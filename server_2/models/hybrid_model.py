import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model, DistilBertModel
from .fusion_layers import FusionLayer, CrossAttentionFusionLayer, DenseFusionLayer
from typing import Dict, Optional
def log_to_file(message):
    print(message)

class HybridGPT2DistilBERTMultiTask(nn.Module):
    def __init__(self, num_intents: int, num_categories: int, num_ner_labels: int,
                 dropout_rate: float, fusion_type: str = "concat",
                 loss_weights: Optional[Dict[str, float]] = None,
                 ner_class_weights: Optional[torch.Tensor] = None,
                 category_class_weights: Optional[torch.Tensor] = None,
                 intent_class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        log_to_file(f"Initializing model with {fusion_type} fusion...")
        self.gpt2_config = GPT2Config.from_pretrained('gpt2')
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        for param in self.gpt2.parameters():
            param.requires_grad = False
        for param in self.distilbert.parameters():
            param.requires_grad = False
        log_to_file("All GPT-2 and DistilBERT layers remain frozen")

        gpt2_dim = self.gpt2_config.n_embd
        bert_dim = self.distilbert.config.hidden_size
        hidden_size = gpt2_dim

        if fusion_type == "concat":
            self.fusion_layer = FusionLayer(gpt2_dim, bert_dim, hidden_size, dropout_rate)
        elif fusion_type == "crossattention":
            self.fusion_layer = CrossAttentionFusionLayer(gpt2_dim, bert_dim, hidden_size, dropout_rate)
        elif fusion_type == "dense":
            self.fusion_layer = DenseFusionLayer(gpt2_dim, bert_dim, hidden_size, dropout_rate)
        else:
            raise ValueError("fusion_type must be 'concat', 'crossattention', or 'dense'")

        self.intent_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_intents)
        )
        self.category_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_categories)
        )
        self.ner_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_ner_labels)
        )

        self.loss_weights = loss_weights or {'intent': 0.3, 'category': 0.3, 'ner': 0.4}
        self.intent_loss_fn = nn.CrossEntropyLoss(weight=intent_class_weights) if intent_class_weights is not None else nn.CrossEntropyLoss()
        self.category_loss_fn = nn.CrossEntropyLoss(weight=category_class_weights) if category_class_weights is not None else nn.CrossEntropyLoss()
        self.ner_loss_fn = nn.CrossEntropyLoss(weight=ner_class_weights) if ner_class_weights is not None else nn.CrossEntropyLoss()

    def forward(self, gpt2_input_ids, gpt2_attention_mask,
                distilbert_input_ids, distilbert_attention_mask,
                intent_labels=None, category_labels=None, ner_labels=None):
        gpt2_outputs = self.gpt2(input_ids=gpt2_input_ids, attention_mask=gpt2_attention_mask)
        distilbert_outputs = self.distilbert(input_ids=distilbert_input_ids, attention_mask=distilbert_attention_mask)

        gpt2_features = gpt2_outputs.last_hidden_state
        bert_features = distilbert_outputs.last_hidden_state

        fused_features = self.fusion_layer(gpt2_features, bert_features, gpt2_attention_mask)

        masked_features = fused_features * gpt2_attention_mask.unsqueeze(-1)
        sequence_repr = masked_features.sum(dim=1) / gpt2_attention_mask.sum(dim=1, keepdim=True)

        intent_logits = self.intent_head(sequence_repr)
        category_logits = self.category_head(sequence_repr)
        ner_logits = self.ner_head(fused_features)

        output_dict = {
            'intent_logits': intent_logits,
            'category_logits': category_logits,
            'ner_logits': ner_logits
        }

        if all(label is not None for label in [intent_labels, category_labels, ner_labels]):
            intent_loss = self.intent_loss_fn(intent_logits, intent_labels)
            category_loss = self.category_loss_fn(category_logits, category_labels)
            combined_mask = (gpt2_attention_mask * distilbert_attention_mask)
            active_loss = combined_mask.view(-1) == 1
            active_logits = ner_logits.view(-1, ner_logits.size(-1))[active_loss]
            active_labels = ner_labels.view(-1)[active_loss]
            ner_loss = self.ner_loss_fn(active_logits, active_labels)

            total_loss = (self.loss_weights['intent'] * intent_loss +
                          self.loss_weights['category'] * category_loss +
                          self.loss_weights['ner'] * ner_loss)

            output_dict.update({
                'loss': total_loss,
                'intent_loss': intent_loss,
                'category_loss': category_loss,
                'ner_loss': ner_loss
            })

        return output_dict
    pass