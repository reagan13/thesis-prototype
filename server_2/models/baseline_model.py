import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model, DistilBertModel
from typing import Dict, Optional
def log_to_file(message):
    """Helper function for logging"""
    print(message)

# Baseline GPT-2 Multi-Task Model
class BaselineGPT2MultiTask(nn.Module):
    def __init__(self, num_intents: int, num_categories: int, num_ner_labels: int,
                 dropout_rate: float, loss_weights: Optional[Dict[str, float]] = None,
                 ner_class_weights: Optional[torch.Tensor] = None,
                 category_class_weights: Optional[torch.Tensor] = None,
                 intent_class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        log_to_file("Initializing classification model...")
        self.config = GPT2Config.from_pretrained('gpt2')
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        hidden_size = self.config.n_embd

        for param in self.gpt2.parameters():
            param.requires_grad = False
        log_to_file("All GPT-2 layers remain frozen")

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

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                intent_labels: Optional[torch.Tensor] = None,
                category_labels: Optional[torch.Tensor] = None,
                ner_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        masked_features = sequence_output * attention_mask.unsqueeze(-1)
        sequence_repr = masked_features.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

        intent_logits = self.intent_head(sequence_repr)
        category_logits = self.category_head(sequence_repr)
        ner_logits = self.ner_head(sequence_output)

        output_dict = {
            'intent_logits': intent_logits,
            'category_logits': category_logits,
            'ner_logits': ner_logits
        }

        if all(label is not None for label in [intent_labels, category_labels, ner_labels]):
            intent_loss = self.intent_loss_fn(intent_logits, intent_labels)
            category_loss = self.category_loss_fn(category_logits, category_labels)
            active_loss = attention_mask.view(-1) == 1
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