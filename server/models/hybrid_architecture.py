# models/model_architecture.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, DistilBertModel
from typing import Dict, Optional

class FusionModule(nn.Module):
    def __init__(self, gpt2_dim: int, bert_dim: int, fusion_dim: int):
        super().__init__()
        # Projection layers
        self.gpt2_proj = nn.Linear(gpt2_dim, fusion_dim)
        self.bert_proj = nn.Linear(bert_dim, fusion_dim)
        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(fusion_dim, num_heads=8, batch_first=True)
        # Fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid()
        )
        self.layer_norm = nn.LayerNorm(fusion_dim)

    def forward(self, gpt2_features: torch.Tensor, bert_features: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        # Project features to same dimension
        gpt2_proj = self.gpt2_proj(gpt2_features)
        bert_proj = self.bert_proj(bert_features)
        # Cross-attention between features
        attn_mask = attention_mask.float().masked_fill(attention_mask == 0, float('-inf'))
        cross_attended, _ = self.cross_attention(gpt2_proj, bert_proj, bert_proj,
                                               key_padding_mask=~attention_mask.bool())
        # Gated fusion
        gate_weights = self.fusion_gate(torch.cat([gpt2_proj, cross_attended], dim=-1))
        fused = gate_weights * gpt2_proj + (1 - gate_weights) * cross_attended
        return self.layer_norm(fused)

class HybridFusionMultiTask(nn.Module):
    def __init__(self, num_intents: int, num_categories: int, num_ner_labels: int,
                 fusion_dim: int = 768, dropout_rate: float = 0.1):
        super().__init__()
        # Base models
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        gpt2_dim = self.gpt2.config.n_embd
        bert_dim = self.distilbert.config.hidden_size
        # Task-specific fusion modules
        self.intent_fusion = FusionModule(gpt2_dim, bert_dim, fusion_dim)
        self.category_fusion = FusionModule(gpt2_dim, bert_dim, fusion_dim)
        self.ner_fusion = FusionModule(gpt2_dim, bert_dim, fusion_dim)
        # Task-specific attention
        self.intent_attention = nn.Sequential(
            nn.Linear(fusion_dim, 1),
            nn.Softmax(dim=1)
        )
        self.category_attention = nn.Sequential(
            nn.Linear(fusion_dim, 1),
            nn.Softmax(dim=1)
        )
        # Classification heads
        classifier_layers = [
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.LayerNorm(fusion_dim // 2),
            nn.Dropout(dropout_rate)
        ]
        self.intent_classifier = nn.Sequential(
            *classifier_layers,
            nn.Linear(fusion_dim // 2, num_intents)
        )
        self.category_classifier = nn.Sequential(
            *classifier_layers,
            nn.Linear(fusion_dim // 2, num_categories)
        )
        self.ner_classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, num_ner_labels)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                intent_labels: Optional[torch.Tensor] = None,
                category_labels: Optional[torch.Tensor] = None,
                ner_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Extract features sequentially
        with torch.no_grad():  # Disable gradients for feature extraction
            gpt2_outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
            distilbert_outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        gpt2_features = gpt2_outputs.last_hidden_state.detach()  # Detach to save memory
        bert_features = distilbert_outputs.last_hidden_state.detach()  # Detach to save memory
        # Fuse features for each task
        intent_fused = self.intent_fusion(gpt2_features, bert_features, attention_mask)
        category_fused = self.category_fusion(gpt2_features, bert_features, attention_mask)
        ner_fused = self.ner_fusion(gpt2_features, bert_features, attention_mask)
        # Task-specific predictions with attention
        intent_weights = self.intent_attention(intent_fused)
        intent_pooled = torch.sum(intent_fused * intent_weights, dim=1)
        intent_logits = self.intent_classifier(intent_pooled)
        category_weights = self.category_attention(category_fused)
        category_pooled = torch.sum(category_fused * category_weights, dim=1)
        category_logits = self.category_classifier(category_pooled)
        ner_logits = self.ner_classifier(ner_fused)
        # Calculate losses if labels are provided
        loss = None
        if all(label is not None for label in [intent_labels, category_labels, ner_labels]):
            intent_loss = F.cross_entropy(intent_logits, intent_labels)
            category_loss = F.cross_entropy(category_logits, category_labels)
            active_loss = attention_mask.view(-1) == 1
            active_logits = ner_logits.view(-1, ner_logits.size(-1))[active_loss]
            active_labels = ner_labels.view(-1)[active_loss]
            ner_loss = F.cross_entropy(active_logits, active_labels)
            loss = intent_loss + category_loss + ner_loss
        return {
            'intent_logits': intent_logits,
            'category_logits': category_logits,
            'ner_logits': ner_logits,
            'loss': loss
        }
        