# models/baseline_model_architecture.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config
from typing import Dict, Optional



# Model definition (unchanged)
class BaselineGPT2MultiTask(nn.Module):
    def __init__(self, num_intents: int, num_categories: int, num_ner_labels: int, dropout_rate: float = 0.1):
        super().__init__()
        self.config = GPT2Config.from_pretrained('gpt2')
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        hidden_size = self.config.n_embd
        for param in self.gpt2.parameters():
            param.requires_grad = False
        self.intent_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_intents)
        )
        self.category_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_categories)
        )
        self.ner_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_ner_labels)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                intent_labels: Optional[torch.Tensor] = None,
                category_labels: Optional[torch.Tensor] = None,
                ner_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        batch_size = sequence_output.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        last_token_indexes = sequence_lengths.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, sequence_output.shape[-1])
        sequence_repr = torch.gather(sequence_output, 1, last_token_indexes).squeeze(1)
        intent_logits = self.intent_head(sequence_repr)
        category_logits = self.category_head(sequence_repr)
        ner_logits = self.ner_head(sequence_output)
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
