import torch
from transformers import GPT2TokenizerFast
import json

# Define the BaselineGPT2MultiTask model architecture
class BaselineGPT2MultiTask(torch.nn.Module):
    def __init__(self, num_intents: int, num_categories: int, num_ner_labels: int, dropout_rate: float = 0.1):
        super().__init__()
        from transformers import GPT2Model, GPT2Config
        self.config = GPT2Config.from_pretrained('gpt2')
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        hidden_size = self.config.n_embd

        # Intent Classification Head
        self.intent_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_size, num_intents)
        )

        # Category Classification Head
        self.category_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_size, num_categories)
        )

        # NER Head
        self.ner_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_size, num_ner_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # Get sequence representation for classification (using last token)
        batch_size = sequence_output.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        last_token_indexes = sequence_lengths.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, sequence_output.shape[-1])
        sequence_repr = torch.gather(sequence_output, 1, last_token_indexes).squeeze(1)

        # Task-specific predictions
        intent_logits = self.intent_head(sequence_repr)
        category_logits = self.category_head(sequence_repr)
        ner_logits = self.ner_head(sequence_output)

        return {
            "intent_logits": intent_logits,
            "category_logits": category_logits,
            "ner_logits": ner_logits
        }


# Load the Baseline model
def load_baseline_model(model_path, label_encoders, device):
    model = BaselineGPT2MultiTask(
        num_intents=len(label_encoders["intent_encoder"]),
        num_categories=len(label_encoders["category_encoder"]),
        num_ner_labels=len(label_encoders["ner_label_encoder"])
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# Load tokenizer for Baseline model
def load_baseline_tokenizer(tokenizer_dir):
    tokenizer = GPT2TokenizerFast.from_pretrained(f"{tokenizer_dir}/tokenizer")
    return tokenizer


# Load label encoders
def load_label_encoders(encoder_path):
    with open(encoder_path, "r") as f:
        label_encoders = json.load(f)
    return label_encoders