import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config
from typing import Dict, Optional

import torch
from transformers import AutoTokenizer
import json

# Define paths for the Baseline Model
BASELINE_MODEL_PATH = "./baseline/model.pth"
BASELINE_TOKENIZER_PATH = "./baseline/tokenizer"
BASELINE_LABEL_ENCODERS_PATH = "./baseline/label_encoders.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaselineGPT2MultiTask(nn.Module):
    def __init__(self, num_intents: int, num_categories: int, num_ner_labels: int, dropout_rate: float = 0.1):
        super().__init__()

        # Load GPT2-large as base model
        self.config = GPT2Config.from_pretrained('gpt2')
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        hidden_size = self.config.n_embd

        # Intent Classification Head
        self.intent_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_intents)
        )

        # Category Classification Head
        self.category_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_categories)
        )

        # NER Head
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

        # Get GPT2 outputs
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

        # Calculate losses if labels are provided
        loss = None
        if all(label is not None for label in [intent_labels, category_labels, ner_labels]):
            # Intent loss
            intent_loss = F.cross_entropy(intent_logits, intent_labels)

            # Category loss
            category_loss = F.cross_entropy(category_logits, category_labels)

            # NER loss (handle padding)
            active_loss = attention_mask.view(-1) == 1
            active_logits = ner_logits.view(-1, ner_logits.size(-1))[active_loss]
            active_labels = ner_labels.view(-1)[active_loss]
            ner_loss = F.cross_entropy(active_logits, active_labels)

            # Total loss
            loss = intent_loss + category_loss + ner_loss

        return {
            'intent_logits': intent_logits,
            'category_logits': category_logits,
            'ner_logits': ner_logits,
            'loss': loss
        }


# Initialize the model architecture
with open(BASELINE_LABEL_ENCODERS_PATH, "r", encoding="utf-8") as f:
    label_encoders = json.load(f)


num_intents = len(label_encoders["intent_encoder"])
num_categories = len(label_encoders["category_encoder"])
num_ner_labels = len(label_encoders["ner_label_encoder"])

baseline_model = BaselineGPT2MultiTask(
    num_intents=num_intents,
    num_categories=num_categories,
    num_ner_labels=num_ner_labels
)
baseline_model.load_state_dict(torch.load(BASELINE_MODEL_PATH, map_location=torch.device(device)))
baseline_model.eval()


# Load the tokenizer
baseline_tokenizer = AutoTokenizer.from_pretrained(BASELINE_TOKENIZER_PATH)

# Load the label encoders
with open(BASELINE_LABEL_ENCODERS_PATH, "r", encoding="utf-8") as f:
    baseline_label_encoders = json.load(f)
    

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
baseline_model.to(device)

def baseline_infer(text: str):
    """
    Perform inference on the Baseline GPT-2 model.
    :param text: Input text to process.
    :return: Dictionary containing predictions and confidence percentages.
    """
    # Tokenize the input text
    encoding = baseline_tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Perform inference
    baseline_model.eval()
    with torch.no_grad():
        outputs = baseline_model(input_ids=input_ids, attention_mask=attention_mask)

    # Extract logits
    intent_logits = outputs["intent_logits"].cpu()
    category_logits = outputs["category_logits"].cpu()
    ner_logits = outputs["ner_logits"].cpu()

    # Convert logits to probabilities using softmax
    intent_probs = torch.softmax(intent_logits, dim=-1).squeeze(0)
    category_probs = torch.softmax(category_logits, dim=-1).squeeze(0)
    ner_probs = torch.softmax(ner_logits, dim=-1).squeeze(0)

    # Get predictions and confidence percentages
    intent_pred_idx = torch.argmax(intent_probs).item()
    category_pred_idx = torch.argmax(category_probs).item()
    intent_encoder = baseline_label_encoders["intent_encoder"]
    category_encoder = baseline_label_encoders["category_encoder"]
    ner_label_encoder = baseline_label_encoders["ner_label_encoder"]

    intent_pred = list(intent_encoder.keys())[list(intent_encoder.values()).index(intent_pred_idx)]
    category_pred = list(category_encoder.keys())[list(category_encoder.values()).index(category_pred_idx)]

    intent_confidence = intent_probs[intent_pred_idx].item() * 100
    category_confidence = category_probs[category_pred_idx].item() * 100

    # Process NER predictions
    tokens = baseline_tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    active_mask = attention_mask.squeeze(0).bool()

    grouped_ner_entities = []
    current_entity = ""
    current_label = None
    current_confidences = []

    for i, token in enumerate(tokens):
        if not active_mask[i] or token in ["", "[PAD]"]:
            continue  # Skip padding and special tokens

        clean_token = token.replace("Ġ", " ").strip()
        ner_prob = ner_probs[i]
        ner_pred_idx = torch.argmax(ner_prob).item()
        ner_pred = list(ner_label_encoder.keys())[list(ner_label_encoder.values()).index(ner_pred_idx)]
        ner_confidence = ner_prob[ner_pred_idx].item() * 100

        if ner_pred != "O":  # Start of a new entity
            if current_entity and current_label == ner_pred:
                current_entity += clean_token
                current_confidences.append(ner_confidence)
            else:
                if current_entity:
                    avg_confidence = sum(current_confidences) / len(current_confidences)
                    grouped_ner_entities.append({
                        "entity": current_entity.strip(),
                        "label": current_label,
                        "confidence": round(avg_confidence, 2)
                    })
                current_entity = clean_token
                current_label = ner_pred
                current_confidences = [ner_confidence]
        else:
            if current_entity:
                avg_confidence = sum(current_confidences) / len(current_confidences)
                grouped_ner_entities.append({
                    "entity": current_entity.strip(),
                    "label": current_label,
                    "confidence": round(avg_confidence, 2)
                })
                current_entity = ""
                current_label = None
                current_confidences = []

    if current_entity:
        avg_confidence = sum(current_confidences) / len(current_confidences)
        grouped_ner_entities.append({
            "entity": current_entity.strip(),
            "label": current_label,
            "confidence": round(avg_confidence, 2)
        })

    return {
        "text": text,
        "intent": {
            "prediction": intent_pred,
            "confidence": round(intent_confidence, 2)
        },
        "category": {
            "prediction": category_pred,
            "confidence": round(category_confidence, 2)
        },
        "ner": grouped_ner_entities
    }