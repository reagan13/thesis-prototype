import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2TokenizerFast, GPT2Model

# Define paths for the Baseline Model
MODEL_PATH = "baseline/model.pth"
TOKENIZER_PATH = "baseline/tokenizer"
LABEL_ENCODERS_PATH = "baseline/label_encoders.json"

# Load label encoders
with open(LABEL_ENCODERS_PATH, "r", encoding="utf-8") as f:
    label_encoders = json.load(f)

# Minimal model definition (required for loading weights)
class BaselineGPT2MultiTask(nn.Module):
    def __init__(self, num_intents, num_categories, num_ner_labels, dropout_rate=0.1, hidden_size=768):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        # Freeze GPT-2
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

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        batch_size = sequence_output.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        last_token_indexes = sequence_lengths.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, sequence_output.shape[-1])
        sequence_repr = torch.gather(sequence_output, 1, last_token_indexes).squeeze(1)
        intent_logits = self.intent_head(sequence_repr)
        category_logits = self.category_head(sequence_repr)
        ner_logits = self.ner_head(sequence_output)
        return {
            'intent_logits': intent_logits,
            'category_logits': category_logits,
            'ner_logits': ner_logits
        }

# Initialize the model architecture
num_intents = len(label_encoders["intent_encoder"])
num_categories = len(label_encoders["category_encoder"])
num_ner_labels = len(label_encoders["ner_label_encoder"])

model = BaselineGPT2MultiTask(
    num_intents=num_intents,
    num_categories=num_categories,
    num_ner_labels=num_ner_labels
)

# Load the tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_PATH)

# Add a padding token if necessary
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Resize the embedding layer of the model
model.gpt2.resize_token_embeddings(len(tokenizer))

# Load the saved weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
model.to(device)
model.eval()

def infer(text: str) -> dict:
    """
    Perform inference on the Baseline GPT-2 model.
    :param text: Input text to process.
    :return: Dictionary containing predictions and confidence percentages.
    """
    # Tokenize the input text
    encoding = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Extract logits
    intent_logits = outputs["intent_logits"].cpu()
    category_logits = outputs["category_logits"].cpu()
    ner_logits = outputs["ner_logits"].cpu()

    # Convert logits to probabilities using softmax
    intent_probs = F.softmax(intent_logits, dim=-1).squeeze(0)
    category_probs = F.softmax(category_logits, dim=-1).squeeze(0)
    ner_probs = F.softmax(ner_logits, dim=-1).squeeze(0)

    # Get predictions and confidence percentages
    intent_pred_idx = torch.argmax(intent_probs).item()
    category_pred_idx = torch.argmax(category_probs).item()
    intent_encoder = label_encoders["intent_encoder"]
    category_encoder = label_encoders["category_encoder"]
    ner_label_encoder = label_encoders["ner_label_encoder"]

    intent_pred = list(intent_encoder.keys())[list(intent_encoder.values()).index(intent_pred_idx)]
    category_pred = list(category_encoder.keys())[list(category_encoder.values()).index(category_pred_idx)]

    intent_confidence = intent_probs[intent_pred_idx].item() * 100
    category_confidence = category_probs[category_pred_idx].item() * 100

    # Process NER predictions
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
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
                # Continue the current entity
                current_entity += clean_token
                current_confidences.append(ner_confidence)
            else:
                # Save the previous entity
                if current_entity:
                    avg_confidence = sum(current_confidences) / len(current_confidences)
                    grouped_ner_entities.append({
                        "entity": current_entity.strip(),
                        "label": current_label,
                        "confidence": round(avg_confidence, 2)
                    })
                # Start a new entity
                current_entity = clean_token
                current_label = ner_pred
                current_confidences = [ner_confidence]
        else:
            # End the current entity
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

    # Add the last entity if it exists
    if current_entity:
        avg_confidence = sum(current_confidences) / len(current_confidences)
        grouped_ner_entities.append({
            "entity": current_entity.strip(),
            "label": current_label,
            "confidence": round(avg_confidence, 2)
        })

    # Return results
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