# models/baseline_model.py

import torch
from transformers import GPT2TokenizerFast
import json
from models.baseline_architecture import BaselineGPT2MultiTask
from utils.baseline import preprocess_input, postprocess_ner

# Define paths
BASELINE_MODEL_PATH = "baseline/model.pth"
GPT2_TOKENIZER_PATH = "baseline/tokenizer"
LABEL_ENCODERS_PATH = "baseline/label_encoders.json"

# Load label encoders
with open(LABEL_ENCODERS_PATH, "r", encoding="utf-8") as f:
    label_encoders = json.load(f)

# Initialize model
num_intents = len(label_encoders["intent_encoder"])
num_categories = len(label_encoders["category_encoder"])
num_ner_labels = len(label_encoders["ner_label_encoder"])

baseline_model = BaselineGPT2MultiTask(
    num_intents=num_intents,
    num_categories=num_categories,
    num_ner_labels=num_ner_labels
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load tokenizer
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(GPT2_TOKENIZER_PATH)


# Add a padding token if necessary
if gpt2_tokenizer.pad_token is None:
    gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    
# Resize the embedding layer of the model
baseline_model.gpt2.resize_token_embeddings(len(gpt2_tokenizer))

baseline_model.load_state_dict(torch.load(BASELINE_MODEL_PATH, map_location=device), strict=False)
baseline_model.eval()
baseline_model.to(device)


def baseline_infer(text):
    """
    Perform inference on the Baseline GPT-2 Multi-Task model.
    :param text: Input text to process.
    :return: Dictionary containing predictions and confidence percentages.
    """
    # Preprocess input
    inputs = preprocess_input(text, gpt2_tokenizer)
    # Move inputs to device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    # Forward pass
    with torch.no_grad():
        outputs = baseline_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    # Intent prediction and confidence
    intent_logits = outputs["intent_logits"][0]
    intent_probs = torch.softmax(intent_logits, dim=-1).cpu().numpy()
    intent_pred_idx = intent_probs.argmax()
    intent_pred = list(label_encoders["intent_encoder"].keys())[
        list(label_encoders["intent_encoder"].values()).index(intent_pred_idx)
    ]
    intent_confidence = float(intent_probs[intent_pred_idx])
    # Category prediction and confidence
    category_logits = outputs["category_logits"][0]
    category_probs = torch.softmax(category_logits, dim=-1).cpu().numpy()
    category_pred_idx = category_probs.argmax()
    category_pred = list(label_encoders["category_encoder"].keys())[
        list(label_encoders["category_encoder"].values()).index(category_pred_idx)
    ]
    category_confidence = float(category_probs[category_pred_idx])
    # NER predictions and confidence
    ner_logits = outputs["ner_logits"][0]
    ner_probs = torch.softmax(ner_logits, dim=-1).cpu().numpy()
    ner_preds = ner_probs.argmax(axis=-1)
    ner_labels = [list(label_encoders["ner_label_encoder"].keys())[p] for p in ner_preds]
    ner_confidences = ner_probs.max(axis=-1).tolist()
    # Postprocess NER labels
    ner_results = postprocess_ner(
        text,
        ner_labels,
        ner_confidences,
        gpt2_tokenizer
    )
    return {
        "intent": {"label": intent_pred, "confidence": intent_confidence},
        "category": {"label": category_pred, "confidence": category_confidence},
        "ner": ner_results["entity_spans"]
    }