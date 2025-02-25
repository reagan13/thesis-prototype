# models/hybrid_model.py

import torch
from transformers import GPT2TokenizerFast, DistilBertTokenizerFast
from models.hybrid_architecture import HybridFusionMultiTask
from utils.hybrid import preprocess_input, postprocess_ner
import json

# Define paths
HYBRID_MODEL_PATH = "hybrid/hybrid_fusion_multitask_model.pth"
GPT2_TOKENIZER_PATH = "hybrid/gpt2_tokenizer"
DISTILBERT_TOKENIZER_PATH = "hybrid/distilbert_tokenizer"
LABEL_ENCODERS_PATH = "hybrid/label_encoders.json"

# Load label encoders
with open(LABEL_ENCODERS_PATH, "r", encoding="utf-8") as f:
    label_encoders = json.load(f)

# Initialize model
num_intents = len(label_encoders["intent_encoder"])
num_categories = len(label_encoders["category_encoder"])
num_ner_labels = len(label_encoders["ner_label_encoder"])

hybrid_model = HybridFusionMultiTask(
    num_intents=num_intents,
    num_categories=num_categories,
    num_ner_labels=num_ner_labels
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hybrid_model.load_state_dict(torch.load(HYBRID_MODEL_PATH, map_location=device))
hybrid_model.eval()
hybrid_model.to(device)

# Load tokenizers
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(GPT2_TOKENIZER_PATH)
distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_TOKENIZER_PATH)

def hybrid_infer(text):
    """
    Perform inference on the Hybrid Fusion Multi-Task model.
    :param text: Input text to process.
    :return: Dictionary containing predictions and confidence percentages.
    """
    # Preprocess input
    inputs = preprocess_input(text, gpt2_tokenizer, distilbert_tokenizer)
    # Move inputs to device
    gpt2_input_ids = inputs["gpt2_input_ids"].to(device)
    gpt2_attention_mask = inputs["gpt2_attention_mask"].to(device)
    distilbert_input_ids = inputs["distilbert_input_ids"].to(device)
    distilbert_attention_mask = inputs["distilbert_attention_mask"].to(device)
    # Forward pass
    with torch.no_grad():
        outputs = hybrid_model(
            input_ids=distilbert_input_ids,
            attention_mask=distilbert_attention_mask
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
        distilbert_tokenizer
    )
    return {
        "intent": {"label": intent_pred, "confidence": intent_confidence},
        "category": {"label": category_pred, "confidence": category_confidence},
        "ner": ner_results["entity_spans"]
    }