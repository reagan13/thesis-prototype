import torch
from transformers import AutoTokenizer
import json
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from transformers import GPT2Model, DistilBertModel, DistilBertTokenizerFast, GPT2TokenizerFast

# Define paths for the Hybrid Model
HYBRID_MODEL_PATH = "./hybrid/hybrid_fusion_multitask_model_v2.pth"
GPT2_TOKENIZER_PATH = "./hybrid/gpt2_tokenizer"
DISTILBERT_TOKENIZER_PATH = "./hybrid/distilbert_tokenizer"
LABEL_ENCODERS_PATH = "./hybrid/label_encoders.json"

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



# Initialize the model architecture
with open(LABEL_ENCODERS_PATH, "r", encoding="utf-8") as f:
    label_encoders = json.load(f)

num_intents = len(label_encoders["intent_encoder"])
num_categories = len(label_encoders["category_encoder"])
num_ner_labels = len(label_encoders["ner_label_encoder"])

hybrid_model = HybridFusionMultiTask(
    num_intents=num_intents,
    num_categories=num_categories,
    num_ner_labels=num_ner_labels
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the saved weights
hybrid_model.load_state_dict(torch.load(HYBRID_MODEL_PATH, map_location=torch.device(device)))
hybrid_model.eval()

# Load the tokenizers
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(GPT2_TOKENIZER_PATH)
distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_TOKENIZER_PATH)

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hybrid_model.to(device)

def preprocess_input(text, gpt2_tokenizer, distilbert_tokenizer, max_length=128):
    """
    Preprocess input text for inference.
    :param text: Input text.
    :param gpt2_tokenizer: GPT-2 tokenizer.
    :param distilbert_tokenizer: DistilBERT tokenizer.
    :param max_length: Maximum sequence length.
    :return: Tokenized inputs as tensors.
    """
    # Tokenize with GPT-2
    gpt2_inputs = gpt2_tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    # Tokenize with DistilBERT
    distilbert_inputs = distilbert_tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return {
        "gpt2_input_ids": gpt2_inputs["input_ids"],
        "gpt2_attention_mask": gpt2_inputs["attention_mask"],
        "distilbert_input_ids": distilbert_inputs["input_ids"],
        "distilbert_attention_mask": distilbert_inputs["attention_mask"]
    }

def postprocess_ner(text, ner_labels, ner_confidences, tokenizer, max_length=128):
    """
    Postprocess NER labels to align with the original tokens and extract entity spans with confidence levels.
    :param text: Original input text.
    :param ner_labels: Predicted NER labels (list of labels).
    :param ner_confidences: Confidence levels for each token (list of floats).
    :param tokenizer: Tokenizer used for tokenization.
    :param max_length: Maximum sequence length.
    :return: Dictionary containing aligned NER labels, extracted entity spans, and confidence levels.
    """
    # Tokenize the text to get the actual tokens
    tokens = tokenizer.tokenize(text)
    # Remove padding and special tokens from NER labels and confidences
    aligned_labels = ner_labels[:len(tokens)]  # Keep only labels corresponding to actual tokens
    aligned_confidences = ner_confidences[:len(tokens)]
    # Extract entity spans
    entity_spans = []
    current_entity = None
    current_tokens = []
    current_confidences = []
    for token, label, confidence in zip(tokens, aligned_labels, aligned_confidences):
        if label.startswith("B-"):  # Beginning of a new entity
            if current_entity:
                # Save the previous entity
                entity_spans.append({
                    "type": current_entity,
                    "text": tokenizer.convert_tokens_to_string(current_tokens),
                    "confidence": sum(current_confidences) / len(current_confidences)  # Average confidence
                })
            # Start a new entity
            current_entity = label[2:]  # Remove "B-" prefix
            current_tokens = [token]
            current_confidences = [confidence]
        elif label.startswith("I-") and current_entity:  # Inside the same entity
            current_tokens.append(token)
            current_confidences.append(confidence)
        else:  # Outside any entity
            if current_entity:
                # Save the previous entity
                entity_spans.append({
                    "type": current_entity,
                    "text": tokenizer.convert_tokens_to_string(current_tokens),
                    "confidence": sum(current_confidences) / len(current_confidences)  # Average confidence
                })
                current_entity = None
                current_tokens = []
                current_confidences = []
    # Add the last entity if it exists
    if current_entity:
        entity_spans.append({
            "type": current_entity,
            "text": tokenizer.convert_tokens_to_string(current_tokens),
            "confidence": sum(current_confidences) / len(current_confidences)  # Average confidence
        })
    return {
        "aligned_labels": aligned_labels,
        "entity_spans": entity_spans
    }
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
    intent_logits = outputs["intent_logits"][0]  # Shape: (num_intents,)
    intent_probs = torch.softmax(intent_logits, dim=-1).cpu().numpy()  # Softmax to get probabilities
    intent_pred_idx = intent_probs.argmax()
    intent_pred = list(label_encoders["intent_encoder"].keys())[
        list(label_encoders["intent_encoder"].values()).index(intent_pred_idx)
    ]
    intent_confidence = float(intent_probs[intent_pred_idx])  # Convert to standard Python float
    # Category prediction and confidence
    category_logits = outputs["category_logits"][0]  # Shape: (num_categories,)
    category_probs = torch.softmax(category_logits, dim=-1).cpu().numpy()
    category_pred_idx = category_probs.argmax()
    category_pred = list(label_encoders["category_encoder"].keys())[
        list(label_encoders["category_encoder"].values()).index(category_pred_idx)
    ]
    category_confidence = float(category_probs[category_pred_idx])  # Convert to standard Python float
    # NER predictions and confidence
    ner_logits = outputs["ner_logits"][0]  # Shape: (max_length, num_ner_labels)
    ner_probs = torch.softmax(ner_logits, dim=-1).cpu().numpy()  # Softmax over NER labels
    ner_preds = ner_probs.argmax(axis=-1)  # Predicted indices
    ner_labels = [list(label_encoders["ner_label_encoder"].keys())[p] for p in ner_preds]
    ner_confidences = ner_probs.max(axis=-1).tolist()  # Convert to list of standard Python floats
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