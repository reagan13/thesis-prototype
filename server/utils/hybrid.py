from flask import Flask, request, jsonify
import torch
from transformers import GPT2TokenizerFast, DistilBertTokenizerFast
import json
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, DistilBertModel
from typing import Dict, Optional



# Preprocessing function
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

# Inference function
def run_hybrid_inference(model, text, gpt2_tokenizer, distilbert_tokenizer, label_encoders, device='cuda'):
    """
    Run inference on a single input text.
    :param model: The hybrid multi-task model.
    :param text: Input text.
    :param gpt2_tokenizer: GPT-2 tokenizer.
    :param distilbert_tokenizer: DistilBERT tokenizer.
    :param label_encoders: Dictionary containing label encoders.
    :param device: Device to run the model on ('cuda' or 'cpu').
    :return: Predictions for intent, category, and NER with confidence levels.
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
        outputs = model(
            input_ids=distilbert_input_ids,
            attention_mask=distilbert_attention_mask
        )

    # Intent prediction and confidence
    intent_logits = outputs["intent_logits"][0]  # Shape: (num_intents,)
    intent_probs = F.softmax(intent_logits, dim=-1).cpu().numpy()  # Softmax to get probabilities
    intent_pred_idx = intent_probs.argmax()
    intent_pred = list(label_encoders["intent_encoder"].keys())[
        list(label_encoders["intent_encoder"].values()).index(intent_pred_idx)
    ]
    intent_confidence = intent_probs[intent_pred_idx]

    # Category prediction and confidence
    category_logits = outputs["category_logits"][0]  # Shape: (num_categories,)
    category_probs = F.softmax(category_logits, dim=-1).cpu().numpy()
    category_pred_idx = category_probs.argmax()
    category_pred = list(label_encoders["category_encoder"].keys())[
        list(label_encoders["category_encoder"].values()).index(category_pred_idx)
    ]
    category_confidence = category_probs[category_pred_idx]

    # NER predictions and confidence
    ner_logits = outputs["ner_logits"][0]  # Shape: (max_length, num_ner_labels)
    ner_probs = F.softmax(ner_logits, dim=-1).cpu().numpy()  # Softmax over NER labels
    ner_preds = ner_probs.argmax(axis=-1)  # Predicted indices
    ner_labels = [list(label_encoders["ner_label_encoder"].keys())[p] for p in ner_preds]
    ner_confidences = ner_probs.max(axis=-1)  # Confidence for each token

    return {
        "intent": {"label": intent_pred, "confidence": float(intent_confidence)},
        "category": {"label": category_pred, "confidence": float(category_confidence)},
        "ner": {
            "labels": ner_labels,
            "confidences": ner_confidences.tolist()
        }
    }

# Postprocess NER results
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
