# models/inference_utils.py

import torch
from transformers import AutoTokenizer
import json

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