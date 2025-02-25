# models/baseline_inference_utils.py

import torch
from transformers import GPT2TokenizerFast

def preprocess_input(text, gpt2_tokenizer, max_length=128):
    """
    Preprocess input text for inference.
    :param text: Input text.
    :param gpt2_tokenizer: GPT-2 tokenizer.
    :param max_length: Maximum sequence length.
    :return: Tokenized inputs as tensors.
    """
    gpt2_inputs = gpt2_tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return {
        "input_ids": gpt2_inputs["input_ids"],
        "attention_mask": gpt2_inputs["attention_mask"],
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
    tokens = tokenizer.tokenize(text)
    aligned_labels = ner_labels[:len(tokens)]
    aligned_confidences = ner_confidences[:len(tokens)]
    entity_spans = []
    current_entity = None
    current_tokens = []
    current_confidences = []
    for token, label, confidence in zip(tokens, aligned_labels, aligned_confidences):
        if label.startswith("B-"):
            if current_entity:
                entity_spans.append({
                    "type": current_entity,
                    "text": tokenizer.convert_tokens_to_string(current_tokens),
                    "confidence": sum(current_confidences) / len(current_confidences)
                })
            current_entity = label[2:]
            current_tokens = [token]
            current_confidences = [confidence]
        elif label.startswith("I-") and current_entity:
            current_tokens.append(token)
            current_confidences.append(confidence)
        else:
            if current_entity:
                entity_spans.append({
                    "type": current_entity,
                    "text": tokenizer.convert_tokens_to_string(current_tokens),
                    "confidence": sum(current_confidences) / len(current_confidences)
                })
                current_entity = None
                current_tokens = []
                current_confidences = []
    if current_entity:
        entity_spans.append({
            "type": current_entity,
            "text": tokenizer.convert_tokens_to_string(current_tokens),
            "confidence": sum(current_confidences) / len(current_confidences)
        })
    return {
        "aligned_labels": aligned_labels,
        "entity_spans": entity_spans
    }