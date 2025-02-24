import torch.nn.functional as F

# Preprocess input text for Baseline model
def preprocess_baseline_input(text, tokenizer, max_length=128):
    inputs = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }


# Run inference for Baseline model
def run_baseline_inference(model, text, tokenizer, label_encoders, device):
    inputs = preprocess_baseline_input(text, tokenizer)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Extract predictions
    intent_logits = outputs["intent_logits"][0]
    intent_probs = F.softmax(intent_logits, dim=-1).cpu().numpy()
    intent_pred_idx = intent_probs.argmax()
    intent_pred = list(label_encoders["intent_encoder"].keys())[
        list(label_encoders["intent_encoder"].values()).index(intent_pred_idx)
    ]
    intent_confidence = intent_probs[intent_pred_idx]

    category_logits = outputs["category_logits"][0]
    category_probs = F.softmax(category_logits, dim=-1).cpu().numpy()
    category_pred_idx = category_probs.argmax()
    category_pred = list(label_encoders["category_encoder"].keys())[
        list(label_encoders["category_encoder"].values()).index(category_pred_idx)
    ]
    category_confidence = category_probs[category_pred_idx]

    ner_logits = outputs["ner_logits"][0]
    ner_probs = F.softmax(ner_logits, dim=-1).cpu().numpy()
    ner_preds = ner_probs.argmax(axis=-1)
    ner_labels = [list(label_encoders["ner_label_encoder"].keys())[p] for p in ner_preds]
    ner_confidences = ner_probs.max(axis=-1)

    return {
        "intent": {"label": intent_pred, "confidence": float(intent_confidence)},
        "category": {"label": category_pred, "confidence": float(category_confidence)},
        "ner": {"labels": ner_labels, "confidences": ner_confidences.tolist()}
    }


# Postprocess NER results
def postprocess_ner(text, ner_labels, ner_confidences, tokenizer, max_length=128):
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

    return {"aligned_labels": aligned_labels, "entity_spans": entity_spans}