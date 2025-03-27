import torch
import re
from transformers import GPT2TokenizerFast, DistilBertTokenizerFast



def inference_hybrid(model, text, gpt2_tokenizer, distilbert_tokenizer, label_encoders, max_length, device):
    model.eval()
    gpt2_encoding = gpt2_tokenizer(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    distilbert_encoding = distilbert_tokenizer(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")

    inputs = {
        "gpt2_input_ids": gpt2_encoding["input_ids"].to(device),
        "gpt2_attention_mask": gpt2_encoding["attention_mask"].to(device),
        "distilbert_input_ids": distilbert_encoding["input_ids"].to(device),
        "distilbert_attention_mask": distilbert_encoding["attention_mask"].to(device)
    }

    with torch.no_grad():
        outputs = model(**inputs)

    intent_logits = outputs["intent_logits"]
    category_logits = outputs["category_logits"]
    ner_logits = outputs["ner_logits"]

    intent_probs = torch.nn.functional.softmax(intent_logits, dim=-1)[0]
    category_probs = torch.nn.functional.softmax(category_logits, dim=-1)[0]
    ner_probs = torch.nn.functional.softmax(ner_logits, dim=-1)

    intent_pred = torch.argmax(intent_probs).cpu().item()
    intent_confidence = intent_probs[intent_pred].cpu().item()
    category_pred = torch.argmax(category_probs).cpu().item()
    category_confidence = category_probs[category_pred].cpu().item()
    ner_preds = torch.argmax(ner_probs, dim=-1).cpu().numpy()[0]
    ner_confidences = torch.max(ner_probs, dim=-1)[0][0].cpu().numpy()

    intent_decoder = {v: k for k, v in label_encoders["intent_encoder"].items()}
    category_decoder = {v: k for k, v in label_encoders["category_encoder"].items()}
    ner_decoder = {v: k for k, v in label_encoders["ner_label_encoder"].items()}

    intent_label = intent_decoder[intent_pred]
    category_label = category_decoder[category_pred]
    tokens = gpt2_tokenizer.convert_ids_to_tokens(inputs["gpt2_input_ids"][0].tolist())
    seq_len = int(inputs["gpt2_attention_mask"][0].sum().item())
    ner_labels = [ner_decoder[pred] for pred in ner_preds[:seq_len]]

    entities = []
    current_entity = None
    entity_tokens = []
    entity_confidences = []
    entity_type = None

    for i, (token, label, confidence) in enumerate(zip(tokens[:seq_len], ner_labels, ner_confidences[:seq_len])):
        if label.startswith("B-"):
            if current_entity is not None:
                entity_text = gpt2_tokenizer.convert_tokens_to_string(entity_tokens).strip()
                if entity_text:
                    avg_confidence = sum(entity_confidences) / len(entity_confidences)
                    entities.append({"entity": entity_text, "label": entity_type, "confidence": avg_confidence})
            current_entity = label[2:]
            entity_type = label[2:]
            entity_tokens = [token]
            entity_confidences = [confidence]
        elif label.startswith("I-") and current_entity == label[2:]:
            entity_tokens.append(token)
            entity_confidences.append(confidence)
        elif current_entity is not None:
            entity_text = gpt2_tokenizer.convert_tokens_to_string(entity_tokens).strip()
            if entity_text:
                avg_confidence = sum(entity_confidences) / len(entity_confidences)
                entities.append({"entity": entity_text, "label": entity_type, "confidence": avg_confidence})
            current_entity = None
            entity_tokens = []
            entity_confidences = []
            entity_type = None

    if current_entity is not None:
        entity_text = gpt2_tokenizer.convert_tokens_to_string(entity_tokens).strip()
        if entity_text:
            avg_confidence = sum(entity_confidences) / len(entity_confidences)
            entities.append({"entity": entity_text, "label": entity_type, "confidence": avg_confidence})

    return {
        "intent": {"label": intent_label, "confidence": intent_confidence},
        "category": {"label": category_label, "confidence": category_confidence},
        "ner": entities
    }
pass

def generate_response(model, tokenizer, instruction, classification, history, max_length=1024, device="cuda"):
    model.eval()
    intent = classification["intent"]["label"] if isinstance(classification["intent"], dict) else classification["intent"]
    category = classification["category"]["label"] if isinstance(classification["category"], dict) else classification["category"]
    if isinstance(intent, str) and "[" in intent:
        intent = intent.strip("[]'")
    if isinstance(category, str) and "[" in category:
        category = category.strip("[]'")
    entities_text = ", ".join([f"{entity['entity']} ({entity['label']})" for entity in classification["ner"]]) if classification["ner"] else "none"

    history_text = ""
    if history:
        history_text = "Previous conversation:\n" + "\n".join([f"User: {h['instruction']}\nAssistant: {h['response']}" for h in history]) + "\n\n"

    input_text = f"[INST] {history_text}Current query: {instruction}\n\nBased on the following classification:\n- Intent: {intent}\n- Category: {category}\n- Entities: {entities_text}\n\nProvide a helpful customer service response: [RESP]"

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,  # Set to 1024 for GPT-2's full max
                num_beams=5,
                no_repeat_ngram_size=2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
        if "[RESP]" in generated_text:
            response = generated_text.split("[RESP]")[1].strip()
            if "[EOS]" in response:
                response = response.split("[EOS]")[0].strip()
        else:
            response = generated_text[len(input_text):].strip()
        steps_pattern = re.search(r'(\d+)\.\s+([A-Z])', response)
        if steps_pattern or "step" in response.lower() or "follow" in response.lower():
            for i in range(1, 10):
                step_marker = f"{i}. "
                if step_marker in response and f"\n{i}. " not in response:
                    response = response.replace(step_marker, f"\n{i}. ")
            response = re.sub(r'\n\s*\n', '\n\n', response)
            response = response.lstrip('\n')
        response = re.sub(r'https?://\S+', '', response)
        response = re.sub(r'<[^>]*>', '', response)
        response = re.sub(r'\{\s*"[^"]*":', '', response)
        response = re.sub(r'\s+', ' ', response).strip()
        return response
    except Exception as e:
        print(f"Error in generate_response: {e}")
        return f"I apologize, but I couldn't generate a response. Error: {str(e)}"
pass
