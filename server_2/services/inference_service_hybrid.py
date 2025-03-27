import json
import os
import time
import torch
from threading import Lock
from transformers import GPT2TokenizerFast, DistilBertTokenizerFast, GPT2LMHeadModel, GPT2Tokenizer
from models.hybrid_model import HybridGPT2DistilBERTMultiTask
from utils.inference_hybrid import inference_hybrid, generate_response
from utils.memory_utils import get_peak_memory_usage

class ConversationalInferenceService:
    def __init__(self, model_paths, fusion_type="concat"):
        self.fusion_type = fusion_type
        self.output_dir = model_paths[f"{fusion_type}_model_dir"]
        self.generation_model_path = model_paths["generation_model_path"]
        self.generation_tokenizer_path = model_paths["generation_tokenizer_path"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conversation_history = {}
        self.max_history = 5
        self.lock = Lock()
        self.load_models()

    def load_models(self):
        print(f"\nLoading {self.fusion_type} model on {self.device.upper()}...")
        encoders_path = os.path.join(self.output_dir, "label_encoders.json")
        hyperparams_path = os.path.join(self.output_dir, "hyperparameters.json")
        model_path = os.path.join(self.output_dir, "hybrid_model.pth")

        with open(encoders_path, 'r', encoding='utf-8') as f:
            self.label_encoders = json.load(f)
        with open(hyperparams_path, 'r', encoding='utf-8') as f:
            self.hyperparameters = json.load(f)

        self.gpt2_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        if self.gpt2_tokenizer.pad_token is None:
            self.gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if self.distilbert_tokenizer.pad_token is None:
            self.distilbert_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.classification_model = HybridGPT2DistilBERTMultiTask(
            num_intents=len(self.label_encoders["intent_encoder"]),
            num_categories=len(self.label_encoders["category_encoder"]),
            num_ner_labels=len(self.label_encoders["ner_label_encoder"]),
            dropout_rate=self.hyperparameters["dropout_rate"],
            fusion_type=self.fusion_type
        )

        if self.gpt2_tokenizer.pad_token_id is not None:
            self.classification_model.gpt2.resize_token_embeddings(len(self.gpt2_tokenizer))
        self.classification_model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
        self.classification_model.to(self.device)
        self.classification_model.eval()

        try:
            self.generation_model = GPT2LMHeadModel.from_pretrained(self.generation_model_path).to(self.device)
            self.generation_tokenizer = GPT2Tokenizer.from_pretrained(self.generation_tokenizer_path)
            self.generation_tokenizer.pad_token = self.generation_tokenizer.eos_token
            self.generation_tokenizer.add_special_tokens({'additional_special_tokens': ['[INST]', '[RESP]', '[EOS]']})
            self.generation_model.resize_token_embeddings(len(self.generation_tokenizer))
        except Exception as e:
            print(f"Error loading generation model: {e}")
            print("Falling back to default GPT2...")
            self.generation_model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
            self.generation_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.generation_tokenizer.pad_token = self.generation_tokenizer.eos_token
            self.generation_tokenizer.add_special_tokens({'additional_special_tokens': ['[INST]', '[RESP]', '[EOS]']})
            self.generation_model.resize_token_embeddings(len(self.generation_tokenizer))
        self.generation_model.eval()

    def process_input(self, instruction, session_id="default"):
        print(f"\nProcessing input: {instruction} for session {session_id} with {self.fusion_type} model")

        with self.lock:
            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = []

            def run_classification():
                return inference_hybrid(
                    self.classification_model,
                    instruction,
                    self.gpt2_tokenizer,
                    self.distilbert_tokenizer,
                    self.label_encoders,
                    self.hyperparameters["max_length"],
                    self.device
                )

            classification_start = time.time()
            classification_result, classification_memory = get_peak_memory_usage(run_classification, device=self.device)
            classification_time = time.time() - classification_start

            intent = classification_result["intent"]["label"]
            intent_confidence = classification_result["intent"]["confidence"]
            category = classification_result["category"]["label"]
            category_confidence = classification_result["category"]["confidence"]
            entities = classification_result["ner"]
            entities_text = ", ".join([f"{entity['entity']} ({entity['label']})" for entity in entities]) if entities else "none"
            
            # CHANGE: Removed confidence scores from classified_input
            classified_input = (
                f"{instruction} [Classified: Intent is '{intent}', "
                f"Category is '{category}', "
                f"Entities are {entities_text}]"
            )

            def run_generation():
                return generate_response(
                    self.generation_model,
                    self.generation_tokenizer,
                    instruction,
                    classification_result,
                    self.conversation_history[session_id][-self.max_history:],
                    device=self.device
                )

            generation_start = time.time()
            generated_response, generation_memory = get_peak_memory_usage(run_generation, device=self.device)
            generation_time = time.time() - generation_start

            overall_time = classification_time + generation_time
            overall_memory = classification_memory + generation_memory

            self.conversation_history[session_id].append({
                "instruction": instruction,
                "response": generated_response
            })

        return {
            "instruction": instruction,
            "classified_input": classified_input,  # Updated field name
            "response": generated_response,
            "classification": {
                "intent": {"label": intent, "confidence": intent_confidence},
                "category": {"label": category, "confidence": category_confidence},
                "ner": entities
            },
            "classification_time": classification_time,
            "generation_time": generation_time,
            "overall_time": overall_time,
            "memory_usage": overall_memory
        }

    def clear_history(self, session_id="default"):
        with self.lock:
            if session_id in self.conversation_history:
                self.conversation_history[session_id] = []
                print(f"Conversation history cleared for session {session_id}")
            else:
                print(f"No history found for session {session_id}")
    pass