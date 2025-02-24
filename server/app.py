from flask import Flask, request, jsonify
import torch
from models.hybrid import load_hybrid_model, load_hybrid_tokenizers, load_label_encoders
from utils.hybrid import run_hybrid_inference, postprocess_ner
from models.text_generation import load_model, load_tokenizer, generate_text, compute_rouge
from transformers import GPT2TokenizerFast, DistilBertTokenizerFast
import requests

# Initialize Flask app
app = Flask(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Hybrid model and tokenizers
hybrid_model_path = "hybrid/hybrid_fusion_multitask_model_v2.pth"
hybrid_tokenizer_dir = "hybrid"
hybrid_encoder_path = "hybrid/label_encoders.json"

hybrid_label_encoders = load_label_encoders(hybrid_encoder_path)
hybrid_gpt2_tokenizer, hybrid_distilbert_tokenizer = load_hybrid_tokenizers(hybrid_tokenizer_dir)
hybrid_model = load_hybrid_model(hybrid_model_path, hybrid_label_encoders, device)



# Paths to the model and tokenizer
model_dir = "text-generation"

# Load the model and tokenizer
model = load_model(model_dir, device)
tokenizer = load_tokenizer(model_dir)


@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        input_text = data.get("text", "")
        reference_text = data.get("reference_text", "")  # Optional reference text for ROUGE-L
        max_length = data.get("max_length", 512)
        num_beams = data.get("num_beams", 5)
        early_stopping = data.get("early_stopping", True)

        if not input_text:
            return jsonify({"error": "Input text is required"}), 400

        # Call the /hybrid endpoint
        hybrid_response = requests.post('http://localhost:5000/hybrid', json={"text": input_text})
        if hybrid_response.status_code != 200:
            return jsonify({"error": "Failed to get response from hybrid endpoint"}), 500

        hybrid_data = hybrid_response.json()
        intent_label = hybrid_data.get("intent", {}).get("label", "")
        category_label = hybrid_data.get("category", {}).get("label", "")
        ner_labels = [ner["label"] for ner in hybrid_data.get("ner", [])]

        # Concatenate intent, category, and ner with the input text
        new_input_text = f"Intent: {intent_label}\nCategory: {category_label}\nNER: {ner_labels}\nText: {input_text}"

        # Generate text using the GPT-2 model
        generated_text = generate_text(
            prompt=new_input_text,
            model=model,
            tokenizer=tokenizer,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=early_stopping,
            device=device
        )

        # Compute ROUGE-L score if reference text is provided
        rouge_l_score = None
        if reference_text:
            rouge_l_score = compute_rouge(generated_text, reference_text)

        # Prepare response
        response = {
            "new_input_text": new_input_text,
            "generated_text": generated_text,
            "intent": intent_label,
            "category": category_label,
            "ner": ner_labels,
            "rouge_l_score": rouge_l_score
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
# Define the /predict endpoint
@app.route('/hybrid', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "Input text is required"}), 400

        # Run inference
        predictions = run_hybrid_inference(
            hybrid_model, text, hybrid_gpt2_tokenizer, hybrid_distilbert_tokenizer, hybrid_label_encoders, device
        )

        # Postprocess NER labels
        ner_results = postprocess_ner(
            text,
            predictions["ner"]["labels"],
            predictions["ner"]["confidences"],
            hybrid_distilbert_tokenizer
        )

        # Prepare response
        response = {
            "intent": predictions["intent"],
            "category": predictions["category"],
            "ner": ner_results["entity_spans"]
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)