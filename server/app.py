from flask import Flask, request, jsonify
# from models.baseline_inference import baseline_infer
from models.hybrid_model import hybrid_infer
from models.generation_inference import generate_text_with_probabilities
from models.baseline_inference import infer
from models.baseline_model import baseline_infer
from datetime import datetime

from flask_cors import CORS
app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/')
def home():
    return "Flask Server is Running!"


@app.route('/hybrid/predict', methods=['POST'])
def hybrid_predict():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        result = hybrid_infer(text)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/baseline/predict', methods=['POST'])
def baesline_predict():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        result = baseline_infer(text)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/hybrid', methods=['POST'])
def hybrid():
    try:
        # Capture the start time
        start_time = datetime.now()
        
        # Step 1: Get JSON data from the request
        data = request.get_json()
        text = data.get('text')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Step 2: Perform inference using the Hybrid Model
        hybrid_result = hybrid_infer(text)
        
        # Extract predictions from the Hybrid Model
        intent = hybrid_result["intent"]["label"]
        category = hybrid_result["category"]["label"]
        ner_entities = hybrid_result["ner"]
        
        # Format NER entities into a readable string
        ner_summary = ", ".join([f"{entity['type']}: {entity['text']}" for entity in ner_entities])
        
        # Step 3: Create a new prompt for the Text Generation Model
        old_prompt = text  # Original user input
        new_prompt = (
            f"Intent: {intent}\n"
            f"Category: {category}\n"
            f"Entities: {ner_summary}\n"
            f"User Input: {text}"
        )
        
        # Step 4: Generate text using the Text Generation Model
        max_length = data.get('max_length', 512)
        num_beams = data.get('num_beams', 5)
        early_stopping = data.get('early_stopping', True)
        generation_result = generate_text_with_probabilities(
            prompt=new_prompt,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=early_stopping
        )
        
        # Capture the end time
        end_time = datetime.now()
        
        # Step 5: Combine results and return as JSON
        result = {
            "old_prompt": old_prompt,  # Original user input
            "new_prompt": new_prompt,  # Prompt constructed from Hybrid Model predictions
            "hybrid_predictions": hybrid_result,
            "generated_text": generation_result["generated_text"],
            "token_probabilities": generation_result["token_probabilities"],
            "weighted_sum": generation_result["weighted_sum"],
            "start_time": start_time.isoformat(),  # Add start time
            "end_time": end_time.isoformat()  # Add end time
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/baseline', methods=['POST'])
def baseline():
    try:
        # Capture the start time
        start_time = datetime.now()
        
        # Step 1: Get JSON data from the request
        data = request.get_json()
        text = data.get('text')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Step 2: Perform inference using the Baseline Model
        baseline_result = baseline_infer(text)
        
        # Extract predictions from the Baseline Model
        intent = baseline_result["intent"]["label"]
        category = baseline_result["category"]["label"]
        ner_entities = baseline_result["ner"]
        
        # Format NER entities into a readable string
        ner_summary = ", ".join([f"{entity['type']}: {entity['text']}" for entity in ner_entities])
        
        # Step 3: Create a new prompt for the Text Generation Model
        old_prompt = text  # Original user input
        new_prompt = (
            f"Intent: {intent}\n"
            f"Category: {category}\n"
            f"Entities: {ner_summary}\n"
            f"User Input: {text}"
        )
        
        # Step 4: Generate text using the Text Generation Model
        max_length = data.get('max_length', 512)
        num_beams = data.get('num_beams', 5)
        early_stopping = data.get('early_stopping', True)
        generation_result = generate_text_with_probabilities(
            prompt=new_prompt,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=early_stopping
        )
        
        # Capture the end time
        end_time = datetime.now()
        # Step 5: Combine results and return as JSON
        result = {
            "old_prompt": old_prompt,  # Original user input
            "new_prompt": new_prompt,  # Prompt constructed from Baseline Model predictions
            "baseline_predictions": baseline_result,
            "generated_text": generation_result["generated_text"],
            "token_probabilities": generation_result["token_probabilities"],
            "weighted_sum": generation_result["weighted_sum"],
            "start_time": start_time.isoformat(),  # Add start time
            "end_time": end_time.isoformat()  # Add end time
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1')