from flask import Flask, request, jsonify
# from models.baseline_inference import baseline_infer
from models.hybrid_inference import hybrid_infer
from models.generation_inference import generate_text_with_probabilities

app = Flask(__name__)

@app.route('/')
def home():
    return "Flask Server is Running!"

# @app.route('/baseline/predict', methods=['POST'])
# def baseline_predict():
#     try:
#         # Get JSON data from the request
#         data = request.get_json()
#         text = data.get('text')

#         if not text:
#             return jsonify({'error': 'No text provided'}), 400

#         # Perform inference using the Baseline Model
#         result = baseline_infer(text)

#         # Return the result as JSON
#         return jsonify(result)

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


@app.route('/hybrid/predict', methods=['POST'])
def hybrid_predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        text = data.get('text')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Perform inference using the Hybrid Model
        result = hybrid_infer(text)

        # Return the result as JSON
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/textgen/generate', methods=['POST'])
def textgen_generate():
    try:
        # Get JSON data from the request
        data = request.get_json()
        prompt = data.get('prompt')

        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400

        # Optional parameters
        max_length = data.get('max_length', 512)
        num_beams = data.get('num_beams', 5)
        early_stopping = data.get('early_stopping', True)

        # Generate text
        result = generate_text_with_probabilities(
            prompt=prompt,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=early_stopping
        )

        # Return the result as JSON
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    


@app.route('/predict_and_generate', methods=['POST'])
def predict_and_generate():
    try:
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

        # Step 5: Combine results and return as JSON
        result = {
            "old_prompt": old_prompt,  # Original user input
            "new_prompt": new_prompt,  # Prompt constructed from Hybrid Model predictions
            "hybrid_predictions": hybrid_result,
            "generated_text": generation_result["generated_text"],
            "token_probabilities": generation_result["token_probabilities"],
            "weighted_sum": generation_result["weighted_sum"]
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1')