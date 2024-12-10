from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import json
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the pre-trained GPT-2 model
GPT2_MODEL_PATH = './models/gpt2_category_v3'

# Initialize GPT-2 tokenizer and model
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL_PATH)
gpt2_model = GPT2ForSequenceClassification.from_pretrained(GPT2_MODEL_PATH)

# Load and invert category labels for GPT-2
with open(f'{GPT2_MODEL_PATH}/category_labels.json', 'r') as f:
    gpt2_category_labels = json.load(f)
    gpt2_inverted_category_labels = {str(v): k for k, v in gpt2_category_labels.items()}

# Set GPT-2 model to evaluation mode
gpt2_model.eval()

# Load the pre-trained DistilBERT model
DISTILBERT_MODEL_PATH = './models/distilbert_category'

# Initialize DistilBERT tokenizer and model
distilbert_tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT_MODEL_PATH)
distilbert_model = DistilBertForSequenceClassification.from_pretrained(DISTILBERT_MODEL_PATH)

# Load and invert category labels for DistilBERT
with open(f'{DISTILBERT_MODEL_PATH}/config.json', 'r') as f:
    distilbert_config = json.load(f)
    distilbert_inverted_category_labels = {str(v): k for k, v in distilbert_config['label2id'].items()}

# Set DistilBERT model to evaluation mode
distilbert_model.eval()

# Load Gradient Boosting model
GB_MODEL_PATH = './models/gradient_boosting/gradient_boosting_model_category.pkl'
gb_model = joblib.load(GB_MODEL_PATH)

def get_model_predictions(instruction):
    # GPT-2 Prediction
    inputs_gpt2 = gpt2_tokenizer(instruction, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs_gpt2 = gpt2_model(**inputs_gpt2)
    gpt2_preds = outputs_gpt2.logits.detach().numpy().flatten()

    # DistilBERT Prediction
    inputs_distilbert = distilbert_tokenizer(instruction, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs_distilbert = distilbert_model(**inputs_distilbert)
    distilbert_preds = outputs_distilbert.logits.detach().numpy().flatten()

    # Combine both GPT-2 and DistilBERT predictions (flattened)
    return list(gpt2_preds) + list(distilbert_preds)

@app.route('/baseline_category', methods=['POST'])
def baseline_category():
    try:
        # Get input text from request
        data = request.json
        input_text = data.get('text', '')

        # Tokenize input
        inputs = gpt2_tokenizer(input_text, return_tensors='pt', 
                                truncation=True, 
                                max_length=512, 
                                padding=True)

        # Perform inference
        with torch.no_grad():
            outputs = gpt2_model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(predictions, dim=1).item()

        # Check if predicted class index exists in gpt2_inverted_category_labels
        if str(predicted_class) in gpt2_inverted_category_labels:
            class_label = gpt2_inverted_category_labels[str(predicted_class)]
        else:
            class_label = 'Unknown'

        # Return classification result
        return jsonify({
            'class': class_label,
            'probabilities': predictions.tolist()[0]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/distilbert_category', methods=['POST'])
def distilbert_category():
    try:
        # Get input text from request
        data = request.json
        input_text = data.get('text', '')

        # Tokenize input
        inputs = distilbert_tokenizer(input_text, return_tensors='pt', 
                                      truncation=True, 
                                      max_length=512, 
                                      padding=True)

        # Perform inference
        with torch.no_grad():
            outputs = distilbert_model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(predictions, dim=1).item()

        # Check if predicted class index exists in distilbert_inverted_category_labels
        if str(predicted_class) in distilbert_inverted_category_labels:
            class_label = distilbert_inverted_category_labels[str(predicted_class)]
        else:
            class_label = 'Unknown'

        # Return classification result
        return jsonify({
            'class': class_label,
            'probabilities': predictions.tolist()[0]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/gradient_boosting_category', methods=['POST'])
def gradient_boosting_category():
    try:
        # Get input text from request
        data = request.json
        input_text = data.get('text', '')

        # Extract features using GPT-2 and DistilBERT
        features = get_model_predictions(input_text)

        # Perform inference with Gradient Boosting model
        predicted_class = gb_model.predict([features])[0]

        # Check if predicted class index exists in gpt2_inverted_category_labels
        if str(predicted_class) in gpt2_inverted_category_labels:
            class_label = gpt2_inverted_category_labels[str(predicted_class)]
        else:
            class_label = 'Unknown'

        # Return classification result
        return jsonify({
            'class': class_label
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'cli':
        # Command-line interface mode
        input_text = input("Enter text to classify: ")

        # Tokenize input for GPT-2
        inputs = gpt2_tokenizer(input_text, return_tensors='pt', 
                                truncation=True, 
                                max_length=512, 
                                padding=True)

        # Perform inference with GPT-2 model
        with torch.no_grad():
            outputs = gpt2_model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(predictions, dim=1).item()

        # Debug: Print predicted class index for GPT-2 model
        print(f'GPT-2 Model - Predicted class index: {predicted_class}')

        # Check if predicted class index exists in gpt2_inverted_category_labels
        if str(predicted_class) in gpt2_inverted_category_labels:
            class_label = gpt2_inverted_category_labels[str(predicted_class)]
        else:
            class_label = 'Unknown'

        # Print classification result for GPT-2 model
        print(f'GPT-2 Model - Class: {class_label}')
        print('GPT-2 Model - Probabilities:')
        for i, prob in enumerate(predictions.tolist()[0]):
            label = gpt2_inverted_category_labels.get(str(i), 'Unknown')
            print(f'  {label}: {prob}')

        # Tokenize input for DistilBERT
        inputs = distilbert_tokenizer(input_text, return_tensors='pt', 
                                      truncation=True, 
                                      max_length=512, 
                                      padding=True)

        # Perform inference with DistilBERT model
        with torch.no_grad():
            outputs = distilbert_model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(predictions, dim=1).item()

        # Debug: Print predicted class index for DistilBERT model
        print(f'DistilBERT Model - Predicted class index: {predicted_class}')

        # Check if predicted class index exists in distilbert_inverted_category_labels
        if str(predicted_class) in distilbert_inverted_category_labels:
            class_label = distilbert_inverted_category_labels[str(predicted_class)]
        else:
            class_label = 'Unknown'

        # Print classification result for DistilBERT model
        print(f'DistilBERT Model - Class: {class_label}')
        print('DistilBERT Model - Probabilities:')
        for i, prob in enumerate(predictions.tolist()[0]):
            label = distilbert_inverted_category_labels.get(str(i), 'Unknown')
            print(f'  {label}: {prob}')

        # Extract features using GPT-2 and DistilBERT
        features = get_model_predictions(input_text)

        # Perform inference with Gradient Boosting model
        predicted_class_gb = gb_model.predict([features])[0]

        # Debug: Print predicted class index for Gradient Boosting model
        print(f'Gradient Boosting Model - Predicted class index: {predicted_class_gb}')

        # Check if predicted class index exists in gpt2_inverted_category_labels
        if str(predicted_class_gb) in gpt2_inverted_category_labels:
            class_label_gb = gpt2_inverted_category_labels[str(predicted_class_gb)]
        else:
            class_label_gb = 'Unknown'

        # Print classification result for Gradient Boosting model
        print(f'Gradient Boosting Model - Class: {class_label_gb}')
    else:
        # Run Flask app
        app.run(debug=True)