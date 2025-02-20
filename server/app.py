from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import json
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the pre-trained GPT-2 model for category
GPT2_MODEL_PATH = './models/gpt2_category_v3'

# Initialize GPT-2 tokenizer and model for category
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL_PATH)
gpt2_model = GPT2ForSequenceClassification.from_pretrained(GPT2_MODEL_PATH)

# Load and invert category labels for GPT-2
with open(f'{GPT2_MODEL_PATH}/category_labels.json', 'r') as f:
    gpt2_category_labels = json.load(f)
    gpt2_inverted_category_labels = {str(v): k for k, v in gpt2_category_labels.items()}

# Set GPT-2 model to evaluation mode
gpt2_model.eval()

# Load the pre-trained DistilBERT model for category
DISTILBERT_MODEL_PATH = './models/distilbert_category'

# Initialize DistilBERT tokenizer and model for category
distilbert_tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT_MODEL_PATH)
distilbert_model = DistilBertForSequenceClassification.from_pretrained(DISTILBERT_MODEL_PATH)

# Load and invert category labels for DistilBERT
with open(f'{DISTILBERT_MODEL_PATH}/config.json', 'r') as f:
    distilbert_config = json.load(f)
    distilbert_inverted_category_labels = {str(v): k for k, v in distilbert_config['label2id'].items()}

# Set DistilBERT model to evaluation mode
distilbert_model.eval()

# Load Gradient Boosting model for category
GB_MODEL_PATH = './models/gradient_boosting/gradient_boosting_model_category.pkl'
gb_model = joblib.load(GB_MODEL_PATH)

# Load and invert category labels for Gradient Boosting model from config.json
with open(f'{DISTILBERT_MODEL_PATH}/config.json', 'r') as f:
    distilbert_config = json.load(f)
    gb_inverted_category_labels = {str(v): k for k, v in distilbert_config['label2id'].items()}

# Load the pre-trained GPT-2 model for intent
GPT2_INTENT_MODEL_PATH = './models/gpt2_intent_v3'

# Load Gradient Boosting model for intent
GB_INTENT_MODEL_PATH = './models/gradient_boosting/gradient_boosting_model_intent.pkl'
gb_intent_model = joblib.load(GB_INTENT_MODEL_PATH)

# Initialize GPT-2 tokenizer and model for intent
gpt2_intent_tokenizer = GPT2Tokenizer.from_pretrained(GPT2_INTENT_MODEL_PATH)
gpt2_intent_model = GPT2ForSequenceClassification.from_pretrained(GPT2_INTENT_MODEL_PATH)

# Load and invert intent labels for GPT-2
with open(f'{GPT2_INTENT_MODEL_PATH}/intent_labels.json', 'r') as f:
    gpt2_intent_labels = json.load(f)
    gpt2_inverted_intent_labels = {str(v): k for k, v in gpt2_intent_labels.items()}

# Set GPT-2 intent model to evaluation mode
gpt2_intent_model.eval()

# Load the pre-trained DistilBERT model for intent
DISTILBERT_INTENT_MODEL_PATH = './models/distilbert_intent'

# Initialize DistilBERT tokenizer and model for intent
distilbert_intent_tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT_INTENT_MODEL_PATH)
distilbert_intent_model = DistilBertForSequenceClassification.from_pretrained(DISTILBERT_INTENT_MODEL_PATH)

# Load and invert intent labels for DistilBERT
with open(f'{DISTILBERT_INTENT_MODEL_PATH}/config.json', 'r') as f:
    distilbert_intent_config = json.load(f)
    distilbert_inverted_intent_labels = {str(v): k for k, v in distilbert_intent_config['label2id'].items()}

# Load and invert intent labels for Gradient Boosting model from config.json
with open(f'{DISTILBERT_INTENT_MODEL_PATH}/config.json', 'r') as f:
    distilbert_intent_config = json.load(f)
    gb_inverted_intent_labels = {str(v): k for k, v in distilbert_intent_config['label2id'].items()}

# Set DistilBERT intent model to evaluation mode
distilbert_intent_model.eval()

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

def get_intent_predictions(instruction):
    # GPT-2 Intent Prediction
    inputs_gpt2 = gpt2_intent_tokenizer(instruction, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs_gpt2 = gpt2_intent_model(**inputs_gpt2)
    gpt2_preds = outputs_gpt2.logits.detach().numpy().flatten()

    # DistilBERT Intent Prediction
    inputs_distilbert = distilbert_intent_tokenizer(instruction, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs_distilbert = distilbert_intent_model(**inputs_distilbert)
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
        probabilities = gb_model.predict_proba([features])[0]

        print(predicted_class,"asdasd")
        print(probabilities,"pro")
        # Check if predicted class index exists in gb_inverted_category_labels
        if str(predicted_class) in gb_inverted_category_labels:
            class_label = gb_inverted_category_labels[str(predicted_class)]
        else:
            class_label = 'Unknown'

        # Return classification result
        return jsonify({
            'class': class_label,
            'probabilities': probabilities.tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/baseline_intent', methods=['POST'])
def baseline_intent():
    try:
        # Get input text from request
        data = request.json
        input_text = data.get('text', '')

        # Tokenize input
        inputs = gpt2_intent_tokenizer(input_text, return_tensors='pt', 
                                       truncation=True, 
                                       max_length=512, 
                                       padding=True)

        # Perform inference
        with torch.no_grad():
            outputs = gpt2_intent_model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(predictions, dim=1).item()

        # Check if predicted class index exists in gpt2_inverted_intent_labels
        if str(predicted_class) in gpt2_inverted_intent_labels:
            class_label = gpt2_inverted_intent_labels[str(predicted_class)]
        else:
            class_label = 'Unknown'

        # Return classification result
        return jsonify({
            'class': class_label,
            'probabilities': predictions.tolist()[0]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/distilbert_intent', methods=['POST'])
def distilbert_intent():
    try:
        # Get input text from request
        data = request.json
        input_text = data.get('text', '')

        # Tokenize input
        inputs = distilbert_intent_tokenizer(input_text, return_tensors='pt', 
                                             truncation=True, 
                                             max_length=512, 
                                             padding=True)

        # Perform inference
        with torch.no_grad():
            outputs = distilbert_intent_model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(predictions, dim=1).item()

        # Check if predicted class index exists in distilbert_inverted_intent_labels
        if str(predicted_class) in distilbert_inverted_intent_labels:
            class_label = distilbert_inverted_intent_labels[str(predicted_class)]
        else:
            class_label = 'Unknown'

        # Return classification result
        return jsonify({
            'class': class_label,
            'probabilities': predictions.tolist()[0]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/gradient_boosting_intent', methods=['POST'])
def gradient_boosting_intent():
    try:
        # Get input text from request
        data = request.json
        input_text = data.get('text', '')

        # Extract features using GPT-2 and DistilBERT for intent
        features_intent = get_intent_predictions(input_text)

        # Perform inference with Gradient Boosting intent model
        probabilities_gb_intent = gb_intent_model.predict_proba([features_intent])[0]
        predicted_class_gb_intent = np.argmax(probabilities_gb_intent)

        # Check if predicted class index exists in gb_inverted_intent_labels
        if str(predicted_class_gb_intent) in gb_inverted_intent_labels:
            class_label_gb_intent = gb_inverted_intent_labels[str(predicted_class_gb_intent)]
        else:
            class_label_gb_intent = 'Unknown'

        # Return classification result
        return jsonify({
            'class': class_label_gb_intent,
            'probabilities': probabilities_gb_intent.tolist()
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

        # Extract features using GPT-2 and DistilBERT for category
        features = get_model_predictions(input_text)

        # Perform inference with Gradient Boosting category model
        predicted_class_gb = gb_model.predict([features])[0]
        probabilities_gb = gb_model.predict_proba([features])[0]       

        # Print classification result for Gradient Boosting category model
        print(f'Gradient Boosting Category Model - Class: {predicted_class_gb}')
        print('Gradient Boosting Category Model - Probabilities:')
        for i, prob in enumerate(probabilities_gb):
            label = gb_inverted_cawtegory_labels.get(str(i), 'Unknown')
            print(f'  {label}: {prob}')

        # Extract features using GPT-2 and DistilBERT for intent
        features_intent = get_intent_predictions(input_text)

        # Perform inference with Gradient Boosting intent model
        predicted_class_gb_intent = gb_intent_model.predict([features_intent])[0]
        probabilities_gb_intent = gb_intent_model.predict_proba([features_intent])[0]


        # Print classification result for Gradient Boosting intent model
        print(f'Gradient Boosting Intent Model - Class: {predicted_class_gb_intent}')
        print('Gradient Boosting Intent Model - Probabilities:')
        for i, prob in enumerate(probabilities_gb_intent):
            label = gb_inverted_intent_labels.get(str(i), 'Unknown')
            print(f'  {label}: {prob}')
    else:
        # Run Flask app
        app.run(debug=True)