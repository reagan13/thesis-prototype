from flask import Flask, request, jsonify
from flask_cors import CORS
from services.inference_service_hybrid import ConversationalInferenceService
from services.inference_service_baseline import BaselineInferenceService

def process_request(service, data):
    print("Received data:", data)  # Debug log to verify incoming data
    if not data or 'text' not in data:  # CHANGE: Updated from 'message' to 'text'
        print("Error: No text provided in data:", data)
        return jsonify({"error": "No text provided"}), 400

    instruction = data['text']  # CHANGE: Updated from 'message' to 'text'
    session_id = data.get('session_id', 'default')
    print("Processing:", instruction, "Session:", session_id)

    if instruction.lower() == "clear":
        service.clear_history(session_id)
        return jsonify({"response": "Conversation history cleared", "session_id": session_id})

    try:
        result = service.process_input(instruction, session_id)
        return jsonify({
            "response": result["response"],
            "classified_input": result["classified_input"],
            "session_id": session_id,
            "classification": {
                "intent": result["classification"]["intent"],
                "category": result["classification"]["category"],
                "ner": result["classification"]["ner"]
            },
            "metrics": {
                "classification_time": result["classification_time"],
                "generation_time": result["generation_time"],
                "overall_time": result["overall_time"],
                "memory_usage": result["memory_usage"]
            }
        })
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

def create_app(model_paths):
    app = Flask(__name__)
    CORS(app)

    baseline_service = BaselineInferenceService(model_paths)
    concat_service = ConversationalInferenceService(model_paths, fusion_type="concat")
    crossattention_service = ConversationalInferenceService(model_paths, fusion_type="crossattention")
    dense_service = ConversationalInferenceService(model_paths, fusion_type="dense")

    @app.route('/baseline', methods=['POST'])
    def baseline_chat():
        return process_request(baseline_service, request.get_json())

    @app.route('/concat', methods=['POST'])
    def concat_chat():
        return process_request(concat_service, request.get_json())

    @app.route('/crossattention', methods=['POST'])
    def crossattention_chat():
        return process_request(crossattention_service, request.get_json())

    @app.route('/dense', methods=['POST'])
    def dense_chat():
        return process_request(dense_service, request.get_json())

    print("Registered routes:", [rule.rule for rule in app.url_map.iter_rules()])
    return app