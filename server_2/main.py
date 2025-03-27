import torch
import os
from api.app import create_app  # Import the factory function

model_paths = {
    "baseline_model_dir": os.path.abspath("../Baseline_freeze_v1"),
    "concat_model_dir": os.path.abspath("../Hybrid_Concat_Freeze"),
    "crossattention_model_dir": os.path.abspath("../Hybrid_Cross_Attention_Freeze"),
    "dense_model_dir": os.path.abspath("../Hybrid_Dense_Layer_Freeze"),
    "generation_model_path": os.path.abspath("../text_generation_results_03-09-25/model"),
    "generation_tokenizer_path": os.path.abspath("../text_generation_results_03-09-25/tokenizer")
}

if __name__ == "__main__":
    device_info = "CUDA" if torch.cuda.is_available() else "CPU"
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Running on: {device_info}")
    app = create_app(model_paths)  # Create the app instance here
    app.run(host='0.0.0.0', port=5000, debug=True)