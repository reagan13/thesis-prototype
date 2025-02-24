import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Path to the saved model and tokenizer
MODEL_PATH = "text-generation"

# Load the model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Ensure pad_token is set (important for generation)
tokenizer.pad_token = tokenizer.eos_token
def generate_text_with_probabilities(prompt, max_length=512, num_beams=5, early_stopping=True):
    """
    Generate text based on the input prompt and compute token probabilities.
    Args:
        prompt (str): The input text/prompt for generation.
        max_length (int): Maximum length of the generated text.
        num_beams (int): Number of beams for beam search (set to 1 for greedy decoding).
        early_stopping (bool): Whether to stop generation early when the model predicts an end token.
    Returns:
        dict: A dictionary containing the generated text, token probabilities, and weighted sum.
    """
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    # Generate text with output scores
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=early_stopping,
            output_scores=True,  # Return token probabilities
            return_dict_in_generate=True  # Return a dictionary with additional info
        )

    # Decode the generated output
    full_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    # Exclude the prompt from the generated text
    response_text = full_text[len(prompt):].strip()

    # Compute token probabilities
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )
    token_probabilities = torch.exp(transition_scores).tolist()

    # Calculate the weighted sum of token probabilities
    weighted_sum = sum(token_probabilities[0]) / len(token_probabilities[0])

    return {
        "generated_text": response_text,  # Only the response text
        "token_probabilities": token_probabilities[0],
        "weighted_sum": round(weighted_sum, 4)
    }