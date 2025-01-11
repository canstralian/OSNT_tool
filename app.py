from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Define the model name (replace with your actual model name)
model_name = "huggingface/transformers"  # Example model name

# Load the tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    print("Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# Add your app logic here (e.g., for inference, etc.)
def predict(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    return outputs

# Example usage
if __name__ == "__main__":
    test_text = "Hello, world!"
    result = predict(test_text)
    print(result)
