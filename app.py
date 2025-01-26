import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import torch

# Sidebar for user input
st.sidebar.header("Model Configuration")
model_choice = st.sidebar.selectbox("Select a model", [
    "CyberAttackDetection",
    "text2shellcommands",
    "pentest_ai"
])

# Model mapping
model_mapping = {
    "CyberAttackDetection": "Canstralian/CyberAttackDetection",
    "text2shellcommands": "Canstralian/text2shellcommands",
    "pentest_ai": "Canstralian/pentest_ai"
}

# Model name retrieval
model_name = model_mapping.get(model_choice, "Canstralian/CyberAttackDetection")

# Custom class labels for classification models
class_labels = {
    "CyberAttackDetection": {0: "Malicious", 1: "Benign"},
    "pentest_ai": {0: "Vulnerable", 1: "Secure"}
}

# Load model and tokenizer on demand with caching
@st.cache_resource
def load_model(model_name):
    """Load model and tokenizer based on the selected model."""
    try:
        st.info(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Detect model type and load accordingly
        if "seq2seq" in model_name.lower():
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        st.success(f"Model {model_name} loaded successfully!")
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load the model and tokenizer
tokenizer, model = load_model(model_name)

# Helper function to perform text2shellcommands prediction
def predict_shell_command(user_input, tokenizer, model):
    """Generate shell commands from text using the text2shellcommands model."""
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model.generate(**inputs)
    generated_command = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_command

# Helper function to perform classification prediction
def predict_classification(user_input, tokenizer, model):
    """Classify text using the selected classification model."""
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    
    # Get the class label based on the model type
    labels = class_labels.get(model_choice, {0: "Unknown", 1: "Unknown"})
    predicted_label = labels.get(predicted_class, "Unknown")
    
    return predicted_label, logits

# Streamlit UI components
st.title(f"{model_choice} Model")
user_input = st.text_area("Enter text:")

# Model prediction when user input is provided
if user_input and model and tokenizer:
    st.spinner("Processing input...")  # Show spinner while processing
    
    # Predict based on the model choice
    if model_choice == "text2shellcommands":
        generated_command = predict_shell_command(user_input, tokenizer, model)
        st.write(f"Generated Shell Command: `{generated_command}`")
    
    else:
        predicted_class, logits = predict_classification(user_input, tokenizer, model)
        st.write(f"Predicted Class: {predicted_class}")
        st.write(f"Logits: {logits}")

else:
    st.info("Please enter some text for prediction.")

# Debugging section
if st.checkbox("Show Debugging Info"):
    st.write(f"Model: {model_name}")
    st.write(f"Tokenizer: {tokenizer}")
    if model:
        st.write(f"Model Type: {type(model)}")
    else:
        st.warning("Model not loaded successfully.")
