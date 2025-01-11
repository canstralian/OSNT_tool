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

# Define the model names
model_mapping = {
    "CyberAttackDetection": "Canstralian/CyberAttackDetection",
    "text2shellcommands": "Canstralian/text2shellcommands",
    "pentest_ai": "Canstralian/pentest_ai"
}

model_name = model_mapping.get(model_choice, "Canstralian/CyberAttackDetection")

# Load model and tokenizer on demand
@st.cache_resource
def load_model(model_name):
    try:
        # Load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_name == "Canstralian/text2shellcommands":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load the model and tokenizer
tokenizer, model = load_model(model_name)

# Input text box in the main panel
st.title(f"{model_choice} Model")
user_input = st.text_area("Enter text:")

# Make prediction if user input is provided
if user_input and model and tokenizer:
    if model_choice == "text2shellcommands":
        # For text2shellcommands model, generate shell commands
        inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model.generate(**inputs)
        generated_command = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write(f"Generated Shell Command: {generated_command}")
    
    else:
        # For CyberAttackDetection and pentest_ai models, perform classification
        inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()
        st.write(f"Predicted Class: {predicted_class}")
        st.write(f"Logits: {logits}")

else:
    st.info("Please enter some text for prediction.")
