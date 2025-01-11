import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Sidebar for user input
st.sidebar.header("Model Configuration")
model_name = st.sidebar.text_input("Enter model name", "huggingface/transformers")

# Load model and tokenizer on demand
@st.cache_resource
def load_model(model_name):
    try:
        # Load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load the model and tokenizer
tokenizer, model = load_model(model_name)

# Input text box in the main panel
st.title("Text Classification with Hugging Face Models")
user_input = st.text_area("Enter text for classification:")

# Make prediction if user input is provided
if user_input and model and tokenizer:
    inputs = tokenizer(user_input, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Display results (e.g., classification logits)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Logits: {logits}")
else:
    st.info("Please enter some text to classify.")

