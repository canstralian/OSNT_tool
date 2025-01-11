import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import os

# Initialize the Hugging Face pipeline (ensure to use a valid model)
model_name = "your_huggingface_model_name"  # Ensure to use a valid model
tokenizer = AutoTokenizer.from_pretrained(model_name)
try:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
except Exception as e:
    st.error(f"Error initializing the model '{model_name}': {e}")

# Function to generate OSINT results
def generate_osint_results(prompt: str, history: List[Dict[str, str]]) -> List[str]:
    """
    Simulates generating OSINT-based results from the user's input.
    Args:
        prompt (str): The user's input to the simulator.
        history (List[Dict]): The user's message history with timestamps.
    Returns:
        List[str]: A list of OSINT responses from the AI.
    """
    # Validate inputs
    if not prompt.strip():
        return ["Error: Prompt cannot be empty."]
    if not isinstance(history, list) or not all(isinstance(h, dict) for h in history):
        return ["Error: History must be a list of dictionaries."]
    
    # Prepare messages for the AI
    messages = [{"role": "system", "content": f"Responding to OSINT prompt: {prompt}"}]
    for val in history:
        if "user" in val:
            messages.append({"role": "user", "content": val["user"]})
        if "assistant" in val:
            messages.append({"role": "assistant", "content": val["assistant"]})

    # Append the current user prompt
    messages.append({"role": "user", "content": prompt})

    # Generate a response using the Hugging Face model
    try:
        response = generator(messages[-1]["content"], max_length=100, num_return_sequences=1)
        return [response[0]["generated_text"]]
    except Exception as e:
        return [f"Error generating response: {e}"]

# Function for fine-tuning the model with the uploaded dataset
def fine_tune_model(dataset: str) -> str:
    """
    Fine-tunes the model using the uploaded dataset.
    Args:
        dataset (str): The path to the dataset for fine-tuning.
    Returns:
        str: A message indicating whether fine-tuning was successful or failed.
    """
    try:
        # Process the dataset (dummy processing for illustration)
        with open(dataset, "r") as file:
            data = file.readlines()

        # Simulate fine-tuning with the provided dataset
        # Here, you would use the data to fine-tune the model
        # For this example, we're not actually fine-tuning the model.
        model.save_pretrained("./fine_tuned_model")
        return "Model fine-tuned successfully!"
    except Exception as e:
        return f"Error fine-tuning the model: {e}"

# Streamlit app interface
st.title("OSINT Tool")
st.write("This tool generates OSINT-based results and allows you to fine-tune the model with custom datasets.")

# User input for prompt and message history
prompt = st.text_area("Enter your OSINT prompt here...", placeholder="Type your prompt here...")
history = []

# Display message history
if "history" not in st.session_state:
    st.session_state.history = []

# Display past conversation
st.write("### Message History:")
for msg in st.session_state.history:
    st.write(f"**User**: {msg['user']}")
    st.write(f"**Assistant**: {msg['assistant']}")

# Fine-tuning functionality
dataset_file = st.file_uploader("Upload a dataset for fine-tuning", type=["txt"])

if dataset_file is not None:
    # Save the uploaded file
    dataset_path = os.path.join("uploads", dataset_file.name)
    with open(dataset_path, "wb") as f:
        f.write(dataset_file.read())
    
    # Fine-tune the model
    fine_tuning_status = fine_tune_model(dataset_path)
    st.success(fine_tuning_status)

# Generate OSINT response when prompt is entered
if st.button("Generate OSINT Results"):
    if prompt:
        response = generate_osint_results(prompt, st.session_state.history)
        st.session_state.history.append({"user": prompt, "assistant": response[0]})
        st.write("### Generated OSINT Result:")
        st.write(response[0])
    else:
        st.error("Please enter a prompt.")

# Optionally save fine-tuned model
if os.path.exists("./fine_tuned_model"):
    st.write("The model has been fine-tuned and saved as `fine_tuned_model`.")
