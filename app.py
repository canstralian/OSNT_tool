import yaml
import huggingface_hub
import requests
from bs4 import BeautifulSoup
import pandas as pd
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from components.sidebar import sidebar
from components.chat_box import chat_box
from components.chat_loop import chat_loop
from components.init_state import init_state
from components.prompt_engineering_dashboard import prompt_engineering_dashboard
import streamlit as st

# Access the Hugging Face token
hf_token = st.secrets["HF_TOKEN"]

# Example usage: if you're using the Hugging Face API
from huggingface_hub import login

login(token=hf_token)

# Load config.yaml
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Streamlit page configuration
st.set_page_config(
    page_title="NCTC OSINT AGENT - Fine-tuning Models",
    page_icon="𓃮",
)

# Initialize session state
init_state(st.session_state, config)

# Custom HTML for title styling
html_title = '''
<style>
.stTitle {
  color: #00008B;  /* Deep blue color */
  font-size: 36px;  /* Adjust font size as desired */
  font-weight: bold;  /* Add boldness (optional) */
}
</style>
<h1 class="stTitle">NCTC OSINT AGENT - Fine-tuning AI Models</h1>
'''

# Display HTML title
st.write(html_title, unsafe_allow_html=True)

# OSINT functions
def get_github_stars_forks(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}"
    response = requests.get(url)
    data = response.json()
    return data['stargazers_count'], data['forks_count']

def get_github_issues(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    response = requests.get(url)
    issues = response.json()
    return len(issues)

def get_github_pull_requests(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
    response = requests.get(url)
    pulls = response.json()
    return len(pulls)

def get_github_license(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/license"
    response = requests.get(url)
    data = response.json()
    return data['license']['name']

def get_last_commit(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    response = requests.get(url)
    commits = response.json()
    return commits[0]['commit']['committer']['date']

def get_github_workflow_status(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs"
    response = requests.get(url)
    runs = response.json()
    return runs['workflow_runs'][0]['status'] if runs['workflow_runs'] else "No workflows found"

# Function to fetch page title from a URL
def fetch_page_title(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.string if soup.title else 'No title found'
            return title
        else:
            return f"Error: Received status code {response.status_code}"
    except Exception as e:
        return f"An error occurred: {e}"

# Main Streamlit app
def main():
    # Display Prompt Engineering Dashboard (testing phase)
    prompt_engineering_dashboard(st.session_state, config)

    # Display sidebar and chat box
    sidebar(st.session_state, config)
    chat_box(st.session_state, config)
    chat_loop(st.session_state, config)

    # GitHub OSINT Analysis
    st.write("### GitHub Repository OSINT Analysis")
    st.write("Enter the GitHub repository owner and name:")

    owner = st.text_input("Repository Owner")
    repo = st.text_input("Repository Name")

    if owner and repo:
        stars, forks = get_github_stars_forks(owner, repo)
        open_issues = get_github_issues(owner, repo)
        open_pulls = get_github_pull_requests(owner, repo)
        license_type = get_github_license(owner, repo)
        last_commit = get_last_commit(owner, repo)
        workflow_status = get_github_workflow_status(owner, repo)

        st.write(f"Stars: {stars}, Forks: {forks}")
        st.write(f"Open Issues: {open_issues}, Open Pull Requests: {open_pulls}")
        st.write(f"License: {license_type}")
        st.write(f"Last Commit: {last_commit}")
        st.write(f"Workflow Status: {workflow_status}")

    # URL Title Fetcher
    st.write("### URL Title Fetcher")
    url = st.text_input("Enter a URL to fetch its title:")
    if url:
        title = fetch_page_title(url)
        st.write(f"Title: {title}")

    # Dataset Upload & Model Fine-Tuning Section
    st.write("### Dataset Upload & Model Fine-Tuning")
    dataset_file = st.file_uploader("Upload a CSV file for fine-tuning", type=["csv"])
    
    if dataset_file:
        df = pd.read_csv(dataset_file)
        st.write("Preview of the uploaded dataset:")
        st.dataframe(df.head())

    # Select model for fine-tuning
    st.write("Select a model for fine-tuning:")
    model_name = st.selectbox("Model", ["bert-base-uncased", "distilbert-base-uncased"])

    if st.button("Fine-tune Model"):
        if dataset_file:
            with st.spinner("Fine-tuning in progress..."):
                dataset = Dataset.from_pandas(df)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)

                def tokenize_function(examples):
                    return tokenizer(examples['text'], padding="max_length", truncation=True)

                tokenized_datasets = dataset.map(tokenize_function, batched=True)
                training_args = TrainingArguments(output_dir="./results", num_train_epochs=1, per_device_train_batch_size=8)
                trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_datasets)
                trainer.train()
                
                st.success("Model fine-tuned successfully!")

    # Load and display OSINT dataset
    st.write("### OSINT Dataset")
    dataset = load_dataset("originalbox/osint")  # Replace with the correct dataset name
    
    # Convert to pandas DataFrame for display
    df = dataset['train'].to_pandas()  # Make sure to use the appropriate split ('train', 'test', etc.)
    st.write(df.head())

if __name__ == "__main__":
    main()
