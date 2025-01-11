import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset

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
        st.write(f"Fetching URL: {url} - Status Code: {response.status_code}")
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
    st.title("OSINT Tool")
    
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

    st.write("### URL Title Fetcher")
    url = st.text_input("Enter a URL to fetch its title:")
    if url:
        title = fetch_page_title(url)
        st.write(f"Title: {title}")
    
    st.write("### Dataset Upload & Model Fine-Tuning")
    dataset_file = st.file_uploader("Upload a CSV file for fine-tuning", type=["csv"])
    if dataset_file:
        df = pd.read_csv(dataset_file)
        st.dataframe(df.head())

    st.write("Select a model for fine-tuning:")
    model_name = st.selectbox("Model", ["bert-base-uncased", "distilbert-base-uncased"])

    if st.button("Fine-tune Model"):
        if dataset_file:
            dataset = Dataset.from_pandas(df)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)

            def tokenize_function(examples):
                return tokenizer(examples['text'], padding="max_length", truncation=True)

            tokenized_datasets = dataset.map(tokenize_function, batched=True)
            training_args = TrainingArguments(output_dir="./results", num_train_epochs=1, per_device_train_batch_size=8)
            trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_datasets)
            trainer.train()
            st.write("Model fine-tuned successfully!")

    # Load and display OSINT dataset
    st.write("### OSINT Dataset")
    dataset = load_dataset("originalbox/osint")  # Replace with the correct dataset name
    
    # Convert to pandas DataFrame for display
    df = dataset['train'].to_pandas()  # Make sure to use the appropriate split ('train', 'test', etc.)
    st.write(df.head())

if __name__ == "__main__":
    main()
