---
title: OSINT Tool
emoji: 🏢
colorFrom: gray
colorTo: indigo
sdk: streamlit
sdk_version: 1.41.1
app_file: app.py
pinned: false
license: mit
---

# OSINT Tool & GitHub Repository Analysis

## Overview
This project is designed to perform Open Source Intelligence (OSINT) analysis on GitHub repositories and fetch titles from URLs. It also provides functionalities to upload datasets in CSV format for fine-tuning machine learning models. Currently, it supports fine-tuning models like `distilbert-base-uncased` for sequence classification tasks.

## Features
- **GitHub Repository Analysis**: Analyze GitHub repositories by entering the repository owner and name.
- **URL Title Fetcher**: Fetch titles from given URLs.
- **Dataset Upload & Model Fine-Tuning**: Upload CSV files for fine-tuning models and perform sequence classification tasks.

## Prerequisites
Before running the project, make sure you have the following dependencies installed:

- Python 3.6 or higher
- PyTorch (for model fine-tuning)
- Hugging Face Transformers
- Other dependencies listed in `requirements.txt`

## Badges
![Build Status](https://img.shields.io/github/workflow/status/canstralian/osint-tool/CI)
![Code Coverage](https://img.shields.io/codecov/c/github/canstralian/osint-tool)
![License](https://img.shields.io/github/license/canstralian/osint-tool)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<username>/<repository>.git
cd <repository>
```

### 2. Create and activate a virtual environment

For Linux/MacOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

For Windows:

```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install PyTorch (if not already installed)

```bash
pip install torch
```

## Usage

### Running the Application

To run the application, simply execute the following command:

```bash
python app.py
```

### Features

#### GitHub Repository Analysis

1. Enter the GitHub repository owner and name.
2. The application will fetch details and analyze the repository.

#### URL Title Fetcher

1. Enter a URL, and the application will fetch the title of the page.

#### Dataset Upload & Model Fine-Tuning

1. Upload a CSV file (limit 200MB).
2. Select the model for fine-tuning (e.g., `distilbert-base-uncased`).
3. Fine-tune the model for sequence classification tasks.

### Example CSV Format for Fine-Tuning

```csv
text,label
"This is an example sentence.",1
"This is another example.",0
```

### Running the Model Fine-Tuning

```bash
python fine_tune.py --model distilbert-base-uncased --data dataset.csv
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your suggestions or fixes.

### Steps to Contribute

1. Fork the repository.
2. Clone your fork: `git clone https://github.com/<your-username>/<repository>.git`
3. Create a branch: `git checkout -b feature/your-feature`
4. Commit your changes: `git commit -m "Add feature"`
5. Push to the branch: `git push origin feature/your-feature`
6. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for providing the pre-trained models.
- [PyTorch](https://pytorch.org/) for deep learning frameworks.
- [Shields.io](https://shields.io/) for providing awesome badges.