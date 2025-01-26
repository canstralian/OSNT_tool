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

## Detailed Explanations and Examples

### GitHub Repository Analysis

The GitHub Repository Analysis feature allows you to gather various statistics and information about a GitHub repository. This includes the number of stars, forks, open issues, pull requests, license type, last commit date, and workflow status.

#### Example Usage

1. Enter the GitHub repository owner and name (e.g., `Chemically-Motivated-Solutions/OSINT_Tool`).
2. The application will display the following information:
   - Stars: 10
   - Forks: 5
   - Open Issues: 2
   - Open Pull Requests: 1
   - License: MIT
   - Last Commit: 2023-08-15
   - Workflow Status: Success

### URL Title Fetcher

The URL Title Fetcher feature allows you to fetch the title of a web page by entering its URL. This can be useful for quickly identifying the content of a web page.

#### Example Usage

1. Enter a URL (e.g., `https://example.com`).
2. The application will display the title of the page (e.g., `Example Domain`).

### Dataset Upload & Model Fine-Tuning

The Dataset Upload & Model Fine-Tuning feature allows you to upload a CSV file containing text data and labels for fine-tuning a machine learning model. This can be useful for customizing a pre-trained model to your specific use case.

#### Example Usage

1. Upload a CSV file with the following format:

```csv
text,label
"This is an example sentence.",1
"This is another example.",0
```

2. Select the model for fine-tuning (e.g., `distilbert-base-uncased`).
3. Fine-tune the model for sequence classification tasks by running the following command:

```bash
python fine_tune.py --model distilbert-base-uncased --data dataset.csv
```

## How to Contribute

We welcome contributions from the community! If you would like to contribute to this project, please follow the guidelines below.

### Submitting Issues

If you encounter any issues or have suggestions for improvements, please open an issue on the GitHub repository. Provide as much detail as possible, including steps to reproduce the issue if applicable.

### Submitting Pull Requests

If you would like to contribute code to the project, please follow these steps:

1. Fork the repository.
2. Clone your fork: `git clone https://github.com/<your-username>/<repository>.git`
3. Create a branch: `git checkout -b feature/your-feature`
4. Make your changes and commit them: `git commit -m "Add feature"`
5. Push to the branch: `git push origin feature/your-feature`
6. Open a pull request on the original repository.

Please ensure that your code follows the project's coding standards and includes appropriate tests.

## Changelog

All notable changes to this project will be documented in the `CHANGELOG.md` file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### [Unreleased]

- Added detailed explanations and examples to the `README.md`.
- Included a section on how to contribute to the project.
- Added a `CHANGELOG.md` file to document changes and updates to the project over time.
