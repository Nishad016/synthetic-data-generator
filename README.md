---
title: Synthetic Data Generator
emoji: 
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

#  Synthetic Data Generator

An LLM-powered application for generating realistic synthetic datasets.

The application uses Meta's Llama 3.1 8B Instruct model through Hugging Face
Inference Providers and provides a Gradio interface for dataset generation.

## Features

- Generate synthetic datasets from natural-language requirements
- Custom JSON schema
- Multiple generation strategies
- Adjustable temperature
- JSON validation
- Pandas DataFrame preview
- Raw JSON output
- Hugging Face deployment

## Architecture

```text
User
  ↓
Gradio UI
  ↓
Prompt Builder
  ↓
Hugging Face InferenceClient
  ↓
Llama 3.1 8B Instruct
  ↓
JSON Parsing
  ↓
Pandas DataFrame
  ↓
Dataset Preview
```
## Technologies

- Python
- Gradio
- Hugging Face Hub
- Llama 3.1 8B Instruct
- Pandas

## Local Setup
### Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/synthetic-data-generator.git
cd synthetic-data-generator
```
### Create a virtual environment:
```bash
python -m venv .venv
```
### Activate it:

### Windows:
```bash
.venv\Scripts\activate
```
### Install dependencies:
```bash
pip install -r requirements.txt
```
### Create .env
```bash
HF_TOKEN=your_huggingface_token
HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
```
### Run:
```bash
python app.py
```
