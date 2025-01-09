# Fine-Tuning BERT Model for Named Entity Recognition (NER)

This project demonstrates the process of fine-tuning a pre-trained BERT model from Hugging Face for the task of Named Entity Recognition (NER). NER involves identifying and classifying named entities in text, such as names, dates, locations, and organizations. By leveraging the power of the BERT model, which is renowned for its contextual understanding of language, this project aims to build a robust system for accurately tagging entities in a given dataset.

## Process Overview

### 1. Dataset Preparation
The input text is tokenized using the BERT tokenizer, and the corresponding entity labels are mapped to match the tokenized structure. The dataset is then formatted for compatibility with Hugging Face's token classification pipeline.

### 2. Model Configuration
A pre-trained BERT model is loaded, and a classification head for token-level predictions is added. This head enables the model to predict the appropriate entity class for each token.

### 3. Training
The model is fine-tuned on a labeled dataset using a loss function suited for token classification (e.g., CrossEntropyLoss). Techniques like learning rate scheduling and gradient clipping are employed to optimize training.

### 4. Evaluation
Performance metrics such as precision, recall, and F1 score are calculated to assess the model's effectiveness on unseen data.

## Results

Below are the outcomes of the fine-tuned model:

### Evaluation Metrics

| Label     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| geo       | 0.86      | 0.87   | 0.86     | 11,585  |
| gpe       | 0.97      | 0.91   | 0.94     | 3,467   |
| org       | 0.72      | 0.69   | 0.71     | 6,785   |
| per       | 0.79      | 0.80   | 0.79     | 5,270   |
| tim       | 0.84      | 0.84   | 0.84     | 4,457   |
| **micro avg** | **0.83** | **0.82** | **0.82** | **31,564** |
| **macro avg** | **0.83** | **0.82** | **0.83** | **31,564** |
| **weighted avg** | **0.83** | **0.82** | **0.82** | **31,564** |

### Sample Output
The model's output on test data:
![Output](bert-ner/output.png)

## Key Features

- Fine-tunes the BERT model using Hugging Face Transformers.
- Implements token classification for NER tasks.
- Achieves high accuracy with optimized training strategies.

## Tools and Libraries

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- PyTorch
- Scikit-learn
- Pandas
- NumPy