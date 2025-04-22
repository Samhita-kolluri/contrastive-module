# Fine-Tuned Stance Model — Contrastive Embedding
This document outlines the process to fine-tune a stance-aware LLM or embedding model using SNLI-style contradiction data for improved stance opposition scoring in the Contrastive Ideas Search Module.

## 🔧 Requirements

Install the required libraries (if not already installed):

```bash
pip install sentence-transformers datasets torch
```

## 📦 Dataset Overview
We use a JSON dataset based on SNLI-style structure, with each entry representing a premise–hypothesis pair that reflects a stance contrast.

**Dataset Source**: [`Samhita-kolluri/snli-contrastive-json-dataset`](https://huggingface.co/datasets/Samhita-kolluri/snli-contrastive-json-dataset)

Each entry in JSON format:

```json
{
  "premise": "Children smiling and waving at camera",
  "hypothesis": "The kids are frowning",
  "id": 1
}
```
## 🧠 Objective
Train a model to recognize stance opposition between two texts. Specifically:

Given a premise, model predicts if the hypothesis holds an opposing stance.

Supervision comes from natural contradictions (negations, opposites, stance flips).

- **Base Model**: `meta-llama/Llama-3.2-3B-Instruct`
- **Fine-Tuned Model**: [`Samhita-kolluri/llama-contrastive-module-stance`](https://huggingface.co/Samhita-kolluri/llama-contrastive-module-stance)  (after fine-tuning on SNLI-style contrastive data) 
- **Framework**: sentence-transformers

## 🛠️ Fine-Tuning Process
- **Load Base Model**:  The base model used for fine-tuning is `meta-llama/Llama-3.2-3B-Instruct`, which is a large instruction-tuned model suitable for understanding and generating human-like responses.
- **Dataset Preparation**:
The dataset [`Samhita-kolluri/snli-contrastive-json-dataset`](https://huggingface.co/datasets/Samhita-kolluri/snli-contrastive-json-dataset)consists of pairs of premise and hypothesis sentences. The fine-tuning process involves training the model to recognize stance opposition between these pairs.  
- **Fine-Tuning**:
Fine-tuning is carried out using the Trainer class from Hugging Face's transformers library and it is saved in this [`Samhita-kolluri/llama-contrastive-module-stance`](https://huggingface.co/Samhita-kolluri/llama-contrastive-module-stance) 
- **Evaluation**:
After fine-tuning, the model's performance is evaluated using contrastive metrics, particularly stance opposition detection.

## 📈 After Fine-tuning

- Using fine-tuned stance LLM and integrating with `ContrastiveScorer` in `scoring.py`
- Optimizing via `benchmarking/rlhf_tuning.py` (RLHF tuning)

---

## ✅ Summary

| Component         | Value                                           |
|-------------------|-------------------------------------------------|
| **Base Model**    | `meta-llama/Llama-3.2-3B-Instruct`             |
| **Fine-Tuned Model** | `Samhita-kolluri/llama-contrastive-module-stance` |
| **Dataset**       | `Samhita-kolluri/snli-contrastive-json-dataset` |
| **Output Path**   | `models/stance-model/`                          |
| **Usage**         | Used in stance scoring and contrastive search   |
| **Framework**     | `sentence-transformers`                         |
