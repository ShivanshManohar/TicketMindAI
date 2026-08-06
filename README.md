# TicketMindAI

> An end-to-end AI Engineering project that investigates whether QLoRA fine-tuning can outperform prompt engineering for structured customer support ticket understanding.

---

# 📌 Problem Statement

Customer support teams receive thousands of unstructured tickets every day, making automatic routing, prioritization, and summarization difficult.

TicketMindAI transforms raw customer support messages into structured JSON containing:

- Category
- Subcategory
- Priority
- Sentiment
- Summary

The project first establishes a prompt-engineered baseline, then fine-tunes an open-source LLM using QLoRA and compares both approaches through a custom benchmarking pipeline.

---

# 🎯 Objectives

- Build a high-quality customer support dataset
- Design an automated LLM-based annotation pipeline
- Fine-tune an open-source LLM using QLoRA
- Compare Prompt Engineering vs Fine-tuned LoRA
- Evaluate model quality using custom benchmarks
- Deploy the model through FastAPI
- Serve structured JSON predictions via REST APIs

---

# ⚙️ Tech Stack

## LLM & Fine-Tuning

- Llama 3.2 3B
- QLoRA
- LoRA
- PEFT
- Unsloth

## AI / Machine Learning

- Hugging Face Transformers
- Hugging Face Datasets
- Google Gemini API
- Pandas

## Backend

- FastAPI
- Uvicorn
- Pydantic

## Development

- Python
- Git
- GitHub

---

# 📂 Dataset Pipeline

```text
Original Dataset (24,635 Tickets)
            │
            ▼
Balanced Sampling
(~3,240 Tickets)
            │
            ▼
Gemini Annotation Pipeline
            │
            ▼
Structured Labels
(Category + Subcategory + Priority + Sentiment + Summary)
            │
            ▼
ChatML Conversion
            │
            ▼
QLoRA Fine-Tuning
            │
            ▼
FastAPI Inference API
```

---

# 🚀 Features

- Balanced dataset generation
- Automated batch annotation pipeline
- Prompt-engineered baseline
- ChatML dataset generation
- QLoRA fine-tuning
- FastAPI REST API
- Swagger UI
- Structured JSON inference
- Benchmarking pipeline

---

# 📊 Dataset

Original Dataset

- 24,635 customer support tickets

Training Dataset

- ~3,240 balanced samples
- Automatically annotated using Gemini
- Converted into ChatML format for supervised fine-tuning

Each training sample predicts:

- Category
- Subcategory
- Priority
- Sentiment
- Summary

---

# 🛠 Project Structure

```
api/
data/
evaluation/
scripts/
training/
models/
```

---

# 🚧 Current Status

| Module | Status |
|---------|--------|
| Dataset Cleaning | ✅ Completed |
| Balanced Dataset Generation | ✅ Completed |
| Gemini Annotation Pipeline | ✅ Completed |
| 3K Ticket Annotation | ✅ Completed |
| ChatML Conversion | ⏳ In Progress |
| QLoRA Fine-Tuning | ⏳ In Progress |
| FastAPI Integration | 🟡 Skeleton Ready |
| Model Benchmarking | ⏳ Pending |
| Documentation | 🟡 Ongoing |

---

# 🎯 Expected Outcome

Compare three approaches on the same benchmark:

- Prompt Engineering
- Base Llama 3.2
- Fine-Tuned Llama 3.2 (QLoRA)

using:

- Classification Accuracy
- Precision
- Recall
- F1 Score
- JSON Validity
- Inference Latency

to evaluate whether domain-specific fine-tuning provides measurable improvements over prompt engineering.