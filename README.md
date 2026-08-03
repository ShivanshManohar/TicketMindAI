# TicketMindAI

> An end-to-end AI Engineering project that explores whether LoRA fine-tuning can outperform prompt engineering for structuring customer support tickets.

---

## 📌 Problem Statement

Customer support teams receive thousands of unstructured tickets every day.

These tickets are difficult to route, prioritize and summarize automatically.

TicketMindAI converts raw customer support messages into structured incident reports using Large Language Models (LLMs).

The project first establishes a prompt-engineered baseline (TicketMind Sentinel), then fine-tunes an open-source LLM using QLoRA and compares both approaches through a custom evaluation benchmark.

Thus, TicketMindAI is an AI-powered middleware for enterprise customer support!

---

## 🎯 Project Goals

- Build a prompt-engineered baseline
- Fine-tune an open-source LLM using LoRA / QLoRA
- Compare baseline vs fine-tuned model
- Build a custom evaluation benchmark
- Detect catastrophic forgetting
- Deploy both models using FastAPI
- Automate evaluation with GitHub Actions

---

## 🛠 Planned Tech Stack

### LLM

- Llama 3.1
- LoRA
- QLoRA
- PEFT

### Machine Learning

- Hugging Face Transformers
- Datasets
- Unsloth
- Weights & Biases

### Backend

- FastAPI

### MLOps

- GitHub Actions
- Automated Regression Testing

---

## 🚧 Current Status

🟢 Phase 1 — Project Planning & Schema Design