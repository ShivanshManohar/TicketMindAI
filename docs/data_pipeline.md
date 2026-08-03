# TicketMindAI Data Pipeline

## Goal

Transform a public customer support dataset into a high-quality dataset suitable for LoRA fine-tuning.

---

## Stage 1 - Raw Dataset

Source:
- Hugging Face

Dataset:
- bitext/Bitext-customer-support-llm-chatbot-training-dataset

Output:
- Raw customer support examples

---

## Stage 2 - Data Cleaning

Tasks:

- Remove duplicate samples
- Remove empty rows
- Normalize whitespace
- Remove invalid examples

Output:
- Clean dataset

---

## Stage 3 - Data Transformation

Input Columns

- instruction
- category
- intent

Output Schema

{
  "categories": [],
  "priority": "",
  "sentiment": "",
  "summary": ""
}

---

## Stage 4 - Dataset Validation

Checks

- Schema validation
- Missing values
- Category consistency
- Intent consistency

---

## Stage 5 - Train / Validation / Test Split

80%

10%

10%

---

## Stage 6 - Benchmark Creation

Create 30–50 manually curated benchmark examples.

These examples will never be used during training.

---

## Stage 7 - LoRA Fine-Tuning

Train using the processed dataset.

---

## Stage 8 - Evaluation

Compare

- Prompt baseline
- Fine-tuned model

Measure

- Accuracy
- JSON validity
- Latency
- Failure cases