# TicketMindAI

> **An end-to-end AI Engineering project investigating whether QLoRA fine-tuning can outperform prompt engineering for structured customer support ticket understanding.**

TicketMindAI fine-tunes a 3.2B-parameter open-source LLM to transform unstructured customer support tickets into structured outputs containing **Category, Subcategory, Priority, Sentiment, and Summary**.

The project compares a prompt-engineered baseline against a QLoRA fine-tuned model using a dedicated **540-ticket benchmark**, demonstrating a substantial improvement in structured ticket classification accuracy.

---

## 🎯 Project Highlights

* Fine-tuned a **3.2B-parameter LLM** using **QLoRA + Unsloth**
* Trained **24.3M parameters (0.75% of the model)** instead of updating the full model
* Fine-tuned on **2,697 ChatML-formatted customer support tickets**
* Built an LLM inference and evaluation pipeline across **540 benchmark tickets**
* Improved combined **Category + Subcategory + Priority accuracy from 34.70% to 98.56%**
* Implemented structured JSON generation and validation
* Compared **Prompt Engineering vs QLoRA Fine-Tuning**
* Built the project as an end-to-end AI/ML pipeline from dataset preparation to inference and evaluation

---

# 📌 Problem Statement

Customer support teams receive large volumes of unstructured tickets every day. Manually categorizing, prioritizing, and summarizing these tickets is time-consuming and difficult to scale.

TicketMindAI investigates whether **domain-specific parameter-efficient fine-tuning** can provide more reliable structured ticket understanding than carefully designed prompting alone.

For every customer support ticket, the model generates:

```json
{
  "category": "...",
  "subcategory": "...",
  "priority": "...",
  "sentiment": "...",
  "summary": "..."
}
```

The project evaluates the model primarily on:

* Category
* Subcategory
* Priority

while also generating sentiment and summary information.

---

# 🏗️ System Overview

```text
                    Customer Support Tickets
                              │
                              ▼
                  Dataset Preparation
                              │
                              ▼
                 Balanced Training Dataset
                              │
                              ▼
                   LLM-Based Annotation
                              │
                              ▼
                    ChatML Conversion
                              │
                              ▼
                  QLoRA / LoRA Fine-Tuning
                              │
                              ▼
                    Fine-Tuned 3.2B LLM
                              │
                              ▼
                    Structured JSON Output
                              │
                              ▼
                 JSON Validation & Evaluation
                              │
                              ▼
                   540-Ticket Benchmark
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
       Prompt Engineering              QLoRA Model
          Baseline
                │                           │
                └─────────────┬─────────────┘
                              ▼
                       Performance Comparison
```

---

# 🎯 Objectives

The project was designed to:

1. Build a balanced customer support ticket dataset.
2. Generate structured labels for supervised fine-tuning.
3. Convert training examples into ChatML format.
4. Fine-tune an open-source LLM using QLoRA.
5. Build a deterministic inference pipeline.
6. Validate structured JSON predictions.
7. Benchmark the prompt-engineered and fine-tuned approaches.
8. Quantify the impact of parameter-efficient fine-tuning on ticket classification.

---

# ⚙️ Tech Stack

## LLM & Fine-Tuning

* Llama 3.2 3B
* QLoRA
* LoRA
* PEFT
* Unsloth
* PyTorch

## AI / Machine Learning

* Hugging Face Transformers
* Hugging Face Datasets
* Google Gemini API
* Pandas
* NumPy

## Evaluation

* Custom benchmark pipeline
* JSON validation
* Category accuracy
* Subcategory accuracy
* Priority accuracy
* Combined structured-output accuracy
* Confusion/error analysis

## Development

* Python
* Git
* GitHub
* Jupyter / notebook-based experimentation

---

# 📊 Dataset Pipeline

The original dataset contains approximately **24,635 customer support tickets**.

The pipeline progressively transforms the raw dataset into a fine-tuning and evaluation dataset:

```text
Original Dataset
24,635 Tickets
       │
       ▼
Balanced Sampling
       │
       ▼
~3,240 Annotated Tickets
       │
       ▼
Quality Filtering
       │
       ▼
2,697 Training Tickets
       │
       ▼
ChatML Conversion
       │
       ▼
QLoRA Fine-Tuning
```

A separate **540-ticket benchmark** is used to compare the prompt-engineered baseline against the fine-tuned model.

---

# 🧠 Data Annotation

Google Gemini was used to automatically generate structured annotations for the training data.

Each ticket was transformed into a structured representation containing:

| Field       | Description                        |
| ----------- | ---------------------------------- |
| Category    | High-level ticket classification   |
| Subcategory | More specific issue classification |
| Priority    | Ticket urgency                     |
| Sentiment   | Customer sentiment                 |
| Summary     | Concise description of the issue   |

These annotations were subsequently converted into **ChatML-formatted supervised fine-tuning examples**.

---

# 🔧 QLoRA Fine-Tuning

Instead of updating all parameters of the 3.2B-parameter LLM, TicketMindAI uses **QLoRA** for parameter-efficient fine-tuning.

### Training configuration

```text
Base Model
Llama 3.2 3B

Training Examples
2,697 ChatML-formatted tickets

Total Model Parameters
~3.2B

Trainable Parameters
24.3M

Trainable Parameter Ratio
0.75%

Fine-Tuning Method
QLoRA

Training Framework
Unsloth + PEFT + PyTorch
```

QLoRA combines low-rank adaptation with quantized model weights, substantially reducing the computational and memory requirements of fine-tuning.

Only a small fraction of the model's parameters are trained while the original model weights remain largely frozen.

---

# 🔄 Inference Pipeline

The inference pipeline processes each ticket through the fine-tuned model and converts the generated response into structured information.

```text
Input Ticket
     │
     ▼
Prompt Construction
     │
     ▼
QLoRA Fine-Tuned LLM
     │
     ▼
Generated Response
     │
     ▼
JSON Extraction
     │
     ▼
Schema / Format Validation
     │
     ▼
Canonical Label Normalization
     │
     ▼
Evaluation
```

The evaluation pipeline is designed to handle malformed or inconsistent model outputs rather than blindly treating every generated response as valid.

---

# 📈 Benchmark Results

The final evaluation uses **540 benchmark tickets**.

### Prompt Engineering vs QLoRA

| Approach               |   Accuracy |
| ---------------------- | ---------: |
| Prompt Engineering     | **34.70%** |
| QLoRA Fine-Tuned Model | **98.56%** |

### Improvement

```text
Prompt Engineering
        │
        │ 34.70%
        ▼
   QLoRA Fine-Tuning
        │
        │ 98.56%
        ▼
+63.86 percentage points
```

The benchmark evaluates structured ticket understanding across:

* **Category**
* **Subcategory**
* **Priority**

The results demonstrate that, for this dataset and task formulation, domain-specific QLoRA fine-tuning substantially outperformed the prompt-engineered baseline.

---

# 🔬 Evaluation Methodology

The benchmark pipeline evaluates the models on the same set of **540 tickets**.

The evaluation process includes:

1. Running inference on each ticket.
2. Extracting the structured model response.
3. Validating the generated JSON.
4. Normalizing predicted labels.
5. Comparing predictions against the ground truth.
6. Computing classification accuracy.
7. Generating error/confusion analysis.

This makes the comparison between the baseline and fine-tuned model more controlled and reproducible.

---

# 🚀 Key Features

* ✅ Balanced customer support dataset
* ✅ Automated LLM annotation
* ✅ ChatML dataset generation
* ✅ Parameter-efficient QLoRA fine-tuning
* ✅ Unsloth-based training
* ✅ 3.2B-parameter LLM
* ✅ 24.3M trainable parameters
* ✅ Structured JSON generation
* ✅ JSON validation
* ✅ Label normalization
* ✅ Deterministic inference
* ✅ 540-ticket evaluation benchmark
* ✅ Prompt Engineering vs QLoRA comparison
* ✅ Accuracy and error analysis

---

# 📂 Project Structure

```text
TicketMindAI/
│
├── api/
│   └── FastAPI inference / API components
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── benchmark/
│
├── evaluation/
│   ├── benchmark/
│   ├── metrics/
│   └── error analysis
│
├── scripts/
│   ├── dataset preparation
│   └── annotation pipelines
│
├── training/
│   ├── ChatML conversion
│   └── QLoRA fine-tuning
│
├── models/
│   └── fine-tuned model / adapter artifacts
│
└── README.md
```

> The exact contents of individual directories may evolve as the project develops.

---

# 🧪 Experimental Comparison

TicketMindAI focuses on answering a practical AI Engineering question:

> **Can parameter-efficient fine-tuning provide substantially better structured customer-support understanding than prompt engineering alone?**

The experiment compares:

### 1. Prompt Engineering

A base LLM is instructed through carefully designed prompts to produce structured ticket classifications.

### 2. QLoRA Fine-Tuning

The same task is learned from domain-specific ChatML training examples using parameter-efficient fine-tuning.

Both approaches are evaluated against the same benchmark.

---

# 📊 Final Results

```text
                           Accuracy
                              │
Prompt Engineering            │ ███████                34.70%
                              │
QLoRA Fine-Tuning             │ ████████████████████   98.56%
                              │
                              └────────────────────────
```

**Result: 98.56% accuracy with QLoRA compared with 34.70% using prompt engineering.**

This represents a **63.86 percentage-point improvement** on the benchmark.

---

# 💡 Engineering Takeaways

### Parameter-efficient fine-tuning

QLoRA allowed the project to adapt a 3.2B-parameter model while training only **24.3M parameters**, or approximately **0.75% of the model**.

### Structured generation

For real-world ticket processing, classification accuracy alone is insufficient. The model must also produce machine-readable output that can be validated and consumed by downstream systems.

### Benchmark-driven development

Rather than evaluating the fine-tuned model subjectively, the project uses a fixed **540-ticket benchmark** to quantify the difference between approaches.

### Domain adaptation

The results suggest that a model trained specifically on the target ticket taxonomy can substantially outperform generic prompt-based inference for this structured classification task.

---

# 🚧 Current Status

| Module                      | Status        |
| --------------------------- | ------------- |
| Dataset Cleaning            | ✅ Completed   |
| Balanced Dataset Generation | ✅ Completed   |
| Gemini Annotation Pipeline  | ✅ Completed   |
| Training Dataset Generation | ✅ Completed   |
| ChatML Conversion           | ✅ Completed   |
| QLoRA Fine-Tuning           | ✅ Completed   |
| Model Inference             | ✅ Completed   |
| JSON Validation             | ✅ Completed   |
| Benchmark Generation        | ✅ Completed   |
| 540-Ticket Evaluation       | ✅ Completed   |
| Error / Confusion Analysis  | ✅ Completed   |
| FastAPI Integration         | ✅ Implemented |

---

# 🎯 Conclusion

TicketMindAI demonstrates an end-to-end workflow for adapting an open-source LLM to a specialized customer-support classification task.

The final experiment shows:

```text
2,697 Training Tickets
        +
QLoRA Fine-Tuning
        +
24.3M Trainable Parameters
        +
540-Ticket Benchmark
        ↓
98.56% Accuracy
```

compared with:

```text
Prompt Engineering
        ↓
34.70% Accuracy
```

The project therefore provides empirical evidence that **parameter-efficient fine-tuning can significantly improve structured domain-specific ticket understanding over prompt engineering alone**, at least for the dataset, taxonomy, and benchmark used in this experiment.

---

## 👨‍💻 Author

**Shivansh Manohar**

B.Tech Computer Science
Vellore Institute of Technology, Vellore

* GitHub: [@ShivanshManohar](https://github.com/ShivanshManohar)
* LinkedIn: [shivanshmanohar](https://linkedin.com/in/shivanshmanohar)
