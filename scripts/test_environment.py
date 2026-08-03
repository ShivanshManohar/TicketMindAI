import pandas
from datasets import load_dataset

print("✅ pandas imported")
print("✅ datasets imported")

dataset = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    split="train[:1]"
)

print("✅ Dataset downloaded")
print(dataset[0])