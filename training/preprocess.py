from datasets import load_dataset
from clean import clean_dataset
from enrich import enrich_dataset
from export import export_jsonl

print("=" * 60)
print("TicketMindAI Preprocessing Pipeline")
print("=" * 60)

dataset = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    split="train"
)

df = dataset.to_pandas()

print(f"Raw Samples : {len(df)}")

df = clean_dataset(df) # First cleaning
df = enrich_dataset(df) # Then enriching (setting priority)

print(f"Clean Samples : {len(df)}")

print("\nPreview\n")

print(
    df[
        [
            "input",
            "category",
            "subcategory",
            "priority"
        ]
    ].head(10)
)

export_jsonl(
    df,
    "data/processed/train.jsonl"
)

df.to_csv(
    "data/processed/ticketmind_dataset.csv",
    index=False
)

print("CSV exported.")