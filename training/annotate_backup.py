import json
import os
import pandas as pd

from training.annotate import annotate

INPUT_FILE = "data/processed/ticketmind_dataset.csv"
OUTPUT_FILE = "data/processed/annotations.json"

SAVE_EVERY = 25

df = pd.read_csv(INPUT_FILE)

if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)
else:
    results = []

start = len(results)

print(f"Resuming from sample {start}")

for i in range(start, len(df)):

    row = df.iloc[i]

    print(f"[{i+1}/{len(df)}]")

    try:

        result = annotate(
            row["input"],
            row["category"],
            row["subcategory"]
        )

        result["category"] = row["category"]
        result["subcategory"] = row["subcategory"]
        result["input"] = row["input"]

        results.append(result)

    except Exception as e:

        print(e)

        continue

    if (i + 1) % SAVE_EVERY == 0:

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print("Checkpoint Saved")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print("Finished")