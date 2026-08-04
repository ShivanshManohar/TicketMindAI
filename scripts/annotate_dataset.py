import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import json
import pandas as pd

from training.annotate import annotate

df = pd.read_csv("data/processed/ticketmind_dataset.csv")

results = []

for i, row in df.head(100).iterrows():

    print(f"Annotating {i+1}/100")

    result = annotate(
        row["input"],
        row["category"],
        row["subcategory"]
    )

    result["input"] = row["input"]

    results.append(result)

with open(
    "data/processed/annotations.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(results, f, indent=4)

print("Finished!")