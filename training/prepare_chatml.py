import json
import pandas as pd

df = pd.read_csv("data/processed/ticketmind_dataset.csv")

chatml_data = []

for _, row in df.iterrows():

    assistant_response = {
        "category": row["category"],
        "subcategory": row["subcategory"],
        "priority": row["priority"]
    }

    chatml_data.append(
        {
            "messages": [
                {
                    "role": "user",
                    "content": row["input"]
                },
                {
                    "role": "assistant",
                    "content": json.dumps(
                                assistant_response,
                                indent=2,
                                ensure_ascii=False
                            )
                }
            ]
        }
    )

with open(
    "data/processed/train_chatml.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(chatml_data, f, indent=4)

print(f"Created {len(chatml_data)} training samples.")