import json


def export_jsonl(df, output_path):

    with open(output_path, "w", encoding="utf-8") as f:

        for _, row in df.iterrows():

            sample = {

                "instruction":
                    "Convert the customer support ticket into structured metadata.",

                "input":
                    row["input"],

                "output":
                    json.dumps(
                        {
                            "category": row["category"],
                            "subcategory": row["subcategory"],
                            "priority": row["priority"]
                        }
                    )

            }

            f.write(json.dumps(sample) + "\n")

    print(f"\nDataset exported to {output_path}")