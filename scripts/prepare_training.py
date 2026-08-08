import json
import logging
import random
import os


INPUT_PATH = "data/processed/annotations_repaired.json"

OUTPUT_PATH = "data/processed/train_chatml.jsonl"

SEED = 42


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("prepare_training")


SYSTEM_PROMPT = """You are TicketMindAI, a customer support ticket classification system.

Analyze the customer's ticket and return structured metadata.

You must determine:
- category
- subcategory
- priority
- sentiment
- summary

Return ONLY valid JSON.

Required format:
{
  "category": "...",
  "subcategory": "...",
  "priority": "HIGH|MEDIUM|LOW",
  "sentiment": "Positive|Neutral|Negative",
  "summary": "..."
}
"""


def build_example(row):

    user_content = f"""Customer Ticket:
{row["input"]}"""

    assistant_content = json.dumps(
        {
            "category": row["category"],
            "subcategory": row["subcategory"],
            "priority": row["priority"],
            "sentiment": row["sentiment"],
            "summary": row["summary"],
        },
        ensure_ascii=False,
    )

    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_content,
            },
            {
                "role": "assistant",
                "content": assistant_content,
            },
        ]
    }


def main():

    logger.info(
        "Loading repaired annotations..."
    )

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    logger.info(
        "Loaded %d samples.",
        len(data),
    )

    random.seed(SEED)

    random.shuffle(data)

    examples = [
        build_example(row)
        for row in data
    ]

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        for example in examples:

            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                )
                + "\n"
            )

    logger.info(
        "Saved %d training examples.",
        len(examples),
    )

    logger.info(
        "Output: %s",
        OUTPUT_PATH,
    )

    # ---------------------------------------------------------
    # Preview
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING DATASET CREATED")
    print("=" * 70)

    print(
        f"\nSamples: {len(examples)}"
    )

    print(
        f"Format : JSONL / ChatML"
    )

    print(
        f"Output : {OUTPUT_PATH}"
    )

    print("\nFIRST EXAMPLE")
    print("-" * 70)

    print(
        json.dumps(
            examples[0],
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()