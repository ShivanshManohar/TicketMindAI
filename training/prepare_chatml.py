import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("chatml_convert")

ANNOTATIONS_PATH = "data/processed/annotations.json"
OUTPUT_PATH = "data/processed/train_chatml.json"

REQUIRED_KEYS = (
    "input",
    "category",
    "subcategory",
    "priority",
    "sentiment",
    "summary",
)

def load_annotations(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_chatml(record: dict) -> dict | None:
    missing = [k for k in REQUIRED_KEYS if not record.get(k)]
    if missing:
        logger.warning("Skipping record, missing keys %s: %r", missing, record.get("input"))
        return None

    assistant_response = {
        "category": record["category"],
        "subcategory": record["subcategory"],
        "priority": record["priority"],
        "sentiment": record["sentiment"],
        "summary": record["summary"],
    }

    return {
        "messages": [
            {"role": "user", "content": record["input"]},
            {
                "role": "assistant",
                # compact, single-line JSON: this is the exact string the model
                # is trained to reproduce, so no pretty-print indentation here
                "content": json.dumps(
                    assistant_response,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            },
        ]
    }


def main():
    records = load_annotations(ANNOTATIONS_PATH)
    logger.info("Loaded %d annotated records from %s", len(records), ANNOTATIONS_PATH)

    chatml_data = []
    skipped = 0

    for record in records:
        sample = to_chatml(record)
        if sample is None:
            skipped += 1
            continue
        chatml_data.append(sample)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chatml_data, f, indent=4, ensure_ascii=False)

    logger.info(
        "Created %d ChatML training samples (%d skipped for missing fields). Written to %s",
        len(chatml_data), skipped, OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()