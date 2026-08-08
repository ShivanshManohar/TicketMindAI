import json
import logging
import os
from collections import Counter


INPUT_PATH = "data/processed/annotations.json"

OUTPUT_PATH = (
    "data/processed/"
    "annotations_repaired.json"
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("repair_annotations")


# ---------------------------------------------------------
# Dataset-discovered priority mapping
# ---------------------------------------------------------

HIGH_CATEGORIES = {
    "Account",
    "Payment",
    "Refund",
}

MEDIUM_CATEGORIES = {
    "Delivery",
    "Shipping",
}

LOW_CATEGORIES = {
    "Cancel",
    "Contact",
    "Feedback",
    "Invoice",
    "Order",
    "Subscription",
}


def get_priority(category):
    if category in HIGH_CATEGORIES:
        return "HIGH"

    if category in MEDIUM_CATEGORIES:
        return "MEDIUM"

    if category in LOW_CATEGORIES:
        return "LOW"

    raise ValueError(
        f"Unknown category: {category}"
    )


def main():

    logger.info(
        "Loading annotations..."
    )

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        annotations = json.load(f)

    logger.info(
        "Loaded %d annotations.",
        len(annotations),
    )

    repaired = []

    old_priority_counts = Counter()
    new_priority_counts = Counter()
    sentiment_counts = Counter()

    for index, item in enumerate(
        annotations
    ):

        required = {
            "input",
            "category",
            "subcategory",
            "priority",
            "sentiment",
            "summary",
        }

        missing = required - set(item.keys())

        if missing:

            raise ValueError(
                f"Row {index} missing fields: "
                f"{sorted(missing)}"
            )

        category = item["category"]

        priority = get_priority(
            category
        )

        old_priority_counts[
            item["priority"]
        ] += 1

        new_priority_counts[
            priority
        ] += 1

        sentiment_counts[
            item["sentiment"]
        ] += 1

        repaired_item = {
            "input": item["input"],
            "category": category,
            "subcategory": item["subcategory"],
            "priority": priority,
            "sentiment": item["sentiment"],
            "summary": item["summary"],
        }

        repaired.append(
            repaired_item
        )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            repaired,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("ANNOTATION REPAIR COMPLETE")
    print("=" * 70)

    print(
        f"\nSamples: {len(repaired)}"
    )

    print("\nOLD PRIORITY DISTRIBUTION")
    print("-" * 70)

    for label in [
        "HIGH",
        "MEDIUM",
        "LOW",
    ]:

        print(
            f"{label:8}: "
            f"{old_priority_counts[label]}"
        )

    print("\nREPAIRED PRIORITY DISTRIBUTION")
    print("-" * 70)

    for label in [
        "HIGH",
        "MEDIUM",
        "LOW",
    ]:

        print(
            f"{label:8}: "
            f"{new_priority_counts[label]}"
        )

    print("\nSENTIMENT DISTRIBUTION")
    print("-" * 70)

    for label in [
        "Positive",
        "Neutral",
        "Negative",
    ]:

        print(
            f"{label:8}: "
            f"{sentiment_counts[label]}"
        )

    print("\nOUTPUT:")
    print(OUTPUT_PATH)

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()