import json
import logging
from collections import Counter

import pandas as pd


ORIGINAL_PATH = "data/processed/ticketmind_dataset.csv"
ANNOTATIONS_PATH = "data/processed/annotations_repaired.json"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("annotation_audit")


def main():

    logger.info("Loading original dataset...")
    original = pd.read_csv(ORIGINAL_PATH)

    logger.info("Loading repaired annotations...")
    with open(
        ANNOTATIONS_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        annotations = json.load(f)

    print("\n" + "=" * 70)
    print("ANNOTATION QUALITY AUDIT")
    print("=" * 70)

    print(f"\nOriginal dataset : {len(original)}")
    print(f"Annotations      : {len(annotations)}")

    original_map = {
        row["input"]: row
        for _, row in original.iterrows()
    }

    # ---------------------------------------------------------
    # 1. Required fields
    # ---------------------------------------------------------

    required = {
        "input",
        "category",
        "subcategory",
        "priority",
        "sentiment",
        "summary",
    }

    missing_fields = 0

    for i, row in enumerate(annotations):

        missing = required - set(row.keys())

        if missing:
            missing_fields += 1
            print(
                f"Missing fields row {i}: {missing}"
            )

    # ---------------------------------------------------------
    # 2. Original dataset consistency
    # ---------------------------------------------------------

    missing_original = 0
    category_mismatch = 0
    subcategory_mismatch = 0
    priority_mismatch = 0

    for row in annotations:

        original_row = original_map.get(
            row["input"]
        )

        if original_row is None:
            missing_original += 1
            continue

        if (
            row["category"]
            != original_row["category"]
        ):
            category_mismatch += 1

        if (
            row["subcategory"]
            != original_row["subcategory"]
        ):
            subcategory_mismatch += 1

        if (
            row["priority"].upper()
            != str(
                original_row["priority"]
            ).upper()
        ):
            priority_mismatch += 1

    # ---------------------------------------------------------
    # 3. Duplicates
    # ---------------------------------------------------------

    inputs = [
        row["input"]
        for row in annotations
    ]

    duplicate_count = (
        len(inputs) - len(set(inputs))
    )

    # ---------------------------------------------------------
    # 4. Summary quality
    # ---------------------------------------------------------

    empty_summaries = 0
    long_summaries = 0

    summary_lengths = []

    for row in annotations:

        summary = str(
            row.get("summary", "")
        ).strip()

        if not summary:
            empty_summaries += 1

        words = len(summary.split())

        summary_lengths.append(words)

        if words > 20:
            long_summaries += 1

    # ---------------------------------------------------------
    # 5. Sentiment
    # ---------------------------------------------------------

    sentiments = Counter(
        row["sentiment"]
        for row in annotations
    )

    # ---------------------------------------------------------
    # 6. Priority
    # ---------------------------------------------------------

    priorities = Counter(
        row["priority"]
        for row in annotations
    )

    # ---------------------------------------------------------
    # 7. Category distribution
    # ---------------------------------------------------------

    categories = Counter(
        row["category"]
        for row in annotations
    )

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------

    print("\nDATA INTEGRITY")
    print("-" * 70)

    print(
        f"Missing required fields : {missing_fields}"
    )

    print(
        f"Missing from original   : {missing_original}"
    )

    print(
        f"Category mismatches     : {category_mismatch}"
    )

    print(
        f"Subcategory mismatches  : {subcategory_mismatch}"
    )

    print(
        f"Priority mismatches     : {priority_mismatch}"
    )

    print(
        f"Duplicate inputs        : {duplicate_count}"
    )

    print("\nSUMMARY QUALITY")
    print("-" * 70)

    print(
        f"Empty summaries         : {empty_summaries}"
    )

    print(
        f"Summaries >20 words    : {long_summaries}"
    )

    print(
        f"Average summary length : "
        f"{sum(summary_lengths) / len(summary_lengths):.2f} words"
    )

    print("\nSENTIMENT")
    print("-" * 70)

    for label in [
        "Positive",
        "Neutral",
        "Negative",
    ]:
        count = sentiments[label]

        print(
            f"{label:10}: "
            f"{count:4} "
            f"({count / len(annotations) * 100:.2f}%)"
        )

    print("\nPRIORITY")
    print("-" * 70)

    for label in [
        "HIGH",
        "MEDIUM",
        "LOW",
    ]:
        count = priorities[label]

        print(
            f"{label:10}: "
            f"{count:4} "
            f"({count / len(annotations) * 100:.2f}%)"
        )

    print("\nCATEGORY COVERAGE")
    print("-" * 70)

    for category, count in sorted(
        categories.items()
    ):
        print(
            f"{category:15}: {count}"
        )

    # ---------------------------------------------------------
    # Final verdict
    # ---------------------------------------------------------

    critical_failures = (
        missing_fields
        + missing_original
        + category_mismatch
        + subcategory_mismatch
        + priority_mismatch
        + duplicate_count
        + empty_summaries
        + long_summaries
    )

    print("\n" + "=" * 70)

    if critical_failures == 0:
        print("AUDIT STATUS: PASS")
        print(
            "Training dataset is structurally clean."
        )
    else:
        print("AUDIT STATUS: REVIEW REQUIRED")
        print(
            f"Potential issues detected: "
            f"{critical_failures}"
        )

    if sentiments["Positive"] == 0:
        print(
            "\nWARNING: Positive sentiment count is 0."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()