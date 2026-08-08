import json
from collections import Counter, defaultdict


PREDICTIONS_PATH = (
    "data/processed/prompt_baseline_predictions.json"
)


def main():

    with open(
        PREDICTIONS_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    print("\n" + "=" * 70)
    print("PRIORITY AUDIT")
    print("=" * 70)

    true_counts = Counter(
        row["true_priority"].upper()
        for row in data
    )

    pred_counts = Counter(
        row["pred_priority"].upper()
        for row in data
    )

    print("\nGROUND TRUTH DISTRIBUTION")
    print("-" * 70)

    for label in ["HIGH", "MEDIUM", "LOW"]:
        count = true_counts[label]
        print(
            f"{label:8} {count:4} "
            f"({count / len(data) * 100:.2f}%)"
        )

    print("\nPREDICTION DISTRIBUTION")
    print("-" * 70)

    for label in ["HIGH", "MEDIUM", "LOW"]:
        count = pred_counts[label]
        print(
            f"{label:8} {count:4} "
            f"({count / len(data) * 100:.2f}%)"
        )

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    matrix = defaultdict(Counter)

    for row in data:

        true = row["true_priority"].upper()
        pred = row["pred_priority"].upper()

        matrix[true][pred] += 1

    print("\nCONFUSION MATRIX")
    print("-" * 70)

    print(
        f"{'TRUE':<12}"
        f"{'HIGH':>10}"
        f"{'MEDIUM':>10}"
        f"{'LOW':>10}"
    )

    for true in ["HIGH", "MEDIUM", "LOW"]:

        print(
            f"{true:<12}"
            f"{matrix[true]['HIGH']:>10}"
            f"{matrix[true]['MEDIUM']:>10}"
            f"{matrix[true]['LOW']:>10}"
        )

    # ---------------------------------------------------------
    # Misclassified examples
    # ---------------------------------------------------------

    print("\nMISCLASSIFIED PRIORITY EXAMPLES")
    print("-" * 70)

    shown = 0

    for row in data:

        true = row["true_priority"].upper()
        pred = row["pred_priority"].upper()

        if true != pred:

            print(
                f"\nTicket: {row['input']}"
            )

            print(
                f"True: {true} | Predicted: {pred}"
            )

            print(
                f"Category: {row['true_category']}"
            )

            print(
                f"Subcategory: {row['true_subcategory']}"
            )

            shown += 1

            if shown >= 25:
                break

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()