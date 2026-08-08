import json
import logging

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("evaluate")


def normalize_priority(value):
    return str(value).strip().upper()


def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    return {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


def calculate_exact_match(y_true, y_pred):
    if not y_true:
        return 0.0

    return sum(
        true == pred
        for true, pred in zip(y_true, y_pred)
    ) / len(y_true)


def evaluate_predictions(predictions):
    """
    Evaluate predictions against ground-truth labels.

    Expected fields:

    true_category
    pred_category

    true_subcategory
    pred_subcategory

    true_priority
    pred_priority
    """

    if not predictions:
        raise ValueError(
            "No predictions were provided."
        )

    # ---------------------------------------------------------
    # Category
    # ---------------------------------------------------------

    category_true = [
        row["true_category"]
        for row in predictions
    ]

    category_pred = [
        row["pred_category"]
        for row in predictions
    ]

    category_metrics = calculate_metrics(
        category_true,
        category_pred,
    )

    category_metrics["exact_match"] = (
        calculate_exact_match(
            category_true,
            category_pred,
        )
    )

    # ---------------------------------------------------------
    # Subcategory
    # ---------------------------------------------------------

    subcategory_true = [
        row["true_subcategory"]
        for row in predictions
    ]

    subcategory_pred = [
        row["pred_subcategory"]
        for row in predictions
    ]

    subcategory_metrics = calculate_metrics(
        subcategory_true,
        subcategory_pred,
    )

    subcategory_metrics["exact_match"] = (
        calculate_exact_match(
            subcategory_true,
            subcategory_pred,
        )
    )

    # ---------------------------------------------------------
    # Priority
    # ---------------------------------------------------------

    priority_true = [
        normalize_priority(
            row["true_priority"]
        )
        for row in predictions
    ]

    priority_pred = [
        normalize_priority(
            row["pred_priority"]
        )
        for row in predictions
    ]

    priority_metrics = calculate_metrics(
        priority_true,
        priority_pred,
    )

    priority_metrics["exact_match"] = (
        calculate_exact_match(
            priority_true,
            priority_pred,
        )
    )

    return {
        "samples": len(predictions),
        "category": category_metrics,
        "subcategory": subcategory_metrics,
        "priority": priority_metrics,
    }


def print_results(results, model_name="Model"):
    print()
    print("=" * 72)
    print(f"TICKETMINDAI — {model_name}")
    print("=" * 72)

    print(
        f"\nEvaluation samples: {results['samples']}"
    )

    for name in [
        "category",
        "subcategory",
        "priority",
    ]:

        metrics = results[name]

        print(
            f"\n{name.upper()}"
        )

        print("-" * 72)

        print(
            f"Accuracy        : "
            f"{metrics['accuracy']:.4f} "
            f"({metrics['accuracy'] * 100:.2f}%)"
        )

        print(
            f"Macro Precision : "
            f"{metrics['macro_precision']:.4f}"
        )

        print(
            f"Macro Recall    : "
            f"{metrics['macro_recall']:.4f}"
        )

        print(
            f"Macro F1        : "
            f"{metrics['macro_f1']:.4f}"
        )

        print(
            f"Exact Match     : "
            f"{metrics['exact_match']:.4f} "
            f"({metrics['exact_match'] * 100:.2f}%)"
        )

    print()
    print("=" * 72)


def main():

    predictions_path = (
        "data/processed/"
        "prompt_baseline_predictions.json"
    )

    logger.info(
        "Loading predictions from %s",
        predictions_path,
    )

    with open(
        predictions_path,
        "r",
        encoding="utf-8",
    ) as f:

        predictions = json.load(f)

    results = evaluate_predictions(
        predictions
    )

    print_results(
        results,
        model_name="PROMPT ENGINEERING BASELINE",
    )


if __name__ == "__main__":
    main()