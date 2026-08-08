import json
import logging
import os
import sys
import time

from evaluate import (
    evaluate_predictions,
    print_results,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("benchmark")


PREDICTIONS_PATH = (
    "data/processed/"
    "prompt_baseline_predictions.json"
)

RESULTS_PATH = (
    "data/processed/"
    "prompt_baseline_metrics.json"
)


def load_predictions():

    if not os.path.exists(
        PREDICTIONS_PATH
    ):
        raise FileNotFoundError(
            f"Prediction file not found:\n"
            f"{PREDICTIONS_PATH}\n\n"
            f"Run prompt_baseline.py first."
        )

    with open(
        PREDICTIONS_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        predictions = json.load(f)

    if not isinstance(
        predictions,
        list,
    ):

        raise ValueError(
            "Prediction file must contain a JSON list."
        )

    if len(predictions) == 0:

        raise ValueError(
            "Prediction file contains zero predictions."
        )

    return predictions


def validate_predictions(predictions):

    required_fields = {
        "input",
        "true_category",
        "true_subcategory",
        "true_priority",
        "pred_category",
        "pred_subcategory",
        "pred_priority",
    }

    for index, row in enumerate(
        predictions
    ):

        missing = (
            required_fields
            - set(row.keys())
        )

        if missing:

            raise ValueError(
                f"Prediction {index} is missing: "
                f"{sorted(missing)}"
            )


def save_results(results):

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )


def main():

    logger.info(
        "Starting benchmark evaluation..."
    )

    predictions = load_predictions()

    logger.info(
        "Loaded %d predictions.",
        len(predictions),
    )

    validate_predictions(
        predictions
    )

    start_time = time.perf_counter()

    results = evaluate_predictions(
        predictions
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    results["evaluation_time_seconds"] = (
        elapsed
    )

    results["model"] = (
        "Gemini 3.5 Flash-Lite "
        "Prompt Engineering"
    )

    save_results(
        results
    )

    print_results(
        results,
        model_name=(
            "PROMPT ENGINEERING BASELINE"
        ),
    )

    logger.info(
        "Metrics saved to: %s",
        RESULTS_PATH,
    )

    logger.info(
        "Benchmark evaluation completed."
    )


if __name__ == "__main__":
    main()