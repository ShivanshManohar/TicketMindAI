import json
import logging

import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("create_benchmark")


ORIGINAL_DATASET = "data/processed/ticketmind_dataset.csv"
ANNOTATIONS_PATH = "data/processed/annotations.json"
OUTPUT_PATH = "data/processed/benchmark.csv"

SAMPLES_PER_INTENT = 20
RANDOM_SEED = 42


def main():

    logger.info("Loading original dataset...")

    df = pd.read_csv(ORIGINAL_DATASET)

    logger.info(
        "Original dataset: %d rows",
        len(df),
    )

    logger.info("Loading training annotations...")

    with open(
        ANNOTATIONS_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        annotations = json.load(f)

    training_inputs = {
        item["input"]
        for item in annotations
        if item.get("input")
    }

    logger.info(
        "Training inputs found: %d",
        len(training_inputs),
    )

    # Remove every ticket that was used for training.
    benchmark_pool = df[
        ~df["input"].isin(training_inputs)
    ].copy()

    logger.info(
        "Remaining benchmark pool: %d rows",
        len(benchmark_pool),
    )

    # Stratify by intent.
    grouped = benchmark_pool.groupby(
        ["category", "subcategory"],
        sort=True,
    )

    benchmark_parts = []

    for (category, subcategory), group in grouped:

        available = len(group)

        if available < SAMPLES_PER_INTENT:

            logger.warning(
                "%s / %s has only %d available samples",
                category,
                subcategory,
                available,
            )

            take = available

        else:

            take = SAMPLES_PER_INTENT

        sampled = group.sample(
            n=take,
            random_state=RANDOM_SEED,
        )

        benchmark_parts.append(sampled)

        logger.info(
            "%-20s | %-30s | %d",
            category,
            subcategory,
            take,
        )

    benchmark = pd.concat(
        benchmark_parts,
        ignore_index=True,
    )

    benchmark = benchmark.sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    benchmark.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    logger.info("")
    logger.info(
        "Benchmark created: %d samples",
        len(benchmark),
    )

    logger.info(
        "Saved to: %s",
        OUTPUT_PATH,
    )

    logger.info(
        "Unique intents: %d",
        benchmark.groupby(
            ["category", "subcategory"]
        ).ngroups,
    )


if __name__ == "__main__":
    main()