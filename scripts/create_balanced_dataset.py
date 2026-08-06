import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger("balanced_dataset")

INPUT_PATH = "data/processed/ticketmind_dataset.csv"
OUTPUT_PATH = "data/processed/ticketmind_balanced.csv"

# Number of samples to keep for each (category, subcategory)

TARGET_DATASET_SIZE = 3000

RANDOM_SEED = 42


def main():

    logger.info("Loading dataset...")

    df = pd.read_csv(INPUT_PATH)

    logger.info("Original samples: %d", len(df))

    grouped = df.groupby(
        ["category", "subcategory"],
        sort=True
    )
    num_intents = grouped.ngroups
    samples_per_intent = TARGET_DATASET_SIZE // num_intents

    balanced = []

    logger.info("Creating balanced dataset...")

    for (category, subcategory), group in grouped:

        available = len(group)

        take = min(
            samples_per_intent,
            available
        )

        sampled = group.sample(
            n=take,
            random_state=RANDOM_SEED
        )

        balanced.append(sampled)

        logger.info(
            "%-20s | %-30s | %4d / %4d",
            category,
            subcategory,
            take,
            available
        )

    balanced_df = (
        pd.concat(balanced)
        .sample(frac=1, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )

    balanced_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    logger.info("")

    logger.info(
        "Balanced dataset created successfully."
    )

    logger.info(
        "Final samples: %d",
        len(balanced_df)
    )

    logger.info(
        "Saved to: %s",
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()

