import os
import sys
import json
import time
import logging

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import pandas as pd

from training.annotate_batch import annotate_batch, BATCH_SIZE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("data/processed/annotate_run.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("annotate_dataset")

DATASET_PATH = "data/processed/ticketmind_balanced.csv"
OUTPUT_PATH = "data/processed/annotations.json"
FAILURE_PATH = "data/processed/annotation_failures.json"


def load_json(path):

    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )


def main():

    df = pd.read_csv(DATASET_PATH)

    annotations = load_json(OUTPUT_PATH)

    failures = load_json(FAILURE_PATH)

    start = len(annotations)

    logger.info("Starting from row %d", start)

    while start < len(df):

        end = min(start + BATCH_SIZE, len(df))

        logger.info(
            "Processing rows %d -> %d",
            start,
            end - 1
        )

        rows = df.iloc[start:end].to_dict("records")

        success = False

        for retry in range(10):

            results = annotate_batch(rows)

            if results is not None:

                annotations.extend(results)

                save_json(
                    OUTPUT_PATH,
                    annotations
                )

                logger.info(
                    "Saved %d annotations",
                    len(annotations)
                )

                success = True

                break

            logger.warning(
                "Retry %d/10...",
                retry + 1
            )

            time.sleep(45)

        if not success:

            failures.append({

                "start": start,

                "end": end - 1

            })

            save_json(
                FAILURE_PATH,
                failures
            )

        start = end

        # Stay below Gemini free-tier RPM
        time.sleep(5)

    logger.info("Annotation completed.")


if __name__ == "__main__":
    main()