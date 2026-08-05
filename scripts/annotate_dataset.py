import os
import sys
import json
import logging
import argparse

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import pandas as pd

from training.annotate import annotate

# ---------------------------------------------------------------------------
# Logging: failures go to a file so a crash or a bad batch is diagnosable
# without re-reading terminal scrollback.
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("data/processed/annotate_run.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("annotate_dataset")

DATA_PATH = "data/processed/ticketmind_dataset.csv"
OUTPUT_PATH = "data/processed/annotations.json"
FAILURES_PATH = "data/processed/annotation_failures.json"
CHECKPOINT_EVERY = 10  # save progress every N rows, not just at the end


def load_existing_results(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            logger.warning("Existing %s is unreadable, starting fresh.", path)
            return []


def save_json(path: str, data: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Annotate ticket dataset via LLM.")
    parser.add_argument("--n", type=int, default=100, help="Number of rows to annotate.")
    parser.add_argument("--start", type=int, default=0, help="Row index to start from.")
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip inputs already present in an existing annotations.json.",
    )
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH)
    subset = df.iloc[args.start: args.start + args.n]

    results = load_existing_results(OUTPUT_PATH) if args.resume else []
    already_done = {r["input"] for r in results if r.get("input")}

    failures = load_existing_results(FAILURES_PATH) if args.resume else []

    total = len(subset)
    processed_since_checkpoint = 0

    for i, (idx, row) in enumerate(subset.iterrows(), start=1):
        ticket_input = row["input"]

        if args.resume and ticket_input in already_done:
            logger.info("Skipping already-annotated row %d/%d (resume)", i, total)
            continue

        logger.info("Annotating %d/%d (row index %d)", i, total, idx)

        result = annotate(ticket_input, row["category"], row["subcategory"])

        if result is None:
            logger.error("Skipping row %d after annotate() failure.", idx)
            failures.append({
                "row_index": int(idx),
                "input": ticket_input,
                "category": row["category"],
                "subcategory": row["subcategory"],
            })
            continue

        result["input"] = ticket_input
        results.append(result)
        processed_since_checkpoint += 1

        if processed_since_checkpoint >= CHECKPOINT_EVERY:
            save_json(OUTPUT_PATH, results)
            save_json(FAILURES_PATH, failures)
            processed_since_checkpoint = 0
            logger.info("Checkpoint saved: %d annotations so far.", len(results))

    # final save
    save_json(OUTPUT_PATH, results)
    save_json(FAILURES_PATH, failures)

    logger.info(
        "Finished. %d annotated, %d failed. Failures logged to %s.",
        len(results), len(failures), FAILURES_PATH,
    )


if __name__ == "__main__":
    main()