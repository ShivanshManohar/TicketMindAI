import pandas as pd


DATASET_PATH = "data/processed/ticketmind_dataset.csv"


def main():

    df = pd.read_csv(DATASET_PATH)

    print()
    print("=" * 80)
    print("ORIGINAL DATASET — PRIORITY MAPPING")
    print("=" * 80)

    mapping = (
        df.groupby(
            ["category", "subcategory", "priority"]
        )
        .size()
        .reset_index(name="count")
    )

    for (category, subcategory), group in mapping.groupby(
        ["category", "subcategory"]
    ):

        total = group["count"].sum()

        print()
        print(
            f"{category} | {subcategory}"
        )

        for _, row in group.iterrows():

            percentage = (
                row["count"] / total * 100
            )

            print(
                f"    {row['priority']:8} "
                f"{row['count']:4} "
                f"({percentage:6.2f}%)"
            )

    print()
    print("=" * 80)
    print("DOMINANT PRIORITY PER INTENT")
    print("=" * 80)

    dominant = (
        df.groupby(
            ["category", "subcategory"]
        )["priority"]
        .agg(
            lambda x: x.value_counts().index[0]
        )
        .reset_index()
    )

    for _, row in dominant.iterrows():

        print(
            f"{row['category']:15} | "
            f"{row['subcategory']:30} | "
            f"{row['priority']}"
        )

    print()
    print("=" * 80)

    # Check whether each intent has a single priority.
    consistency = (
        df.groupby(
            ["category", "subcategory"]
        )["priority"]
        .nunique()
        .reset_index(name="unique_priorities")
    )

    inconsistent = consistency[
        consistency["unique_priorities"] > 1
    ]

    print(
        f"\nTotal intents: {len(consistency)}"
    )

    print(
        f"Consistent intents: "
        f"{len(consistency) - len(inconsistent)}"
    )

    print(
        f"Inconsistent intents: "
        f"{len(inconsistent)}"
    )

    if len(inconsistent) > 0:

        print()
        print("INTENTS WITH MULTIPLE PRIORITIES")
        print("-" * 80)

        for _, row in inconsistent.iterrows():

            print(
                f"{row['category']:15} | "
                f"{row['subcategory']:30}"
            )


if __name__ == "__main__":
    main()