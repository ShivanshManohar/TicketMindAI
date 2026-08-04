import pandas as pd


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw Bitext dataset.
    """

    # Rename columns
    df = df.rename(
        columns={
            "instruction": "input",
            "intent": "subcategory"
        }
    )

    # Normalize text
    df["category"] = (
        df["category"]
        .str.title()
        .str.replace("_", " ", regex=False)
    )

    df["subcategory"] = (
        df["subcategory"]
        .str.title()
        .str.replace("_", " ", regex=False)
    )

    # Remove duplicates
    df = df.drop_duplicates(subset=["input"])

    # Remove empty instructions
    df = df[df["input"].str.strip() != ""]

    return df