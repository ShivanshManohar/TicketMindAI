import pandas as pd


HIGH_PRIORITY = {
    "Payment",
    "Refund",
    "Account"
}

MEDIUM_PRIORITY = {
    "Delivery",
    "Shipping"
}


def assign_priority(category: str) -> str:
    """
    Assign priority based on ticket category.
    """

    if category in HIGH_PRIORITY:
        return "High"

    if category in MEDIUM_PRIORITY:
        return "Medium"

    return "Low"


def enrich_dataset(df: pd.DataFrame) -> pd.DataFrame:

    df["priority"] = df["category"].apply(assign_priority)

    return df