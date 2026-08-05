from config import *


def load_model():
    """
    Tomorrow we will load:

    tokenizer
    base model
    LoRA adapter

    from here.
    """

    return None, None


model, tokenizer = load_model()


def predict_ticket(ticket: str):

    """
    Tomorrow this function will call:

    tokenizer(...)
        ↓
    model.generate(...)
        ↓
    json.loads(...)
    """

    return {
        "category": "Order",
        "subcategory": "Cancel Order",
        "priority": "Medium",
        "sentiment": "Negative",
        "summary": "Customer requests order cancellation."
    }