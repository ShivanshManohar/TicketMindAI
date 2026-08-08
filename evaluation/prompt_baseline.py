import os
import json
import time
import logging

import pandas as pd
from dotenv import load_dotenv
from google import genai


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

MODEL = "gemini-3.5-flash-lite"

INPUT_PATH = "data/processed/benchmark.csv"
OUTPUT_PATH = "data/processed/prompt_baseline_predictions.json"

BATCH_SIZE = 10
MAX_RETRIES = 5


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("prompt_baseline")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a senior customer support operations manager and expert ticket
classification system.

Your task is to classify customer support tickets into structured metadata.

You are NOT answering the customer.

You are NOT providing customer support.

You are ONLY classifying the ticket.

For every ticket, determine:

1. category
2. subcategory
3. priority

Use ONLY the information contained in the customer ticket.

Do not invent facts.

Do not infer information that is not reasonably supported by the ticket.

Classification must be consistent:
if two tickets express the same underlying intent, they should receive
the same category, subcategory, and priority.

==========================================================
CATEGORY AND SUBCATEGORY
==========================================================

Choose the category and subcategory that best represent the customer's
PRIMARY intent.

The available taxonomy is:

Account
- Create Account
- Delete Account
- Edit Account
- Recover Password
- Registration Problems
- Switch Account

Cancel
- Check Cancellation Fee

Contact
- Contact Customer Service
- Contact Human Agent

Delivery
- Delivery Options
- Delivery Period

Feedback
- Complaint
- Review

Invoice
- Check Invoice
- Get Invoice

Order
- Cancel Order
- Change Order
- Place Order
- Track Order

Payment
- Check Payment Methods
- Payment Issue

Refund
- Check Refund Policy
- Get Refund
- Track Refund

Shipping
- Change Shipping Address
- Set Up Shipping Address

Subscription
- Newsletter Subscription

Always select EXACTLY one category and one subcategory from this taxonomy.

Do not create new categories.

Do not create new subcategories.

----------------------------------------------------------
CATEGORY/SUBCATEGORY DECISION GUIDANCE
----------------------------------------------------------

CREATE ACCOUNT

Use when the customer wants to open, create, or register a new account.

DELETE ACCOUNT

Use when the customer wants to close, remove, or delete an existing account.

EDIT ACCOUNT

Use when the customer wants to change existing account information.

RECOVER PASSWORD

Use when the customer wants to recover, retrieve, reset, or regain access
to a password, PIN, key, or account credential.

REGISTRATION PROBLEMS

Use when the customer reports an error or problem during registration,
signup, or account creation.

SWITCH ACCOUNT

Use when the customer wants to switch between existing accounts or
change which account they are using.

CANCEL ORDER

Use when the customer wants to cancel an existing order.

CHANGE ORDER

Use when the customer wants to modify an existing order, such as changing
or removing an item.

PLACE ORDER

Use when the customer wants to place or make a new order.

TRACK ORDER

Use when the customer wants to track an existing order.

PAYMENT ISSUE

Use when the customer reports a payment failure, payment problem,
transfer problem, or inability to complete a payment.

CHECK PAYMENT METHODS

Use when the customer asks which payment methods are available.

GET REFUND

Use when the customer wants to obtain or request a refund.

TRACK REFUND

Use when the customer is asking about the status of an already-requested
refund.

CHECK REFUND POLICY

Use when the customer asks about refund rules, eligibility, or conditions.

CHANGE SHIPPING ADDRESS

Use when the customer wants to change or correct the shipping address
for an order.

SET UP SHIPPING ADDRESS

Use when the customer wants to add, create, or configure a shipping address.

CHECK INVOICE

Use when the customer wants to view, inspect, or check an invoice.

GET INVOICE

Use when the customer wants to obtain, receive, or retrieve an invoice.

DELIVERY OPTIONS

Use when the customer asks about available delivery/shipping methods.

DELIVERY PERIOD

Use when the customer asks how long delivery will take.

CHECK CANCELLATION FEE

Use when the customer asks about fees or charges associated with
cancellation.

CONTACT CUSTOMER SERVICE

Use when the customer wants to contact customer support generally.

CONTACT HUMAN AGENT

Use when the customer specifically wants to speak to a human/person/agent.

NEWSLETTER SUBSCRIPTION

Use when the customer wants to subscribe, unsubscribe, receive, or change
their newsletter subscription.

COMPLAINT

Use when the primary intent is to make a complaint rather than perform
one of the more specific actions above.

REVIEW

Use when the customer wants to leave feedback, an opinion, rating, or review.

==========================================================
PRIORITY
==========================================================

Priority measures BUSINESS URGENCY and BUSINESS IMPACT.

Priority does NOT measure:

- politeness
- grammar
- capitalization
- profanity
- emotional intensity alone

Choose exactly one:

HIGH
MEDIUM
LOW

----------------------------------------------------------
HIGH
----------------------------------------------------------

Use HIGH ONLY when at least one of these is true:

SECURITY

- Unauthorized account access
- Account compromise
- Fraud
- Suspicious activity
- Security breach
- Stolen account
- Unauthorized transaction

PLATFORM-CAUSED FINANCIAL LOSS

- Duplicate charge
- Incorrect charge
- Unexpected deduction
- Wrong amount charged
- Unauthorized payment caused by the platform
- Other confirmed financial loss caused by the company/platform

CRITICAL SERVICE FAILURE

- Customer is completely blocked from a critical service
- Critical functionality is unavailable
- Critical service outage
- Account access is completely blocked due to a system/service problem

Do NOT assign HIGH merely because the customer is angry.

Do NOT assign HIGH merely because the customer uses profanity.

Do NOT assign HIGH merely because the customer says they cannot afford
something.

----------------------------------------------------------
MEDIUM
----------------------------------------------------------

Use MEDIUM when the customer is experiencing a problem or needs assistance
to complete an important action, but there is no HIGH-level security,
financial-loss, or critical-service condition.

Examples:

- Payment failed
- Customer cannot complete payment
- Customer cannot cancel an order
- Customer cannot modify an order
- Customer cannot create an account
- Customer reports a technical problem
- Refund is delayed
- Customer needs assistance completing an action
- Customer is unable to complete an action
- Customer wants to cancel an existing order
- Customer cannot afford an order
- Customer wants to change account information

IMPORTANT:

Financial hardship alone is MEDIUM, not HIGH.

For example:

"I cannot afford this order."

→ MEDIUM

"I can no longer pay for this purchase."

→ MEDIUM

But:

"I was charged twice for this purchase."

→ HIGH

----------------------------------------------------------
LOW
----------------------------------------------------------

Use LOW when the customer is primarily asking for information or asking
HOW to perform a routine action without reporting that the action is
currently failing.

Examples:

- "How do I cancel my order?"
- "What payment methods are available?"
- "Where can I see the refund policy?"
- "How can I recover my password?"
- "How do I change my address?"
- "What are the delivery options?"

IMPORTANT DISTINCTION:

If the customer only asks HOW to perform an action:

→ LOW

If the customer needs assistance completing an action:

→ MEDIUM

If the customer explicitly reports being unable to complete an action:

→ MEDIUM

----------------------------------------------------------
PRIORITY TIE-BREAK RULE
----------------------------------------------------------

When multiple signals exist, use this precedence:

1. Security / fraud / unauthorized activity
2. Platform-caused financial loss
3. Critical service blockage
4. Active problem preventing an important task
5. Routine request or information question

The strongest applicable business-impact signal determines priority.

==========================================================
IMPORTANT SENTIMENT PRINCIPLE
==========================================================

Sentiment is NOT being evaluated in this benchmark.

Do not generate sentiment.

Profanity, frustration, or emotional language should NOT alter category,
subcategory, or priority unless it provides actual business-impact
information.

For example:

"help me cancel my fucking order"

is still:

category = Order
subcategory = Cancel Order
priority = MEDIUM

==========================================================
OUTPUT FORMAT
==========================================================

Return ONLY a JSON array.

The array must contain exactly one object for every ticket.

Each object MUST contain exactly:

{
    "category": "...",
    "subcategory": "...",
    "priority": "HIGH|MEDIUM|LOW"
}

Do not include:

- sentiment
- summary
- explanation
- confidence
- reasoning
- additional fields

Do not use markdown.

Do not use code fences.

Return valid JSON only.
"""


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def extract_json_array(text: str):

    text = text.strip()

    if text.startswith("```"):
        text = (
            text.replace("```json", "")
            .replace("```", "")
            .strip()
        )

    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1:
        raise ValueError(
            "Could not find JSON array in Gemini response."
        )

    return json.loads(
        text[start:end + 1]
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

VALID_PRIORITIES = {
    "HIGH",
    "MEDIUM",
    "LOW",
}

VALID_CATEGORIES = {
    "Account",
    "Cancel",
    "Contact",
    "Delivery",
    "Feedback",
    "Invoice",
    "Order",
    "Payment",
    "Refund",
    "Shipping",
    "Subscription",
}

VALID_SUBCATEGORIES = {
    "Create Account",
    "Delete Account",
    "Edit Account",
    "Recover Password",
    "Registration Problems",
    "Switch Account",
    "Check Cancellation Fee",
    "Contact Customer Service",
    "Contact Human Agent",
    "Delivery Options",
    "Delivery Period",
    "Complaint",
    "Review",
    "Check Invoice",
    "Get Invoice",
    "Cancel Order",
    "Change Order",
    "Place Order",
    "Track Order",
    "Check Payment Methods",
    "Payment Issue",
    "Check Refund Policy",
    "Get Refund",
    "Track Refund",
    "Change Shipping Address",
    "Set Up Shipping Address",
    "Newsletter Subscription",
}


def validate_prediction(prediction):

    if not isinstance(prediction, dict):
        return False

    if prediction.get("category") not in VALID_CATEGORIES:
        return False

    if prediction.get("subcategory") not in VALID_SUBCATEGORIES:
        return False

    if prediction.get("priority") not in VALID_PRIORITIES:
        return False

    return True


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(rows):

    sections = []

    for index, row in enumerate(rows, start=1):

        sections.append(
            f"""
TICKET {index}

Customer Ticket:
{row["input"]}
"""
        )

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Gemini inference
# ---------------------------------------------------------------------------

def predict_batch(rows):

    prompt = build_prompt(rows)

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = client.models.generate_content(
                model=MODEL,
                contents=(
                    SYSTEM_PROMPT
                    + "\n\n"
                    + prompt
                    + "\n\n"
                    + "Return exactly one JSON object for each ticket."
                ),
            )

            predictions = extract_json_array(
                response.text
            )

            if not isinstance(predictions, list):
                raise ValueError(
                    "Gemini response is not a JSON array."
                )

            if len(predictions) != len(rows):

                raise ValueError(
                    f"Expected {len(rows)} predictions, "
                    f"received {len(predictions)}."
                )

            for prediction in predictions:

                if not validate_prediction(prediction):

                    raise ValueError(
                        f"Invalid prediction: {prediction}"
                    )

            return predictions

        except Exception as error:

            logger.warning(
                "Batch attempt %d/%d failed: %s",
                attempt,
                MAX_RETRIES,
                error,
            )

            error_text = str(error)

            if (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
            ):

                wait_time = 45

                logger.warning(
                    "Gemini rate limit reached. "
                    "Waiting %d seconds.",
                    wait_time,
                )

                time.sleep(wait_time)

            else:

                time.sleep(
                    3 * attempt
                )

    raise RuntimeError(
        "Gemini batch failed after all retries."
    )


# ---------------------------------------------------------------------------
# Main benchmark generation
# ---------------------------------------------------------------------------

def main():

    df = pd.read_csv(INPUT_PATH)

    logger.info(
        "Loaded %d benchmark samples.",
        len(df),
    )

    predictions = []

    total = len(df)

    start_time = time.time()

    for start in range(
        0,
        total,
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            total,
        )

        rows = df.iloc[
            start:end
        ].to_dict("records")

        logger.info(
            "Processing benchmark samples %d-%d / %d",
            start + 1,
            end,
            total,
        )

        batch_predictions = predict_batch(
            rows
        )

        for row, prediction in zip(
            rows,
            batch_predictions,
        ):

            predictions.append(
                {
                    "input": row["input"],

                    "true_category": row["category"],
                    "true_subcategory": row["subcategory"],
                    "true_priority": row["priority"],

                    "pred_category": prediction["category"],
                    "pred_subcategory": prediction["subcategory"],
                    "pred_priority": prediction["priority"],
                }
            )

        with open(
            OUTPUT_PATH,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                predictions,
                f,
                indent=2,
                ensure_ascii=False,
            )

        elapsed = time.time() - start_time

        rate = (
            len(predictions) / elapsed
            if elapsed > 0
            else 0
        )

        remaining = (
            total - len(predictions)
        )

        eta_seconds = (
            remaining / rate
            if rate > 0
            else 0
        )

        logger.info(
            "Progress: %d/%d | %.2f tickets/sec | ETA %.1f min",
            len(predictions),
            total,
            rate,
            eta_seconds / 60,
        )

        # Keep comfortably below the free-tier RPM.
        time.sleep(5)

    logger.info("")
    logger.info(
        "Prompt baseline completed successfully."
    )

    logger.info(
        "Predictions: %d",
        len(predictions),
    )

    logger.info(
        "Saved to: %s",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()