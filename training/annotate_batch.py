import os
import json
import time
import logging
import re

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

logger = logging.getLogger("annotate_batch")

MODEL = "gemini-3.5-flash-lite"

BATCH_SIZE = 10

MAX_RETRIES = 5

VALID_PRIORITIES = {
    "HIGH",
    "MEDIUM",
    "LOW"
}

VALID_SENTIMENTS = {
    "Positive",
    "Neutral",
    "Negative"
}

SYSTEM_PROMPT = """
You are a Senior Customer Support Operations Manager and Data Annotation Specialist.

Your task is to generate CONSISTENT, HIGH-QUALITY labels for a customer support machine learning dataset.

You are NOT answering the customer.

You are NOT providing customer support.

You are ONLY annotating existing support tickets.

Each ticket already has the correct Category and Subcategory.

Your job is ONLY to determine:

1. Priority
2. Sentiment
3. Summary

Your annotations must be deterministic.

If two tickets describe the same problem, they should receive the same labels.

Never invent information.

Never infer details that are not present.

Use ONLY the ticket content.

==========================================================
PRIORITY
==========================================================

Priority measures BUSINESS IMPACT.

It does NOT measure:

- politeness
- frustration
- grammar
- writing quality
- profanity

Always choose ONE of:

HIGH
MEDIUM
LOW

----------------------------------------------------------
HIGH
----------------------------------------------------------

Assign HIGH ONLY if at least ONE condition is true.

SECURITY

• account hacked
• unauthorized login
• account compromise
• suspicious activity
• fraud
• phishing
• stolen account
• identity theft

FINANCIAL LOSS CAUSED BY THE PLATFORM

• duplicate charge

• charged twice

• wrong amount charged

• unexpected deduction

• unauthorized payment

• incorrect refund amount

• billing system error

CRITICAL SERVICE FAILURE

• customer completely blocked

• cannot access essential service

• essential functionality unavailable

• system outage

• account inaccessible

Examples

"I was charged twice."

HIGH

"My account was hacked."

HIGH

"I cannot log in because my account was compromised."

HIGH

----------------------------------------------------------
MEDIUM
----------------------------------------------------------

Assign MEDIUM when the customer needs assistance to complete an important task but there is no confirmed security incident or financial loss caused by the platform.

Examples

• payment failed

• refund delayed

• cannot cancel order

• cannot modify order

• technical issue

• cannot complete checkout

• order issue

• wants refund

• wants cancellation

• subscription issue

• account issue

Financial hardship belongs here.

Examples

"I cannot afford this order."

MEDIUM

"I no longer want this purchase."

MEDIUM

"I changed my mind."

MEDIUM

----------------------------------------------------------
LOW
----------------------------------------------------------

Assign LOW only when the ticket is informational.

Examples

• how do I...

• where can I...

• what is...

• policy question

• documentation

• account information

• pricing question

• product information

Examples

"How do I cancel my order?"

LOW

"What payment methods do you support?"

LOW

==========================================================
IMPORTANT PRIORITY RULES
==========================================================

If the customer explicitly needs help completing a task
or reports being unable to complete it,
assign MEDIUM.

If the customer only asks for information,
assign LOW.

Customer emotion NEVER increases priority.

Profanity NEVER increases priority.

Writing in capital letters NEVER increases priority.

Financial hardship alone is NOT HIGH.

Changing one's mind is NOT HIGH.

Routine cancellation is NOT HIGH.

Only actual business impact determines priority.

If the customer asks HOW to perform an action
→ LOW

If the customer requests assistance completing an action
→ MEDIUM

If the customer reports they cannot complete an action
→ MEDIUM

==========================================================
SENTIMENT
==========================================================

Choose exactly ONE.

Positive

Customer expresses satisfaction, appreciation, happiness or gratitude.

Examples

"Thank you."

"I love this service."

Neutral

Simple request.

Question.

Information seeking.

No emotional language.

Examples

"I want to cancel my order."

"What payment methods are available?"

Negative

Customer expresses

• frustration

• disappointment

• anger

• concern

• financial hardship

• complaint

• profanity

Examples

"This is ridiculous."

"I cannot afford this."

"My account was hacked."

"I've been charged twice."

==========================================================
SUMMARY
==========================================================

Produce ONE concise summary.

Rules

Summary Rules

Maximum 20 words.

Start with

Customer

Then describe the action.

Examples

Customer requests a refund.

Customer reports duplicate charges.

Customer cannot access the account.

Customer wants to cancel an order.

Customer asks about payment methods.

Avoid words like

information

inquires

regarding

requests information

tries to

attempts to

Keep summaries simple and action-focused.

==========================================================
OUTPUT FORMAT
==========================================================

Return ONLY valid JSON.

No markdown.

No explanation.

No extra text.

Exactly:

{
    "priority":"HIGH|MEDIUM|LOW",
    "sentiment":"Positive|Neutral|Negative",
    "summary":"..."
}

Never include category.

Never include subcategory.

Never include input.

Return ONLY JSON.
"""


def validate(result):

    if not isinstance(result, dict):
        return False

    if result.get("priority") not in VALID_PRIORITIES:
        return False

    if result.get("sentiment") not in VALID_SENTIMENTS:
        return False

    summary = result.get("summary")

    if not isinstance(summary, str):
        return False

    if len(summary.split()) > 20:
        return False

    return True


def build_prompt(rows):

    sections = []

    for idx, row in enumerate(rows, start=1):

        sections.append(
            f"""
Ticket {idx}

Customer Ticket:
{row["input"]}

Existing Category:
{row["category"]}

Existing Subcategory:
{row["subcategory"]}
"""
        )

    return "\n".join(sections)

def extract_json(text: str):

    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text)
        text = text.replace("```", "").strip()

    start = text.find("[")

    end = text.rfind("]")

    if start == -1 or end == -1:
        raise ValueError("Could not locate JSON array.")

    return json.loads(text[start:end + 1])


def annotate_batch(rows):

    prompt = build_prompt(rows)

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = client.models.generate_content(

                model=MODEL,

                contents=f"{SYSTEM_PROMPT}\n\n{prompt}"

            )

            predictions = extract_json(response.text)

            if not isinstance(predictions, list):

                raise ValueError(
                    "Gemini did not return a JSON array."
                )

            if len(predictions) != len(rows):

                raise ValueError(
                    f"Expected {len(rows)} predictions "
                    f"but received {len(predictions)}."
                )

            final = []

            for row, prediction in zip(rows, predictions):

                if not validate(prediction):

                    raise ValueError(
                        f"Validation failed:\n{prediction}"
                    )

                prediction["category"] = row["category"]

                prediction["subcategory"] = row["subcategory"]

                prediction["input"] = row["input"]

                final.append(prediction)

            return final

        except Exception as e:

            last_error = e

            error = str(e)

            logger.warning(
                "Batch attempt %d/%d failed:\n%s",
                attempt,
                MAX_RETRIES,
                error
            )

            if (
                "RESOURCE_EXHAUSTED" in error
                or
                "429" in error
            ):

                wait_time = 45

                match = re.search(
                    r"retry in\s+(\d+)",
                    error,
                    flags=re.IGNORECASE
                )

                if match:

                    wait_time = int(match.group(1)) + 2

                logger.info(
                    "Rate limit reached. Waiting %d seconds...",
                    wait_time
                )

                time.sleep(wait_time)

                continue

            if (
                "Expected" in error
                or
                "Validation failed" in error
                or
                "JSON" in error
            ):

                logger.info(
                    "Retrying after malformed model output..."
                )

                time.sleep(5)

                continue

            time.sleep(2 * attempt)

    logger.error(
        "Batch failed after %d retries.\n%s",
        MAX_RETRIES,
        last_error
    )

    return None