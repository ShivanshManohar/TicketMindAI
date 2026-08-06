import os
import json
import time
import logging

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

logger = logging.getLogger("annotate_gemini")

MODEL = "gemini-3.5-flash-lite"

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2

VALID_PRIORITIES = {"HIGH", "MEDIUM", "LOW"}
VALID_SENTIMENTS = {"Positive", "Neutral", "Negative"}

SYSTEM_PROMPT = """
You are a senior customer support operations manager creating
gold-standard labels for a machine learning training dataset.

You are given:
1. Customer ticket
2. Existing category
3. Existing subcategory

Your ONLY job is to output:

1. priority
2. sentiment
3. summary

=== PRIORITY RULES ===

Priority reflects business urgency, not politeness or grammar.

HIGH — use ONLY when at least one of these is true:

- Unauthorized access, account compromise, fraud, or suspected security breach.
- Customer has already suffered financial loss due to a platform/company error
  (duplicate charge, incorrect charge, unexpected deduction).
- Customer is completely blocked from accessing a critical service.
- Critical service outage affecting essential functionality.

DO NOT use HIGH simply because the customer cannot afford an order.

MEDIUM — use when:

- Payment failed.
- Refund delayed.
- Customer cannot complete an important task.
- Order cancellation / modification.
- Technical issue affecting workflow.
- Customer wants to cancel because they changed their mind or cannot afford it.

LOW — use when:

- Information request.
- How-to question.
- Routine account request.
- Product or policy question.

Tie-break:
If BOTH a cancellation request and financial hardship are mentioned,
use HIGH.

=== SENTIMENT ===

Positive:
Customer expresses appreciation or satisfaction.

Neutral:
Routine request without emotional language.

Negative:
Frustration, financial hardship, complaints, profanity.

=== SUMMARY ===

- Maximum 20 words.
- Third person.
- Factual.
- Do not include placeholders like {{Order Number}}.

=== FEW SHOT ===

"I cannot afford this order, cancel purchase"
→ HIGH
→ Negative
→ Customer cannot afford the order and requests cancellation

"I bought the same item twice."
→ MEDIUM
→ Neutral
→ Customer wants to cancel a duplicate purchase

"How do I cancel my purchase?"
→ LOW
→ Neutral
→ Customer asks how to cancel a purchase

Return ONLY valid JSON.

{
    "priority":"HIGH|MEDIUM|LOW",
    "sentiment":"Positive|Neutral|Negative",
    "summary":"..."
}
"""


def _validate(result):

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


def annotate(ticket, category, subcategory):

    prompt = f"""
Customer Ticket:
{ticket}

Existing Category:
{category}

Existing Subcategory:
{subcategory}

Return ONLY JSON.
"""

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = client.models.generate_content(
                model=MODEL,
                contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
            )

            output = response.text.strip()

            if output.startswith("```"):
                output = (
                    output.replace("```json", "")
                    .replace("```", "")
                    .strip()
                )

            result = json.loads(output)

            if not _validate(result):
                raise ValueError(f"Validation failed: {result}")

            result["category"] = category
            result["subcategory"] = subcategory

            return result

        except Exception as e:

            last_error = e

            logger.warning(
                "Attempt %d/%d failed: %s",
                attempt,
                MAX_RETRIES,
                e,
            )

            error = str(e)

            if "RESOURCE_EXHAUSTED" in error or "429" in error:

                logger.info(
                    "Gemini rate limit reached. Waiting 45 seconds..."
                )

                time.sleep(45)

            else:

                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    logger.error("All retries failed: %s", last_error)

    return None