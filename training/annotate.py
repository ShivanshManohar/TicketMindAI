import os
import json
import time
import logging

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

logger = logging.getLogger("annotate")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "llama-3.3-70b-versatile"
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2

VALID_PRIORITIES = {"HIGH", "MEDIUM", "LOW"}
VALID_SENTIMENTS = {"Positive", "Neutral", "Negative"}

# ---------------------------------------------------------------------------
# System prompt
#
# Design notes (why this differs from the original):
# 1. The original prompt gave abstract category definitions but no worked
#    examples at the boundary between HIGH and MEDIUM. On your own sample
#    data this caused "I can no longer afford X" -> MEDIUM but
#    "I can no longer pay for X" -> HIGH for functionally identical tickets.
#    Few-shot anchors below fix that specific boundary.
# 2. Sentiment guidance is added because raw negative-affect words
#    ("goddamn", "damn item") were previously left to the model's judgement
#    with no rule, risking inconsistent labeling of profanity-driven venting
#    vs genuine dissatisfaction.
# 3. JSON is now enforced via response_format at the API level (see below),
#    so the prompt's job is only to get CONTENT right, not syntax.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior customer support operations manager creating
gold-standard labels for a machine learning training dataset.

You are given a customer ticket, plus its already-correct category and
subcategory. Your ONLY job is to output:

1. priority
2. sentiment
3. summary

=== PRIORITY RULES ===

Priority reflects business urgency, not politeness or grammar.

HIGH — use ONLY when at least one of these is true:

- Unauthorized access, account compromise, fraud, or suspected security breach.
- Customer has already suffered financial loss due to a platform/company error
  (e.g., duplicate charge, incorrect charge, money deducted unexpectedly).
- Customer is completely blocked from accessing or using a critical service
  with no available workaround.
- Critical service outage affecting essential functionality.

DO NOT use HIGH simply because the customer says they cannot afford or cannot
pay for an order. Financial hardship alone is NOT a HIGH priority unless it is
caused by a platform or payment system issue.


MEDIUM — use when:

- Customer cannot complete an important task.
- Payment failed or could not be completed.
- Customer cannot cancel, modify, or place an order.
- Refund is delayed.
- Technical issue interrupts an important workflow.
- Customer wants to cancel an order due to personal reasons
  (e.g., changed mind, cannot afford it, no longer needs it).
  
  
LOW — use when:
- Pure question, "how do I..." / "where do I..." with no request yet acted on.
- Customer expresses confusion but not urgency ("I don't know how to...").
- General informational or routine account request.
- Product or policy questions.

Tie-break rule: if a ticket contains BOTH a routine cancellation reason AND
a financial-hardship statement ("can't afford", "can't pay", "cannot afford"),
priority is HIGH — hardship overrides routine framing.

=== SENTIMENT RULES ===

Positive — customer expresses satisfaction, gratitude, or a compliment.
Neutral — plain informational tone, ordinary request phrasing, no strong
  affect either way, even if terse or blunt.
Negative — customer expresses frustration, distress, financial hardship, or
  uses profanity/venting language ("goddamn", "damn"). Profanity is always
  Negative even if the underlying request is routine.

=== SUMMARY RULES ===

- Maximum 20 words.
- Third person, factual, no restated customer emotion words unless
  necessary for clarity (e.g. do not write "customer is very upset", write
  what they want).
- Do not include the order number placeholder in the summary.

=== FEW-SHOT EXAMPLES (boundary cases) ===

Ticket: "I cannot afford this order, cancel purchase {{Order Number}}"
-> {"priority": "HIGH", "sentiment": "Negative", "summary": "Customer cannot afford the order and requests cancellation"}

Ticket: "I can no longer pay for order {{Order Number}}"
-> {"priority": "HIGH", "sentiment": "Negative", "summary": "Customer can no longer pay for the order"}

Ticket: "I bought the same item twice, cancel order {{Order Number}}"
-> {"priority": "MEDIUM", "sentiment": "Neutral", "summary": "Customer wants to cancel a duplicate order"}

Ticket: "what do I have to do to cancel purchase {{Order Number}}?"
-> {"priority": "LOW", "sentiment": "Neutral", "summary": "Customer asks how to cancel their purchase"}

Ticket: "help me to cancel my last goddamn purchase"
-> {"priority": "MEDIUM", "sentiment": "Negative", "summary": "Customer wants to cancel their last purchase"}

=== OUTPUT FORMAT ===

Return ONLY a JSON object, no markdown fences, no explanation:

{"priority": "HIGH|MEDIUM|LOW", "sentiment": "Positive|Neutral|Negative", "summary": "..."}
"""


def _build_user_prompt(ticket: str, category: str, subcategory: str) -> str:
    return f"""Customer Ticket:
{ticket}

Existing Category:
{category}

Existing Subcategory:
{subcategory}

Generate ONLY priority, sentiment, summary as JSON."""


def _validate(result: dict) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("priority") not in VALID_PRIORITIES:
        return False
    if result.get("sentiment") not in VALID_SENTIMENTS:
        return False
    summary = result.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return False
    if len(summary.split()) > 25:  # small buffer over the 20-word instruction
        return False
    return True


def annotate(ticket: str, category: str, subcategory: str) -> dict | None:
    """
    Calls the LLM to annotate one ticket. Retries on malformed / invalid
    output up to MAX_RETRIES times, using JSON mode to eliminate most
    formatting failures outright. Returns None only if all retries fail,
    so the caller can log and skip instead of crashing.
    """

    user_prompt = _build_user_prompt(ticket, category, subcategory)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                response_format={"type": "json_object"},
                messages=messages,
            )

            output = response.choices[0].message.content.strip()
            result = json.loads(output)

            if not _validate(result):
                raise ValueError(f"Failed validation: {result}")

            result["category"] = category
            result["subcategory"] = subcategory
            return result

        except Exception as e:
            last_error = e
            logger.warning(
                "Attempt %d/%d failed for ticket %r: %s",
                attempt, MAX_RETRIES, ticket[:60], e,
            )
            # Tell the model what went wrong and ask it to correct itself
            messages.append({
                "role": "user",
                "content": (
                    "Your previous response was invalid or malformed JSON. "
                    "Respond again with ONLY a valid JSON object matching "
                    "the required schema, using allowed enum values exactly."
                ),
            })
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    logger.error("All retries exhausted for ticket %r: %s", ticket[:60], last_error)
    return None