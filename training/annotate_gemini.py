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

SYSTEM_PROMPT = """
You are a customer support dataset annotator.

Given:

1. Customer ticket
2. Existing category
3. Existing subcategory

Return ONLY valid JSON.

{
    "priority":"HIGH|MEDIUM|LOW",
    "sentiment":"Positive|Neutral|Negative",
    "summary":"..."
}

Rules:

Priority

HIGH
- Security breach
- Fraud
- Unauthorized access
- Duplicate/incorrect charges
- Customer completely blocked

MEDIUM
- Payment failed
- Refund delayed
- Order modification
- Cancellation requests
- Technical issues

LOW
- Questions
- Information requests
- Routine account requests

Summary:
Maximum 20 words.
Return JSON only.
"""


def annotate(ticket, category, subcategory):

    prompt = f"""
Ticket:
{ticket}

Category:
{category}

Subcategory:
{subcategory}
"""

    for attempt in range(MAX_RETRIES):

        try:

            response = client.models.generate_content(
                model=MODEL,
                contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
            )

            text = response.text.strip()

            if text.startswith("```"):
                text = (
                    text.replace("```json", "")
                        .replace("```", "")
                        .strip()
                )

            result = json.loads(text)

            result["category"] = category
            result["subcategory"] = subcategory

            return result

        except Exception as e:

            logger.warning(
                "Attempt %d failed: %s",
                attempt + 1,
                e,
            )

            time.sleep(2 * (attempt + 1))

    return None