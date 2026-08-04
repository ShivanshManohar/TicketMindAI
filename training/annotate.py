import os
import json
import pandas as pd

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """
You are a senior customer support manager.

Given:
1. Customer ticket
2. Existing category
3. Existing subcategory

Return ONLY valid JSON.

{
    "category":"",
    "subcategory":"",
    "priority":"",
    "sentiment":"",
    "summary":""
}

Rules:
- Priority: Low / Medium / High
- Sentiment: Positive / Neutral / Negative
- Summary: Maximum 20 words

Do not explain anything.

Return ONLY valid JSON.
Do not wrap the response in markdown.
Do not explain.
Do not use ```json.
Your first character must be {
Your last character must be }
"""


def annotate(ticket, category, subcategory):

    prompt = f"""
Customer Ticket:
{ticket}

Existing Category:
{category}

Existing Subcategory:
{subcategory}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    output = response.choices[0].message.content.strip()

    try:
        return json.loads(output)

    except Exception:

        print("\nFAILED OUTPUT\n")
        print(output)

        return None