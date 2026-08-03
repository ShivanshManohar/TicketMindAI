# TicketMindAI Output Schema (Version 1)

## Purpose

This document defines the standard response format that every AI-generated customer support ticket must follow.

The goal is to make the output predictable, structured, and easy for downstream systems and support agents to process.

---

# Output Schema

```json
{
  "categories": [
    {
      "category": "",
      "subcategory": ""
    }
  ],
  "priority": "",
  "sentiment": "",
  "summary": ""
}
```

---

# Field Explanations

## Categories

Purpose:

Allows a ticket to belong to one or more business domains.

Examples:

- Billing
- Technical
- Account
- Shipping

Each category also contains a more specific subcategory.

---

## Priority

Purpose:

Represents the business urgency of the issue.

Allowed Values:

- Low
- Medium
- High
- Critical

Priority is determined by business impact rather than customer emotion.

---

## Sentiment

Purpose:

Captures the customer's emotional tone.

Possible Values:

- Positive
- Neutral
- Negative

Sentiment is used for analytics and customer experience reporting.

It should not automatically determine priority.

---

## Summary

Purpose:

Provides a concise one-sentence description of the customer's issue.

Maximum Length:

20–25 words.

---

# Rejected Fields

## Assigned Team

Reason:

Departments differ across companies.

Routing should be handled by backend business logic instead of the language model.

---

# Open Questions

- Should confidence scores be included?
- Should multiple priorities ever exist?
- Should the model predict issue severity separately from business priority?