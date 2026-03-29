"""
Generate cold email openers from 10-K RAG context using Claude.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict

import anthropic
from dotenv import load_dotenv

_BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(_BASE_DIR / ".env")

MODEL = "claude-opus-4-5"
MAX_TOKENS = 2000

SYSTEM_PROMPT = """You are a sales intelligence assistant. Your job is to read excerpts from a \
company's 10-K annual filing and generate 5 cold email openers for a sales rep.

Each opener must:
- Reference a specific detail from the 10-K excerpts provided, such as a named \
risk, a financial metric, a strategic initiative, or a direct quote from management
- Be warm and conversational in tone, like a curious and informed peer
- Be between 2 and 4 sentences long
- Never mention the sales rep's product directly
- Never be generic. Phrases like "I noticed your company is growing" are forbidden
- Feel like it was written by someone who actually read the filing

Return your response as valid JSON in exactly this format:
{
  "openers": [
    {
      "text": "the email opener text",
      "signal": "2 to 4 word label for what 10-K detail this references",
      "why": "one sentence explaining why this will resonate with this prospect"
    }
  ]
}"""


def _parse_json_response(text: str) -> Dict[str, Any]:
    """Extract and parse JSON from Claude output (handles optional markdown fences)."""
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```\s*$", "", raw)
    return json.loads(raw)


def generate_openers(
    prospect_name: str,
    company_name: str,
    prospect_role: str,
    ae_product: str,
    rag_context: str,
) -> Dict[str, Any]:
    """
    Build prompts, call Claude, and return parsed JSON as a dict with an "openers" list.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")

    user_prompt = f"""Prospect name: {prospect_name}
Company: {company_name}
Prospect role: {prospect_role}
What the rep sells (for your situational awareness only — do not mention this in the openers): {ae_product}

10-K excerpts to use:
{rag_context}

Generate exactly 5 cold email openers following the system instructions. Output only valid JSON."""

    client = anthropic.Anthropic(api_key=api_key)

    print("Sending context to Claude...")
    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    print("Response received. Parsing openers...")
    combined = "".join(
        b.text for b in message.content if getattr(b, "type", None) == "text"
    )
    result = _parse_json_response(combined)

    openers = result.get("openers", [])
    print(f"Done. Generated {len(openers)} openers.")
    return result


if __name__ == "__main__":
    sample_context = """
    --- Excerpt 1 ---
    Rivian faces significant supply chain risk due to sole-source battery suppliers.

    --- Excerpt 2 ---
    The company is scaling its Normal Illinois plant to support Commercial Van production.

    --- Excerpt 3 ---
    Gross profit per vehicle remains negative as the company works toward unit economics.
    """

    result = generate_openers(
        prospect_name="Sarah Chen",
        company_name="Rivian Automotive",
        prospect_role="VP of Supply Chain",
        ae_product="Supplier risk intelligence platform",
        rag_context=sample_context,
    )

    for i, opener in enumerate(result["openers"]):
        print(f"\nOpener {i+1} [{opener['signal']}]")
        print(opener["text"])
        print(f"Why: {opener['why']}")
