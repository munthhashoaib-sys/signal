import json
import os
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

_BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(_BASE_DIR / ".env")

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def revise_openers(rejected_openers, approved_chunks, prospect_name, prospect_role, prospect_company, ae_product):
    """
    Takes rejected openers and judge feedback.
    Builds a targeted correction prompt and regenerates only the rejected ones.
    Returns revised openers.
    """
    if not rejected_openers:
        print("No rejected openers to revise.")
        return []

    print(f"Reviser regenerating {len(rejected_openers)} rejected openers...")

    context_text = ""
    for i, chunk in enumerate(approved_chunks[:4]):
        text = chunk.get("text", "") if isinstance(chunk, dict) else chunk
        signal = chunk.get("key_signal", "") if isinstance(chunk, dict) else ""
        context_text += f"\n--- Filing Excerpt {i+1} ---\n"
        if signal:
            context_text += f"Key signal: {signal}\n"
        context_text += f"{text[:500]}\n"

    rejection_feedback = ""
    for i, opener in enumerate(rejected_openers):
        rejection_feedback += f"\nRejected opener {i+1}:\n"
        rejection_feedback += f"Text: {opener.get('text', '')}\n"
        rejection_feedback += f"Score: {opener.get('score', 0)}/10\n"
        rejection_feedback += f"Weakness: {opener.get('weakness', 'Too generic')}\n"
        if opener.get('specific_improvement'):
            rejection_feedback += f"Required fix: {opener.get('specific_improvement')}\n"

    system_prompt = """You are a senior sales strategist rewriting cold email openers that failed a quality review.

You will be given:
1. The specific reasons each opener was rejected
2. The actual filing excerpts containing the signal you must use
3. The prospect context

Your rewrites must:
- Reference a specific named detail from the filing excerpts provided (a metric, named program, named risk, named product, specific figure)
- Sound like a curious, informed peer — not a salesperson
- Be 2-3 sentences maximum
- End with a genuine question relevant to the prospect's role
- Never use phrases like: I noticed, I saw that, I wanted to reach out, I hope this finds you well

Return ONLY valid JSON:
{
  "revised_openers": [
    {
      "original_index": 0,
      "text": "the rewritten opener",
      "signal": "2-4 word label for what filing detail this references",
      "why": "one sentence explaining why this will resonate"
    }
  ]
}"""

    user_message = f"""Prospect: {prospect_name}, {prospect_role} at {prospect_company}
What the rep sells: {ae_product}

FILING EXCERPTS TO USE:
{context_text}

REJECTED OPENERS AND WHY THEY FAILED:
{rejection_feedback}

Rewrite each rejected opener. Use specific details from the filing excerpts above.
Return {len(rejected_openers)} revised openers."""

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}]
    )

    raw = response.content[0].text.strip()
    clean = raw.replace("```json", "").replace("```", "").strip()
    result = json.loads(clean)
    revised = result.get("revised_openers", [])

    print(f"Reviser produced {len(revised)} revised openers.")
    for i, opener in enumerate(revised):
        print(f"  Revised {i+1} [{opener.get('signal', '')}]: {opener.get('text', '')[:100]}...")

    return revised


if __name__ == "__main__":
    rejected_openers = [
        {
            "index": 0,
            "text": "Sarah, I noticed Rivian is growing fast and wanted to reach out.",
            "score": 2,
            "weakness": "Generic observation with I noticed phrasing. Applies to any fast-growing company.",
            "specific_improvement": "Reference a specific filing detail such as the sole-source battery supplier risk or the Amazon EDV ramp."
        },
        {
            "index": 3,
            "text": "I saw that Rivian is expanding and thought our platform could be a good fit.",
            "score": 2,
            "weakness": "Generic I saw that phrasing with vague fit statement.",
            "specific_improvement": "Use the tariff exposure on raw materials or the 100,000 EDV order as a specific hook."
        }
    ]

    approved_chunks = [
        {
            "text": "Rivian faces significant supply chain risk due to sole-source dependency on battery cell suppliers. The company has identified this as a top operational risk in its most recent 10-K filing.",
            "key_signal": "Sole-source battery cell supplier dependency"
        },
        {
            "text": "Amazon placed an initial order of 100,000 Electric Delivery Vans globally. The Commercial Van platform underpins the EDV variant and requires coordinated supplier ramps across multiple component categories.",
            "key_signal": "Amazon 100,000 EDV order requiring supplier ramp coordination"
        }
    ]

    revised = revise_openers(
        rejected_openers=rejected_openers,
        approved_chunks=approved_chunks,
        prospect_name="Sarah Chen",
        prospect_role="VP of Supply Chain",
        prospect_company="Rivian Automotive",
        ae_product="Supplier risk intelligence platform"
    )

    print(f"\nRevised openers:")
    for opener in revised:
        print(f"\n[{opener.get('signal', '')}]")
        print(opener.get('text', ''))
        print(f"Why: {opener.get('why', '')}")
