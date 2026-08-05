import json
import sys
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv
import os

_BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(_BASE_DIR / ".env")

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

JUDGE_RUBRIC = """
You are a sales intelligence quality judge. Your job is to evaluate whether a chunk of text from a 10-K filing contains genuine, specific signal that a sales rep can use to write a personalized cold email to a specific prospect.

Score each chunk from 1 to 10 using this rubric:

9-10: PASS — The chunk contains a named, specific detail (a metric, a named risk, a named initiative, a named product, a specific geographic market, a specific financial figure) that directly connects to this prospect's role and would not appear in a press release or company homepage.

7-8: PASS — The chunk contains useful context that is role-relevant and specific to this company, even if not deeply granular. A rep could reference it credibly.

5-6: BORDERLINE — The chunk is vague, generic, or only tangentially relevant to this prospect's function. A rep could technically use it but it would not impress.

1-4: FAIL — The chunk is boilerplate, financial table data, legal disclaimer, or generic language that applies to any company. No specific signal for a sales rep.

CRITICAL RULES:
- A chunk full of XBRL tags, accounting codes, or raw financial table data is always a 1-2.
- A chunk that mentions a specific risk, initiative, product line, geographic market, or executive statement is always at least a 7.
- Generic phrases like "we face competition" or "our business depends on key personnel" are always below 5.
- Your scores must be consistent. Two chunks of similar quality must receive similar scores.

Return ONLY valid JSON, no explanation, no markdown:
{
  "evaluations": [
    {
      "chunk_index": 0,
      "score": 8,
      "reason": "one sentence explaining the score",
      "key_signal": "the specific detail that makes this useful, or null if score below 7"
    }
  ]
}
"""

def judge_chunks(chunks, prospect_name, prospect_role, prospect_company, ae_product):
    """
    Score each chunk for sales relevance.
    Returns (approved_chunks, rejected_chunks, evaluations)
    """
    print(f"Judge evaluating {len(chunks)} chunks...")

    chunks_text = ""
    for i, chunk in enumerate(chunks):
        chunks_text += f"\n--- CHUNK {i} ---\n{chunk[:600]}\n"

    user_message = f"""Prospect: {prospect_name}, {prospect_role} at {prospect_company}
What the rep sells: {ae_product}

Evaluate each chunk below for sales signal quality given this specific prospect and role.

{chunks_text}

Score all {len(chunks)} chunks and return JSON."""

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=2000,
        system=JUDGE_RUBRIC,
        messages=[{"role": "user", "content": user_message}]
    )

    raw = response.content[0].text.strip()
    clean = raw.replace("```json", "").replace("```", "").strip()
    result = json.loads(clean)
    evaluations = result.get("evaluations", [])

    approved_chunks = []
    rejected_chunks = []

    for ev in evaluations:
        idx = ev.get("chunk_index", 0)
        score = ev.get("score", 0)
        if idx < len(chunks):
            chunk_with_meta = {
                "text": chunks[idx],
                "score": score,
                "reason": ev.get("reason", ""),
                "key_signal": ev.get("key_signal", None)
            }
            if score >= 7:
                approved_chunks.append(chunk_with_meta)
            else:
                rejected_chunks.append(chunk_with_meta)

    approved_chunks.sort(key=lambda x: x["score"], reverse=True)

    print(f"Judge result: {len(approved_chunks)} approved, {len(rejected_chunks)} rejected")
    for ev in evaluations:
        idx = ev.get("chunk_index", 0)
        score = ev.get("score", 0)
        status = "PASS" if score >= 7 else "FAIL"
        print(f"  Chunk {idx}: {score}/10 [{status}] — {ev.get('reason', '')[:80]}")

    return approved_chunks, rejected_chunks, evaluations


def needs_fallback(approved_chunks, min_required=3):
    """Returns True if we don't have enough high-quality chunks."""
    return len(approved_chunks) < min_required


if __name__ == "__main__":
    sys.path.insert(0, str(_BASE_DIR))
    from backend.retriever import retrieve

    print("Testing judge with Rivian supply chain query...")
    chunks, scores = retrieve("supply chain risks and strategic initiatives at Rivian")

    approved, rejected, evals = judge_chunks(
        chunks=chunks,
        prospect_name="Sarah Chen",
        prospect_role="VP of Supply Chain",
        prospect_company="Rivian Automotive",
        ae_product="Supplier risk intelligence platform"
    )

    print(f"\nApproved chunks: {len(approved)}")
    for i, chunk in enumerate(approved):
        print(f"\n[{i+1}] Score: {chunk['score']}/10")
        print(f"Signal: {chunk['key_signal']}")
        print(f"Text preview: {chunk['text'][:150]}...")

    print(f"\nNeeds fallback: {needs_fallback(approved)}")
