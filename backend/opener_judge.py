import json
import os
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

_BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(_BASE_DIR / ".env")

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

OPENER_JUDGE_RUBRIC = """
You are an expert sales coach evaluating cold email openers for B2B sales reps.
Score each opener from 1 to 10.

9-10: APPROVE - References a specific named detail from the filing. Sounds human. Role-relevant.
7-8: APPROVE_WITH_NOTE - Good but one weakness. Flag it clearly.
5-6: REJECT - Too vague, sounds like AI, or applies to any company.
1-4: REJECT - Generic, cringe-worthy, or role-irrelevant.

RULES:
- Phrases like I noticed, I saw that, I wanted to reach out score maximum 5.
- Openers applying to any company in the industry score maximum 4.
- Openers referencing a specific number, named program, or named risk score minimum 7.
- Role mismatch scores maximum 5.

Return ONLY valid JSON:
{
  "evaluations": [
    {
      "opener_index": 0,
      "score": 8,
      "verdict": "APPROVE",
      "weakness": null,
      "specific_improvement": null
    }
  ]
}
"""

def judge_openers(openers, prospect_name, prospect_role, prospect_company, ae_product):
    print(f"Opener judge evaluating {len(openers)} openers...")

    openers_text = ""
    for i, opener in enumerate(openers):
        text = opener.get("text", "") if isinstance(opener, dict) else opener
        openers_text += f"\n--- OPENER {i} ---\n{text}\n"

    user_message = f"""Prospect: {prospect_name}, {prospect_role} at {prospect_company}
What the rep sells: {ae_product}

Evaluate each opener. Be ruthless. Generic openers must be rejected.

{openers_text}

Score all {len(openers)} openers and return JSON."""

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=2000,
        system=OPENER_JUDGE_RUBRIC,
        messages=[{"role": "user", "content": user_message}]
    )

    raw = response.content[0].text.strip()
    clean = raw.replace("```json", "").replace("```", "").strip()
    result = json.loads(clean)
    evaluations = result.get("evaluations", [])

    approved = []
    flagged = []
    rejected = []

    for ev in evaluations:
        idx = ev.get("opener_index", 0)
        score = ev.get("score", 0)
        verdict = ev.get("verdict", "REJECT")

        opener_text = ""
        if idx < len(openers):
            o = openers[idx]
            opener_text = o.get("text", "") if isinstance(o, dict) else o

        entry = {
            "index": idx,
            "text": opener_text,
            "score": score,
            "verdict": verdict,
            "weakness": ev.get("weakness"),
            "specific_improvement": ev.get("specific_improvement")
        }

        if isinstance(openers[idx], dict):
            entry["signal"] = openers[idx].get("signal", "")
            entry["why"] = openers[idx].get("why", "")

        if verdict == "APPROVE":
            approved.append(entry)
        elif verdict == "APPROVE_WITH_NOTE":
            flagged.append(entry)
        else:
            rejected.append(entry)

    print(f"Opener judge: {len(approved)} approved, {len(flagged)} flagged, {len(rejected)} rejected")
    for ev in evaluations:
        idx = ev.get("opener_index", 0)
        score = ev.get("score", 0)
        verdict = ev.get("verdict", "REJECT")
        weakness = ev.get("weakness", "")
        print(f"  Opener {idx+1}: {score}/10 [{verdict}]" + (f" - {weakness[:80]}" if weakness else ""))

    return approved, flagged, rejected, evaluations


if __name__ == "__main__":
    sample_openers = [
        {"text": "Sarah, I noticed Rivian is growing fast and wanted to reach out.", "signal": "generic"},
        {"text": "Sarah, Rivian 10-K flags sole-source dependency on battery cell suppliers as a top operational risk - curious whether your team is building redundancy or still in assessment mode.", "signal": "sole-source battery risk"},
        {"text": "The Amazon EDV order of 100,000 units caught my attention - coordinating supplier ramps across that volume while keeping R1 lines fed must create real prioritization tensions.", "signal": "Amazon EDV scale"},
        {"text": "I saw that Rivian is expanding and thought our platform could be a good fit.", "signal": "generic"},
        {"text": "Rivian filing mentions tariff exposure on raw materials as a named risk to manufacturing timelines - I imagine that uncertainty shows up in how you think about safety stock. Is that accurate?", "signal": "tariff risk"}
    ]

    approved, flagged, rejected, evals = judge_openers(
        openers=sample_openers,
        prospect_name="Sarah Chen",
        prospect_role="VP of Supply Chain",
        prospect_company="Rivian Automotive",
        ae_product="Supplier risk intelligence platform"
    )

    print(f"\nFinal: {len(approved)} approved, {len(flagged)} flagged, {len(rejected)} rejected")
