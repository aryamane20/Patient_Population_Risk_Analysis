"""
care_plan.py - turn a patient's risk factors into an actionable care plan (Day 27).

Two paths, same output contract:
  1. Claude (when ANTHROPIC_API_KEY is set): a structured prompt asks Claude for an
     evidence-based 30-day readmission-prevention plan as strict JSON.
  2. Rule-based fallback (no key / any API error): deterministic keyword →
     intervention mapping so the dashboard ALWAYS produces a plan for the demo.

Output schema (both paths):
    {
      "summary": str,
      "interventions": [ {"category": str, "action": str, "priority": str}, ... ],
      "source": "claude" | "rule_based",
    }
"""

from __future__ import annotations

import json
import os

# default model is overridable so we never hard-fail on a renamed model id
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-3-5-sonnet-latest")
PRIORITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}


def claude_available() -> bool:
    """True only if the SDK is importable AND an API key is present."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────
# Rule-based fallback
# ─────────────────────────────────────────────
# keyword found in a risk factor -> (category, action, priority)
_RULES = [
    ("prior inpatient", ("Transitional care",
        "Enroll in Transitional Care Management; nurse call within 48h of discharge", "High")),
    ("prior admissions", ("Transitional care",
        "Assign a care coordinator for 30-day post-discharge monitoring", "High")),
    ("prior er", ("Utilization",
        "Review ED utilization drivers; set up an ambulatory care access plan", "High")),
    ("insulin", ("Diabetes management",
        "Diabetes educator review of insulin regimen; confirm glucometer + supplies", "High")),
    ("diabetes", ("Diabetes management",
        "Reinforce diabetes self-management education and dietary counseling", "Medium")),
    ("hba1c", ("Diabetes management",
        "Endocrinology follow-up for glycemic control; recheck HbA1c in 4-6 weeks", "Medium")),
    ("glucose", ("Diabetes management",
        "Establish home glucose monitoring with logging and clinician review", "Medium")),
    ("diagnoses on record", ("Complex care",
        "Complete medication reconciliation across all conditions", "High")),
    ("medications changed", ("Medication safety",
        "Pharmacist review of medication changes; counsel on adherence", "High")),
    ("active medications", ("Medication safety",
        "Assess polypharmacy risk; simplify regimen where possible", "Medium")),
    ("medications", ("Medication safety",
        "Pharmacist-led medication reconciliation and adherence check", "Medium")),
    ("days in hospital", ("Home support",
        "Order home health assessment given the prolonged length of stay", "Medium")),
    ("discharge disposition", ("Care transitions",
        "Confirm discharge destination has adequate support and follow-up", "Medium")),
    ("primary diagnosis", ("Disease-specific",
        "Coordinate specialist follow-up for the primary diagnosis", "Medium")),
    ("age", ("Geriatric care",
        "Fall-risk and functional assessment; consider geriatric consult", "Medium")),
    ("procedures", ("Post-procedure",
        "Post-procedure wound/recovery check and symptom education", "Low")),
]


def _rule_based_plan(patient: dict) -> dict:
    tier = str(patient.get("tier", "Medium"))
    factors = str(patient.get("top_3_risk_factors", "")).lower()

    interventions: list[dict] = []
    seen_categories = set()
    for keyword, (category, action, priority) in _RULES:
        if keyword in factors and category not in seen_categories:
            interventions.append({"category": category, "action": action, "priority": priority})
            seen_categories.add(category)

    # universal baseline for any discharged patient
    interventions.append({
        "category": "Primary care",
        "action": "Schedule PCP follow-up within 7 days of discharge",
        "priority": "High" if tier == "High" else "Medium",
    })

    if tier == "High":
        interventions.insert(0, {
            "category": "Escalation",
            "action": "Flag for intensive case management; daily check-ins for the first week",
            "priority": "High",
        })

    interventions.sort(key=lambda i: PRIORITY_ORDER.get(i["priority"], 3))

    summary = (
        f"{tier}-risk patient (score {patient.get('risk_score', 'NA')}/100). "
        f"Key drivers: {patient.get('top_3_risk_factors', 'n/a')}. "
        "Plan targets the top modifiable readmission drivers with a structured "
        "30-day transition."
    )
    return {"summary": summary, "interventions": interventions, "source": "rule_based"}


# ─────────────────────────────────────────────
# Claude path
# ─────────────────────────────────────────────
_SYSTEM = (
    "You are a clinical care-coordination assistant supporting hospital discharge "
    "planning. You produce concise, evidence-based 30-day readmission-prevention "
    "care plans. You never invent patient facts beyond what is provided. "
    "Respond with STRICT JSON only, no prose or markdown."
)

_USER_TEMPLATE = """Create a 30-day readmission-prevention care plan for this discharged patient.

Patient:
- Risk tier: {tier}
- Risk score: {risk_score}/100 (probability {probability})
- Top risk factors (from a SHAP-explained model): {factors}

Return JSON with EXACTLY this shape:
{{
  "summary": "1-2 sentence clinical summary of the patient's readmission risk",
  "interventions": [
    {{"category": "short category", "action": "specific, actionable step", "priority": "High|Medium|Low"}}
  ]
}}
Provide 4-6 interventions, ordered most important first, each tied to a stated risk factor where possible."""


def _claude_plan(patient: dict) -> dict:
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    prompt = _USER_TEMPLATE.format(
        tier=patient.get("tier"),
        risk_score=patient.get("risk_score"),
        probability=patient.get("probability"),
        factors=patient.get("top_3_risk_factors"),
    )
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=900,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in msg.content if getattr(block, "type", None) == "text")
    # tolerate accidental code fences
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(text)
    interventions = data.get("interventions", [])
    interventions.sort(key=lambda i: PRIORITY_ORDER.get(i.get("priority", "Low"), 3))
    return {
        "summary": data.get("summary", ""),
        "interventions": interventions,
        "source": "claude",
    }


def generate_care_plan(patient: dict, use_claude: bool | None = None) -> dict:
    """
    Generate a care plan for one patient.

    `patient` needs: tier, risk_score, probability, top_3_risk_factors.
    `use_claude=None` -> auto (Claude if available, else rule-based). Any Claude
    failure falls back to the rule-based plan so the caller always gets a result.
    """
    want_claude = claude_available() if use_claude is None else use_claude
    if want_claude:
        try:
            return _claude_plan(patient)
        except Exception as exc:  # never break the UI on an API hiccup
            plan = _rule_based_plan(patient)
            plan["summary"] = f"[Claude unavailable: {exc}] " + plan["summary"]
            return plan
    return _rule_based_plan(patient)


if __name__ == "__main__":
    demo = {
        "patient_id": 111042378, "tier": "High", "risk_score": 92,
        "probability": 0.9231,
        "top_3_risk_factors": "prior inpatient visits: 3; insulin: Down; prior admissions: 3",
    }
    print("claude_available():", claude_available())
    plan = generate_care_plan(demo)
    print(json.dumps(plan, indent=2))
