"""Phase 7 Level-2 explainability: guarded plain-language narrative (plan.md §11.2, §12.3).

Hard architectural rule (plan.md §11.2 / Gap D2): the LLM only *translates*
already-computed SHAP values into prose — it never judges creditworthiness
and never sees the case file, only the top-5 driver tuples. The templated
narrative is the always-available default; the LLM path activates only when
ANTHROPIC_API_KEY is set, and whatever the LLM returns must pass
verify_narrative() (every stated number traced back to an actual SHAP value,
feature value, or the calibrated PD) before display — on any mismatch or API
error the system falls back to the template rather than show an unverified
number (§12.3 guardrail).
"""

import os
import re

import requests

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
ANTHROPIC_VERSION = "2023-06-01"
LLM_MAX_TOKENS = 250
LLM_TIMEOUT_S = 20

# The fixed survival horizon — the one number a narrative may always state.
HORIZON_MONTHS = 12

# Fixed-structure prompt (§11.2 Level 2): receives ONLY the top-5 SHAP
# feature/value pairs plus grade and calibrated PD — never the case file.
NARRATIVE_PROMPT_TEMPLATE = """\
You are a credit-risk narrative writer for a regulated early-warning system.
Using ONLY the verified model outputs below, write ONE short paragraph
(3-5 sentences) explaining this account's risk, in this fixed structure:
driver -> direction -> magnitude -> risk implication.
Rules: do not state any number that is not listed below; do not invent
drivers, causes, or recommendations; do not judge creditworthiness beyond
what the listed drivers say.

Risk grade: {risk_grade}
Calibrated {horizon}-month default probability: {pd_pct}%
Top SHAP drivers (feature | value | SHAP contribution | direction):
{driver_lines}
"""

# Narrative cache per (risk-grade, top-3-driver combination) — the §12.3
# latency budget: identical explanation patterns never re-call the LLM.
_NARRATIVE_CACHE: dict[tuple, dict] = {}

_NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


def _driver_lines(top_drivers: list[dict]) -> str:
    lines = []
    for d in top_drivers:
        value = "n/a" if d.get("value") is None else f"{d['value']:g}"
        lines.append(f"- {d['label']} | value={value} | shap={d['shap']:+g} | {d['direction']}")
    return "\n".join(lines)


def templated_narrative(top_drivers: list[dict], risk_grade: str, calibrated_pd: float) -> str:
    """Deterministic plain-language paragraph — the fallback AND the no-key default."""
    lead = top_drivers[0]
    lead_val = "" if lead.get("value") is None else f" (value {lead['value']:g})"
    parts = [
        f"This account is graded {risk_grade} with a calibrated {HORIZON_MONTHS}-month "
        f"default probability of {calibrated_pd * 100:.1f}%.",
        f"The largest driver is {lead['label'].lower()}{lead_val}, which {lead['direction']} "
        f"with a SHAP contribution of {lead['shap']:+g}.",
    ]
    rest = []
    for d in top_drivers[1:]:
        val = "" if d.get("value") is None else f" at {d['value']:g}"
        rest.append(f"{d['label'].lower()}{val} ({d['direction']}, SHAP {d['shap']:+g})")
    if rest:
        parts.append("Also contributing: " + "; ".join(rest) + ".")
    parts.append(
        "Taken together, these drivers determine the account's position on the unified "
        "risk-grade scale and the mapped monitoring action."
    )
    return " ".join(parts)


def _allowed_numbers(top_drivers: list[dict], calibrated_pd: float) -> set[float]:
    allowed = {float(HORIZON_MONTHS), float(calibrated_pd), float(calibrated_pd) * 100.0}
    for d in top_drivers:
        if d.get("value") is not None:
            allowed.add(float(d["value"]))
        allowed.add(float(d["shap"]))
        allowed.add(abs(float(d["shap"])))
    return allowed


def verify_narrative(text: str, top_drivers: list[dict], calibrated_pd: float) -> bool:
    """THE GUARDRAIL (§12.3): every number stated in the text must match one of
    the actual SHAP values, feature values, the calibrated PD (raw or as a
    percentage), or the fixed 12-month horizon, within a rounding tolerance.
    Returns False on any unverifiable number."""
    # Driver names are verified text: digits inside a label or feature name
    # (e.g. "External credit score 2") are not stated statistics.
    for d in top_drivers:
        for name in (d["label"], d["feature"]):
            text = re.sub(re.escape(name), " ", text, flags=re.IGNORECASE)
    allowed = _allowed_numbers(top_drivers, calibrated_pd)
    for match in _NUMBER_RE.finditer(text):
        x = float(match.group())
        if not any(abs(x - a) <= max(0.05 * abs(a), 0.006) for a in allowed):
            return False
    return True


def _call_llm(prompt: str) -> str:
    """Raw Messages-API call via requests — deliberately no SDK dependency."""
    resp = requests.post(
        ANTHROPIC_API_URL,
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        },
        json={
            "model": ANTHROPIC_MODEL,
            "max_tokens": LLM_MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=LLM_TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


def _cache_key(risk_grade: str, top_drivers: list[dict]) -> tuple:
    return (risk_grade, tuple(d["feature"] for d in top_drivers[:3]))


def clear_narrative_cache() -> None:
    _NARRATIVE_CACHE.clear()


def generate_narrative(top_drivers: list[dict], risk_grade: str, calibrated_pd: float) -> dict:
    """LLM narrative when ANTHROPIC_API_KEY is set (guardrail-verified),
    templated narrative otherwise or on any mismatch/API error.

    Returns {"text": str, "source": "llm"|"template", "verified": bool}.
    """
    key = _cache_key(risk_grade, top_drivers)
    if key in _NARRATIVE_CACHE:
        return _NARRATIVE_CACHE[key]

    result = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        prompt = NARRATIVE_PROMPT_TEMPLATE.format(
            risk_grade=risk_grade,
            horizon=HORIZON_MONTHS,
            pd_pct=f"{calibrated_pd * 100:.1f}",
            driver_lines=_driver_lines(top_drivers),
        )
        try:
            text = _call_llm(prompt)
            if verify_narrative(text, top_drivers, calibrated_pd):
                result = {"text": text.strip(), "source": "llm", "verified": True}
        except (requests.RequestException, KeyError, IndexError, ValueError, TypeError):
            result = None  # any API/parse failure -> templated fallback

    if result is None:
        text = templated_narrative(top_drivers, risk_grade, calibrated_pd)
        result = {
            "text": text,
            "source": "template",
            "verified": verify_narrative(text, top_drivers, calibrated_pd),
        }

    _NARRATIVE_CACHE[key] = result
    return result
