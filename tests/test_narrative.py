"""Phase 7 narrative guardrail tests (plan.md §12.3): a wrong number is never displayed."""

import pytest
import requests

import src.explainability.narrative as narrative

DRIVERS = [
    {"feature": "ext_source_2", "label": "External credit score 2", "value": 0.21, "shap": 0.42, "direction": "increases risk"},
    {"feature": "credit_income_ratio", "label": "Credit-to-income ratio", "value": 6.75, "shap": 0.31, "direction": "increases risk"},
    {"feature": "prior_late_rate", "label": "Late-payment rate on earlier loans", "value": 0.18, "shap": 0.12, "direction": "increases risk"},
    {"feature": "age_years", "label": "Applicant age (years)", "value": 29.0, "shap": -0.05, "direction": "decreases risk"},
    {"feature": "ext_source_3", "label": "External credit score 3", "value": None, "shap": 0.04, "direction": "increases risk"},
]
PD = 0.104
GRADE = "E"


@pytest.fixture(autouse=True)
def _fresh_cache():
    narrative.clear_narrative_cache()
    yield
    narrative.clear_narrative_cache()


def test_templated_narrative_states_pd_and_passes_guardrail():
    text = narrative.templated_narrative(DRIVERS, GRADE, PD)
    assert "10.4%" in text
    assert "External credit score 2".lower() in text.lower()
    assert narrative.verify_narrative(text, DRIVERS, PD) is True


def test_guardrail_rejects_wrong_number():
    # Mandated guardrail test: a deliberately wrong number -> False.
    bad = "The 12-month default probability is 87.3%, driven by external credit score 2."
    assert narrative.verify_narrative(bad, DRIVERS, PD) is False


def test_guardrail_accepts_rounded_true_numbers():
    ok = ("Default probability is about 10% over 12 months; external credit score 2 "
          "at 0.21 increases risk (SHAP 0.42), while age (29) decreases risk.")
    assert narrative.verify_narrative(ok, DRIVERS, PD) is True


def test_llm_wrong_number_falls_back_to_template(monkeypatch):
    # Mandated: LLM returns an unverifiable number -> discard, show template.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(narrative, "_call_llm", lambda prompt: "PD is 55% because credit usage rose 300%.")
    out = narrative.generate_narrative(DRIVERS, GRADE, PD)
    assert out["source"] == "template"
    assert out["text"] == narrative.templated_narrative(DRIVERS, GRADE, PD)
    assert out["verified"] is True


def test_llm_api_error_falls_back_to_template(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def boom(prompt):
        raise requests.ConnectionError("no network")

    monkeypatch.setattr(narrative, "_call_llm", boom)
    out = narrative.generate_narrative(DRIVERS, GRADE, PD)
    assert out["source"] == "template"
    assert out["verified"] is True


def test_no_api_key_uses_template_without_calling_llm(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def fail(prompt):  # pragma: no cover - must never run
        raise AssertionError("LLM must not be called without an API key")

    monkeypatch.setattr(narrative, "_call_llm", fail)
    out = narrative.generate_narrative(DRIVERS, GRADE, PD)
    assert out["source"] == "template"
    assert out["verified"] is True


def test_verified_llm_narrative_is_cached_per_grade_and_top3_drivers(monkeypatch):
    # §12.3 latency budget: identical (grade, top-3 drivers) never re-calls the LLM.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    calls = {"n": 0}

    def fake_llm(prompt):
        calls["n"] += 1
        return ("Grade E account: 12-month PD of 10.4%. External credit score 2 at 0.21 "
                "increases risk (SHAP 0.42), raising near-term default risk.")

    monkeypatch.setattr(narrative, "_call_llm", fake_llm)
    first = narrative.generate_narrative(DRIVERS, GRADE, PD)
    second = narrative.generate_narrative(DRIVERS, GRADE, PD)
    assert first["source"] == "llm" and first["verified"] is True
    assert second == first
    assert calls["n"] == 1
