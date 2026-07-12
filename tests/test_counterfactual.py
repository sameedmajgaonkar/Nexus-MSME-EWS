"""Phase 7 counterfactual test (plan.md §11.2 Level 3): the suggested change,
applied and re-scored through the real trained artifacts, actually improves the grade."""

from pathlib import Path

# joblib here loads only this repo's own trained model artifacts (trusted).
import joblib
import pandas as pd
import pytest

from src.explainability.counterfactual import (
    GRADE_ORDER,
    IMPROVEMENT_BOUNDS,
    find_counterfactual,
    score_loan,
)
from src.models.explain import build_explainer, top_drivers
from src.models.hazard import MAX_MONTHS, align_columns, build_model_matrix

ROOT = Path(__file__).resolve().parents[1]
MAX_CANDIDATE_LOANS = 30


@pytest.fixture(scope="module")
def artifacts():
    paths = {
        "hazard": ROOT / "models" / "hazard.joblib",
        "calibrators": ROOT / "models" / "calibrators.joblib",
        "features": ROOT / "data" / "processed" / "serving_features.parquet",
        "scored": ROOT / "data" / "processed" / "phase4_scored_test.parquet",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        pytest.skip(f"trained artifacts missing: {missing}")
    bundle = joblib.load(paths["hazard"])
    return {
        "model": bundle["model"],
        "columns": bundle["columns"],
        "calibrators": joblib.load(paths["calibrators"]),
        "features": pd.read_parquet(paths["features"]),
        "scored": pd.read_parquet(paths["scored"]),
    }


def _drivers_for(art, loan_row, explainer):
    loan_matrix = loan_row.copy()
    loan_matrix["months_since_origination"] = MAX_MONTHS
    X_row = align_columns(build_model_matrix(loan_matrix), art["columns"])
    return top_drivers(explainer, X_row)


def test_counterfactual_change_actually_improves_grade(artifacts):
    art = artifacts
    explainer = build_explainer(art["model"])

    # Sample high-risk loans: grade D or worse, nearest the boundary first so
    # a small realistic change can plausibly cross into the next-better grade.
    high_risk = (
        art["scored"][art["scored"]["risk_grade"].isin(list("DEFG"))]
        .sort_values("calibrated_pd_12m")
        .head(MAX_CANDIDATE_LOANS)
    )
    assert not high_risk.empty

    verified_one = False
    for sk_id in high_risk["SK_ID_CURR"]:
        loan_row = art["features"][art["features"]["SK_ID_CURR"] == sk_id].reset_index(drop=True)
        if loan_row.empty:
            continue
        segment = loan_row.iloc[0]["loan_type_segment"]
        drivers = _drivers_for(art, loan_row, explainer)

        cf = find_counterfactual(
            art["model"], art["columns"], art["calibrators"], segment, loan_row, drivers
        )
        if cf.get("feature") is None:
            continue

        # Only actionable numeric levers may ever be suggested.
        assert cf["feature"] in IMPROVEMENT_BOUNDS

        base_pd, base_grade = score_loan(
            art["model"], art["columns"], art["calibrators"], segment, loan_row
        )
        assert base_grade >= "D"

        # Apply the suggested change and re-score: the grade must improve to
        # exactly the returned new_grade.
        perturbed = loan_row.copy()
        perturbed[cf["feature"]] = cf["suggested_value"]
        new_pd, new_grade = score_loan(
            art["model"], art["columns"], art["calibrators"], segment, perturbed
        )
        assert new_grade == cf["new_grade"]
        assert GRADE_ORDER.index(new_grade) < GRADE_ORDER.index(base_grade)
        assert new_pd == pytest.approx(cf["new_pd"], abs=1e-4)
        assert new_pd < base_pd
        verified_one = True
        break

    assert verified_one, "no counterfactual found for any sampled high-risk loan"


def test_structural_and_non_actionable_features_never_suggested(artifacts):
    art = artifacts
    loan_row = art["features"].head(1).reset_index(drop=True)
    segment = loan_row.iloc[0]["loan_type_segment"]
    fake_drivers = [
        {"feature": "loan_type_segment_term_loan_proxy", "label": "segment dummy", "value": 1.0, "shap": 0.9, "direction": "increases risk"},
        {"feature": "months_since_origination", "label": "months", "value": 12.0, "shap": 0.8, "direction": "increases risk"},
        {"feature": "age_years", "label": "Applicant age (years)", "value": 30.0, "shap": 0.7, "direction": "increases risk"},
    ]
    cf = find_counterfactual(
        art["model"], art["columns"], art["calibrators"], segment, loan_row, fake_drivers
    )
    assert cf["feature"] is None
    assert "no actionable" in cf["reason"]
