"""Phase 7 Level-3 explainability: minimal counterfactual via grid re-scoring (plan.md §11.2, §12.3).

Hackathon-realistic constrained search from §11.2: grid re-scoring of the SAME
trained hazard model over small perturbations of the top 2-3 SHAP-flagged
numeric features, reporting the smallest realistic change that moves the
account to the next-better risk grade (Gap D3). Features are only moved in
the risk-reducing direction implied by their positive SHAP contribution, and
only within realistic bounds (rates/ratios floor at 0, external scores cap
at 1). One-hot segment columns and months_since_origination are structural,
not actionable, and are never perturbed.
"""

import pandas as pd

from src.models.calibration import apply_calibration, pd_to_grade
from src.models.hazard import cumulative_pd, predict_hazard_curve

GRADE_ORDER = "ABCDEFG"

# Actionable numeric levers only: risk-reducing direction + realistic bound.
# Whitelist doubles as the exclusion of one-hot segment columns, structural
# columns (months_since_origination) and non-actionable traits (age).
IMPROVEMENT_BOUNDS = {
    "credit_income_ratio": ("down", 0.0),
    "annuity_income_ratio": ("down", 0.0),
    "bureau_active_count": ("down", 0.0),
    "bureau_overdue_flag": ("down", 0.0),
    "prior_late_rate": ("down", 0.0),
    "ext_source_1": ("up", 1.0),
    "ext_source_2": ("up", 1.0),
    "ext_source_3": ("up", 1.0),
}
INTEGER_FEATURES = {"bureau_active_count", "bureau_overdue_flag"}
MAX_CANDIDATE_FEATURES = 3


def score_loan(model, columns: list[str], calibrators: dict, segment: str, loan_row: pd.DataFrame) -> tuple[float, str]:
    """Full Phase 3/4 scoring chain for one loan: 12-month hazard expansion ->
    cumulative PD -> per-segment isotonic calibration -> unified grade."""
    raw = cumulative_pd(predict_hazard_curve(model, columns, loan_row))
    calibrated = float(apply_calibration(calibrators, pd.Series([segment]), pd.Series([raw]))[0])
    return calibrated, pd_to_grade(calibrated)


def next_better_grade(grade: str) -> str | None:
    idx = GRADE_ORDER.index(grade)
    return None if idx == 0 else GRADE_ORDER[idx - 1]


def find_counterfactual(
    model,
    columns: list[str],
    calibrators: dict,
    segment: str,
    loan_row: pd.DataFrame,
    top_drivers: list[dict],
    grid_steps: int = 10,
) -> dict:
    """Smallest single-feature change that reaches the next-better grade.

    Returns {feature, label, current_value, suggested_value, new_pd, new_grade}
    on success; {"feature": None, "reason": ...} when no counterfactual exists.
    """
    current_pd, current_grade = score_loan(model, columns, calibrators, segment, loan_row)
    target = next_better_grade(current_grade)
    if target is None:
        return {"feature": None, "reason": "already at the best grade (A)"}

    candidates = [
        d
        for d in top_drivers
        if d["shap"] > 0 and d["feature"] in IMPROVEMENT_BOUNDS and d.get("value") is not None
    ][:MAX_CANDIDATE_FEATURES]
    if not candidates:
        return {
            "feature": None,
            "reason": "no actionable risk-increasing numeric feature among the top SHAP drivers",
        }

    best: dict | None = None
    for d in candidates:
        feat = d["feature"]
        direction, bound = IMPROVEMENT_BOUNDS[feat]
        current = float(loan_row.iloc[0][feat])
        span = (current - bound) if direction == "down" else (bound - current)
        if span <= 0:
            continue  # already at (or past) the realistic bound
        tried: set[float] = set()
        for k in range(1, grid_steps + 1):
            frac = k / grid_steps
            value = current - frac * span if direction == "down" else current + frac * span
            if feat in INTEGER_FEATURES:
                value = float(round(value))
            if value in tried or value == current:
                continue
            tried.add(value)
            perturbed = loan_row.copy()
            perturbed[feat] = value
            new_pd, new_grade = score_loan(model, columns, calibrators, segment, perturbed)
            if GRADE_ORDER.index(new_grade) <= GRADE_ORDER.index(target):
                if best is None or frac < best["_frac"]:
                    best = {
                        "feature": feat,
                        "label": d["label"],
                        "current_value": round(current, 4),
                        "suggested_value": round(value, 4),
                        "new_pd": round(new_pd, 5),
                        "new_grade": new_grade,
                        "_frac": frac,
                    }
                break  # smallest change for this feature found; try the next lever

    if best is None:
        return {
            "feature": None,
            "reason": (
                f"no single-feature move within realistic bounds reaches grade {target} "
                f"(current grade {current_grade}, calibrated PD {current_pd:.3f})"
            ),
        }
    best.pop("_frac")
    return best
