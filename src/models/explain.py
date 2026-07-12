"""Phase 4 Level-1 explainability: SHAP top-k drivers per prediction (plan.md §11.2).

SHAP is the source of truth; anything rendered on top (narrative, later) only
translates these values — it never invents a driver.
"""

import numpy as np
import pandas as pd
import shap

# Structural feature of the survival framing, not a borrower risk driver.
EXCLUDE_FROM_DRIVERS = ("months_since_origination",)

FEATURE_LABELS = {
    "credit_income_ratio": "Credit-to-income ratio",
    "annuity_income_ratio": "Annuity-to-income ratio",
    "bureau_active_count": "Active credit lines (bureau)",
    "bureau_overdue_flag": "Existing overdue on other credits (bureau)",
    "ext_source_1": "External credit score 1",
    "ext_source_2": "External credit score 2",
    "ext_source_3": "External credit score 3",
    "age_years": "Applicant age (years)",
    "employed_years": "Employment tenure (years)",
    "prior_late_rate": "Late-payment rate on earlier loans",
    "prior_n_installments": "Installments observed on earlier loans",
}


def build_explainer(model) -> shap.TreeExplainer:
    return shap.TreeExplainer(model)


def top_drivers(
    explainer: shap.TreeExplainer,
    X_row: pd.DataFrame,
    k: int = 5,
) -> list[dict]:
    """Top-k |SHAP| drivers for a single model-matrix row, plain-language labeled."""
    sv = explainer.shap_values(X_row)
    if isinstance(sv, list):  # older shap returns [class0, class1]
        sv = sv[1]
    sv = np.asarray(sv).reshape(-1)

    rows = []
    for feat, value, shap_val in zip(X_row.columns, X_row.iloc[0], sv):
        if feat in EXCLUDE_FROM_DRIVERS:
            continue
        rows.append(
            {
                "feature": feat,
                "label": FEATURE_LABELS.get(feat, feat.replace("_", " ")),
                "value": None if pd.isna(value) else round(float(value), 4),
                "shap": round(float(shap_val), 5),
                "direction": "increases risk" if shap_val > 0 else "decreases risk",
            }
        )
    rows.sort(key=lambda r: abs(r["shap"]), reverse=True)
    return rows[:k]
