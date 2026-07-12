"""Phase 3 discrete-time hazard model on the person-period panel (plan.md §9.2 Option A).

Feature note: installment-derived Phase 1 features (late_installment_rate etc.)
are deliberately EXCLUDED here — they are computed from the same installment
records that define the panel's event, which would be direct label leakage.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping

STATIC_FEATURES = [
    "credit_income_ratio",
    "annuity_income_ratio",
    "bureau_active_count",
    "bureau_overdue_flag",
    "ext_source_1",
    "ext_source_2",
    "ext_source_3",
    "age_years",
    "employed_years",
    # late-rate over the applicant's OTHER (older) previous loans — behavioral
    # signal that excludes the panel loan itself, so no label leakage.
    "prior_late_rate",
    "prior_n_installments",
]
SEGMENT_COLS = ["loan_type_segment", "sector_segment", "data_richness"]
MAX_MONTHS = 12


def build_model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Static features + one-hot segments + months_since_origination."""
    X = df[STATIC_FEATURES + ["months_since_origination"]].copy()
    for col in SEGMENT_COLS:
        dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
        X = pd.concat([X, dummies], axis=1)
    return X


def align_columns(X: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Reindex a scoring matrix onto the training columns (missing dummies -> 0)."""
    return X.reindex(columns=columns, fill_value=0)


def train_hazard_model(panel_features: pd.DataFrame, split: pd.Series) -> tuple[LGBMClassifier, list[str]]:
    X = build_model_matrix(panel_features)
    y = panel_features["event"]

    train_mask = (split == "train").to_numpy()
    calib_mask = (split == "calib").to_numpy()

    # No scale_pos_weight here, deliberately: reweighting saturates monthly
    # hazards toward 1, and the 12-month product 1-prod(1-h) then collapses to
    # 1.0 for every loan (all ties, AUC ~0.5). A hazard model needs native
    # probabilities; miscalibration is handled downstream by isotonic (§9.4).
    model = LGBMClassifier(
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=1000,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        verbose=-1,
    )
    model.fit(
        X[train_mask],
        y[train_mask],
        eval_set=[(X[calib_mask], y[calib_mask])],
        eval_metric="auc",
        callbacks=[early_stopping(50, verbose=False)],
    )
    return model, list(X.columns)


def predict_hazard_curve(model: LGBMClassifier, columns: list[str], loan_row: pd.DataFrame) -> np.ndarray:
    """Predict the 12-value monthly hazard vector for one loan's static features."""
    months = pd.concat([loan_row] * MAX_MONTHS, ignore_index=True)
    months["months_since_origination"] = np.arange(1, MAX_MONTHS + 1)
    X = align_columns(build_model_matrix(months), columns)
    return model.predict_proba(X)[:, 1]


def cumulative_pd(hazard_curve: np.ndarray) -> float:
    """P(event within 12 months) = 1 - prod(1 - h_m)."""
    return float(1.0 - np.prod(1.0 - hazard_curve))
