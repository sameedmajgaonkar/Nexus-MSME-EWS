"""Phase 6 thin-file specialist: TabPFN for the NTC/NTB segment (plan.md §9.3, §12.9).

TabPFN is a pretrained tabular foundation model that performs in-context
inference in a single forward pass — no task-specific training loop — which is
exactly the regime the New-to-Credit/New-to-Bank sub-segment sits in. Per
plan.md §9.3, local CPU inference is kept under ~1,000 training rows.
train_lgbm_small is the mandated side-by-side comparator (§12.9): LightGBM on
the *same* tiny subsample, so the TabPFN choice is validated, not asserted.

Explainability: `tabpfn_extensions`' interpretability module is used when
importable; it is NOT currently installed, so the active path is a
shap.PermutationExplainer over predict_proba with a small background sample.
Either path emits the exact schema of src/models/explain.py:top_drivers.
"""

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier
from tabpfn.model_loading import ModelVersion

from src.models.explain import EXCLUDE_FROM_DRIVERS, FEATURE_LABELS

# CPU guidance from plan.md §9.3 — beyond this, use a GPU or tabpfn_client.
MAX_TABPFN_TRAIN_ROWS = 1000
# Structured features the thin-file model consumes (same static set as the
# hazard model — installment-behavior features on the scored loan are excluded
# there for leakage reasons and are empty for NTC/NTB borrowers anyway).
THIN_FILE_FEATURES = [
    "credit_income_ratio",
    "annuity_income_ratio",
    "bureau_active_count",
    "bureau_overdue_flag",
    "ext_source_1",
    "ext_source_2",
    "ext_source_3",
    "age_years",
    "employed_years",
    "prior_late_rate",
    "prior_n_installments",
]
# NaN handling: both TabPFN (v2+, verified against tabpfn 8.0.8) and LightGBM
# accept NaN natively, so raw NaNs are passed through — the §12.9 comparison
# stays on *identical* inputs, and a constant-fill run measurably hurt TabPFN
# (AUC 0.515 filled vs 0.585 NaN-native on the NTC/NTB subsample).
BACKGROUND_ROWS = 32  # background sample size for the permutation explainer

try:  # pragma: no cover - optional dependency, not in the current env
    from tabpfn_extensions.interpretability import shap as tabpfn_shap

    HAS_TABPFN_EXTENSIONS = True
except ImportError:
    tabpfn_shap = None
    HAS_TABPFN_EXTENSIONS = False

SHAP_PATH = "tabpfn_extensions.interpretability" if HAS_TABPFN_EXTENSIONS else "shap.PermutationExplainer fallback"


def train_tabpfn(X_train: pd.DataFrame, y_train) -> TabPFNClassifier:
    """Fit ('condition') a CPU TabPFNClassifier; default n_estimators (8).

    Pinned to the v2 checkpoint — the Nature-published model plan.md §9.3
    cites — because it is the only version with an ungated direct download;
    v2.5/v2.6/v3 weights require an interactive Prior Labs license login,
    which a non-interactive pipeline cannot perform.
    """
    if len(X_train) > MAX_TABPFN_TRAIN_ROWS:
        raise ValueError(
            f"TabPFN CPU path capped at {MAX_TABPFN_TRAIN_ROWS} training rows "
            f"(plan.md §9.3); got {len(X_train)}. Subsample first."
        )
    model = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2, device="cpu", random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_lgbm_small(X_train: pd.DataFrame, y_train) -> LGBMClassifier:
    """LightGBM comparator with small-sample-appropriate capacity (plan.md §12.9)."""
    model = LGBMClassifier(
        n_estimators=300,
        num_leaves=7,
        max_depth=3,
        learning_rate=0.05,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary",
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def _predict_pd(model, columns: list[str]):
    """Wrap predict_proba[:, 1] so shap can call the model on raw arrays."""

    def f(X) -> np.ndarray:
        return model.predict_proba(pd.DataFrame(np.asarray(X), columns=columns))[:, 1]

    return f


def _shap_values_permutation(model, X_background: pd.DataFrame, x_row: pd.DataFrame) -> np.ndarray:
    background = X_background.sample(
        n=min(BACKGROUND_ROWS, len(X_background)), random_state=42
    )
    explainer = shap.PermutationExplainer(
        _predict_pd(model, list(X_background.columns)), background
    )
    # Minimum evals for one full antithetic permutation over all features.
    explanation = explainer(x_row, max_evals=2 * x_row.shape[1] + 2, silent=True)
    return np.asarray(explanation.values).reshape(-1)


def _shap_values_extensions(model, X_background: pd.DataFrame, x_row: pd.DataFrame) -> np.ndarray:
    """tabpfn_extensions interpretability path (only when the package is installed)."""
    values = tabpfn_shap.get_shap_values(model, x_row, attribute_names=list(x_row.columns))
    values = np.asarray(values)
    if values.ndim == 3:  # (rows, features, classes) -> positive class
        values = values[..., -1]
    return values.reshape(-1)


def tabpfn_top_drivers(
    model,
    X_background: pd.DataFrame,
    x_row: pd.DataFrame,
    k: int = 5,
) -> list[dict]:
    """Top-k |SHAP| drivers for one thin-file loan, in explain.py:top_drivers schema.

    Returns [{feature, label, value, shap, direction}], sorted by |shap|.
    """
    x_row = x_row[list(X_background.columns)]
    if HAS_TABPFN_EXTENSIONS:
        try:
            sv = _shap_values_extensions(model, X_background, x_row)
        except Exception:
            sv = _shap_values_permutation(model, X_background, x_row)
    else:
        sv = _shap_values_permutation(model, X_background, x_row)

    rows = []
    for feat, value, shap_val in zip(x_row.columns, x_row.iloc[0], sv):
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
