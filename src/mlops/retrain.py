"""Retrain-and-gate loop — the plan.md §13.2 flowchart.

Challenger recipe mirrors scripts/run_phase5.py's fused model exactly:
LightGBM on structured + graph + text features, loan-level TARGET outcome
(raw monthly data unavailable — BUILD_CONTEXT constraint), pseudo-OOT split,
seed 42. The gate compares challenger vs the CURRENT Production artifact
(models/fused.joblib) on the SAME OOT test window via the shared metrics
harness (src/models/evaluate.py) and NEVER auto-promotes a worse model.
"""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping

from src.mlops import tracking
from src.models.evaluate import metrics_row
from src.models.hazard import SEGMENT_COLS

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
SEED = 42
FUSED_MODEL_NAME = "fused"

# Same feature recipe as scripts/run_phase5.py (fused arm).
STRUCTURED_FEATURES = [
    "credit_income_ratio",
    "annuity_income_ratio",
    "ext_source_1",
    "ext_source_2",
    "ext_source_3",
    "age_years",
    "employed_years",
    "bureau_active_count",
    "bureau_overdue_flag",
    "n_installments",
    "late_installment_rate",
    "max_days_late",
    "avg_days_late",
    "prior_late_rate",
    "prior_n_installments",
]
GRAPH_FEATURES = [
    "counterparty_concentration",
    "degree_centrality",
    "anchor_linkage_flag",
    "network_churn",
]
TEXT_FEATURES = [f"text_pc_{i}" for i in range(1, 13)] + [
    "sentiment_signed",
    "distress_keyword_flag",
]
FUSED_FEATURES = STRUCTURED_FEATURES + GRAPH_FEATURES + TEXT_FEATURES

DEFAULT_PARAMS = dict(
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=1000,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    random_state=SEED,
    verbose=-1,
)


def loan_matrix(df: pd.DataFrame, feature_cols: list[str] = FUSED_FEATURES) -> pd.DataFrame:
    """Numeric features + one-hot segment dummies (identical to run_phase5)."""
    X = df[feature_cols].copy()
    for col in SEGMENT_COLS:
        X = pd.concat([X, pd.get_dummies(df[col], prefix=col, dtype=int)], axis=1)
    return X


def train_challenger(
    data: pd.DataFrame,
    split: pd.Series,
    params_override: dict | None = None,
    sample_frac: float | None = None,
    seed: int = SEED,
) -> tuple[LGBMClassifier, list[str]]:
    """Train a challenger on the train window (optionally subsampled for runtime)."""
    X = loan_matrix(data)
    y = data["TARGET"]
    train_idx = data.index[(split == "train").to_numpy()]
    if sample_frac is not None and sample_frac < 1.0:
        rng = np.random.default_rng(seed)
        train_idx = train_idx[rng.random(len(train_idx)) < sample_frac]
    calib_mask = (split == "calib").to_numpy()

    y_train = y.loc[train_idx]
    params = dict(DEFAULT_PARAMS)
    params["scale_pos_weight"] = float(
        (y_train == 0).sum() / max(int((y_train == 1).sum()), 1)
    )
    if params_override:
        params.update(params_override)

    model = LGBMClassifier(**params)
    model.fit(
        X.loc[train_idx],
        y_train,
        eval_set=[(X[calib_mask], y[calib_mask])],
        eval_metric="auc",
        callbacks=[early_stopping(50, verbose=False)],
    )
    return model, list(X.columns)


def evaluate_production(
    data: pd.DataFrame, split: pd.Series, models_dir: str | Path = MODELS_DIR
) -> dict:
    """Score the CURRENT Production artifact (models/fused.joblib) on the OOT test window."""
    # joblib.load is safe here: the artifact is produced by this project's own
    # training scripts (run_phase5 / this module), never from an external source.
    bundle = joblib.load(Path(models_dir) / "fused.joblib")
    test_mask = (split == "test").to_numpy()
    X_test = loan_matrix(data[test_mask]).reindex(columns=bundle["columns"], fill_value=0)
    scores = bundle["model"].predict_proba(X_test)[:, 1]
    row = metrics_row("production_fused", data["TARGET"][test_mask], scores)
    row.pop("model")
    return row


def retrain_and_gate(
    data: pd.DataFrame,
    split: pd.Series,
    challenger_params: dict | None = None,
    sample_frac: float | None = None,
    models_dir: str | Path = MODELS_DIR,
    model_name: str = FUSED_MODEL_NAME,
) -> dict:
    """§13.2 flowchart: retrain -> compare vs Production on the SAME OOT test
    window -> promote ONLY if challenger AUC >= Production AUC, else hold.

    Promotion = MLflow registry alias move to 'production' + overwrite of
    models/fused.joblib. A hold logs the run with the reason and changes nothing.
    """
    models_dir = Path(models_dir)
    t0 = time.time()
    model, columns = train_challenger(data, split, challenger_params, sample_frac)
    train_seconds = round(time.time() - t0, 1)

    test_mask = (split == "test").to_numpy()
    X_test = loan_matrix(data[test_mask]).reindex(columns=columns, fill_value=0)
    scores = model.predict_proba(X_test)[:, 1]
    challenger = metrics_row("challenger_fused", data["TARGET"][test_mask], scores)
    challenger.pop("model")

    production = evaluate_production(data, split, models_dir=models_dir)

    promoted = challenger["auc_roc"] >= production["auc_roc"]
    if promoted:
        reason = (
            f"PROMOTED: challenger AUC {challenger['auc_roc']:.4f} >= production AUC "
            f"{production['auc_roc']:.4f} on the same OOT test window."
        )
        joblib.dump({"model": model, "columns": columns}, models_dir / "fused.joblib")
    else:
        reason = (
            f"HELD: challenger AUC {challenger['auc_roc']:.4f} < production AUC "
            f"{production['auc_roc']:.4f} on the same OOT test window — a worse model "
            f"is never auto-promoted (plan.md §13.2); flagged for manual review."
        )

    logged_params = {**DEFAULT_PARAMS, **(challenger_params or {})}
    logged_params["sample_frac"] = 1.0 if sample_frac is None else sample_frac
    run_id = tracking.log_model_run(
        "retrain_challenger_fused",
        params=logged_params,
        metrics=challenger,
        artifact_path=(models_dir / "fused.joblib") if promoted else None,
        tags={
            "phase": "10",
            "gate_decision": "promoted" if promoted else "held",
            "gate_reason": reason,
        },
    )

    result = {
        "promoted": promoted,
        "reason": reason,
        "run_id": run_id,
        "challenger_metrics": challenger,
        "production_metrics": production,
        "train_seconds": train_seconds,
    }
    if promoted:
        result["registered_version"] = tracking.register_and_stage(
            run_id, model_name, "production"
        )
    return result
