"""Shared metrics harness (plan.md §2.1, §15.1, §15.3).

Every model in the project reports through this module so the naive-baseline
comparison is always present and always computed the same way.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)


def ks_statistic(y_true, y_score) -> float:
    """Max separation between cumulative defaulter / non-defaulter score distributions."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(np.abs(tpr - fpr)))


def recall_at_fpr(y_true, y_score, max_fpr: float = 0.10) -> float:
    """Of true events, the fraction caught while capping false alarms at max_fpr."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(tpr[fpr <= max_fpr].max()) if (fpr <= max_fpr).any() else 0.0


def metrics_row(name: str, y_true, y_score) -> dict:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    return {
        "model": name,
        "auc_roc": round(roc_auc_score(y_true, y_score), 4),
        "ks": round(ks_statistic(y_true, y_score), 4),
        "pr_auc": round(average_precision_score(y_true, y_score), 4),
        "recall_at_fpr10": round(recall_at_fpr(y_true, y_score), 4),
        "brier": round(brier_score_loss(y_true, np.clip(y_score, 0, 1)), 4),
    }


def naive_row(y_true) -> dict:
    """The 'always predict no-default' baseline (plan.md §15.3), computed analytically.

    A constant score has no ranking power (AUC 0.5, KS 0, catches nothing at any
    capped FPR) and its Brier score equals the event rate. Its raw 'accuracy' is
    1 - event_rate — shown only to expose why accuracy is the wrong metric.
    """
    y_true = np.asarray(y_true)
    event_rate = float(y_true.mean())
    return {
        "model": "naive_always_no_default",
        "auc_roc": 0.5,
        "ks": 0.0,
        "pr_auc": round(event_rate, 4),  # AP of a random/constant ranker = prevalence
        "recall_at_fpr10": 0.0,
        "brier": round(event_rate, 4),
        "accuracy_for_contrast_only": round(1 - event_rate, 4),
    }


def metrics_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index("model")
