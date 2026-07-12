"""Phase 4: per-segment isotonic calibration + the shared risk-grade scale (plan.md §9.4, §9.7).

This layer is the literal implementation of the "common interpretation
framework": every segment model's raw score is mapped onto one calibrated PD
scale, then onto one A-G grade ladder, so a grade means the same real-world
likelihood regardless of which segment produced it.
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

GLOBAL_KEY = "__global__"

# Calibrated 12-month PD -> unified grade (upper bound, grade).
GRADE_BANDS = [
    (0.01, "A"),
    (0.02, "B"),
    (0.04, "C"),
    (0.07, "D"),
    (0.12, "E"),
    (0.20, "F"),
    (1.01, "G"),
]

# §9.7 risk-band-to-action mapping, keyed off the unified grade.
ACTION_BY_GRADE = {
    "A": "Normal monitoring cadence",
    "B": "Normal monitoring cadence",
    "C": "Monthly review; enter Watch List",
    "D": "Monthly review; enter Watch List",
    "E": "Enhanced monitoring; proactive restructuring offer",
    "F": "Enhanced monitoring; proactive restructuring offer",
    "G": "Immediate intervention; exposure-limitation protocol",
}


def pd_to_grade(p: float) -> str:
    for upper, grade in GRADE_BANDS:
        if p < upper:
            return grade
    return "G"


def fit_calibrators(
    calib_df: pd.DataFrame,
    segment_col: str = "loan_type_segment",
    score_col: str = "hazard_cum_pd_12m",
    label_col: str = "event",
) -> dict[str, IsotonicRegression]:
    """Fit one isotonic regressor per segment plus a global fallback.

    Fit only on loans with a determinable 12-month outcome (event, or observed
    a full 12 months) — early-censored loans have no binary label to calibrate to.
    """
    calibrators: dict[str, IsotonicRegression] = {}
    grouped = [(GLOBAL_KEY, calib_df)] + list(calib_df.groupby(segment_col))
    for key, seg_df in grouped:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(seg_df[score_col], seg_df[label_col])
        calibrators[key] = iso
    return calibrators


def apply_calibration(
    calibrators: dict[str, IsotonicRegression],
    segments: pd.Series,
    raw_scores: pd.Series,
) -> np.ndarray:
    out = np.empty(len(raw_scores))
    for key, idx in segments.groupby(segments).groups.items():
        iso = calibrators.get(key, calibrators[GLOBAL_KEY])
        out[segments.index.get_indexer(idx)] = iso.predict(raw_scores.loc[idx])
    return out


def reliability_table(y_true, y_pred, n_bins: int = 10) -> pd.DataFrame:
    """Quantile-binned predicted vs observed rates — the §9.4 reliability diagram data."""
    df = pd.DataFrame({"pred": y_pred, "obs": y_true})
    df["bin"] = pd.qcut(df["pred"], q=n_bins, duplicates="drop")
    return (
        df.groupby("bin", observed=True)
        .agg(mean_predicted=("pred", "mean"), observed_rate=("obs", "mean"), n=("obs", "size"))
        .reset_index(drop=True)
    )
