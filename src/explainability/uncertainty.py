"""Phase 7 lightweight uncertainty quantification (plan.md §9.6).

Calibration/test-set residuals attach a "± X" confidence band to every
prediction — the §9.6 phrasing: "PD 42% ± 6%, calibrated on n=380 comparable
accounts". Bands are fitted per segment in calibrated-PD bins; the half-width
combines the bin's residual bias (observed rate minus mean predicted PD) with
a 95% sampling term that widens on small n. Wide bands, tiny comparable
samples, or TabPFN-vs-hazard disagreement beyond a margin all flag the case
for MANDATORY human review. Full conformal prediction stays Tier 3 (§12.10).
"""

import math

import pandas as pd

from src.models.calibration import GLOBAL_KEY
from src.models.hazard import MAX_MONTHS

WIDE_BAND_HALF_WIDTH = 0.10  # half-width above this -> mandatory human review
MIN_COMPARABLE_N = 50  # fewer comparable accounts than this -> mandatory human review
DISAGREEMENT_MARGIN = 0.15  # TabPFN vs hazard PD gap beyond this -> mandatory human review
Z_95 = 1.96
DEFAULT_BINS = 8
MIN_SEGMENT_ROWS_FOR_BINS = 200  # below this, one pooled (wide) bin for the whole segment


def _band_rows(seg_df: pd.DataFrame, pd_col: str, label_col: str, n_bins: int) -> list[dict]:
    if len(seg_df) < MIN_SEGMENT_ROWS_FOR_BINS:
        n_bins = 1
    binned = pd.qcut(seg_df[pd_col], q=n_bins, duplicates="drop")
    rows = []
    for interval, grp in seg_df.groupby(binned, observed=True):
        n = len(grp)
        mean_pd = float(grp[pd_col].mean())
        obs = float(grp[label_col].mean())
        # Laplace-smoothed rate so a zero-event bin still carries sampling width.
        p_hat = (obs * n + 1.0) / (n + 2.0)
        half_width = abs(obs - mean_pd) + Z_95 * math.sqrt(p_hat * (1.0 - p_hat) / n)
        rows.append(
            {
                "lo": float(interval.left),
                "hi": float(interval.right),
                "half_width": round(half_width, 5),
                "n": int(n),
            }
        )
    rows.sort(key=lambda r: r["lo"])
    rows[0]["lo"] = 0.0
    rows[-1]["hi"] = 1.0
    return rows


def fit_confidence_bands(
    scored_df: pd.DataFrame,
    segment_col: str = "loan_type_segment",
    pd_col: str = "calibrated_pd_12m",
    label_col: str = "event",
    n_bins: int = DEFAULT_BINS,
) -> dict[str, list[dict]]:
    """Fit per-segment residual-spread bands from a phase4_scored_test-shaped frame.

    Only loans with a determinable 12-month outcome (event, or observed the
    full horizon) contribute residuals — same evaluability rule as calibration.
    """
    df = scored_df
    if "duration" in df.columns:
        df = df[(df[label_col] == 1) | (df["duration"] >= MAX_MONTHS)]
    bands = {GLOBAL_KEY: _band_rows(df, pd_col, label_col, n_bins)}
    for segment, seg_df in df.groupby(segment_col, observed=True):
        bands[segment] = _band_rows(seg_df, pd_col, label_col, n_bins)
    return bands


def confidence_band(bands: dict[str, list[dict]], segment: str, pd_value: float) -> dict:
    """± band for one prediction: {half_width, n_comparable, wide_band_flag}.

    wide_band_flag=True mandates human review (§9.6): the band is wide or the
    bin was calibrated on too few comparable accounts."""
    seg_bands = bands.get(segment, bands[GLOBAL_KEY])
    row = next(
        (r for r in seg_bands if r["lo"] <= pd_value <= r["hi"]),
        min(seg_bands, key=lambda r: abs((r["lo"] + r["hi"]) / 2.0 - pd_value)),
    )
    return {
        "half_width": row["half_width"],
        "n_comparable": row["n"],
        "wide_band_flag": bool(row["half_width"] > WIDE_BAND_HALF_WIDTH or row["n"] < MIN_COMPARABLE_N),
    }


def models_disagree(pd_a: float, pd_b: float, margin: float = DISAGREEMENT_MARGIN) -> bool:
    """TabPFN-vs-hazard disagreement flag (§9.6) — only meaningful where both
    models scored the same case (e.g. the Phase 6 validation comparison).
    A gap exactly at the margin is not a disagreement (float-safe)."""
    return bool(abs(float(pd_a) - float(pd_b)) - margin > 1e-12)
