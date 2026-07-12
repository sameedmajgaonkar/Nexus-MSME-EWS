"""Evidently drift monitoring + PSI retrain trigger (plan.md §13.3, §15.1, §13.2).

evidently 0.7.x API: ``Report([DataDriftPreset(method="psi")]).run(current, reference)``
returns a Snapshot with ``save_html()`` / ``dict()``; the preset exposes a
per-column PSI via its ValueDrift metrics. We ALSO compute PSI ourselves with
the standard binned formula (plan.md §15.1) so the retrain trigger is exact,
unit-testable, and independent of preset defaults — the Evidently snapshot is
kept as the full visual report (reports/drift_report.html).

reports/drift_summary.json is read VERBATIM by the serving API
(GET /api/drift/report) — the field names here are a contract:
{generated_at, psi_by_feature, drifted_features, calibration_status,
 plain_language, recommendation, retrain_triggered}.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.models.hazard import STATIC_FEATURES

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"

# Tier-1 = the 11 core underwriting features of the committed hazard model.
TIER1_FEATURES = list(STATIC_FEATURES)
PSI_RETRAIN_THRESHOLD = 0.2  # plan.md §13.2 out-of-cycle retrain trigger
SCORE_PSI_TOLERANCE = 0.1  # calibration proxy: model-score distribution stability
N_BINS = 10
_EPS = 1e-4


def psi(expected, actual, n_bins: int = N_BINS) -> float:
    """Population Stability Index, standard binned formula (plan.md §15.1):

        PSI = sum over bins of (actual% - expected%) * ln(actual% / expected%)

    Bins are expected-side deciles; discrete columns (<= n_bins unique values,
    e.g. binary flags) use the values themselves as bins. NaNs are dropped;
    zero proportions are clipped at 1e-4.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    unique_expected = np.unique(expected)
    if len(unique_expected) <= n_bins:
        categories = np.unique(np.concatenate([unique_expected, np.unique(actual)]))
        expected_pct = np.array([(expected == c).mean() for c in categories])
        actual_pct = np.array([(actual == c).mean() for c in categories])
    else:
        breaks = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
        breaks[0], breaks[-1] = -np.inf, np.inf
        expected_pct = np.histogram(expected, bins=breaks)[0] / len(expected)
        actual_pct = np.histogram(actual, bins=breaks)[0] / len(actual)

    expected_pct = np.clip(expected_pct, _EPS, None)
    actual_pct = np.clip(actual_pct, _EPS, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def _plain_language(
    psi_by_feature: dict, tier1_drifted: list[str], calibration_status: str
) -> tuple[str, str]:
    """(plain_language, recommendation) — risk-officer wording, not a stat dump (§13.3)."""
    worst_feature = max(psi_by_feature, key=psi_by_feature.get)
    worst_psi = psi_by_feature[worst_feature]
    n_tier1_checked = sum(1 for f in psi_by_feature if f in TIER1_FEATURES)

    if tier1_drifted:
        plain = (
            f"{len(tier1_drifted)} of {n_tier1_checked} core underwriting features have "
            f"shifted materially since the model was trained — the worst is "
            f"'{worst_feature}' (PSI {worst_psi:.2f}, above the {PSI_RETRAIN_THRESHOLD} "
            f"retrain threshold). Scores for borrowers affected by this shift may no "
            f"longer be reliable, so an out-of-cycle retrain has been triggered "
            f"automatically. The retrained model will only replace the current one if "
            f"it wins on the held-out comparison window."
        )
        recommendation = (
            f"Out-of-cycle retrain triggered: PSI > {PSI_RETRAIN_THRESHOLD} on Tier-1 "
            f"feature(s) {tier1_drifted}. The challenger is gated against the current "
            f"Production model on the OOT test window before any promotion."
        )
    else:
        plain = (
            f"No material population shift detected: the worst feature-level PSI is "
            f"{worst_psi:.2f} on '{worst_feature}', below the {PSI_RETRAIN_THRESHOLD} "
            f"retrain threshold. Borrowers being scored in the current window still "
            f"look like the population the model was trained on."
        )
        if calibration_status == "score_distribution_shifted":
            plain += (
                " The model's score distribution has shifted, however — a "
                "recalibration check is recommended."
            )
            recommendation = (
                "No Tier-1 feature drift; score-distribution shift detected — "
                "recheck calibration (Phase 4 isotonic recalibration) before the "
                "next scheduled cycle."
            )
        else:
            plain += " No action needed beyond routine monthly monitoring."
            recommendation = (
                "Continue routine monitoring; next retrain check runs on the "
                "scheduled monthly cycle."
            )
    return plain, recommendation


def run_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str],
    prediction_col: str | None = None,
    reports_dir: str | Path = REPORTS_DIR,
) -> dict:
    """Compare current vs reference windows; write drift_report.html + drift_summary.json.

    Returns the summary dict (same content as reports/drift_summary.json).
    Retrain rule (plan.md §13.2): PSI > 0.2 on ANY Tier-1 feature sets
    retrain_triggered=True. ``prediction_col``, if present in both frames,
    drives calibration_status via score-distribution PSI — a stability proxy;
    the full reliability-curve check lives in Phase 4's calibration module.
    """
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    report = Report(
        [DataDriftPreset(columns=list(features), method="psi",
                         threshold=PSI_RETRAIN_THRESHOLD)]
    )
    snapshot = report.run(current_df[list(features)], reference_df[list(features)])
    snapshot.save_html(str(reports_dir / "drift_report.html"))

    psi_by_feature = {
        f: round(psi(reference_df[f], current_df[f]), 4) for f in features
    }
    drifted_features = [
        f for f, v in psi_by_feature.items() if v > PSI_RETRAIN_THRESHOLD
    ]
    tier1_drifted = [f for f in drifted_features if f in TIER1_FEATURES]
    retrain_triggered = len(tier1_drifted) > 0

    calibration_status = "not_evaluated"
    score_psi = None
    if (
        prediction_col
        and prediction_col in reference_df.columns
        and prediction_col in current_df.columns
    ):
        score_psi = round(psi(reference_df[prediction_col], current_df[prediction_col]), 4)
        calibration_status = (
            "stable" if score_psi < SCORE_PSI_TOLERANCE else "score_distribution_shifted"
        )

    plain_language, recommendation = _plain_language(
        psi_by_feature, tier1_drifted, calibration_status
    )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "psi_by_feature": psi_by_feature,
        "drifted_features": drifted_features,
        "calibration_status": calibration_status,
        "score_psi": score_psi,
        "plain_language": plain_language,
        "recommendation": recommendation,
        "retrain_triggered": retrain_triggered,
    }
    (reports_dir / "drift_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
