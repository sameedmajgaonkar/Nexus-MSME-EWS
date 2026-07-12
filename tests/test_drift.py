"""Phase 10 drift tests: PSI formula behavior + drift_summary.json contract.

The summary schema is load-bearing — the serving API returns
reports/drift_summary.json verbatim from GET /api/drift/report.
"""

import json

import numpy as np
import pandas as pd

from src.mlops.drift import PSI_RETRAIN_THRESHOLD, run_drift_report, psi

SUMMARY_FIELDS = [
    "generated_at",
    "psi_by_feature",
    "drifted_features",
    "calibration_status",
    "plain_language",
    "recommendation",
    "retrain_triggered",
]


def test_psi_identical_distribution_is_near_zero():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 5000)
    assert psi(x, x) < 1e-9  # same sample: exactly stable
    y = rng.normal(0, 1, 5000)  # fresh draw from the same distribution
    assert psi(x, y) < 0.05


def test_psi_shifted_distribution_crosses_threshold():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 5000)
    y = rng.normal(1.0, 1, 5000)  # one-sigma mean shift
    assert psi(x, y) > PSI_RETRAIN_THRESHOLD


def test_psi_handles_discrete_and_nan_columns():
    rng = np.random.default_rng(42)
    # binary flag: 10% -> 50% incidence must register as large drift
    flags_ref = (rng.random(4000) < 0.10).astype(float)
    flags_cur = (rng.random(4000) < 0.50).astype(float)
    assert psi(flags_ref, flags_cur) > PSI_RETRAIN_THRESHOLD
    # NaNs are dropped, not propagated
    x = rng.normal(0, 1, 4000)
    x_nan = x.copy()
    x_nan[:1000] = np.nan
    assert np.isfinite(psi(x, x_nan))
    assert psi(x, x_nan) < 0.05


def _windows(shift: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    n = 1500
    ref = pd.DataFrame({
        "credit_income_ratio": rng.lognormal(1.0, 0.4, n),  # Tier-1 feature
        "age_years": rng.normal(40, 10, n),                 # Tier-1 feature
    })
    cur = pd.DataFrame({
        "credit_income_ratio": rng.lognormal(1.0, 0.4, n) * shift,
        "age_years": rng.normal(40, 10, n),
    })
    return ref, cur


def test_drift_summary_schema_and_retrain_flag(tmp_path):
    ref, cur = _windows(shift=1.6)
    summary = run_drift_report(ref, cur, list(ref.columns), reports_dir=tmp_path)

    for field in SUMMARY_FIELDS:
        assert field in summary, f"missing contract field '{field}'"
    assert (tmp_path / "drift_report.html").exists()
    on_disk = json.loads((tmp_path / "drift_summary.json").read_text())
    assert on_disk == summary

    assert "credit_income_ratio" in summary["drifted_features"]
    assert summary["retrain_triggered"] is True
    assert isinstance(summary["plain_language"], str) and summary["plain_language"]


def test_drift_summary_quiet_when_no_shift(tmp_path):
    ref, cur = _windows(shift=1.0)
    summary = run_drift_report(ref, cur, list(ref.columns), reports_dir=tmp_path)
    assert summary["drifted_features"] == []
    assert summary["retrain_triggered"] is False
    assert summary["calibration_status"] == "not_evaluated"  # no prediction col given
