"""Phase 7 uncertainty tests (plan.md §9.6): sane bands, tiny-n review flag, disagreement margin."""

import numpy as np
import pandas as pd
import pytest

from src.explainability.uncertainty import (
    MIN_COMPARABLE_N,
    confidence_band,
    fit_confidence_bands,
    models_disagree,
)
from src.models.calibration import GLOBAL_KEY


@pytest.fixture(scope="module")
def bands():
    rng = np.random.default_rng(42)
    n_big, n_tiny = 4000, 8
    pd_big = rng.uniform(0.005, 0.30, n_big)
    big = pd.DataFrame(
        {
            "loan_type_segment": "term_loan_proxy",
            "calibrated_pd_12m": pd_big,
            "event": rng.binomial(1, pd_big),  # well-calibrated by construction
            "duration": 12,
        }
    )
    pd_tiny = rng.uniform(0.05, 0.25, n_tiny)
    tiny = pd.DataFrame(
        {
            "loan_type_segment": "other_proxy",
            "calibrated_pd_12m": pd_tiny,
            "event": rng.binomial(1, pd_tiny),
            "duration": 12,
        }
    )
    return fit_confidence_bands(pd.concat([big, tiny], ignore_index=True)), n_big, n_tiny


def test_bands_are_monotone_sane(bands):
    fitted, n_big, n_tiny = bands
    assert GLOBAL_KEY in fitted and "term_loan_proxy" in fitted and "other_proxy" in fitted
    for rows in fitted.values():
        # Bins ordered, contiguous, covering [0, 1], with sane widths and counts.
        assert rows[0]["lo"] == 0.0 and rows[-1]["hi"] == 1.0
        for prev, cur in zip(rows, rows[1:]):
            assert prev["hi"] == cur["lo"]
        for r in rows:
            assert 0.0 < r["half_width"] <= 0.5
            assert r["n"] > 0
    assert sum(r["n"] for r in fitted["term_loan_proxy"]) == n_big
    assert sum(r["n"] for r in fitted["other_proxy"]) == n_tiny


def test_censored_loans_are_excluded_from_fitting():
    df = pd.DataFrame(
        {
            "loan_type_segment": "term_loan_proxy",
            "calibrated_pd_12m": np.linspace(0.01, 0.3, 400),
            "event": [0, 1] * 200,
            "duration": [3, 12] * 200,  # duration=3 & event=0 -> not evaluable
        }
    )
    fitted = fit_confidence_bands(df)
    # Only event==1 (200 rows) is evaluable: every event==0 row is censored at
    # duration=3, before the 12-month horizon.
    assert sum(r["n"] for r in fitted[GLOBAL_KEY]) == 200


def test_large_segment_band_is_tight_and_not_flagged(bands):
    fitted, _, _ = bands
    band = confidence_band(fitted, "term_loan_proxy", 0.10)
    assert band["n_comparable"] >= MIN_COMPARABLE_N
    assert band["half_width"] <= 0.10
    assert band["wide_band_flag"] is False


def test_tiny_n_segment_triggers_mandatory_review_flag(bands):
    fitted, _, n_tiny = bands
    band = confidence_band(fitted, "other_proxy", 0.10)
    assert band["n_comparable"] == n_tiny
    assert band["wide_band_flag"] is True  # n < MIN_COMPARABLE_N -> human review


def test_unknown_segment_falls_back_to_global(bands):
    fitted, _, _ = bands
    band = confidence_band(fitted, "never_seen_segment", 0.10)
    assert band["n_comparable"] > 0


def test_out_of_range_pd_maps_to_nearest_bin(bands):
    fitted, _, _ = bands
    band = confidence_band(fitted, "term_loan_proxy", 0.999)
    assert band["n_comparable"] > 0


def test_models_disagree_boundary():
    assert models_disagree(0.30, 0.50) is True
    assert models_disagree(0.30, 0.44) is False
    assert models_disagree(0.30, 0.45) is False  # exactly at margin: not a disagreement
    assert models_disagree(0.30, 0.4501) is True
    assert models_disagree(0.50, 0.30, margin=0.10) is True  # symmetric, custom margin
