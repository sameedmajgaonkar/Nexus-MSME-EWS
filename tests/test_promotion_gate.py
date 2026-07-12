"""Phase 10 promotion-gate tests (plan.md §13.2): never auto-promote a worse model.

Production metrics are monkeypatched (no real fused.joblib needed) and MLflow's
file store is pointed at tmp_path, so no server and no repo-side mlruns writes.
"""

import numpy as np
import pandas as pd
import pytest

from src.mlops import retrain, tracking
from src.mlops.retrain import FUSED_FEATURES, retrain_and_gate
from src.models.splits import pseudo_oot_split

FAKE_PROD_METRICS = {"auc_roc": None}  # value set per-test


def _synthetic_loans(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Tiny loan table with every fused-recipe column and a learnable signal."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.normal(0, 1, n) for f in FUSED_FEATURES})
    df["SK_ID_CURR"] = np.arange(100000, 100000 + n)
    df["loan_type_segment"] = rng.choice(["term_loan_proxy", "working_capital_proxy"], n)
    df["sector_segment"] = rng.choice(["Manufacturing", "Retail_Trade"], n)
    df["data_richness"] = rng.choice(["established", "ntc_ntb"], n)
    logits = -1.0 + 1.5 * df["credit_income_ratio"] - 1.0 * df["ext_source_2"]
    df["TARGET"] = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(int)
    return df


@pytest.fixture()
def gate_env(monkeypatch, tmp_path):
    """Isolated MLflow file store + patched production evaluation."""
    monkeypatch.setattr(tracking, "TRACKING_URI", (tmp_path / "mlruns").as_uri())
    monkeypatch.setattr(
        retrain,
        "evaluate_production",
        lambda data, split, models_dir=None: dict(FAKE_PROD_METRICS),
    )
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    # sentinel bytes: retrain_and_gate must not touch this file on a HELD decision
    (models_dir / "fused.joblib").write_bytes(b"sentinel-production-artifact")
    return models_dir


def test_gate_refuses_worse_challenger(gate_env):
    FAKE_PROD_METRICS["auc_roc"] = 0.99  # unbeatable production model
    df = _synthetic_loans()
    result = retrain_and_gate(
        df, pseudo_oot_split(df),
        challenger_params={"n_estimators": 20},
        models_dir=gate_env,
    )
    assert result["promoted"] is False
    assert "HELD" in result["reason"]
    assert "registered_version" not in result
    # production artifact untouched
    assert (gate_env / "fused.joblib").read_bytes() == b"sentinel-production-artifact"


def test_gate_accepts_better_challenger(gate_env):
    FAKE_PROD_METRICS["auc_roc"] = 0.50  # production model no better than chance
    df = _synthetic_loans()
    result = retrain_and_gate(
        df, pseudo_oot_split(df),
        challenger_params={"n_estimators": 20},
        models_dir=gate_env,
    )
    assert result["promoted"] is True
    assert result["challenger_metrics"]["auc_roc"] >= 0.50
    assert result["registered_version"] >= 1
    # artifact replaced with a real model bundle
    import joblib  # own artifact written seconds ago — trusted pickle

    bundle = joblib.load(gate_env / "fused.joblib")
    assert set(bundle) == {"model", "columns"}
    # registry alias 'production' now points at the challenger's run metrics
    prod = tracking.get_production_metrics("fused")
    assert prod is not None
    assert prod["auc_roc"] == pytest.approx(result["challenger_metrics"]["auc_roc"])
