"""Phase 5 graph-lite feature tests (no model downloads, small synthetic graph)."""

import numpy as np
import pandas as pd
import pytest

from src.features.graph import (
    GraphSeparabilityError,
    build_graph_features,
    generate_counterparty_graph,
    networkx_cross_check,
    validate_separability,
)


def toy_features(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(100_000, 100_000 + n),
            "late_installment_rate": rng.uniform(0.0, 0.5, n),
            "bureau_overdue_flag": rng.integers(0, 2, n).astype(float),
        }
    )


def test_separability_validation_passes_on_small_graph():
    df = toy_features()
    edges, profile = generate_counterparty_graph(df, seed=42, n_counterparties=200)
    feats = build_graph_features(edges, profile, seed=42)
    result = validate_separability(feats, profile)
    assert result["auc_concentration_alone"] >= 0.8
    assert result["mean_concentrated"] > result["mean_diversified"]
    assert result["cohens_d"] > 0
    assert result["n_concentrated"] > 0 and result["n_diversified"] > 0


def test_separability_raises_when_absent():
    df = toy_features(n=200)
    edges, profile = generate_counterparty_graph(df, seed=42, n_counterparties=100)
    feats = build_graph_features(edges, profile, seed=42)
    # Shuffle labels so the concentration feature can no longer separate groups.
    broken = profile.copy()
    broken["concentrated_flag"] = np.resize([0, 1], len(broken))
    with pytest.raises(GraphSeparabilityError):
        validate_separability(feats, broken, min_auc=0.8)


def test_feature_ranges_and_provenance():
    df = toy_features()
    edges, profile = generate_counterparty_graph(df, seed=42, n_counterparties=200)
    feats = build_graph_features(edges, profile, seed=42)

    assert set(feats["SK_ID_CURR"]) == set(df["SK_ID_CURR"])
    assert feats["counterparty_concentration"].between(0.0, 1.0).all()
    assert feats["degree_centrality"].between(0.0, 1.0).all()
    assert feats["anchor_linkage_flag"].isin([0, 1]).all()
    assert feats["network_churn"].between(0.0, 1.0).all()
    assert (feats["data_provenance"] == "synthetic_graph").all()
    assert (profile["data_provenance"] == "synthetic_graph").all()
    # Concentrated firms have 1-2 counterparties -> concentration >= 0.5.
    merged = feats.merge(profile[["SK_ID_CURR", "concentrated_flag"]], on="SK_ID_CURR")
    assert (merged.loc[merged["concentrated_flag"] == 1, "counterparty_concentration"] >= 0.5).all()


def test_generation_deterministic_under_fixed_seed():
    df = toy_features()
    edges_a, profile_a = generate_counterparty_graph(df, seed=42, n_counterparties=200)
    edges_b, profile_b = generate_counterparty_graph(df, seed=42, n_counterparties=200)
    pd.testing.assert_frame_equal(edges_a, edges_b)
    pd.testing.assert_frame_equal(profile_a, profile_b)

    feats_a = build_graph_features(edges_a, profile_a, seed=42)
    feats_b = build_graph_features(edges_b, profile_b, seed=42)
    pd.testing.assert_frame_equal(feats_a, feats_b)

    edges_c, _ = generate_counterparty_graph(df, seed=7, n_counterparties=200)
    assert not edges_a.equals(edges_c)


def test_networkx_cross_check_agrees_with_vectorized_degrees():
    df = toy_features(n=300)
    edges, _ = generate_counterparty_graph(df, seed=42, n_counterparties=150)
    result = networkx_cross_check(edges, n_firms=100, seed=42)
    assert result["max_abs_centrality_diff"] <= 1e-12
    assert result["firms_checked"] == 100
