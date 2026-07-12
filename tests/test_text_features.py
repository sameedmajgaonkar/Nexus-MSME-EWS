"""Phase 5 text-pipeline tests — non-network parts only (no model downloads;
MiniLM/FinBERT are exercised by scripts/run_phase5.py instead)."""

import numpy as np
import pandas as pd

from src.features.text import (
    DISTRESS_TEMPLATES,
    NEUTRAL_TEMPLATES,
    POSITIVE_TEMPLATES,
    TEMPLATE_POOLS,
    distress_keyword_flag,
    generate_officer_notes,
    signed_sentiment,
    unique_note_index,
)


def toy_features(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(200_000, 200_000 + n),
            "late_installment_rate": rng.uniform(0.0, 0.6, n),
            "bureau_overdue_flag": rng.integers(0, 2, n).astype(float),
        }
    )


def test_distress_flag_fires_on_distress_not_on_positive():
    notes = pd.Series(
        [
            "Payment stoppage reported at the unit following a customer dispute.",
            "Repayments regular for the past year; account conduct satisfactory.",
        ]
    )
    assert distress_keyword_flag(notes).tolist() == [1, 0]


def test_distress_flag_covers_all_templates_and_only_distress_ones():
    assert distress_keyword_flag(pd.Series(DISTRESS_TEMPLATES)).eq(1).all()
    assert distress_keyword_flag(pd.Series(POSITIVE_TEMPLATES)).eq(0).all()
    assert distress_keyword_flag(pd.Series(NEUTRAL_TEMPLATES)).eq(0).all()


def test_template_pool_is_finite_and_in_spec_range():
    n_unique = len(set(POSITIVE_TEMPLATES + NEUTRAL_TEMPLATES + DISTRESS_TEMPLATES))
    assert n_unique == len(POSITIVE_TEMPLATES) + len(NEUTRAL_TEMPLATES) + len(DISTRESS_TEMPLATES)
    assert 60 <= n_unique <= 120


def test_note_generation_deterministic_and_from_pool():
    df = toy_features()
    notes_a = generate_officer_notes(df, seed=42)
    notes_b = generate_officer_notes(df, seed=42)
    pd.testing.assert_frame_equal(notes_a, notes_b)
    assert not notes_a["officer_note"].equals(generate_officer_notes(df, seed=7)["officer_note"])

    all_templates = set(POSITIVE_TEMPLATES) | set(NEUTRAL_TEMPLATES) | set(DISTRESS_TEMPLATES)
    assert set(notes_a["officer_note"]).issubset(all_templates)
    assert (notes_a["data_provenance"] == "synthetic_text").all()
    # note_tone matches the pool the sentence came from
    for tone, pool in TEMPLATE_POOLS.items():
        mask = notes_a["note_tone"] == tone
        assert set(notes_a.loc[mask, "officer_note"]).issubset(set(pool))


def test_note_tone_conditions_on_observable_behavior():
    n = 2000
    df = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(n),
            "late_installment_rate": np.concatenate([np.zeros(n // 2), np.full(n // 2, 0.6)]),
            "bureau_overdue_flag": np.concatenate([np.zeros(n // 2), np.ones(n // 2)]),
        }
    )
    notes = generate_officer_notes(df, seed=42)
    distress_rate_clean = (notes["note_tone"][: n // 2] == "distress").mean()
    distress_rate_stressed = (notes["note_tone"][n // 2 :] == "distress").mean()
    assert distress_rate_stressed > distress_rate_clean + 0.3


def test_unique_note_index_roundtrip():
    notes = pd.Series(["b note", "a note", "b note", "c note", "a note"])
    uniques, codes = unique_note_index(notes)
    assert sorted(uniques) == uniques  # categorical order, deterministic
    assert len(uniques) == 3
    reconstructed = [uniques[c] for c in codes]
    assert reconstructed == notes.tolist()


def test_signed_sentiment_encoding():
    labels = pd.Series(["positive", "negative", "neutral"])
    scores = pd.Series([0.9, 0.8, 0.99])
    assert signed_sentiment(labels, scores).tolist() == [0.9, -0.8, 0.0]
