"""Phase 5: graph-lite + text enrichment, separability validation, honest ablation.

Pipeline (PHASES.md Phase 5, plan.md §8.2, §8.3, §9.5, §12.1):
  1. Load serving features, run the data-quality gate.
  2. Synthetic counterparty graph -> graph-lite features -> NON-OPTIONAL
     separability validation (concentrated vs diversified firms).
  3. Synthetic officer notes -> MiniLM embeddings -> FinBERT sentiment ->
     PCA(12) dense text vector + distress-keyword flag. FinBERT sanity-checked
     on hand-picked sentences.
  4. Ablation (honest, LOAN level — raw monthly data unavailable, so outcome
     is TARGET with pseudo_oot_split; the committed hazard model stays the
     core 12-month engine): LightGBM structured-only vs structured+graph+text,
     both evaluated on the OOT test window via the shared metrics harness.

Caveat stated up front: graph and text features are SYNTHETIC proxies whose
generators condition on observable behavior already present in the structured
matrix, so the ablation uplift is illustrative of the fusion MECHANISM, not
evidence of real-world signal. Small or negative uplift is a legitimate
finding and is reported verbatim.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping

from src.data.quality import serving_features_suite
from src.features.graph import (
    build_graph_features,
    generate_counterparty_graph,
    networkx_cross_check,
    validate_separability,
)
from src.features.text import (
    build_text_features,
    finbert_sentiment,
    generate_officer_notes,
)
from src.models.evaluate import metrics_row, metrics_table, naive_row
from src.models.hazard import SEGMENT_COLS
from src.models.splits import pseudo_oot_split

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
SEED = 42

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

FINBERT_SANITY_SENTENCES = [
    ("The company delivered record quarterly revenue and healthy profit growth.", "positive"),
    ("The plant is facing closure after repeated payment delays and mounting losses.", "negative"),
    ("The firm operates a mid-sized workshop in the industrial area.", "neutral"),
]


def loan_matrix(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Numeric features + one-hot segment dummies (same treatment for both arms)."""
    X = df[feature_cols].copy()
    for col in SEGMENT_COLS:
        X = pd.concat([X, pd.get_dummies(df[col], prefix=col, dtype=int)], axis=1)
    return X


def train_loan_lgbm(X: pd.DataFrame, y: pd.Series, split: pd.Series) -> LGBMClassifier:
    train_mask = (split == "train").to_numpy()
    calib_mask = (split == "calib").to_numpy()
    y_train = y[train_mask]
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    model = LGBMClassifier(
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=1000,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        random_state=SEED,
        verbose=-1,
    )
    model.fit(
        X[train_mask],
        y_train,
        eval_set=[(X[calib_mask], y[calib_mask])],
        eval_metric="auc",
        callbacks=[early_stopping(50, verbose=False)],
    )
    return model


def main():
    print("=== Phase 5 — Graph-lite + Text Feature Enrichment ===\n")

    features = pd.read_parquet(PROCESSED / "serving_features.parquet")
    serving_features_suite(features)
    print(f"Data-quality gate PASSED for serving_features ({len(features):,} loans).")

    # ---------- Graph-lite features (plan.md §8.2, §12.1 Stages 1-3) ----------
    print("\nGenerating synthetic bipartite firm<->counterparty graph...")
    edges, firm_profile = generate_counterparty_graph(features, seed=SEED)
    print(f"  {firm_profile['SK_ID_CURR'].nunique():,} firms, "
          f"{edges['counterparty_id'].nunique():,} counterparties, {len(edges):,} edges "
          f"({firm_profile['concentrated_flag'].mean():.1%} concentrated firms)")

    nx_check = networkx_cross_check(edges, seed=SEED)
    print(f"  networkx cross-check on validation subgraph: {nx_check}")

    graph_feats = build_graph_features(edges, firm_profile, validate_networkx=False, seed=SEED)

    separability = validate_separability(graph_feats, firm_profile)
    print("\nSeparability validation (NON-OPTIONAL, concentration feature alone):")
    for key, val in separability.items():
        print(f"  {key}: {val}")

    # ---------- Text features (plan.md §8.3, §9.5) ----------
    print("\nGenerating synthetic officer notes from finite template pool...")
    notes = generate_officer_notes(features, seed=SEED)
    print("  note tone distribution:")
    print(notes["note_tone"].value_counts(normalize=True).round(4).to_string())
    print(f"  unique sentences: {notes['officer_note'].nunique()} "
          f"(embeddings/FinBERT run on uniques only, mapped back to {len(notes):,} loans)")

    print("\nFinBERT sanity check on hand-picked sentences:")
    sanity = finbert_sentiment([s for s, _ in FINBERT_SANITY_SENTENCES])
    for (sentence, expected), (_, row) in zip(FINBERT_SANITY_SENTENCES, sanity.iterrows()):
        status = "OK" if row["sentiment_label"] == expected else "MISMATCH"
        print(f"  [{status}] expected={expected:8s} got={row['sentiment_label']:8s} "
              f"(score={row['sentiment_score']:.3f}) :: {sentence}")

    print("\nBuilding text features (MiniLM -> PCA(12) + FinBERT + distress flag)...")
    text_feats = build_text_features(notes, seed=SEED)
    print(f"  text feature columns: {[c for c in text_feats.columns if c != 'SK_ID_CURR']}")

    # ---------- Save feature tables ----------
    graph_feats.to_parquet(PROCESSED / "graph_features.parquet", index=False)
    text_feats.to_parquet(PROCESSED / "text_features.parquet", index=False)

    enriched = graph_feats.rename(columns={"data_provenance": "graph_data_provenance"}).merge(
        text_feats.rename(columns={"data_provenance": "text_data_provenance"}),
        on="SK_ID_CURR",
        how="inner",
    )
    enriched.to_parquet(PROCESSED / "enriched_features.parquet", index=False)

    serving_enriched = features.merge(enriched, on="SK_ID_CURR", how="left")
    serving_enriched.to_parquet(PROCESSED / "serving_features_enriched.parquet", index=False)
    print(f"Saved graph_features / text_features / enriched_features / "
          f"serving_features_enriched parquets ({len(serving_enriched):,} rows).")

    # ---------- Ablation (honest, loan level) ----------
    print("\n=== Ablation: structured-only vs structured + graph + text (LOAN level) ===")
    print("Note: raw monthly data unavailable -> ablation runs at loan level "
          "(outcome=TARGET) with pseudo_oot_split; the committed hazard model "
          "remains the core 12-month engine.")

    df = features.merge(graph_feats.drop(columns=["data_provenance"]), on="SK_ID_CURR").merge(
        text_feats[["SK_ID_CURR"] + TEXT_FEATURES], on="SK_ID_CURR"
    )
    split = pseudo_oot_split(df)
    y = df["TARGET"]
    test_mask = (split == "test").to_numpy()

    print("\nTraining (a) structured-only LightGBM...")
    X_struct = loan_matrix(df, STRUCTURED_FEATURES)
    model_struct = train_loan_lgbm(X_struct, y, split)
    print(f"  best iteration: {model_struct.best_iteration_}")

    print("Training (b) fused LightGBM (structured + graph + text-PCA + sentiment + distress)...")
    fused_cols = STRUCTURED_FEATURES + GRAPH_FEATURES + TEXT_FEATURES
    X_fused = loan_matrix(df, fused_cols)
    model_fused = train_loan_lgbm(X_fused, y, split)
    print(f"  best iteration: {model_fused.best_iteration_}")

    y_test = y[test_mask]
    score_struct = model_struct.predict_proba(X_struct[test_mask])[:, 1]
    score_fused = model_fused.predict_proba(X_fused[test_mask])[:, 1]

    table = metrics_table(
        [
            naive_row(y_test),
            metrics_row("structured_only_lgbm", y_test, score_struct),
            metrics_row("fused_structured_graph_text", y_test, score_fused),
        ]
    )
    print(f"\n=== Ablation table — OOT test window (n={int(test_mask.sum()):,}), "
          f"outcome = TARGET ===")
    print(table.to_string())

    auc_struct = table.loc["structured_only_lgbm", "auc_roc"]
    auc_fused = table.loc["fused_structured_graph_text", "auc_roc"]
    ks_struct = table.loc["structured_only_lgbm", "ks"]
    ks_fused = table.loc["fused_structured_graph_text", "ks"]
    print(f"\nUplift from graph+text enrichment: "
          f"AUC {auc_struct:.4f} -> {auc_fused:.4f} (delta {auc_fused - auc_struct:+.4f}), "
          f"KS {ks_struct:.4f} -> {ks_fused:.4f} (delta {ks_fused - ks_struct:+.4f}).")
    print("Caveat: graph/text features are synthetic proxies conditioned on observable "
          "behavior already in the structured matrix, so this uplift illustrates the "
          "fusion mechanism — it is NOT evidence of real-world graph/text signal. "
          "A small or negative delta is a legitimate, honest finding.")

    joblib.dump({"model": model_fused, "columns": list(X_fused.columns)},
                MODELS_DIR / "fused.joblib")
    table.to_json(MODELS_DIR / "phase5_metrics.json")
    print("\nSaved models/fused.joblib and models/phase5_metrics.json.")


if __name__ == "__main__":
    main()
