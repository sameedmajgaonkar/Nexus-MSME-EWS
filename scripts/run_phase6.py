"""Phase 6: TabPFN thin-file specialist vs LightGBM on identical small samples
(plan.md §9.3, §12.9). Two experiments:

1. German Credit (1,000 rows) — TabPFN's own proof-of-concept dataset, with the
   project's ordered 70/10/20 pseudo-OOT protocol (row order as pseudo-time).
2. A deterministic 500-train / 300-test subsample of the Home Credit NTC/NTB
   segment (data_richness == 'ntc_ntb'), simulating the real thin-file regime.

Prints both side-by-side AUC/KS tables, saves models/thin_file_tabpfn.joblib
({model, columns}) and models/phase6_metrics.json, and prints one sample
loan's SHAP top drivers to show format consistency with Phase 4.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from src.data.load_german_credit import load_german_credit
from src.models.evaluate import metrics_row, metrics_table, naive_row
from src.models.splits import pseudo_oot_split
from src.models.thin_file import (
    SHAP_PATH,
    THIN_FILE_FEATURES,
    tabpfn_top_drivers,
    train_lgbm_small,
    train_tabpfn,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

NTC_TRAIN_ROWS = 500  # train+calib context for TabPFN (plan.md §12.9: 200-500)
NTC_TEST_ROWS = 300


def side_by_side(name: str, y_test, tabpfn_scores, lgbm_scores) -> pd.DataFrame:
    table = metrics_table(
        [
            naive_row(y_test),
            metrics_row("tabpfn_thin_file", y_test, tabpfn_scores),
            metrics_row("lgbm_small_sample", y_test, lgbm_scores),
        ]
    )
    print(f"\n=== {name}: TabPFN vs LightGBM on the IDENTICAL sample (§12.9) ===")
    print(table.to_string())
    return table


def experiment_german_credit() -> pd.DataFrame:
    df = load_german_credit()
    split = pseudo_oot_split(df, id_col="row_id")
    feature_cols = [c for c in df.columns if c not in ("row_id", "TARGET", "data_provenance")]

    train_mask = (split == "train").to_numpy()  # 700 rows — under the 1,000 CPU cap
    test_mask = (split == "test").to_numpy()
    X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, "TARGET"]
    X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, "TARGET"]
    print(
        f"German Credit: {len(df)} rows, {len(feature_cols)} numeric features "
        f"(one-hot encoded), train={train_mask.sum()}, test={test_mask.sum()}, "
        f"test default rate={y_test.mean():.3f}"
    )

    tabpfn = train_tabpfn(X_train, y_train)
    lgbm = train_lgbm_small(X_train, y_train)
    return side_by_side(
        "Experiment 1 — German Credit",
        y_test,
        tabpfn.predict_proba(X_test)[:, 1],
        lgbm.predict_proba(X_test)[:, 1],
    )


def experiment_ntc_ntb() -> pd.DataFrame:
    features = pd.read_parquet(ROOT / "data" / "processed" / "serving_features.parquet")
    thin = (
        features[features["data_richness"] == "ntc_ntb"]
        .sort_values("SK_ID_CURR")
        .reset_index(drop=True)
    )
    # Deterministic subsample across the ID range, then ordered train/test:
    # earlier pseudo-time rows condition TabPFN, later rows are held-out test.
    rng = np.random.default_rng(42)
    positions = np.sort(
        rng.choice(len(thin), size=NTC_TRAIN_ROWS + NTC_TEST_ROWS, replace=False)
    )
    sample = thin.iloc[positions]
    train_df = sample.iloc[:NTC_TRAIN_ROWS]
    test_df = sample.iloc[NTC_TRAIN_ROWS:]

    # TabPFN v2 and LightGBM both accept NaN natively, so NaNs pass through —
    # identical inputs for both models (§12.9); constant-filling hurt TabPFN.
    X_train = train_df[THIN_FILE_FEATURES]
    X_test = test_df[THIN_FILE_FEATURES]
    y_train, y_test = train_df["TARGET"], test_df["TARGET"]
    print(
        f"\nNTC/NTB thin-file subsample: {len(thin):,} eligible loans -> "
        f"train={len(X_train)}, test={len(X_test)} (disjoint, ID-ordered), "
        f"train default rate={y_train.mean():.3f}, test default rate={y_test.mean():.3f}"
    )

    tabpfn = train_tabpfn(X_train, y_train)
    lgbm = train_lgbm_small(X_train, y_train)
    table = side_by_side(
        "Experiment 2 — Home Credit NTC/NTB subsample",
        y_test,
        tabpfn.predict_proba(X_test)[:, 1],
        lgbm.predict_proba(X_test)[:, 1],
    )

    joblib.dump(
        {"model": tabpfn, "columns": THIN_FILE_FEATURES},
        MODELS_DIR / "thin_file_tabpfn.joblib",
    )
    print(f"\nSaved {MODELS_DIR / 'thin_file_tabpfn.joblib'}")

    # SHAP format-consistency demo: highest-scored test loan's top drivers.
    scores = tabpfn.predict_proba(X_test)[:, 1]
    idx = int(np.argmax(scores))
    sample_id = int(test_df.iloc[idx]["SK_ID_CURR"])
    x_row = X_test.iloc[[idx]].reset_index(drop=True)
    print(f"\nSHAP path in use: {SHAP_PATH}")
    drivers = tabpfn_top_drivers(tabpfn, X_train, x_row)
    print(f"=== TabPFN SHAP top-5 drivers, thin-file loan SK_ID_CURR={sample_id} ===")
    for d in drivers:
        print(f"  {d['label']}: value={d['value']}, shap={d['shap']:+.5f} ({d['direction']})")

    return table


def main():
    german_table = experiment_german_credit()
    ntc_table = experiment_ntc_ntb()

    metrics_path = MODELS_DIR / "phase6_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "german_credit": german_table.to_dict(),
                "ntc_ntb_subsample": ntc_table.to_dict(),
            },
            indent=2,
        )
    )
    print(f"\nSaved {metrics_path}")


if __name__ == "__main__":
    main()
