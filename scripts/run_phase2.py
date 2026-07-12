"""Phase 2: train the WOE-logistic baseline and print the mandatory metrics table."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from src.models.baseline import train_baseline_scorecard
from src.models.evaluate import metrics_row, metrics_table, naive_row
from src.models.splits import pseudo_oot_split

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def main():
    features = pd.read_parquet(ROOT / "data" / "processed" / "phase1_features.parquet")
    split = pseudo_oot_split(features)
    print("Split sizes:\n", split.value_counts(), sep="")

    print("\nFitting WOE binners on train window + training logistic scorecard...")
    model, binners, scores = train_baseline_scorecard(features, split)

    test_mask = split == "test"
    y_test = features.loc[test_mask, "TARGET"]
    s_test = scores[test_mask]

    table = metrics_table(
        [naive_row(y_test), metrics_row("woe_logistic_baseline", y_test, s_test)]
    )
    print("\n=== Phase 2 metrics (OOT test window) ===")
    print(table.to_string())

    joblib.dump({"model": model, "binners": binners}, MODELS_DIR / "baseline.joblib")
    scores_df = features[["SK_ID_CURR", "TARGET"]].copy()
    scores_df["split"] = split
    scores_df["baseline_pd"] = scores
    scores_df.to_parquet(ROOT / "data" / "processed" / "phase2_baseline_scores.parquet", index=False)
    table.to_json(MODELS_DIR / "phase2_metrics.json")
    print("\nSaved baseline model, scores, and metrics.")


if __name__ == "__main__":
    main()
