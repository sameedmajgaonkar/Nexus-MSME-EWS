"""Phase 4: fit per-segment isotonic calibration, produce the unified grade table,
reliability diagram, and a SHAP top-5 demo for one loan."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.panel import MAX_MONTHS
from src.models.calibration import (
    ACTION_BY_GRADE,
    apply_calibration,
    fit_calibrators,
    pd_to_grade,
    reliability_table,
)
from src.models.evaluate import metrics_row, metrics_table, naive_row
from src.models.explain import build_explainer, top_drivers
from src.models.hazard import align_columns, build_model_matrix

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def main():
    calib = pd.read_parquet(ROOT / "data" / "processed" / "phase3_calib_scores.parquet")
    test = pd.read_parquet(ROOT / "data" / "processed" / "phase3_test_scores.parquet")

    calib_eval = calib[(calib["event"] == 1) | (calib["duration"] >= MAX_MONTHS)]
    print(f"Fitting isotonic calibrators on {len(calib_eval):,} evaluable calib loans, per segment...")
    calibrators = fit_calibrators(calib_eval)

    test = test.copy()
    test["calibrated_pd_12m"] = apply_calibration(
        calibrators, test["loan_type_segment"], test["hazard_cum_pd_12m"]
    )
    test["risk_grade"] = test["calibrated_pd_12m"].apply(pd_to_grade)
    test["recommended_action"] = test["risk_grade"].map(ACTION_BY_GRADE)

    # --- Reliability diagram (test evaluable set) ---
    test_eval = test[(test["event"] == 1) | (test["duration"] >= MAX_MONTHS)]
    rel = reliability_table(test_eval["event"], test_eval["calibrated_pd_12m"])
    print("\n=== Reliability table (calibrated PD vs observed 12m delinquency rate) ===")
    print(rel.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(rel["mean_predicted"], rel["observed_rate"], "o-", label="calibrated hazard PD")
    lim = max(rel["mean_predicted"].max(), rel["observed_rate"].max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="perfect calibration")
    ax.set_xlabel("Mean predicted 12m PD")
    ax.set_ylabel("Observed 12m delinquency rate")
    ax.set_title("Reliability diagram — per-segment isotonic calibration (OOT test)")
    ax.legend()
    fig.savefig(REPORTS_DIR / "reliability_diagram.png", dpi=120, bbox_inches="tight")
    print(f"Saved {REPORTS_DIR / 'reliability_diagram.png'}")

    # Brier before/after calibration.
    y = test_eval["event"]
    table = metrics_table(
        [
            naive_row(y),
            metrics_row("hazard_raw_cum_pd", y, test_eval["hazard_cum_pd_12m"]),
            metrics_row("hazard_calibrated_pd", y, test_eval["calibrated_pd_12m"]),
        ]
    )
    print("\n=== Calibration effect (OOT test, outcome = 12m serious delinquency) ===")
    print(table.to_string())

    # --- DoD item #4: unified grade table across >= 2 segments (§12.8 style) ---
    print("\n=== Unified grade scale across segments ===")
    seg_table = (
        test.groupby(["loan_type_segment", "risk_grade"], observed=True)
        .agg(n=("SK_ID_CURR", "size"), mean_calibrated_pd=("calibrated_pd_12m", "mean"))
        .reset_index()
    )
    print(seg_table.to_string(index=False))

    # --- SHAP top-5 for one sample high-risk loan ---
    bundle = joblib.load(MODELS_DIR / "hazard.joblib")
    model, columns = bundle["model"], bundle["columns"]
    explainer = build_explainer(model)

    features = pd.read_parquet(ROOT / "data" / "processed" / "serving_features.parquet")
    sample_id = int(test.sort_values("calibrated_pd_12m", ascending=False).iloc[0]["SK_ID_CURR"])
    loan_row = features[features["SK_ID_CURR"] == sample_id].reset_index(drop=True)
    loan_row["months_since_origination"] = MAX_MONTHS
    X_row = align_columns(build_model_matrix(loan_row), columns)

    drivers = top_drivers(explainer, X_row)
    print(f"\n=== SHAP top-5 drivers for sample loan SK_ID_CURR={sample_id} ===")
    for d in drivers:
        print(f"  {d['label']}: value={d['value']}, shap={d['shap']:+.4f} ({d['direction']})")

    joblib.dump(calibrators, MODELS_DIR / "calibrators.joblib")
    test.to_parquet(ROOT / "data" / "processed" / "phase4_scored_test.parquet", index=False)
    table.to_json(MODELS_DIR / "phase4_metrics.json")
    print("\nSaved calibrators + graded test scores.")


if __name__ == "__main__":
    main()
