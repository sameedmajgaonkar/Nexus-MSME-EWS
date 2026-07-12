"""Phase 3: build the person-period panel, train the hazard model, extend the metrics table.

Two evaluation tables, deliberately separate:
  Table 1 — outcome = TARGET (current-loan default): baseline vs hazard, both leak-free.
  Table 2 — outcome = serious delinquency within 12m (the hazard model's native
            event): hazard vs naive only. The WOE baseline is excluded here
            because its late_installment_rate feature derives from the same
            installments that define this event (would be leakage-flattered).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from src.data.load_home_credit import load_installments
from src.features.panel import MAX_MONTHS, build_person_period_panel, prior_loan_behavior
from src.models.evaluate import metrics_row, metrics_table, naive_row
from src.models.hazard import align_columns, build_model_matrix, train_hazard_model
from src.models.splits import pseudo_oot_split

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


def score_loans(model, columns, loans: pd.DataFrame) -> np.ndarray:
    """12-month hazard matrix for a frame of unique-loan static features."""
    expanded = loans.loc[loans.index.repeat(MAX_MONTHS)].reset_index(drop=True)
    expanded["months_since_origination"] = np.tile(np.arange(1, MAX_MONTHS + 1), len(loans))
    X = align_columns(build_model_matrix(expanded), columns)
    return model.predict_proba(X)[:, 1].reshape(len(loans), MAX_MONTHS)


def main():
    print("Loading installments (13.6M rows) and building person-period panel...")
    installments = load_installments(
        usecols=[
            "SK_ID_CURR",
            "SK_ID_PREV",
            "NUM_INSTALMENT_NUMBER",
            "DAYS_INSTALMENT",
            "DAYS_ENTRY_PAYMENT",
        ]
    )
    panel = build_person_period_panel(installments)
    prior = prior_loan_behavior(installments)
    print(f"Panel: {len(panel):,} person-period rows, {panel['SK_ID_CURR'].nunique():,} loans")

    features = pd.read_parquet(ROOT / "data" / "processed" / "phase1_features.parquet")
    features = features.merge(prior, on="SK_ID_CURR", how="left")
    panel_features = panel.merge(features, on="SK_ID_CURR", how="inner")
    print(f"Panel joined with features: {len(panel_features):,} rows, "
          f"row event rate = {panel_features['event'].mean():.4f}")

    split = pseudo_oot_split(panel_features)  # by SK_ID_CURR -> a loan never straddles windows

    print("Training LightGBM discrete-time hazard model...")
    model, columns = train_hazard_model(panel_features, split)
    print(f"Best iteration: {model.best_iteration_}")

    durations = panel_features.groupby("SK_ID_CURR").agg(
        duration=("months_since_origination", "max"), event=("event", "max")
    )

    test_loans = (
        panel_features[split == "test"].drop_duplicates("SK_ID_CURR").reset_index(drop=True)
    )
    hazards = score_loans(model, columns, test_loans)
    cum_pd = 1.0 - np.prod(1.0 - hazards, axis=1)

    scored = test_loans[["SK_ID_CURR", "TARGET", "loan_type_segment"]].copy()
    scored["hazard_cum_pd_12m"] = cum_pd
    scored = scored.merge(durations, on="SK_ID_CURR")

    baseline_scores = pd.read_parquet(ROOT / "data" / "processed" / "phase2_baseline_scores.parquet")
    scored = scored.merge(baseline_scores[["SK_ID_CURR", "baseline_pd"]], on="SK_ID_CURR")

    # --- Table 1: common outcome = TARGET (current-loan default), both models leak-free ---
    y1 = scored["TARGET"]
    table1 = metrics_table(
        [
            naive_row(y1),
            metrics_row("woe_logistic_baseline", y1, scored["baseline_pd"]),
            metrics_row("hazard_model_12m_cum_pd", y1, scored["hazard_cum_pd_12m"]),
        ]
    )
    print("\n=== Table 1 — OOT test, outcome = TARGET (current-loan default) ===")
    print(table1.to_string())

    # --- Table 2: hazard model's native outcome = serious delinquency within 12 months ---
    evaluable = scored[(scored["event"] == 1) | (scored["duration"] >= MAX_MONTHS)]
    y2 = evaluable["event"]
    table2 = metrics_table(
        [
            naive_row(y2),
            metrics_row("hazard_model_12m_cum_pd", y2, evaluable["hazard_cum_pd_12m"]),
        ]
    )
    print(f"\n=== Table 2 — OOT test, outcome = serious delinquency within 12m "
          f"(n={len(evaluable):,} evaluable loans) ===")
    print(table2.to_string())

    # DoD item #1: a real hazard curve for one sample loan.
    sample_idx = int(np.argmax(cum_pd))
    print(f"\nSample loan SK_ID_CURR={test_loans.loc[sample_idx, 'SK_ID_CURR']} hazard curve:")
    print("  monthly hazards:", np.round(hazards[sample_idx], 4).tolist())
    print(f"  cumulative 12m PD: {cum_pd[sample_idx]:.3f}")

    joblib.dump({"model": model, "columns": columns}, MODELS_DIR / "hazard.joblib")
    scored.to_parquet(ROOT / "data" / "processed" / "phase3_test_scores.parquet", index=False)

    calib_loans = (
        panel_features[split == "calib"].drop_duplicates("SK_ID_CURR").reset_index(drop=True)
    )
    hazards_c = score_loans(model, columns, calib_loans)
    calib_scored = calib_loans[["SK_ID_CURR", "TARGET", "loan_type_segment"]].copy()
    calib_scored["hazard_cum_pd_12m"] = 1.0 - np.prod(1.0 - hazards_c, axis=1)
    calib_scored = calib_scored.merge(durations, on="SK_ID_CURR")
    calib_scored.to_parquet(ROOT / "data" / "processed" / "phase3_calib_scores.parquet", index=False)

    # Full feature snapshot for serving (Phase 5 scores any loan on demand).
    features.to_parquet(ROOT / "data" / "processed" / "serving_features.parquet", index=False)

    table1.to_json(MODELS_DIR / "phase3_metrics_target.json")
    table2.to_json(MODELS_DIR / "phase3_metrics_delinquency.json")
    print("\nSaved hazard model, test/calib loan-level scores, serving feature snapshot.")


if __name__ == "__main__":
    main()
