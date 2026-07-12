"""Phase 7: narrative, counterfactual & uncertainty layer — DoD item #3 end-to-end.

Fits confidence bands from the Phase 4 OOT test residuals, persists them to
models/confidence_bands.joblib, then runs the full SHAP -> narrative ->
counterfactual -> confidence-band chain for one real high-risk loan
(plan.md §11.2, §12.3, §9.6)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# joblib load/dump here only touches models/*.joblib artifacts produced by this
# repo's own phase scripts (trusted, project-internal) — same pattern as run_phase4.
import joblib
import pandas as pd

from src.explainability.counterfactual import find_counterfactual, score_loan
from src.explainability.narrative import generate_narrative
from src.explainability.uncertainty import confidence_band, fit_confidence_bands, models_disagree
from src.models.calibration import ACTION_BY_GRADE
from src.models.explain import build_explainer, top_drivers
from src.models.hazard import MAX_MONTHS, align_columns, build_model_matrix

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
PROCESSED = ROOT / "data" / "processed"
MAX_CANDIDATE_LOANS = 30  # high-risk loans to try until one yields a counterfactual


def main():
    # --- 1. Confidence bands from Phase 4 OOT test residuals (§9.6) ---
    scored = pd.read_parquet(PROCESSED / "phase4_scored_test.parquet")
    bands = fit_confidence_bands(scored)
    joblib.dump(bands, MODELS_DIR / "confidence_bands.joblib")
    print(f"Fitted confidence bands for {len(bands)} segment keys "
          f"({sum(len(v) for v in bands.values())} PD bins total); "
          f"saved {MODELS_DIR / 'confidence_bands.joblib'}")

    # --- 2. Load the trained scoring chain ---
    bundle = joblib.load(MODELS_DIR / "hazard.joblib")
    model, columns = bundle["model"], bundle["columns"]
    calibrators = joblib.load(MODELS_DIR / "calibrators.joblib")
    explainer = build_explainer(model)
    features = pd.read_parquet(PROCESSED / "serving_features.parquet")

    # High-risk = grade E or worse; try the ones nearest the next-better grade
    # boundary first, so the smallest realistic counterfactual exists.
    high_risk = (
        scored[scored["risk_grade"].isin(list("EFG"))]
        .sort_values("calibrated_pd_12m")
        .head(MAX_CANDIDATE_LOANS)
    )

    chosen = None
    for sk_id in high_risk["SK_ID_CURR"]:
        loan_row = features[features["SK_ID_CURR"] == sk_id].reset_index(drop=True)
        if loan_row.empty:
            continue
        segment = loan_row.iloc[0]["loan_type_segment"]

        loan_matrix = loan_row.copy()
        loan_matrix["months_since_origination"] = MAX_MONTHS
        X_row = align_columns(build_model_matrix(loan_matrix), columns)
        drivers = top_drivers(explainer, X_row)

        calibrated_pd, grade = score_loan(model, columns, calibrators, segment, loan_row)
        cf = find_counterfactual(model, columns, calibrators, segment, loan_row, drivers)
        chosen = (sk_id, segment, loan_row, drivers, calibrated_pd, grade, cf)
        if cf.get("feature") is not None:
            break

    sk_id, segment, loan_row, drivers, calibrated_pd, grade, cf = chosen

    # --- 3. The DoD-item-3 chain: SHAP -> narrative -> counterfactual -> band ---
    print(f"\n=== Sample high-risk loan SK_ID_CURR={int(sk_id)} "
          f"(segment={segment}, grade={grade}, calibrated 12m PD={calibrated_pd:.4f}) ===")
    print(f"Recommended action: {ACTION_BY_GRADE[grade]}")

    print("\n--- Level 1: SHAP top-5 drivers ---")
    for d in drivers:
        print(f"  {d['label']}: value={d['value']}, shap={d['shap']:+.4f} ({d['direction']})")

    print("\n--- Level 2: narrative (guardrail-verified) ---")
    narrative = generate_narrative(drivers, grade, calibrated_pd)
    print(f"  source={narrative['source']}, verified={narrative['verified']}")
    print(f"  {narrative['text']}")

    print("\n--- Level 3: counterfactual (smallest change to next-better grade) ---")
    if cf.get("feature") is None:
        print(f"  No counterfactual found: {cf['reason']}")
    else:
        print(f"  If {cf['label']} moved from {cf['current_value']} to {cf['suggested_value']}, "
              f"calibrated PD falls from {calibrated_pd:.4f} to {cf['new_pd']:.4f} "
              f"(grade {grade} -> {cf['new_grade']}).")

    print("\n--- Uncertainty: confidence band (§9.6) ---")
    band = confidence_band(bands, segment, calibrated_pd)
    print(f"  PD {calibrated_pd * 100:.0f}% ± {band['half_width'] * 100:.0f}%, "
          f"calibrated on n={band['n_comparable']} comparable accounts")
    print(f"  wide_band_flag={band['wide_band_flag']}"
          + (" -> mandatory human review" if band["wide_band_flag"] else ""))
    print(f"  models_disagree demo (hazard {calibrated_pd:.2f} vs hypothetical TabPFN "
          f"{calibrated_pd + 0.20:.2f}): {models_disagree(calibrated_pd, calibrated_pd + 0.20)}")

    print("\nPhase 7 chain complete: SHAP -> narrative -> counterfactual -> confidence band.")


if __name__ == "__main__":
    main()
