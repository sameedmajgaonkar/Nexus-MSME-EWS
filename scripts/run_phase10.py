"""Phase 10 — MLOps: tracking backfill, drift monitoring, retrain gate (plan.md §13.2-§13.4).

Steps (PHASES.md Phase 10 "How to verify"):
  a. Backfill every historical phaseN_metrics.json as MLflow runs (one run per
     model per phase, tagged phase=N) and register baseline / hazard / fused /
     thin_file_tabpfn — fused gets the 'production' alias (MLflow >= 2.9
     deprecates registry stages, so aliases implement Staging -> Production).
  b. Honest drift check: reference = pseudo-OOT train window, current = test
     window of serving_features_enriched (expect little/no drift).
  c. Deliberately shift a Tier-1 feature (credit_income_ratio x 1.6) in a copy
     of the current window -> PSI flag must fire -> retrain_and_gate runs LIVE
     (train window subsampled to 30% to keep runtime < ~5 min; the OOT test
     window is NOT subsampled, so the gate comparison stays honest).
  d. Promotion-gate hold-back proof: a deliberately crippled challenger
     (n_estimators=5, 2% of train rows) must be HELD and models/fused.joblib
     must be byte-identical before/after (sha256).
  e. PASS/FAIL per verify item + where to browse the MLflow UI.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from src.mlops import tracking
from src.mlops.drift import PSI_RETRAIN_THRESHOLD, TIER1_FEATURES, run_drift_report
from src.mlops.retrain import loan_matrix, retrain_and_gate
from src.models.splits import pseudo_oot_split

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
PROCESSED = ROOT / "data" / "processed"

# (phase tag, metrics file, extra tags) — dict shape: metric -> {model -> value}.
PHASE_METRIC_FILES = [
    ("2", "phase2_metrics.json", {}),
    ("3", "phase3_metrics_target.json", {"outcome": "target"}),
    ("3", "phase3_metrics_delinquency.json", {"outcome": "delinquency"}),
    ("4", "phase4_metrics.json", {}),
    ("5", "phase5_metrics.json", {}),
    # phase6 is nested one level deeper: dataset -> metric -> {model -> value}
    ("6", "phase6_metrics.json", {}),
]

# registered name -> (phase, source file, model column, artifact, alias)
REGISTRATIONS = {
    "baseline": ("2", "phase2_metrics.json", "woe_logistic_baseline",
                 "baseline.joblib", "staging"),
    "hazard": ("3", "phase3_metrics_delinquency.json", "hazard_model_12m_cum_pd",
               "hazard.joblib", "staging"),
    "fused": ("5", "phase5_metrics.json", "fused_structured_graph_text",
              "fused.joblib", "production"),
    "thin_file_tabpfn": ("6", "phase6_metrics.json:german_credit", "tabpfn_thin_file",
                         "thin_file_tabpfn.joblib", "staging"),
}


def _metrics_by_model(table: dict) -> dict:
    """Invert {metric: {model: value}} -> {model: {metric: value}}."""
    out: dict = {}
    for metric, per_model in table.items():
        for model, value in per_model.items():
            out.setdefault(model, {})[metric] = value
    return out


def backfill_registry() -> dict:
    """Step (a): one MLflow run per model per phase + registry entries.

    Runs to be registered get the model .joblib attached as an artifact.
    """
    reg_lookup = {
        (source_key, model_col): (reg_name, artifact, alias)
        for reg_name, (_, source_key, model_col, artifact, alias) in REGISTRATIONS.items()
    }
    run_ids: dict = {}  # (source_key, model) -> run_id
    n_runs = 0
    for phase, filename, extra_tags in PHASE_METRIC_FILES:
        raw = json.loads((MODELS_DIR / filename).read_text())  # json accepts NaN literals
        # phase6 nests by dataset
        blocks = (
            {f"{filename}:{ds}": tbl for ds, tbl in raw.items()}
            if phase == "6"
            else {filename: raw}
        )
        for source_key, table in blocks.items():
            for model, metrics in _metrics_by_model(table).items():
                reg_name, artifact, _alias = reg_lookup.get(
                    (source_key, model), (None, None, None)
                )
                tags = {"phase": phase, "model": model, "backfill": "true",
                        "source": source_key, **extra_tags}
                if reg_name:
                    tags["registered_as"] = reg_name
                run_id = tracking.log_model_run(
                    f"phase{phase}_{model}",
                    params={"source_file": source_key},
                    metrics=metrics,
                    artifact_path=(MODELS_DIR / artifact) if artifact else None,
                    tags=tags,
                )
                run_ids[(source_key, model)] = run_id
                n_runs += 1
    print(f"  Backfilled {n_runs} historical runs from "
          f"{len(PHASE_METRIC_FILES)} metrics files.")

    versions = {}
    for reg_name, (phase, source_key, model_col, artifact, alias) in REGISTRATIONS.items():
        version = tracking.register_and_stage(
            run_ids[(source_key, model_col)], reg_name, alias
        )
        versions[reg_name] = (version, alias)
        print(f"  Registered '{reg_name}' v{version} (alias '{alias}') "
              f"from phase {phase} run, artifact models/{artifact}.")
    return versions


def _score_with_production(df: pd.DataFrame) -> pd.Series:
    """Score rows with the current Production fused artifact (own artifact — trusted)."""
    bundle = joblib.load(MODELS_DIR / "fused.joblib")
    X = loan_matrix(df).reindex(columns=bundle["columns"], fill_value=0)
    return pd.Series(bundle["model"].predict_proba(X)[:, 1], index=df.index)


def _print_summary(summary: dict, label: str) -> None:
    print(f"\n--- Drift summary ({label}) ---")
    worst = sorted(summary["psi_by_feature"].items(), key=lambda kv: -kv[1])[:5]
    print("  top PSI: " + ", ".join(f"{f}={v:.3f}" for f, v in worst))
    print(f"  drifted_features: {summary['drifted_features']}")
    print(f"  calibration_status: {summary['calibration_status']} "
          f"(score_psi={summary['score_psi']})")
    print(f"  retrain_triggered: {summary['retrain_triggered']}")
    print(f"  plain_language: {summary['plain_language']}")
    print(f"  recommendation: {summary['recommendation']}")


def main() -> None:
    print("=== Phase 10 — MLOps: Tracking, Drift Monitoring, Retrain Loop ===\n")
    results = {}

    # ---------- (a) MLflow backfill + registry ----------
    print("[a] Backfilling historical training runs into MLflow (file:./mlruns)...")
    print("    NOTE: MLflow >= 2.9 deprecates registry STAGES; using registered-model")
    print("    ALIASES 'staging'/'production' instead (MLflow "
          f"{__import__('mlflow').__version__}).")
    backfill_registry()

    expected_models = set(REGISTRATIONS)
    client = tracking._client()
    registered = {m.name for m in client.search_registered_models()}
    prod_metrics = tracking.get_production_metrics("fused")
    results["registry_backfill"] = expected_models <= registered and bool(prod_metrics)
    print(f"  Registered models present: {sorted(registered)}")
    print(f"  'fused'@production logged metrics: {prod_metrics}")

    # ---------- (b) honest drift check ----------
    print("\n[b] Drift check on honest windows (reference = pseudo-OOT train, "
          "current = test)...")
    df = pd.read_parquet(PROCESSED / "serving_features_enriched.parquet")
    split = pseudo_oot_split(df)
    reference = df[(split == "train").to_numpy()].copy()
    current = df[(split == "test").to_numpy()].copy()
    print(f"  reference n={len(reference):,}, current n={len(current):,}")
    print("  scoring both windows with Production fused model for the "
          "calibration/prediction-drift check...")
    reference["prediction"] = _score_with_production(reference)
    current["prediction"] = _score_with_production(current)

    honest = run_drift_report(reference, current, TIER1_FEATURES,
                              prediction_col="prediction")
    _print_summary(honest, "honest windows")

    # ---------- (c) deliberately shifted Tier-1 feature -> retrain fires ----------
    print("\n[c] Deliberately shifting Tier-1 feature 'credit_income_ratio' x 1.6 "
          "in a copy of the current window...")
    shifted = current.copy()
    shifted["credit_income_ratio"] = shifted["credit_income_ratio"] * 1.6
    shifted["prediction"] = _score_with_production(shifted)
    drifted = run_drift_report(reference, shifted, TIER1_FEATURES,
                               prediction_col="prediction")
    _print_summary(drifted, "shifted credit_income_ratio x1.6")

    assert "credit_income_ratio" in drifted["drifted_features"], (
        "PSI flag did not fire on the shifted Tier-1 feature")
    assert drifted["retrain_triggered"] is True, "retrain flag did not fire"
    print(f"\n  PSI flag fired (credit_income_ratio PSI="
          f"{drifted['psi_by_feature']['credit_income_ratio']:.3f} > "
          f"{PSI_RETRAIN_THRESHOLD}) -> invoking retrain_and_gate LIVE...")
    print("  (train window subsampled to 30% for runtime; OOT test window kept "
          "in full so the gate comparison is honest)")

    t0 = time.time()
    live = retrain_and_gate(df, split, sample_frac=0.30)
    print(f"  retrain_and_gate finished in {time.time() - t0:.1f}s "
          f"(training {live['train_seconds']}s)")
    print(f"  challenger: {live['challenger_metrics']}")
    print(f"  production: {live['production_metrics']}")
    print(f"  gate decision: {'PROMOTED' if live['promoted'] else 'HELD'} — "
          f"{live['reason']}")
    results["psi_flag_fires_retrain"] = True

    # ---------- (d) promotion gate holds back a deliberately worse model ----------
    print("\n[d] Promotion-gate hold-back proof: crippled challenger "
          "(n_estimators=5, 2% of train rows)...")
    fused_path = MODELS_DIR / "fused.joblib"
    hash_before = hashlib.sha256(fused_path.read_bytes()).hexdigest()
    crippled = retrain_and_gate(
        df, split, challenger_params={"n_estimators": 5}, sample_frac=0.02
    )
    hash_after = hashlib.sha256(fused_path.read_bytes()).hexdigest()
    print(f"  challenger AUC {crippled['challenger_metrics']['auc_roc']:.4f} vs "
          f"production AUC {crippled['production_metrics']['auc_roc']:.4f}")
    print(f"  gate decision: {'PROMOTED' if crippled['promoted'] else 'HELD'} — "
          f"{crippled['reason']}")
    print(f"  models/fused.joblib sha256 before: {hash_before[:16]}... "
          f"after: {hash_after[:16]}...")
    assert crippled["promoted"] is False, "gate promoted a worse model!"
    assert hash_before == hash_after, "fused.joblib changed despite a HELD decision!"
    results["gate_holds_back_worse_model"] = True

    # restore the honest summary so the serving API (/api/drift/report) shows
    # the real state of the portfolio, not the synthetic-shift demo
    run_drift_report(reference, current, TIER1_FEATURES, prediction_col="prediction")
    print("\n  reports/drift_summary.json restored to the honest (unshifted) "
          "comparison for the serving API.")

    # ---------- (e) verify lines ----------
    print("\n[e] Browse the MLflow UI with:")
    print("    uv run mlflow ui --backend-store-uri file:./mlruns")
    print("    (set MLFLOW_ALLOW_FILE_STORE=true — MLflow 3.x gates the local "
          "file store behind it)")

    print("\n=== PHASES.md Phase-10 verification ===")
    checks = [
        ("Phase 2/3/5/6 models in MLflow registry with logged metrics",
         results.get("registry_backfill", False)),
        ("Shifted feature raises PSI-above-threshold flag and fires retrain",
         results.get("psi_flag_fires_retrain", False)),
        ("Promotion gate holds back a deliberately worse model",
         results.get("gate_holds_back_worse_model", False)),
    ]
    all_ok = True
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        all_ok &= ok
    print(f"\nPhase 10: {'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
