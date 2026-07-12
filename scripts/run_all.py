"""Phase 12 — full-chain integration dry run (PHASES.md Phase 12, plan.md §19.4).

Exercises the entire system in one pass: data-quality gate -> enrichment artifacts ->
segment routing -> scoring -> calibration -> explanation (SHAP/narrative/counterfactual/
confidence band) -> immutable audit -> governance endpoints -> drift report -> stress
test -> streaming alert. Prints one PASS/FAIL line per stage and exits non-zero on any
failure. Run: `uv run python scripts/run_all.py [--thin-file]`
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS: list[tuple[str, bool, str]] = []


def stage(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def main() -> int:
    import pandas as pd

    # 1. Data-quality gate (Phase 0/§6.5)
    from src.data.quality import DataQualityError, serving_features_suite

    enriched_path = ROOT / "data/processed/serving_features_enriched.parquet"
    features = pd.read_parquet(enriched_path)
    try:
        serving_features_suite(features)
        stage("data-quality gate", True, f"{len(features):,} rows validated")
    except DataQualityError as exc:
        stage("data-quality gate", False, str(exc))

    # 2. Model artifacts present (Phases 2-7, 10)
    artifacts = [
        "models/baseline.joblib", "models/hazard.joblib", "models/calibrators.joblib",
        "models/fused.joblib", "models/thin_file_tabpfn.joblib", "models/confidence_bands.joblib",
    ]
    missing = [a for a in artifacts if not (ROOT / a).exists()]
    stage("model artifacts", not missing, "all present" if not missing else f"missing: {missing}")

    # 3. Enrichment columns (Phase 5)
    needed = {"counterparty_concentration", "degree_centrality", "anchor_linkage_flag",
              "network_churn", "text_pc_1", "distress_keyword_flag"}
    have = needed.issubset(features.columns)
    stage("graph+text enrichment columns", have,
          "present" if have else f"missing: {needed - set(features.columns)}")

    # 4. Segment router (Phase 6)
    from src.models.router import route_segment

    est = features[features["data_richness"] == "established"].iloc[0]
    ntc = features[features["data_richness"] == "ntc_ntb"].iloc[0]
    ok = route_segment(est) == "hazard" and route_segment(ntc) == "tabpfn_thin_file"
    stage("segment routing", ok, f"established->hazard, ntc_ntb->tabpfn_thin_file")

    # 5-11. Serving chain over the live app (Phases 8, 9-backend, 11)
    from fastapi.testclient import TestClient

    from src.serving.app import app

    with TestClient(app) as client:
        loans = client.get("/api/loans", params={"per_grade": 2}).json()
        loan_id = next(l["loan_id"] for l in loans if l["risk_grade"] in ("C", "D", "E"))

        # 5. Six-part §11.3 score
        score = client.post(f"/api/score/{loan_id}").json()
        six = ["calibrated_pd_12m", "hazard_curve", "risk_grade", "top_drivers",
               "counterfactual", "recommended_action", "narrative", "confidence_band"]
        missing_fields = [f for f in six if f not in score]
        stage("score: six-part output (§11.3)", not missing_fields,
              f"loan {loan_id} grade {score.get('risk_grade')} pd {score.get('calibrated_pd_12m'):.4f}"
              if not missing_fields else f"missing {missing_fields}")
        stage("score: 12-value hazard curve", isinstance(score.get("hazard_curve"), list)
              and len(score["hazard_curve"]) == 12, "curve, not a flag")
        stage("score: narrative guardrail-verified", bool(score.get("narrative", {}).get("verified")),
              f"source={score.get('narrative', {}).get('source')}")

        # 6. Cached explanation without re-scoring
        explain = client.get(f"/api/explain/{loan_id}")
        stage("explain: cached retrieval", explain.status_code == 200)

        # 7. Override governance: reason mandatory + append-only audit
        bad = client.post(f"/api/override/{loan_id}", json={"decision": "override"})
        good = client.post(f"/api/override/{loan_id}",
                           json={"decision": "override", "reason": "integration dry-run"})
        stage("override: reason mandatory", bad.status_code == 422 and good.status_code == 200,
              f"no-reason->{bad.status_code}, with-reason->{good.status_code}")

        db_path = ROOT / "data/overrides.db"
        with sqlite3.connect(db_path) as conn:
            n_audit = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
            try:
                conn.execute("UPDATE overrides SET reason='tampered'")
                append_only = False
            except sqlite3.IntegrityError:
                append_only = True
            except sqlite3.OperationalError:
                append_only = True
        stage("audit log: immutable rows", n_audit > 0 and append_only,
              f"{n_audit} audit rows; UPDATE blocked by trigger")

        # 8. Governance endpoints (Phase 9 backend)
        fairness = client.get("/api/fairness/audit").json()
        disparity = any(
            g.get("disparity_ratio") not in (None, 1.0)
            for dim in fairness.get("dimensions", {}).values() for g in dim
        )
        stage("fairness audit: real disparity number", disparity)
        rbi = client.get("/api/rbi/mapping").json()
        stage("RBI SMA/EWS/RFA/CRILC mapping", len(rbi.get("ml_trigger_mapping", [])) >= 5)

        # 9. Drift report (Phase 10)
        drift = client.get("/api/drift/report").json()
        stage("drift report served in plain language", "plain_language" in drift,
              drift.get("plain_language", drift.get("status", ""))[:80])

        # 10. Stress test (Phase 11 / §12.7)
        st = client.post("/api/stress-test", json={"shock_type": "revenue", "magnitude": -0.15}).json()
        worse = st["delta_expected_loss_rate"] > 0
        stage("stress test: -15% revenue worsens book", worse,
              f"dEL rate {st['delta_expected_loss_rate']:+.6f}")

        # 11. Streaming: GST event moves a risk band, no manual re-score (DoD #5)
        target = client.get("/api/loans", params={"per_grade": 2}).json()
        mid = next(l["loan_id"] for l in target if l["risk_grade"] == "B")
        before_alerts = {a["id"] for a in client.get("/api/alerts").json()}
        client.post("/api/events/simulate", json={"loan_id": mid, "type": "gst_filing"})
        moved = None
        for _ in range(20):
            time.sleep(1)
            new = [a for a in client.get("/api/alerts").json()
                   if a["id"] not in before_alerts and a["loan_id"] == mid]
            if new:
                moved = new[0]
                break
        stage("streaming: GST event moves risk band (DoD #5)", moved is not None,
              moved["message"] if moved else "no alert within 20s")

        # restore the demo loan's online features from the offline store
        from src.serving import feature_store

        offline_row = features.loc[features["SK_ID_CURR"] == mid]
        if len(offline_row):
            clean = offline_row.iloc[0].drop(labels=["TARGET"], errors="ignore")
            feature_store.update_features(mid, json.loads(clean.to_json()))

    # 12. Thin-file scoring (optional; ~30s TabPFN CPU inference)
    if "--thin-file" in sys.argv:
        with TestClient(app) as client:
            ntc_id = int(ntc["SK_ID_CURR"])
            resp = client.post(f"/api/score/{ntc_id}").json()
            stage("thin-file: TabPFN single-point estimate",
                  resp.get("model_used") == "tabpfn_thin_file" and resp.get("point_estimate") is True,
                  f"pd={resp.get('calibrated_pd_12m')}")

    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{'=' * 60}\nIntegration dry run: {len(RESULTS) - len(failed)}/{len(RESULTS)} stages passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
