"""Phase 8 + Phase 11 serving-API tests (PHASES.md verify items).

All DB paths are redirected to a pytest tmp dir via EWS_DB_PATH /
ONLINE_STORE_DB before the app starts, so tests never pollute data/.
ANTHROPIC_API_KEY is stripped so narratives are the deterministic template.
"""

import json
import os
import sqlite3
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

_ENV_KEYS = (
    "EWS_DB_PATH",
    "ONLINE_STORE_DB",
    "ANTHROPIC_API_KEY",
    "KAFKA_BOOTSTRAP_SERVERS",
    "DATABASE_URL",
)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("phase8")
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    os.environ["EWS_DB_PATH"] = str(tmp / "overrides.db")
    os.environ["ONLINE_STORE_DB"] = str(tmp / "online.db")
    for k in ("ANTHROPIC_API_KEY", "KAFKA_BOOTSTRAP_SERVERS", "DATABASE_URL"):
        os.environ.pop(k, None)

    from src.serving.app import app

    with TestClient(app) as c:
        yield c

    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _state():
    import src.serving.app as appmod

    return appmod.STATE


def _established_loans(grade: str) -> list[int]:
    sj = _state()["scored_join"]
    sel = sj[(sj["data_richness"] == "established") & (sj["risk_grade"] == grade)]
    return [int(i) for i in sel["SK_ID_CURR"]]


def _poll(fn, timeout=20.0, interval=0.25):
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = fn()
        if out:
            return out
        time.sleep(interval)
    return None


SIX_PART_KEYS = (
    "calibrated_pd_12m",  # 1
    "hazard_curve",       # 2
    "risk_grade",         # 3
    "top_drivers",        # 4
    "counterfactual",     # 5
    "recommended_action", # 6
)


def test_score_established_six_parts_and_audit(client):
    loan_id = _established_loans("C")[0]
    r = client.post(f"/api/score/{loan_id}")
    assert r.status_code == 200
    body = r.json()

    for key in SIX_PART_KEYS:
        assert key in body, f"missing §11.3 part: {key}"
    assert body["model_used"] == "hazard"
    assert body["point_estimate"] is False
    assert isinstance(body["hazard_curve"], list) and len(body["hazard_curve"]) == 12
    assert 0.0 <= body["calibrated_pd_12m"] <= 1.0
    assert body["risk_grade"] in "ABCDEFG"
    assert 1 <= len(body["top_drivers"]) <= 5
    assert {"feature", "label", "value", "shap", "direction"} <= set(body["top_drivers"][0])
    assert {"text", "source", "verified"} <= set(body["narrative"])
    assert body["narrative"]["source"] == "template"  # no API key in tests
    assert {"half_width", "n_comparable", "wide_band_flag"} <= set(body["confidence_band"])
    assert "disclosure" in body

    # audit_log row written, append-only content intact
    with sqlite3.connect(os.environ["EWS_DB_PATH"]) as conn:
        rows = conn.execute(
            "SELECT endpoint, model_used, risk_grade, narrative_source FROM audit_log "
            "WHERE loan_id = ?",
            (loan_id,),
        ).fetchall()
    assert len(rows) >= 1
    assert rows[0][0] == f"/api/score/{loan_id}"
    assert rows[0][1] == "hazard"

    # /explain returns the cached explanation without re-scoring
    e = client.get(f"/api/explain/{loan_id}")
    assert e.status_code == 200
    assert e.json()["risk_grade"] == body["risk_grade"]
    missing = client.get("/api/explain/999999999")
    assert missing.status_code == 404
    assert "hint" in missing.json()["detail"]


def test_second_score_reads_online_store(client):
    """Phase 8 verify item: re-scoring serves features from the ONLINE store —
    asserted via the store's source counters, not timing."""
    from src.serving import feature_store

    loan_id = _established_loans("C")[0]
    feature_store.reset_source_counts()
    assert client.post(f"/api/score/{loan_id}").status_code == 200
    assert client.post(f"/api/score/{loan_id}").status_code == 200
    counts = feature_store.get_source_counts()
    assert counts["online"] >= 2
    assert counts["offline_fallback"] == 0


def test_thin_file_routing_monkeypatched(client):
    """Router dispatch for ntc_ntb, deterministic via a stub model (fast path)."""
    state = _state()
    feats = state["features"]
    ntc_id = int(feats[feats["data_richness"] == "ntc_ntb"]["SK_ID_CURR"].iloc[0])

    class StubModel:
        def predict_proba(self, X):
            return np.tile([0.8, 0.2], (len(X), 1))

    original = state["thin_file"]
    state["thin_file"] = {"model": StubModel(), "columns": original["columns"]}
    try:
        r = client.post(f"/api/score/{ntc_id}")
    finally:
        state["thin_file"] = original

    assert r.status_code == 200
    body = r.json()
    assert body["model_used"] == "tabpfn_thin_file"
    assert body["point_estimate"] is True
    assert body["hazard_curve"] is None
    assert body["counterfactual"]["feature"] is None
    for key in SIX_PART_KEYS:
        assert key in body


def test_thin_file_real_tabpfn(client):
    """Real TabPFN inference (~30s CPU: ~8s predict + ~22s permutation SHAP —
    measured under the 60s limit, so the real test stays)."""
    feats = _state()["features"]
    ntc_id = int(feats[feats["data_richness"] == "ntc_ntb"]["SK_ID_CURR"].iloc[1])
    r = client.post(f"/api/score/{ntc_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["model_used"] == "tabpfn_thin_file"
    assert body["point_estimate"] is True
    assert body["hazard_curve"] is None
    assert 0.0 <= body["calibrated_pd_12m"] <= 1.0
    assert 1 <= len(body["top_drivers"]) <= 5


def test_override_rules_and_append_only(client):
    loan_id = _established_loans("D")[0]

    assert client.post(f"/api/override/{loan_id}", json={"decision": "override"}).status_code == 422
    assert client.post(f"/api/override/{loan_id}", json={"decision": "modify"}).status_code == 422
    assert client.post(f"/api/override/{loan_id}", json={"decision": "bogus"}).status_code == 422
    assert client.post(f"/api/override/{loan_id}", json={"decision": "accept"}).status_code == 200

    r = client.post(
        f"/api/override/{loan_id}",
        json={
            "decision": "modify",
            "reason": "collateral revaluation received",
            "delta": {"risk_grade": "D -> C"},
        },
    )
    assert r.status_code == 200
    r2 = client.post(
        f"/api/override/{loan_id}",
        json={"decision": "override", "reason": "site visit contradicts bureau data"},
    )
    assert r2.status_code == 200

    listed = client.get("/api/overrides").json()
    ours = [d for d in listed if d["loan_id"] == loan_id]
    assert {d["decision"] for d in ours} == {"accept", "modify", "override"}
    modify_row = next(d for d in ours if d["decision"] == "modify")
    assert json.loads(modify_row["delta"]) == {"risk_grade": "D -> C"}

    # APPEND-ONLY enforcement: direct UPDATE/DELETE must abort via trigger.
    with sqlite3.connect(os.environ["EWS_DB_PATH"]) as conn:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute("UPDATE overrides SET reason = 'tampered'")
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute("DELETE FROM overrides")
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute("UPDATE audit_log SET risk_grade = 'A'")


def test_stress_test_revenue_shock(client):
    r = client.post("/api/stress-test", json={"shock_type": "revenue", "magnitude": -0.15})
    assert r.status_code == 200
    body = r.json()
    pre, post = body["pre_shock_distribution"], body["post_shock_distribution"]
    assert set(pre) == set("ABCDEFG")
    # visibly worse: fewer top grades, more E-or-worse, positive delta EL rate
    assert post["A"] + post["B"] < pre["A"] + pre["B"]
    assert sum(post[g] for g in "EFG") >= sum(pre[g] for g in "EFG")
    assert body["delta_expected_loss_rate"] > 0
    assert body["el_convention"]["lgd"] == 0.45

    assert client.post(
        "/api/stress-test", json={"shock_type": "bogus", "magnitude": -0.15}
    ).status_code == 422
    assert client.post(
        "/api/stress-test", json={"shock_type": "sector_demand", "magnitude": -0.2}
    ).status_code == 422  # sector required


def test_event_big_distress_moves_grade_and_alerts(client):
    """DoD item #5 (backend half): distress event -> bus -> consumer -> online
    store refresh -> re-score -> grade moves -> alert row, within seconds."""
    loan_id = _established_loans("B")[1]
    r = client.post(
        "/api/events/simulate",
        json={
            "loan_id": loan_id,
            "type": "gst_filing",
            "feature_updates": {
                "ext_source_1": 0.01,
                "ext_source_2": 0.01,
                "ext_source_3": 0.01,
                "credit_income_ratio": 20.0,
                "annuity_income_ratio": 2.0,
                "bureau_overdue_flag": 1.0,
                "prior_late_rate": 1.0,
            },
        },
    )
    assert r.status_code == 200
    assert r.json()["published"] is True
    event_id = r.json()["event_id"]
    assert event_id

    def find_alert():
        alerts = client.get("/api/alerts?limit=20").json()
        return [a for a in alerts if a["loan_id"] == loan_id]

    found = _poll(find_alert)
    assert found, "no alert appeared within the polling window"
    alert = found[0]
    assert alert["event_type"] == "gst_filing"
    assert alert["old_grade"] != alert["new_grade"]
    assert "ABCDEFG".index(alert["new_grade"]) > "ABCDEFG".index(alert["old_grade"])

    # trajectory row exists too
    tl = client.get(f"/api/timeline/{loan_id}").json()
    assert len(tl) >= 1


def test_event_tiny_update_silent_timeline_no_alert(client):
    loan_id = _established_loans("B")[2]
    from src.serving import feature_store

    current = feature_store.get_online_features(loan_id)["credit_income_ratio"]
    r = client.post(
        "/api/events/simulate",
        json={
            "loan_id": loan_id,
            "type": "repayment",
            "feature_updates": {"credit_income_ratio": current * 1.0001},
        },
    )
    assert r.status_code == 200

    timeline = _poll(lambda: client.get(f"/api/timeline/{loan_id}").json())
    assert timeline, "no timeline row appeared within the polling window"
    assert timeline[0]["grade"] in "ABCDEFG"

    alerts = client.get("/api/alerts?limit=50").json()
    assert not [a for a in alerts if a["loan_id"] == loan_id]


def test_fairness_audit_and_rbi_mapping(client):
    f = client.get("/api/fairness/audit").json()
    assert f["n_loans"] > 0
    assert set(f["dimensions"]) == {"sector_segment", "loan_type_segment", "data_richness"}
    sector_rows = f["dimensions"]["sector_segment"]
    assert len(sector_rows) >= 2
    for row in sector_rows:
        assert row["n"] > 0
        assert isinstance(row["avg_calibrated_pd"], float)
        assert row["disparity_ratio"] >= 1.0
    assert any(r["disparity_ratio"] > 1.0 for r in sector_rows)  # >=1 real disparity number
    assert "gender_of_promoter" in f["unavailable_slices"]

    m = client.get("/api/rbi/mapping").json()
    assert len(m["ml_trigger_mapping"]) >= 5
    assert {"ml_trigger", "rbi_indicator", "regulatory_action"} <= set(m["ml_trigger_mapping"][0])
    assert m["state_machine"]["states"]
    assert m["state_machine"]["transitions"]


def test_portfolio_summary_and_drift_placeholder(client):
    p = client.get("/api/portfolio/summary").json()
    assert p["n_loans"] > 0
    assert set(p["grade_distribution"]) == set("ABCDEFG")
    assert 0.0 < p["expected_loss_rate"] < 1.0
    assert p["el_convention"]["lgd"] == 0.45
    assert "loan_type_segment" in p["counts_by_segment"]

    d = client.get("/api/drift/report").json()
    assert d.get("status") == "not_yet_generated" or "psi" in json.dumps(d).lower()
