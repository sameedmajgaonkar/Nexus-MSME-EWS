# MVP Build Phases — MSME Credit Risk Predictive AI (Hackathon Showcase)

This is the **prioritized subset** of `PHASES.md` / `plan.md`, scoped for a solo or 2-person build with no fixed hour budget. Every feature is sorted into **Must-Have**, **Should-Have**, or **Cut**. Must-Have (P0–P5) alone is a complete, demoable, defensible entry. Should-Have (P6–P8) is added only after P0–P5 works end-to-end, in the order listed — each was chosen for being cheap relative to its pitch value.

## Priority Summary

| Tier | Phases | What it proves |
|---|---|---|
| **Must-Have (MVP)** | P0–P5 | Default is a *when*, not *if* (hazard curve); scores are calibrated and comparable across segments; every prediction is SHAP-explained; it's a naive-baseline-beating, live-servable system, not a notebook. |
| **Should-Have** | P6–P8 | The three-level explainability story is real (narrative+counterfactual); the supply-chain-network differentiator works (graph-lite); there's a second live "wow" moment (stress-test). |
| **Cut — roadmap slide only** | — | TabPFN thin-file specialist, text/FinBERT pipeline, Feast, MLflow/Evidently MLOps, fairness dashboard, Redpanda streaming, full FREE-AI governance UX, and all of `plan.md`'s original Tier-3 items (GNN, neural survival, federated learning, causal reasoning, cross-jurisdiction adaptation, conformal prediction). |

---

## Phase 0 — Environment + Primary Dataset (Must-Have)

**Goal:** One dataset loaded and validated, environment reproducible.

**Build:**
- Repo skeleton: `src/`, `notebooks/`, `data/`, `tests/`. `requirements.txt`: `pandas`, `scikit-learn`, `lightgbm`, `optbinning`, `shap`, `networkx`, `fastapi`, `streamlit`, `pytest`.
- **Primary dataset: Home Credit Default Risk** (Kaggle). Use `application_train.csv` + `bureau.csv` + `previous_application.csv` + `installments_payments.csv` — the last one gives real month-by-month repayment history, which is what makes the Phase 3 hazard model genuine rather than a synthesized time axis. No Freddie Mac, no German Credit for the MVP.
- **Resolve immediately:** Home Credit needs a Kaggle account + API token (`kaggle.json`). If unavailable, fall back to German Credit (UCI, no auth) and accept a weaker survival story — decide this now, not mid-build.

**Verify:** `application_train.csv` loads (~307K rows); `installments_payments.csv` loads and joins by `SK_ID_CURR`.

---

## Phase 1 — Segmentation + Core Structured Features (Must-Have)

**Goal:** Every loan tagged with a segment; a WOE-transformed feature table ready for modeling.

**Build:**
- Proxy segmentation (Home Credit has no real MSME loan-type field): `NAME_CONTRACT_TYPE` / `OCCUPATION_TYPE` / `ORGANIZATION_TYPE` stand in for loan-type/sector; a `data_richness` flag derived from bureau-history row-count stands in for Established vs. NTC/NTB. State this substitution explicitly in the pitch as "proxy segmentation, schema-swappable to real MSME fields."
- A small, strong feature set only: DPD history + payment-delay frequency (from `installments_payments.csv`), 2–3 ratios (credit-to-income, annuity-to-income), bureau aggregates (existing lines, prior defaults). Skip the exhaustive §8.1 list.
- WOE binning (`optbinning`) alongside raw values.

**Verify:** segment-distribution table printed and plausible; WOE bins monotonic in event rate for the top 3 features.

---

## Phase 2 — Baseline Scorecard + Metrics Harness (Must-Have)

**Goal:** First trained model, plus the shared evaluation utility every later phase reuses.

**Build:**
- `LogisticRegression` on WOE features.
- Metrics-table utility: naive "always no-default" baseline vs. trained model — AUC-ROC, KS, PR-AUC, Recall@FPR=10%, Brier. Build once, reuse everywhere.
- Out-of-time split by application date — never random k-fold.

**Verify:** metrics table prints both rows side by side; baseline clears naive AUC 0.50 by a real margin.

---

## Phase 3 — Discrete-Time Hazard Model (Must-Have, the flagship)

**Goal:** The model that actually proves "12 months in advance."

**Build (Option A only — skip Cox PH for MVP):**
- Expand `installments_payments.csv` into person-period format (one row per loan per month), binary `event` = missed/defaulted, right-censoring for loans that finish the window without defaulting.
- `months_since_origination` feature.
- LightGBM on the expanded panel; predict across 12 month-indices at inference to emit a hazard vector + cumulative PD.

**Verify:** unit test — a censored loan contributes multiple `event=0` rows, never one negative label; one sample loan prints a real 12-value hazard curve; metrics table extended, hazard model vs. Phase 2 baseline.

---

## Phase 4 — Calibration + Unified Grade + SHAP (Must-Have, the credibility centerpiece)

**Goal:** Comparable scores across segments, plus a top-5-driver explanation per prediction.

**Build:**
- Isotonic calibration per segment on a held-out split; map to a shared A–G+ grade.
- Risk-band-to-action lookup table.
- `shap.TreeExplainer` on the hazard model; top-5 feature/value pairs per prediction.

**Verify:** reliability diagram tracks the diagonal after calibration; unified-grade table reproduced across **≥2 segments** on the same scale; SHAP top-5 drivers printed for one sample loan.

---

## Phase 5 — Minimal Serving + Dashboard (Must-Have — MVP complete here)

**Goal:** A live, clickable demo, not a notebook.

**Build:**
- `POST /score/{loan_id}` (FastAPI) → calibrated PD, hazard curve, risk grade, SHAP top-5, recommended action. No Feast — compute features on the fly from an in-memory/parquet cache; name Feast as the production upgrade path in the pitch.
- One-page Streamlit dashboard: pick a loan → hazard-curve chart, grade, SHAP bar chart, metrics table.
- Governance stand-in: a visible "AI-assisted assessment" disclosure line + an Accept/Override button appending to a local SQLite table (not the full audit/incident-log system).

**Verify — MVP Definition of Done:**
1. Score a real loan, see a 12-month hazard curve (not a flag).
2. See the naive-vs-model metrics table live.
3. See SHAP top-5 drivers for that loan.
4. See the unified grade reproduced for 2 segments.
5. Click Override, confirm it's logged.

**If nothing else gets built, this alone is a complete, defensible entry.**

---

## Phase 6 — Narrative + Counterfactual Layer (Should-Have)

**Goal:** Turn SHAP numbers into the "why" — cheap (no retraining), makes the three-level explainability claim real.

**Build:**
- Fixed-structure LLM prompt fed only the top-5 SHAP pairs → one paragraph narrative.
- Guardrail: parse any number the LLM states, compare to actual SHAP values, fall back to a templated sentence on mismatch.
- Counterfactual: grid re-scoring over top 2–3 SHAP features to find the smallest change that flips the risk band.

**Verify:** guardrail test with a deliberately wrong number falls back correctly; counterfactual actually moves a sample loan to the next-better grade when applied.

---

## Phase 7 — Graph-lite Supply-Chain Features (Should-Have)

**Goal:** The flagship differentiator — no commercial Indian product publicly demonstrates this.

**Build:**
- Synthetic bipartite firm↔counterparty graph (`networkx`) with a deliberate mix of concentrated vs. diversified firms.
- Graph-lite features: counterparty concentration, degree centrality, anchor-linkage flag, network churn.
- Mandatory validation before wiring anywhere real: confirm the features separate concentrated from diversified synthetic firms.
- Concatenate onto the Phase 3 feature matrix, retrain, report the AUC/KS delta honestly.

**Verify:** separability validation passes; ablation table (with vs. without graph features) printed.

---

## Phase 8 — Stress-Test Simulator (Should-Have)

**Goal:** A second live "wow" moment, near-zero cost — pure re-inference, no retraining.

**Build:**
- One more endpoint/button: apply a shock (e.g. −15% revenue) to a slice of scored loans, re-run through the existing calibrated model, show before/after grade-distribution shift and ΔExpected Loss (`PD × LGD × EAD`).

**Verify:** running the shock produces a visibly worse post-shock distribution and higher ΔEL than baseline.

---

## The Honest Cut List (say this out loud in the pitch)

TabPFN thin-file specialist, text/FinBERT sentiment pipeline, Feast/MLflow/Evidently production MLOps, fairness dashboard, live Kafka/Redpanda streaming, full FREE-AI governance UX, and the original Tier-3 items (GNN, neural survival, federated learning, causal reasoning, cross-jurisdiction adaptation, conformal prediction) — each with its one-line "why not now" and "what library, next," already written out in `plan.md` §12.10. A named, deliberate cut list reads as engineering maturity to a banking judge, not as incompleteness.
