# Build Phases — MSME Credit Risk Predictive AI

This document turns `plan.md` (the full architectural blueprint) into a **sequential build order**: 13 phases, each producing one working, testable artifact before the next phase starts. Every phase cites the `plan.md` section(s) it implements — read that section for full rationale before building.

Tier 1 = Phases 0–4 (must-build core). Tier 2 = Phases 5–11 (differentiators). Phase 12 = integration. Tier 3 items (§12.10: full GNN, neural survival models, federated learning, causal reasoning, cross-jurisdiction adaptation, conformal prediction) are **never built** — they stay a roadmap slide only.

Total budget: ~42h (Tier 1 ~15h, Tier 2 ~23h, integration ~4h). If time runs short, compress or narrate-only Phases 9–11 first; never skip a Tier-1 phase's test.

---

## Phase 0 — Foundation & Data Setup (~2h)

**Goal:** A working environment with the proxy datasets loaded and validated.

**Plan sections:** §6.1, §6.2, §6.5, §16

**How to build it:**
1. Create the repo skeleton: `src/`, `notebooks/`, `data/raw/`, `data/processed/`, `tests/`, `docker-compose.yml`.
2. Write `requirements.txt` (or `pyproject.toml`) pinning at least: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `optbinning`, `lifelines`, `shap`, `networkx`, `sentence-transformers`, `transformers`, `great-expectations`, `pytest`.
3. Download/ingest the datasets from §6.2:
   - Home Credit Default Risk (Kaggle — needs a Kaggle API token in `~/.kaggle/kaggle.json`).
   - Freddie Mac Single-Family Loan-Level Dataset (needs a free Freddie Mac account).
   - Statlog German Credit Data (UCI — no auth needed, direct download).
   - HMEQ (Kaggle mirror or direct CSV).
   - Financial PhraseBank (`huggingface.co/datasets/financial_phrasebank`).
   - If Kaggle/Freddie Mac credentials aren't available yet, start with German Credit + HMEQ (no-auth) and stub loaders for the rest so the schema is ready when credentials arrive.
4. Add a `data_provenance` column to every loaded table: `real_sandbox` / `public_proxy` / `synthetic_graph` (§6.5) — every row must be tagged from the moment it's loaded.
5. Write one Great-Expectations-style check per table: null-rate threshold, range validation on key ratios, duplicate-row detection.

**How to verify it's done:**
- `docker-compose config` validates without error.
- `pip install -r requirements.txt` completes cleanly in a fresh virtualenv.
- Each dataset loads into a DataFrame with the expected shape (e.g. Home Credit `application_train.csv` → 307,511 rows).
- The data-quality check runs against at least one table and passes (or fails loudly with a clear reason — never silently).

---

## Phase 1 — Segmentation + Structured Feature Pipeline (~3h)

**Goal:** Every loan record tagged with a segment, and a structured feature table ready for modeling.

**Plan sections:** §7.2, §7.3, §8.1

**How to build it:**
1. Implement the segmentation decision tree from §7.2 as a pure function: `loan_type` (Personal/Home/Auto/MSME) → for MSME, branch on `has_12mo_history` (Established vs NTC/NTB) → for Established MSME, branch on `sector` (Manufacturing/Retail/Services/Agriculture). Output columns: `loan_type`, `data_richness`.
2. Build the structured features from §8.1 per segment-weighting table in §7.3: DPD history, EMI delay frequency, SMA-0/1/2 counts, debt-to-income, utilization ratio, liquidity ratio, rolling 3/6/12-month cash-flow average and coefficient of variation, bureau fields (existing lines, enquiry count, prior defaults), GST on-time filing rate, input-output tax ratio, UPI/transaction velocity, counterparty diversity.
3. Apply coarse binning (5–10 bins) + WOE transform (`optbinning`) to every candidate feature, computed **alongside** the raw value (not replacing it) — the logistic baseline will consume WOE, the hazard model will consume raw values.
4. Treat missingness as its own bin, not an imputed/dropped value (§6.4 data quality plan).

**How to verify it's done:**
- Unit test: feed 4–5 hand-crafted records (one per segment) through the segmentation function and assert each lands in the expected segment.
- For each WOE-binned feature, assert the bins are monotonic in event rate (a broken WOE bin is a red flag before any model is trained).
- Print a segment-distribution table (row counts per segment) and sanity-check it against expectations (e.g. MSME-NTC should be a minority of rows).

---

## Phase 2 — Baseline WOE-Logistic Scorecard (~2h)

**Goal:** The first trainable model, plus the evaluation harness every later model reuses.

**Plan sections:** §9.1, §2.1, §15.2, §15.3

**How to build it:**
1. Train `sklearn.LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', max_iter=1000)` on WOE-transformed structured features only (no graph/text) from Phase 1, joined against `bureau.csv`.
2. Build the **out-of-time (OOT) split** now, once, as a shared utility: train on data up to time T, validate T+1..T+k, test on a final held-out future window. Random k-fold is banned project-wide (§15.2) — every later phase's model reuses this split function.
3. Build the metrics-table utility (§15.3): AUC-ROC, KS-statistic, PR-AUC, Recall@FPR=10%, Brier score — computed for any model + labels, always printed next to the naive "always predict no-default" baseline.

**How to verify it's done:**
- Confirm the split function partitions strictly by time, not randomly (write a test that asserts no validation-window row has a date earlier than the latest training-window row).
- Run the metrics table for the naive baseline and the trained scorecard side by side; scorecard AUC clears 0.50 by a meaningful margin.
- This metrics-table + OOT-split utility becomes the one place every future phase reports numbers from — don't reimplement it per phase.

---

## Phase 3 — Discrete-Time Hazard Model (core engine) (~5h)

**Goal:** The model that actually answers "12 months in advance" — a 12-value hazard curve per loan, not a single flag.

**Plan sections:** §9.2 (both options), §2.2

**How to build it:**
1. Expand loan-level data into person-period format: one row per loan per month, with a binary `event` column (did default happen in *this* month).
2. Right-censor correctly: a loan that survives to month 12 (or to the end of the observed window) without defaulting contributes rows with `event=0` for every month it survived — it is never labeled as a single "safe" row.
3. Add `months_since_origination` as a feature — this is the feature that lets the model learn how risk changes over a loan's life.
4. Train LightGBM (`num_leaves=31, learning_rate=0.05, n_estimators=1000, early_stopping_rounds=50, scale_pos_weight=<neg:pos ratio>, objective='binary', metric='auc'`) on the expanded panel.
5. At inference, predict across all 12 month-indices for a given loan to produce the 12-length hazard vector + cumulative PD.
6. Optionally also fit Cox PH via `lifelines` (`penalizer=0.1`, `strata='loan_type'`) as a second, independent validation angle (concordance index) — cheap to add, and it's the more "classically correct" survival model for judges/stakeholders who ask about it.

**How to verify it's done:**
- Unit test: construct a loan censored at month 8; assert it produces exactly 8 person-period rows, all with `event=0` — not a single negative-labeled row.
- For one sample loan, print the 12-value hazard curve and cumulative PD (not a single 0/1 flag).
- Extend the Phase 2 metrics table with this model's AUC/KS/Recall@FPR; it should beat the Phase 2 baseline (if it doesn't, that's a real finding to report, not something to hide).
- **This satisfies Definition-of-Done item #1 (§19.4):** a working hazard curve, end-to-end from raw data to output.

---

## Phase 4 — Calibration + Unified Risk Grade + SHAP (~3h)

**Goal:** Scores from different segments become comparable, and every prediction gets a top-5 driver explanation.

**Plan sections:** §9.4, §9.7, §11.2 (Level 1 only)

**How to build it:**
1. Fit `sklearn.isotonic.IsotonicRegression` **per segment**, on a held-out calibration split that was used in neither training nor validation.
2. Map the calibrated PD onto a shared 10–13-band grade (A→G+, CRIF-Rank-analogous per §9.4).
3. Implement the risk-band-to-action table from §9.7 (0–20% → monitor, 20–50% → watch list, 50–80% → enhanced monitoring, 80%+ → immediate intervention) as a pure lookup function.
4. Wire up SHAP: `shap.TreeExplainer` for the LightGBM hazard model, `shap.LinearExplainer` for the logistic baseline. Extract top-5 feature/value pairs per prediction.

**How to verify it's done:**
- Plot a reliability diagram (binned predicted PD vs. observed default rate) — it should track the diagonal after calibration, visibly better than before.
- Reproduce the unified-grade table (§12.8 style) for **at least two different segments** landing on the same A–G+ scale — **this satisfies Definition-of-Done item #4**.
- Print SHAP top-5 drivers for one sample loan — **partially satisfies Definition-of-Done item #3** (narrative/counterfactual come in Phase 7).

**Tier 1 is complete here.** At this checkpoint there should be a real, calibrated, explainable, segment-aware hazard model that beats a naive baseline. Confirm all of Phases 0–4's exit criteria pass together before moving to Tier 2.

---

## Phase 5 — Graph-lite + Text Feature Enrichment (~4h)

**Goal:** Add supply-chain-network signal and unstructured-text signal, and prove they help (or report honestly that they don't).

**Plan sections:** §12.1 (Stages 1–3 only — Stage 4/GAT is Tier 3, never built), §8.2, §8.3, §9.5

**How to build it:**
1. Generate a synthetic bipartite firm↔counterparty graph with `networkx`, deliberately including some firms with concentrated (1–2 counterparties) and some with diversified trading patterns.
2. Compute graph-lite tabular features: counterparty concentration (share of value from top counterparty), degree centrality, anchor-linkage flag (is the top counterparty itself large/stable?), network stability (month-over-month counterparty churn).
3. **Before wiring to any real data**, validate that these features actually separate the concentrated vs. diversified synthetic firms — this validation step is explicitly called out in the plan as non-optional.
4. Build the text pipeline: `sentence-transformers` (`all-MiniLM-L6-v2`) embeddings on free-text fields → `ProsusAI/finbert` sentiment classification → PCA to 10–20 components → a regex-based distress-keyword flag (words like "delay," "dispute," "stoppage," "closure").
5. Concatenate both feature families as **dense vectors** (not scalars) onto the Phase 3 structured matrix, and retrain the hazard model (§9.5's "hackathon-realistic" fusion — the learned cross-attention layer is Tier 3 and is not built).

**How to verify it's done:**
- The synthetic-graph validation step passes: concentrated firms score measurably differently from diversified firms using the graph features alone.
- FinBERT produces the expected sentiment label on 2–3 hand-picked, obviously positive/negative sentences.
- Report the retrained (fused) model's AUC/KS against the Phase 4 model as an explicit ablation — state the uplift number honestly, even if it's small or negative.

---

## Phase 6 — TabPFN Thin-File Specialist (~2h)

**Goal:** A purpose-built model for the New-to-Credit/New-to-Bank segment, validated against the alternative (LightGBM on the same tiny sample).

**Plan sections:** §9.3, §12.9

**How to build it:**
1. Install `tabpfn` (local, CPU-only under ~1,000 rows) or `tabpfn_client` (hosted, no local GPU needed — note the data-privacy caveat in §9.3 if this is ever pointed at real sandbox data).
2. Train/run TabPFN on German Credit Data (matches TabPFN's own published benchmark) and on a 200–500-row NTC/NTB subsample of Home Credit.
3. Train a LightGBM model on the **exact same** subsample, for the side-by-side comparison mandated by §12.9 — this comparison is what makes the differentiator credible instead of a novelty pick.
4. Wrap explainability with `tabpfn-extensions`' interpretability module so SHAP output stays consistent in format with Phase 4's SHAP pipeline.
5. Update the Phase 1 segment router: any loan tagged `data_richness=NTC/NTB` routes to TabPFN instead of the hazard model.

**How to verify it's done:**
- Print a side-by-side AUC/KS table: TabPFN vs. LightGBM, both on the identical thin-file subsample.
- Unit test: a synthetic NTC/NTB loan record, run through the router, dispatches to TabPFN — not the hazard model.

---

## Phase 7 — Narrative, Counterfactual & Uncertainty Layer (~3h)

**Goal:** Turn SHAP numbers into a trustworthy plain-language explanation, plus a concrete "what would fix this," plus a confidence signal.

**Plan sections:** §11.2 (Levels 2–3), §12.3, §9.6

**How to build it:**
1. Write a **fixed-structure** LLM prompt template that receives *only* the top-5 SHAP feature/value pairs (never the full case file) and returns one short paragraph: driver → direction → magnitude → risk implication.
2. Add the guardrail: after the LLM responds, parse any stated percentage/number out of its text and compare it against the actual SHAP values. If they don't match, discard the LLM output and fall back to a templated (non-LLM) sentence instead of showing an unverified number.
3. Cache narratives per (risk-band, top-3-driver-combination) tuple so a live demo doesn't re-call the LLM for identical explanation patterns (latency budget note, §12.3).
4. Build the counterfactual generator: a grid re-scoring of the trained model over small perturbations to the top 2–3 SHAP-flagged features, reporting the smallest realistic change that moves the account to the next-better risk band.
5. Build the confidence band: use calibration-set residuals to attach a `± X%` band to every prediction; flag for mandatory human review when the band is wide or when TabPFN and the hazard model disagree beyond a set margin (only relevant where both scored the same case, e.g. during the Phase 6 validation comparison).

**How to verify it's done:**
- Guardrail test: deliberately feed the guardrail a narrative with a wrong number; confirm it falls back to the templated sentence rather than displaying the wrong number.
- Counterfactual test: for one sample high-risk loan, apply the suggested feature change and confirm the model's output actually moves to the next-better grade.
- **This satisfies Definition-of-Done item #3 in full:** SHAP → narrative → counterfactual, end-to-end, for one live prediction.

---

## Phase 8 — Serving API + Feature Store + Audit Log (~4h)

**Goal:** Everything built so far becomes callable over HTTP, with features served consistently and every call logged immutably.

**Plan sections:** §17, §13.1, §12.5 (audit-log portion only)

**How to build it:**
1. Stand up `Feast` with an offline store (historical features from Phases 1/5/6, for training) and an online store (Redis or SQLite, for low-latency serving) — Feast's point-in-time-correct joins reinforce the no-leakage discipline from Phase 2.
2. Build FastAPI endpoints, starting with the three most load-bearing ones from §17's table:
   - `POST /score/{loan_id}` — runs the full pipeline (route by segment → score → calibrate → explain), returns the six-part structure from §11.3.
   - `GET /explain/{loan_id}` — retrieves the already-computed explanation without re-scoring.
   - `POST /override/{loan_id}` — records accept/modify/override; `reason` is a mandatory field on override.
3. Stand up PostgreSQL with an append-only audit table; every `/score` call and every `/override` call writes an immutable row.

**How to verify it's done:**
- `POST /score/{loan_id}` on a sample loan returns all six fields from §11.3: calibrated PD, 12-value hazard curve, risk grade, top-5 drivers, counterfactual, recommended action.
- `POST /override/{loan_id}` without a `reason` is rejected (4xx); with a reason, a new audit-log row appears.
- Re-calling `/score` for the same loan without new upstream data returns features from the **online** Feast store, not a recomputation from raw tables (confirm via logging/timing, not just correctness).

---

## Phase 9 — Dashboard + Governance UX (~4h)

**Goal:** The risk-officer console and the governance surfaces (disclosure, override, fairness) that turn this from a model into a product.

**Plan sections:** §18, §12.4, §12.5, §12.6

**How to build it:**
1. Build the Streamlit risk-officer console matching the §18.1 wireframe: portfolio summary panel + per-borrower detail panel (risk grade, hazard curve, top drivers, narrative, counterfactual, recommended action, Accept/Modify/Override buttons), wired to the Phase 8 API.
2. Add the always-visible AI-disclosure banner (never a settings-page toggle — §18.3 makes this explicit).
3. Build the minimal borrower-facing view (§18.2): disclosure banner + plain-language risk band + counterfactual only — never raw SHAP numbers or model internals.
4. Add the RBI SMA/EWS/RFA/CRILC mapping table and state diagram (§12.4) as a read-only panel showing which regulatory indicator each ML-derived trigger maps to.
5. Add the fairness dashboard (§12.6): group-by aggregation of approval rate / average calibrated PD / false-positive rate, sliced by sector, region, and gender-of-promoter.

**How to verify it's done:**
- Manual walkthrough: score a loan through the dashboard UI and confirm hazard curve, SHAP drivers, narrative, and counterfactual render exactly as in the §18.1 wireframe.
- Attempt an override with no reason in the UI — it's blocked; with a reason, the audit log entry appears (cross-check against the Phase 8 audit table).
- Fairness dashboard renders at least one real disparity number (e.g. average PD by sector) computed from actual scored loans, not placeholder data.

---

## Phase 10 — MLOps: Tracking, Drift Monitoring, Retrain Loop (~3h)

**Goal:** Every model version is tracked, drift is visible in plain language, and a bad new model can't silently replace a good one.

**Plan sections:** §13.2, §13.3, §13.4

**How to build it:**
1. Wrap every training run since Phase 2 (baseline, hazard model, TabPFN) with `MLflow` tracking: log hyperparameters, metrics (AUC/KS/Recall@FPR/Brier/PSI), and the model artifact.
2. Register promoted models in the MLflow Model Registry with `Staging` → `Production` stage transitions.
3. Wire `Evidently AI` to generate data-drift, prediction-drift, and calibration reports comparing a rolling "current" window against the training-time reference distribution — render the summary in plain language on the Phase 9 dashboard (not a raw statistical dump).
4. Implement the retrain-trigger flowchart from §13.2: scheduled monthly retrain by default, plus an out-of-cycle trigger when PSI crosses a threshold on any Tier-1 feature or calibration deviation exceeds tolerance. On retrain, compare the new model against current Production on the OOT set before promoting — never auto-promote a worse model.
5. Orchestrate with a simple cron job or `Prefect` flow (Airflow is the named production upgrade path, not built now).

**How to verify it's done:**
- Confirm every model trained in Phases 2/3/5/6 shows up in the MLflow registry with its logged metrics.
- Feed a deliberately shifted feature distribution into Evidently and confirm it raises a PSI-above-threshold flag, which fires the retrain job.
- Retrain with a deliberately worse model and confirm the promotion gate holds it back rather than promoting it.

---

## Phase 11 — Stress-Test Simulator + Streaming Demo (~3h)

**Goal:** Two "wow" demo moments — a portfolio-level what-if tool, and a live event moving a risk band on screen.

**Plan sections:** §12.7, §14

**How to build it:**
1. Build `POST /stress-test`: re-score a filtered slice of the book after systematically perturbing a chosen feature (revenue shock, rate shock, sector demand shock) through the already-trained calibrated model — pure re-inference, no retraining. Report before/after risk-grade distribution and Δ Expected Loss (`EL = PD × LGD × EAD`).
2. Stand up `Redpanda` (Kafka-API-compatible, single Docker container, no JVM/Zookeeper) as the event bus.
3. Build a feature-refresh consumer service that listens for simulated events (GST filing, bank transaction, repayment), updates the Feast online store, and triggers a re-score of the affected loan.
4. Wire a dashboard alert: if the re-score moves the loan's risk band, push a visible alert to the Phase 9 dashboard; if not, update the risk-trajectory timeline silently.

**How to verify it's done:**
- Run the stress-test with a −15% revenue shock and confirm the post-shock grade distribution is visibly worse, with a higher ΔEL, than the baseline.
- **Definition-of-Done item #5:** simulate a GST-drop event on the bus and confirm one borrower's risk band visibly changes on the dashboard within seconds, with no manual re-scoring step triggered by a human.

---

## Phase 12 — Integration, DoD Verification & Pitch Prep (~4h)

**Goal:** Prove the whole system works end-to-end, live, in one pass — and package the honest Tier-3 roadmap.

**Plan sections:** §19.4, §20, §12.10

**How to build it:**
1. Run one complete dry run of the entire chain: raw data → segmentation → features → scoring → calibration → explanation → dashboard → audit log → drift monitor → stress test → streaming alert. Fix whatever breaks under the full chain (integration bugs that don't show up when phases are tested in isolation are expected here — budget time for this).
2. Prepare the Tier-3 roadmap slide (§12.10 table: full GNN, neural survival models, federated learning, causal-uplift reasoning, cross-jurisdiction adaptation, conformal prediction) — each with its named library and honest reason it's out of scope, framed as "what's next," never as "half-built."
3. Rehearse the demo script from §20.1: reframe → metrics table → live demo (score a loan → SHAP/narrative/counterfactual → streaming event → unified cross-segment table → stress test) → governance slide → roadmap close.

**How to verify it's done — run the full §19.4 checklist live, in order:**
1. Working segment-specific hazard curve for a real account, raw data → output. (built in Phase 3)
2. Naive-baseline vs. built-model comparison on AUC/KS/Recall@FPR, not accuracy. (Phase 2/3)
3. Live SHAP → narrative → counterfactual chain for one prediction. (Phase 7)
4. Calibration reliability diagram + unified grade across ≥2 segments. (Phase 4)
5. Live streaming demo moving a risk band. (Phase 11)
6. One clear slide on the Tier-3 roadmap.

If time runs out before this phase, the honest fallback order to cut is: Phase 11 (narrate the streaming demo instead of running it live) → Phase 10 (mention MLOps, don't demo the retrain loop) → Phase 9 (show a static screenshot instead of a live dashboard walkthrough) — never cut Phases 0–4 or Phase 7.
