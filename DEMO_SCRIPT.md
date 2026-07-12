# Demo Script & Definition-of-Done Checklist

Pitch arc per plan.md §20.1 (8–10 min), with the exact commands/clicks per step.
Run `uv run uvicorn src.serving.app:app --port 8000` and `cd frontend && npm run dev` first
(or serve the built SPA straight from FastAPI via `frontend/dist`).

## 1. Open with the reframe (60s)
"16–22% accuracy sounds like a modeling problem. It isn't. We treat default as *when*, not *if* —
a 12-month hazard curve, calibrated onto one scale across segments, explained in language a bank
examiner would accept."

## 2. Honest metric reframe (60s)
Dashboard → Console → metrics table (GET /api/metrics): naive always-no-default baseline
next to WOE-logistic (AUC 0.739), hazard model, and fused ablation — AUC/KS/Recall@FPR10/Brier,
never accuracy.

## 3. Live demo (4–5 min, in this order)
1. **Score a loan** — pick a mid-grade borrower in the Console: 12-value hazard curve (not a flag),
   grade, recommended action.
2. **Explainability chain** — same screen: SHAP top-5 → guardrail-verified narrative →
   counterfactual ("if ext_source_1 moved 0.15→0.32, PD 7.1%→6.7%, grade E→D") →
   confidence band ("PD 7% ± 1%, n=2,832 comparable").
3. **Streaming moment (DoD #5)** — borrower detail → Simulate event → `gst_filing`:
   watch the alert banner fire within seconds as the grade moves (evidence run: B→G) with
   no manual re-scoring.
4. **Unified cross-segment table (DoD #4)** — grades A–G on one scale across
   term-loan/working-capital segments (per-segment isotonic calibration), plus the TabPFN
   thin-file route (`model_used` chip) landing on the same grade scale.
5. **Stress test** — Stress Test tab: −15% revenue shock → pre/post grade distribution shift
   + ΔExpected-Loss rate (unit-EAD, LGD 0.45 convention stated on screen).
6. **Governance** — Fairness tab (real disparity: Agriculture_Allied FPR 0.078 vs Retail_Trade
   0.027), RBI SMA/EWS/RFA/CRILC mapping tab, always-on AI disclosure, Accept/Modify/Override
   with mandatory reason (SQLite-trigger-enforced append-only audit), drift status chip
   (Evidently + PSI, plain language).

## 4. Governance slide (60s)
FREE-AI Sutra → shipped feature mapping (plan.md §12.5 table). Every /score call writes an
immutable audit row; overrides cannot be updated or deleted (enforced, not conventional).

## 5. Close with the honest roadmap (60s)
ROADMAP.md — Tier-3 items named with real libraries; synthetic-data provenance stated out loud.

---

## Definition-of-Done checklist (§19.4) — verified live in order

| # | Item | Where | Status |
|---|---|---|---|
| 1 | Segment-specific 12-month hazard curve, raw data → output | Phase 3 model, `/api/score/{id}` | ✅ |
| 2 | Naive vs built model on AUC/KS/Recall@FPR (not accuracy) | `/api/metrics`, models/phase*_metrics.json | ✅ |
| 3 | Live SHAP → narrative → counterfactual chain | Phase 7, `scripts/run_phase7.py` + Console | ✅ |
| 4 | Calibration reliability + unified grade across ≥2 segments | Phase 4 + `/api/portfolio/summary` | ✅ |
| 5 | Live streaming demo moving a risk band | `/api/events/simulate` → alert (B→G in ~4s) | ✅ |
| 6 | One-slide Tier-3 roadmap | ROADMAP.md | ✅ |

Integration dry-run: `uv run python scripts/run_all.py` executes the full chain
(quality gate → enrichment → routing → scoring → explanation → audit → drift → stress test →
streaming alert) and prints a PASS/FAIL line per stage.
