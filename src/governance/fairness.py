"""Fairness & bias audit: disparate impact over real scored loans (plan.md §12.6).

Group-by aggregation over the Phase 4 OOT scored output joined with the
enriched serving features, sliced by the dimensions the Home Credit proxy
actually carries: sector_segment, loan_type_segment, data_richness. Per group:
n, average calibrated PD, high-risk share (grade E or worse), false-positive
rate (flagged high-risk but no observed default among outcome-evaluable
loans), and a disparity ratio vs the best (lowest-average-PD) group.

Honest scope note (plan.md §12.6): the FREE-AI fairness lens also names
region/geography and gender-of-promoter slices. Neither attribute exists in
the Home Credit proxy snapshot, so those slices are NOT computed here — they
activate unchanged (same group-by code path) once real sandbox data with
those columns arrives. The payload carries this note verbatim.
"""

from pathlib import Path

import pandas as pd

from src.models.hazard import MAX_MONTHS

ROOT = Path(__file__).resolve().parents[2]
SCORED_PATH = ROOT / "data" / "processed" / "phase4_scored_test.parquet"
FEATURES_PATH = ROOT / "data" / "processed" / "serving_features_enriched.parquet"

HIGH_RISK_GRADES = ("E", "F", "G")
DIMENSIONS = ("sector_segment", "loan_type_segment", "data_richness")

UNAVAILABLE_SLICES_NOTE = (
    "Region/geography and gender-of-promoter slices (plan.md §12.6) are not "
    "computable on the Home Credit proxy snapshot — those attributes do not "
    "exist in the data. The same group-by pipeline activates for them "
    "unchanged once real sandbox data carrying region and promoter-gender "
    "columns is connected."
)


def _group_rows(df: pd.DataFrame, dimension: str) -> list[dict]:
    rows = []
    for group, g in df.groupby(dimension, observed=True):
        flagged = g["risk_grade"].isin(HIGH_RISK_GRADES)
        # FPR only over loans whose 12-month outcome is determinable (§9.4 rule):
        # defaulted, or observed the full horizon without defaulting.
        evaluable = g[(g["event"] == 1) | (g["duration"] >= MAX_MONTHS)]
        negatives = evaluable[evaluable["event"] == 0]
        fpr = (
            float(negatives["risk_grade"].isin(HIGH_RISK_GRADES).mean())
            if len(negatives)
            else None
        )
        rows.append(
            {
                "group": str(group),
                "n": int(len(g)),
                "avg_calibrated_pd": round(float(g["calibrated_pd_12m"].mean()), 5),
                "high_risk_share": round(float(flagged.mean()), 5),
                "false_positive_rate": None if fpr is None else round(fpr, 5),
            }
        )
    best = min(r["avg_calibrated_pd"] for r in rows) or 1e-9
    for r in rows:
        r["disparity_ratio"] = round(r["avg_calibrated_pd"] / best, 3)
    rows.sort(key=lambda r: r["avg_calibrated_pd"])
    return rows


def fairness_audit(
    scored_path: Path = SCORED_PATH, features_path: Path = FEATURES_PATH
) -> dict:
    """The §12.6 disparate-impact summary served at GET /api/fairness/audit."""
    scored = pd.read_parquet(scored_path)
    features = pd.read_parquet(features_path)[
        ["SK_ID_CURR", "sector_segment", "data_richness"]
    ]
    df = scored.merge(features, on="SK_ID_CURR", how="left")

    return {
        "n_loans": int(len(df)),
        "basis": (
            "Phase 4 OOT scored test set joined with enriched serving features; "
            "high-risk = unified grade E or worse; FPR computed only over loans "
            "with a determinable 12-month outcome."
        ),
        "dimensions": {dim: _group_rows(df, dim) for dim in DIMENSIONS},
        "unavailable_slices": ["region_geography", "gender_of_promoter"],
        "note": UNAVAILABLE_SLICES_NOTE,
    }
