"""Scheduled drift -> retrain check (plan.md §13.2, §13.4). Run monthly.

Drift check: reference = pseudo-OOT train window (the training-time
distribution), current = the most recent pseudo-OOT window (with real sandbox
data this becomes the latest scoring month). If PSI > 0.2 on any Tier-1
feature, the §13.2 retrain-and-gate loop runs; the gate never promotes a
challenger that loses to Production on the OOT test window.

Schedule with cron / Windows Task Scheduler — see scripts/retrain_cron.md.
Exit code: 0 = no action or promotion; 1 = drift fired and challenger was HELD
(manual review needed), matching the flowchart's "Hold; flag for manual review".
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.mlops.drift import TIER1_FEATURES, run_drift_report
from src.mlops.retrain import retrain_and_gate
from src.models.splits import pseudo_oot_split

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def main() -> int:
    df = pd.read_parquet(PROCESSED / "serving_features_enriched.parquet")
    split = pseudo_oot_split(df)
    reference = df[(split == "train").to_numpy()]
    current = df[(split == "test").to_numpy()]

    summary = run_drift_report(reference, current, TIER1_FEATURES)
    print(f"drift check: retrain_triggered={summary['retrain_triggered']}")
    print(f"  {summary['plain_language']}")

    if not summary["retrain_triggered"]:
        print("no retrain needed; next check on the scheduled cycle.")
        return 0

    print("retrain triggered -> running retrain_and_gate...")
    result = retrain_and_gate(df, split)
    print(f"  gate decision: {'PROMOTED' if result['promoted'] else 'HELD'}")
    print(f"  {result['reason']}")
    return 0 if result["promoted"] else 1


if __name__ == "__main__":
    sys.exit(main())
