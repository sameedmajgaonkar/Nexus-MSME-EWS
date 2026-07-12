"""Phase 1 real-data verification: segment distribution + WOE monotonicity check."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.load_home_credit import load_applications, load_bureau, load_installments
from src.features.segmentation import segment
from src.features.structured import build_feature_table
from src.features.woe import event_rate_by_bin, fit_woe_transform, is_monotonic

pd.set_option("display.width", 120)


def main():
    print("Loading raw tables...")
    applications = load_applications()
    bureau = load_bureau()
    installments = load_installments(
        usecols=["SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"]
    )

    print("Segmenting...")
    applications = segment(applications, bureau)

    print("\n=== Segment distribution (loan_type_segment x data_richness) ===")
    print(
        pd.crosstab(applications["loan_type_segment"], applications["data_richness"], margins=True)
    )

    print("\n=== Segment distribution (sector_segment) ===")
    print(applications["sector_segment"].value_counts())

    print("\nBuilding structured feature table...")
    feature_table = build_feature_table(applications, bureau, installments)
    print(f"Feature table shape: {feature_table.shape}")

    print("\nFitting WOE transforms...")
    top3 = ["credit_income_ratio", "late_installment_rate", "bureau_overdue_flag"]
    woe_df, binners = fit_woe_transform(feature_table, features=top3)

    print("\n=== WOE monotonicity check (top 3 features) ===")
    all_monotonic = True
    for feat in top3:
        rates = event_rate_by_bin(binners[feat])
        monotonic = is_monotonic(rates)
        all_monotonic &= monotonic
        print(f"{feat}: event rates per bin = {rates.tolist()} -> monotonic={monotonic}")

    print(f"\nAll top-3 features monotonic: {all_monotonic}")

    out_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "phase1_features.parquet"
    feature_table.to_parquet(out_path, index=False)
    print(f"\nSaved feature table to {out_path}")


if __name__ == "__main__":
    main()
