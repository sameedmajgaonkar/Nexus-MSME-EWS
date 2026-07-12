"""Phase 1 WOE binning on top of the structured feature table (plan.md §8.1)."""

import pandas as pd
from optbinning import OptimalBinning

WOE_CANDIDATE_FEATURES = [
    "credit_income_ratio",
    "annuity_income_ratio",
    "late_installment_rate",
    "max_days_late",
    "avg_days_late",
    "bureau_active_count",
    "bureau_overdue_flag",
    "ext_source_1",
    "ext_source_2",
    "ext_source_3",
    "age_years",
    "employed_years",
]


FEATURE_MIN_BIN_SIZE = {
    # bureau_overdue_flag=1 is a real signal but only ~1.1% of rows; the
    # default min_bin_size would merge it away, so allow a smaller bin here.
    "bureau_overdue_flag": 0.005,
}

# The pre-binning step (which runs before optimization) also has its own
# ~5% default floor and merges rare categories away before the optimizer
# ever sees them — lower it wherever min_bin_size is also lowered.
FEATURE_MIN_PREBIN_SIZE = {
    "bureau_overdue_flag": 0.001,
}

# optbinning's default monotonic_trend='auto' permits peak/valley shapes,
# which are non-monotonic overall. A WOE scorecard needs a strict direction
# per feature (plan.md §9.1's whole point is auditable, monotonic bins), so
# every candidate feature here is forced to its expected risk direction.
FEATURE_MONOTONIC_TREND = {
    "credit_income_ratio": "ascending",
    "annuity_income_ratio": "ascending",
    "late_installment_rate": "ascending",
    "max_days_late": "ascending",
    "avg_days_late": "ascending",
    "bureau_active_count": "ascending",
    "bureau_overdue_flag": "ascending",
    # higher external score / age / tenure = lower default risk
    "ext_source_1": "descending",
    "ext_source_2": "descending",
    "ext_source_3": "descending",
    "age_years": "descending",
    "employed_years": "descending",
}


def fit_woe_transform(df: pd.DataFrame, target_col: str = "TARGET", features=None):
    """Fit one OptimalBinning per feature and return (woe_dataframe, fitted_binners)."""
    features = features or WOE_CANDIDATE_FEATURES
    binners = {}
    woe_df = pd.DataFrame(index=df.index)
    for feat in features:
        x = df[feat].values
        y = df[target_col].values
        kwargs = {"monotonic_trend": FEATURE_MONOTONIC_TREND.get(feat, "auto")}
        if feat in FEATURE_MIN_BIN_SIZE:
            kwargs["min_bin_size"] = FEATURE_MIN_BIN_SIZE[feat]
        if feat in FEATURE_MIN_PREBIN_SIZE:
            kwargs["min_prebin_size"] = FEATURE_MIN_PREBIN_SIZE[feat]
        binning = OptimalBinning(name=feat, dtype="numerical", solver="cp", **kwargs)
        binning.fit(x, y)
        binners[feat] = binning
        woe_df[f"{feat}_woe"] = binning.transform(x, metric="woe")
    return woe_df, binners


def event_rate_by_bin(binning: OptimalBinning) -> pd.Series:
    """Extract the per-bin observed event rate from a fitted binning table, in bin order."""
    table = binning.binning_table.build().reset_index()
    real_bins = table[
        (table["index"] != "Totals") & (~table["Bin"].isin(["Special", "Missing"]))
    ]
    return real_bins["Event rate"].astype(float).reset_index(drop=True)


def is_monotonic(series: pd.Series) -> bool:
    return series.is_monotonic_increasing or series.is_monotonic_decreasing
