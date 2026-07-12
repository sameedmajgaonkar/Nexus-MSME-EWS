"""Phase 1 core structured features (plan.md §8.1, deliberately scoped down for the MVP:
a handful of strong features rather than the full exhaustive list)."""

import numpy as np
import pandas as pd


def build_installment_features(installments: pd.DataFrame) -> pd.DataFrame:
    df = installments[["SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"]].copy()
    df["days_late"] = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
    df["is_late"] = (df["days_late"] > 0).astype(int)

    agg = (
        df.groupby("SK_ID_CURR")
        .agg(
            n_installments=("days_late", "size"),
            late_installment_rate=("is_late", "mean"),
            max_days_late=("days_late", "max"),
            avg_days_late=("days_late", "mean"),
        )
        .reset_index()
    )
    agg["max_days_late"] = agg["max_days_late"].clip(lower=0)
    return agg


def build_bureau_features(bureau: pd.DataFrame) -> pd.DataFrame:
    agg = (
        bureau.groupby("SK_ID_CURR")
        .agg(
            bureau_active_count=("CREDIT_ACTIVE", lambda s: (s == "Active").sum()),
            bureau_overdue_flag=("CREDIT_DAY_OVERDUE", lambda s: int((s > 0).any())),
        )
        .reset_index()
    )
    return agg


def build_ratio_features(applications: pd.DataFrame) -> pd.DataFrame:
    df = applications[["SK_ID_CURR", "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY"]].copy()
    df["credit_income_ratio"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["annuity_income_ratio"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    return df[["SK_ID_CURR", "credit_income_ratio", "annuity_income_ratio"]]


DAYS_EMPLOYED_SENTINEL = 365243  # Home Credit's placeholder for pensioner/unemployed


def build_applicant_features(applications: pd.DataFrame) -> pd.DataFrame:
    """External bureau-style scores + demographics (proxy for CIBIL/CRIF-grade fields)."""
    df = applications[
        ["SK_ID_CURR", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH", "DAYS_EMPLOYED"]
    ].copy()
    df = df.rename(
        columns={
            "EXT_SOURCE_1": "ext_source_1",
            "EXT_SOURCE_2": "ext_source_2",
            "EXT_SOURCE_3": "ext_source_3",
        }
    )
    df["age_years"] = -df["DAYS_BIRTH"] / 365.25
    days_employed = df["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_SENTINEL, np.nan)
    df["employed_years"] = (-days_employed / 365.25).clip(lower=0)
    return df[
        ["SK_ID_CURR", "ext_source_1", "ext_source_2", "ext_source_3", "age_years", "employed_years"]
    ]


FEATURE_FILL_DEFAULTS = {
    "bureau_active_count": 0,
    "bureau_overdue_flag": 0,
    "n_installments": 0,
    "late_installment_rate": 0,
    "max_days_late": 0,
    "avg_days_late": 0,
}


def build_feature_table(
    applications: pd.DataFrame, bureau: pd.DataFrame, installments: pd.DataFrame
) -> pd.DataFrame:
    ratios = build_ratio_features(applications)
    applicant_feats = build_applicant_features(applications)
    bureau_feats = build_bureau_features(bureau)
    inst_feats = build_installment_features(installments)

    keep_cols = ["SK_ID_CURR", "TARGET", "loan_type_segment", "sector_segment", "data_richness"]
    df = applications[keep_cols].copy()
    df = df.merge(ratios, on="SK_ID_CURR", how="left")
    df = df.merge(applicant_feats, on="SK_ID_CURR", how="left")
    df = df.merge(bureau_feats, on="SK_ID_CURR", how="left")
    df = df.merge(inst_feats, on="SK_ID_CURR", how="left")

    for col, default in FEATURE_FILL_DEFAULTS.items():
        df[col] = df[col].fillna(default)

    return df
