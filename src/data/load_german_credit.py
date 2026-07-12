"""Phase 6 loader for the Statlog German Credit dataset (plan.md §9.3).

German Credit (1,000 rows, 20 attributes) is the dataset TabPFN's own creators
used for their proof-of-concept, which is why it is the first thin-file test
bed here. Column names are snake_cased from the attribute descriptions in
data/raw/german_credit/german.doc. Label: 1 = good -> TARGET 0, 2 = bad ->
TARGET 1. Categoricals (A-coded symbols) are one-hot encoded so both TabPFN
and LightGBM receive a fully numeric matrix.
"""

from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "german_credit" / "german.data"

DATA_PROVENANCE = "public_proxy"

# Attribute names per german.doc, in file order (20 attributes + label).
COLUMN_NAMES = [
    "checking_account_status",       # 1  qualitative
    "duration_months",               # 2  numerical
    "credit_history",                # 3  qualitative
    "purpose",                       # 4  qualitative
    "credit_amount",                 # 5  numerical
    "savings_account",               # 6  qualitative
    "employment_since",              # 7  qualitative
    "installment_rate_pct_income",   # 8  numerical
    "personal_status_sex",           # 9  qualitative
    "other_debtors",                 # 10 qualitative
    "residence_since",               # 11 numerical
    "property",                      # 12 qualitative
    "age_years",                     # 13 numerical
    "other_installment_plans",       # 14 qualitative
    "housing",                       # 15 qualitative
    "existing_credits_count",        # 16 numerical
    "job",                           # 17 qualitative
    "dependents_count",              # 18 numerical
    "telephone",                     # 19 qualitative
    "foreign_worker",                # 20 qualitative
]

CATEGORICAL_COLUMNS = [
    "checking_account_status",
    "credit_history",
    "purpose",
    "savings_account",
    "employment_since",
    "personal_status_sex",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
]


def load_german_credit_raw() -> pd.DataFrame:
    """Parse german.data as-is: 20 named attributes + TARGET (bad=1, good=0)."""
    df = pd.read_csv(RAW_PATH, sep=r"\s+", header=None, names=COLUMN_NAMES + ["label"])
    df["TARGET"] = (df["label"] == 2).astype(int)  # 2 = bad credit risk -> default proxy
    df = df.drop(columns="label")
    df["data_provenance"] = DATA_PROVENANCE
    return df


def load_german_credit() -> pd.DataFrame:
    """Numeric model-ready table: one-hot categoricals + row_id pseudo-time key.

    German Credit has no application timestamp or ID; row order in the file is
    the only sequence available, so row_id serves as the ordered-split key the
    same way SK_ID_CURR does for Home Credit (plan.md §15.2).
    """
    df = load_german_credit_raw()
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, dtype=int)
    df.insert(0, "row_id", range(len(df)))
    return df
