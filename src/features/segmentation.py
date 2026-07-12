"""Phase 1 proxy segmentation for the Home Credit dataset (plan.md §7.2, §7.3).

Home Credit has no real MSME loan-type/sector field, so this module documents
explicit proxy mappings, schema-swappable to real MSME fields once available:
  - NAME_CONTRACT_TYPE -> loan_type_segment (Cash loans ~ term loan,
    Revolving loans ~ working-capital/overdraft)
  - ORGANIZATION_TYPE  -> sector_segment (Manufacturing / Retail_Trade /
    Services_IT / Agriculture_Allied / Other_Public)
  - bureau history depth -> data_richness (established vs ntc_ntb)
"""

import pandas as pd

LOAN_TYPE_MAP = {
    "Cash loans": "term_loan_proxy",
    "Revolving loans": "working_capital_proxy",
}

SECTOR_BUCKETS = {
    "Manufacturing": [f"Industry: type {i}" for i in range(1, 14)] + ["Construction"],
    "Retail_Trade": [f"Trade: type {i}" for i in range(1, 8)] + ["Restaurant", "Realtor", "Hotel"],
    "Services_IT": [
        "Business Entity Type 1",
        "Business Entity Type 2",
        "Business Entity Type 3",
        "Bank",
        "Insurance",
        "Legal Services",
        "Advertising",
        "Telecom",
        "Mobile",
        "Services",
        "Cleaning",
        "Security",
        "Self-employed",
    ],
    "Agriculture_Allied": ["Agriculture"],
}


def _sector_proxy(org_type: str) -> str:
    for sector, values in SECTOR_BUCKETS.items():
        if org_type in values:
            return sector
    return "Other_Public"


def add_loan_type_segment(applications: pd.DataFrame) -> pd.DataFrame:
    df = applications.copy()
    df["loan_type_segment"] = df["NAME_CONTRACT_TYPE"].map(LOAN_TYPE_MAP).fillna("other_proxy")
    df["sector_segment"] = df["ORGANIZATION_TYPE"].apply(_sector_proxy)
    return df


def add_data_richness(
    applications: pd.DataFrame, bureau: pd.DataFrame, min_bureau_records: int = 2
) -> pd.DataFrame:
    bureau_counts = bureau.groupby("SK_ID_CURR").size().rename("bureau_record_count")
    df = applications.merge(bureau_counts, on="SK_ID_CURR", how="left")
    df["bureau_record_count"] = df["bureau_record_count"].fillna(0).astype(int)
    df["data_richness"] = "ntc_ntb"
    df.loc[df["bureau_record_count"] >= min_bureau_records, "data_richness"] = "established"
    return df


def segment(applications: pd.DataFrame, bureau: pd.DataFrame) -> pd.DataFrame:
    df = add_loan_type_segment(applications)
    df = add_data_richness(df, bureau)
    return df
