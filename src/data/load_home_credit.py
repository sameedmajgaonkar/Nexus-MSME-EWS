"""Phase 0 loaders for the Home Credit Default Risk proxy dataset (plan.md §6.2, §6.5)."""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

DATA_PROVENANCE = "public_proxy"


def _tag_provenance(df: pd.DataFrame) -> pd.DataFrame:
    df["data_provenance"] = DATA_PROVENANCE
    return df


def load_applications() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "application_train.csv")
    return _tag_provenance(df)


def load_bureau() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "bureau.csv")
    return _tag_provenance(df)


def load_previous_application() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "previous_application.csv")
    return _tag_provenance(df)


def load_installments(usecols: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "installments_payments.csv", usecols=usecols)
    return _tag_provenance(df)
