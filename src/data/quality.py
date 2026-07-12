"""Great-Expectations-style data-quality gate (plan.md §6.5, PHASES.md Phase 0).

Lightweight pandas checks with the same contract as a GE suite: each check
returns a result dict, `run_checks` fails loudly (raises) when any expectation
is violated — never silently. Great Expectations itself is the named production
upgrade path; these checks are deliberately dependency-free.
"""
from __future__ import annotations

import pandas as pd


class DataQualityError(ValueError):
    """Raised when a data-quality expectation fails."""


def expect_null_rate_below(df: pd.DataFrame, column: str, threshold: float) -> dict:
    rate = float(df[column].isna().mean())
    return {
        "check": f"null_rate({column}) < {threshold}",
        "observed": rate,
        "passed": rate < threshold,
    }


def expect_values_between(
    df: pd.DataFrame, column: str, low: float, high: float, allow_null: bool = True
) -> dict:
    series = df[column].dropna() if allow_null else df[column]
    in_range = series.between(low, high)
    frac = float(in_range.mean()) if len(series) else 1.0
    return {
        "check": f"{column} in [{low}, {high}]",
        "observed": frac,
        "passed": bool(in_range.all()) if len(series) else True,
    }


def expect_no_duplicate_rows(df: pd.DataFrame, subset: list[str] | None = None) -> dict:
    dupes = int(df.duplicated(subset=subset).sum())
    return {
        "check": f"no duplicate rows (subset={subset})",
        "observed": dupes,
        "passed": dupes == 0,
    }


def expect_column_present(df: pd.DataFrame, column: str) -> dict:
    return {
        "check": f"column present: {column}",
        "observed": column in df.columns,
        "passed": column in df.columns,
    }


def run_checks(table_name: str, results: list[dict]) -> list[dict]:
    """Evaluate a list of check results; raise loudly on any failure."""
    failures = [r for r in results if not r["passed"]]
    if failures:
        detail = "; ".join(f"{r['check']} (observed={r['observed']})" for r in failures)
        raise DataQualityError(f"Data-quality gate FAILED for {table_name}: {detail}")
    return results


def serving_features_suite(df: pd.DataFrame) -> list[dict]:
    """The standing suite for the serving feature snapshot."""
    return run_checks(
        "serving_features",
        [
            expect_column_present(df, "SK_ID_CURR"),
            expect_column_present(df, "loan_type_segment"),
            expect_column_present(df, "data_richness"),
            expect_no_duplicate_rows(df, subset=["SK_ID_CURR"]),
            expect_null_rate_below(df, "credit_income_ratio", 0.05),
            expect_values_between(df, "late_installment_rate", 0.0, 1.0),
            expect_values_between(df, "age_years", 18.0, 100.0),
        ],
    )
