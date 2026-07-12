import pandas as pd
import pytest

from src.data.quality import (
    DataQualityError,
    expect_no_duplicate_rows,
    expect_null_rate_below,
    expect_values_between,
    run_checks,
)


def _frame():
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4],
            "late_installment_rate": [0.0, 0.5, 1.0, 0.2],
            "credit_income_ratio": [1.2, 3.4, None, 2.0],
        }
    )


def test_passing_suite_returns_results():
    df = _frame()
    results = run_checks(
        "t",
        [
            expect_no_duplicate_rows(df, subset=["SK_ID_CURR"]),
            expect_values_between(df, "late_installment_rate", 0.0, 1.0),
            expect_null_rate_below(df, "credit_income_ratio", 0.5),
        ],
    )
    assert all(r["passed"] for r in results)


def test_failure_raises_loudly_with_reason():
    df = _frame()
    df.loc[0, "late_installment_rate"] = 2.5  # out of range
    with pytest.raises(DataQualityError, match="late_installment_rate"):
        run_checks("t", [expect_values_between(df, "late_installment_rate", 0.0, 1.0)])


def test_duplicate_detection_fails():
    df = pd.DataFrame({"SK_ID_CURR": [1, 1]})
    with pytest.raises(DataQualityError, match="duplicate"):
        run_checks("t", [expect_no_duplicate_rows(df, subset=["SK_ID_CURR"])])
