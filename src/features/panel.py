"""Phase 3 person-period panel construction (plan.md §9.2 Option A).

Home Credit's TARGET has no event timing, so the survival event here is
**first serious delinquency (30+ days late, or an unpaid installment) within
the first 12 installment-months of the applicant's most recent previous
loan** — real month-by-month repayment history from installments_payments.csv.

Proxy caveat (stated, not hidden): application-level features are as-of the
current application, i.e. after the panel window. With real MSME sandbox data
the features become point-in-time and the machinery below is unchanged.

Censoring: a loan observed for m < 12 months with no event contributes m rows
with event=0 — it is never labeled "safe".
"""

import numpy as np
import pandas as pd

SERIOUS_DELINQ_DAYS = 30
MAX_MONTHS = 12
UNPAID_DAYS_LATE = 9999  # missing payment date = installment never paid


def most_recent_prev_loan(installments: pd.DataFrame) -> pd.DataFrame:
    """Keep only each applicant's most recently active previous loan."""
    recency = (
        installments.groupby(["SK_ID_CURR", "SK_ID_PREV"])["DAYS_INSTALMENT"]
        .max()
        .reset_index()
        .sort_values(["SK_ID_CURR", "DAYS_INSTALMENT"])
        .drop_duplicates("SK_ID_CURR", keep="last")
    )
    return installments.merge(recency[["SK_ID_CURR", "SK_ID_PREV"]], on=["SK_ID_CURR", "SK_ID_PREV"])


def loan_durations(installments: pd.DataFrame) -> pd.DataFrame:
    """Collapse one previous loan per applicant into (duration, event).

    duration = months until first serious delinquency, or until censoring
    (end of observed installments or the 12-month cap), whichever is first.
    """
    df = installments.copy()
    df["days_late"] = (df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]).fillna(UNPAID_DAYS_LATE)
    df["month"] = df["NUM_INSTALMENT_NUMBER"].astype(int)
    df = df[df["month"] >= 1]

    monthly = (
        df.groupby(["SK_ID_CURR", "month"])["days_late"].max().reset_index()
    )
    monthly["is_event_month"] = monthly["days_late"] > SERIOUS_DELINQ_DAYS

    agg = monthly.groupby("SK_ID_CURR").agg(
        months_observed=("month", "max"),
        first_event_month=(
            "month",
            lambda m: np.nan,  # placeholder, filled below
        ),
    )
    first_event = (
        monthly[monthly["is_event_month"]].groupby("SK_ID_CURR")["month"].min().rename("first_event_month")
    )
    agg = agg.drop(columns="first_event_month").join(first_event)

    agg["months_observed"] = agg["months_observed"].clip(upper=MAX_MONTHS)
    has_event = agg["first_event_month"].notna() & (agg["first_event_month"] <= agg["months_observed"])
    agg["event"] = has_event.astype(int)
    agg["duration"] = np.where(has_event, agg["first_event_month"], agg["months_observed"]).astype(int)
    return agg.reset_index()[["SK_ID_CURR", "duration", "event"]]


def expand_person_period(durations: pd.DataFrame) -> pd.DataFrame:
    """One row per loan per month survived; event=1 only on an event's final row."""
    reps = durations["duration"].to_numpy()
    idx = np.repeat(durations.index.to_numpy(), reps)
    panel = durations.loc[idx, ["SK_ID_CURR"]].reset_index(drop=True)
    panel["months_since_origination"] = np.concatenate([np.arange(1, d + 1) for d in reps])

    last_row = panel["months_since_origination"] == np.repeat(reps, reps)
    panel["event"] = (last_row & np.repeat(durations["event"].to_numpy() == 1, reps)).astype(int)
    return panel


def build_person_period_panel(installments: pd.DataFrame) -> pd.DataFrame:
    recent = most_recent_prev_loan(installments)
    durations = loan_durations(recent)
    return expand_person_period(durations)


def prior_loan_behavior(installments: pd.DataFrame) -> pd.DataFrame:
    """Behavioral features from the applicant's OTHER (older) previous loans.

    The panel loan itself is excluded, so these features never see the
    installments that define the panel's event — leakage-safe by construction.
    Applicants with a single previous loan get NaN (LightGBM handles natively).
    """
    recent = most_recent_prev_loan(installments)
    panel_pairs = recent[["SK_ID_CURR", "SK_ID_PREV"]].drop_duplicates()
    merged = installments.merge(
        panel_pairs, on=["SK_ID_CURR", "SK_ID_PREV"], how="left", indicator=True
    )
    others = merged[merged["_merge"] == "left_only"].copy()

    others["days_late"] = (
        others["DAYS_ENTRY_PAYMENT"] - others["DAYS_INSTALMENT"]
    ).fillna(UNPAID_DAYS_LATE)
    others["is_late"] = others["days_late"] > SERIOUS_DELINQ_DAYS

    return (
        others.groupby("SK_ID_CURR")
        .agg(prior_late_rate=("is_late", "mean"), prior_n_installments=("is_late", "size"))
        .reset_index()
    )
