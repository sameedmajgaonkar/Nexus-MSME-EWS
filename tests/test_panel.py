import pandas as pd

from src.features.panel import build_person_period_panel, expand_person_period, loan_durations


def _installments(rows):
    """rows: list of (sk_id_curr, sk_id_prev, num, days_instalment, days_entry_payment)"""
    return pd.DataFrame(
        rows,
        columns=[
            "SK_ID_CURR",
            "SK_ID_PREV",
            "NUM_INSTALMENT_NUMBER",
            "DAYS_INSTALMENT",
            "DAYS_ENTRY_PAYMENT",
        ],
    )


def test_censored_loan_contributes_event0_rows_not_a_safe_label():
    # Loan observed for 8 months, every installment paid on time -> censored at 8.
    rows = [(1, 10, m, -300 + 30 * m, -300 + 30 * m) for m in range(1, 9)]
    panel = build_person_period_panel(_installments(rows))

    assert len(panel) == 8  # 8 person-period rows, NOT one row
    assert (panel["event"] == 0).all()  # never labeled as an outcome
    assert list(panel["months_since_origination"]) == list(range(1, 9))


def test_event_loan_truncates_at_event_month():
    # Paid on time months 1-4, then 45 days late in month 5, on time after.
    rows = [(2, 20, m, -400 + 30 * m, -400 + 30 * m) for m in range(1, 5)]
    rows.append((2, 20, 5, -250, -205))  # 45 days late
    rows += [(2, 20, m, -400 + 30 * m, -400 + 30 * m) for m in range(6, 10)]
    panel = build_person_period_panel(_installments(rows))

    assert len(panel) == 5  # rows stop at the event month
    assert list(panel["event"]) == [0, 0, 0, 0, 1]  # event only on the final row


def test_panel_caps_at_12_months():
    rows = [(3, 30, m, -900 + 30 * m, -900 + 30 * m) for m in range(1, 25)]
    panel = build_person_period_panel(_installments(rows))
    assert len(panel) == 12
    assert panel["months_since_origination"].max() == 12


def test_unpaid_installment_is_an_event():
    # Month 3's payment date is missing = never paid -> event at month 3.
    rows = [(4, 40, 1, -90, -90), (4, 40, 2, -60, -60), (4, 40, 3, -30, None)]
    panel = build_person_period_panel(_installments(rows))
    assert list(panel["event"]) == [0, 0, 1]


def test_only_most_recent_prev_loan_is_used():
    old = [(5, 50, m, -1000 + 30 * m, -1000 + 30 * m) for m in range(1, 7)]
    recent = [(5, 51, m, -200 + 30 * m, -200 + 30 * m) for m in range(1, 4)]
    panel = build_person_period_panel(_installments(old + recent))
    assert len(panel) == 3  # only the recent loan's 3 months
