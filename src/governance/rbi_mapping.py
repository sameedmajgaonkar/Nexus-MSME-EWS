"""RBI-native early-warning mapping, served read-only (plan.md §12.4).

The ML hazard curve is a FORWARD-LOOKING layer sitting ahead of RBI's own
(structurally reactive) SMA ladder — not a replacement for it. Every
ML-derived trigger is mapped onto the codified indicator taxonomy of RBI's
July 2024 Master Directions on Fraud Risk Management, so a flagged account is
immediately usable by existing EWS/RFA (Red-Flagged Account) and CRILC
reporting workflows rather than being a parallel, unexplained score.

Plain data structures only — the API serves these verbatim and the Phase 9
dashboard renders them as a read-only panel."""

# ML-derived trigger -> RBI indicator -> the regulatory action it enables (§12.4 table).
ML_TRIGGER_TO_RBI = [
    {
        "ml_trigger": "GST turnover drop >X% month-over-month",
        "rbi_indicator": "'Default in payment to statutory bodies' family of EWS indicators",
        "regulatory_action": "Feeds existing bank EWS/RFA workflow",
    },
    {
        "ml_trigger": "Sudden spike in cash withdrawal relative to account activity",
        "rbi_indicator": "'Heavy cash withdrawal in loan accounts'",
        "regulatory_action": "Existing RFA escalation path",
    },
    {
        "ml_trigger": "Large RTGS transfer to an unrelated, newly-added counterparty",
        "rbi_indicator": "'High-value RTGS to unrelated parties'",
        "regulatory_action": "Existing RFA escalation path",
    },
    {
        "ml_trigger": "Repeated ad-hoc credit-limit requests",
        "rbi_indicator": "'Frequent ad-hoc sanction requests'",
        "regulatory_action": "Existing RFA escalation path",
    },
    {
        "ml_trigger": "PD crosses Stage-1 -> Stage-2 threshold",
        "rbi_indicator": "Ind AS 109 'significant increase in credit risk'",
        "regulatory_action": "Triggers lifetime-ECL provisioning review",
    },
]

# RBI Special Mention Account ladder — the existing, reactive regulatory EWS.
SMA_LADDER = [
    {"bucket": "SMA-0", "definition": "Principal/interest overdue 1-30 days"},
    {"bucket": "SMA-1", "definition": "Principal/interest overdue 31-60 days"},
    {"bucket": "SMA-2", "definition": "Principal/interest overdue 61-90 days"},
    {
        "bucket": "NPA",
        "definition": "Overdue 90+ days — regulatory NPA per RBI IRAC norms; "
        "Stage 3 ECL under Ind AS 109",
    },
]

# §12.4 state machine: the forward-looking ML layer feeding the regulatory states.
EWS_STATES = ["Performing", "WatchList", "HighAlert", "Critical", "NPA_Stage3"]

EWS_TRANSITIONS = [
    {
        "from": "Performing",
        "to": "WatchList",
        "condition": "ML hazard curve crosses segment-calibrated threshold "
        "(e.g. 12-month PD > 20%)",
    },
    {
        "from": "WatchList",
        "to": "HighAlert",
        "condition": "PD > 50% OR an EWS indicator fires (GST turnover drop, cheque "
        "bounce, RTGS to unrelated party, cash-withdrawal anomaly)",
    },
    {
        "from": "HighAlert",
        "to": "Critical",
        "condition": "PD > 75% OR SMA-1/SMA-2 delinquency confirms the signal",
    },
    {
        "from": "Critical",
        "to": "NPA_Stage3",
        "condition": "90+ DPD — regulatory NPA per RBI IRAC norms; Stage 3 ECL "
        "under Ind AS 109",
    },
    {"from": "WatchList", "to": "Performing", "condition": "Risk factors resolve"},
    {"from": "HighAlert", "to": "WatchList", "condition": "Restructuring succeeds"},
]


def get_rbi_mapping() -> dict:
    """The full read-only §12.4 payload served at GET /api/rbi/mapping."""
    return {
        "positioning": (
            "The ML hazard curve is a forward-looking layer sitting AHEAD of RBI's "
            "SMA framework (whose earliest signal, SMA-0, fires only after a payment "
            "is already 1 day overdue). Every ML trigger maps onto RBI's codified "
            "indicator taxonomy so flagged accounts flow into existing EWS/RFA and "
            "CRILC workflows."
        ),
        "source": (
            "RBI Master Directions on Fraud Risk Management (July 2024); RBI IRAC "
            "norms; Ind AS 109. See plan.md §12.4."
        ),
        "ml_trigger_mapping": ML_TRIGGER_TO_RBI,
        "sma_ladder": SMA_LADDER,
        "state_machine": {"states": EWS_STATES, "transitions": EWS_TRANSITIONS},
    }
