"""Phase 6 router tests (plan.md §7.2): NTC/NTB dispatches to TabPFN, not hazard.

Router is pure logic — no TabPFN inference here (far too slow for unit tests).
"""

import pandas as pd

from src.models.router import MODEL_REGISTRY, route_segment


def _synthetic_loan(richness: str) -> dict:
    return {
        "SK_ID_CURR": 999001,
        "loan_type_segment": "working_capital_proxy",
        "sector_segment": "Retail_Trade",
        "data_richness": richness,
        "credit_income_ratio": 3.2,
        "annuity_income_ratio": 0.18,
        "age_years": 34.0,
        "employed_years": 1.5,
    }


def test_ntc_ntb_routes_to_tabpfn():
    assert route_segment(_synthetic_loan("ntc_ntb")) == "tabpfn_thin_file"


def test_established_routes_to_hazard():
    assert route_segment(_synthetic_loan("established")) == "hazard"


def test_router_accepts_pandas_series():
    row = pd.Series(_synthetic_loan("ntc_ntb"))
    assert route_segment(row) == "tabpfn_thin_file"


def test_model_registry_has_both_models():
    assert set(MODEL_REGISTRY) >= {"hazard", "tabpfn_thin_file"}
    assert MODEL_REGISTRY["hazard"].name == "hazard.joblib"
    assert MODEL_REGISTRY["tabpfn_thin_file"].name == "thin_file_tabpfn.joblib"
