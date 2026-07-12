import pandas as pd

from src.features.segmentation import segment


def _applications():
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4, 5],
            "NAME_CONTRACT_TYPE": [
                "Cash loans",
                "Revolving loans",
                "Cash loans",
                "Cash loans",
                "Revolving loans",
            ],
            "ORGANIZATION_TYPE": [
                "Industry: type 3",
                "Trade: type 2",
                "Business Entity Type 3",
                "Agriculture",
                "XNA",
            ],
        }
    )


def _bureau():
    # SK_ID_CURR 1 and 3 have >=2 bureau records (established); 2, 4, 5 have <2 (ntc_ntb)
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 1, 2, 3, 3, 4],
        }
    )


def test_loan_type_and_sector_mapping():
    df = segment(_applications(), _bureau())
    row = lambda sk: df.loc[df["SK_ID_CURR"] == sk].iloc[0]

    assert row(1)["loan_type_segment"] == "term_loan_proxy"
    assert row(1)["sector_segment"] == "Manufacturing"

    assert row(2)["loan_type_segment"] == "working_capital_proxy"
    assert row(2)["sector_segment"] == "Retail_Trade"

    assert row(3)["loan_type_segment"] == "term_loan_proxy"
    assert row(3)["sector_segment"] == "Services_IT"

    assert row(4)["loan_type_segment"] == "term_loan_proxy"
    assert row(4)["sector_segment"] == "Agriculture_Allied"

    assert row(5)["loan_type_segment"] == "working_capital_proxy"
    assert row(5)["sector_segment"] == "Other_Public"


def test_data_richness_from_bureau_history():
    df = segment(_applications(), _bureau())
    row = lambda sk: df.loc[df["SK_ID_CURR"] == sk].iloc[0]

    assert row(1)["data_richness"] == "established"  # 3 bureau records
    assert row(2)["data_richness"] == "ntc_ntb"  # 1 bureau record
    assert row(3)["data_richness"] == "established"  # 2 bureau records
    assert row(4)["data_richness"] == "ntc_ntb"  # 1 bureau record
    assert row(5)["data_richness"] == "ntc_ntb"  # 0 bureau records
