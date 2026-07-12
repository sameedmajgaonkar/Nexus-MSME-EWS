"""Segment router (plan.md §7.2 segmentation decision tree).

The §7.2 flowchart sends any MSME loan without 12+ months of repayment/bureau
history — i.e. data_richness == 'ntc_ntb' — to the TabPFN thin-file specialist
(§9.3); every other loan goes to the discrete-time hazard engine. Routing is
pure data logic: no model is loaded here, so the router can run inside tests
and hot request paths for free.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODEL_REGISTRY = {
    "hazard": ROOT / "models" / "hazard.joblib",
    "tabpfn_thin_file": ROOT / "models" / "thin_file_tabpfn.joblib",
}

THIN_FILE_RICHNESS = "ntc_ntb"


def route_segment(loan) -> str:
    """Return the model name in MODEL_REGISTRY that should score this loan.

    Accepts a dict or a pandas Series (anything with dict-style access to a
    'data_richness' field).
    """
    richness = loan.get("data_richness") if hasattr(loan, "get") else loan["data_richness"]
    return "tabpfn_thin_file" if richness == THIN_FILE_RICHNESS else "hazard"
