"""Phase 5 graph-lite features from a synthetic firm-counterparty graph (plan.md §8.2, §12.1).

Stage 1-3 of the §12.1 build path: a synthetic bipartite firm<->counterparty
transaction graph, hand-computed graph-lite tabular features, and the
NON-OPTIONAL synthetic-graph separability validation. Stage 4 (trained GAT)
is Tier 3 roadmap and is never built here.

Honesty note (plan.md §6.5, §12.1): this graph is a DOCUMENTED SIMULATION of
the empirical concentration->distress link — every row it produces carries
data_provenance='synthetic_graph'. Concentration propensity conditions ONLY
on observable behavioral columns (late_installment_rate, bureau_overdue_flag),
NEVER on TARGET, so no label information can leak into the features.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

PROVENANCE = "synthetic_graph"
N_COUNTERPARTIES_DEFAULT = 5000
DIVERSIFIED_CP_LOW, DIVERSIFIED_CP_HIGH = 5, 15  # PHASES.md Phase 5: diversified firms
CONCENTRATED_CP_HIGH = 2  # concentrated firms trade with 1-2 counterparties
ANCHOR_VOLUME_PERCENTILE = 0.8  # top counterparty is an "anchor" above this volume pctile
SEPARABILITY_MIN_AUC = 0.80


class GraphSeparabilityError(ValueError):
    """Raised when graph features fail to separate concentrated vs diversified firms."""


def generate_counterparty_graph(
    features_df: pd.DataFrame,
    seed: int = 42,
    n_counterparties: int = N_COUNTERPARTIES_DEFAULT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate a bipartite firm<->counterparty transaction graph, one firm per SK_ID_CURR.

    Deliberate mix of concentrated (1-2 counterparties) and diversified (5-15)
    firms per plan.md §12.1 Stage 3. The propensity to be concentrated rises
    with OBSERVABLE behavior only (late_installment_rate, bureau_overdue_flag)
    — a documented simulation of the empirical concentration->distress link;
    TARGET is never read. Counterparty popularity is Zipf-skewed so a handful
    of large "anchor" counterparties emerge naturally.

    Returns (edges, firm_profile):
      edges        — firm_id (=SK_ID_CURR), counterparty_id, txn_value
      firm_profile — SK_ID_CURR, concentrated_flag, network_churn,
                     data_provenance='synthetic_graph'
    """
    rng = np.random.default_rng(seed)
    firms = features_df["SK_ID_CURR"].to_numpy()
    late = features_df["late_installment_rate"].fillna(0).to_numpy(dtype=float)
    overdue = features_df["bureau_overdue_flag"].fillna(0).to_numpy(dtype=float)
    n_firms = len(firms)

    # Observable-behavior-only concentration propensity (never TARGET).
    p_concentrated = np.clip(0.15 + 0.9 * late + 0.25 * overdue, 0.05, 0.90)
    concentrated = rng.random(n_firms) < p_concentrated
    n_cp = np.where(
        concentrated,
        rng.integers(1, CONCENTRATED_CP_HIGH + 1, n_firms),
        rng.integers(DIVERSIFIED_CP_LOW, DIVERSIFIED_CP_HIGH + 1, n_firms),
    )

    # Zipf-skewed counterparty popularity -> natural anchor counterparties.
    ranks = np.arange(1, n_counterparties + 1, dtype=float)
    popularity = 1.0 / ranks**0.8
    popularity /= popularity.sum()

    firm_idx = np.repeat(np.arange(n_firms), n_cp)
    cp_ids = rng.choice(n_counterparties, size=int(n_cp.sum()), p=popularity)
    txn_value = rng.lognormal(mean=12.0, sigma=1.0, size=int(n_cp.sum()))

    edges = pd.DataFrame(
        {"firm_id": firms[firm_idx], "counterparty_id": cp_ids, "txn_value": txn_value}
    )
    # A firm drawing the same counterparty twice is one relationship: sum values.
    edges = edges.groupby(["firm_id", "counterparty_id"], as_index=False)["txn_value"].sum()

    # Simulated month-over-month counterparty churn rate — again conditioned
    # only on observable behavior (unstable relationships correlate with
    # payment stress), simulated directly rather than from monthly snapshots.
    churn = np.clip(0.05 + 0.5 * late + 0.10 * overdue + rng.normal(0.0, 0.03, n_firms), 0.0, 1.0)

    firm_profile = pd.DataFrame(
        {
            "SK_ID_CURR": firms,
            "concentrated_flag": concentrated.astype(int),
            "network_churn": churn,
            "data_provenance": PROVENANCE,
        }
    )
    return edges, firm_profile


def networkx_cross_check(
    edges: pd.DataFrame, n_firms: int = 1500, seed: int = 42, atol: float = 1e-12
) -> dict:
    """Build an nx.Graph validation subgraph and cross-check vectorized degrees.

    The full-graph features are computed vectorized for speed (307K firms);
    this keeps the networkx path real: nx.degree_centrality on a sampled
    bipartite subgraph must match degree/(n_nodes-1) from the edge table.
    """
    rng = np.random.default_rng(seed)
    all_firms = edges["firm_id"].unique()
    sample = rng.choice(all_firms, size=min(n_firms, len(all_firms)), replace=False)
    sub = edges[edges["firm_id"].isin(sample)]

    graph = nx.Graph()
    graph.add_nodes_from(sub["firm_id"].unique(), bipartite=0)
    cp_nodes = "C" + sub["counterparty_id"].astype(str)
    graph.add_nodes_from(cp_nodes.unique(), bipartite=1)
    graph.add_edges_from(zip(sub["firm_id"], cp_nodes))

    centrality = nx.degree_centrality(graph)
    n_nodes = graph.number_of_nodes()
    vec_degree = sub.groupby("firm_id").size()
    nx_vals = np.array([centrality[f] for f in vec_degree.index])
    vec_vals = vec_degree.to_numpy() / (n_nodes - 1)
    max_diff = float(np.max(np.abs(nx_vals - vec_vals)))
    if max_diff > atol:
        raise ValueError(f"networkx degree-centrality cross-check failed (max diff {max_diff})")
    return {
        "subgraph_nodes": n_nodes,
        "subgraph_edges": graph.number_of_edges(),
        "firms_checked": int(len(vec_degree)),
        "max_abs_centrality_diff": max_diff,
    }


def build_graph_features(
    edges: pd.DataFrame,
    firm_profile: pd.DataFrame,
    validate_networkx: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """Graph-lite tabular features per firm (plan.md §8.2 table), keyed by SK_ID_CURR.

    counterparty_concentration — top-counterparty share of total transaction value
    degree_centrality          — degree / (n_nodes - 1) on the bipartite graph
    anchor_linkage_flag        — top counterparty's own volume percentile > 0.8
    network_churn              — simulated month-over-month counterparty churn
    """
    by_firm = edges.groupby("firm_id")["txn_value"]
    total_value = by_firm.sum()
    top_value = by_firm.max()
    concentration = (top_value / total_value).rename("counterparty_concentration")

    degree = edges.groupby("firm_id").size()
    n_nodes = edges["firm_id"].nunique() + edges["counterparty_id"].nunique()
    degree_centrality = (degree / (n_nodes - 1)).rename("degree_centrality")

    # Anchor linkage: is the firm's top counterparty itself large/stable?
    cp_volume_pct = edges.groupby("counterparty_id")["txn_value"].sum().rank(pct=True)
    top_cp = edges.loc[
        edges.groupby("firm_id")["txn_value"].idxmax(), ["firm_id", "counterparty_id"]
    ].set_index("firm_id")["counterparty_id"]
    anchor_flag = (
        (cp_volume_pct.reindex(top_cp.to_numpy()).to_numpy() > ANCHOR_VOLUME_PERCENTILE)
        .astype(int)
    )

    feats = pd.DataFrame(
        {
            "SK_ID_CURR": concentration.index,
            "counterparty_concentration": concentration.to_numpy(),
            "degree_centrality": degree_centrality.to_numpy(),
            "anchor_linkage_flag": anchor_flag,
        }
    )
    feats = feats.merge(
        firm_profile[["SK_ID_CURR", "network_churn"]], on="SK_ID_CURR", how="left"
    )
    feats["data_provenance"] = PROVENANCE

    if validate_networkx:
        networkx_cross_check(edges, seed=seed)
    return feats


def validate_separability(
    graph_features: pd.DataFrame,
    firm_profile: pd.DataFrame,
    min_auc: float = SEPARABILITY_MIN_AUC,
) -> dict:
    """NON-OPTIONAL check (PHASES.md Phase 5 / plan.md §12.1 Stage 3).

    Using counterparty_concentration ALONE, concentrated firms must rank
    measurably above diversified ones. Returns AUC, group means and Cohen's d;
    raises GraphSeparabilityError if separation is absent.
    """
    from sklearn.metrics import roc_auc_score

    df = graph_features.merge(
        firm_profile[["SK_ID_CURR", "concentrated_flag"]], on="SK_ID_CURR"
    )
    y = df["concentrated_flag"].to_numpy()
    score = df["counterparty_concentration"].to_numpy()
    if y.min() == y.max():
        raise GraphSeparabilityError("Synthetic graph produced only one firm type.")

    conc, div = score[y == 1], score[y == 0]
    pooled_std = float(np.sqrt((conc.var(ddof=1) + div.var(ddof=1)) / 2))
    result = {
        "auc_concentration_alone": round(float(roc_auc_score(y, score)), 4),
        "mean_concentrated": round(float(conc.mean()), 4),
        "mean_diversified": round(float(div.mean()), 4),
        "cohens_d": round(float((conc.mean() - div.mean()) / pooled_std), 4),
        "n_concentrated": int(y.sum()),
        "n_diversified": int((1 - y).sum()),
    }
    if result["auc_concentration_alone"] < min_auc:
        raise GraphSeparabilityError(
            f"Graph features do NOT separate concentrated vs diversified firms: "
            f"AUC={result['auc_concentration_alone']} < {min_auc}. Result: {result}"
        )
    return result
