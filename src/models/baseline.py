"""Phase 2 baseline: WOE-binned logistic regression scorecard (plan.md §9.1).

WOE binners are fit on the training window only, then applied to calib/test —
fitting them on the full table would leak future information into the bins.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.features.woe import WOE_CANDIDATE_FEATURES, fit_woe_transform


def train_baseline_scorecard(
    features: pd.DataFrame,
    split: pd.Series,
    target_col: str = "TARGET",
    woe_features: list[str] | None = None,
):
    """Returns (model, binners, scores) where scores is a Series of PDs for all rows."""
    woe_features = woe_features or WOE_CANDIDATE_FEATURES
    train_df = features[split == "train"]

    _, binners = fit_woe_transform(train_df, target_col=target_col, features=woe_features)

    woe_all = pd.DataFrame(index=features.index)
    for feat in woe_features:
        woe_all[f"{feat}_woe"] = binners[feat].transform(features[feat].values, metric="woe")

    model = LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", max_iter=1000)
    model.fit(woe_all[split == "train"], train_df[target_col])

    scores = pd.Series(model.predict_proba(woe_all)[:, 1], index=features.index, name="baseline_pd")
    return model, binners, scores
