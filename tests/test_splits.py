import numpy as np
import pandas as pd

from src.models.splits import pseudo_oot_split


def test_split_is_strictly_ordered_in_pseudo_time():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"SK_ID_CURR": rng.permutation(np.arange(100000, 101000))})
    split = pseudo_oot_split(df)

    train_ids = df.loc[split == "train", "SK_ID_CURR"]
    calib_ids = df.loc[split == "calib", "SK_ID_CURR"]
    test_ids = df.loc[split == "test", "SK_ID_CURR"]

    # No calib/test row may precede the latest training row: this is the
    # temporal-leakage guarantee that replaces random k-fold (plan.md §15.2).
    assert train_ids.max() < calib_ids.min()
    assert calib_ids.max() < test_ids.min()

    # All rows are assigned, in roughly the configured proportions.
    assert len(train_ids) + len(calib_ids) + len(test_ids) == len(df)
    assert abs(len(train_ids) / len(df) - 0.70) < 0.01
    assert abs(len(calib_ids) / len(df) - 0.10) < 0.01
