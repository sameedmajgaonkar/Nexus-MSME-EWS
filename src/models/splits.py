"""Out-of-time split utility (plan.md §15.2 — random k-fold is banned project-wide).

Home Credit's application_train has no absolute application date (all DAYS_*
columns are relative to application). SK_ID_CURR is assigned sequentially, so
sorted-ID order is used as a documented pseudo-temporal proxy. With real
sandbox data this swaps to the true application timestamp — the interface
(train / calib / test masks, strictly ordered) stays identical.
"""

import pandas as pd

TRAIN_FRAC = 0.70
CALIB_FRAC = 0.10  # calibration window sits between train and test in pseudo-time


def pseudo_oot_split(
    df: pd.DataFrame,
    id_col: str = "SK_ID_CURR",
    train_frac: float = TRAIN_FRAC,
    calib_frac: float = CALIB_FRAC,
) -> pd.Series:
    """Return a Series aligned to df.index with values 'train' | 'calib' | 'test'."""
    ids = df[id_col].sort_values().to_numpy()
    n = len(ids)
    train_end_id = ids[int(n * train_frac) - 1]
    calib_end_id = ids[int(n * (train_frac + calib_frac)) - 1]

    split = pd.Series("test", index=df.index, name="split")
    split[df[id_col] <= train_end_id] = "train"
    split[(df[id_col] > train_end_id) & (df[id_col] <= calib_end_id)] = "calib"
    return split
