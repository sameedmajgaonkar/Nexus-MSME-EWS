"""Phase 8 point-in-time feature store — offline/online split (plan.md §13.1).

Feast is the named production feature store (plan.md §13.1). This module is a
deliberately thin store that honours the same contract Feast enforces — an
OFFLINE store (historical snapshot, for training) strictly separated from an
ONLINE store (low-latency key-value lookup, for serving) — without pulling in
Feast's dependency tree. Swapping in real Feast (offline: parquet source,
online: Redis via the docker-compose 'redis' service) is a definition-file
change, not an API change: `materialize` / `get_online_features` mirror
Feast's own verbs.

Offline store : data/processed/serving_features_enriched.parquet — the Phase 1
                structured + Phase 5 graph-lite/text feature snapshot, one row
                per SK_ID_CURR. Point-in-time correctness holds trivially here
                because the snapshot is a single origination-time cut (no
                future rows exist to leak).
Online store  : SQLite (data/online_store.db by default; ONLINE_STORE_DB env
                var overrides), one row per loan holding the feature payload
                as JSON plus updated_at. This is THE serving path: every
                lookup is logged and counted (see SOURCE_COUNTS) so "the API
                reads the online store, not a recomputation" is provable.
Streaming     : `update_features` merges event-driven deltas into the online
                row (Phase 11 feature-refresh consumer, plan.md §14).

The TARGET label column is never materialized into the online store — the
serving path must not expose (or be able to accidentally consume) the label.
"""

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OFFLINE_PATH = ROOT / "data" / "processed" / "serving_features_enriched.parquet"
DEFAULT_ONLINE_DB = ROOT / "data" / "online_store.db"
KEY_COL = "SK_ID_CURR"
EXCLUDE_FROM_ONLINE = ("TARGET",)  # labels never enter the serving path

logger = logging.getLogger("feature_store")

# Proof-of-path counters (Phase 8 verify item): every get_online_features call
# increments exactly one of these.
SOURCE_COUNTS = {"online": 0, "offline_fallback": 0}

_OFFLINE_CACHE: dict[str, pd.DataFrame] = {}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS online_features (
    loan_id INTEGER PRIMARY KEY,
    features TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _online_db_path() -> Path:
    return Path(os.environ.get("ONLINE_STORE_DB", DEFAULT_ONLINE_DB))


def _connect() -> sqlite3.Connection:
    path = _online_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(_SCHEMA)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_offline() -> pd.DataFrame:
    key = str(OFFLINE_PATH)
    if key not in _OFFLINE_CACHE:
        df = pd.read_parquet(OFFLINE_PATH)
        _OFFLINE_CACHE[key] = df.set_index(KEY_COL, drop=False)
    return _OFFLINE_CACHE[key]


def _json_scalar(obj):
    """json.dumps fallback for numpy scalars (np.int64/np.bool_ are not JSON-native)."""
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def _row_to_payload(row: pd.Series) -> dict:
    """One offline row -> JSON-safe feature dict (NaN -> None, labels dropped)."""
    out = {}
    for col, val in row.items():
        if col in EXCLUDE_FROM_ONLINE:
            continue
        if pd.isna(val):
            out[col] = None
        elif hasattr(val, "item"):  # numpy scalar -> python scalar
            out[col] = val.item()
        else:
            out[col] = val
    return out


def online_count() -> int:
    with _connect() as conn:
        return int(conn.execute("SELECT COUNT(*) FROM online_features").fetchone()[0])


def materialize(loan_ids: list[int] | None = None, limit: int | None = None) -> int:
    """Bulk-load offline -> online (Feast's `materialize` verb).

    loan_ids restricts to a subset (the API materializes the scoreable
    demo book at startup); limit caps rows for tests. Returns rows written.
    """
    t0 = time.perf_counter()
    offline = _load_offline()
    if loan_ids is not None:
        idx = offline.index.intersection(pd.Index(loan_ids))
        offline = offline.loc[idx]
    if limit is not None:
        offline = offline.head(limit)
    now = _now()
    sub = offline.drop(columns=[c for c in EXCLUDE_FROM_ONLINE if c in offline.columns])
    records = sub.astype(object).where(sub.notna(), None).to_dict("records")
    rows = [
        (int(rec[KEY_COL]), json.dumps(rec, default=_json_scalar), now) for rec in records
    ]
    with _connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO online_features (loan_id, features, updated_at) "
            "VALUES (?, ?, ?)",
            rows,
        )
    logger.info(
        "materialized %d rows offline->online in %.1fs", len(rows), time.perf_counter() - t0
    )
    return len(rows)


def get_online_features(loan_id: int) -> dict:
    """THE serving read path: online store first, offline write-through on miss.

    Raises KeyError for a loan unknown to both stores. Source + latency are
    logged and counted so the Phase 8 "online store, not recomputation"
    verification is assertable (see SOURCE_COUNTS / get_source_counts)."""
    t0 = time.perf_counter()
    with _connect() as conn:
        row = conn.execute(
            "SELECT features FROM online_features WHERE loan_id = ?", (int(loan_id),)
        ).fetchone()
    if row is not None:
        SOURCE_COUNTS["online"] += 1
        logger.info(
            "features loan=%s source=online %.2fms",
            loan_id,
            (time.perf_counter() - t0) * 1000,
        )
        return json.loads(row[0])

    offline = _load_offline()
    if int(loan_id) not in offline.index:
        raise KeyError(f"loan_id {loan_id} not found in offline or online feature store")
    payload = _row_to_payload(offline.loc[int(loan_id)])
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO online_features (loan_id, features, updated_at) "
            "VALUES (?, ?, ?)",
            (int(loan_id), json.dumps(payload), _now()),
        )
    SOURCE_COUNTS["offline_fallback"] += 1
    logger.info(
        "features loan=%s source=offline_fallback (write-through) %.2fms",
        loan_id,
        (time.perf_counter() - t0) * 1000,
    )
    return payload


def update_features(loan_id: int, updates: dict) -> dict:
    """Merge streaming feature deltas into the online row (plan.md §14).

    The loan is pulled through get_online_features first (write-through on
    miss), so an event on a never-scored loan still lands correctly."""
    payload = get_online_features(loan_id)
    payload.update(updates)
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO online_features (loan_id, features, updated_at) "
            "VALUES (?, ?, ?)",
            (int(loan_id), json.dumps(payload), _now()),
        )
    logger.info("features loan=%s updated cols=%s", loan_id, sorted(updates))
    return payload


def get_source_counts() -> dict:
    return dict(SOURCE_COUNTS)


def reset_source_counts() -> None:
    SOURCE_COUNTS["online"] = 0
    SOURCE_COUNTS["offline_fallback"] = 0
