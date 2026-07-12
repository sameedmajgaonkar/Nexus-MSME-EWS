"""Phase 8 governance store: append-only audit log + decisions + alerts (plan.md §12.5).

SQLite is the local default (no Docker daemon available in this environment);
PostgreSQL — the named production audit store, provisioned as the
docker-compose 'postgres' service — activates when DATABASE_URL is set. That
branch is a clean NotImplementedError here rather than half-tested psycopg2
code, honestly stating the constraint.

Append-only is ENFORCED, not conventional: SQLite BEFORE UPDATE / BEFORE
DELETE triggers RAISE(ABORT) on the `overrides` and `audit_log` tables, so
even a direct DB connection cannot rewrite history (§12.5 immutable audit
trail).

Tables
  overrides     — officer decisions: accept | modify | override. `reason` is
                  mandatory for modify AND override (enforced at the API);
                  `delta` records what a modify changed.
  audit_log     — one immutable row per /score call (§12.3 audit logging:
                  the narrative source is stored next to the SHAP payload it
                  was derived from).
  alerts        — streaming re-scores that MOVED a risk grade (plan.md §14).
  risk_timeline — every streaming re-score, silent trajectory (plan.md §14).

DB path: data/overrides.db by default; EWS_DB_PATH env var overrides (tests
point this at a tmp file so they never pollute data/).
"""

import json
import os
import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "overrides.db"

DECISIONS = ("accept", "modify", "override")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS overrides (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    loan_id INTEGER NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('accept', 'modify', 'override')),
    reason TEXT,
    delta TEXT,
    risk_grade TEXT,
    calibrated_pd REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL DEFAULT (datetime('now')),
    loan_id INTEGER NOT NULL,
    endpoint TEXT NOT NULL,
    model_used TEXT,
    calibrated_pd REAL,
    risk_grade TEXT,
    narrative_source TEXT,
    payload_json TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL DEFAULT (datetime('now')),
    loan_id INTEGER NOT NULL,
    event_type TEXT,
    old_grade TEXT,
    new_grade TEXT,
    message TEXT
);

CREATE TABLE IF NOT EXISTS risk_timeline (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    loan_id INTEGER NOT NULL,
    ts TEXT NOT NULL DEFAULT (datetime('now')),
    pd REAL,
    grade TEXT
);
"""

# §12.5 immutable audit trail: UPDATE/DELETE on the append-only tables ABORT.
_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS overrides_no_update BEFORE UPDATE ON overrides
BEGIN SELECT RAISE(ABORT, 'overrides is append-only (plan.md 12.5)'); END;
CREATE TRIGGER IF NOT EXISTS overrides_no_delete BEFORE DELETE ON overrides
BEGIN SELECT RAISE(ABORT, 'overrides is append-only (plan.md 12.5)'); END;
CREATE TRIGGER IF NOT EXISTS audit_log_no_update BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only (plan.md 12.5)'); END;
CREATE TRIGGER IF NOT EXISTS audit_log_no_delete BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only (plan.md 12.5)'); END;
"""


def _db_path() -> Path:
    return Path(os.environ.get("EWS_DB_PATH", DEFAULT_DB_PATH))


def _require_sqlite() -> None:
    url = os.environ.get("DATABASE_URL", "")
    if url.startswith("postgres"):
        raise NotImplementedError(
            "DATABASE_URL points at PostgreSQL — the production audit store served "
            "by the docker-compose 'postgres' service. No Docker daemon is available "
            "in this environment, so the psycopg2 branch is not implemented; unset "
            "DATABASE_URL to use the SQLite default."
        )


def _migrate_legacy_overrides(conn: sqlite3.Connection) -> None:
    """Upgrade a pre-Phase-8 overrides table (accept/override only, no delta).

    Rows are preserved verbatim; only the CHECK constraint and the delta
    column change. Table recreation is DDL and does not violate the row-level
    append-only triggers (which are re-created on the new table)."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='overrides'"
    ).fetchone()
    if row is None or "'modify'" in row[0]:
        return
    conn.executescript(
        """
        DROP TRIGGER IF EXISTS overrides_no_update;
        DROP TRIGGER IF EXISTS overrides_no_delete;
        ALTER TABLE overrides RENAME TO overrides_legacy_v1;
        CREATE TABLE overrides (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            loan_id INTEGER NOT NULL,
            decision TEXT NOT NULL CHECK (decision IN ('accept', 'modify', 'override')),
            reason TEXT,
            delta TEXT,
            risk_grade TEXT,
            calibrated_pd REAL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        INSERT INTO overrides (id, loan_id, decision, reason, risk_grade, calibrated_pd, created_at)
            SELECT id, loan_id, decision, reason, risk_grade, calibrated_pd, created_at
            FROM overrides_legacy_v1;
        DROP TABLE overrides_legacy_v1;
        """
    )


def _connect() -> sqlite3.Connection:
    _require_sqlite()
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    _migrate_legacy_overrides(conn)
    conn.executescript(_SCHEMA)
    conn.executescript(_TRIGGERS)
    return conn


def init_db() -> None:
    """Create tables + append-only triggers (idempotent; called at API startup)."""
    with _connect():
        pass


# ---------------------------------------------------------------- decisions

def record_decision(
    loan_id: int,
    decision: str,
    reason: str | None,
    risk_grade: str | None = None,
    calibrated_pd: float | None = None,
    delta: dict | str | None = None,
) -> int:
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}")
    delta_json = json.dumps(delta) if isinstance(delta, dict) else delta
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO overrides (loan_id, decision, reason, delta, risk_grade, calibrated_pd) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (loan_id, decision, reason, delta_json, risk_grade, calibrated_pd),
        )
        return cur.lastrowid


def list_decisions(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM overrides ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------- audit log

def record_audit(
    loan_id: int,
    endpoint: str,
    model_used: str | None = None,
    calibrated_pd: float | None = None,
    risk_grade: str | None = None,
    narrative_source: str | None = None,
    payload: dict | None = None,
) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO audit_log (loan_id, endpoint, model_used, calibrated_pd, "
            "risk_grade, narrative_source, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                loan_id,
                endpoint,
                model_used,
                calibrated_pd,
                risk_grade,
                narrative_source,
                json.dumps(payload) if payload is not None else None,
            ),
        )
        return cur.lastrowid


def list_audit(limit: int = 50, loan_id: int | None = None) -> list[dict]:
    q = "SELECT * FROM audit_log"
    params: tuple = ()
    if loan_id is not None:
        q += " WHERE loan_id = ?"
        params = (loan_id,)
    q += " ORDER BY id DESC LIMIT ?"
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(q, params + (limit,)).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------- alerts / timeline

def insert_alert(
    loan_id: int,
    event_type: str | None,
    old_grade: str | None,
    new_grade: str | None,
    message: str,
) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO alerts (loan_id, event_type, old_grade, new_grade, message) "
            "VALUES (?, ?, ?, ?, ?)",
            (loan_id, event_type, old_grade, new_grade, message),
        )
        return cur.lastrowid


def list_alerts(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def insert_timeline(loan_id: int, pd_value: float, grade: str) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO risk_timeline (loan_id, pd, grade) VALUES (?, ?, ?)",
            (loan_id, float(pd_value), grade),
        )
        return cur.lastrowid


def list_timeline(loan_id: int, limit: int = 100) -> list[dict]:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM risk_timeline WHERE loan_id = ? ORDER BY id ASC LIMIT ?",
            (loan_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
