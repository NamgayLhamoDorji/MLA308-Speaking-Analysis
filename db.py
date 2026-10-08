"""
db.py — Member D (Frontend, Data & Presentation) owns this file.

Contract with main.py (unchanged):

    init() -> None
    save_session(session_id: str, report: dict) -> None
    get_recent_sessions(limit: int) -> list[dict]   newest first
    is_ready() -> bool

Storage: one SQLite file (sessions.db, next to main.py). Each row keeps
the full report as JSON plus the six per-parameter scores pulled out
into their own columns.

Change: added the "tone" parameter (voice intonation). Existing
sessions.db files are upgraded automatically (tone_score column is added);
old sessions simply have no tone score.

Thread-safety: every call opens its own short-lived connection.
"""

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "sessions.db"

PARAMETERS = ("content", "delivery", "posture", "expression", "language", "tone")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id       TEXT PRIMARY KEY,
    created_at       TEXT NOT NULL,           -- ISO-8601, UTC
    duration_seconds REAL,
    content_score    REAL,
    delivery_score   REAL,
    posture_score    REAL,
    expression_score REAL,
    language_score   REAL,
    tone_score       REAL,
    report_json      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions (created_at);
"""


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, timeout=10)


def init() -> None:
    with closing(_connect()) as conn:
        conn.executescript(_SCHEMA)
        # Upgrade databases created before the tone parameter existed.
        cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
        if "tone_score" not in cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN tone_score REAL")
        conn.commit()


def _score(report: dict, key: str):
    value = (report.get("parameters", {}).get(key) or {}).get("score")
    return float(value) if isinstance(value, (int, float)) else None


def save_session(session_id: str, report: dict) -> None:
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with closing(_connect()) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO sessions
               (session_id, created_at, duration_seconds,
                content_score, delivery_score, posture_score,
                expression_score, language_score, tone_score, report_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                created_at,
                report.get("duration_seconds"),
                *(_score(report, k) for k in PARAMETERS),
                json.dumps(report),
            ),
        )
        conn.commit()


def get_recent_sessions(limit: int = 20) -> list:
    limit = max(1, min(int(limit), 200))
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT created_at, report_json FROM sessions "
            "ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (limit,),
        ).fetchall()

    sessions = []
    for created_at, report_json in rows:
        report = json.loads(report_json)
        report["created_at"] = created_at
        sessions.append(report)
    return sessions


def is_ready() -> bool:
    try:
        with closing(_connect()) as conn:
            conn.execute("SELECT 1 FROM sessions LIMIT 1")
        return True
    except sqlite3.Error:
        return False