"""
db.py — Member D (Frontend, Data & Presentation) owns this file.

Contract with main.py (unchanged):

    init() -> None
    save_session(session_id: str, report: dict, profile: str = "Guest") -> None
    get_recent_sessions(limit: int, profile: str | None = None) -> list[dict]   newest first
    clean_profile(name: str) -> str
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
import re
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
    profile          TEXT NOT NULL DEFAULT 'Guest',
    report_json      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions (created_at);
"""

DEFAULT_PROFILE = "Guest"
PROFILE_MAX_LEN = 30


def clean_profile(name) -> str:
    """Make a profile name safe and consistent: trimmed, single spaces, letters/digits/space/_ - . only,
    at most 30 characters. Empty names become "Guest". Used for both saving and looking up history."""
    name = re.sub(r"[^\w .\-]", "", str(name or ""))
    name = re.sub(r"\s+", " ", name).strip()[:PROFILE_MAX_LEN].strip()
    return name or DEFAULT_PROFILE


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, timeout=10)


def init() -> None:
    with closing(_connect()) as conn:
        conn.executescript(_SCHEMA)
        # Upgrade databases created before the tone parameter existed.
        cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
        if "tone_score" not in cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN tone_score REAL")
        # Databases created before profiles existed: old sessions become the "Guest" profile.
        if "profile" not in cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN profile TEXT NOT NULL DEFAULT 'Guest'")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_profile ON sessions (profile COLLATE NOCASE)")
        conn.commit()


def _score(report: dict, key: str):
    value = (report.get("parameters", {}).get(key) or {}).get("score")
    return float(value) if isinstance(value, (int, float)) else None


def save_session(session_id: str, report: dict, profile: str = DEFAULT_PROFILE) -> None:
    profile = clean_profile(profile)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with closing(_connect()) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO sessions
               (session_id, created_at, duration_seconds,
                content_score, delivery_score, posture_score,
                expression_score, language_score, tone_score, profile, report_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                created_at,
                report.get("duration_seconds"),
                *(_score(report, k) for k in PARAMETERS),
                profile,
                json.dumps(report),
            ),
        )
        conn.commit()


def get_recent_sessions(limit: int = 20, profile=None) -> list:
    """Newest first. With a profile name, only that profile's sessions (case-insensitive);
    without one, every session."""
    limit = max(1, min(int(limit), 200))
    sql = "SELECT created_at, report_json FROM sessions "
    args = []
    if profile is not None:
        sql += "WHERE profile = ? COLLATE NOCASE "
        args.append(clean_profile(profile))
    sql += "ORDER BY created_at DESC, rowid DESC LIMIT ?"
    args.append(limit)
    with closing(_connect()) as conn:
        rows = conn.execute(sql, args).fetchall()

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