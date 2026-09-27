"""
db.py — Member D (Frontend, Data & Presentation) owns this file.

Contract with main.py:

    init() -> None                          create the DB/table if missing
    save_session(session_id: str, report: dict) -> None
    get_recent_sessions(limit: int) -> list[dict]   backs the progress view
    is_ready() -> bool                      used by /api/health

TODO(Member D): replace with a real SQLite table (session_id, timestamp,
full report JSON, and pulled-out per-parameter scores for the
progress-over-time chart). This stub keeps everything in memory so
main.py and the frontend can be developed against it before the real
schema is ready — swap the internals, keep the four function signatures.
"""

_sessions = []  # STUB: in-memory list, cleared on restart


def init() -> None:
    # TODO(Member D): sqlite3.connect("sessions.db"), CREATE TABLE IF NOT EXISTS ...
    pass


def save_session(session_id: str, report: dict) -> None:
    # TODO(Member D): INSERT into SQLite instead of appending to a list.
    _sessions.append(report)


def get_recent_sessions(limit: int = 20) -> list:
    # TODO(Member D): SELECT ... ORDER BY timestamp DESC LIMIT ?
    return _sessions[-limit:][::-1]


def is_ready() -> bool:
    return True
