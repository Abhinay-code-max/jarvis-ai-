"""
jarvis_core/voice/store.py
=============================
Durable local transcript log for the voice layer — deliberately its own
small store (own sqlite3 connection, own DDL, own exception type)
rather than an extension of jarvis_core/db.py's DecisionStore, per this
feature's own constraint not to touch db.py's internals or the
decisions/approvals tables.

Same on-disk file as jarvis_core's other local state
(jarvis_core.config.resolve_db_path(), same JARVIS_DB_PATH) — one
SQLite file per install, not a second database file to manage — but a
completely separate table (`voice_turns`) and connection object, so
nothing here can corrupt or lock out DecisionStore's own transactions.

Deliberately does NOT store raw audio, only transcript text, per this
feature's explicit scope: "transcript text is what's durable, not raw
audio."
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_DDL = """
CREATE TABLE IF NOT EXISTS voice_turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    text        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_voice_turns_session ON voice_turns(session_id);
"""


class VoiceStoreError(Exception):
    """Raised for any SQLite failure. Never swallowed at the point it
    occurs — see session.py for why a failure here is logged but does
    not abort an in-progress voice turn."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class VoiceTurn:
    id: int
    session_id: str
    role: str
    text: str
    created_at: str


class VoiceStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.executescript(_DDL)
            self._conn.commit()
        except sqlite3.Error as e:
            raise VoiceStoreError(f"failed to initialize voice_turns table at {db_path}: {e}") from e

    def close(self) -> None:
        self._conn.close()

    def record_turn(self, session_id: str, role: str, text: str) -> int:
        try:
            cur = self._conn.execute(
                "INSERT INTO voice_turns (session_id, role, text, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, text, _now()),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            raise VoiceStoreError(f"failed to record {role} turn for session {session_id}: {e}") from e
        return cur.lastrowid

    def list_session_turns(self, session_id: str) -> list[VoiceTurn]:
        try:
            rows = self._conn.execute(
                "SELECT * FROM voice_turns WHERE session_id = ? ORDER BY id ASC", (session_id,)
            ).fetchall()
        except sqlite3.Error as e:
            raise VoiceStoreError(f"failed to list turns for session {session_id}: {e}") from e
        return [VoiceTurn(**dict(r)) for r in rows]

    def list_recent_turns(self, limit: int = 200) -> list[VoiceTurn]:
        try:
            rows = self._conn.execute(
                "SELECT * FROM voice_turns ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        except sqlite3.Error as e:
            raise VoiceStoreError(f"failed to list recent turns: {e}") from e
        return [VoiceTurn(**dict(r)) for r in rows]
