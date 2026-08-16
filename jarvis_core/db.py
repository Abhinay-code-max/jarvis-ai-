"""
jarvis_core/db.py
===================
Durable local decision log for B.2 (B.2.3) — conceptually the JARVIS
equivalent of the Hermes task_events/log_events pattern, reimplemented
standalone for this codebase: one durable row per queue item, updated
in place as reasoning and routing proceed. Plain stdlib sqlite3 — this
is a single-process local audit trail, not a shared multi-writer
database, so nothing heavier is warranted.

Schema (see _DDL for the authoritative source):

  decisions  — one row per queue item ever reasoned over. `queue_item_id`
               is UNIQUE — that uniqueness IS the idempotency mechanism
               B.2's reliability requirements ask for: a queue item that
               already has a row here is never reasoned over or routed
               again by this process (see engine.py._process_item).
               `status` tracks where the item is in its lifecycle:

                 pending -> reasoned -> in_progress -> {completed | resolved
                                                          | failed | waiting_approval}
                 waiting_approval -> {completed | rejected | failed}

               A row still `in_progress` at process startup is an orphan
               from a prior crash — the action may or may not have
               completed. recover_orphaned_decisions() in engine.py
               surfaces these distinctly (a warning log line, not a
               retry) rather than silently re-running them, which is
               exactly the accidental-duplicate-execution B.2's
               reliability requirements call out.

  approvals  — the B.4 placeholder store. See
               approval_interface.py's module docstring for why B.4 is
               implemented here rather than as a separate service, and
               for the explicit note that this is NOT JARVIS-XL's
               task_approval.py.

Migration mechanism: init happens via idempotent `CREATE TABLE IF NOT
EXISTS` / `CREATE INDEX IF NOT EXISTS` statements, run every time
DecisionStore is constructed — sufficient for a single, append-mostly
schema like this one and safe to re-run on every process start. If the
schema ever needs a real migration (column rename/type change), that's
the point to introduce a schema_version table; not needed yet, per B.2's
"do not over-engineer the database" instruction.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_log = logging.getLogger("jarvis_core.db")

STATUS_PENDING = "pending"
STATUS_REASONED = "reasoned"
STATUS_IN_PROGRESS = "in_progress"
STATUS_WAITING_APPROVAL = "waiting_approval"
STATUS_COMPLETED = "completed"
STATUS_RESOLVED = "resolved"
STATUS_FAILED = "failed"
STATUS_REJECTED = "rejected"


class DecisionStoreError(Exception):
    """Raised for any SQLite failure. Never swallowed — callers in
    engine.py either fail the specific decision row (if a row id
    already exists to fail) or let a startup failure abort the process,
    per B.2's 'do not swallow exceptions silently' requirement."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class DecisionRecord:
    id: int
    queue_item_id: str
    source_item: dict[str, Any]
    decision: str | None
    outcome: str | None
    reason: str | None
    action_type: str | None
    action_payload: dict[str, Any] | None
    confidence: float | None
    status: str
    error: str | None
    approval_id: str | None
    created_at: str
    updated_at: str
    completed_at: str | None


_DDL = """
CREATE TABLE IF NOT EXISTS decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    queue_item_id   TEXT NOT NULL UNIQUE,
    source_item     TEXT NOT NULL,
    decision        TEXT,
    outcome         TEXT,
    reason          TEXT,
    action_type     TEXT,
    action_payload  TEXT,
    confidence      REAL,
    status          TEXT NOT NULL,
    error           TEXT,
    approval_id     TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    completed_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_status ON decisions(status);

CREATE TABLE IF NOT EXISTS approvals (
    id               TEXT PRIMARY KEY,
    decision_id      INTEGER NOT NULL REFERENCES decisions(id),
    queue_item_id    TEXT NOT NULL,
    proposed_action  TEXT NOT NULL,
    reasoning        TEXT NOT NULL,
    context          TEXT,
    status           TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    resolved_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_approvals_status ON approvals(status);
"""


class DecisionStore:
    """Not thread-safe beyond sqlite3's own connection-level locking —
    jarvis_core is single-threaded (see poller.py's PollerLoop, which
    jarvis_core/main.py reuses unmodified), so this has never needed to
    be more than that."""

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.executescript(_DDL)
            self._conn.commit()
        except sqlite3.Error as e:
            raise DecisionStoreError(f"failed to initialize database at {db_path}: {e}") from e

    def close(self) -> None:
        self._conn.close()

    # -- decisions --------------------------------------------------

    def get_by_queue_item_id(self, queue_item_id: str) -> DecisionRecord | None:
        try:
            row = self._conn.execute(
                "SELECT * FROM decisions WHERE queue_item_id = ?", (queue_item_id,)
            ).fetchone()
        except sqlite3.Error as e:
            raise DecisionStoreError(f"failed to read decision for {queue_item_id}: {e}") from e
        return self._row_to_record(row) if row else None

    def get_by_id(self, decision_id: int) -> DecisionRecord | None:
        try:
            row = self._conn.execute("SELECT * FROM decisions WHERE id = ?", (decision_id,)).fetchone()
        except sqlite3.Error as e:
            raise DecisionStoreError(f"failed to read decision {decision_id}: {e}") from e
        return self._row_to_record(row) if row else None

    def create_pending(self, queue_item_id: str, source_item: dict[str, Any]) -> int:
        now = _now()
        try:
            cur = self._conn.execute(
                "INSERT INTO decisions (queue_item_id, source_item, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (queue_item_id, json.dumps(source_item, default=str), STATUS_PENDING, now, now),
            )
            self._conn.commit()
        except sqlite3.IntegrityError as e:
            raise DecisionStoreError(f"decision row for {queue_item_id} already exists: {e}") from e
        except sqlite3.Error as e:
            raise DecisionStoreError(f"failed to create decision row for {queue_item_id}: {e}") from e
        return cur.lastrowid

    def mark_reasoned(
        self,
        decision_id: int,
        *,
        decision: str,
        reason: str,
        action_type: str | None,
        action_payload: dict[str, Any],
        confidence: float,
    ) -> None:
        self._update(
            decision_id,
            status=STATUS_REASONED,
            decision=decision,
            reason=reason,
            action_type=action_type,
            action_payload=json.dumps(action_payload, default=str),
            confidence=confidence,
        )

    def mark_in_progress(self, decision_id: int) -> None:
        self._update(decision_id, status=STATUS_IN_PROGRESS)

    def mark_completed(self, decision_id: int, outcome: str) -> None:
        self._update(decision_id, status=STATUS_COMPLETED, outcome=outcome, completed_at=_now())

    def mark_resolved(self, decision_id: int, outcome: str = "no action required") -> None:
        self._update(decision_id, status=STATUS_RESOLVED, outcome=outcome, completed_at=_now())

    def mark_waiting_approval(self, decision_id: int, approval_id: str) -> None:
        self._update(decision_id, status=STATUS_WAITING_APPROVAL, approval_id=approval_id)

    def mark_failed(self, decision_id: int, error: str) -> None:
        self._update(decision_id, status=STATUS_FAILED, error=error, completed_at=_now())

    def mark_rejected(self, decision_id: int, outcome: str) -> None:
        self._update(decision_id, status=STATUS_REJECTED, outcome=outcome, completed_at=_now())

    def list_in_progress(self) -> list[DecisionRecord]:
        try:
            rows = self._conn.execute(
                "SELECT * FROM decisions WHERE status = ?", (STATUS_IN_PROGRESS,)
            ).fetchall()
        except sqlite3.Error as e:
            raise DecisionStoreError(f"failed to list in-progress decisions: {e}") from e
        return [self._row_to_record(r) for r in rows]

    def list_waiting_approval(self) -> list[DecisionRecord]:
        try:
            rows = self._conn.execute(
                "SELECT * FROM decisions WHERE status = ?", (STATUS_WAITING_APPROVAL,)
            ).fetchall()
        except sqlite3.Error as e:
            raise DecisionStoreError(f"failed to list waiting-approval decisions: {e}") from e
        return [self._row_to_record(r) for r in rows]

    def list_recent(self, limit: int = 200) -> list[DecisionRecord]:
        """Most-recently-created decisions first, capped at `limit`.
        Read-only — used by dashboard.py's live decisions view (B.4) and
        safe to call as often as a UI wants to poll; never mutates
        anything."""
        try:
            rows = self._conn.execute(
                "SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        except sqlite3.Error as e:
            raise DecisionStoreError(f"failed to list recent decisions: {e}") from e
        return [self._row_to_record(r) for r in rows]

    def _update(self, decision_id: int, **fields: Any) -> None:
        fields["updated_at"] = _now()
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        try:
            self._conn.execute(
                f"UPDATE decisions SET {set_clause} WHERE id = ?",
                (*fields.values(), decision_id),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            raise DecisionStoreError(f"failed to update decision {decision_id}: {e}") from e

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> DecisionRecord:
        return DecisionRecord(
            id=row["id"],
            queue_item_id=row["queue_item_id"],
            source_item=json.loads(row["source_item"]),
            decision=row["decision"],
            outcome=row["outcome"],
            reason=row["reason"],
            action_type=row["action_type"],
            action_payload=json.loads(row["action_payload"]) if row["action_payload"] else None,
            confidence=row["confidence"],
            status=row["status"],
            error=row["error"],
            approval_id=row["approval_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            completed_at=row["completed_at"],
        )

    # -- approvals (B.4 placeholder store; see approval_interface.py) --

    def create_approval(
        self,
        approval_id: str,
        decision_id: int,
        queue_item_id: str,
        proposed_action: dict[str, Any],
        reasoning: str,
        context: dict[str, Any],
    ) -> None:
        now = _now()
        try:
            self._conn.execute(
                "INSERT INTO approvals (id, decision_id, queue_item_id, proposed_action, "
                "reasoning, context, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    approval_id,
                    decision_id,
                    queue_item_id,
                    json.dumps(proposed_action, default=str),
                    reasoning,
                    json.dumps(context, default=str),
                    "pending",
                    now,
                ),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            raise DecisionStoreError(f"failed to create approval {approval_id}: {e}") from e

    def get_approval(self, approval_id: str) -> dict[str, Any] | None:
        try:
            row = self._conn.execute("SELECT * FROM approvals WHERE id = ?", (approval_id,)).fetchone()
        except sqlite3.Error as e:
            raise DecisionStoreError(f"failed to read approval {approval_id}: {e}") from e
        return dict(row) if row is not None else None

    def list_pending_approvals(self) -> list[dict[str, Any]]:
        try:
            rows = self._conn.execute("SELECT * FROM approvals WHERE status = 'pending'").fetchall()
        except sqlite3.Error as e:
            raise DecisionStoreError(f"failed to list pending approvals: {e}") from e
        return [dict(r) for r in rows]

    def set_approval_status(self, approval_id: str, status: str) -> None:
        try:
            self._conn.execute(
                "UPDATE approvals SET status = ?, resolved_at = ? WHERE id = ?",
                (status, _now(), approval_id),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            raise DecisionStoreError(f"failed to update approval {approval_id}: {e}") from e
