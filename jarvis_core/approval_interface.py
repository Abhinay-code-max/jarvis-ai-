"""
jarvis_core/approval_interface.py
===================================
B.2.2's "invoke the B.4 approval mechanism" — implemented here as the
smallest clean abstraction, because a real, separate B.4 approval
service does not exist yet in this codebase.

IMPORTANT NAMING NOTE: this is NOT JARVIS-XL's task_approval.py.
JARVIS-XL (the desktop assistant) has its own, already-completed "B.4"
— a permission/approval model for a different, unrelated codebase. This
module does not read, import, or depend on it, per this task's explicit
constraint not to revive or depend on JARVIS-XL/Hermes infrastructure.
"B.4" in this file's own comments refers only to the new, not-yet-built
EYV-agent-system approval mechanism this reasoning core is designed to
call — do not go looking for JARVIS-XL's task_approval.py as a reference
for this one; the two share a roadmap label and nothing else.

ApprovalInterface is the seam: engine.py calls only create_approval()
and get_status() against this interface, never against DecisionStore
directly for approval state. The concrete implementation below persists
approvals in the same local SQLite database DecisionStore already owns
(the `approvals` table) — reusing the one local datastore this process
already has, rather than standing up a second store for a placeholder.
When a real B.4 service exists (e.g. an HTTP API a human approves
through), swap this class for a client of that service; engine.py does
not need to change, since it only depends on the create_approval() /
get_status() contract below.

Resolving a pending approval (approve/reject) is a human operator's
call, never automated by this process. approve()/reject()/list_pending()
below are that one resolution surface — approvals_cli.py (a terminal
tool) and jarvis_core/dashboard.py (a PyQt6 GUI, B.4's "Approval UI")
both call these same three methods and add no approve/reject logic of
their own, so the two front ends can never drift apart on what
"approve" or "reject" actually does to the database.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from jarvis_core.db import DecisionStore, DecisionStoreError

_log = logging.getLogger("jarvis_core.approval_interface")


class ApprovalError(Exception):
    """Raised when an approval cannot be created or read. Never
    swallowed — engine.py treats this as a routing failure (the decision
    row is marked failed), never as silent auto-approval or a silent
    skip."""


class ApprovalInterface:
    def __init__(self, store: DecisionStore):
        self._store = store

    def create_approval(
        self,
        decision_id: int,
        queue_item_id: str,
        proposed_action: dict[str, Any],
        reasoning: str,
        context: dict[str, Any],
    ) -> str:
        """Creates a pending approval (queue item reference, proposed
        action, reasoning, context, and a created_at timestamp are all
        recorded — see db.py's `approvals` table) and returns its id.
        Never executes the proposed action itself — creating the record
        IS the entire effect, per B.2.2's 'never automatically execute
        an action classified as requiring human sign-off'."""
        approval_id = str(uuid.uuid4())
        try:
            self._store.create_approval(
                approval_id, decision_id, queue_item_id, proposed_action, reasoning, context
            )
        except DecisionStoreError as e:
            raise ApprovalError(f"failed to create approval for queue item {queue_item_id}: {e}") from e
        _log.info(
            "Created approval %s for queue item %s — awaiting operator review "
            "(see 'python -m jarvis_core.approvals_cli list')",
            approval_id, queue_item_id,
        )
        return approval_id

    def get_status(self, approval_id: str) -> str | None:
        """Returns 'pending' / 'approved' / 'rejected', or None if the
        approval id is unknown."""
        try:
            approval = self._store.get_approval(approval_id)
        except DecisionStoreError as e:
            raise ApprovalError(f"failed to read approval {approval_id}: {e}") from e
        return approval["status"] if approval else None

    def list_pending(self) -> list[dict[str, Any]]:
        """Approvals currently awaiting sign-off — the data source for
        both approvals_cli.py's `list` command and dashboard.py's
        Pending Approvals view."""
        try:
            return self._store.list_pending_approvals()
        except DecisionStoreError as e:
            raise ApprovalError(f"failed to list pending approvals: {e}") from e

    def approve(self, approval_id: str) -> None:
        """Marks a pending approval 'approved'. Raises ApprovalError if
        the id is unknown or the write fails. Does not itself execute
        anything — engine.py's resume_waiting_approvals() is what reads
        this status change and performs the originally-proposed action
        on jarvis_core's next poll cycle (see that method's docstring).
        This is the one call site both front ends use to approve —
        neither reimplements the status transition."""
        self._resolve(approval_id, "approved")

    def reject(self, approval_id: str) -> None:
        """Marks a pending approval 'rejected'. Same single-call-site
        reasoning as approve() above — engine.py picks this up on its
        next poll and marks the underlying decision rejected without
        executing anything."""
        self._resolve(approval_id, "rejected")

    def _resolve(self, approval_id: str, status: str) -> None:
        try:
            existing = self._store.get_approval(approval_id)
        except DecisionStoreError as e:
            raise ApprovalError(f"failed to read approval {approval_id}: {e}") from e
        if existing is None:
            raise ApprovalError(f"no approval found with id {approval_id}")
        try:
            self._store.set_approval_status(approval_id, status)
        except DecisionStoreError as e:
            raise ApprovalError(f"failed to update approval {approval_id}: {e}") from e
        _log.info("Approval %s marked %s", approval_id, status)
