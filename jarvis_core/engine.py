"""
jarvis_core/engine.py
=======================
B.2's reasoning-and-routing core (B.2.1 + B.2.2). Owns exactly one
thing per queue item: reason -> validate -> (maybe override for low
confidence) -> record -> route -> record outcome. Every step that can
fail is caught here and recorded as a `failed` (or, where appropriate,
left untouched) decision row rather than raised past this module — see
eyv_poller/poller.py's own docstring for why a single bad item must
never be allowed to stop the whole poll loop, a guarantee this module
extends down to the level of a single queue item.

Idempotency: process_queue_snapshot() checks DecisionStore for an
existing row for each queue_item_id BEFORE calling Claude. If one
exists, the item is skipped entirely (logged at debug level) — this is
what satisfies "prevent accidental duplicate execution" for the normal
case. The other half of that requirement — a row left `in_progress` by
a process that crashed mid-item — is handled separately by
recover_orphaned_decisions(), which only logs (distinctly, at startup)
and never auto-retries; see its own docstring.

Confidence gate: any decision with confidence below
config.confidence_threshold is forced to NEEDS_APPROVAL regardless of
what Claude originally classified it as. The original decision and
confidence are preserved in the recorded `reason` as context for the
human reviewer, never silently dropped.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from eyv_poller.schema import QueueSnapshot

from jarvis_core.approval_interface import ApprovalError, ApprovalInterface
from jarvis_core.claude_reasoner import ClaudeReasoner, ClaudeReasonerError
from jarvis_core.code_interface import CodeInterface
from jarvis_core.db import DecisionRecord, DecisionStore, DecisionStoreError
from jarvis_core.decision_schema import DecisionValidationError, ReasoningDecision
from jarvis_core.marketing_client import MarketingClientError, MarketingDecisionClient

_log = logging.getLogger("jarvis_core.engine")


class JarvisEngine:
    def __init__(
        self,
        store: DecisionStore,
        reasoner: ClaudeReasoner,
        code_interface: CodeInterface,
        marketing_client: MarketingDecisionClient,
        approval_interface: ApprovalInterface,
        confidence_threshold: float,
    ):
        self._store = store
        self._reasoner = reasoner
        self._code = code_interface
        self._marketing = marketing_client
        self._approval = approval_interface
        self._confidence_threshold = confidence_threshold

    # -- startup recovery -------------------------------------------

    def recover_orphaned_decisions(self) -> None:
        """Call once at process startup, before the first poll. Logs
        (distinctly, at warning level) any decision row left
        'in_progress' by a prior crash — the action it was routing to
        may or may not have completed. Deliberately does NOT auto-retry
        or auto-resolve these: guessing either way risks exactly the
        duplicate-execution B.2's reliability requirements warn against.
        An operator has to look at these and decide (e.g. by checking
        whether the underlying code change / marketing post actually
        happened, then updating the row by hand)."""
        try:
            orphans = self._store.list_in_progress()
        except DecisionStoreError:
            _log.exception("Could not check for orphaned in-progress decisions at startup")
            return
        if not orphans:
            return
        for row in orphans:
            _log.warning(
                "ORPHANED DECISION on startup: id=%d queue_item_id=%s decision=%s action_type=%s "
                "— an action may have been attempted before a prior crash and its outcome is "
                "unknown. Status left unchanged pending manual review; not retried automatically.",
                row.id, row.queue_item_id, row.decision, row.action_type,
            )

    # -- main entry point ---------------------------------------------

    def process_queue_snapshot(self, snapshot: QueueSnapshot) -> None:
        """Wired as eyv_poller's PollerLoop `process` callback (see
        main.py) — called once per successful poll with whatever items
        EYV returned. First checks whether any previously-created
        approvals have been resolved and resumes those decisions, then
        reasons over and routes every new item in `snapshot`. One item's
        failure never stops the rest — poller.py's own try/except around
        this whole call is the outer safety net; the per-item try/except
        below is the inner one, so a single bad item can't abort the
        batch either."""
        self.resume_waiting_approvals()

        for item in snapshot.items:
            queue_item_id = item.item_id or self._synthesize_id(item.raw)
            try:
                self._process_item(queue_item_id, item.raw)
            except Exception:
                _log.exception(
                    "Unexpected error processing queue item %s — continuing with next item",
                    queue_item_id,
                )

    @staticmethod
    def _synthesize_id(raw: dict[str, Any]) -> str:
        """QueueItem.item_id is best-effort (see eyv_poller/schema.py) —
        for the rare item with no recognizable id field, derive a stable
        one from its content so idempotency still holds across polls for
        an otherwise-identical item."""
        digest = hashlib.sha256(json.dumps(raw, sort_keys=True, default=str).encode()).hexdigest()[:16]
        return f"unidentified:{digest}"

    def _process_item(self, queue_item_id: str, raw_item: dict[str, Any]) -> None:
        existing = self._store.get_by_queue_item_id(queue_item_id)
        if existing is not None:
            _log.debug(
                "Skipping queue item %s — already has a decision row (status=%s)",
                queue_item_id, existing.status,
            )
            return

        try:
            decision_id = self._store.create_pending(queue_item_id, raw_item)
        except DecisionStoreError:
            _log.exception("Could not create a decision row for queue item %s — skipping this poll", queue_item_id)
            return

        try:
            decision = self._reasoner.reason({"queue_item_id": queue_item_id, "payload": raw_item})
        except ClaudeReasonerError as e:
            _log.error("Claude reasoning failed for queue item %s: %s", queue_item_id, e)
            self._safe_mark_failed(decision_id, f"reasoning failed: {e}")
            return
        except DecisionValidationError as e:
            _log.error("Claude returned a malformed decision for queue item %s: %s", queue_item_id, e)
            self._safe_mark_failed(decision_id, f"malformed Claude response: {e}")
            return
        except Exception as e:
            _log.exception("Unexpected error reasoning over queue item %s", queue_item_id)
            self._safe_mark_failed(decision_id, f"unexpected reasoning error: {e}")
            return

        decision = self._apply_confidence_gate(decision)

        try:
            self._store.mark_reasoned(
                decision_id,
                decision=decision.decision,
                reason=decision.reason,
                action_type=decision.action_type,
                action_payload=decision.action_payload,
                confidence=decision.confidence,
            )
        except DecisionStoreError:
            _log.exception("Could not record the reasoned decision for queue item %s — not routing", queue_item_id)
            return

        self._route(decision_id, queue_item_id, raw_item, decision)

    def _apply_confidence_gate(self, decision: ReasoningDecision) -> ReasoningDecision:
        if decision.decision == "NEEDS_APPROVAL" or decision.confidence >= self._confidence_threshold:
            return decision
        _log.info(
            "Confidence %.2f below threshold %.2f — forcing NEEDS_APPROVAL (originally %s)",
            decision.confidence, self._confidence_threshold, decision.decision,
        )
        overridden_reason = (
            f"[low-confidence override: originally classified {decision.decision} at "
            f"confidence {decision.confidence:.2f}, below the {self._confidence_threshold:.2f} "
            f"threshold] {decision.reason}"
        )
        return ReasoningDecision(
            decision="NEEDS_APPROVAL",
            reason=overridden_reason,
            action_type=decision.action_type,
            action_payload=decision.action_payload,
            confidence=decision.confidence,
            raw=decision.raw,
        )

    def _route(self, decision_id: int, queue_item_id: str, raw_item: dict[str, Any], decision: ReasoningDecision) -> None:
        self._safe_mark_in_progress(decision_id)

        if decision.decision == "RESOLVED":
            self._safe_mark_resolved(decision_id)
            return

        if decision.decision == "NEEDS_CODE":
            instructions = decision.action_payload.get("instructions") or decision.reason
            result = self._code.submit(instructions, {"queue_item_id": queue_item_id, "payload": raw_item})
            if result.success:
                self._safe_mark_completed(decision_id, result.detail)
            else:
                self._safe_mark_failed(decision_id, result.detail)
            return

        if decision.decision == "MARKETING_ACTION":
            payload = {
                "queue_item_id": queue_item_id,
                "action": decision.action_payload,
                "reason": decision.reason,
                "context": raw_item,
            }
            try:
                self._marketing.post_decision(payload)
            except MarketingClientError as e:
                _log.error("Failed to POST /jarvis/decisions for queue item %s: %s", queue_item_id, e)
                self._safe_mark_failed(decision_id, f"marketing POST failed: {e}")
                return
            self._safe_mark_completed(decision_id, "posted to /jarvis/decisions for Bob")
            return

        if decision.decision == "NEEDS_APPROVAL":
            try:
                approval_id = self._approval.create_approval(
                    decision_id,
                    queue_item_id,
                    decision.action_payload,
                    decision.reason,
                    {"queue_item_id": queue_item_id, "payload": raw_item},
                )
            except ApprovalError as e:
                _log.error("Failed to create approval for queue item %s: %s", queue_item_id, e)
                self._safe_mark_failed(decision_id, f"approval creation failed: {e}")
                return
            self._safe_mark_waiting_approval(decision_id, approval_id)
            return

        # Unreachable given decision_schema's validation (only the four
        # known decision strings pass parse_decision), but never
        # silently drop an unrecognized decision type either.
        self._safe_mark_failed(decision_id, f"no route for decision type {decision.decision!r}")

    # -- approval resumption --------------------------------------------

    def resume_waiting_approvals(self) -> None:
        """Called at the start of every poll cycle. For each decision
        still 'waiting_approval', checks whether its B.4 approval has
        since been resolved and, if so, resumes it: an 'approved'
        approval executes the originally-proposed action (via B.3 for a
        `code` action, via Bob for a `marketing` action, or is simply
        marked resolved if the proposal carried no executable action —
        e.g. a purely human sign-off); a 'rejected' approval marks the
        decision rejected without executing anything. Approvals still
        'pending' are left completely untouched — never automatically
        executed, per B.2.2's 'never automatically execute an action
        Claude classified as requiring human sign-off'."""
        try:
            waiting = self._store.list_waiting_approval()
        except DecisionStoreError:
            _log.exception("Could not list waiting-approval decisions")
            return

        for row in waiting:
            if not row.approval_id:
                continue
            try:
                status = self._approval.get_status(row.approval_id)
            except ApprovalError:
                _log.exception("Could not check approval status for decision %d — leaving it waiting", row.id)
                continue

            if status in (None, "pending"):
                continue
            if status == "rejected":
                self._safe_mark_rejected(row.id, "approval rejected by operator")
                continue
            if status == "approved":
                self._resume_approved(row)
            else:
                _log.warning("Unrecognized approval status %r for decision %d — leaving it waiting", status, row.id)

    def _resume_approved(self, row: DecisionRecord) -> None:
        action_type = row.action_type
        action_payload = row.action_payload or {}

        if action_type == "code":
            instructions = action_payload.get("instructions") or row.reason or ""
            result = self._code.submit(instructions, {"queue_item_id": row.queue_item_id, "payload": row.source_item})
            if result.success:
                self._safe_mark_completed(row.id, f"approved, then executed via B.3: {result.detail}")
            else:
                self._safe_mark_failed(row.id, f"approved, but B.3 execution failed: {result.detail}")
            return

        if action_type == "marketing":
            payload = {
                "queue_item_id": row.queue_item_id,
                "action": action_payload,
                "reason": row.reason,
                "context": row.source_item,
            }
            try:
                self._marketing.post_decision(payload)
            except MarketingClientError as e:
                _log.error("Approved marketing action failed to POST for queue item %s: %s", row.queue_item_id, e)
                self._safe_mark_failed(row.id, f"approved, but marketing POST failed: {e}")
                return
            self._safe_mark_completed(row.id, "approved, then posted to /jarvis/decisions for Bob")
            return

        # No systematically-executable action type on the proposal —
        # the approval itself was the entire outcome (e.g. sign-off on a
        # purely human action). Nothing further for JARVIS to execute.
        self._safe_mark_completed(row.id, "approved by operator — no further automated action required")

    # -- DecisionStore write wrappers ------------------------------------
    # A failure to WRITE the outcome of routing must never look like a
    # crash of the whole poll (poller.py already guards the outer call),
    # but it also must never be silently swallowed — every failure here
    # is logged with a full traceback.

    def _safe_mark_in_progress(self, decision_id: int) -> None:
        try:
            self._store.mark_in_progress(decision_id)
        except DecisionStoreError:
            _log.exception("Could not mark decision %d in_progress", decision_id)

    def _safe_mark_completed(self, decision_id: int, outcome: str) -> None:
        try:
            self._store.mark_completed(decision_id, outcome)
        except DecisionStoreError:
            _log.exception("Could not mark decision %d completed", decision_id)

    def _safe_mark_resolved(self, decision_id: int) -> None:
        try:
            self._store.mark_resolved(decision_id)
        except DecisionStoreError:
            _log.exception("Could not mark decision %d resolved", decision_id)

    def _safe_mark_failed(self, decision_id: int, error: str) -> None:
        try:
            self._store.mark_failed(decision_id, error)
        except DecisionStoreError:
            _log.exception("Could not mark decision %d failed (original error: %s)", decision_id, error)

    def _safe_mark_waiting_approval(self, decision_id: int, approval_id: str) -> None:
        try:
            self._store.mark_waiting_approval(decision_id, approval_id)
        except DecisionStoreError:
            _log.exception("Could not mark decision %d waiting_approval", decision_id)

    def _safe_mark_rejected(self, decision_id: int, outcome: str) -> None:
        try:
            self._store.mark_rejected(decision_id, outcome)
        except DecisionStoreError:
            _log.exception("Could not mark decision %d rejected", decision_id)
