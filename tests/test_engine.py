"""
tests/test_engine.py
======================
JarvisEngine (B.2 reasoning + routing) exercised against hand-written
stubs for every external collaborator — Claude, B.3 (Claude Code),
/jarvis/decisions (Bob), and B.4 (approvals). Never against the live
Claude API or a real HTTP endpoint, per this codebase's standing rule
(see tests/test_claude_reasoner.py's module docstring). Same
"hand-written stub over mocking framework" convention eyv_poller's own
tests use (see tests/test_poller.py's _StubClient).

Covers every decision path from B.2.2 (NEEDS_CODE, MARKETING_ACTION,
RESOLVED, NEEDS_APPROVAL), the malformed-response and external-failure
cases from B.2's "Reliability requirements", the low-confidence
override, idempotency (no duplicate execution for an already-decided
queue item), and orphan recovery at startup.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eyv_poller.schema import QueueItem, QueueSnapshot  # noqa: E402

from jarvis_core.approval_interface import ApprovalError, ApprovalInterface  # noqa: E402
from jarvis_core.claude_reasoner import ClaudeReasonerError  # noqa: E402
from jarvis_core.code_interface import ExecutionResult  # noqa: E402
from jarvis_core.db import DecisionStore  # noqa: E402
from jarvis_core.decision_schema import DecisionValidationError, ReasoningDecision  # noqa: E402
from jarvis_core.engine import JarvisEngine  # noqa: E402
from jarvis_core.marketing_client import MarketingClientUnavailable  # noqa: E402


class _StubReasoner:
    """reason() returns/raises whatever the next entry in `plan` says,
    then holds on the last entry for any extra calls — same shape as
    eyv_poller's own _StubClient."""

    def __init__(self, plan):
        self._plan = list(plan)
        self.calls = []

    def reason(self, context):
        self.calls.append(context)
        step = self._plan[min(len(self.calls), len(self._plan)) - 1]
        if isinstance(step, Exception):
            raise step
        return step


class _StubCodeInterface:
    def __init__(self, result):
        self._result = result
        self.calls = []

    def submit(self, instructions, context):
        self.calls.append((instructions, context))
        return self._result


class _StubMarketingClient:
    def __init__(self, error=None):
        self._error = error
        self.calls = []

    def post_decision(self, payload):
        self.calls.append(payload)
        if self._error is not None:
            raise self._error


class _FailingApprovalInterface:
    """Every call raises — used to test the approval-creation-failure path."""

    def create_approval(self, *args, **kwargs):
        raise ApprovalError("db write failed")

    def get_status(self, approval_id):
        return None


def _decision(decision_type, action=None, confidence=0.95, reason="because"):
    action = action or {}
    return ReasoningDecision(
        decision=decision_type,
        reason=reason,
        action_type=action.get("type"),
        action_payload=action,
        confidence=confidence,
        raw={"decision": decision_type},
    )


class JarvisEngineTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.store = DecisionStore(Path(self._tmpdir.name) / "jarvis.db")
        self.addCleanup(self.store.close)

    def _engine(self, reasoner, code=None, marketing=None, approval=None, threshold=0.7):
        return JarvisEngine(
            store=self.store,
            reasoner=reasoner,
            code_interface=code or _StubCodeInterface(ExecutionResult(True, "done")),
            marketing_client=marketing or _StubMarketingClient(),
            approval_interface=approval or ApprovalInterface(self.store),
            confidence_threshold=threshold,
        )

    @staticmethod
    def _snapshot(item_id="1", raw=None):
        return QueueSnapshot(items=[QueueItem(item_id=item_id, raw=raw or {"id": item_id})])

    # -- NEEDS_CODE -----------------------------------------------------

    def test_needs_code_routes_to_code_interface_and_completes(self):
        code = _StubCodeInterface(ExecutionResult(True, "implemented"))
        engine = self._engine(
            _StubReasoner([_decision("NEEDS_CODE", {"type": "code", "instructions": "add X"})]),
            code=code,
        )
        engine.process_queue_snapshot(self._snapshot())

        self.assertEqual(len(code.calls), 1)
        self.assertEqual(code.calls[0][0], "add X")
        row = self.store.get_by_queue_item_id("1")
        self.assertEqual(row.decision, "NEEDS_CODE")
        self.assertEqual(row.status, "completed")
        self.assertIn("implemented", row.outcome)

    def test_needs_code_execution_failure_is_recorded_not_hidden(self):
        code = _StubCodeInterface(ExecutionResult(False, "compile error"))
        engine = self._engine(
            _StubReasoner([_decision("NEEDS_CODE", {"type": "code", "instructions": "add X"})]),
            code=code,
        )
        engine.process_queue_snapshot(self._snapshot())

        row = self.store.get_by_queue_item_id("1")
        self.assertEqual(row.status, "failed")
        self.assertIn("compile error", row.error)

    # -- MARKETING_ACTION -------------------------------------------------

    def test_marketing_action_posts_to_decisions_endpoint(self):
        marketing = _StubMarketingClient()
        engine = self._engine(
            _StubReasoner([_decision("MARKETING_ACTION", {"type": "marketing", "instructions": "tweet it"})]),
            marketing=marketing,
        )
        engine.process_queue_snapshot(self._snapshot())

        self.assertEqual(len(marketing.calls), 1)
        self.assertEqual(marketing.calls[0]["queue_item_id"], "1")
        self.assertEqual(marketing.calls[0]["action"]["instructions"], "tweet it")
        row = self.store.get_by_queue_item_id("1")
        self.assertEqual(row.status, "completed")

    def test_marketing_action_http_failure_is_recorded_not_hidden(self):
        marketing = _StubMarketingClient(error=MarketingClientUnavailable("EYV down"))
        engine = self._engine(
            _StubReasoner([_decision("MARKETING_ACTION", {"type": "marketing"})]),
            marketing=marketing,
        )
        engine.process_queue_snapshot(self._snapshot())

        row = self.store.get_by_queue_item_id("1")
        self.assertEqual(row.status, "failed")
        self.assertIn("EYV down", row.error)

    # -- RESOLVED -----------------------------------------------------------

    def test_resolved_marks_row_resolved_with_no_external_call(self):
        code = _StubCodeInterface(ExecutionResult(True, "should not be called"))
        marketing = _StubMarketingClient()
        engine = self._engine(_StubReasoner([_decision("RESOLVED")]), code=code, marketing=marketing)
        engine.process_queue_snapshot(self._snapshot())

        self.assertEqual(code.calls, [])
        self.assertEqual(marketing.calls, [])
        row = self.store.get_by_queue_item_id("1")
        self.assertEqual(row.status, "resolved")

    # -- NEEDS_APPROVAL -------------------------------------------------

    def test_needs_approval_creates_approval_and_never_auto_executes(self):
        code = _StubCodeInterface(ExecutionResult(True, "should not run yet"))
        engine = self._engine(
            _StubReasoner([_decision("NEEDS_APPROVAL", {"type": "code", "instructions": "risky change"})]),
            code=code,
        )
        engine.process_queue_snapshot(self._snapshot())

        self.assertEqual(code.calls, [])
        row = self.store.get_by_queue_item_id("1")
        self.assertEqual(row.status, "waiting_approval")
        self.assertIsNotNone(row.approval_id)
        approval = self.store.get_approval(row.approval_id)
        self.assertEqual(approval["status"], "pending")
        self.assertEqual(approval["queue_item_id"], "1")

    def test_approved_decision_resumes_and_executes_on_next_poll(self):
        code = _StubCodeInterface(ExecutionResult(True, "implemented after approval"))
        engine = self._engine(
            _StubReasoner([_decision("NEEDS_APPROVAL", {"type": "code", "instructions": "risky change"})]),
            code=code,
        )
        engine.process_queue_snapshot(self._snapshot())
        row = self.store.get_by_queue_item_id("1")
        self.store.set_approval_status(row.approval_id, "approved")

        # Any subsequent poll checks waiting approvals before new items.
        engine.process_queue_snapshot(self._snapshot(item_id="2", raw={"id": "2"}))

        row = self.store.get_by_queue_item_id("1")
        self.assertEqual(row.status, "completed")
        self.assertEqual(len(code.calls), 1)

    def test_rejected_decision_marks_row_rejected_without_executing(self):
        code = _StubCodeInterface(ExecutionResult(True, "should never run"))
        engine = self._engine(
            _StubReasoner([_decision("NEEDS_APPROVAL", {"type": "code", "instructions": "risky change"})]),
            code=code,
        )
        engine.process_queue_snapshot(self._snapshot())
        row = self.store.get_by_queue_item_id("1")
        self.store.set_approval_status(row.approval_id, "rejected")

        engine.process_queue_snapshot(self._snapshot(item_id="2", raw={"id": "2"}))

        row = self.store.get_by_queue_item_id("1")
        self.assertEqual(row.status, "rejected")
        self.assertEqual(code.calls, [])

    def test_still_pending_approval_is_left_untouched(self):
        code = _StubCodeInterface(ExecutionResult(True, "should never run"))
        engine = self._engine(
            _StubReasoner([_decision("NEEDS_APPROVAL", {"type": "code", "instructions": "risky change"})]),
            code=code,
        )
        engine.process_queue_snapshot(self._snapshot())
        engine.process_queue_snapshot(self._snapshot(item_id="2", raw={"id": "2"}))  # still pending

        row = self.store.get_by_queue_item_id("1")
        self.assertEqual(row.status, "waiting_approval")
        self.assertEqual(code.calls, [])

    # -- low-confidence override ----------------------------------------

    def test_low_confidence_forces_needs_approval(self):
        engine = self._engine(
            _StubReasoner([_decision("NEEDS_CODE", {"type": "code", "instructions": "x"}, confidence=0.4)]),
            threshold=0.7,
        )
        engine.process_queue_snapshot(self._snapshot())

        row = self.store.get_by_queue_item_id("1")
        self.assertEqual(row.status, "waiting_approval")
        self.assertEqual(row.decision, "NEEDS_APPROVAL")
        self.assertIn("low-confidence override", row.reason)
        self.assertIn("NEEDS_CODE", row.reason)

    def test_confidence_at_or_above_threshold_is_not_overridden(self):
        engine = self._engine(
            _StubReasoner([_decision("RESOLVED", confidence=0.7)]),
            threshold=0.7,
        )
        engine.process_queue_snapshot(self._snapshot())

        row = self.store.get_by_queue_item_id("1")
        self.assertEqual(row.decision, "RESOLVED")
        self.assertEqual(row.status, "resolved")

    def test_needs_approval_is_never_overridden_regardless_of_confidence(self):
        engine = self._engine(
            _StubReasoner([_decision("NEEDS_APPROVAL", confidence=0.1)]),
            threshold=0.7,
        )
        engine.process_queue_snapshot(self._snapshot())

        row = self.store.get_by_queue_item_id("1")
        self.assertNotIn("low-confidence override", row.reason)

    # -- malformed / external failures -----------------------------------

    def test_malformed_claude_response_is_recorded_as_failed_not_silently_continued(self):
        engine = self._engine(_StubReasoner([DecisionValidationError("bad decision value")]))
        engine.process_queue_snapshot(self._snapshot())

        row = self.store.get_by_queue_item_id("1")
        self.assertEqual(row.status, "failed")
        self.assertIn("malformed", row.error)
        self.assertIn("bad decision value", row.error)

    def test_claude_api_failure_is_recorded_as_failed(self):
        engine = self._engine(_StubReasoner([ClaudeReasonerError("network timeout")]))
        engine.process_queue_snapshot(self._snapshot())

        row = self.store.get_by_queue_item_id("1")
        self.assertEqual(row.status, "failed")
        self.assertIn("network timeout", row.error)

    def test_unexpected_reasoning_exception_is_recorded_not_raised(self):
        engine = self._engine(_StubReasoner([RuntimeError("totally unanticipated bug")]))
        engine.process_queue_snapshot(self._snapshot())  # must not raise

        row = self.store.get_by_queue_item_id("1")
        self.assertEqual(row.status, "failed")
        self.assertIn("unexpected", row.error)

    def test_approval_creation_failure_is_recorded_as_failed(self):
        engine = self._engine(
            _StubReasoner([_decision("NEEDS_APPROVAL", {"type": "code"})]),
            approval=_FailingApprovalInterface(),
        )
        engine.process_queue_snapshot(self._snapshot())

        row = self.store.get_by_queue_item_id("1")
        self.assertEqual(row.status, "failed")
        self.assertIn("db write failed", row.error)

    def test_one_bad_item_does_not_block_the_rest_of_the_snapshot(self):
        reasoner = _StubReasoner([ClaudeReasonerError("boom"), _decision("RESOLVED")])
        engine = self._engine(reasoner)
        snapshot = QueueSnapshot(items=[
            QueueItem(item_id="bad", raw={"id": "bad"}),
            QueueItem(item_id="good", raw={"id": "good"}),
        ])
        engine.process_queue_snapshot(snapshot)

        self.assertEqual(self.store.get_by_queue_item_id("bad").status, "failed")
        self.assertEqual(self.store.get_by_queue_item_id("good").status, "resolved")

    # -- idempotency ------------------------------------------------------

    def test_queue_item_already_decided_is_not_reprocessed(self):
        reasoner = _StubReasoner([_decision("RESOLVED"), _decision("NEEDS_CODE", {"type": "code"})])
        engine = self._engine(reasoner)

        engine.process_queue_snapshot(self._snapshot())
        engine.process_queue_snapshot(self._snapshot())  # same item_id="1" again

        self.assertEqual(len(reasoner.calls), 1)
        self.assertEqual(self.store.get_by_queue_item_id("1").decision, "RESOLVED")

    def test_item_with_no_id_gets_a_stable_synthesized_id(self):
        reasoner = _StubReasoner([_decision("RESOLVED")])
        engine = self._engine(reasoner)
        snapshot = QueueSnapshot(items=[QueueItem(item_id=None, raw={"title": "no id ticket"})])

        engine.process_queue_snapshot(snapshot)
        engine.process_queue_snapshot(snapshot)  # identical raw content again

        self.assertEqual(len(reasoner.calls), 1)  # second poll recognized it as already-decided

    # -- orphan recovery ---------------------------------------------------

    def test_recover_orphaned_decisions_logs_without_retrying_or_changing_status(self):
        decision_id = self.store.create_pending("orphan-1", {"id": "orphan-1"})
        self.store.mark_in_progress(decision_id)

        engine = self._engine(_StubReasoner([]))
        engine.recover_orphaned_decisions()  # must not raise

        row = self.store.get_by_queue_item_id("orphan-1")
        self.assertEqual(row.status, "in_progress")  # left untouched, not auto-retried

    def test_recover_orphaned_decisions_is_a_noop_when_nothing_is_in_progress(self):
        engine = self._engine(_StubReasoner([]))
        engine.recover_orphaned_decisions()  # must not raise on an empty store


if __name__ == "__main__":
    unittest.main(verbosity=2)
