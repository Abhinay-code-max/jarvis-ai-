"""
tests/test_db.py
===================
jarvis_core.db.DecisionStore against a real temp-directory SQLite file
(same "exercise a real component over a mock" convention as
eyv_poller's own tests — see tests/test_client_resilience.py). Covers
the parts B.2.3 explicitly asks the database to answer: what did JARVIS
decide, which queue item caused it, when, what action was attempted,
did it succeed/fail/why, and is it waiting for approval — plus the
UNIQUE-constraint idempotency mechanism engine.py relies on.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis_core.db import DecisionStore, DecisionStoreError  # noqa: E402


class DecisionStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.store = DecisionStore(Path(self._tmpdir.name) / "nested" / "jarvis.db")
        self.addCleanup(self.store.close)

    def test_db_file_is_created_including_missing_parent_dirs(self):
        self.assertTrue((Path(self._tmpdir.name) / "nested" / "jarvis.db").exists())

    def test_create_pending_then_get_by_queue_item_id_round_trips(self):
        decision_id = self.store.create_pending("t-1", {"id": "t-1", "kind": "bug"})
        row = self.store.get_by_queue_item_id("t-1")
        self.assertEqual(row.id, decision_id)
        self.assertEqual(row.queue_item_id, "t-1")
        self.assertEqual(row.source_item, {"id": "t-1", "kind": "bug"})
        self.assertEqual(row.status, "pending")
        self.assertIsNone(row.decision)

    def test_unknown_queue_item_id_returns_none(self):
        self.assertIsNone(self.store.get_by_queue_item_id("does-not-exist"))

    def test_duplicate_queue_item_id_is_rejected(self):
        self.store.create_pending("t-1", {"id": "t-1"})
        with self.assertRaises(DecisionStoreError):
            self.store.create_pending("t-1", {"id": "t-1"})

    def test_full_lifecycle_needs_code_completed(self):
        decision_id = self.store.create_pending("t-1", {"id": "t-1"})
        self.store.mark_reasoned(
            decision_id, decision="NEEDS_CODE", reason="fix the bug",
            action_type="code", action_payload={"type": "code", "instructions": "fix it"},
            confidence=0.95,
        )
        self.store.mark_in_progress(decision_id)
        self.store.mark_completed(decision_id, "implemented and merged")

        row = self.store.get_by_id(decision_id)
        self.assertEqual(row.decision, "NEEDS_CODE")
        self.assertEqual(row.action_type, "code")
        self.assertEqual(row.confidence, 0.95)
        self.assertEqual(row.status, "completed")
        self.assertEqual(row.outcome, "implemented and merged")
        self.assertIsNotNone(row.completed_at)

    def test_mark_failed_records_the_error(self):
        decision_id = self.store.create_pending("t-1", {"id": "t-1"})
        self.store.mark_failed(decision_id, "Claude API timed out")
        row = self.store.get_by_id(decision_id)
        self.assertEqual(row.status, "failed")
        self.assertEqual(row.error, "Claude API timed out")
        self.assertIsNotNone(row.completed_at)

    def test_list_in_progress_finds_only_in_progress_rows(self):
        a = self.store.create_pending("a", {"id": "a"})
        b = self.store.create_pending("b", {"id": "b"})
        self.store.create_pending("c", {"id": "c"})  # stays pending
        self.store.mark_in_progress(a)
        self.store.mark_in_progress(b)
        self.store.mark_completed(b, "done")

        in_progress = {r.queue_item_id for r in self.store.list_in_progress()}
        self.assertEqual(in_progress, {"a"})

    def test_list_waiting_approval_finds_only_those_rows(self):
        a = self.store.create_pending("a", {"id": "a"})
        self.store.create_pending("b", {"id": "b"})
        self.store.mark_waiting_approval(a, "approval-1")

        waiting = self.store.list_waiting_approval()
        self.assertEqual(len(waiting), 1)
        self.assertEqual(waiting[0].queue_item_id, "a")
        self.assertEqual(waiting[0].approval_id, "approval-1")

    # -- approvals ----------------------------------------------------

    def test_create_and_read_approval(self):
        decision_id = self.store.create_pending("t-1", {"id": "t-1"})
        self.store.create_approval(
            "approval-1", decision_id, "t-1",
            {"type": "code", "instructions": "risky"}, "needs sign-off", {"payload": {"id": "t-1"}},
        )
        approval = self.store.get_approval("approval-1")
        self.assertEqual(approval["status"], "pending")
        self.assertEqual(approval["queue_item_id"], "t-1")

    def test_set_approval_status_updates_and_stamps_resolved_at(self):
        decision_id = self.store.create_pending("t-1", {"id": "t-1"})
        self.store.create_approval("approval-1", decision_id, "t-1", {}, "reason", {})
        self.store.set_approval_status("approval-1", "approved")
        approval = self.store.get_approval("approval-1")
        self.assertEqual(approval["status"], "approved")
        self.assertIsNotNone(approval["resolved_at"])

    def test_list_pending_approvals_excludes_resolved_ones(self):
        d1 = self.store.create_pending("t-1", {"id": "t-1"})
        d2 = self.store.create_pending("t-2", {"id": "t-2"})
        self.store.create_approval("approval-1", d1, "t-1", {}, "r1", {})
        self.store.create_approval("approval-2", d2, "t-2", {}, "r2", {})
        self.store.set_approval_status("approval-2", "rejected")

        pending = self.store.list_pending_approvals()
        self.assertEqual([a["id"] for a in pending], ["approval-1"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
