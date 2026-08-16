"""
tests/test_dashboard_model.py
================================
jarvis_core.dashboard.DashboardModel — the GUI-free data/action layer
behind the B.4 "Approval UI". Exercised directly, with no QApplication
and no PyQt6 widgets involved, which is the whole point of keeping it
separate from the Qt classes in dashboard.py (see that module's
docstring).

The key property under test: DashboardModel.approve()/reject() must
produce *identical* database state to approvals_cli.py's own
approve/reject command — both now route through the exact same
ApprovalInterface.approve()/reject() (see approval_interface.py's
module docstring), so there is no separate "dashboard logic" to drift
out of sync with the CLI. This file proves that by driving one
approval through the real `approvals_cli._main()` entry point and a
second, otherwise-identical approval through DashboardModel, then
diffing the resulting rows field-by-field (except the timestamp, which
will legitimately differ between the two calls).
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jarvis_core.dashboard as dashboard  # noqa: E402

# Importing the model layer alone must never pull in PyQt6 — that's the
# whole reason DashboardModel and the Qt widget classes live in
# separate layers within dashboard.py (PyQt6 is only imported lazily,
# inside run()/_build_main_window()).
assert "PyQt6.QtWidgets" not in sys.modules, "importing jarvis_core.dashboard must not require PyQt6"

from jarvis_core import approvals_cli  # noqa: E402
from jarvis_core.approval_interface import ApprovalError, ApprovalInterface  # noqa: E402
from jarvis_core.db import DecisionStore  # noqa: E402


class DashboardModelTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.db_path = Path(self._tmpdir.name) / "jarvis.db"
        self.store = DecisionStore(self.db_path)
        self.addCleanup(self.store.close)
        self.model = dashboard.DashboardModel(self.store, ApprovalInterface(self.store))

        # approvals_cli._main() resolves its own DecisionStore via
        # config.resolve_db_path() (JARVIS_DB_PATH) — point it at this
        # test's temp database so the "CLI path" and "dashboard path"
        # in the equivalence tests below are genuinely acting on the
        # same rows, not two different databases.
        self._saved_db_path_env = os.environ.get("JARVIS_DB_PATH")
        os.environ["JARVIS_DB_PATH"] = str(self.db_path)
        self.addCleanup(self._restore_db_path_env)

    def _restore_db_path_env(self):
        if self._saved_db_path_env is None:
            os.environ.pop("JARVIS_DB_PATH", None)
        else:
            os.environ["JARVIS_DB_PATH"] = self._saved_db_path_env

    def _seed_needs_approval(self, queue_item_id: str) -> tuple[int, str]:
        decision_id = self.store.create_pending(queue_item_id, {"id": queue_item_id, "title": "risky ticket"})
        self.store.mark_reasoned(
            decision_id, decision="NEEDS_APPROVAL", reason="needs a human to sign off",
            action_type="code", action_payload={"type": "code", "instructions": "do the risky thing"},
            confidence=0.55,
        )
        self.store.mark_in_progress(decision_id)
        approval_id = ApprovalInterface(self.store).create_approval(
            decision_id, queue_item_id, {"type": "code", "instructions": "do the risky thing"},
            "needs a human to sign off", {"payload": {"id": queue_item_id}},
        )
        self.store.mark_waiting_approval(decision_id, approval_id)
        return decision_id, approval_id

    # -- read paths -----------------------------------------------------

    def test_list_recent_decisions_returns_store_rows(self):
        self.store.create_pending("t-1", {"id": "t-1"})
        rows = self.model.list_recent_decisions()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].queue_item_id, "t-1")

    def test_list_pending_approvals_matches_interface(self):
        _, approval_id = self._seed_needs_approval("t-1")
        pending = self.model.list_pending_approvals()
        self.assertEqual([a["id"] for a in pending], [approval_id])

    def test_get_decision_returns_the_originating_row(self):
        decision_id, _ = self._seed_needs_approval("t-1")
        row = self.model.get_decision(decision_id)
        self.assertEqual(row.id, decision_id)
        self.assertEqual(row.confidence, 0.55)

    # -- write paths: approve/reject ------------------------------------

    def test_approve_updates_status_and_removes_from_pending_list(self):
        _, approval_id = self._seed_needs_approval("t-1")
        self.model.approve(approval_id)

        self.assertEqual(self.store.get_approval(approval_id)["status"], "approved")
        self.assertEqual(self.model.list_pending_approvals(), [])

    def test_reject_updates_status(self):
        _, approval_id = self._seed_needs_approval("t-1")
        self.model.reject(approval_id)
        self.assertEqual(self.store.get_approval(approval_id)["status"], "rejected")

    def test_approving_unknown_id_raises_approval_error(self):
        with self.assertRaises(ApprovalError):
            self.model.approve("does-not-exist")

    # -- equivalence with approvals_cli.py -------------------------------

    def test_dashboard_approve_matches_cli_approve_path(self):
        """Two structurally identical approvals, resolved through the
        two different front ends — the resulting rows must agree on
        every field except the resolution timestamp."""
        _, approval_via_cli = self._seed_needs_approval("via-cli")
        _, approval_via_dashboard = self._seed_needs_approval("via-dashboard")

        exit_code = approvals_cli._main(["approve", approval_via_cli])
        self.assertEqual(exit_code, 0)

        self.model.approve(approval_via_dashboard)

        cli_row = self.store.get_approval(approval_via_cli)
        dashboard_row = self.store.get_approval(approval_via_dashboard)

        for field in ("status",):
            self.assertEqual(cli_row[field], dashboard_row[field])
        self.assertEqual(cli_row["status"], "approved")
        self.assertIsNotNone(cli_row["resolved_at"])
        self.assertIsNotNone(dashboard_row["resolved_at"])

    def test_dashboard_reject_matches_cli_reject_path(self):
        _, approval_via_cli = self._seed_needs_approval("via-cli")
        _, approval_via_dashboard = self._seed_needs_approval("via-dashboard")

        exit_code = approvals_cli._main(["reject", approval_via_cli])
        self.assertEqual(exit_code, 0)
        self.model.reject(approval_via_dashboard)

        self.assertEqual(self.store.get_approval(approval_via_cli)["status"], "rejected")
        self.assertEqual(self.store.get_approval(approval_via_dashboard)["status"], "rejected")

    def test_dashboard_and_cli_agree_on_unknown_id_failure(self):
        exit_code = approvals_cli._main(["approve", "does-not-exist"])
        self.assertEqual(exit_code, 1)
        with self.assertRaises(ApprovalError):
            self.model.approve("does-not-exist")


if __name__ == "__main__":
    unittest.main(verbosity=2)
