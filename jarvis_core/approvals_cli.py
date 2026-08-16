"""
jarvis_core/approvals_cli.py
==============================
Minimal terminal tool for resolving B.4 approvals — a thin wrapper over
ApprovalInterface's list_pending()/approve()/reject(). jarvis_core/
dashboard.py (the PyQt6 "Approval UI") calls those exact same three
methods, so this CLI and the dashboard can never drift apart on what
"approve" or "reject" actually does; neither reimplements the status
transition itself (see approval_interface.py's module docstring).

Usage:
    python -m jarvis_core.approvals_cli list
    python -m jarvis_core.approvals_cli approve <approval_id>
    python -m jarvis_core.approvals_cli reject <approval_id>

Approving/rejecting here only updates the approval's own status —
actually resuming the underlying decision (executing the originally
proposed NEEDS_CODE/MARKETING_ACTION, or just marking a non-executable
approval resolved) happens the next time jarvis_core polls
(JarvisEngine.resume_waiting_approvals(), called at the start of every
poll cycle — see engine.py). This tool only ever touches the local
decision database, so it resolves just JARVIS_DB_PATH
(config.resolve_db_path()) rather than the rest of B.2's configuration
— it has no reason to require a Claude API key just to approve or
reject something.
"""
from __future__ import annotations

import sys

from jarvis_core.approval_interface import ApprovalError, ApprovalInterface
from jarvis_core.config import resolve_db_path
from jarvis_core.db import DecisionStore, DecisionStoreError


def _main(argv: list[str]) -> int:
    try:
        store = DecisionStore(resolve_db_path())
    except DecisionStoreError as e:
        print(f"Database error: {e}", file=sys.stderr)
        return 1

    approval_interface = ApprovalInterface(store)
    try:
        if len(argv) == 1 and argv[0] == "list":
            pending = approval_interface.list_pending()
            if not pending:
                print("No pending approvals.")
                return 0
            for approval in pending:
                print(f"{approval['id']}  queue_item={approval['queue_item_id']}  created_at={approval['created_at']}")
                print(f"    reasoning: {approval['reasoning']}")
            return 0

        if len(argv) == 2 and argv[0] in ("approve", "reject"):
            approval_id = argv[1]
            try:
                if argv[0] == "approve":
                    approval_interface.approve(approval_id)
                else:
                    approval_interface.reject(approval_id)
            except ApprovalError as e:
                print(str(e), file=sys.stderr)
                return 1
            status = "approved" if argv[0] == "approve" else "rejected"
            print(f"Approval {approval_id} marked {status}.")
            return 0

        print("Usage: python -m jarvis_core.approvals_cli {list|approve <id>|reject <id>}")
        return 1
    except ApprovalError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
