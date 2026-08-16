"""
jarvis_core/dashboard.py
==========================
B.4's "Approval UI" — a local PyQt6 dashboard over jarvis_core's
existing SQLite decision log (db.py) and the approval_interface.py
seam. Launched independently of jarvis_core's main polling/reasoning
process:

    python -m jarvis_core          # background: poll -> reason -> route
    python -m jarvis_core.dashboard  # separate: viewer + approval-actioner

Same separation JARVIS-XL keeps between its UI (ui.py) and its
background execution (main.py) — this file does not import from or
modify ui.py, only reuses the same PyQt6 library for stack consistency,
per this task's own instruction.

Read-only safety (by construction, not just convention): this module
imports nothing from claude_reasoner.py, code_interface.py,
marketing_client.py, engine.py, or eyv_poller — there is no code path
here that can call the Claude API, execute a code change, POST to
/jarvis/decisions, or run a queue poll. The only writes this file can
ever make are ApprovalInterface.approve()/reject() on an *existing*
pending approval; engine.py's decision/confidence logic is never
touched or reachable from here.

Two layers, deliberately kept separate so the data/action logic is
testable without a Qt event loop:

  DashboardModel  — plain Python. Wraps DecisionStore (reads) and
                    ApprovalInterface (reads + the two approve/reject
                    writes). No PyQt6 import — this is what
                    tests/test_dashboard_model.py exercises directly,
                    and it is the exact same approve()/reject() call
                    path approvals_cli.py uses (via ApprovalInterface),
                    so there is nowhere for the two front ends to drift
                    apart on what "approve"/"reject" does.

  MainWindow, DecisionsTab, ApprovalsTab  — PyQt6 widgets. Poll
                    DashboardModel on a QTimer and render it; every
                    button click calls straight into DashboardModel
                    with no intermediate approve/reject logic of its
                    own.
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Any

from jarvis_core.approval_interface import ApprovalError, ApprovalInterface
from jarvis_core.config import resolve_db_path
from jarvis_core.db import DecisionRecord, DecisionStore, DecisionStoreError

_log = logging.getLogger("jarvis_core.dashboard")

DEFAULT_REFRESH_MS = 3000
DEFAULT_DECISION_LIMIT = 200

# Decision-row statuses the dashboard flags as "needs attention" rather
# than a normal terminal outcome. `in_progress` in particular covers
# both an item genuinely mid-flight in a live jarvis_core process AND a
# crash orphan (see engine.py's recover_orphaned_decisions()) — the
# database carries no separate orphan flag, and this dashboard doesn't
# add one (that would mean touching engine.py's own bookkeeping), so a
# persistently `in_progress` row is exactly the signal an operator
# should look at, whichever of the two it turns out to be.
ATTENTION_STATUSES = {"in_progress", "failed", "waiting_approval"}


class DashboardModel:
    """Read + approval-action layer with no GUI dependency."""

    def __init__(self, store: DecisionStore, approval_interface: ApprovalInterface):
        self._store = store
        self._approval = approval_interface

    def list_recent_decisions(self, limit: int = DEFAULT_DECISION_LIMIT) -> list[DecisionRecord]:
        return self._store.list_recent(limit)

    def list_pending_approvals(self) -> list[dict[str, Any]]:
        return self._approval.list_pending()

    def get_decision(self, decision_id: int) -> DecisionRecord | None:
        """Looks up the decision row an approval was created for — used
        to show its confidence alongside the approval (the approvals
        table itself doesn't duplicate that column; it's already on the
        decision row via `decision_id`)."""
        return self._store.get_by_id(decision_id)

    def approve(self, approval_id: str) -> None:
        """Identical call ApprovalInterface.approve() makes for
        approvals_cli.py's `approve` command — see that module and
        approval_interface.py's module docstring."""
        self._approval.approve(approval_id)

    def reject(self, approval_id: str) -> None:
        """Identical call ApprovalInterface.reject() makes for
        approvals_cli.py's `reject` command."""
        self._approval.reject(approval_id)

    def close(self) -> None:
        self._store.close()


def _fmt_confidence(confidence: float | None) -> str:
    return f"{confidence:.2f}" if confidence is not None else ""


def _fmt_outcome(row: DecisionRecord) -> str:
    if row.error:
        return f"ERROR: {row.error}"
    if row.outcome:
        return row.outcome
    return ""


def _pretty_json(value: Any) -> str:
    try:
        return json.dumps(value, indent=2, default=str)
    except (TypeError, ValueError):
        return str(value)


def run() -> int:
    """Entry point for `python -m jarvis_core.dashboard`."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    db_path = resolve_db_path()
    try:
        store = DecisionStore(db_path)
    except DecisionStoreError as e:
        print(f"Could not open the decision database at {db_path}: {e}", file=sys.stderr)
        return 1

    # Imported lazily so `import jarvis_core.dashboard` (e.g. from tests
    # exercising DashboardModel only) never requires PyQt6 to be
    # installed, and this module's data/action layer stays importable
    # in headless environments.
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    model = DashboardModel(store, ApprovalInterface(store))
    window_cls = _build_main_window()
    window = window_cls(model, db_path)
    window.show()
    exit_code = app.exec()
    model.close()
    return exit_code


def _build_main_window():
    """Defines the PyQt6 widget classes on first use, so importing this
    module never requires PyQt6 unless a caller actually wants the GUI
    (run() or MainWindow below)."""
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QColor
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    # Loosely dark/cyan, in the same family as JARVIS-XL's ui.py palette
    # for visual consistency — defined independently here, not imported
    # from ui.py (per this task's constraint not to touch that file).
    BG = "#00060a"
    PANEL = "#010d14"
    BORDER = "#0d3347"
    PRI = "#00d4ff"
    TEXT = "#c8f8ff"
    TEXT_DIM = "#5ab8cc"
    GREEN = "#00cc70"
    RED = "#ff3355"
    AMBER = "#ffb020"
    GRAY = "#3a5a66"

    _STATUS_COLORS = {
        "pending": GRAY,
        "reasoned": GRAY,
        "in_progress": AMBER,
        "waiting_approval": PRI,
        "completed": GREEN,
        "resolved": GREEN,
        "failed": RED,
        "rejected": TEXT_DIM,
    }

    STYLESHEET = f"""
        QMainWindow, QWidget {{ background: {BG}; color: {TEXT}; }}
        QTableWidget {{
            background: {PANEL}; color: {TEXT}; gridline-color: {BORDER};
            border: 1px solid {BORDER};
        }}
        QHeaderView::section {{
            background: {PANEL}; color: {PRI}; border: 1px solid {BORDER};
            padding: 4px;
        }}
        QTabWidget::pane {{ border: 1px solid {BORDER}; }}
        QTabBar::tab {{
            background: {PANEL}; color: {TEXT_DIM}; padding: 8px 16px;
            border: 1px solid {BORDER};
        }}
        QTabBar::tab:selected {{ color: {PRI}; }}
        QTextEdit {{ background: {PANEL}; color: {TEXT}; border: 1px solid {BORDER}; }}
        QLabel#heading {{ color: {PRI}; font-weight: bold; font-size: 13pt; }}
        QPushButton {{
            background: {PANEL}; color: {TEXT}; border: 1px solid {BORDER};
            padding: 4px 12px; border-radius: 3px;
        }}
        QPushButton:hover {{ border-color: {PRI}; }}
        QPushButton#approve {{ color: {GREEN}; }}
        QPushButton#reject {{ color: {RED}; }}
    """

    class DecisionsTab(QWidget):
        _COLUMNS = ["Created", "Queue Item", "Decision", "Confidence", "Status", "Outcome / Error"]

        def __init__(self, model: DashboardModel, parent=None):
            super().__init__(parent)
            self._model = model

            layout = QVBoxLayout(self)
            heading = QLabel("Recent decisions")
            heading.setObjectName("heading")
            layout.addWidget(heading)

            self._table = QTableWidget(0, len(self._COLUMNS))
            self._table.setHorizontalHeaderLabels(self._COLUMNS)
            self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self._table.horizontalHeader().setSectionResizeMode(
                len(self._COLUMNS) - 1, QHeaderView.ResizeMode.Stretch
            )
            layout.addWidget(self._table)

        def refresh(self) -> None:
            try:
                rows = self._model.list_recent_decisions()
            except DecisionStoreError:
                _log.exception("Could not load recent decisions")
                return

            self._table.setRowCount(len(rows))
            for i, row in enumerate(rows):
                values = [
                    row.created_at or "",
                    row.queue_item_id,
                    row.decision or "",
                    _fmt_confidence(row.confidence),
                    row.status,
                    _fmt_outcome(row),
                ]
                color = QColor(_STATUS_COLORS.get(row.status, GRAY))
                for col, value in enumerate(values):
                    item = QTableWidgetItem(str(value))
                    if row.status in ATTENTION_STATUSES:
                        item.setForeground(color)
                    self._table.setItem(i, col, item)

    class ApprovalsTab(QWidget):
        _COLUMNS = ["Queue Item", "Reason", "Confidence", "Action Type", "Created", "Approve", "Reject"]

        def __init__(self, model: DashboardModel, on_changed, parent=None):
            super().__init__(parent)
            self._model = model
            self._on_changed = on_changed
            self._approvals: list[dict[str, Any]] = []

            layout = QVBoxLayout(self)
            heading = QLabel("Pending approvals")
            heading.setObjectName("heading")
            layout.addWidget(heading)

            splitter = QSplitter(Qt.Orientation.Vertical)

            self._table = QTableWidget(0, len(self._COLUMNS))
            self._table.setHorizontalHeaderLabels(self._COLUMNS)
            self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            self._table.itemSelectionChanged.connect(self._show_selected_detail)
            splitter.addWidget(self._table)

            self._detail = QTextEdit()
            self._detail.setReadOnly(True)
            self._detail.setPlaceholderText("Select a pending approval to see the full queue item and proposed action.")
            splitter.addWidget(self._detail)
            splitter.setSizes([300, 200])

            layout.addWidget(splitter)

        def refresh(self) -> None:
            try:
                self._approvals = self._model.list_pending_approvals()
            except ApprovalError:
                _log.exception("Could not load pending approvals")
                return

            self._table.setRowCount(len(self._approvals))
            for i, approval in enumerate(self._approvals):
                proposed_action = json.loads(approval["proposed_action"]) if approval["proposed_action"] else {}
                decision = self._model.get_decision(approval["decision_id"])
                values = [
                    approval["queue_item_id"],
                    approval["reasoning"],
                    _fmt_confidence(decision.confidence if decision else None),
                    proposed_action.get("type", ""),
                    approval["created_at"],
                ]
                for col, value in enumerate(values):
                    self._table.setItem(i, col, QTableWidgetItem(str(value)))

                approve_btn = QPushButton("Approve")
                approve_btn.setObjectName("approve")
                approve_btn.clicked.connect(lambda _checked=False, aid=approval["id"]: self._resolve(aid, approve=True))
                self._table.setCellWidget(i, len(values), approve_btn)

                reject_btn = QPushButton("Reject")
                reject_btn.setObjectName("reject")
                reject_btn.clicked.connect(lambda _checked=False, aid=approval["id"]: self._resolve(aid, approve=False))
                self._table.setCellWidget(i, len(values) + 1, reject_btn)

        def _show_selected_detail(self) -> None:
            rows = self._table.selectionModel().selectedRows()
            if not rows:
                return
            approval = self._approvals[rows[0].row()]
            proposed_action = json.loads(approval["proposed_action"]) if approval["proposed_action"] else {}
            context = json.loads(approval["context"]) if approval["context"] else {}
            self._detail.setPlainText(
                f"Approval: {approval['id']}\n"
                f"Queue item: {approval['queue_item_id']}\n"
                f"Reasoning: {approval['reasoning']}\n\n"
                f"Proposed action:\n{_pretty_json(proposed_action)}\n\n"
                f"Context:\n{_pretty_json(context)}"
            )

        def _resolve(self, approval_id: str, approve: bool) -> None:
            try:
                if approve:
                    self._model.approve(approval_id)
                else:
                    self._model.reject(approval_id)
            except ApprovalError as e:
                QMessageBox.critical(self, "Approval failed", str(e))
                return
            self._on_changed()

    class MainWindow(QMainWindow):
        def __init__(self, model: DashboardModel, db_path):
            super().__init__()
            self._model = model
            self.setWindowTitle(f"JARVIS Decisions Dashboard — {db_path}")
            self.resize(1100, 700)
            self.setStyleSheet(STYLESHEET)

            self._decisions_tab = DecisionsTab(model)
            self._approvals_tab = ApprovalsTab(model, on_changed=self.refresh_now)

            tabs = QTabWidget()
            tabs.addTab(self._decisions_tab, "Decisions")
            tabs.addTab(self._approvals_tab, "Pending Approvals")
            self.setCentralWidget(tabs)

            self._timer = QTimer(self)
            self._timer.setInterval(DEFAULT_REFRESH_MS)
            self._timer.timeout.connect(self.refresh_now)
            self._timer.start()

            self.refresh_now()

        def refresh_now(self) -> None:
            self._decisions_tab.refresh()
            self._approvals_tab.refresh()

    return MainWindow


if __name__ == "__main__":
    sys.exit(run())
