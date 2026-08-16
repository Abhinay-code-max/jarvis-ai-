"""
tests/test_live_smoke.py
==========================
The one deliberate exception to tests/test_claude_reasoner.py's standing
rule ("every test that touches Claude reasoning mocks the API client,
full stop"). All 112+ other tests covering engine.py/claude_reasoner.py
mock the Claude API — there has never been a real end-to-end run through
claude_reasoner.py -> engine.py with the actual Claude API before this
file. That gap is what this test closes.

Skipped by default — never runs under `python -m unittest discover` or
plain `pytest` — because it costs a real API call and depends on
ANTHROPIC_API_KEY being set. Opt in explicitly:

    JARVIS_LIVE_TEST=1 python -m unittest tests.test_live_smoke -v

Same "explicit opt-in env var, excluded from the default run" shape as
EYV's `make eval --live`. No live EYV connection is required — the
queue item is a hand-crafted fixture, not fetched from GET /jarvis/queue.

CodeInterface and MarketingDecisionClient are real, unmocked instances
here too (this is meant to smoke-test engine.py's *full* pipeline, not
just the Claude call) — but both already degrade to a recorded
`ExecutionResult`/exception rather than raising when their downstream
dependency (the `claude` CLI, a real EYV `/jarvis/decisions` endpoint)
isn't available, so this stays safe to run without either being set up.
The fixture below is deliberately phrased so the most likely decision
(NEEDS_APPROVAL, given how ambiguous/high-risk it reads) never reaches
either of them anyway.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eyv_poller.config import _load_dotenv  # noqa: E402
from jarvis_core import config  # noqa: E402
from jarvis_core.approval_interface import ApprovalInterface  # noqa: E402
from jarvis_core.claude_reasoner import ClaudeReasoner  # noqa: E402
from jarvis_core.code_interface import CodeInterface  # noqa: E402
from jarvis_core.db import DecisionStore  # noqa: E402
from jarvis_core.decision_schema import DECISION_TYPES  # noqa: E402
from jarvis_core.engine import JarvisEngine  # noqa: E402
from jarvis_core.marketing_client import MarketingDecisionClient  # noqa: E402

# A hand-crafted queue item shaped like a real GET /jarvis/queue entry
# (see docs/jarvis_architecture.md's "queue you're reasoning over").
# Deliberately ambiguous/high-risk so the most likely real decision is
# NEEDS_APPROVAL, which never calls out to CodeInterface or
# MarketingDecisionClient — keeping this test side-effect-free even
# though both are real, unmocked instances.
_FIXTURE_QUEUE_ITEM = {
    "id": "live-smoke-test-0001",
    "kind": "bug",
    "approval": "approved",
    "title": "Delete all soft-deleted user accounts older than 30 days",
    "description": (
        "A customer reported that a soft-deleted account of theirs was "
        "still visible in an internal admin export. Someone suggested we "
        "just permanently purge every soft-deleted user account older "
        "than 30 days from the production database to 'clean this up'. "
        "No spec for what 'visible in an export' actually means yet, and "
        "no confirmation this is the right fix rather than a report/export "
        "bug."
    ),
}


@unittest.skipUnless(
    os.environ.get("JARVIS_LIVE_TEST") == "1",
    "live Claude API smoke test — set JARVIS_LIVE_TEST=1 to run (costs a real API call)",
)
class LiveSmokeTest(unittest.TestCase):
    def setUp(self):
        _load_dotenv(Path.cwd() / ".env")
        if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            self.skipTest("ANTHROPIC_API_KEY is not set — cannot make a real Claude API call")

        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        db_path = Path(self._tmpdir.name) / "jarvis_live_smoke.db"
        self.store = DecisionStore(db_path)
        self.addCleanup(self.store.close)

        doc_path_raw = os.environ.get("JARVIS_ARCHITECTURE_DOC_PATH", "").strip()
        architecture_doc_path = Path(doc_path_raw) if doc_path_raw else config.DEFAULT_ARCHITECTURE_DOC_PATH

        self.reasoner = ClaudeReasoner(
            architecture_doc_path=architecture_doc_path,
            model=os.environ.get("JARVIS_CLAUDE_MODEL", config.DEFAULT_CLAUDE_MODEL),
            max_tokens=os.environ.get("JARVIS_CLAUDE_MAX_TOKENS", config.DEFAULT_CLAUDE_MAX_TOKENS),
        )
        self.confidence_threshold = float(
            os.environ.get("JARVIS_CONFIDENCE_THRESHOLD", config.DEFAULT_CONFIDENCE_THRESHOLD)
        )
        self.engine = JarvisEngine(
            store=self.store,
            reasoner=self.reasoner,
            code_interface=CodeInterface(),
            marketing_client=MarketingDecisionClient(
                base_url="https://example.invalid",
                decisions_path=config.DEFAULT_DECISIONS_PATH,
                get_token=lambda: None,
                request_timeout_sec=config.DEFAULT_REQUEST_TIMEOUT_SEC,
                max_retries=0,
                retry_backoff_base_sec=config.DEFAULT_RETRY_BACKOFF_BASE_SEC,
            ),
            approval_interface=ApprovalInterface(self.store),
            confidence_threshold=self.confidence_threshold,
        )

    def test_one_fixture_item_through_the_real_pipeline(self):
        from eyv_poller.schema import QueueItem, QueueSnapshot

        item = QueueItem(item_id=_FIXTURE_QUEUE_ITEM["id"], raw=_FIXTURE_QUEUE_ITEM)
        snapshot = QueueSnapshot(items=[item])

        self.engine.process_queue_snapshot(snapshot)

        row = self.store.get_by_queue_item_id(_FIXTURE_QUEUE_ITEM["id"])
        self.assertIsNotNone(row, "engine.py must durably record a decision row for the fixture item")

        # decision_schema.py's DECISION_TYPES is the ground truth for what
        # a valid `decision` value is — if Claude's response didn't parse,
        # process_queue_snapshot() would have recorded status="failed"
        # with no `decision` value at all instead of getting here.
        self.assertIn(row.decision, DECISION_TYPES)
        self.assertIsNotNone(row.confidence)
        self.assertIsNotNone(row.reason)

        # Confidence-gate invariant (engine.py._apply_confidence_gate):
        # whatever Claude originally returned, a final confidence below
        # threshold must have been forced to NEEDS_APPROVAL.
        if row.confidence < self.confidence_threshold:
            self.assertEqual(
                row.decision, "NEEDS_APPROVAL",
                "confidence gate did not force a below-threshold decision to NEEDS_APPROVAL",
            )

        self.assertIn(
            row.status,
            ("waiting_approval", "completed", "resolved", "failed"),
            "decision row must have progressed past 'reasoned' — routing must have run",
        )

        print("\n" + "=" * 70)
        print("LIVE SMOKE TEST — actual Claude decision for the fixture item")
        print("=" * 70)
        print(f"decision:    {row.decision}")
        print(f"confidence:  {row.confidence}")
        print(f"reason:      {row.reason}")
        print(f"action_type: {row.action_type}")
        print(f"status:      {row.status}")
        print("=" * 70)


if __name__ == "__main__":
    unittest.main(verbosity=2)
