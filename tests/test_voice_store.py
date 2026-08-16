"""
tests/test_voice_store.py
============================
jarvis_core.voice.store.VoiceStore against a real temp-directory SQLite
file — same convention as tests/test_db.py. Also confirms it never
touches DecisionStore's decisions/approvals tables even when pointed at
the exact same database file (the "reuse db.py's SQLite setup with a
new table" requirement from jarvis_voice_ui_prompt.md).
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis_core.db import DecisionStore  # noqa: E402
from jarvis_core.voice.store import VoiceStore, VoiceStoreError  # noqa: E402


class VoiceStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.db_path = Path(self._tmpdir.name) / "jarvis.db"
        self.store = VoiceStore(self.db_path)
        self.addCleanup(self.store.close)

    def test_db_file_is_created(self):
        self.assertTrue(self.db_path.exists())

    def test_record_and_list_session_turns_round_trips_in_order(self):
        self.store.record_turn("session-1", "user", "hello jarvis")
        self.store.record_turn("session-1", "assistant", "hello there")

        turns = self.store.list_session_turns("session-1")

        self.assertEqual([t.role for t in turns], ["user", "assistant"])
        self.assertEqual([t.text for t in turns], ["hello jarvis", "hello there"])
        self.assertTrue(all(t.session_id == "session-1" for t in turns))
        self.assertTrue(all(t.created_at for t in turns))

    def test_sessions_do_not_leak_into_each_other(self):
        self.store.record_turn("session-1", "user", "from session 1")
        self.store.record_turn("session-2", "user", "from session 2")

        self.assertEqual([t.text for t in self.store.list_session_turns("session-1")], ["from session 1"])
        self.assertEqual([t.text for t in self.store.list_session_turns("session-2")], ["from session 2"])

    def test_unknown_session_returns_empty_list(self):
        self.assertEqual(self.store.list_session_turns("no-such-session"), [])

    def test_list_recent_turns_most_recent_first_and_respects_limit(self):
        for i in range(5):
            self.store.record_turn("session-1", "user", f"turn {i}")

        recent = self.store.list_recent_turns(limit=3)

        self.assertEqual(len(recent), 3)
        self.assertEqual(recent[0].text, "turn 4")  # most recent first

    def test_voice_store_never_touches_decisions_or_approvals_tables(self):
        """Same DB file, opened by both stores — proves VoiceStore's own
        table doesn't collide with or corrupt DecisionStore's."""
        decision_store = DecisionStore(self.db_path)
        self.addCleanup(decision_store.close)
        decision_store.create_pending("t-1", {"id": "t-1"})

        self.store.record_turn("session-1", "user", "unrelated voice turn")

        self.assertIsNotNone(decision_store.get_by_queue_item_id("t-1"))
        self.assertEqual(len(self.store.list_session_turns("session-1")), 1)

    def test_close_then_use_raises_voice_store_error(self):
        self.store.close()
        with self.assertRaises(VoiceStoreError):
            self.store.record_turn("session-1", "user", "after close")


if __name__ == "__main__":
    unittest.main(verbosity=2)
