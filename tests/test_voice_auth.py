"""
tests/test_voice_auth.py
===========================
jarvis_core/voice/auth.py: ElevenLabs key storage. Real filesystem,
temp directory per test — same convention as tests/test_auth.py (the
whole point is verifying the actual file and its permission bits).
Only covers set/show, since — unlike eyv_poller's EYV queue token —
there's nothing to locally `generate` for a key ElevenLabs itself
mints (see auth.py's module docstring).
"""
from __future__ import annotations

import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis_core.voice import auth  # noqa: E402


class VoiceAuthTest(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="jarvis_voice_test_keys_"))
        self.key_path = self.tmp_dir / "nested" / "elevenlabs_key.txt"

    def test_get_key_returns_none_when_not_set_up(self):
        self.assertIsNone(auth.get_key(self.key_path))

    def test_set_key_round_trips(self):
        auth.set_key("sk_test_elevenlabs_key", self.key_path)
        self.assertEqual(auth.get_key(self.key_path), "sk_test_elevenlabs_key")

    def test_set_key_creates_parent_directories(self):
        self.assertFalse(self.key_path.parent.exists())
        auth.set_key("sk_test_elevenlabs_key", self.key_path)
        self.assertTrue(self.key_path.exists())

    def test_set_key_strips_whitespace(self):
        auth.set_key("  sk_test_elevenlabs_key  ", self.key_path)
        self.assertEqual(auth.get_key(self.key_path), "sk_test_elevenlabs_key")

    def test_set_key_rejects_empty(self):
        with self.assertRaises(auth.VoiceAuthError):
            auth.set_key("   ", self.key_path)

    def test_set_key_overwrites_previous_value(self):
        auth.set_key("first-key", self.key_path)
        auth.set_key("second-key", self.key_path)
        self.assertEqual(auth.get_key(self.key_path), "second-key")

    @unittest.skipIf(sys.platform == "win32", "POSIX permission bits aren't meaningful on Windows")
    def test_key_file_is_owner_only_readable(self):
        auth.set_key("sk_test_elevenlabs_key", self.key_path)
        mode = stat.S_IMODE(self.key_path.stat().st_mode)
        self.assertEqual(mode, 0o600)

    def test_get_key_survives_unreadable_file_gracefully(self):
        bad_path = self.tmp_dir / "a_directory_not_a_file"
        bad_path.mkdir()
        self.assertIsNone(auth.get_key(bad_path))

    def test_no_generate_command_exists(self):
        # Unlike eyv_poller.auth, there's nothing to cryptographically
        # generate locally — an ElevenLabs key is minted by ElevenLabs.
        self.assertFalse(hasattr(auth, "generate_key"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
