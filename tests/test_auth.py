"""
tests/test_auth.py
=====================
eyv_poller/auth.py: token storage. Real filesystem, temp directory per
test — no mocking of Path/open, since the whole point is to verify the
actual file gets written with the actual permission bits.
"""
from __future__ import annotations

import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eyv_poller import auth  # noqa: E402


class AuthTest(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="eyv_poller_test_tokens_"))
        self.token_path = self.tmp_dir / "nested" / "queue_token.txt"

    def test_get_token_returns_none_when_not_set_up(self):
        self.assertIsNone(auth.get_token(self.token_path))

    def test_generate_token_round_trips(self):
        token = auth.generate_token(self.token_path)
        self.assertTrue(token)
        self.assertEqual(auth.get_token(self.token_path), token)

    def test_generate_token_creates_parent_directories(self):
        self.assertFalse(self.token_path.parent.exists())
        auth.generate_token(self.token_path)
        self.assertTrue(self.token_path.exists())

    def test_generate_token_is_unique_each_time(self):
        first = auth.generate_token(self.token_path)
        second = auth.generate_token(self.token_path)
        self.assertNotEqual(first, second)
        self.assertEqual(auth.get_token(self.token_path), second)

    def test_set_token_stores_pasted_value(self):
        auth.set_token("  pasted-token-value  ", self.token_path)
        self.assertEqual(auth.get_token(self.token_path), "pasted-token-value")

    def test_set_token_rejects_empty(self):
        with self.assertRaises(auth.TokenError):
            auth.set_token("   ", self.token_path)

    @unittest.skipIf(sys.platform == "win32", "POSIX permission bits aren't meaningful on Windows")
    def test_token_file_is_owner_only_readable(self):
        auth.generate_token(self.token_path)
        mode = stat.S_IMODE(self.token_path.stat().st_mode)
        self.assertEqual(mode, 0o600)

    def test_get_token_survives_unreadable_file_gracefully(self):
        # A directory where a file is expected should be reported as
        # "not set up" rather than raising — get_token() must never crash
        # the caller.
        bad_path = self.tmp_dir / "a_directory_not_a_file"
        bad_path.mkdir()
        self.assertIsNone(auth.get_token(bad_path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
