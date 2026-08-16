"""
tests/test_config.py
=======================
eyv_poller/config.py — in particular, that EYV_POLL_INTERVAL_SEC has no
silent default (B.1's explicit requirement: the polling interval is an
open decision, not something this service gets to invent on its own).
Manipulates real os.environ for the duration of each test (restored in
tearDown) rather than mocking os.environ, since Config.load() reads it
directly and a real env is simplest to reason about here.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eyv_poller.config import Config, ConfigError  # noqa: E402

_RELEVANT_KEYS = [
    "EYV_BASE_URL", "EYV_POLL_INTERVAL_SEC", "EYV_QUEUE_PATH", "EYV_TOKEN_PATH",
    "EYV_REQUEST_TIMEOUT_SEC", "EYV_MAX_RETRIES", "EYV_RETRY_BACKOFF_BASE_SEC",
    "EYV_LOG_DIR", "EYV_LOG_LEVEL",
]


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in _RELEVANT_KEYS}
        for k in _RELEVANT_KEYS:
            os.environ.pop(k, None)
        # Points at a nonexistent .env so no real local file leaks into
        # these tests via Config.load()'s default dotenv_path=Path.cwd().
        self._missing_dotenv = Path(tempfile.mkdtemp(prefix="eyv_poller_test_env_")) / "missing.env"

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_missing_base_url_raises(self):
        os.environ["EYV_POLL_INTERVAL_SEC"] = "90"
        with self.assertRaises(ConfigError):
            Config.load(dotenv_path=self._missing_dotenv)

    def test_missing_poll_interval_raises_with_no_default(self):
        os.environ["EYV_BASE_URL"] = "https://example.up.railway.app"
        with self.assertRaises(ConfigError):
            Config.load(dotenv_path=self._missing_dotenv)

    def test_non_numeric_poll_interval_raises(self):
        os.environ["EYV_BASE_URL"] = "https://example.up.railway.app"
        os.environ["EYV_POLL_INTERVAL_SEC"] = "soon"
        with self.assertRaises(ConfigError):
            Config.load(dotenv_path=self._missing_dotenv)

    def test_zero_or_negative_poll_interval_raises(self):
        os.environ["EYV_BASE_URL"] = "https://example.up.railway.app"
        os.environ["EYV_POLL_INTERVAL_SEC"] = "0"
        with self.assertRaises(ConfigError):
            Config.load(dotenv_path=self._missing_dotenv)

    def test_valid_config_loads_with_expected_defaults(self):
        os.environ["EYV_BASE_URL"] = "https://example.up.railway.app/"
        os.environ["EYV_POLL_INTERVAL_SEC"] = "90"
        config = Config.load(dotenv_path=self._missing_dotenv)
        self.assertEqual(config.poll_interval_sec, 90.0)
        self.assertEqual(config.queue_path, "/jarvis/queue")
        self.assertEqual(config.queue_url, "https://example.up.railway.app/jarvis/queue")
        self.assertEqual(config.max_retries, 3)

    def test_overrides_are_respected(self):
        os.environ["EYV_BASE_URL"] = "https://example.up.railway.app"
        os.environ["EYV_POLL_INTERVAL_SEC"] = "60"
        os.environ["EYV_MAX_RETRIES"] = "7"
        os.environ["EYV_QUEUE_PATH"] = "/jarvis/queue/v2"
        config = Config.load(dotenv_path=self._missing_dotenv)
        self.assertEqual(config.max_retries, 7)
        self.assertEqual(config.queue_url, "https://example.up.railway.app/jarvis/queue/v2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
