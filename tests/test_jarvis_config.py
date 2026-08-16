"""
tests/test_jarvis_config.py
==============================
jarvis_core/config.py's JarvisConfig — required-vs-defaulted settings,
same "manipulate real os.environ, restore in tearDown" convention as
eyv_poller's own tests/test_config.py. In particular: ANTHROPIC_API_KEY
must be set (B.2.1 can't reason without it) and the architecture
document must actually exist on disk (it's used verbatim as the system
prompt — a missing file must fail loudly at startup, not at the first
reasoning call).
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis_core.config import JarvisConfig, JarvisConfigError  # noqa: E402

_RELEVANT_KEYS = [
    "ANTHROPIC_API_KEY", "JARVIS_ARCHITECTURE_DOC_PATH", "JARVIS_CLAUDE_MODEL",
    "JARVIS_CLAUDE_MAX_TOKENS", "JARVIS_CONFIDENCE_THRESHOLD", "JARVIS_DB_PATH",
    "JARVIS_DECISIONS_PATH", "JARVIS_REQUEST_TIMEOUT_SEC", "JARVIS_MAX_RETRIES",
    "JARVIS_RETRY_BACKOFF_BASE_SEC", "JARVIS_LOG_DIR", "JARVIS_LOG_LEVEL",
]


class JarvisConfigTest(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in _RELEVANT_KEYS}
        for k in _RELEVANT_KEYS:
            os.environ.pop(k, None)
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self._missing_dotenv = Path(self._tmpdir.name) / "missing.env"
        self._doc_path = Path(self._tmpdir.name) / "arch.md"
        self._doc_path.write_text("You are JARVIS.", encoding="utf-8")

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _set_required(self):
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
        os.environ["JARVIS_ARCHITECTURE_DOC_PATH"] = str(self._doc_path)

    def test_missing_api_key_raises(self):
        os.environ["JARVIS_ARCHITECTURE_DOC_PATH"] = str(self._doc_path)
        with self.assertRaises(JarvisConfigError):
            JarvisConfig.load(dotenv_path=self._missing_dotenv)

    def test_missing_architecture_doc_raises(self):
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
        os.environ["JARVIS_ARCHITECTURE_DOC_PATH"] = str(Path(self._tmpdir.name) / "does_not_exist.md")
        with self.assertRaises(JarvisConfigError):
            JarvisConfig.load(dotenv_path=self._missing_dotenv)

    def test_confidence_threshold_out_of_range_raises(self):
        self._set_required()
        os.environ["JARVIS_CONFIDENCE_THRESHOLD"] = "1.5"
        with self.assertRaises(JarvisConfigError):
            JarvisConfig.load(dotenv_path=self._missing_dotenv)

    def test_non_numeric_confidence_threshold_raises(self):
        self._set_required()
        os.environ["JARVIS_CONFIDENCE_THRESHOLD"] = "very sure"
        with self.assertRaises(JarvisConfigError):
            JarvisConfig.load(dotenv_path=self._missing_dotenv)

    def test_valid_config_loads_with_expected_defaults(self):
        self._set_required()
        config = JarvisConfig.load(dotenv_path=self._missing_dotenv)
        self.assertEqual(config.claude_model, "claude-opus-5")
        self.assertEqual(config.confidence_threshold, 0.7)
        self.assertEqual(config.decisions_path, "/jarvis/decisions")
        self.assertEqual(config.max_retries, 3)
        self.assertEqual(config.architecture_doc_path, self._doc_path)

    def test_overrides_are_respected(self):
        self._set_required()
        os.environ["JARVIS_CLAUDE_MODEL"] = "claude-sonnet-5"
        os.environ["JARVIS_CONFIDENCE_THRESHOLD"] = "0.85"
        os.environ["JARVIS_MAX_RETRIES"] = "5"
        config = JarvisConfig.load(dotenv_path=self._missing_dotenv)
        self.assertEqual(config.claude_model, "claude-sonnet-5")
        self.assertEqual(config.confidence_threshold, 0.85)
        self.assertEqual(config.max_retries, 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
