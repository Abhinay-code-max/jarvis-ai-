"""
tests/test_voice_config.py
=============================
jarvis_core.voice.config.VoiceConfig — engine-selection validation in
particular (JARVIS_VOICE_STT_ENGINE / JARVIS_VOICE_TTS_ENGINE / the
Vosk-requires-a-model-path rule), same "manipulate real os.environ,
restore in tearDown" convention as tests/test_jarvis_config.py.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis_core.voice.config import VoiceConfig, VoiceConfigError  # noqa: E402

_RELEVANT_KEYS = [
    "ANTHROPIC_API_KEY", "JARVIS_VOICE_STT_ENGINE", "JARVIS_VOICE_VOSK_MODEL_PATH",
    "JARVIS_VOICE_TTS_ENGINE", "JARVIS_VOICE_EDGE_TTS_VOICE",
]


class VoiceConfigEngineSelectionTest(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in _RELEVANT_KEYS}
        for k in _RELEVANT_KEYS:
            os.environ.pop(k, None)
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-key"
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self._missing_dotenv = Path(self._tmpdir.name) / "missing.env"

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_defaults_to_whisper_and_elevenlabs(self):
        config = VoiceConfig.load(self._missing_dotenv)

        self.assertEqual(config.stt_engine, "whisper")
        self.assertIsNone(config.vosk_model_path)
        self.assertEqual(config.tts_engine, "elevenlabs")

    def test_invalid_stt_engine_raises(self):
        os.environ["JARVIS_VOICE_STT_ENGINE"] = "carrier-pigeon"
        with self.assertRaises(VoiceConfigError):
            VoiceConfig.load(self._missing_dotenv)

    def test_invalid_tts_engine_raises(self):
        os.environ["JARVIS_VOICE_TTS_ENGINE"] = "shouting"
        with self.assertRaises(VoiceConfigError):
            VoiceConfig.load(self._missing_dotenv)

    def test_vosk_without_model_path_raises(self):
        os.environ["JARVIS_VOICE_STT_ENGINE"] = "vosk"
        with self.assertRaises(VoiceConfigError):
            VoiceConfig.load(self._missing_dotenv)

    def test_vosk_with_model_path_succeeds(self):
        os.environ["JARVIS_VOICE_STT_ENGINE"] = "vosk"
        os.environ["JARVIS_VOICE_VOSK_MODEL_PATH"] = str(Path(self._tmpdir.name) / "some-model")

        config = VoiceConfig.load(self._missing_dotenv)

        self.assertEqual(config.stt_engine, "vosk")
        self.assertEqual(config.vosk_model_path, Path(self._tmpdir.name) / "some-model")

    def test_edgetts_engine_needs_no_extra_config(self):
        os.environ["JARVIS_VOICE_TTS_ENGINE"] = "edgetts"

        config = VoiceConfig.load(self._missing_dotenv)

        self.assertEqual(config.tts_engine, "edgetts")
        self.assertTrue(config.edge_tts_voice)

    def test_engine_names_are_case_insensitive(self):
        os.environ["JARVIS_VOICE_STT_ENGINE"] = "WHISPER"
        os.environ["JARVIS_VOICE_TTS_ENGINE"] = "ElevenLabs"

        config = VoiceConfig.load(self._missing_dotenv)

        self.assertEqual(config.stt_engine, "whisper")
        self.assertEqual(config.tts_engine, "elevenlabs")


if __name__ == "__main__":
    unittest.main(verbosity=2)
