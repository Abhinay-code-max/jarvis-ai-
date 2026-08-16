"""
tests/test_voice_stt_vosk.py
===============================
jarvis_core.voice.stt_vosk.VoskTranscriber, exercised against a
hand-written fake KaldiRecognizer — never a real Vosk model load (no
downloaded model directory needed). Same "mock the heavy local ML
backend" convention as tests/test_voice_stt.py's WhisperTranscriber
tests, applied to Vosk's very different API shape (JSON string results,
16-bit PCM bytes input rather than a float32 array).
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from jarvis_core.voice.stt import SttError  # noqa: E402
from jarvis_core.voice.stt_vosk import VoskTranscriber  # noqa: E402

_SAMPLE_RATE = 16000


class _FakeRecognizer:
    def __init__(self, result_text: str, exc: Exception | None = None):
        self._result_text = result_text
        self._exc = exc
        self.accepted_waveforms: list[bytes] = []

    def AcceptWaveform(self, pcm_bytes):
        if self._exc is not None:
            raise self._exc
        self.accepted_waveforms.append(pcm_bytes)

    def FinalResult(self):
        return json.dumps({"text": self._result_text})


def _transcriber(recognizer: _FakeRecognizer, min_utterance_sec: float = 0.4) -> VoskTranscriber:
    return VoskTranscriber(
        model_path=Path("unused"),
        model=object(),  # never touched when recognizer_factory is injected
        recognizer_factory=lambda model, sample_rate: recognizer,
        min_utterance_sec=min_utterance_sec,
    )


def _audio(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * _SAMPLE_RATE), dtype=np.float32)


class VoskTranscriberTest(unittest.TestCase):
    def test_audio_shorter_than_min_utterance_is_treated_as_noise_without_calling_recognizer(self):
        recognizer = _FakeRecognizer("should never be reached")
        transcriber = _transcriber(recognizer, min_utterance_sec=0.4)

        result = transcriber.transcribe(_audio(0.1), _SAMPLE_RATE)

        self.assertIsNone(result)
        self.assertEqual(recognizer.accepted_waveforms, [])

    def test_normal_speech_returns_text(self):
        transcriber = _transcriber(_FakeRecognizer("hello jarvis"))

        result = transcriber.transcribe(_audio(1.0), _SAMPLE_RATE)

        self.assertEqual(result, "hello jarvis")

    def test_empty_text_returns_none(self):
        transcriber = _transcriber(_FakeRecognizer(""))

        result = transcriber.transcribe(_audio(1.0), _SAMPLE_RATE)

        self.assertIsNone(result)

    def test_recognizer_failure_raises_stt_error(self):
        transcriber = _transcriber(_FakeRecognizer("", exc=RuntimeError("kaldi exploded")))

        with self.assertRaises(SttError):
            transcriber.transcribe(_audio(1.0), _SAMPLE_RATE)

    def test_audio_is_converted_to_16bit_pcm_bytes(self):
        recognizer = _FakeRecognizer("hi")
        transcriber = _transcriber(recognizer)

        transcriber.transcribe(_audio(1.0), _SAMPLE_RATE)

        self.assertEqual(len(recognizer.accepted_waveforms), 1)
        # int16 mono PCM: 2 bytes per sample.
        self.assertEqual(len(recognizer.accepted_waveforms[0]), int(1.0 * _SAMPLE_RATE) * 2)

    def test_missing_model_path_raises_stt_error(self):
        with self.assertRaises(SttError):
            VoskTranscriber(model_path=Path("/definitely/does/not/exist"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
