"""
tests/test_voice_stt.py
==========================
jarvis_core.voice.stt.WhisperTranscriber, exercised against a
hand-written fake model object — never a real faster-whisper model load
(no network, no GPU/CPU model-file download). This mirrors
tests/test_claude_reasoner.py's "mock the API client" convention,
applied to a local ML model instead of a remote API: the point is the
same — never let a test depend on a slow, heavy, or non-deterministic
real backend.

Covers the misfire-resilience contract this module exists for:
transcribe() must return None (never an empty/garbage string) for audio
that's too short, silent, or empty after Whisper's own output — and
must raise SttError, not return None, for a genuine transcription
failure.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from jarvis_core.voice.stt import SttError, WhisperTranscriber  # noqa: E402

_SAMPLE_RATE = 16000


class _FakeModel:
    """transcribe() returns whatever `plan` says for this call, then
    holds on the last entry — same shape as this codebase's other
    hand-written stubs (see tests/test_poller.py's _StubClient)."""

    def __init__(self, plan):
        self._plan = list(plan)
        self.calls = []

    def transcribe(self, audio, beam_size=5):
        self.calls.append(audio)
        step = self._plan[min(len(self.calls), len(self._plan)) - 1]
        if isinstance(step, Exception):
            raise step
        segments, info = step
        return iter(segments), info


def _segment(text, no_speech_prob=0.0):
    return SimpleNamespace(text=text, no_speech_prob=no_speech_prob)


def _audio(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * _SAMPLE_RATE), dtype=np.float32)


class WhisperTranscriberTest(unittest.TestCase):
    def test_normal_speech_returns_joined_text(self):
        model = _FakeModel([([_segment("hello "), _segment("jarvis")], SimpleNamespace())])
        transcriber = WhisperTranscriber(model=model, min_utterance_sec=0.4)

        result = transcriber.transcribe(_audio(2.0), _SAMPLE_RATE)

        self.assertEqual(result, "hello jarvis")
        self.assertEqual(len(model.calls), 1)

    def test_audio_shorter_than_min_utterance_is_treated_as_noise_without_calling_model(self):
        model = _FakeModel([([_segment("should never see this")], SimpleNamespace())])
        transcriber = WhisperTranscriber(model=model, min_utterance_sec=0.4)

        result = transcriber.transcribe(_audio(0.1), _SAMPLE_RATE)

        self.assertIsNone(result)
        self.assertEqual(model.calls, [])  # never even asked Whisper

    def test_empty_segments_returns_none(self):
        model = _FakeModel([([], SimpleNamespace())])
        transcriber = WhisperTranscriber(model=model, min_utterance_sec=0.4)

        self.assertIsNone(transcriber.transcribe(_audio(2.0), _SAMPLE_RATE))

    def test_high_no_speech_probability_returns_none(self):
        model = _FakeModel([([_segment("noise", no_speech_prob=0.95)], SimpleNamespace())])
        transcriber = WhisperTranscriber(model=model, min_utterance_sec=0.4, no_speech_prob_threshold=0.6)

        self.assertIsNone(transcriber.transcribe(_audio(2.0), _SAMPLE_RATE))

    def test_low_no_speech_probability_is_kept(self):
        model = _FakeModel([([_segment("real speech", no_speech_prob=0.1)], SimpleNamespace())])
        transcriber = WhisperTranscriber(model=model, min_utterance_sec=0.4, no_speech_prob_threshold=0.6)

        self.assertEqual(transcriber.transcribe(_audio(2.0), _SAMPLE_RATE), "real speech")

    def test_empty_text_after_stripping_returns_none(self):
        model = _FakeModel([([_segment("   ")], SimpleNamespace())])
        transcriber = WhisperTranscriber(model=model, min_utterance_sec=0.4)

        self.assertIsNone(transcriber.transcribe(_audio(2.0), _SAMPLE_RATE))

    def test_model_failure_raises_stt_error_not_none(self):
        model = _FakeModel([RuntimeError("model exploded")])
        transcriber = WhisperTranscriber(model=model, min_utterance_sec=0.4)

        with self.assertRaises(SttError):
            transcriber.transcribe(_audio(2.0), _SAMPLE_RATE)

    def test_zero_sample_rate_is_treated_as_too_short_rather_than_dividing_by_zero(self):
        model = _FakeModel([([_segment("x")], SimpleNamespace())])
        transcriber = WhisperTranscriber(model=model, min_utterance_sec=0.4)

        self.assertIsNone(transcriber.transcribe(_audio(2.0), 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
