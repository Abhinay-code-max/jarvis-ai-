"""
jarvis_core/voice/stt.py
===========================
Speech-to-text via faster-whisper, run entirely locally — chosen over a
cloud STT API specifically to avoid a third external service/credential
for a local, single-user tool, and to keep voice audio on-device (see
jarvis_voice_ui_prompt.md's STT tradeoff discussion).

`faster_whisper` is imported lazily, inside WhisperTranscriber.__init__,
so importing this module — and running its tests — never requires the
package (or a downloaded model) to be present. Tests inject a fake
model object via the `model=` constructor argument instead of loading a
real one; this mirrors dashboard.py's lazy-PyQt6-import pattern, applied
here to a heavy ML dependency instead of a GUI toolkit.

Misfire resilience is the whole point of this module's public
contract: transcribe() returns None — never an empty or garbage string
— whenever the audio is too short to plausibly contain speech, or
Whisper's own no-speech probability is high, or the resulting text is
empty after stripping. session.py must never forward a None transcript
to Claude; that's what makes JARVIS resilient to silence/background
noise on an open mic. An actual STT failure (a crash inside Whisper, a
bad audio array) is a different case — that raises SttError, which
session.py treats as an error to report, not a normal "no speech"
outcome.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

_log = logging.getLogger("jarvis_core.voice.stt")


class SttError(Exception):
    """Raised for an unexpected transcription failure (not the ordinary
    "no speech detected" case, which returns None rather than raising)."""


class WhisperTranscriber:
    def __init__(
        self,
        model: Any | None = None,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        min_utterance_sec: float = 0.4,
        no_speech_prob_threshold: float = 0.6,
    ):
        self._min_utterance_sec = min_utterance_sec
        self._no_speech_prob_threshold = no_speech_prob_threshold
        if model is not None:
            self._model = model
        else:
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:
                raise SttError(
                    "faster-whisper is not installed — run 'pip install faster-whisper' "
                    "(see requirements.txt)"
                ) from e
            self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str | None:
        """Returns the transcribed text, or None if the audio doesn't
        plausibly contain speech (too short, silent, or Whisper itself
        reports a high no-speech probability). Raises SttError only for
        a genuine failure to run transcription at all."""
        duration_sec = len(audio) / float(sample_rate) if sample_rate else 0.0
        if duration_sec < self._min_utterance_sec:
            _log.info(
                "Utterance too short (%.2fs < %.2fs threshold) — treating as noise, skipping Whisper",
                duration_sec, self._min_utterance_sec,
            )
            return None

        try:
            segments, _info = self._model.transcribe(audio, beam_size=5)
            segments = list(segments)
        except Exception as e:
            raise SttError(f"Whisper transcription failed: {e}") from e

        if not segments:
            _log.info("Whisper returned no segments — treating as no speech")
            return None

        no_speech_probs = [getattr(seg, "no_speech_prob", 0.0) for seg in segments]
        avg_no_speech_prob = sum(no_speech_probs) / len(no_speech_probs)
        if avg_no_speech_prob > self._no_speech_prob_threshold:
            _log.info(
                "Average no-speech probability %.2f exceeds threshold %.2f — treating as noise",
                avg_no_speech_prob, self._no_speech_prob_threshold,
            )
            return None

        text = " ".join(seg.text.strip() for seg in segments).strip()
        if not text:
            _log.info("Whisper returned only empty text — treating as no speech")
            return None

        return text
