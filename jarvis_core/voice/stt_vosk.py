"""
jarvis_core/voice/stt_vosk.py
================================
Vosk speech-to-text — a selectable alternate STT engine alongside
stt.py's WhisperTranscriber (the default). Same public contract
(`transcribe(audio, sample_rate) -> str | None`, same misfire-resilience
guarantee: never an empty/garbage string, None for "no speech", raises
SttError only for a genuine failure) so session.py and ui.py never need
to know which engine is actually running — see jarvis_core/voice/ui.py's
run() for the JARVIS_VOICE_STT_ENGINE switch.

Unlike faster-whisper, Vosk does NOT auto-download a model on first
use — a Vosk model is a folder you download yourself from
https://alphacephei.com/vosk/models and point `model_path` at (see
voice/config.py's JARVIS_VOICE_VOSK_MODEL_PATH). VoskTranscriber raises
SttError with a clear message if that path doesn't exist, rather than
letting the underlying C++ binding fail with a cryptic error.

`vosk` is imported lazily, inside __init__, for the same reason
stt.py's faster_whisper import is lazy — importing this module must
never require the package to be installed.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np

from jarvis_core.voice.stt import SttError

_log = logging.getLogger("jarvis_core.voice.stt_vosk")


def _real_recognizer_factory(model: Any, sample_rate: int) -> Any:
    from vosk import KaldiRecognizer

    return KaldiRecognizer(model, sample_rate)


class VoskTranscriber:
    def __init__(
        self,
        model_path: Path,
        model: Any | None = None,
        recognizer_factory: Callable[[Any, int], Any] | None = None,
        min_utterance_sec: float = 0.4,
    ):
        self._min_utterance_sec = min_utterance_sec
        # `recognizer_factory` mirrors `model`'s injection role: tests supply
        # a hand-written fake for both rather than needing a real downloaded
        # Vosk model or the real `vosk` package's C++ binding at all.
        self._recognizer_factory = recognizer_factory or _real_recognizer_factory
        if model is not None:
            self._model = model
        else:
            if not Path(model_path).is_dir():
                raise SttError(
                    f"Vosk model not found at {model_path} — download one from "
                    "https://alphacephei.com/vosk/models and set "
                    "JARVIS_VOICE_VOSK_MODEL_PATH to the extracted folder"
                )
            try:
                from vosk import Model
            except ImportError as e:
                raise SttError(
                    "vosk is not installed — run 'pip install vosk' (see requirements.txt)"
                ) from e
            self._model = Model(str(model_path))

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str | None:
        """Same contract as WhisperTranscriber.transcribe(): returns the
        transcribed text, or None if the audio doesn't plausibly contain
        speech (too short, or Vosk returns empty text). Raises SttError
        only for a genuine failure to run recognition at all."""
        duration_sec = len(audio) / float(sample_rate) if sample_rate else 0.0
        if duration_sec < self._min_utterance_sec:
            _log.info(
                "Utterance too short (%.2fs < %.2fs threshold) — treating as noise, skipping Vosk",
                duration_sec, self._min_utterance_sec,
            )
            return None

        # Vosk's recognizer wants 16-bit PCM bytes, not the float32 array
        # the rest of this codebase (recorder, Whisper) uses internally.
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

        try:
            recognizer = self._recognizer_factory(self._model, sample_rate)
            recognizer.AcceptWaveform(pcm16)
            result = json.loads(recognizer.FinalResult())
        except Exception as e:
            raise SttError(f"Vosk transcription failed: {e}") from e

        text = (result.get("text") or "").strip()
        if not text:
            _log.info("Vosk returned only empty text — treating as no speech")
            return None

        return text
