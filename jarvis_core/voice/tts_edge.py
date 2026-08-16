"""
jarvis_core/voice/tts_edge.py
================================
EdgeTTS (Microsoft Edge's free, online TTS service) — a selectable
alternate TTS engine alongside tts.py's ElevenLabsTTS (the default).
Same public contract (`synthesize(text) -> (pcm_bytes, sample_rate)`)
so session.py/ui.py never need to know which engine is running — see
jarvis_core/voice/ui.py's run() for the JARVIS_VOICE_TTS_ENGINE switch.

Unlike ElevenLabs, Microsoft's service only ever returns MP3 — there is
no raw-PCM request option — so this module, unlike tts.py, does need an
audio-decoding step: `pydub` (which shells out to a real `ffmpeg`
binary that must be separately installed and on PATH; this is a real
new deployment dependency this project didn't have before, called out
explicitly rather than assumed). No API key is needed (EdgeTTS is
free), so there is no auth-error class here the way ElevenLabsTTS has
one — a failure here is either a network/service problem or a missing
ffmpeg install, both raised as EdgeTTSError.

`edge_tts` and `pydub` are imported lazily, inside __init__/synthesize,
matching this project's standing pattern for optional/heavy
dependencies (see stt.py's faster_whisper import, audio_io.py's
sounddevice import) — importing this module never requires either
package to be installed.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

_log = logging.getLogger("jarvis_core.voice.tts_edge")


class EdgeTTSError(Exception):
    """Raised for any EdgeTTS synthesis failure: a network/service
    error from Microsoft's endpoint, or a decode failure (most likely a
    missing/broken ffmpeg install — pydub needs a real ffmpeg binary on
    PATH to decode the MP3 EdgeTTS returns)."""


def _real_communicate_factory(text: str, voice: str) -> Any:
    import edge_tts

    return edge_tts.Communicate(text, voice=voice)


class EdgeTTS:
    def __init__(
        self,
        voice: str,
        sample_rate: int = 16000,
        communicate_factory: Callable[[str, str], Any] | None = None,
    ):
        self._voice = voice
        self._sample_rate = sample_rate
        # Mirrors ElevenLabsTTS's injectable `session` — tests supply a
        # hand-written fake with a save_sync(path) method instead of
        # making a real call to Microsoft's service.
        self._communicate_factory = communicate_factory or _real_communicate_factory

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def synthesize(self, text: str) -> tuple[bytes, int]:
        """Synthesizes `text` via Microsoft's EdgeTTS service and
        returns (pcm_bytes, sample_rate) — 16-bit little-endian mono PCM
        at `sample_rate`, the same shape ElevenLabsTTS.synthesize()
        returns, decoded from EdgeTTS's MP3 response via pydub/ffmpeg.
        Raises EdgeTTSError on any failure; never returns without actual
        decoded audio bytes."""
        try:
            from pydub import AudioSegment
        except ImportError as e:
            raise EdgeTTSError("pydub is not installed — run 'pip install pydub' (see requirements.txt)") from e

        # A plain temp file, not tempfile.TemporaryDirectory()'s
        # auto-cleanup: pydub/ffprobe can open mp3_path for its own
        # read-ahead check and, on a failed decode (e.g. no ffmpeg on
        # PATH), that handle isn't guaranteed closed by the time the
        # exception reaches us. On Windows that leaves the file locked,
        # and TemporaryDirectory's __exit__ then raises PermissionError
        # while deleting it — masking the real EdgeTTSError with an
        # unrelated cleanup crash. Cleaning up best-effort in `finally`
        # avoids that: a leaked temp file is a far smaller problem than
        # losing the actual error.
        fd, mp3_path_str = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        mp3_path = Path(mp3_path_str)
        try:
            try:
                self._communicate_factory(text, self._voice).save_sync(str(mp3_path))
            except Exception as e:
                raise EdgeTTSError(f"EdgeTTS synthesis request failed: {e}") from e

            if mp3_path.stat().st_size == 0:
                raise EdgeTTSError("EdgeTTS returned no audio data")

            try:
                segment = AudioSegment.from_file(str(mp3_path), format="mp3")
                segment = segment.set_channels(1).set_frame_rate(self._sample_rate).set_sample_width(2)
                pcm_bytes = segment.raw_data
            except Exception as e:
                raise EdgeTTSError(
                    f"could not decode EdgeTTS's MP3 response (is ffmpeg installed and on PATH?): {e}"
                ) from e
        finally:
            try:
                mp3_path.unlink(missing_ok=True)
            except OSError:
                _log.warning("Could not delete temporary file %s (still in use)", mp3_path)

        _log.info("EdgeTTS synthesis succeeded (%d PCM bytes)", len(pcm_bytes))
        return pcm_bytes, self._sample_rate
