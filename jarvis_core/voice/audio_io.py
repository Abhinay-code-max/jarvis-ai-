"""
jarvis_core/voice/audio_io.py
================================
Microphone capture and speaker playback primitives — push-to-talk
recording in, raw PCM playback out. No Claude/STT/TTS logic lives here;
session.py is the only caller. Built on `sounddevice` (PortAudio
bindings) directly rather than a heavier framework, since all this
needs is "record while the button's held" and "play these samples, and
let me interrupt them" — see AudioPlayer.stop(), which is how barge-in
is implemented.

`sounddevice` is imported lazily, inside each class's __init__, so
importing this module never requires a working audio device or even the
package to be installed — tests exercise the surrounding logic
(session.py) against hand-written stand-ins for these two classes
rather than real hardware.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

_log = logging.getLogger("jarvis_core.voice.audio_io")


class AudioIOError(Exception):
    """Raised for a device-level failure: no microphone/speaker, device
    busy, or an invalid operation (e.g. stopping a recorder that was
    never started)."""


class MicRecorder:
    """Push-to-talk capture: start() begins buffering audio, stop()
    ends it and returns everything captured as one float32 mono array
    at `sample_rate`."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1, sd_module: Any | None = None):
        self.sample_rate = sample_rate
        self._channels = channels
        self._sd = sd_module
        self._stream = None
        self._frames: list[np.ndarray] = []
        self._recording = False

    def _sounddevice(self):
        if self._sd is not None:
            return self._sd
        try:
            import sounddevice as sd
        except ImportError as e:
            raise AudioIOError(
                "sounddevice is not installed — run 'pip install sounddevice' (see requirements.txt)"
            ) from e
        self._sd = sd
        return sd

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self) -> None:
        if self._recording:
            raise AudioIOError("recording is already in progress")
        sd = self._sounddevice()
        self._frames = []

        def _callback(indata, _frames, _time_info, status):
            if status:
                _log.warning("Mic input status: %s", status)
            self._frames.append(indata.copy())

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate, channels=self._channels, dtype="float32", callback=_callback,
                latency="high",  # more PortAudio buffering slack; push-to-talk has no real-time-monitoring
                                  # need for low latency here, and this is cheap insurance against the input
                                  # buffer overflowing when something else briefly holds the GIL (e.g. UI paint work)
            )
            self._stream.start()
        except Exception as e:
            self._stream = None
            raise AudioIOError(f"could not start microphone recording: {e}") from e
        self._recording = True

    def stop(self) -> np.ndarray:
        if not self._recording:
            raise AudioIOError("no recording is in progress")
        try:
            self._stream.stop()
            self._stream.close()
        except Exception as e:
            raise AudioIOError(f"could not stop microphone recording: {e}") from e
        finally:
            self._stream = None
            self._recording = False

        if not self._frames:
            return np.array([], dtype=np.float32)
        return np.concatenate(self._frames, axis=0).reshape(-1).astype(np.float32)


class AudioPlayer:
    """Blocking playback with a thread-safe stop() — the barge-in
    control. play() blocks the calling thread until playback finishes
    or stop() is called (from any thread); the polling loop checking
    `_stop_event` is what makes an in-progress call to stop() actually
    cut playback short rather than only preventing a future play()."""

    def __init__(self, sd_module: Any | None = None, poll_interval_sec: float = 0.05):
        self._sd = sd_module
        self._poll_interval_sec = poll_interval_sec
        self._stop_event = threading.Event()

    def _sounddevice(self):
        if self._sd is not None:
            return self._sd
        try:
            import sounddevice as sd
        except ImportError as e:
            raise AudioIOError(
                "sounddevice is not installed — run 'pip install sounddevice' (see requirements.txt)"
            ) from e
        self._sd = sd
        return sd

    def play(self, pcm_bytes: bytes, sample_rate: int) -> None:
        if not pcm_bytes:
            return
        sd = self._sounddevice()
        audio = np.frombuffer(pcm_bytes, dtype=np.int16)
        self._stop_event.clear()
        try:
            sd.play(audio, sample_rate)
            while True:
                stream = sd.get_stream()
                if stream is None or not stream.active:
                    break
                if self._stop_event.is_set():
                    sd.stop()
                    break
                time.sleep(self._poll_interval_sec)
        except Exception as e:
            raise AudioIOError(f"could not play audio: {e}") from e

    def stop(self) -> None:
        """Safe to call from any thread, at any time (including when
        nothing is playing — a no-op in that case)."""
        self._stop_event.set()
        if self._sd is not None:
            try:
                self._sd.stop()
            except Exception:
                _log.exception("Error while stopping audio playback")
