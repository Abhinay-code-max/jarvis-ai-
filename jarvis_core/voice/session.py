"""
jarvis_core/voice/session.py
===============================
VoiceSession — orchestrates one push-to-talk turn: recorded audio -> STT
-> (if a real transcript) Claude -> TTS -> playback, plus the running
conversation history and the interrupt/barge-in control. No PyQt6
import here — this is the GUI-free logic layer, following the same
split dashboard.py established between DashboardModel (plain Python,
tested directly) and its Qt widget classes (thin, untested-in-isolation
wiring). jarvis_core/voice/ui.py wires Qt signals and a background
thread around this class; the class itself needs neither to be
exercised in tests.

Every external failure (recorder, STT, Claude, TTS) stops the turn at
that point and is reported via `on_error` rather than raised past this
class — same "never crash the whole session over one bad turn"
philosophy as jarvis_core/engine.py's per-item error containment,
applied here per-utterance instead of per-queue-item. A silent/noisy
utterance (STT returns None) is explicitly NOT an error: it's the
misfire-resilience case this whole feature exists to handle correctly,
so it ends the turn quietly, with no Claude call and no error callback.

Persisting a turn to VoiceStore is treated as best-effort, not
turn-blocking: a database write failure is logged (with a full
traceback — never swallowed) but does not stop the rest of the
pipeline or surface as a user-facing error, since a background
persistence hiccup losing one row of transcript history is not a
reason to also fail (or interrupt) the live conversation the user is
having right now.
"""
from __future__ import annotations

import logging
import uuid
from enum import Enum
from typing import Any, Callable

import numpy as np

from jarvis_core.voice.audio_io import AudioIOError, AudioPlayer, MicRecorder
from jarvis_core.voice.stt import SttError, WhisperTranscriber
from jarvis_core.voice.store import VoiceStore, VoiceStoreError
from jarvis_core.voice.tts import ElevenLabsError, ElevenLabsTTS
from jarvis_core.voice.voice_reasoner import VoiceReasoner, VoiceReasonerError

_log = logging.getLogger("jarvis_core.voice.session")


class VoiceState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class VoiceSession:
    def __init__(
        self,
        recorder: MicRecorder,
        player: AudioPlayer,
        transcriber: WhisperTranscriber,
        reasoner: VoiceReasoner,
        tts: ElevenLabsTTS,
        store: VoiceStore,
        session_id: str | None = None,
        on_state_changed: Callable[[VoiceState], None] | None = None,
        on_transcript: Callable[[str, str], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ):
        self._recorder = recorder
        self._player = player
        self._transcriber = transcriber
        self._reasoner = reasoner
        self._tts = tts
        self._store = store
        self.session_id = session_id or str(uuid.uuid4())
        self._on_state_changed = on_state_changed
        self._on_transcript = on_transcript
        self._on_error = on_error

        self._state = VoiceState.IDLE
        self._messages: list[dict[str, Any]] = []

    @property
    def state(self) -> VoiceState:
        return self._state

    # -- push-to-talk entry points ---------------------------------------

    def start_listening(self) -> None:
        """Call when the push-to-talk control is pressed."""
        if self._state != VoiceState.IDLE:
            _log.warning("start_listening() called while state is %s — ignoring", self._state)
            return
        try:
            self._recorder.start()
        except AudioIOError as e:
            self._emit_error(f"could not start recording: {e}")
            return
        self._set_state(VoiceState.LISTENING)

    def stop_listening_and_respond(self) -> None:
        """Call when the push-to-talk control is released. Runs the
        whole STT -> Claude -> TTS pipeline synchronously — the caller
        (ui.py) is expected to run this off the UI thread."""
        if self._state != VoiceState.LISTENING:
            _log.warning("stop_listening_and_respond() called while state is %s — ignoring", self._state)
            return
        try:
            audio = self._recorder.stop()
        except AudioIOError as e:
            self._set_state(VoiceState.IDLE)
            self._emit_error(f"could not stop recording: {e}")
            return

        self._set_state(VoiceState.THINKING)
        self._handle_utterance(audio)

    def interrupt(self) -> None:
        """Barge-in: stop whatever's currently playing, or discard an
        in-progress recording. Safe to call at any time, from any
        thread — AudioPlayer.stop() is explicitly designed for that
        (see its own docstring)."""
        self._player.stop()
        if self._state == VoiceState.LISTENING:
            try:
                self._recorder.stop()
            except AudioIOError:
                _log.exception("Error discarding in-progress recording on interrupt")
        self._set_state(VoiceState.IDLE)

    # -- pipeline ---------------------------------------------------------

    def _handle_utterance(self, audio: np.ndarray) -> None:
        try:
            transcript = self._transcriber.transcribe(audio, self._recorder.sample_rate)
        except SttError as e:
            self._set_state(VoiceState.IDLE)
            self._emit_error(f"speech-to-text failed: {e}")
            return

        if not transcript:
            # Explicitly not an error — see module docstring.
            _log.info("No speech detected in this utterance — nothing sent to Claude")
            self._set_state(VoiceState.IDLE)
            return

        self._emit_transcript("user", transcript)
        self._safe_record_turn("user", transcript)
        self._messages.append({"role": "user", "content": transcript})

        try:
            reply = self._reasoner.respond(self._messages)
        except VoiceReasonerError as e:
            self._set_state(VoiceState.IDLE)
            self._emit_error(f"Claude request failed: {e}")
            return

        self._messages.append({"role": "assistant", "content": reply})
        self._emit_transcript("assistant", reply)
        self._safe_record_turn("assistant", reply)

        self._set_state(VoiceState.SPEAKING)
        try:
            pcm, sample_rate = self._tts.synthesize(reply)
        except ElevenLabsError as e:
            self._set_state(VoiceState.IDLE)
            self._emit_error(f"text-to-speech failed: {e}")
            return

        try:
            self._player.play(pcm, sample_rate)
        except AudioIOError as e:
            self._emit_error(f"audio playback failed: {e}")

        self._set_state(VoiceState.IDLE)

    # -- callbacks ----------------------------------------------------------

    def _set_state(self, state: VoiceState) -> None:
        self._state = state
        if self._on_state_changed:
            self._on_state_changed(state)

    def _emit_transcript(self, role: str, text: str) -> None:
        if self._on_transcript:
            self._on_transcript(role, text)

    def _emit_error(self, message: str) -> None:
        _log.error(message)
        if self._on_error:
            self._on_error(message)

    def _safe_record_turn(self, role: str, text: str) -> None:
        try:
            self._store.record_turn(self.session_id, role, text)
        except VoiceStoreError:
            _log.exception(
                "Could not persist %s turn for session %s — continuing the live conversation anyway",
                role, self.session_id,
            )
