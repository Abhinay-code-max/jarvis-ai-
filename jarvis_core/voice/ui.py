"""
jarvis_core/voice/ui.py
==========================
PyQt6 front end for the voice layer — a transcript view, a state
indicator (idle/listening/thinking/speaking), a push-to-talk button,
a Stop (barge-in) button, and a New Session button. Same stack as
jarvis_core/dashboard.py, for consistency; this file does not import
from or modify JARVIS-XL's ui.py.

All the actual pipeline logic lives in session.py's VoiceSession, which
has no Qt dependency — this file only wires Qt signals/slots and a
background thread around it:

  * `QPushButton.pressed`/`released` drive push-to-talk directly (no
    custom mouse-event handling needed — QAbstractButton already
    exposes both signals).
  * The STT -> Claude -> TTS pipeline (VoiceSession.stop_listening_and_respond())
    is slow and blocking, so it runs on a plain background thread, never
    the UI thread. VoiceSession's callbacks run on that same background
    thread; `_SessionSignals` (a QObject living in the main thread)
    is what makes updating the transcript/state labels from there safe —
    emitting a signal from a worker thread onto a receiver that lives on
    the main thread is queued and delivered by Qt's event loop, which is
    the standard safe cross-thread pattern in PyQt6.
  * The Stop button calls VoiceSession.interrupt() directly from the UI
    thread; AudioPlayer.stop() (what interrupt() ultimately calls) is
    explicitly designed to be callable while playback is running on the
    worker thread — see audio_io.py.

Launched independently, alongside (not inside) jarvis_core's other two
processes: `python -m jarvis_core.voice`.
"""
from __future__ import annotations

import logging
import sys
import threading

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from jarvis_core.config import resolve_db_path
from jarvis_core.voice import auth as voice_auth
from jarvis_core.voice.audio_io import AudioPlayer, MicRecorder
from jarvis_core.voice.config import VoiceConfig, VoiceConfigError
from jarvis_core.voice.session import VoiceSession, VoiceState
from jarvis_core.voice.stt import WhisperTranscriber
from jarvis_core.voice.store import VoiceStore, VoiceStoreError
from jarvis_core.voice.tts import ElevenLabsTTS
from jarvis_core.voice.voice_reasoner import VoiceReasoner

_log = logging.getLogger("jarvis_core.voice.ui")

_STATE_COLORS = {
    VoiceState.IDLE.value: "#5ab8cc",
    VoiceState.LISTENING.value: "#00d4ff",
    VoiceState.THINKING.value: "#ffb020",
    VoiceState.SPEAKING.value: "#00cc70",
}


class _SessionSignals(QObject):
    """Cross-thread bridge: VoiceSession's callbacks (invoked from the
    background pipeline thread) emit these; MainWindow's slots (running
    on the main/UI thread) receive them via Qt's queued connection."""

    state_changed = pyqtSignal(str)
    transcript = pyqtSignal(str, str)
    error = pyqtSignal(str)


class MainWindow(QMainWindow):
    def __init__(
        self,
        recorder: MicRecorder,
        player: AudioPlayer,
        transcriber: WhisperTranscriber,
        reasoner: VoiceReasoner,
        tts: ElevenLabsTTS,
        store: VoiceStore,
    ):
        super().__init__()
        self._recorder = recorder
        self._player = player
        self._transcriber = transcriber
        self._reasoner = reasoner
        self._tts = tts
        self._store = store

        self._signals = _SessionSignals()
        self._signals.state_changed.connect(self._on_state_changed)
        self._signals.transcript.connect(self._on_transcript)
        self._signals.error.connect(self._on_error)

        self._session = self._new_session()

        self.setWindowTitle("JARVIS Voice")
        self.resize(760, 640)
        self._build_ui()

    def _new_session(self) -> VoiceSession:
        return VoiceSession(
            recorder=self._recorder,
            player=self._player,
            transcriber=self._transcriber,
            reasoner=self._reasoner,
            tts=self._tts,
            store=self._store,
            on_state_changed=lambda state: self._signals.state_changed.emit(state.value),
            on_transcript=lambda role, text: self._signals.transcript.emit(role, text),
            on_error=lambda message: self._signals.error.emit(message),
        )

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)

        self._state_label = QLabel("Idle")
        self._state_label.setStyleSheet(f"font-weight: bold; font-size: 14pt; color: {_STATE_COLORS[VoiceState.IDLE.value]};")
        layout.addWidget(self._state_label)

        self._transcript = QTextEdit()
        self._transcript.setReadOnly(True)
        layout.addWidget(self._transcript)

        controls = QHBoxLayout()

        self._talk_btn = QPushButton("Hold to Talk")
        self._talk_btn.pressed.connect(self._on_talk_pressed)
        self._talk_btn.released.connect(self._on_talk_released)
        controls.addWidget(self._talk_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        controls.addWidget(self._stop_btn)

        self._new_session_btn = QPushButton("New Session")
        self._new_session_btn.clicked.connect(self._on_new_session_clicked)
        controls.addWidget(self._new_session_btn)

        layout.addLayout(controls)
        self.setCentralWidget(central)

    # -- button handlers --------------------------------------------------

    def _on_talk_pressed(self) -> None:
        if self._session.state != VoiceState.IDLE:
            return
        self._talk_btn.setEnabled(False)  # re-enabled once state returns to idle
        self._session.start_listening()

    def _on_talk_released(self) -> None:
        if self._session.state != VoiceState.LISTENING:
            return
        thread = threading.Thread(target=self._session.stop_listening_and_respond, daemon=True)
        thread.start()

    def _on_stop_clicked(self) -> None:
        self._session.interrupt()

    def _on_new_session_clicked(self) -> None:
        self._session.interrupt()
        self._session = self._new_session()
        self._transcript.clear()
        self._state_label.setText("Idle")
        self._state_label.setStyleSheet(f"font-weight: bold; font-size: 14pt; color: {_STATE_COLORS[VoiceState.IDLE.value]};")
        self._talk_btn.setEnabled(True)

    # -- signal slots (main thread) ----------------------------------------

    def _on_state_changed(self, state_value: str) -> None:
        self._state_label.setText(state_value.capitalize())
        self._state_label.setStyleSheet(
            f"font-weight: bold; font-size: 14pt; color: {_STATE_COLORS.get(state_value, '#ffffff')};"
        )
        if state_value == VoiceState.IDLE.value:
            self._talk_btn.setEnabled(True)

    def _on_transcript(self, role: str, text: str) -> None:
        speaker = "You" if role == "user" else "JARVIS"
        self._transcript.append(f"<b>{speaker}:</b> {text}")

    def _on_error(self, message: str) -> None:
        self._transcript.append(f'<span style="color:#ff3355;">Error: {message}</span>')


def run() -> int:
    """Entry point for `python -m jarvis_core.voice`."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    try:
        config = VoiceConfig.load()
    except VoiceConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    if voice_auth.get_key(config.elevenlabs_key_path) is None:
        print(
            f"No ElevenLabs API key stored at {config.elevenlabs_key_path} — run "
            "'python -m jarvis_core.voice.auth set' first.",
            file=sys.stderr,
        )
        return 1

    db_path = resolve_db_path()
    try:
        store = VoiceStore(db_path)
    except VoiceStoreError as e:
        print(f"Could not open the voice transcript database at {db_path}: {e}", file=sys.stderr)
        return 1

    recorder = MicRecorder(sample_rate=config.sample_rate)
    player = AudioPlayer()
    transcriber = WhisperTranscriber(
        model_size=config.whisper_model_size,
        device=config.whisper_device,
        compute_type=config.whisper_compute_type,
        min_utterance_sec=config.min_utterance_sec,
        no_speech_prob_threshold=config.no_speech_prob_threshold,
    )
    reasoner = VoiceReasoner(model=config.claude_model, max_tokens=config.claude_max_tokens, effort=config.claude_effort)
    tts = ElevenLabsTTS(
        voice_id=config.elevenlabs_voice_id,
        model_id=config.elevenlabs_model_id,
        output_format=config.elevenlabs_output_format,
        get_api_key=lambda: voice_auth.get_key(config.elevenlabs_key_path),
        request_timeout_sec=config.request_timeout_sec,
        max_retries=config.max_retries,
        retry_backoff_base_sec=config.retry_backoff_base_sec,
    )

    app = QApplication(sys.argv)
    window = MainWindow(recorder, player, transcriber, reasoner, tts, store)
    window.show()
    exit_code = app.exec()
    store.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(run())
