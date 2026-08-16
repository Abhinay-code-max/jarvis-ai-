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

Visual design (Phase A restyle): dark/cyan palette, deliberately the
exact same hex constants as jarvis_core/dashboard.py's STYLESHEET (see
that module's own comment on this) so the two JARVIS UIs read as one
system — defined independently here rather than imported from
dashboard.py, since neither module should depend on the other for
something this small. `StateIndicator` is a hand-painted, animated
concentric-ring widget (QPainter + QPropertyAnimation) standing in for
the reference design's glowing circular voice-state indicator — same
visual language (rings, glow, centered, state-colored), not a pixel
copy. The conversation panel is a scrollable column of chat-bubble
widgets (`_add_bubble`) rather than a single QTextEdit, so user and
JARVIS turns can be styled/aligned independently.

The left sidebar (`self.side_panel_layout`) is deliberately built and
wired up empty in this phase — Phase B's System Stats / Uptime /
Weather / Camera panels are meant to `addWidget()` into it without any
further layout changes here.

Launched independently, alongside (not inside) jarvis_core's other two
processes: `python -m jarvis_core.voice`.
"""
from __future__ import annotations

import logging
import sys
import threading
from datetime import datetime

from PyQt6.QtCore import QEasingCurve, QPointF, QPropertyAnimation, Qt, QTimer, pyqtProperty, pyqtSignal, QObject
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
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

# Same family as dashboard.py's palette constants, reused verbatim for
# visual consistency between the two JARVIS UIs (see that module's own
# comment on this — defined independently here, not imported, so
# neither UI module depends on the other).
BG = "#00060a"
PANEL = "#010d14"
BORDER = "#0d3347"
PRI = "#00d4ff"
TEXT = "#c8f8ff"
TEXT_DIM = "#5ab8cc"
GREEN = "#00cc70"
RED = "#ff3355"
AMBER = "#ffb020"
GRAY = "#3a5a66"

_STATE_COLORS = {
    VoiceState.IDLE.value: TEXT_DIM,
    VoiceState.LISTENING.value: PRI,
    VoiceState.THINKING.value: AMBER,
    VoiceState.SPEAKING.value: GREEN,
}

# Ring pulse period per state, ms — faster pulse reads as "more active"
# in lieu of any real audio-amplitude signal to drive it from.
_STATE_PULSE_MS = {
    VoiceState.IDLE.value: 2400,
    VoiceState.LISTENING.value: 1100,
    VoiceState.THINKING.value: 700,
    VoiceState.SPEAKING.value: 900,
}

STYLESHEET = f"""
    QMainWindow, QWidget {{ background: {BG}; color: {TEXT}; }}
    QFrame#topBar {{ background: {PANEL}; border-bottom: 1px solid {BORDER}; }}
    QFrame#sidePanel {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; }}
    QFrame#bottomBar {{ background: {PANEL}; border-top: 1px solid {BORDER}; }}
    QLabel#statusPill {{
        color: {GREEN}; border: 1px solid {GREEN}; border-radius: 9px;
        padding: 2px 10px; font-weight: bold; font-size: 9pt;
    }}
    QLabel#clock {{ color: {PRI}; font-size: 14pt; font-weight: bold; }}
    QLabel#dateLabel {{ color: {TEXT_DIM}; font-size: 9pt; }}
    QLabel#stateName {{ font-weight: bold; font-size: 12pt; letter-spacing: 2px; }}
    QScrollArea {{ border: none; background: {BG}; }}
    QPushButton[class="iconBtn"] {{
        background: {PANEL}; color: {PRI}; border: 2px solid {BORDER};
        border-radius: 32px; font-size: 20pt; min-width: 64px; min-height: 64px;
        max-width: 64px; max-height: 64px;
    }}
    QPushButton[class="iconBtn"]:hover {{ border-color: {PRI}; }}
    QPushButton[class="iconBtn"]:pressed {{ background: {BORDER}; }}
    QPushButton[class="iconBtn"]:disabled {{ color: {GRAY}; border-color: {BORDER}; }}
    QPushButton#stopBtn {{ color: {RED}; }}
    QPushButton#newSessionBtn {{ color: {TEXT_DIM}; font-size: 14pt; }}
"""

_BUBBLE_STYLE_USER = f"""
    background: {PANEL}; color: {TEXT}; border: 1px solid {PRI};
    border-radius: 12px; padding: 8px 12px;
"""
_BUBBLE_STYLE_JARVIS = f"""
    background: {PANEL}; color: {TEXT}; border: 1px solid {BORDER};
    border-radius: 12px; padding: 8px 12px;
"""
_BUBBLE_STYLE_ERROR = f"""
    background: {PANEL}; color: {RED}; border: 1px solid {RED};
    border-radius: 12px; padding: 8px 12px;
"""


class StateIndicator(QWidget):
    """Hand-painted concentric-ring voice-state indicator. A looping
    QPropertyAnimation drives `pulse` (0..1) once per frame-ish tick;
    paintEvent() reads it to place three fading rings expanding outward
    from a solid center dot, all tinted by the current VoiceState's
    color. set_state() swaps the color and restarts the animation at
    a state-appropriate speed (see _STATE_PULSE_MS)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(220, 220)
        self._state = VoiceState.IDLE.value
        self._pulse = 0.0

        self._anim = QPropertyAnimation(self, b"pulse")
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setLoopCount(-1)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutSine)
        self._apply_state_speed()
        self._anim.start()

    def _get_pulse(self) -> float:
        return self._pulse

    def _set_pulse(self, value: float) -> None:
        self._pulse = value
        self.update()

    pulse = pyqtProperty(float, _get_pulse, _set_pulse)

    def _apply_state_speed(self) -> None:
        self._anim.setDuration(_STATE_PULSE_MS.get(self._state, 2000))

    def set_state(self, state_value: str) -> None:
        self._state = state_value
        self._apply_state_speed()
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        cx, cy = rect.center().x(), rect.center().y()
        base_radius = min(rect.width(), rect.height()) / 2 - 12
        color = QColor(_STATE_COLORS.get(self._state, PRI))

        for i in range(3):
            t = (self._pulse + i / 3.0) % 1.0
            radius = base_radius * (0.45 + 0.55 * t)
            ring_color = QColor(color)
            ring_color.setAlpha(int(200 * (1.0 - t)))
            pen = QPen(ring_color)
            pen.setWidthF(2.5)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(cx, cy), radius, radius)

        halo = QColor(color)
        halo.setAlpha(50)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(halo)
        painter.drawEllipse(QPointF(cx, cy), base_radius * 0.32, base_radius * 0.32)

        painter.setBrush(color)
        painter.drawEllipse(QPointF(cx, cy), base_radius * 0.18, base_radius * 0.18)


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
        self.resize(1040, 700)
        self.setStyleSheet(STYLESHEET)
        self._build_ui()
        self._start_clock()

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

    # -- layout --------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_top_bar())

        content = QHBoxLayout()
        content.setContentsMargins(16, 16, 16, 16)
        content.setSpacing(16)

        self.side_panel = QFrame()
        self.side_panel.setObjectName("sidePanel")
        self.side_panel.setFixedWidth(240)
        self.side_panel_layout = QVBoxLayout(self.side_panel)
        self.side_panel_layout.setContentsMargins(12, 12, 12, 12)
        self.side_panel_layout.addStretch(1)  # Phase B panels insert above this
        content.addWidget(self.side_panel)

        content.addLayout(self._build_center_column(), stretch=1)
        content.addLayout(self._build_conversation_panel(), stretch=1)

        root.addLayout(content, stretch=1)
        root.addWidget(self._build_bottom_bar())

        self.setCentralWidget(central)

    def _build_top_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("topBar")
        bar.setFixedHeight(56)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(20, 8, 20, 8)

        title = QLabel("JARVIS")
        title.setStyleSheet(f"color: {PRI}; font-weight: bold; font-size: 13pt; letter-spacing: 3px;")
        layout.addWidget(title)

        self._status_pill = QLabel("● ONLINE")
        self._status_pill.setObjectName("statusPill")
        layout.addWidget(self._status_pill)

        layout.addItem(QSpacerItem(20, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

        date_clock = QVBoxLayout()
        date_clock.setSpacing(0)
        self._clock_label = QLabel("--:--:--")
        self._clock_label.setObjectName("clock")
        self._clock_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._date_label = QLabel("")
        self._date_label.setObjectName("dateLabel")
        self._date_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        date_clock.addWidget(self._clock_label)
        date_clock.addWidget(self._date_label)
        layout.addLayout(date_clock)

        return bar

    def _build_center_column(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.addStretch(1)

        self._indicator = StateIndicator()
        col.addWidget(self._indicator, alignment=Qt.AlignmentFlag.AlignHCenter)

        self._state_label = QLabel(VoiceState.IDLE.value.upper())
        self._state_label.setObjectName("stateName")
        self._state_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._state_label.setStyleSheet(f"color: {_STATE_COLORS[VoiceState.IDLE.value]};")
        col.addWidget(self._state_label)

        col.addStretch(1)
        return col

    def _build_conversation_panel(self) -> QVBoxLayout:
        col = QVBoxLayout()

        heading = QLabel("CONVERSATION")
        heading.setStyleSheet(f"color: {TEXT_DIM}; font-weight: bold; font-size: 9pt; letter-spacing: 2px;")
        col.addWidget(heading)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._bubble_container = QWidget()
        self._bubble_layout = QVBoxLayout(self._bubble_container)
        self._bubble_layout.addStretch(1)
        self._bubble_layout.setSpacing(10)
        self._scroll.setWidget(self._bubble_container)
        col.addWidget(self._scroll, stretch=1)

        return col

    def _build_bottom_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("bottomBar")
        bar.setFixedHeight(96)
        layout = QHBoxLayout(bar)
        layout.addStretch(1)

        self._new_session_btn = QPushButton("⟳")
        self._new_session_btn.setObjectName("newSessionBtn")
        self._new_session_btn.setProperty("class", "iconBtn")
        self._new_session_btn.setToolTip("New Session")
        self._new_session_btn.clicked.connect(self._on_new_session_clicked)
        layout.addWidget(self._new_session_btn)

        self._talk_btn = QPushButton("🎙")
        self._talk_btn.setProperty("class", "iconBtn")
        self._talk_btn.setToolTip("Hold to Talk")
        self._talk_btn.pressed.connect(self._on_talk_pressed)
        self._talk_btn.released.connect(self._on_talk_released)
        layout.addWidget(self._talk_btn)

        self._stop_btn = QPushButton("■")
        self._stop_btn.setObjectName("stopBtn")
        self._stop_btn.setProperty("class", "iconBtn")
        self._stop_btn.setToolTip("Stop")
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        layout.addWidget(self._stop_btn)

        layout.addStretch(1)
        return bar

    # -- clock -----------------------------------------------------------

    def _start_clock(self) -> None:
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._tick_clock)
        self._clock_timer.start(1000)
        self._tick_clock()

    def _tick_clock(self) -> None:
        now = datetime.now()
        self._clock_label.setText(now.strftime("%H:%M:%S"))
        self._date_label.setText(now.strftime("%A, %B %d"))

    # -- conversation bubbles ---------------------------------------------

    def _add_bubble(self, role: str, text: str, *, is_error: bool = False) -> None:
        bubble = QLabel(text)
        bubble.setWordWrap(True)
        bubble.setMaximumWidth(320)
        if is_error:
            bubble.setStyleSheet(_BUBBLE_STYLE_ERROR)
        elif role == "user":
            bubble.setStyleSheet(_BUBBLE_STYLE_USER)
        else:
            bubble.setStyleSheet(_BUBBLE_STYLE_JARVIS)

        row = QHBoxLayout()
        if role == "user" and not is_error:
            row.addStretch(1)
            row.addWidget(bubble)
        else:
            row.addWidget(bubble)
            row.addStretch(1)

        row_widget = QWidget()
        row_widget.setLayout(row)
        # Insert before the trailing stretch so new bubbles append at the bottom.
        self._bubble_layout.insertWidget(self._bubble_layout.count() - 1, row_widget)

        QTimer.singleShot(0, lambda: self._scroll.verticalScrollBar().setValue(
            self._scroll.verticalScrollBar().maximum()
        ))

    def _clear_bubbles(self) -> None:
        while self._bubble_layout.count() > 1:  # keep the trailing stretch
            item = self._bubble_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

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
        self._clear_bubbles()
        self._on_state_changed(VoiceState.IDLE.value)
        self._talk_btn.setEnabled(True)

    # -- signal slots (main thread) ----------------------------------------

    def _on_state_changed(self, state_value: str) -> None:
        self._state_label.setText(state_value.upper())
        color = _STATE_COLORS.get(state_value, PRI)
        self._state_label.setStyleSheet(f"color: {color};")
        self._indicator.set_state(state_value)
        if state_value == VoiceState.IDLE.value:
            self._talk_btn.setEnabled(True)

    def _on_transcript(self, role: str, text: str) -> None:
        self._add_bubble(role, text)

    def _on_error(self, message: str) -> None:
        self._add_bubble("error", f"Error: {message}", is_error=True)


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
