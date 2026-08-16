"""
jarvis_core/voice/ui.py
==========================
PyQt6 front end for the voice layer. All the actual pipeline logic
lives in session.py's VoiceSession, which has no Qt dependency — this
file only wires Qt signals/slots and a background thread around it:

  * `QPushButton.pressed`/`released` drive push-to-talk directly (no
    custom mouse-event handling needed — QAbstractButton already
    exposes both signals).
  * The STT -> Claude -> TTS pipeline (VoiceSession.stop_listening_and_respond())
    is slow and blocking, so it runs on a plain background thread, never
    the UI thread. VoiceSession's callbacks run on that same background
    thread; `_SessionSignals` (a QObject living in the main thread)
    is what makes updating the UI from there safe — emitting a signal
    from a worker thread onto a receiver that lives on the main thread
    is queued and delivered by Qt's event loop, the standard safe
    cross-thread pattern in PyQt6.
  * The Stop button calls VoiceSession.interrupt() directly from the UI
    thread; AudioPlayer.stop() (what interrupt() ultimately calls) is
    explicitly designed to be callable while playback is running on the
    worker thread — see audio_io.py.

Visual design: this HUD (palette, HudCanvas's animated ring/scan/orb
indicator, MetricBar, LogWidget, the header/left-panel/center/right-panel/
footer composition) is ported from the user's prior JARVIS-XL/MARK XL
project's ui.py, adapted to this codebase rather than copied verbatim:
  * HudCanvas drives off this project's real VoiceState values
    (idle/listening/thinking/speaking) instead of MARK XL's own
    muted/speaking/state-string model, and never loads a face image —
    there's no face asset in this project, so it always renders the
    orb + "J.A.R.V.I.S" fallback.
  * The left panel's MetricBar/_SysMetrics (CPU + MEM only — GPU/NET/TMP
    and their platform-specific subprocess probes were dropped, out of
    scope here) double as this UI's "System Stats" panel; the uptime
    label tracks this *process's* start time, not OS boot time.
    Both were separately agreed, no-open-questions Phase B items.
  * LogWidget (a typewriter-effect activity log) replaces the plain
    chat-bubble panel from an earlier revision of this file, matching
    the ported design faithfully instead of mixing two visual languages.
  * Not ported: SetupOverlay/StartupPanel (multi-engine STT/LLM/TTS
    picker — this project's config is entirely `.env`-driven, no
    swappable engines) and FileDropZone (drag-drop file upload — never
    requested here). Push-to-talk stays press-and-hold (session.py's
    actual state machine), not MARK XL's click-to-mute toggle — the
    mic control is restyled to match, not behaviorally replaced.

Face recognition: on startup, a background thread calls
FaceIdentifier.identify() (see face_id.py) for up to ~5s and reports
the result (or "not recognized") into the header/activity log. A small
enroll control lets you save a new face encoding. Both directions
degrade silently (no camera, no match, dependencies missing) rather
than blocking startup or crashing the UI — see face_id.py's own
docstring for exactly what is and isn't stored.

Launched independently, alongside (not inside) jarvis_core's other two
processes: `python -m jarvis_core.voice`.
"""
from __future__ import annotations

import logging
import math
import random
import sys
import threading
import time
from datetime import datetime

import psutil
from PyQt6.QtCore import QEasingCurve, QObject, QPointF, QRectF, Qt, QTimer, pyqtProperty, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from jarvis_core.config import resolve_db_path
from jarvis_core.voice import auth as voice_auth
from jarvis_core.voice.audio_io import AudioPlayer, MicRecorder
from jarvis_core.voice.config import VoiceConfig, VoiceConfigError
from jarvis_core.voice.face_id import FaceIdentifier
from jarvis_core.voice.session import VoiceSession, VoiceState
from jarvis_core.voice.stt import WhisperTranscriber
from jarvis_core.voice.store import VoiceStore, VoiceStoreError
from jarvis_core.voice.tts import ElevenLabsTTS
from jarvis_core.voice.voice_reasoner import VoiceReasoner

_log = logging.getLogger("jarvis_core.voice.ui")

# Full palette ported from MARK XL's `C` class verbatim — same family
# dashboard.py's own palette constants already use (see that module's
# comment), extended here with the rest of the ported HUD's colors.
BG = "#00060a"
PANEL = "#010d14"
PANEL2 = "#010f18"
BORDER = "#0d3347"
BORDER_B = "#1a5c7a"
BORDER_A = "#0f4060"
PRI = "#00d4ff"
PRI_DIM = "#007a99"
PRI_GHO = "#001f2e"
ACC = "#ff6b00"
ACC2 = "#ffcc00"
GREEN = "#00ff88"
GREEN_D = "#00aa55"
RED = "#ff3355"
TEXT = "#8ffcff"
TEXT_DIM = "#3a8a9a"
TEXT_MED = "#5ab8cc"
WHITE = "#d8f8ff"
DARK = "#000d14"
BAR_BG = "#011520"

_STATE_COLORS = {
    VoiceState.IDLE.value: PRI,
    VoiceState.LISTENING.value: GREEN,
    VoiceState.THINKING.value: ACC2,
    VoiceState.SPEAKING.value: ACC,
}


def qcol(hex_str: str, alpha: int = 255) -> QColor:
    c = QColor(hex_str)
    c.setAlpha(alpha)
    return c


class _SysMetrics:
    """Background-sampled CPU/RAM, matching Phase B's approved "System
    Stats" scope exactly (no GPU/NET/TMP — those aren't requested and
    would add untested, platform-specific subprocess probes)."""

    def __init__(self):
        self.cpu = 0.0
        self.mem = 0.0
        self._lock = threading.Lock()
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._running:
            try:
                cpu = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory().percent
                with self._lock:
                    self.cpu = cpu
                    self.mem = mem
            except Exception:
                pass
            time.sleep(1.5)

    def snapshot(self) -> dict:
        with self._lock:
            return {"cpu": self.cpu, "mem": self.mem}


_metrics = _SysMetrics()


class HudCanvas(QWidget):
    """Hand-painted animated voice-state indicator — ported from MARK
    XL's HudCanvas. Grid dots, halo glow, expanding pulse rings,
    spinning arc rings, radar scanners, tick marks, crosshair, corner
    brackets, a central orb, particle bursts while speaking, animated
    status text, and a waveform bar — all driven by a 60fps QTimer, all
    tinted by the current VoiceState's color. No face image (this
    project has no face asset for JARVIS itself); the orb + "J.A.R.V.I.S"
    text is always the center content."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.state = VoiceState.IDLE.value
        self.speaking = False

        self._tick = 0
        self._scale = 1.0
        self._tgt_scale = 1.0
        self._halo = 55.0
        self._tgt_halo = 55.0
        self._last_t = time.time()
        self._scan = 0.0
        self._scan2 = 180.0
        self._rings = [0.0, 120.0, 240.0]
        self._pulses: list[float] = [0.0, 50.0, 100.0]
        self._blink = True
        self._blink_tick = 0
        self._particles: list[list[float]] = []

        # ~25fps by default rather than the original port's 60fps (16ms):
        # this widget's paintEvent() does a lot of QPainter work in plain
        # Python every tick (grid dots, halo rings, arc rings, tick marks,
        # particles, waveform bars) — on a modest CPU that's real,
        # continuous main-thread load, felt as general UI/interaction lag
        # even outside any audio-capture concern. 25fps is still visibly
        # smooth for this kind of ambient animation.
        self._tmr = QTimer(self)
        self._tmr.timeout.connect(self._step)
        self._tmr.start(40)

    def set_state(self, state_value: str) -> None:
        self.state = state_value
        self.speaking = state_value == VoiceState.SPEAKING.value
        # Throttle further while the mic is actually capturing: Python's
        # GIL means busy main-thread paint work can delay sounddevice's
        # real-time input callback (also Python, also needs the GIL) long
        # enough to overflow PortAudio's input buffer — heard as repeated
        # "Mic input status: input overflow" warnings. Recording is the one
        # state where audio fidelity matters more than animation smoothness.
        self._tmr.setInterval(200 if state_value == VoiceState.LISTENING.value else 40)

    def _step(self):
        self._tick += 1
        now = time.time()
        if now - self._last_t > (0.12 if self.speaking else 0.5):
            if self.speaking:
                self._tgt_scale = random.uniform(1.06, 1.14)
                self._tgt_halo = random.uniform(145, 190)
            else:
                self._tgt_scale = random.uniform(1.001, 1.008)
                self._tgt_halo = random.uniform(48, 68)
            self._last_t = now

        sp = 0.38 if self.speaking else 0.15
        self._scale += (self._tgt_scale - self._scale) * sp
        self._halo += (self._tgt_halo - self._halo) * sp

        speeds = [1.3, -0.9, 2.0] if self.speaking else [0.55, -0.35, 0.9]
        for i, spd in enumerate(speeds):
            self._rings[i] = (self._rings[i] + spd) % 360

        self._scan = (self._scan + (3.0 if self.speaking else 1.3)) % 360
        self._scan2 = (self._scan2 + (-2.0 if self.speaking else -0.75)) % 360

        fw = min(self.width(), self.height())
        lim = fw * 0.74
        spd = 4.2 if self.speaking else 2.0
        self._pulses = [r + spd for r in self._pulses if r + spd < lim]
        if len(self._pulses) < 3 and random.random() < (0.07 if self.speaking else 0.025):
            self._pulses.append(0.0)

        if self.speaking and random.random() < 0.28:
            cx, cy = self.width() / 2, self.height() / 2
            ang = random.uniform(0, 2 * math.pi)
            r_s = fw * 0.28
            self._particles.append([
                cx + math.cos(ang) * r_s, cy + math.sin(ang) * r_s,
                math.cos(ang) * random.uniform(0.9, 2.4),
                math.sin(ang) * random.uniform(0.9, 2.4) - 0.4, 1.0,
            ])
        self._particles = [
            [p[0] + p[2], p[1] + p[3], p[2] * 0.97, p[3] * 0.97, p[4] - 0.028]
            for p in self._particles if p[4] > 0
        ]

        self._blink_tick += 1
        if self._blink_tick >= 38:
            self._blink = not self._blink
            self._blink_tick = 0
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), qcol(BG))

        W, H = self.width(), self.height()
        cx, cy = W / 2, H / 2
        fw = min(W, H)
        pri = qcol(_STATE_COLORS.get(self.state, PRI))

        p.setPen(QPen(qcol(PRI_GHO), 1))
        for x in range(0, W, 48):
            for y in range(0, H, 48):
                p.drawPoint(x, y)

        r_face = fw * 0.31

        for i in range(10):
            r = r_face * (1.8 - i * 0.08)
            frc = 1.0 - i / 10
            a = max(0, min(255, int(self._halo * 0.085 * frc)))
            col = QColor(pri)
            col.setAlpha(a)
            p.setPen(QPen(col, 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))

        for pr in self._pulses:
            a = max(0, int(230 * (1.0 - pr / (fw * 0.74))))
            col = QColor(pri)
            col.setAlpha(a)
            p.setPen(QPen(col, 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QRectF(cx - pr, cy - pr, pr * 2, pr * 2))

        for idx, (r_frac, w_r, arc_l, gap) in enumerate(
            [(0.48, 3, 115, 78), (0.40, 2, 78, 55), (0.32, 1, 56, 40)]
        ):
            ring_r = fw * r_frac
            base = self._rings[idx]
            a_val = max(0, min(255, int(self._halo * (1.0 - idx * 0.18))))
            col = QColor(pri)
            col.setAlpha(a_val)
            p.setPen(QPen(col, w_r))
            p.setBrush(Qt.BrushStyle.NoBrush)
            angle = base
            rect = QRectF(cx - ring_r, cy - ring_r, ring_r * 2, ring_r * 2)
            while angle < base + 360:
                p.drawArc(rect, int(angle * 16), int(arc_l * 16))
                angle += arc_l + gap

        sr = fw * 0.50
        sa = min(255, int(self._halo * 1.5))
        ex = 75 if self.speaking else 44
        scan_col = QColor(pri)
        scan_col.setAlpha(sa)
        p.setPen(QPen(scan_col, 2.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        srect = QRectF(cx - sr, cy - sr, sr * 2, sr * 2)
        p.drawArc(srect, int(self._scan * 16), int(ex * 16))
        p.setPen(QPen(qcol(ACC, sa // 2), 1.5))
        p.drawArc(srect, int(self._scan2 * 16), int(ex * 16))

        t_out, t_in = fw * 0.497, fw * 0.474
        p.setPen(QPen(qcol(PRI, 140), 1))
        for deg in range(0, 360, 10):
            rad = math.radians(deg)
            inn = t_in if deg % 30 == 0 else t_in + 6
            p.drawLine(
                QPointF(cx + t_out * math.cos(rad), cy - t_out * math.sin(rad)),
                QPointF(cx + inn * math.cos(rad), cy - inn * math.sin(rad)),
            )

        ch_r, gap_h = fw * 0.51, fw * 0.16
        p.setPen(QPen(qcol(PRI, int(self._halo * 0.5)), 1))
        p.drawLine(QPointF(cx - ch_r, cy), QPointF(cx - gap_h, cy))
        p.drawLine(QPointF(cx + gap_h, cy), QPointF(cx + ch_r, cy))
        p.drawLine(QPointF(cx, cy - ch_r), QPointF(cx, cy - gap_h))
        p.drawLine(QPointF(cx, cy + gap_h), QPointF(cx, cy + ch_r))

        bl = 24
        bc = qcol(PRI, 210)
        hl, hr = cx - fw // 2, cx + fw // 2
        ht, hb = cy - fw // 2, cy + fw // 2
        p.setPen(QPen(bc, 2))
        for bx, by, dx, dy in [(hl, ht, 1, 1), (hr, ht, -1, 1), (hl, hb, 1, -1), (hr, hb, -1, -1)]:
            p.drawLine(QPointF(bx, by), QPointF(bx + dx * bl, by))
            p.drawLine(QPointF(bx, by), QPointF(bx, by + dy * bl))

        orb_r = int(fw * 0.27 * self._scale)
        for i in range(8, 0, -1):
            r2 = int(orb_r * i / 8)
            frc = i / 8
            a = max(0, min(255, int(self._halo * 1.1 * frc)))
            base_col = QColor(pri)
            p.setBrush(QBrush(QColor(
                int(base_col.red() * frc), int(base_col.green() * frc), int(base_col.blue() * frc), a
            )))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QRectF(cx - r2, cy - r2, r2 * 2, r2 * 2))
        p.setPen(QPen(qcol(PRI, min(255, int(self._halo * 2))), 1))
        p.setFont(QFont("Courier New", 13, QFont.Weight.Bold))
        p.drawText(QRectF(cx - 80, cy - 14, 160, 28), Qt.AlignmentFlag.AlignCenter, "J.A.R.V.I.S")

        for pt in self._particles:
            a = max(0, min(255, int(pt[4] * 255)))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(qcol(PRI, a)))
            p.drawEllipse(QPointF(pt[0], pt[1]), 2.5, 2.5)

        sy = cy + fw * 0.40
        sym = "●" if self._blink else "○"
        label = self.state.upper()
        p.setPen(QPen(pri, 1))
        p.setFont(QFont("Courier New", 11, QFont.Weight.Bold))
        p.drawText(QRectF(0, sy, W, 26), Qt.AlignmentFlag.AlignCenter, f"{sym}  {label}")

        wy = sy + 30
        N, bw = 36, 8
        wx0 = (W - N * bw) / 2
        for i in range(N):
            if self.speaking:
                hgt = random.randint(3, 20)
                cl = pri if hgt > 12 else qcol(PRI_DIM)
            else:
                hgt = int(3 + 2 * math.sin(self._tick * 0.09 + i * 0.6))
                cl = qcol(BORDER_B)
            p.fillRect(QRectF(wx0 + i * bw, wy + 20 - hgt, bw - 1, hgt), cl)


class MetricBar(QWidget):
    def __init__(self, label: str, color: str = PRI, parent=None):
        super().__init__(parent)
        self._label = label
        self._color = color
        self._value = 0.0
        self._text = "--"
        self.setFixedHeight(38)
        self.setMinimumWidth(80)

    def set_value(self, pct: float, text: str) -> None:
        self._value = max(0.0, min(100.0, pct))
        self._text = text
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()

        p.setBrush(QBrush(qcol(PANEL2)))
        p.setPen(QPen(qcol(BORDER_A), 1))
        p.drawRoundedRect(QRectF(1, 1, W - 2, H - 2), 4, 4)

        bar_h = 4
        bar_y = H - bar_h - 5
        bar_w = W - 12
        bar_x = 6
        fill_w = int(bar_w * self._value / 100)

        p.setBrush(QBrush(qcol(BAR_BG)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QRectF(bar_x, bar_y, bar_w, bar_h), 2, 2)

        if self._value > 85:
            bar_col = qcol(RED)
        elif self._value > 65:
            bar_col = qcol(ACC)
        else:
            bar_col = qcol(self._color)

        if fill_w > 0:
            p.setBrush(QBrush(bar_col))
            p.drawRoundedRect(QRectF(bar_x, bar_y, fill_w, bar_h), 2, 2)

        p.setFont(QFont("Courier New", 7, QFont.Weight.Bold))
        p.setPen(QPen(qcol(TEXT_DIM), 1))
        p.drawText(QRectF(8, 5, 50, 14), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self._label)

        p.setFont(QFont("Courier New", 9, QFont.Weight.Bold))
        p.setPen(QPen(bar_col if self._text != "--" else qcol(TEXT_DIM), 1))
        p.drawText(QRectF(0, 4, W - 6, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, self._text)


class LogWidget(QTextEdit):
    """Typewriter-effect activity log — ported from MARK XL. Lines are
    queued and revealed a character at a time; a "You:"/"JARVIS:"/
    "SYS:"/"ERR:" prefix picks the line's color."""

    _sig = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Courier New", 9))
        self.setStyleSheet(f"""
            QTextEdit {{
                background: {PANEL}; color: {TEXT}; border: 1px solid {BORDER};
                border-radius: 4px; padding: 6px; selection-background-color: {PRI_GHO};
            }}
            QScrollBar:vertical {{ background: {BG}; width: 8px; border: none; }}
            QScrollBar::handle:vertical {{ background: {BORDER_B}; border-radius: 4px; min-height: 20px; }}
        """)
        self._queue: list[str] = []
        self._typing = False
        self._text = ""
        self._pos = 0
        self._tag = "sys"
        self._tmr = QTimer(self)
        self._tmr.timeout.connect(self._step)
        self._sig.connect(self._enqueue)

    def append_log(self, text: str) -> None:
        self._sig.emit(text)

    def _enqueue(self, text: str) -> None:
        self._queue.append(text)
        if not self._typing:
            self._next()

    def _next(self) -> None:
        if not self._queue:
            self._typing = False
            return
        self._typing = True
        self._text = self._queue.pop(0)
        self._pos = 0
        tl = self._text.lower()
        if tl.startswith("you:"):
            self._tag = "you"
        elif tl.startswith("jarvis:"):
            self._tag = "ai"
        elif tl.startswith("identity:"):
            self._tag = "identity"
        elif "err" in tl:
            self._tag = "err"
        else:
            self._tag = "sys"
        self._tmr.start(6)

    def _step(self) -> None:
        if self._pos < len(self._text):
            ch = self._text[self._pos]
            cur = self.textCursor()
            fmt = cur.charFormat()
            col = {
                "you": qcol(WHITE),
                "ai": qcol(PRI),
                "err": qcol(RED),
                "identity": qcol(GREEN),
                "sys": qcol(ACC2),
            }.get(self._tag, qcol(TEXT))
            fmt.setForeground(QBrush(col))
            cur.movePosition(cur.MoveOperation.End)
            cur.insertText(ch, fmt)
            self.setTextCursor(cur)
            self.ensureCursorVisible()
            self._pos += 1
        else:
            self._tmr.stop()
            cur = self.textCursor()
            cur.movePosition(cur.MoveOperation.End)
            cur.insertText("\n")
            self.setTextCursor(cur)
            self.ensureCursorVisible()
            QTimer.singleShot(20, self._next)


class _SessionSignals(QObject):
    """Cross-thread bridge: VoiceSession's callbacks (invoked from the
    background pipeline thread) emit these; MainWindow's slots (running
    on the main/UI thread) receive them via Qt's queued connection."""

    state_changed = pyqtSignal(str)
    transcript = pyqtSignal(str, str)
    error = pyqtSignal(str)
    identity = pyqtSignal(str, float)  # name ("" if unrecognized), confidence


class MainWindow(QMainWindow):
    def __init__(
        self,
        recorder: MicRecorder,
        player: AudioPlayer,
        transcriber: WhisperTranscriber,
        reasoner: VoiceReasoner,
        tts: ElevenLabsTTS,
        store: VoiceStore,
        face_identifier: FaceIdentifier,
    ):
        super().__init__()
        self._recorder = recorder
        self._player = player
        self._transcriber = transcriber
        self._reasoner = reasoner
        self._tts = tts
        self._store = store
        self._face_identifier = face_identifier
        self._process_start = time.time()
        self._identified_name: str | None = None

        self._signals = _SessionSignals()
        self._signals.state_changed.connect(self._on_state_changed)
        self._signals.transcript.connect(self._on_transcript)
        self._signals.error.connect(self._on_error)
        self._signals.identity.connect(self._on_identity)

        self._session = self._new_session()

        self.setWindowTitle("JARVIS Voice")
        self.resize(1100, 720)
        self.setStyleSheet(f"QMainWindow, QWidget {{ background: {BG}; color: {TEXT}; }}")
        self._build_ui()
        self._start_clock()
        self._start_metrics_timer()
        self._start_identity_scan()

    def _new_session(self) -> VoiceSession:
        session = VoiceSession(
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
        # Identity is a property of who's physically present, not of one
        # conversation's history — carry it forward into a fresh session
        # (e.g. after "New Session") rather than forgetting it.
        session.set_user_name(self._identified_name)
        return session

    # -- layout --------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        body.addWidget(self._build_left_panel(), stretch=0)

        self._hud = HudCanvas()
        body.addWidget(self._hud, stretch=5)

        body.addWidget(self._build_right_panel(), stretch=0)

        root.addLayout(body, stretch=1)
        root.addWidget(self._build_footer())

        self.setCentralWidget(central)

    def _build_header(self) -> QFrame:
        bar = QFrame()
        bar.setFixedHeight(54)
        bar.setStyleSheet(f"background: {DARK}; border-bottom: 1px solid {BORDER_B};")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 0, 16, 0)

        title = QLabel("J.A.R.V.I.S")
        title.setFont(QFont("Courier New", 13, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {PRI}; letter-spacing: 3px;")
        layout.addWidget(title)

        self._status_pill = QLabel("● ONLINE")
        self._status_pill.setFont(QFont("Courier New", 8, QFont.Weight.Bold))
        self._status_pill.setStyleSheet(
            f"color: {GREEN}; border: 1px solid {GREEN}; border-radius: 9px; padding: 2px 10px;"
        )
        layout.addWidget(self._status_pill)

        self._identity_label = QLabel("◇ SCANNING...")
        self._identity_label.setFont(QFont("Courier New", 8, QFont.Weight.Bold))
        self._identity_label.setStyleSheet(f"color: {TEXT_MED}; padding: 2px 10px;")
        layout.addWidget(self._identity_label)

        layout.addItem(QSpacerItem(20, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

        date_clock = QVBoxLayout()
        date_clock.setSpacing(0)
        self._clock_label = QLabel("--:--:--")
        self._clock_label.setFont(QFont("Courier New", 14, QFont.Weight.Bold))
        self._clock_label.setStyleSheet(f"color: {PRI};")
        self._clock_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._date_label = QLabel("")
        self._date_label.setFont(QFont("Courier New", 8))
        self._date_label.setStyleSheet(f"color: {TEXT_DIM};")
        self._date_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        date_clock.addWidget(self._clock_label)
        date_clock.addWidget(self._date_label)
        layout.addLayout(date_clock)

        return bar

    def _build_left_panel(self) -> QWidget:
        w = QWidget()
        w.setFixedWidth(160)
        w.setStyleSheet(f"background: {DARK}; border-right: 1px solid {BORDER};")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 10, 8, 10)
        lay.setSpacing(6)

        hdr = QLabel("◈ SYS MONITOR")
        hdr.setFont(QFont("Courier New", 7, QFont.Weight.Bold))
        hdr.setStyleSheet(f"color: {PRI}; border-bottom: 1px solid {BORDER}; padding-bottom: 4px;")
        lay.addWidget(hdr)
        lay.addSpacing(2)

        self._bar_cpu = MetricBar("CPU", PRI)
        self._bar_mem = MetricBar("MEM", ACC2)
        lay.addWidget(self._bar_cpu)
        lay.addWidget(self._bar_mem)
        lay.addSpacing(4)

        info_panel = QWidget()
        info_panel.setStyleSheet(f"background: {PANEL2}; border: 1px solid {BORDER}; border-radius: 4px;")
        ip_lay = QVBoxLayout(info_panel)
        ip_lay.setContentsMargins(6, 5, 6, 5)
        ip_lay.setSpacing(3)

        self._uptime_lbl = QLabel("UP  00:00:00")
        self._uptime_lbl.setFont(QFont("Courier New", 8, QFont.Weight.Bold))
        self._uptime_lbl.setStyleSheet(f"color: {GREEN}; border: none;")
        ip_lay.addWidget(self._uptime_lbl)

        lay.addWidget(info_panel)
        lay.addStretch()

        return w

    def _build_right_panel(self) -> QWidget:
        w = QWidget()
        w.setFixedWidth(340)
        w.setStyleSheet(f"background: {DARK}; border-left: 1px solid {BORDER};")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        def _sec(txt: str) -> QLabel:
            label = QLabel(f"▸ {txt}")
            label.setFont(QFont("Courier New", 7, QFont.Weight.Bold))
            label.setStyleSheet(f"color: {TEXT_MED};")
            return label

        lay.addWidget(_sec("ACTIVITY LOG"))
        self._log = LogWidget()
        lay.addWidget(self._log, stretch=1)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color: {BORDER}; margin: 2px 0;")
        lay.addWidget(sep)

        lay.addWidget(_sec("ENROLL FACE"))
        lay.addLayout(self._build_enroll_row())

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet(f"color: {BORDER}; margin: 2px 0;")
        lay.addWidget(sep2)

        lay.addWidget(_sec("VOICE CONTROL"))

        self._talk_btn = QPushButton("🎙  HOLD TO TALK")
        self._talk_btn.setFixedHeight(30)
        self._talk_btn.setFont(QFont("Courier New", 8, QFont.Weight.Bold))
        self._talk_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._talk_btn.pressed.connect(self._on_talk_pressed)
        self._talk_btn.released.connect(self._on_talk_released)
        self._style_talk_btn(active=True)
        lay.addWidget(self._talk_btn)

        row = QHBoxLayout()
        self._stop_btn = QPushButton("■  STOP")
        self._stop_btn.setFixedHeight(26)
        self._stop_btn.setFont(QFont("Courier New", 7))
        self._stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stop_btn.setStyleSheet(f"""
            QPushButton {{ background: transparent; color: {RED}; border: 1px solid {BORDER}; border-radius: 3px; }}
            QPushButton:hover {{ border: 1px solid {RED}; }}
        """)
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        row.addWidget(self._stop_btn)

        self._new_session_btn = QPushButton("⟳  NEW SESSION")
        self._new_session_btn.setFixedHeight(26)
        self._new_session_btn.setFont(QFont("Courier New", 7))
        self._new_session_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._new_session_btn.setStyleSheet(f"""
            QPushButton {{ background: transparent; color: {TEXT_MED}; border: 1px solid {BORDER}; border-radius: 3px; }}
            QPushButton:hover {{ color: {PRI}; border: 1px solid {BORDER_B}; }}
        """)
        self._new_session_btn.clicked.connect(self._on_new_session_clicked)
        row.addWidget(self._new_session_btn)

        lay.addLayout(row)

        return w

    def _build_enroll_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(5)

        self._enroll_input = QLineEdit()
        self._enroll_input.setPlaceholderText("Name to enroll...")
        self._enroll_input.setFont(QFont("Courier New", 9))
        self._enroll_input.setFixedHeight(28)
        self._enroll_input.setStyleSheet(f"""
            QLineEdit {{ background: {DARK}; color: {WHITE}; border: 1px solid {BORDER}; border-radius: 3px; padding: 3px 7px; }}
            QLineEdit:focus {{ border: 1px solid {PRI}; }}
        """)
        self._enroll_input.returnPressed.connect(self._on_enroll_clicked)
        row.addWidget(self._enroll_input)

        enroll_btn = QPushButton("+")
        enroll_btn.setFixedSize(28, 28)
        enroll_btn.setFont(QFont("Courier New", 11, QFont.Weight.Bold))
        enroll_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        enroll_btn.setStyleSheet(f"""
            QPushButton {{ background: {PANEL}; color: {PRI}; border: 1px solid {PRI_DIM}; border-radius: 3px; }}
            QPushButton:hover {{ background: {PRI_GHO}; border: 1px solid {PRI}; }}
        """)
        enroll_btn.clicked.connect(self._on_enroll_clicked)
        row.addWidget(enroll_btn)

        return row

    def _build_footer(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(22)
        w.setStyleSheet(f"background: {DARK}; border-top: 1px solid {BORDER};")
        lay = QHBoxLayout(w)
        lay.setContentsMargins(14, 0, 14, 0)

        def _fl(txt: str, color: str = TEXT_MED) -> QLabel:
            label = QLabel(txt)
            label.setFont(QFont("Courier New", 7))
            label.setStyleSheet(f"color: {color};")
            return label

        lay.addWidget(_fl("Hold mic to speak  ·  local face ID enrollment"))
        lay.addStretch()
        lay.addWidget(_fl("JARVIS Voice", PRI_DIM))
        return w

    def _style_talk_btn(self, *, active: bool) -> None:
        if active:
            self._talk_btn.setStyleSheet(f"""
                QPushButton {{ background: #00140a; color: {GREEN}; border: 1px solid {GREEN}; border-radius: 3px; }}
                QPushButton:hover {{ background: #001f10; }}
                QPushButton:disabled {{ background: {PANEL}; color: {TEXT_DIM}; border: 1px solid {BORDER}; }}
            """)
        self._talk_btn.setEnabled(active)

    # -- clock / metrics / identity --------------------------------------

    def _start_clock(self) -> None:
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._tick_clock)
        self._clock_timer.start(1000)
        self._tick_clock()

    def _tick_clock(self) -> None:
        now = datetime.now()
        self._clock_label.setText(now.strftime("%H:%M:%S"))
        self._date_label.setText(now.strftime("%A, %B %d"))

    def _start_metrics_timer(self) -> None:
        self._metric_timer = QTimer(self)
        self._metric_timer.timeout.connect(self._update_metrics)
        self._metric_timer.start(2000)
        self._update_metrics()

    def _update_metrics(self) -> None:
        snap = _metrics.snapshot()
        self._bar_cpu.set_value(snap["cpu"], f"{snap['cpu']:.0f}%")
        self._bar_mem.set_value(snap["mem"], f"{snap['mem']:.0f}%")

        elapsed = time.time() - self._process_start
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        self._uptime_lbl.setText(f"UP  {h:02d}:{m:02d}:{s:02d}")

    def _start_identity_scan(self) -> None:
        def scan():
            name, confidence = self._face_identifier.identify(timeout=5.0)
            self._signals.identity.emit(name or "", confidence)

        threading.Thread(target=scan, daemon=True).start()

    def _on_identity(self, name: str, confidence: float) -> None:
        self._identified_name = name or None
        self._session.set_user_name(self._identified_name)
        if name:
            self._identity_label.setText(f"◆ {name.upper()}")
            self._identity_label.setStyleSheet(f"color: {GREEN}; padding: 2px 10px;")
            self._log.append_log(f"IDENTITY: recognized '{name}' (confidence {confidence:.2f})")
        else:
            self._identity_label.setText("◇ UNRECOGNIZED")
            self._identity_label.setStyleSheet(f"color: {TEXT_MED}; padding: 2px 10px;")
            self._log.append_log("IDENTITY: no enrolled face recognized")

    def _on_enroll_clicked(self) -> None:
        name = self._enroll_input.text().strip()
        if not name:
            return
        self._enroll_input.clear()
        self._log.append_log(f"SYS: Enrolling face for '{name}' — look at the camera...")

        def do_enroll():
            ok = self._face_identifier.register(name)
            if ok:
                self._log.append_log(f"SYS: Enrolled '{name}' successfully.")
            else:
                self._log.append_log(f"ERR: Could not enroll '{name}' — no camera or no face detected.")

        threading.Thread(target=do_enroll, daemon=True).start()

    # -- button handlers --------------------------------------------------

    def _on_talk_pressed(self) -> None:
        if self._session.state != VoiceState.IDLE:
            return
        self._style_talk_btn(active=False)
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
        self._on_state_changed(VoiceState.IDLE.value)
        self._log.append_log("SYS: New session started.")

    # -- signal slots (main thread) ----------------------------------------

    def _on_state_changed(self, state_value: str) -> None:
        self._hud.set_state(state_value)
        if state_value == VoiceState.IDLE.value:
            self._style_talk_btn(active=True)

    def _on_transcript(self, role: str, text: str) -> None:
        speaker = "You" if role == "user" else "JARVIS"
        self._log.append_log(f"{speaker}: {text}")

    def _on_error(self, message: str) -> None:
        self._log.append_log(f"ERR: {message}")


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
        beam_size=config.whisper_beam_size,
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
    face_identifier = FaceIdentifier(config.faces_dir)

    app = QApplication(sys.argv)
    window = MainWindow(recorder, player, transcriber, reasoner, tts, store, face_identifier)
    window.show()
    exit_code = app.exec()
    store.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(run())
