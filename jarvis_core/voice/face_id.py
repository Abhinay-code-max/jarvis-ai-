"""
jarvis_core/voice/face_id.py
==============================
Face recognition for the voice UI — ported from the user's prior
JARVIS-XL/MARK XL project (recognition/face_id.py), adapted to this
codebase's conventions: storage moves from that project's local
`recognition/faces/` directory to jarvis_core's own `~/.jarvis/`
convention (see config.py's `faces_dir`, same 0600-local-file family as
the ElevenLabs key), and logging uses this project's namespace. The
actual algorithm (enrollment, cosine/L2 matching) is unchanged.

How it works:
  * On startup  -> scans the webcam for a few seconds, tries to match
    against enrolled faces (see identify()).
  * On register -> captures N frames, averages their encodings, saves
    the mean encoding to disk (see register()).
  * Matching    -> Euclidean distance against every enrolled encoding;
    closest one under THRESHOLD wins.

This is real, persistent biometric identification, not a live camera
preview: enrolling a face writes a permanent 128-dimension encoding to
`faces_dir` (an ordinary NumPy .npy file, not an image), which is
enough to re-recognize that person indefinitely until deleted via
delete_user(). Nothing here saves raw video/images — only the derived
numeric encoding.

Both `identify()` and `register()` degrade to a no-op-ish failure
(`(None, 0.0)` / `False`) rather than raising if `face_recognition`/
`cv2` aren't installed or no camera is available — this module is
optional, off-by-default functionality; its absence must never crash
the rest of the voice UI (same "never raise past this seam" pattern as
code_interface.py's CLI-availability handling).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

_log = logging.getLogger("jarvis_core.voice.face_id")


class FaceIdentifier:
    THRESHOLD = 0.55      # cosine distance; lower = stricter match
    CAPTURE_N = 15         # frames to capture during registration
    SCAN_FRAMES = 20       # max frames to grab during identification scan

    def __init__(self, faces_dir: Path):
        self._dir = Path(faces_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._encodings: dict[str, np.ndarray] = {}
        self._load_all()

    # -- load saved encodings --------------------------------------------

    def _load_all(self) -> None:
        for p in self._dir.glob("*.npy"):
            name = p.stem
            try:
                self._encodings[name] = np.load(str(p))
                _log.debug("Loaded face encoding for '%s'", name)
            except Exception as e:
                _log.warning("Could not load face encoding %s: %s", p, e)

    # -- identify ----------------------------------------------------------

    def identify(self, timeout: float = 5.0) -> tuple[Optional[str], float]:
        """Opens the webcam, grabs up to SCAN_FRAMES frames, and attempts
        to match any detected face against enrolled profiles. Returns
        (name, confidence) or (None, 0.0) if no match, no camera, or the
        optional dependencies aren't installed."""
        if not self._encodings:
            return None, 0.0

        try:
            import face_recognition
            import cv2
        except ImportError:
            _log.warning("face_recognition or cv2 not installed — skipping face scan.")
            return None, 0.0

        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            _log.warning("No camera available for face identification.")
            return None, 0.0

        best_name = None
        best_conf = 0.0
        deadline = time.time() + timeout
        frames_checked = 0

        try:
            while time.time() < deadline and frames_checked < self.SCAN_FRAMES:
                ret, frame = cam.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                frames_checked += 1

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                locs = face_recognition.face_locations(rgb, model="hog")
                if not locs:
                    continue

                encs = face_recognition.face_encodings(rgb, locs)
                for enc in encs:
                    name, conf = self._match(enc)
                    if conf > best_conf:
                        best_conf = conf
                        best_name = name

                if best_conf >= (1 - self.THRESHOLD):
                    break  # confident early exit
        finally:
            cam.release()

        if best_conf < (1 - self.THRESHOLD):
            return None, 0.0
        return best_name, best_conf

    # -- register ------------------------------------------------------------

    def register(self, name: str) -> bool:
        """Captures CAPTURE_N frames from the webcam and saves the mean
        face encoding to `faces_dir/<name>.npy`. Returns False (never
        raises) if the camera or optional dependencies aren't available,
        or if no face was ever detected across the capture attempt."""
        try:
            import face_recognition
            import cv2
        except ImportError:
            _log.warning("face_recognition or cv2 not installed.")
            return False

        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            return False

        collected = []
        _log.info("Capturing %d frames for '%s'...", self.CAPTURE_N, name)

        try:
            while len(collected) < self.CAPTURE_N:
                ret, frame = cam.read()
                if not ret:
                    time.sleep(0.05)
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                locs = face_recognition.face_locations(rgb, model="hog")
                if not locs:
                    continue
                encs = face_recognition.face_encodings(rgb, locs)
                if encs:
                    collected.append(encs[0])
        finally:
            cam.release()

        if not collected:
            return False

        mean_enc = np.mean(collected, axis=0)
        out_path = self._dir / f"{name}.npy"
        np.save(str(out_path), mean_enc)
        self._encodings[name] = mean_enc
        _log.info("Registered face for '%s' (%d frames).", name, len(collected))
        return True

    # -- matching helper -------------------------------------------------------

    def _match(self, encoding: np.ndarray) -> tuple[Optional[str], float]:
        """Returns (best_name, confidence) where confidence = 1 - L2_dist / 2."""
        best_name = None
        best_dist = 1.0
        for name, ref in self._encodings.items():
            dist = float(np.linalg.norm(encoding - ref))
            if dist < best_dist:
                best_dist = dist
                best_name = name
        confidence = max(0.0, 1.0 - best_dist / 2.0)
        return best_name, confidence

    # -- utility -----------------------------------------------------------

    def list_users(self) -> list[str]:
        return list(self._encodings.keys())

    def delete_user(self, name: str) -> bool:
        p = self._dir / f"{name}.npy"
        if p.exists():
            p.unlink()
            self._encodings.pop(name, None)
            return True
        return False
