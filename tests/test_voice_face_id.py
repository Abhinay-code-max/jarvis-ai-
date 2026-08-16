"""
tests/test_voice_face_id.py
==============================
jarvis_core.voice.face_id.FaceIdentifier — the pure-Python parts only
(loading/matching/listing/deleting saved encodings), against real
temp-directory .npy files rather than mocks, same convention as the
rest of this project's tests.

identify() and register() are deliberately NOT exercised here — both
open a real webcam via cv2.VideoCapture(0), which this suite has no
business doing automatically (no camera available in CI, and even
locally this would be a slow, non-deterministic hardware dependency).
This mirrors audio_io.py's own precedent: no test file touches its
real microphone/speaker classes either, for the same reason. What's
tested here is everything that doesn't require a camera: reading
already-enrolled encodings back off disk, the nearest-match logic, and
list/delete — the parts a missing/failing camera should never affect.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from jarvis_core.voice.face_id import FaceIdentifier  # noqa: E402


class FaceIdentifierTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.faces_dir = Path(self._tmpdir.name) / "faces"

    def _save_encoding(self, name: str, vector: np.ndarray) -> None:
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(self.faces_dir / f"{name}.npy"), vector)

    def test_faces_dir_is_created_if_missing(self):
        self.assertFalse(self.faces_dir.exists())
        FaceIdentifier(self.faces_dir)
        self.assertTrue(self.faces_dir.is_dir())

    def test_loads_previously_enrolled_encodings_on_construction(self):
        self._save_encoding("alice", np.array([1.0, 0.0, 0.0]))
        self._save_encoding("bob", np.array([0.0, 1.0, 0.0]))

        identifier = FaceIdentifier(self.faces_dir)

        self.assertEqual(sorted(identifier.list_users()), ["alice", "bob"])

    def test_match_returns_closest_enrolled_encoding(self):
        self._save_encoding("alice", np.array([1.0, 0.0, 0.0]))
        self._save_encoding("bob", np.array([0.0, 1.0, 0.0]))
        identifier = FaceIdentifier(self.faces_dir)

        name, confidence = identifier._match(np.array([0.95, 0.05, 0.0]))

        self.assertEqual(name, "alice")
        self.assertGreater(confidence, 0.0)

    def test_match_against_no_enrolled_faces_returns_no_name(self):
        # _match() is never actually called with zero enrolled faces via
        # the public API (identify() short-circuits first) — this just
        # pins _match()'s own direct behavior: no candidate to compare
        # against means no name, regardless of the sentinel confidence.
        identifier = FaceIdentifier(self.faces_dir)

        name, _ = identifier._match(np.array([1.0, 0.0, 0.0]))

        self.assertIsNone(name)

    def test_identify_with_no_enrolled_faces_short_circuits_without_camera(self):
        # No encodings means identify() must return early, before ever
        # trying to import cv2/face_recognition or open a camera.
        identifier = FaceIdentifier(self.faces_dir)

        name, confidence = identifier.identify(timeout=0.01)

        self.assertIsNone(name)
        self.assertEqual(confidence, 0.0)

    def test_delete_user_removes_encoding_and_file(self):
        self._save_encoding("alice", np.array([1.0, 0.0, 0.0]))
        identifier = FaceIdentifier(self.faces_dir)

        deleted = identifier.delete_user("alice")

        self.assertTrue(deleted)
        self.assertEqual(identifier.list_users(), [])
        self.assertFalse((self.faces_dir / "alice.npy").exists())

    def test_delete_unknown_user_returns_false(self):
        identifier = FaceIdentifier(self.faces_dir)

        self.assertFalse(identifier.delete_user("nobody"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
