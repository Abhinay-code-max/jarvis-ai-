"""
tests/test_voice_tts_edge.py
===============================
jarvis_core.voice.tts_edge.EdgeTTS — the request/error-handling layer
exercised against a hand-written fake Communicate object (never a real
call to Microsoft's service, same "mock the external API" convention
as tests/test_voice_reasoner.py). The actual MP3->PCM decode step goes
through the real `pydub`, which needs a real `ffmpeg` binary on PATH —
that one path is gated behind a shutil.which("ffmpeg") check and skips
cleanly where ffmpeg isn't installed (it isn't, on the machine this was
written on), same pattern as tests/test_code_interface.py's `claude`
CLI availability check.
"""
from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis_core.voice.tts_edge import EdgeTTS, EdgeTTSError  # noqa: E402


class _FakeCommunicate:
    def __init__(self, exc: Exception | None = None, write_bytes: bytes | None = b"not-really-mp3-audio"):
        self._exc = exc
        self._write_bytes = write_bytes
        self.saved_to: str | None = None

    def save_sync(self, path: str) -> None:
        if self._exc is not None:
            raise self._exc
        self.saved_to = path
        if self._write_bytes is not None:
            Path(path).write_bytes(self._write_bytes)
        # else: simulate EdgeTTS silently producing no file at all.


class EdgeTTSTest(unittest.TestCase):
    def test_communicate_failure_raises_edgetts_error(self):
        factory = lambda text, voice: _FakeCommunicate(exc=ConnectionError("service unreachable"))
        tts = EdgeTTS(voice="en-US-GuyNeural", communicate_factory=factory)

        with self.assertRaises(EdgeTTSError):
            tts.synthesize("hello")

    def test_no_output_file_raises_edgetts_error(self):
        factory = lambda text, voice: _FakeCommunicate(write_bytes=None)
        tts = EdgeTTS(voice="en-US-GuyNeural", communicate_factory=factory)

        with self.assertRaises(EdgeTTSError):
            tts.synthesize("hello")

    def test_empty_output_file_raises_edgetts_error(self):
        factory = lambda text, voice: _FakeCommunicate(write_bytes=b"")
        tts = EdgeTTS(voice="en-US-GuyNeural", communicate_factory=factory)

        with self.assertRaises(EdgeTTSError):
            tts.synthesize("hello")

    def test_invalid_audio_data_raises_edgetts_error_not_a_crash(self):
        # Garbage bytes will fail to decode as MP3 regardless of whether
        # ffmpeg is actually installed on this machine — either way it's
        # a decode failure, wrapped as EdgeTTSError rather than an
        # uncaught exception from pydub/ffmpeg.
        factory = lambda text, voice: _FakeCommunicate(write_bytes=b"definitely not audio data")
        tts = EdgeTTS(voice="en-US-GuyNeural", communicate_factory=factory)

        with self.assertRaises(EdgeTTSError):
            tts.synthesize("hello")

    def test_communicate_is_called_with_text_and_voice(self):
        calls = []

        def factory(text, voice):
            calls.append((text, voice))
            return _FakeCommunicate(write_bytes=None)  # fails fast after recording the call

        tts = EdgeTTS(voice="en-US-GuyNeural", communicate_factory=factory)
        with self.assertRaises(EdgeTTSError):
            tts.synthesize("hello there")

        self.assertEqual(calls, [("hello there", "en-US-GuyNeural")])

    @unittest.skipUnless(shutil.which("ffmpeg"), "real ffmpeg binary not installed/on PATH on this machine")
    def test_real_decode_of_a_real_small_mp3_succeeds(self):
        # Synthesizes a real, tiny silent MP3 via pydub itself (not
        # edge_tts) to exercise the real decode path end to end without
        # a network call — this only runs where ffmpeg is actually usable.
        from pydub import AudioSegment
        from pydub.generators import Sine

        with __import__("tempfile").TemporaryDirectory() as tmpdir:
            mp3_path = Path(tmpdir) / "tone.mp3"
            Sine(440).to_audio_segment(duration=300).export(str(mp3_path), format="mp3")
            mp3_bytes = mp3_path.read_bytes()

        factory = lambda text, voice: _FakeCommunicate(write_bytes=mp3_bytes)
        tts = EdgeTTS(voice="en-US-GuyNeural", sample_rate=16000, communicate_factory=factory)

        pcm_bytes, sample_rate = tts.synthesize("hello")

        self.assertEqual(sample_rate, 16000)
        self.assertGreater(len(pcm_bytes), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
