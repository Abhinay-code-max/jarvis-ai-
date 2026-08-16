"""
tests/test_voice_tts.py
==========================
jarvis_core.voice.tts.ElevenLabsTTS: retry/backoff, auth-error
short-circuiting, and non-2xx handling. Runs a real ThreadingHTTPServer
on 127.0.0.1 rather than mocking `requests` — same "real component over
mock" convention as tests/test_client_resilience.py and
tests/test_marketing_client.py — this is the "at least one real local
smoke test path" the voice layer's spec explicitly asks for, applied to
the one voice-layer HTTP client (ElevenLabs); Claude and faster-whisper
are mocked elsewhere (test_voice_reasoner.py, test_voice_stt.py) since
neither has a "real local server" to stand in for it.
"""
from __future__ import annotations

import socket
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis_core.voice.tts import (  # noqa: E402
    ElevenLabsAuthError,
    ElevenLabsError,
    ElevenLabsTTS,
    ElevenLabsUnavailable,
    _sample_rate_from_output_format,
)

_FAKE_PCM = b"\x01\x02" * 500


class _TTSHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 (http.server's required method name)
        server = self.server
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        with server.lock:
            server.request_count += 1
            server.received_api_keys.append(self.headers.get("xi-api-key"))
            server.received_paths.append(self.path)
            server.received_bodies.append(body)
            if server.responses:
                status, resp_body, content_type = server.responses.pop(0)
            else:
                status, resp_body, content_type = server.default_response

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.end_headers()
        self.wfile.write(resp_body if isinstance(resp_body, bytes) else resp_body.encode("utf-8"))

    def log_message(self, format, *args):  # silence default stderr access logging
        pass


class _FakeElevenLabsServer:
    def __init__(self, default_response=(200, _FAKE_PCM, "application/octet-stream")):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _TTSHandler)
        self.httpd.lock = threading.Lock()
        self.httpd.request_count = 0
        self.httpd.received_api_keys = []
        self.httpd.received_paths = []
        self.httpd.received_bodies = []
        self.httpd.responses = []
        self.httpd.default_response = default_response
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_port}"

    @property
    def request_count(self) -> int:
        return self.httpd.request_count

    @property
    def received_api_keys(self) -> list:
        return self.httpd.received_api_keys

    @property
    def received_paths(self) -> list:
        return self.httpd.received_paths

    def queue_response(self, status: int, body, content_type: str = "application/octet-stream"):
        self.httpd.responses.append((status, body, content_type))

    def start(self):
        self._thread.start()

    def stop(self):
        self.httpd.shutdown()
        self._thread.join(timeout=5.0)
        self.httpd.server_close()


def _client_for(server, api_key="test-key", max_retries=3, backoff=0.01) -> ElevenLabsTTS:
    # ElevenLabsTTS.synthesize() reads the module-level `_BASE_URL` at
    # call time (not at construction time), so patching it in
    # setUp()/tearDown() (see _patch_base_url() below) is enough to
    # route every request at this fake local server instead of the
    # real ElevenLabs host — no base_url constructor argument needed.
    return ElevenLabsTTS(
        voice_id="voice-123",
        model_id="eleven_multilingual_v2",
        output_format="pcm_16000",
        get_api_key=lambda: api_key,
        request_timeout_sec=2.0,
        max_retries=max_retries,
        retry_backoff_base_sec=backoff,
    )


class ElevenLabsTTSTest(unittest.TestCase):
    def setUp(self):
        self.server = _FakeElevenLabsServer()
        self.server.start()
        self._patch_base_url()

    def tearDown(self):
        self.server.stop()
        self._unpatch_base_url()

    def _patch_base_url(self):
        import jarvis_core.voice.tts as tts_module

        self._original_base_url = tts_module._BASE_URL
        tts_module._BASE_URL = self.server.base_url

    def _unpatch_base_url(self):
        import jarvis_core.voice.tts as tts_module

        tts_module._BASE_URL = self._original_base_url

    def test_successful_synthesis_returns_pcm_and_sample_rate(self):
        self.server.queue_response(200, _FAKE_PCM)
        client = _client_for(self.server)

        pcm, sample_rate = client.synthesize("hello there")

        self.assertEqual(pcm, _FAKE_PCM)
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(self.server.request_count, 1)
        self.assertIn("/v1/text-to-speech/voice-123", self.server.received_paths[0])

    def test_api_key_header_is_sent(self):
        self.server.queue_response(200, _FAKE_PCM)
        client = _client_for(self.server, api_key="the-real-key")
        client.synthesize("hi")
        self.assertEqual(self.server.received_api_keys, ["the-real-key"])

    def test_transient_server_error_then_success_is_retried_to_completion(self):
        self.server.queue_response(500, "server error", "text/plain")
        self.server.queue_response(200, _FAKE_PCM)
        client = _client_for(self.server, max_retries=3)

        pcm, _rate = client.synthesize("hi")

        self.assertEqual(pcm, _FAKE_PCM)
        self.assertEqual(self.server.request_count, 2)

    def test_persistent_server_error_exhausts_retries_and_raises_unavailable(self):
        self.server.httpd.default_response = (503, "still down", "text/plain")
        client = _client_for(self.server, max_retries=2)
        with self.assertRaises(ElevenLabsUnavailable):
            client.synthesize("hi")
        self.assertEqual(self.server.request_count, 3)  # 1 initial + 2 retries

    def test_auth_error_is_not_retried(self):
        self.server.httpd.default_response = (401, "unauthorized", "text/plain")
        client = _client_for(self.server, max_retries=3)
        with self.assertRaises(ElevenLabsAuthError):
            client.synthesize("hi")
        self.assertEqual(self.server.request_count, 1)

    def test_forbidden_is_treated_as_auth_error(self):
        self.server.httpd.default_response = (403, "forbidden", "text/plain")
        client = _client_for(self.server, max_retries=3)
        with self.assertRaises(ElevenLabsAuthError):
            client.synthesize("hi")
        self.assertEqual(self.server.request_count, 1)

    def test_other_client_error_is_not_retried(self):
        self.server.httpd.default_response = (422, "bad request", "text/plain")
        client = _client_for(self.server, max_retries=3)
        with self.assertRaises(ElevenLabsUnavailable):
            client.synthesize("hi")
        self.assertEqual(self.server.request_count, 1)

    def test_empty_success_body_raises_unavailable(self):
        self.server.queue_response(200, b"")
        client = _client_for(self.server, max_retries=0)
        with self.assertRaises(ElevenLabsUnavailable):
            client.synthesize("hi")

    def test_missing_api_key_raises_auth_error_without_any_network_call(self):
        client = ElevenLabsTTS(
            voice_id="voice-123", model_id="eleven_multilingual_v2", output_format="pcm_16000",
            get_api_key=lambda: None, request_timeout_sec=2.0, max_retries=3, retry_backoff_base_sec=0.01,
        )
        with self.assertRaises(ElevenLabsAuthError):
            client.synthesize("hi")
        self.assertEqual(self.server.request_count, 0)

    def test_connection_failure_is_retried_then_raises_unavailable(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            unused_port = s.getsockname()[1]
        import jarvis_core.voice.tts as tts_module

        tts_module._BASE_URL = f"http://127.0.0.1:{unused_port}"
        client = ElevenLabsTTS(
            voice_id="voice-123", model_id="eleven_multilingual_v2", output_format="pcm_16000",
            get_api_key=lambda: "t", request_timeout_sec=1.0, max_retries=2, retry_backoff_base_sec=0.01,
        )
        with self.assertRaises(ElevenLabsUnavailable):
            client.synthesize("hi")


class SampleRateParsingTest(unittest.TestCase):
    def test_parses_pcm_formats(self):
        self.assertEqual(_sample_rate_from_output_format("pcm_16000"), 16000)
        self.assertEqual(_sample_rate_from_output_format("pcm_24000"), 24000)

    def test_non_pcm_format_raises(self):
        with self.assertRaises(ElevenLabsError):
            _sample_rate_from_output_format("mp3_44100_128")

    def test_malformed_pcm_format_raises(self):
        with self.assertRaises(ElevenLabsError):
            _sample_rate_from_output_format("pcm_fast")


if __name__ == "__main__":
    unittest.main(verbosity=2)
