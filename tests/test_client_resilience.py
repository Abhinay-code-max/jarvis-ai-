"""
tests/test_client_resilience.py
=================================
eyv_poller/client.py: retry/backoff, auth-error short-circuiting, and
malformed-response handling. Runs a real ThreadingHTTPServer on
127.0.0.1 (an ephemeral port) rather than mocking `requests` — the retry
logic genuinely drives real socket connections/timeouts, so a fake
transport would leave the actual thing under test (does a real
connection failure get retried the right number of times, with backoff,
before giving up) unverified. Same "real component over mock" bias
tests/test_hermes_client.py uses for JARVIS-XL's own HTTP client, just
with stdlib http.server here instead of uvicorn since this service has
no ASGI app of its own to serve.
"""
from __future__ import annotations

import json
import socket
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eyv_poller.client import EYVAuthError, EYVQueueClient, EYVUnavailable  # noqa: E402
from eyv_poller.schema import MalformedResponseError  # noqa: E402


class _QueueHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 (http.server's required method name)
        server = self.server
        with server.lock:
            server.request_count += 1
            server.received_authorization.append(self.headers.get("Authorization"))
            if server.responses:
                status, body, content_type = server.responses.pop(0)
            else:
                status, body, content_type = server.default_response

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.end_headers()
        self.wfile.write(body.encode("utf-8") if isinstance(body, str) else body)

    def log_message(self, format, *args):  # silence default stderr access logging
        pass


class _FakeEYVServer:
    """responses: a queue of (status, body, content_type) tuples served
    in order, one per request; once exhausted, default_response is
    served for every subsequent request."""

    def __init__(self, default_response=(200, "[]", "application/json")):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _QueueHandler)
        self.httpd.lock = threading.Lock()
        self.httpd.request_count = 0
        self.httpd.received_authorization = []
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
    def received_authorization(self) -> list:
        return self.httpd.received_authorization

    def queue_response(self, status: int, body: str, content_type: str = "application/json"):
        self.httpd.responses.append((status, body, content_type))

    def start(self):
        self._thread.start()

    def stop(self):
        self.httpd.shutdown()
        self._thread.join(timeout=5.0)
        self.httpd.server_close()


def _client_for(server: _FakeEYVServer, token="test-token", max_retries=3, backoff=0.01) -> EYVQueueClient:
    return EYVQueueClient(
        base_url=server.base_url,
        queue_path="/jarvis/queue",
        get_token=lambda: token,
        request_timeout_sec=2.0,
        max_retries=max_retries,
        retry_backoff_base_sec=backoff,
    )


class ClientResilienceTest(unittest.TestCase):
    def setUp(self):
        self.server = _FakeEYVServer()
        self.server.start()

    def tearDown(self):
        self.server.stop()

    def test_successful_poll_returns_snapshot(self):
        self.server.queue_response(200, json.dumps([{"id": "1"}, {"id": "2"}]))
        client = _client_for(self.server)
        snapshot = client.fetch_queue()
        self.assertEqual(snapshot.count, 2)
        self.assertEqual(self.server.request_count, 1)

    def test_bearer_token_is_sent(self):
        self.server.queue_response(200, "[]")
        client = _client_for(self.server, token="the-real-token")
        client.fetch_queue()
        self.assertEqual(self.server.received_authorization, ["Bearer the-real-token"])

    def test_transient_server_error_then_success_is_retried_to_completion(self):
        self.server.queue_response(500, "server error", "text/plain")
        self.server.queue_response(200, json.dumps([{"id": "1"}]))
        client = _client_for(self.server, max_retries=3)
        snapshot = client.fetch_queue()
        self.assertEqual(snapshot.count, 1)
        self.assertEqual(self.server.request_count, 2)

    def test_persistent_server_error_exhausts_retries_and_raises_unavailable(self):
        self.server.httpd.default_response = (503, "still down", "text/plain")
        client = _client_for(self.server, max_retries=2)
        with self.assertRaises(EYVUnavailable):
            client.fetch_queue()
        self.assertEqual(self.server.request_count, 3)  # 1 initial + 2 retries

    def test_auth_error_is_not_retried(self):
        self.server.httpd.default_response = (401, "unauthorized", "text/plain")
        client = _client_for(self.server, max_retries=3)
        with self.assertRaises(EYVAuthError):
            client.fetch_queue()
        self.assertEqual(self.server.request_count, 1)

    def test_forbidden_is_treated_as_auth_error(self):
        self.server.httpd.default_response = (403, "forbidden", "text/plain")
        client = _client_for(self.server, max_retries=3)
        with self.assertRaises(EYVAuthError):
            client.fetch_queue()
        self.assertEqual(self.server.request_count, 1)

    def test_malformed_json_body_is_not_retried(self):
        self.server.queue_response(200, "this is not json", "application/json")
        client = _client_for(self.server, max_retries=3)
        with self.assertRaises(MalformedResponseError):
            client.fetch_queue()
        self.assertEqual(self.server.request_count, 1)

    def test_unexpected_json_shape_is_not_retried(self):
        self.server.queue_response(200, json.dumps({"status": "ok"}))
        client = _client_for(self.server, max_retries=3)
        with self.assertRaises(MalformedResponseError):
            client.fetch_queue()
        self.assertEqual(self.server.request_count, 1)

    def test_other_client_error_is_not_retried(self):
        self.server.httpd.default_response = (404, "not found", "text/plain")
        client = _client_for(self.server, max_retries=3)
        with self.assertRaises(EYVUnavailable):
            client.fetch_queue()
        self.assertEqual(self.server.request_count, 1)

    def test_missing_token_raises_auth_error_without_any_network_call(self):
        client = EYVQueueClient(
            base_url=self.server.base_url, queue_path="/jarvis/queue",
            get_token=lambda: None, request_timeout_sec=2.0,
            max_retries=3, retry_backoff_base_sec=0.01,
        )
        with self.assertRaises(EYVAuthError):
            client.fetch_queue()
        self.assertEqual(self.server.request_count, 0)

    def test_connection_failure_is_retried_then_raises_unavailable(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            unused_port = s.getsockname()[1]
        client = EYVQueueClient(
            base_url=f"http://127.0.0.1:{unused_port}", queue_path="/jarvis/queue",
            get_token=lambda: "t", request_timeout_sec=1.0,
            max_retries=2, retry_backoff_base_sec=0.01,
        )
        with self.assertRaises(EYVUnavailable):
            client.fetch_queue()


if __name__ == "__main__":
    unittest.main(verbosity=2)
