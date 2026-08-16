"""
tests/test_marketing_client.py
================================
jarvis_core/marketing_client.py: retry/backoff, auth-error
short-circuiting, and non-2xx handling for POST /jarvis/decisions. Runs
a real ThreadingHTTPServer on 127.0.0.1 rather than mocking `requests`
— same "real component over mock" convention as
tests/test_client_resilience.py (eyv_poller's own HTTP client tests),
just POST instead of GET.
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

from jarvis_core.marketing_client import (  # noqa: E402
    MarketingClientAuthError,
    MarketingClientUnavailable,
    MarketingDecisionClient,
)


class _DecisionsHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 (http.server's required method name)
        server = self.server
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        with server.lock:
            server.request_count += 1
            server.received_authorization.append(self.headers.get("Authorization"))
            server.received_bodies.append(body)
            if server.responses:
                status, resp_body, content_type = server.responses.pop(0)
            else:
                status, resp_body, content_type = server.default_response

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.end_headers()
        self.wfile.write(resp_body.encode("utf-8") if isinstance(resp_body, str) else resp_body)

    def log_message(self, format, *args):  # silence default stderr access logging
        pass


class _FakeDecisionsServer:
    def __init__(self, default_response=(200, "{}", "application/json")):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _DecisionsHandler)
        self.httpd.lock = threading.Lock()
        self.httpd.request_count = 0
        self.httpd.received_authorization = []
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
    def received_authorization(self) -> list:
        return self.httpd.received_authorization

    @property
    def received_bodies(self) -> list:
        return self.httpd.received_bodies

    def queue_response(self, status: int, body: str, content_type: str = "application/json"):
        self.httpd.responses.append((status, body, content_type))

    def start(self):
        self._thread.start()

    def stop(self):
        self.httpd.shutdown()
        self._thread.join(timeout=5.0)
        self.httpd.server_close()


def _client_for(server: _FakeDecisionsServer, token="test-token", max_retries=3, backoff=0.01) -> MarketingDecisionClient:
    return MarketingDecisionClient(
        base_url=server.base_url,
        decisions_path="/jarvis/decisions",
        get_token=lambda: token,
        request_timeout_sec=2.0,
        max_retries=max_retries,
        retry_backoff_base_sec=backoff,
    )


class MarketingDecisionClientTest(unittest.TestCase):
    def setUp(self):
        self.server = _FakeDecisionsServer()
        self.server.start()

    def tearDown(self):
        self.server.stop()

    def test_successful_post_returns_normally(self):
        self.server.queue_response(200, "{}")
        client = _client_for(self.server)
        client.post_decision({"queue_item_id": "1", "action": {"type": "marketing"}})
        self.assertEqual(self.server.request_count, 1)
        self.assertEqual(json.loads(self.server.received_bodies[0])["queue_item_id"], "1")

    def test_bearer_token_is_sent(self):
        self.server.queue_response(200, "{}")
        client = _client_for(self.server, token="the-real-token")
        client.post_decision({"queue_item_id": "1"})
        self.assertEqual(self.server.received_authorization, ["Bearer the-real-token"])

    def test_transient_server_error_then_success_is_retried_to_completion(self):
        self.server.queue_response(500, "server error", "text/plain")
        self.server.queue_response(200, "{}")
        client = _client_for(self.server, max_retries=3)
        client.post_decision({"queue_item_id": "1"})  # must not raise
        self.assertEqual(self.server.request_count, 2)

    def test_persistent_server_error_exhausts_retries_and_raises_unavailable(self):
        self.server.httpd.default_response = (503, "still down", "text/plain")
        client = _client_for(self.server, max_retries=2)
        with self.assertRaises(MarketingClientUnavailable):
            client.post_decision({"queue_item_id": "1"})
        self.assertEqual(self.server.request_count, 3)  # 1 initial + 2 retries

    def test_auth_error_is_not_retried(self):
        self.server.httpd.default_response = (401, "unauthorized", "text/plain")
        client = _client_for(self.server, max_retries=3)
        with self.assertRaises(MarketingClientAuthError):
            client.post_decision({"queue_item_id": "1"})
        self.assertEqual(self.server.request_count, 1)

    def test_forbidden_is_treated_as_auth_error(self):
        self.server.httpd.default_response = (403, "forbidden", "text/plain")
        client = _client_for(self.server, max_retries=3)
        with self.assertRaises(MarketingClientAuthError):
            client.post_decision({"queue_item_id": "1"})
        self.assertEqual(self.server.request_count, 1)

    def test_other_client_error_is_not_retried(self):
        self.server.httpd.default_response = (404, "not found", "text/plain")
        client = _client_for(self.server, max_retries=3)
        with self.assertRaises(MarketingClientUnavailable):
            client.post_decision({"queue_item_id": "1"})
        self.assertEqual(self.server.request_count, 1)

    def test_missing_token_raises_auth_error_without_any_network_call(self):
        client = MarketingDecisionClient(
            base_url=self.server.base_url, decisions_path="/jarvis/decisions",
            get_token=lambda: None, request_timeout_sec=2.0,
            max_retries=3, retry_backoff_base_sec=0.01,
        )
        with self.assertRaises(MarketingClientAuthError):
            client.post_decision({"queue_item_id": "1"})
        self.assertEqual(self.server.request_count, 0)

    def test_connection_failure_is_retried_then_raises_unavailable(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            unused_port = s.getsockname()[1]
        client = MarketingDecisionClient(
            base_url=f"http://127.0.0.1:{unused_port}", decisions_path="/jarvis/decisions",
            get_token=lambda: "t", request_timeout_sec=1.0,
            max_retries=2, retry_backoff_base_sec=0.01,
        )
        with self.assertRaises(MarketingClientUnavailable):
            client.post_decision({"queue_item_id": "1"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
