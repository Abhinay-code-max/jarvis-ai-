"""
eyv_poller/client.py
======================
Thin HTTP client for EYV's GET /jarvis/queue. Owns exactly two things:
authentication (attaching the bearer token) and transient-failure retry
— response *parsing* lives in eyv_poller/schema.py, kept separate so a
schema change never touches this module.

Retry policy is deliberately simple (a fixed number of attempts, linear
exponential backoff, no jitter library, no third-party retry framework):
connection errors, timeouts, and 5xx responses are retried; 4xx
responses are not — a bad/expired token or a wrong URL won't fix itself
by trying again, and retrying it just delays surfacing the real problem.
"""
from __future__ import annotations

import logging
import time

import requests

from eyv_poller.schema import MalformedResponseError, QueueSnapshot, parse_queue_response

_log = logging.getLogger("eyv_poller.client")


class EYVRequestError(Exception):
    """Base class for everything this client raises. Never carries the
    Authorization header or token value in its message."""


class EYVAuthError(EYVRequestError):
    """401/403 from EYV — the stored token is missing, wrong, or revoked.
    Never retried here: a bad token doesn't become a good one by asking
    again."""


class EYVUnavailable(EYVRequestError):
    """Connection error, timeout, or a 5xx response that persisted
    through every retry attempt."""


class EYVQueueClient:
    def __init__(
        self,
        base_url: str,
        queue_path: str,
        get_token,
        request_timeout_sec: float,
        max_retries: int,
        retry_backoff_base_sec: float,
        session: requests.Session | None = None,
    ):
        self._url = base_url.rstrip("/") + queue_path
        self._get_token = get_token
        self._timeout = request_timeout_sec
        self._max_retries = max(0, max_retries)
        self._backoff_base = retry_backoff_base_sec
        self._session = session or requests.Session()

    @property
    def url(self) -> str:
        return self._url

    def _headers(self) -> dict[str, str]:
        token = self._get_token()
        if not token:
            raise EYVAuthError(
                "no EYV queue token stored — run 'python -m eyv_poller.auth generate' (or 'set') first"
            )
        return {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    def fetch_queue(self) -> QueueSnapshot:
        """Fetches and parses the current EYV queue. Raises EYVAuthError
        (no retry), EYVUnavailable (retries exhausted), or
        MalformedResponseError (parseable HTTP response, unparseable
        body — not retried, since a schema mismatch won't change on
        retry). A single successful attempt returns normally at any
        point in the loop."""
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                delay = self._backoff_base * (2 ** (attempt - 1))
                _log.info("Retrying EYV queue poll (attempt %d/%d) in %.1fs", attempt + 1, self._max_retries + 1, delay)
                time.sleep(delay)

            try:
                response = self._session.get(self._url, headers=self._headers(), timeout=self._timeout)
            except EYVAuthError:
                raise
            except requests.RequestException as e:
                last_error = e
                _log.warning("EYV queue poll network failure on attempt %d/%d: %s", attempt + 1, self._max_retries + 1, e)
                continue

            if response.status_code in (401, 403):
                raise EYVAuthError(f"EYV rejected the queue token (HTTP {response.status_code})")

            if response.status_code >= 500:
                last_error = EYVUnavailable(f"EYV returned HTTP {response.status_code}")
                _log.warning(
                    "EYV queue poll got a server error on attempt %d/%d: HTTP %d",
                    attempt + 1, self._max_retries + 1, response.status_code,
                )
                continue

            if response.status_code >= 400:
                # Any other 4xx (bad request, not found, etc.) is a
                # configuration problem — e.g. a wrong queue_path — not a
                # transient condition, so this doesn't retry either.
                raise EYVUnavailable(f"EYV returned HTTP {response.status_code}: {response.text[:200]}")

            try:
                payload = response.json()
            except ValueError as e:
                raise MalformedResponseError(f"EYV response body was not valid JSON: {e}") from e

            return parse_queue_response(payload)

        raise EYVUnavailable(f"EYV queue poll failed after {self._max_retries + 1} attempts: {last_error}")
