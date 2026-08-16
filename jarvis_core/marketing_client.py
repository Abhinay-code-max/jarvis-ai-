"""
jarvis_core/marketing_client.py
=================================
B.2.2's MARKETING_ACTION route: POST /jarvis/decisions, instructing Bob
what marketing action is required. Mirrors eyv_poller/client.py's
conventions exactly (same bearer-token header, same
retry-on-network-error/5xx / no-retry-on-4xx policy) rather than
inventing a new HTTP client shape — see that module's docstring for the
full reasoning behind each choice.

ASSUMPTION: /jarvis/decisions does not exist on EYV's backend yet as of
this writing — only GET /jarvis/queue is implemented (see
eyv-website-main/backend/internal_jarvis_api.py). This client is built
against the contract this task's own B.2.2 spec describes (a JSON POST
carrying what/why/which-item/content/constraints) and assumes the real
endpoint will be added under the same /jarvis/* prefix, guarded by the
same require_jarvis_queue_token bearer-token dependency eyv_poller
already authenticates with — so this reuses eyv_poller.auth.get_token()
rather than inventing a second credential (see main.py for the wiring).
If the real endpoint's contract differs once it exists, only this
module needs to change.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

import requests

_log = logging.getLogger("jarvis_core.marketing_client")


class MarketingClientError(Exception):
    """Base class for everything this client raises. Never carries the
    Authorization header or token value in its message."""


class MarketingClientAuthError(MarketingClientError):
    """401/403 from EYV — the stored token is missing, wrong, or
    revoked. Never retried: a bad token doesn't become a good one by
    asking again."""


class MarketingClientUnavailable(MarketingClientError):
    """Connection error, timeout, or a 5xx response that persisted
    through every retry attempt, or any other non-2xx response."""


class MarketingDecisionClient:
    def __init__(
        self,
        base_url: str,
        decisions_path: str,
        get_token: Callable[[], str | None],
        request_timeout_sec: float,
        max_retries: int,
        retry_backoff_base_sec: float,
        session: requests.Session | None = None,
    ):
        self._url = base_url.rstrip("/") + decisions_path
        self._get_token = get_token
        self._timeout = request_timeout_sec
        self._max_retries = max(0, max_retries)
        self._backoff_base = retry_backoff_base_sec
        self._session = session or requests.Session()

    @property
    def url(self) -> str:
        return self._url

    def post_decision(self, payload: dict[str, Any]) -> None:
        """POSTs `payload` to /jarvis/decisions. Raises
        MarketingClientAuthError (no retry) or MarketingClientUnavailable
        (retries exhausted, or a non-retryable non-2xx response) on any
        failure — never returns normally without a confirmed 2xx
        response, so callers can trust that a normal return means Bob
        was actually notified."""
        token = self._get_token()
        if not token:
            raise MarketingClientAuthError(
                "no EYV queue token stored — cannot authenticate to /jarvis/decisions"
            )
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                delay = self._backoff_base * (2 ** (attempt - 1))
                _log.info(
                    "Retrying POST /jarvis/decisions (attempt %d/%d) in %.1fs",
                    attempt + 1, self._max_retries + 1, delay,
                )
                time.sleep(delay)

            try:
                response = self._session.post(self._url, json=payload, headers=headers, timeout=self._timeout)
            except requests.RequestException as e:
                last_error = e
                _log.warning(
                    "POST /jarvis/decisions network failure on attempt %d/%d: %s",
                    attempt + 1, self._max_retries + 1, e,
                )
                continue

            if response.status_code in (401, 403):
                raise MarketingClientAuthError(f"EYV rejected the /jarvis/decisions token (HTTP {response.status_code})")

            if response.status_code >= 500:
                last_error = MarketingClientUnavailable(f"EYV returned HTTP {response.status_code}")
                _log.warning(
                    "POST /jarvis/decisions server error on attempt %d/%d: HTTP %d",
                    attempt + 1, self._max_retries + 1, response.status_code,
                )
                continue

            if response.status_code >= 400:
                raise MarketingClientUnavailable(
                    f"EYV returned HTTP {response.status_code}: {response.text[:200]}"
                )

            _log.info("POST /jarvis/decisions succeeded (HTTP %d)", response.status_code)
            return

        raise MarketingClientUnavailable(
            f"POST /jarvis/decisions failed after {self._max_retries + 1} attempts: {last_error}"
        )
