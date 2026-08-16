"""
jarvis_core/voice/tts.py
===========================
ElevenLabs text-to-speech client. Mirrors jarvis_core/marketing_client.py's
HTTP conventions exactly — same retry-on-network-error/5xx,
no-retry-on-4xx policy, same auth-vs-unavailable exception split —
because this is the same shape of problem (a JSON-in, binary-out POST
to a third-party API) with a different endpoint and credential.

Endpoint/params confirmed against ElevenLabs' current API reference:
`POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}`, auth
header `xi-api-key`, JSON body `{"text": ..., "model_id": ...}`, an
`output_format` query parameter. This client requests raw PCM
(`pcm_16000` by default) rather than MP3 specifically so playback
(audio_io.py) needs no audio-decoding dependency — the response body is
signed 16-bit little-endian mono samples at the requested rate, ready
to hand straight to sounddevice.
"""
from __future__ import annotations

import logging
import time
from typing import Callable

import requests

_log = logging.getLogger("jarvis_core.voice.tts")

_BASE_URL = "https://api.elevenlabs.io"


class ElevenLabsError(Exception):
    """Base class for everything this client raises. Never carries the
    API key in its message."""


class ElevenLabsAuthError(ElevenLabsError):
    """401/403 from ElevenLabs — the stored key is missing, wrong, or
    revoked. Never retried: a bad key doesn't become a good one by
    asking again."""


class ElevenLabsUnavailable(ElevenLabsError):
    """Connection error, timeout, or a 5xx response that persisted
    through every retry attempt, or any other non-2xx response."""


def _sample_rate_from_output_format(output_format: str) -> int:
    """`pcm_16000` -> 16000, etc. Raises ElevenLabsError for any
    non-PCM format — this client only ever requests PCM (see module
    docstring), so a non-`pcm_*` value here means misconfiguration, not
    something to silently guess a sample rate for."""
    if not output_format.startswith("pcm_"):
        raise ElevenLabsError(
            f"unsupported output_format {output_format!r} — this client only supports raw "
            "PCM formats ('pcm_16000', 'pcm_24000', etc.) since it plays audio back without "
            "an MP3/Opus decoder"
        )
    try:
        return int(output_format.removeprefix("pcm_"))
    except ValueError:
        raise ElevenLabsError(f"could not parse a sample rate out of output_format {output_format!r}") from None


class ElevenLabsTTS:
    def __init__(
        self,
        voice_id: str,
        model_id: str,
        output_format: str,
        get_api_key: Callable[[], str | None],
        request_timeout_sec: float,
        max_retries: int,
        retry_backoff_base_sec: float,
        session: requests.Session | None = None,
    ):
        self._voice_id = voice_id
        self._model_id = model_id
        self._output_format = output_format
        self._sample_rate = _sample_rate_from_output_format(output_format)
        self._get_api_key = get_api_key
        self._timeout = request_timeout_sec
        self._max_retries = max(0, max_retries)
        self._backoff_base = retry_backoff_base_sec
        self._session = session or requests.Session()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def synthesize(self, text: str) -> tuple[bytes, int]:
        """Synthesizes `text` and returns (pcm_bytes, sample_rate).
        Raises ElevenLabsAuthError (no retry) or ElevenLabsUnavailable
        (retries exhausted, or a non-retryable non-2xx response) on any
        failure — never returns without a confirmed 2xx response with a
        non-empty body."""
        api_key = self._get_api_key()
        if not api_key:
            raise ElevenLabsAuthError("no ElevenLabs API key stored — cannot synthesize speech")

        url = f"{_BASE_URL}/v1/text-to-speech/{self._voice_id}"
        params = {"output_format": self._output_format}
        headers = {"xi-api-key": api_key, "Content-Type": "application/json", "Accept": "audio/*"}
        body = {"text": text, "model_id": self._model_id}

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                delay = self._backoff_base * (2 ** (attempt - 1))
                _log.info(
                    "Retrying ElevenLabs synthesis (attempt %d/%d) in %.1fs",
                    attempt + 1, self._max_retries + 1, delay,
                )
                time.sleep(delay)

            try:
                response = self._session.post(
                    url, params=params, json=body, headers=headers, timeout=self._timeout
                )
            except requests.RequestException as e:
                last_error = e
                _log.warning(
                    "ElevenLabs synthesis network failure on attempt %d/%d: %s",
                    attempt + 1, self._max_retries + 1, e,
                )
                continue

            if response.status_code in (401, 403):
                raise ElevenLabsAuthError(f"ElevenLabs rejected the API key (HTTP {response.status_code})")

            if response.status_code >= 500:
                last_error = ElevenLabsUnavailable(f"ElevenLabs returned HTTP {response.status_code}")
                _log.warning(
                    "ElevenLabs synthesis server error on attempt %d/%d: HTTP %d",
                    attempt + 1, self._max_retries + 1, response.status_code,
                )
                continue

            if response.status_code >= 400:
                raise ElevenLabsUnavailable(
                    f"ElevenLabs returned HTTP {response.status_code}: {response.text[:200]}"
                )

            if not response.content:
                raise ElevenLabsUnavailable("ElevenLabs returned a 2xx response with an empty body")

            _log.info("ElevenLabs synthesis succeeded (%d bytes)", len(response.content))
            return response.content, self._sample_rate

        raise ElevenLabsUnavailable(
            f"ElevenLabs synthesis failed after {self._max_retries + 1} attempts: {last_error}"
        )
