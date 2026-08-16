"""
eyv_poller/config.py
=====================
All tunables for the EYV queue poller, sourced from environment
variables (optionally backfilled from a local ".env" file — see
_load_dotenv() below). Real process environment always wins over
".env" so this stays container/systemd-friendly.

POLL_INTERVAL_SEC is the one setting this module refuses to default:
B.1's brief is explicit that "a minute or two" is a plausible guess for
ticket-escalation polling, not a confirmed decision, and that this
service must not silently bake in a guess. So EYV_POLL_INTERVAL_SEC has
no fallback — Config.load() raises ConfigError until it's set
explicitly, forcing whoever runs this service to make (or ask for) that
call rather than inherit an arbitrary constant. See README.md's
"Open decision" section.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Same ~/.jarvis/ convention JARVIS-XL uses for the GitHub PAT and Hermes
# tokens — a per-user directory outside any repo, so nothing here ever
# needs a .gitignore entry to stay out of source control.
JARVIS_DIR   = Path.home() / ".jarvis"
DEFAULT_TOKEN_PATH = JARVIS_DIR / "eyv_poller" / "queue_token.txt"
DEFAULT_LOG_DIR    = JARVIS_DIR / "eyv_poller" / "logs"

DEFAULT_QUEUE_PATH             = "/jarvis/queue"
DEFAULT_REQUEST_TIMEOUT_SEC    = 10.0
DEFAULT_MAX_RETRIES            = 3
DEFAULT_RETRY_BACKOFF_BASE_SEC = 1.5
DEFAULT_LOG_LEVEL              = "INFO"


class ConfigError(Exception):
    """Raised for missing/invalid required configuration. Always caught
    at the top of main.py and reported as a clear startup error — never
    meant to surface as a bare traceback to an operator."""


def _load_dotenv(path: Path) -> None:
    """Minimal ".env" loader (KEY=VALUE per line, '#' comments, blank
    lines skipped) — deliberately not a dependency on python-dotenv for
    something this small. Only fills in variables not already present in
    the real environment, so a real deployment's env always wins over a
    leftover local .env file."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise ConfigError(f"{name}={raw!r} is not a valid number") from None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{name}={raw!r} is not a valid integer") from None


@dataclass(frozen=True)
class Config:
    base_url:            str
    queue_path:           str
    token_path:           Path
    poll_interval_sec:    float
    request_timeout_sec:  float
    max_retries:          int
    retry_backoff_base_sec: float
    log_dir:              Path
    log_level:            str

    @property
    def queue_url(self) -> str:
        return self.base_url.rstrip("/") + self.queue_path

    @classmethod
    def load(cls, dotenv_path: Path | None = None) -> "Config":
        _load_dotenv(dotenv_path or Path.cwd() / ".env")

        base_url = os.environ.get("EYV_BASE_URL", "").strip()
        if not base_url:
            raise ConfigError(
                "EYV_BASE_URL is not set — point it at EYV's Railway deployment "
                "(e.g. https://eyv-website-2.up.railway.app)."
            )

        poll_interval_raw = os.environ.get("EYV_POLL_INTERVAL_SEC", "").strip()
        if not poll_interval_raw:
            raise ConfigError(
                "EYV_POLL_INTERVAL_SEC is not set. The polling interval is an "
                "open decision (see README.md) — a value must be chosen "
                "explicitly rather than inherited from a hard-coded default. "
                "Set it in the environment or .env once an interval is confirmed "
                "(a minute or two has been suggested as a starting point for "
                "manual local testing only)."
            )
        try:
            poll_interval_sec = float(poll_interval_raw)
        except ValueError:
            raise ConfigError(f"EYV_POLL_INTERVAL_SEC={poll_interval_raw!r} is not a valid number") from None
        if poll_interval_sec <= 0:
            raise ConfigError("EYV_POLL_INTERVAL_SEC must be a positive number of seconds")

        token_path_raw = os.environ.get("EYV_TOKEN_PATH", "").strip()
        token_path = Path(token_path_raw) if token_path_raw else DEFAULT_TOKEN_PATH

        log_dir_raw = os.environ.get("EYV_LOG_DIR", "").strip()
        log_dir = Path(log_dir_raw) if log_dir_raw else DEFAULT_LOG_DIR

        return cls(
            base_url=base_url,
            queue_path=os.environ.get("EYV_QUEUE_PATH", DEFAULT_QUEUE_PATH),
            token_path=token_path,
            poll_interval_sec=poll_interval_sec,
            request_timeout_sec=_env_float("EYV_REQUEST_TIMEOUT_SEC", DEFAULT_REQUEST_TIMEOUT_SEC),
            max_retries=_env_int("EYV_MAX_RETRIES", DEFAULT_MAX_RETRIES),
            retry_backoff_base_sec=_env_float("EYV_RETRY_BACKOFF_BASE_SEC", DEFAULT_RETRY_BACKOFF_BASE_SEC),
            log_dir=log_dir,
            log_level=os.environ.get("EYV_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper(),
        )
