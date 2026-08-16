"""
jarvis_core/config.py
=======================
B.2-specific tunables, sourced from environment variables (optionally
backfilled from a local ".env" file — reuses eyv_poller.config's
_load_dotenv() so one ".env" file covers both processes rather than
inventing a second loader).

Queue polling itself (EYV_BASE_URL, EYV_POLL_INTERVAL_SEC, EYV_QUEUE_PATH,
EYV_TOKEN_PATH, retry/timeout settings, the bearer token) is deliberately
NOT duplicated here. That's already B.1's job — this module only adds
what B.2 needs on top: Claude API settings, the local decision database,
and the /jarvis/decisions endpoint. See jarvis_core/main.py for how the
two Config objects are used together.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from eyv_poller.config import _load_dotenv

# Same ~/.jarvis/ convention eyv_poller/config.py uses.
JARVIS_DIR = Path.home() / ".jarvis"
DEFAULT_DB_PATH = JARVIS_DIR / "jarvis_core" / "jarvis.db"
DEFAULT_LOG_DIR = JARVIS_DIR / "jarvis_core" / "logs"

# Repo-relative default so `python -m jarvis_core` works out of the box
# right after cloning, without requiring JARVIS_ARCHITECTURE_DOC_PATH to
# be set first. See docs/jarvis_architecture.md's own header for what
# this file is standing in for.
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARCHITECTURE_DOC_PATH = _REPO_ROOT / "docs" / "jarvis_architecture.md"

DEFAULT_CLAUDE_MODEL = "claude-opus-5"
DEFAULT_CLAUDE_MAX_TOKENS = 4096
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_DECISIONS_PATH = "/jarvis/decisions"
DEFAULT_REQUEST_TIMEOUT_SEC = 10.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_BASE_SEC = 1.5
DEFAULT_LOG_LEVEL = "INFO"


class JarvisConfigError(Exception):
    """Raised for missing/invalid required B.2 configuration. Always
    caught at the top of main.py — never meant to surface as a bare
    traceback to an operator."""


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise JarvisConfigError(f"{name}={raw!r} is not a valid number") from None


def resolve_db_path(dotenv_path: Path | None = None) -> Path:
    """Resolves JARVIS_DB_PATH on its own, without requiring the rest of
    B.2's config (ANTHROPIC_API_KEY, the architecture document). Used by
    approvals_cli.py and jarvis_core/dashboard.py, both of which only
    ever read/write the local decision database and have no reason to
    require Claude API credentials just to view or resolve approvals —
    JarvisConfig.load() is deliberately not reused for that reason."""
    _load_dotenv(dotenv_path or Path.cwd() / ".env")
    db_path_raw = os.environ.get("JARVIS_DB_PATH", "").strip()
    return Path(db_path_raw) if db_path_raw else DEFAULT_DB_PATH


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise JarvisConfigError(f"{name}={raw!r} is not a valid integer") from None


@dataclass(frozen=True)
class JarvisConfig:
    architecture_doc_path: Path
    claude_model: str
    claude_max_tokens: int
    confidence_threshold: float
    db_path: Path
    decisions_path: str
    request_timeout_sec: float
    max_retries: int
    retry_backoff_base_sec: float
    log_dir: Path
    log_level: str

    @classmethod
    def load(cls, dotenv_path: Path | None = None) -> "JarvisConfig":
        _load_dotenv(dotenv_path or Path.cwd() / ".env")

        if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            raise JarvisConfigError(
                "ANTHROPIC_API_KEY is not set. It's required to call the Claude API "
                "for reasoning (B.2.1). Set it in the environment or .env."
            )

        doc_path_raw = os.environ.get("JARVIS_ARCHITECTURE_DOC_PATH", "").strip()
        architecture_doc_path = Path(doc_path_raw) if doc_path_raw else DEFAULT_ARCHITECTURE_DOC_PATH
        if not architecture_doc_path.is_file():
            raise JarvisConfigError(
                f"JARVIS architecture document not found at {architecture_doc_path}. "
                "This file is used verbatim as Claude's system prompt (B.2.1) — set "
                "JARVIS_ARCHITECTURE_DOC_PATH if the real one lives somewhere else."
            )

        confidence_threshold = _env_float("JARVIS_CONFIDENCE_THRESHOLD", DEFAULT_CONFIDENCE_THRESHOLD)
        if not (0.0 <= confidence_threshold <= 1.0):
            raise JarvisConfigError(
                f"JARVIS_CONFIDENCE_THRESHOLD must be between 0 and 1, got {confidence_threshold}"
            )

        db_path_raw = os.environ.get("JARVIS_DB_PATH", "").strip()
        db_path = Path(db_path_raw) if db_path_raw else DEFAULT_DB_PATH

        log_dir_raw = os.environ.get("JARVIS_LOG_DIR", "").strip()
        log_dir = Path(log_dir_raw) if log_dir_raw else DEFAULT_LOG_DIR

        return cls(
            architecture_doc_path=architecture_doc_path,
            claude_model=os.environ.get("JARVIS_CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL),
            claude_max_tokens=_env_int("JARVIS_CLAUDE_MAX_TOKENS", DEFAULT_CLAUDE_MAX_TOKENS),
            confidence_threshold=confidence_threshold,
            db_path=db_path,
            decisions_path=os.environ.get("JARVIS_DECISIONS_PATH", DEFAULT_DECISIONS_PATH),
            request_timeout_sec=_env_float("JARVIS_REQUEST_TIMEOUT_SEC", DEFAULT_REQUEST_TIMEOUT_SEC),
            max_retries=_env_int("JARVIS_MAX_RETRIES", DEFAULT_MAX_RETRIES),
            retry_backoff_base_sec=_env_float("JARVIS_RETRY_BACKOFF_BASE_SEC", DEFAULT_RETRY_BACKOFF_BASE_SEC),
            log_dir=log_dir,
            log_level=os.environ.get("JARVIS_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper(),
        )
