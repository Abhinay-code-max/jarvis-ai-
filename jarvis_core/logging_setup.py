"""
jarvis_core/logging_setup.py
==============================
Console + rotating local file logging for jarvis_core — same shape as
eyv_poller/logging_setup.py (see that module's docstring for the full
reasoning), just under its own "jarvis_core" logger namespace and log
directory so the two processes' logs never interleave in one file even
when run on the same machine.

Same hard rule, enforced by convention rather than code: no call site in
this package logs a token, an Authorization header, a Claude API key, or
raw ticket content that hasn't already been through decision_schema
validation.
"""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)-30s %(message)s"


def init_logging(log_dir: Path, level: str = "INFO") -> None:
    """Call once, at process startup. Idempotent — safe to call more than
    once (e.g. in tests) without stacking duplicate handlers."""
    root = logging.getLogger("jarvis_core")
    if root.handlers:
        return

    root.setLevel(level)
    root.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "jarvis_core.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        root.warning(
            "Could not set up file logging at %s — continuing with console logging only",
            log_dir, exc_info=True,
        )
