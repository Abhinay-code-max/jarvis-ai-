"""
eyv_poller/logging_setup.py
=============================
Local operational logging for the poller: console + a rotating file
under Config.log_dir. Deliberately just stdlib logging — this is a
single small process, not JARVIS-XL's multi-threaded desktop app, so
there's no non-blocking-queue requirement to satisfy the way
core/logging_setup.py has (see that module's docstring for why it needs
one and this doesn't).

Hard rule enforced by convention, not code: no call site in this package
ever logs a token, an Authorization header, or any other secret value.
Log statements pass counts, status codes, and URLs — never headers or
credentials.
"""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)-24s %(message)s"


def init_logging(log_dir: Path, level: str = "INFO") -> None:
    """Call once, at process startup. Idempotent — safe to call more than
    once (e.g. in tests) without stacking duplicate handlers."""
    root = logging.getLogger("eyv_poller")
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
            log_dir / "eyv_poller.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        # A logging setup failure (e.g. unwritable directory) must never
        # prevent the service from starting — it just runs console-only.
        root.warning("Could not set up file logging at %s — continuing with console logging only", log_dir, exc_info=True)
