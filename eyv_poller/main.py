"""
eyv_poller/main.py
=====================
Entry point: `python -m eyv_poller.main` (or `python -m eyv_poller` —
see __main__.py). Loads config, wires up logging/auth/client/poller, and
runs until SIGINT/SIGTERM or Ctrl+C.
"""
from __future__ import annotations

import logging
import signal
import sys
import threading

from eyv_poller import auth
from eyv_poller.client import EYVQueueClient
from eyv_poller.config import Config, ConfigError
from eyv_poller.logging_setup import init_logging
from eyv_poller.poller import PollerLoop

_log = logging.getLogger("eyv_poller.main")


def run() -> int:
    try:
        config = Config.load()
    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    init_logging(config.log_dir, config.log_level)
    _log.info("Starting EYV poller (base_url=%s, queue_path=%s)", config.base_url, config.queue_path)

    if auth.get_token(config.token_path) is None:
        _log.error(
            "No EYV queue token stored at %s — run 'python -m eyv_poller.auth generate' "
            "(or 'set') before starting the poller.",
            config.token_path,
        )
        return 1

    client = EYVQueueClient(
        base_url=config.base_url,
        queue_path=config.queue_path,
        get_token=lambda: auth.get_token(config.token_path),
        request_timeout_sec=config.request_timeout_sec,
        max_retries=config.max_retries,
        retry_backoff_base_sec=config.retry_backoff_base_sec,
    )

    stop_event = threading.Event()
    loop = PollerLoop(client, config.poll_interval_sec, stop_event=stop_event)

    def _handle_signal(signum, _frame):
        _log.info("Received signal %s — shutting down after the current poll", signum)
        loop.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    try:
        loop.run()
    except KeyboardInterrupt:
        _log.info("Interrupted — shutting down")
        loop.stop()

    return 0


if __name__ == "__main__":
    sys.exit(run())
