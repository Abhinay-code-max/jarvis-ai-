"""
jarvis_core/main.py
=====================
Entry point for the standalone B.2 JARVIS reasoning process:
`python -m jarvis_core` (or `python -m jarvis_core.main`).

Reuses B.1's queue polling wholesale — eyv_poller.config.Config,
eyv_poller.auth, eyv_poller.client.EYVQueueClient, and
eyv_poller.poller.PollerLoop — rather than re-implementing polling. B.2
starts at JarvisEngine.process_queue_snapshot, wired in here as the
poller's `process` callback; everything upstream of that (auth, retry,
the sleep/poll loop, per-tick failure containment) is exactly B.1's
existing, already-tested implementation. See README.md's B.2 section
for the full list of environment variables this process needs.
"""
from __future__ import annotations

import logging
import signal
import sys
import threading

from eyv_poller import auth
from eyv_poller.client import EYVQueueClient
from eyv_poller.config import Config, ConfigError
from eyv_poller.poller import PollerLoop

from jarvis_core.approval_interface import ApprovalInterface
from jarvis_core.claude_reasoner import ClaudeReasoner
from jarvis_core.code_interface import CodeInterface
from jarvis_core.config import JarvisConfig, JarvisConfigError
from jarvis_core.db import DecisionStore, DecisionStoreError
from jarvis_core.engine import JarvisEngine
from jarvis_core.logging_setup import init_logging
from jarvis_core.marketing_client import MarketingDecisionClient

_log = logging.getLogger("jarvis_core.main")


def run() -> int:
    try:
        queue_config = Config.load()
        jarvis_config = JarvisConfig.load()
    except (ConfigError, JarvisConfigError) as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    init_logging(jarvis_config.log_dir, jarvis_config.log_level)
    _log.info(
        "Starting JARVIS reasoning core (B.2) — model=%s, confidence_threshold=%.2f",
        jarvis_config.claude_model, jarvis_config.confidence_threshold,
    )

    if auth.get_token(queue_config.token_path) is None:
        _log.error(
            "No EYV queue token stored at %s — run 'python -m eyv_poller.auth generate' "
            "(or 'set') before starting jarvis_core. B.2 reuses the same token B.1 uses "
            "for /jarvis/queue, since /jarvis/decisions is expected to sit under the same "
            "auth gate.",
            queue_config.token_path,
        )
        return 1

    get_token = lambda: auth.get_token(queue_config.token_path)  # noqa: E731

    queue_client = EYVQueueClient(
        base_url=queue_config.base_url,
        queue_path=queue_config.queue_path,
        get_token=get_token,
        request_timeout_sec=queue_config.request_timeout_sec,
        max_retries=queue_config.max_retries,
        retry_backoff_base_sec=queue_config.retry_backoff_base_sec,
    )

    try:
        store = DecisionStore(jarvis_config.db_path)
    except DecisionStoreError:
        _log.exception("Failed to initialize the decision database at %s", jarvis_config.db_path)
        return 1

    try:
        reasoner = ClaudeReasoner(
            architecture_doc_path=jarvis_config.architecture_doc_path,
            model=jarvis_config.claude_model,
            max_tokens=jarvis_config.claude_max_tokens,
        )
    except Exception:
        _log.exception("Failed to initialize the Claude reasoner")
        store.close()
        return 1

    code_interface = CodeInterface()
    marketing_client = MarketingDecisionClient(
        base_url=queue_config.base_url,
        decisions_path=jarvis_config.decisions_path,
        get_token=get_token,
        request_timeout_sec=jarvis_config.request_timeout_sec,
        max_retries=jarvis_config.max_retries,
        retry_backoff_base_sec=jarvis_config.retry_backoff_base_sec,
    )
    approval_interface = ApprovalInterface(store)

    engine = JarvisEngine(
        store=store,
        reasoner=reasoner,
        code_interface=code_interface,
        marketing_client=marketing_client,
        approval_interface=approval_interface,
        confidence_threshold=jarvis_config.confidence_threshold,
    )
    engine.recover_orphaned_decisions()

    stop_event = threading.Event()
    loop = PollerLoop(
        queue_client,
        queue_config.poll_interval_sec,
        process=engine.process_queue_snapshot,
        stop_event=stop_event,
    )

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
    finally:
        store.close()

    return 0


if __name__ == "__main__":
    sys.exit(run())
