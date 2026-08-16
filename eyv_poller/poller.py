"""
eyv_poller/poller.py
======================
The polling loop itself: wait, poll, process, repeat. Deliberately
small — a single-purpose service like this needs a sleep loop, not a job
scheduler.

Resilience boundary: eyv_poller/client.py already retries transient
HTTP/network failures internally, but this loop wraps each tick in its
own try/except anyway. That is not redundant — it's what guarantees that
*any* unexpected failure in a single tick (an exhausted retry budget
surfacing as EYVUnavailable, a MalformedResponseError from an
unanticipated response shape, or a genuine bug in processor.py) logs and
is survived rather than ever taking the whole process down. One bad poll
must never be the last poll.

Shutdown is cooperative: stop() sets a threading.Event that both ends
the sleep early and exits the loop after the current tick, so `run()`
never needs an OS signal handler wired directly into it — see
eyv_poller/main.py for where SIGINT/SIGTERM get connected to stop().
"""
from __future__ import annotations

import logging
import threading
from typing import Callable

from eyv_poller.client import EYVQueueClient, EYVRequestError
from eyv_poller.processor import process_queue_snapshot
from eyv_poller.schema import MalformedResponseError, QueueSnapshot

_log = logging.getLogger("eyv_poller.poller")


class PollerLoop:
    def __init__(
        self,
        client: EYVQueueClient,
        interval_sec: float,
        process: Callable[[QueueSnapshot], None] = process_queue_snapshot,
        stop_event: threading.Event | None = None,
    ):
        self._client       = client
        self._interval_sec = interval_sec
        self._process       = process
        self._stop_event    = stop_event or threading.Event()

    def _poll_once(self) -> None:
        _log.info("Polling EYV queue")
        try:
            snapshot = self._client.fetch_queue()
        except EYVRequestError as e:
            _log.error("EYV queue poll failed: %s", e)
            return
        except MalformedResponseError as e:
            _log.error("EYV queue poll returned an unexpected response shape: %s", e)
            return
        except Exception:
            # Catch-all so a bug anywhere below this line (including
            # inside self._process) can never terminate the loop — see
            # module docstring. Anything caught only here, rather than by
            # one of the specific branches above, is worth a full
            # traceback since it's by definition unanticipated.
            _log.exception("Unexpected error during EYV queue poll")
            return

        _log.info("Poll succeeded — %d item(s) on the queue", snapshot.count)
        try:
            self._process(snapshot)
        except Exception:
            _log.exception("Queue item processing failed for this poll — will retry next cycle")

    def run(self) -> None:
        _log.info("EYV poller starting (interval=%.1fs, queue=%s)", self._interval_sec, self._client.url)
        while not self._stop_event.is_set():
            self._poll_once()
            self._stop_event.wait(self._interval_sec)
        _log.info("EYV poller stopped")

    def stop(self) -> None:
        self._stop_event.set()
