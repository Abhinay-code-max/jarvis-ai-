"""
tests/test_poller.py
=======================
eyv_poller/poller.py: the loop's own resilience contract — a failing
tick (network error, malformed response, or a bug in the processing
callback) must never stop subsequent ticks, and stop() must end run()
promptly. Uses a small hand-written stub in place of EYVQueueClient
(a plain object with a fetch_queue() method, not a mocking-framework
double) since what's under test here is the loop's control flow, not
HTTP behavior — that's tests/test_client_resilience.py's job.
"""
from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eyv_poller.client import EYVAuthError, EYVUnavailable  # noqa: E402
from eyv_poller.poller import PollerLoop  # noqa: E402
from eyv_poller.schema import MalformedResponseError, QueueItem, QueueSnapshot  # noqa: E402


class _StubClient:
    """fetch_queue() returns/raises whatever the next entry in `plan`
    says, then holds on the last entry for any extra calls."""

    def __init__(self, plan):
        self._plan = list(plan)
        self.url = "http://stub/jarvis/queue"
        self.call_count = 0

    def fetch_queue(self):
        self.call_count += 1
        step = self._plan[min(self.call_count, len(self._plan)) - 1]
        if isinstance(step, Exception):
            raise step
        return step


def _run_until(loop: PollerLoop, condition, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


class PollerLoopTest(unittest.TestCase):
    def test_successful_poll_is_handed_to_processor(self):
        snapshot = QueueSnapshot(items=[QueueItem(item_id="1", raw={"id": "1"})])
        client = _StubClient([snapshot])
        processed = []
        stop_event = threading.Event()
        loop = PollerLoop(client, interval_sec=0.05, process=processed.append, stop_event=stop_event)

        t = threading.Thread(target=loop.run, daemon=True)
        t.start()
        self.assertTrue(_run_until(loop, lambda: len(processed) >= 1))
        loop.stop()
        t.join(timeout=2.0)
        self.assertEqual(processed[0].count, 1)

    def test_request_error_on_one_tick_does_not_stop_the_loop(self):
        good = QueueSnapshot(items=[])
        client = _StubClient([EYVUnavailable("simulated outage"), good, good])
        processed = []
        stop_event = threading.Event()
        loop = PollerLoop(client, interval_sec=0.05, process=processed.append, stop_event=stop_event)

        t = threading.Thread(target=loop.run, daemon=True)
        t.start()
        self.assertTrue(_run_until(loop, lambda: len(processed) >= 1))
        loop.stop()
        t.join(timeout=2.0)
        # The first tick failed (no process() call for it); the loop kept
        # going and a later tick succeeded.
        self.assertGreaterEqual(client.call_count, 2)

    def test_auth_error_does_not_stop_the_loop(self):
        client = _StubClient([EYVAuthError("bad token"), QueueSnapshot(items=[])])
        processed = []
        loop = PollerLoop(client, interval_sec=0.05, process=processed.append)

        t = threading.Thread(target=loop.run, daemon=True)
        t.start()
        self.assertTrue(_run_until(loop, lambda: len(processed) >= 1))
        loop.stop()
        t.join(timeout=2.0)

    def test_malformed_response_does_not_stop_the_loop(self):
        client = _StubClient([MalformedResponseError("bad shape"), QueueSnapshot(items=[])])
        processed = []
        loop = PollerLoop(client, interval_sec=0.05, process=processed.append)

        t = threading.Thread(target=loop.run, daemon=True)
        t.start()
        self.assertTrue(_run_until(loop, lambda: len(processed) >= 1))
        loop.stop()
        t.join(timeout=2.0)

    def test_processor_exception_does_not_stop_the_loop(self):
        good = QueueSnapshot(items=[])
        client = _StubClient([good, good, good])
        call_count = {"n": 0}

        def flaky_process(_snapshot):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("processor bug")

        loop = PollerLoop(client, interval_sec=0.05, process=flaky_process)
        t = threading.Thread(target=loop.run, daemon=True)
        t.start()
        self.assertTrue(_run_until(loop, lambda: call_count["n"] >= 2))
        loop.stop()
        t.join(timeout=2.0)

    def test_unexpected_exception_from_fetch_queue_does_not_stop_the_loop(self):
        client = _StubClient([RuntimeError("totally unanticipated bug"), QueueSnapshot(items=[])])
        processed = []
        loop = PollerLoop(client, interval_sec=0.05, process=processed.append)

        t = threading.Thread(target=loop.run, daemon=True)
        t.start()
        self.assertTrue(_run_until(loop, lambda: len(processed) >= 1))
        loop.stop()
        t.join(timeout=2.0)

    def test_stop_ends_run_promptly_even_mid_sleep(self):
        client = _StubClient([QueueSnapshot(items=[])])
        loop = PollerLoop(client, interval_sec=30.0, process=lambda s: None)

        t = threading.Thread(target=loop.run, daemon=True)
        t.start()
        self.assertTrue(_run_until(loop, lambda: client.call_count >= 1))
        started_stop = time.time()
        loop.stop()
        t.join(timeout=2.0)
        self.assertLess(time.time() - started_stop, 2.0)
        self.assertFalse(t.is_alive())


if __name__ == "__main__":
    unittest.main(verbosity=2)
