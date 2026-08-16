"""
eyv_poller/processor.py
=========================
Extensibility point for what happens to items once they're pulled off
the EYV queue. B.1's scope is polling only — this module deliberately
does nothing but log what arrived. Future queue-handling work (routing
escalations, deduping, acking/removing handled items, etc.) plugs in
here without eyv_poller/poller.py or client.py needing to change.
"""
from __future__ import annotations

import logging

from eyv_poller.schema import QueueSnapshot

_log = logging.getLogger("eyv_poller.processor")


def process_queue_snapshot(snapshot: QueueSnapshot) -> None:
    """Called once per successful poll with whatever EYV returned (an
    empty snapshot on a quiet queue is a normal, frequent call, not an
    edge case). Logs item ids only — never full item bodies, since
    ticket content isn't known to be free of sensitive data and doesn't
    belong in a local log file."""
    if snapshot.count == 0:
        _log.debug("Queue empty — nothing to process")
        return

    ids = [item.item_id or "(no id)" for item in snapshot.items]
    _log.info("Processing %d queue item(s): %s", snapshot.count, ids)
    # No further handling in B.1 — see module docstring.
