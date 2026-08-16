"""
eyv_poller/schema.py
======================
Parsing for GET /jarvis/queue's response body — isolated in its own
module, deliberately separate from eyv_poller/client.py's HTTP
transport, because the exact response schema is NOT yet confirmed on
EYV's side (B.1's brief is explicit: /jarvis/queue doesn't exist there
yet). Nothing about the field names below should be treated as
authoritative — they're a best guess at a plausible shape (a JSON object
with an "items" list) chosen only so this service has something concrete
to poll against and test today.

When the real schema is confirmed, only parse_queue_response() (and
QueueItem/QueueSnapshot's fields, if the shape genuinely differs) should
need to change — eyv_poller/client.py, poller.py, and processor.py never
touch raw response bytes/JSON themselves, only QueueSnapshot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class MalformedResponseError(Exception):
    """Raised when the response body isn't valid JSON, or doesn't match
    even the loose shape parse_queue_response() expects. Callers treat
    this the same as any other recoverable poll failure — log it and
    keep polling — never a reason to crash the process."""


@dataclass(frozen=True)
class QueueItem:
    """One queue entry. Deliberately loose: `raw` always carries the
    complete, unmodified item dict so eyv_poller/processor.py has access
    to every field EYV actually sends, even fields this best-guess schema
    doesn't know about yet. `item_id` is best-effort — None if no
    recognizable id field is present, rather than raising."""
    item_id: str | None
    raw:     dict[str, Any]


@dataclass(frozen=True)
class QueueSnapshot:
    items: list[QueueItem] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.items)


_ID_KEYS = ("id", "item_id", "ticket_id", "escalation_id")


def _extract_item_id(item: dict[str, Any]) -> str | None:
    for key in _ID_KEYS:
        value = item.get(key)
        if value is not None:
            return str(value)
    return None


def parse_queue_response(payload: Any) -> QueueSnapshot:
    """Accepts either a bare JSON list of items, or an object carrying
    the list under a plausible key ("items"/"queue"/"results") — both
    are common REST shapes and there's no confirmed EYV convention yet
    to narrow this to one. Raises MalformedResponseError for anything
    else (not a list/dict, or a dict with no recognizable list field, or
    a list containing a non-dict entry) rather than guessing further."""
    if isinstance(payload, list):
        raw_items = payload
    elif isinstance(payload, dict):
        raw_items = None
        for key in ("items", "queue", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                raw_items = value
                break
        if raw_items is None:
            raise MalformedResponseError(
                f"response object has no recognizable list field (looked for "
                f"items/queue/results); top-level keys were {sorted(payload.keys())}"
            )
    else:
        raise MalformedResponseError(f"expected a JSON list or object, got {type(payload).__name__}")

    items = []
    for i, entry in enumerate(raw_items):
        if not isinstance(entry, dict):
            raise MalformedResponseError(f"queue item at index {i} is not a JSON object (got {type(entry).__name__})")
        items.append(QueueItem(item_id=_extract_item_id(entry), raw=entry))

    return QueueSnapshot(items=items)
