"""
tests/test_schema.py
=======================
eyv_poller/schema.py: parse_queue_response()'s handling of the plausible
shapes it accepts and the malformed ones it must reject cleanly, since
the real EYV /jarvis/queue schema isn't confirmed yet and this is the
one seam meant to absorb that uncertainty (see that module's docstring).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eyv_poller.schema import MalformedResponseError, parse_queue_response  # noqa: E402


class ParseQueueResponseTest(unittest.TestCase):
    def test_bare_list_of_items(self):
        snapshot = parse_queue_response([{"id": "1"}, {"id": "2"}])
        self.assertEqual(snapshot.count, 2)
        self.assertEqual([i.item_id for i in snapshot.items], ["1", "2"])

    def test_object_with_items_key(self):
        snapshot = parse_queue_response({"items": [{"id": "abc"}]})
        self.assertEqual(snapshot.count, 1)
        self.assertEqual(snapshot.items[0].item_id, "abc")

    def test_object_with_queue_key(self):
        snapshot = parse_queue_response({"queue": [{"ticket_id": "t-1"}]})
        self.assertEqual(snapshot.count, 1)
        self.assertEqual(snapshot.items[0].item_id, "t-1")

    def test_object_with_results_key(self):
        snapshot = parse_queue_response({"results": [{"escalation_id": 42}]})
        self.assertEqual(snapshot.count, 1)
        self.assertEqual(snapshot.items[0].item_id, "42")

    def test_empty_list_is_a_valid_empty_queue(self):
        snapshot = parse_queue_response([])
        self.assertEqual(snapshot.count, 0)

    def test_item_missing_recognizable_id_is_kept_with_none_id(self):
        snapshot = parse_queue_response([{"subject": "no id field here"}])
        self.assertEqual(snapshot.count, 1)
        self.assertIsNone(snapshot.items[0].item_id)
        self.assertEqual(snapshot.items[0].raw, {"subject": "no id field here"})

    def test_raw_item_is_preserved_verbatim(self):
        item = {"id": "1", "priority": "high", "nested": {"a": 1}}
        snapshot = parse_queue_response([item])
        self.assertEqual(snapshot.items[0].raw, item)

    def test_object_with_no_recognizable_list_field_raises(self):
        with self.assertRaises(MalformedResponseError):
            parse_queue_response({"status": "ok"})

    def test_top_level_scalar_raises(self):
        with self.assertRaises(MalformedResponseError):
            parse_queue_response("not a list or object")

    def test_top_level_none_raises(self):
        with self.assertRaises(MalformedResponseError):
            parse_queue_response(None)

    def test_non_dict_item_in_list_raises(self):
        with self.assertRaises(MalformedResponseError):
            parse_queue_response(["not", "a", "dict"])

    def test_non_dict_item_in_object_list_raises(self):
        with self.assertRaises(MalformedResponseError):
            parse_queue_response({"items": [{"id": "1"}, "bad"]})


if __name__ == "__main__":
    unittest.main(verbosity=2)
