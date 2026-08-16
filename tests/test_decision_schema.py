"""
tests/test_decision_schema.py
================================
jarvis_core.decision_schema.parse_decision(): the strict-validation
boundary between "whatever Claude's tool call returned" and a trusted
ReasoningDecision. Every malformed-input case here corresponds directly
to a way a real (or misbehaving) Claude response could arrive.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis_core.decision_schema import DecisionValidationError, parse_decision  # noqa: E402


def _valid(**overrides):
    base = {"decision": "RESOLVED", "reason": "nothing to do", "confidence": 0.9}
    base.update(overrides)
    return base


class ParseDecisionTest(unittest.TestCase):
    def test_valid_minimal_decision_parses(self):
        decision = parse_decision(_valid())
        self.assertEqual(decision.decision, "RESOLVED")
        self.assertEqual(decision.reason, "nothing to do")
        self.assertEqual(decision.confidence, 0.9)
        self.assertIsNone(decision.action_type)
        self.assertEqual(decision.action_payload, {})

    def test_valid_decision_with_action_parses(self):
        decision = parse_decision(_valid(
            decision="NEEDS_CODE",
            action={"type": "code", "instructions": "add a retry"},
        ))
        self.assertEqual(decision.action_type, "code")
        self.assertEqual(decision.action_payload["instructions"], "add a retry")

    def test_non_dict_input_raises(self):
        with self.assertRaises(DecisionValidationError):
            parse_decision("not a dict")

    def test_missing_decision_field_raises(self):
        raw = _valid()
        del raw["decision"]
        with self.assertRaises(DecisionValidationError):
            parse_decision(raw)

    def test_unrecognized_decision_value_raises(self):
        with self.assertRaises(DecisionValidationError):
            parse_decision(_valid(decision="SUPPORT_ACTION"))

    def test_empty_reason_raises(self):
        with self.assertRaises(DecisionValidationError):
            parse_decision(_valid(reason="   "))

    def test_missing_reason_raises(self):
        raw = _valid()
        del raw["reason"]
        with self.assertRaises(DecisionValidationError):
            parse_decision(raw)

    def test_non_numeric_confidence_raises(self):
        with self.assertRaises(DecisionValidationError):
            parse_decision(_valid(confidence="high"))

    def test_boolean_confidence_raises(self):
        # bool is a subclass of int in Python — must be rejected explicitly.
        with self.assertRaises(DecisionValidationError):
            parse_decision(_valid(confidence=True))

    def test_confidence_out_of_range_raises(self):
        with self.assertRaises(DecisionValidationError):
            parse_decision(_valid(confidence=1.5))
        with self.assertRaises(DecisionValidationError):
            parse_decision(_valid(confidence=-0.1))

    def test_confidence_boundary_values_are_accepted(self):
        parse_decision(_valid(confidence=0.0))
        parse_decision(_valid(confidence=1.0))

    def test_non_dict_action_raises(self):
        with self.assertRaises(DecisionValidationError):
            parse_decision(_valid(action="do something"))

    def test_non_string_action_type_raises(self):
        with self.assertRaises(DecisionValidationError):
            parse_decision(_valid(action={"type": 123}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
