"""
tests/test_claude_reasoner.py
================================
ClaudeReasoner exercised against a hand-written fake Anthropic client —
never a real API call. This is a standing rule for this codebase (it's
been hit before, elsewhere, as unmocked LLM calls spawning real
processes in tests): every test that touches Claude reasoning mocks the
API client, full stop.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic  # noqa: E402
import httpx  # noqa: E402

from jarvis_core.claude_reasoner import ClaudeReasoner, ClaudeReasonerError  # noqa: E402
from jarvis_core.decision_schema import DecisionValidationError  # noqa: E402


class _FakeToolUseBlock:
    def __init__(self, name, input_):
        self.type = "tool_use"
        self.name = name
        self.input = input_


class _FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeMessages:
    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if self._exc is not None:
            raise self._exc
        return self._response


class _FakeClient:
    def __init__(self, response=None, exc=None):
        self.messages = _FakeMessages(response, exc)


def _write_doc(tmpdir) -> Path:
    path = Path(tmpdir) / "arch.md"
    path.write_text("You are JARVIS. Decide.", encoding="utf-8")
    return path


class ClaudeReasonerTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.doc_path = _write_doc(self._tmpdir.name)

    def _reasoner(self, client):
        return ClaudeReasoner(self.doc_path, "claude-opus-5", 1024, client=client)

    def test_valid_tool_call_parses_into_decision(self):
        response = SimpleNamespace(
            stop_reason="tool_use",
            content=[_FakeToolUseBlock("record_decision", {
                "decision": "RESOLVED", "reason": "no action needed", "confidence": 0.9,
            })],
        )
        client = _FakeClient(response)
        decision = self._reasoner(client).reason({"queue_item_id": "1", "payload": {}})

        self.assertEqual(decision.decision, "RESOLVED")
        self.assertEqual(decision.confidence, 0.9)
        # The system prompt is the architecture doc's exact contents, and
        # the tool call is forced (not left to auto tool_choice).
        self.assertEqual(client.messages.last_kwargs["system"], "You are JARVIS. Decide.")
        self.assertEqual(client.messages.last_kwargs["tool_choice"], {"type": "tool", "name": "record_decision"})

    def test_ignores_stray_text_block_and_finds_the_tool_call(self):
        response = SimpleNamespace(
            stop_reason="tool_use",
            content=[
                _FakeTextBlock("thinking out loud"),
                _FakeToolUseBlock("record_decision", {
                    "decision": "MARKETING_ACTION", "reason": "tell users", "confidence": 0.8,
                    "action": {"type": "marketing", "instructions": "post an update"},
                }),
            ],
        )
        decision = self._reasoner(_FakeClient(response)).reason({"queue_item_id": "1", "payload": {}})
        self.assertEqual(decision.decision, "MARKETING_ACTION")
        self.assertEqual(decision.action_payload["instructions"], "post an update")

    def test_missing_tool_call_raises_reasoner_error(self):
        response = SimpleNamespace(stop_reason="end_turn", content=[_FakeTextBlock("no tool call")])
        with self.assertRaises(ClaudeReasonerError):
            self._reasoner(_FakeClient(response)).reason({"queue_item_id": "1", "payload": {}})

    def test_refusal_stop_reason_raises_reasoner_error(self):
        response = SimpleNamespace(stop_reason="refusal", content=[])
        with self.assertRaises(ClaudeReasonerError):
            self._reasoner(_FakeClient(response)).reason({"queue_item_id": "1", "payload": {}})

    def test_malformed_tool_input_raises_validation_error_not_reasoner_error(self):
        response = SimpleNamespace(
            stop_reason="tool_use",
            content=[_FakeToolUseBlock("record_decision", {
                "decision": "NOT_A_REAL_DECISION", "reason": "x", "confidence": 0.5,
            })],
        )
        with self.assertRaises(DecisionValidationError):
            self._reasoner(_FakeClient(response)).reason({"queue_item_id": "1", "payload": {}})

    def test_out_of_range_confidence_raises_validation_error(self):
        response = SimpleNamespace(
            stop_reason="tool_use",
            content=[_FakeToolUseBlock("record_decision", {
                "decision": "RESOLVED", "reason": "x", "confidence": 5,
            })],
        )
        with self.assertRaises(DecisionValidationError):
            self._reasoner(_FakeClient(response)).reason({"queue_item_id": "1", "payload": {}})

    def test_api_connection_error_is_wrapped_as_reasoner_error(self):
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        exc = anthropic.APIConnectionError(request=request)
        with self.assertRaises(ClaudeReasonerError):
            self._reasoner(_FakeClient(exc=exc)).reason({"queue_item_id": "1", "payload": {}})

    def test_rate_limit_error_is_wrapped_as_reasoner_error(self):
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        response = httpx.Response(429, request=request)
        exc = anthropic.RateLimitError("rate limited", response=response, body=None)
        with self.assertRaises(ClaudeReasonerError):
            self._reasoner(_FakeClient(exc=exc)).reason({"queue_item_id": "1", "payload": {}})


if __name__ == "__main__":
    unittest.main(verbosity=2)
