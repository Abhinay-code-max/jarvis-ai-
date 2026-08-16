"""
tests/test_voice_reasoner.py
===============================
jarvis_core.voice.voice_reasoner.VoiceReasoner, exercised against a
hand-written fake Anthropic client — never a real API call. Same
pattern as tests/test_claude_reasoner.py, applied to the voice layer's
separate, non-tool-forced conversational reasoning path. Also verifies
that path stays separate: no `tools`/`tool_choice` are ever sent.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic  # noqa: E402
import httpx  # noqa: E402

from jarvis_core.voice.voice_reasoner import SYSTEM_PROMPT, VoiceReasoner, VoiceReasonerError  # noqa: E402


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


def _reasoner(client, model="claude-opus-5", max_tokens=1024, effort="low") -> VoiceReasoner:
    return VoiceReasoner(model=model, max_tokens=max_tokens, effort=effort, client=client)


class VoiceReasonerTest(unittest.TestCase):
    def test_valid_response_returns_joined_text(self):
        response = SimpleNamespace(stop_reason="end_turn", content=[_FakeTextBlock("Hello! How can I help?")])
        client = _FakeClient(response)

        text = _reasoner(client).respond([{"role": "user", "content": "hi"}])

        self.assertEqual(text, "Hello! How can I help?")

    def test_no_tools_or_tool_choice_are_sent(self):
        response = SimpleNamespace(stop_reason="end_turn", content=[_FakeTextBlock("hi")])
        client = _FakeClient(response)

        _reasoner(client).respond([{"role": "user", "content": "hi"}])

        self.assertNotIn("tools", client.messages.last_kwargs)
        self.assertNotIn("tool_choice", client.messages.last_kwargs)

    def test_system_prompt_and_effort_are_passed(self):
        response = SimpleNamespace(stop_reason="end_turn", content=[_FakeTextBlock("hi")])
        client = _FakeClient(response)

        _reasoner(client, effort="low").respond([{"role": "user", "content": "hi"}])

        self.assertIn("JARVIS", client.messages.last_kwargs["system"])
        self.assertEqual(client.messages.last_kwargs["output_config"], {"effort": "low"})

    def test_user_name_is_appended_to_system_prompt_when_given(self):
        response = SimpleNamespace(stop_reason="end_turn", content=[_FakeTextBlock("hi")])
        client = _FakeClient(response)

        _reasoner(client).respond([{"role": "user", "content": "hi"}], user_name="Abhinay")

        self.assertIn("Abhinay", client.messages.last_kwargs["system"])

    def test_no_user_name_leaves_system_prompt_unchanged(self):
        response = SimpleNamespace(stop_reason="end_turn", content=[_FakeTextBlock("hi")])
        client = _FakeClient(response)

        _reasoner(client).respond([{"role": "user", "content": "hi"}])

        self.assertEqual(client.messages.last_kwargs["system"], SYSTEM_PROMPT)

    def test_full_message_history_is_forwarded_unchanged(self):
        response = SimpleNamespace(stop_reason="end_turn", content=[_FakeTextBlock("hi")])
        client = _FakeClient(response)
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "what's the weather"},
        ]

        _reasoner(client).respond(history)

        self.assertEqual(client.messages.last_kwargs["messages"], history)

    def test_refusal_stop_reason_raises(self):
        response = SimpleNamespace(stop_reason="refusal", content=[])
        with self.assertRaises(VoiceReasonerError):
            _reasoner(_FakeClient(response)).respond([{"role": "user", "content": "hi"}])

    def test_empty_text_response_raises(self):
        response = SimpleNamespace(stop_reason="end_turn", content=[])
        with self.assertRaises(VoiceReasonerError):
            _reasoner(_FakeClient(response)).respond([{"role": "user", "content": "hi"}])

    def test_whitespace_only_response_raises(self):
        response = SimpleNamespace(stop_reason="end_turn", content=[_FakeTextBlock("   ")])
        with self.assertRaises(VoiceReasonerError):
            _reasoner(_FakeClient(response)).respond([{"role": "user", "content": "hi"}])

    def test_api_connection_error_is_wrapped(self):
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        exc = anthropic.APIConnectionError(request=request)
        with self.assertRaises(VoiceReasonerError):
            _reasoner(_FakeClient(exc=exc)).respond([{"role": "user", "content": "hi"}])

    def test_rate_limit_error_is_wrapped(self):
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        response = httpx.Response(429, request=request)
        exc = anthropic.RateLimitError("rate limited", response=response, body=None)
        with self.assertRaises(VoiceReasonerError):
            _reasoner(_FakeClient(exc=exc)).respond([{"role": "user", "content": "hi"}])


if __name__ == "__main__":
    unittest.main(verbosity=2)
