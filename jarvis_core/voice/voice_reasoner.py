"""
jarvis_core/voice/voice_reasoner.py
======================================
Conversational Claude calls for the voice layer — deliberately separate
from jarvis_core/claude_reasoner.py, which forces every response
through decision_schema.py's 4-type structured-decision tool call.
Voice queries are open conversation, not ticket routing, so they get
their own system prompt here and no forced tool_choice — see
this module never imports decision_schema.py or anything from engine.py.

Model defaults to claude-opus-5, per this project's standing rule to
default to the most capable model unless told otherwise. Latency for a
live back-and-forth conversation is controlled via `output_config.effort`
("low" by default), not by swapping to a cheaper model — lower effort
still uses adaptive thinking and is the documented mitigation for
verbosity/latency on Claude Opus 5, rather than disabling thinking
outright (which has its own failure modes: tool-call-as-text and
thinking-tag leakage — moot here since this reasoner never defines
tools, but the effort lever is still the right one to reach for first).
"""
from __future__ import annotations

import logging
from typing import Any

import anthropic

_log = logging.getLogger("jarvis_core.voice.voice_reasoner")

SYSTEM_PROMPT = """\
You are JARVIS, speaking with the user out loud through a voice interface.

Respond the way a person would speak in conversation: short, natural
sentences. Never use markdown, bullet points, numbered lists, code
blocks, or headings — everything you say gets read aloud by
text-to-speech, and formatting characters would be spoken as literal
symbols. If you need to list a few things, say them as a spoken
sentence ("first ... then ... and finally ...") rather than a list.

Keep answers focused and conversational rather than exhaustive — this
is a spoken back-and-forth, not a written report. If the user's request
sounds like it should become an actual EYV engineering or marketing
ticket (a bug, a feature request, a marketing action), say so and
describe what you'd do, but do not claim to have already filed or
executed anything — this voice layer only talks; it never files
tickets, writes code, or takes action on JARVIS's decision queue.
"""


class VoiceReasonerError(Exception):
    """Base class for reasoning failures: network errors, timeouts,
    rate limits, non-2xx API responses, safety refusals, or an empty
    response. Never carries the raw conversation content in its
    message."""


class VoiceReasoner:
    def __init__(
        self,
        model: str,
        max_tokens: int,
        effort: str,
        client: Any | None = None,
    ):
        self._model = model
        self._max_tokens = max_tokens
        self._effort = effort
        self._client = client or anthropic.Anthropic()

    def respond(self, messages: list[dict[str, Any]]) -> str:
        """`messages` is the full conversation so far (the API is
        stateless — see claude-api conventions), each entry
        {"role": "user"|"assistant", "content": str}. Returns Claude's
        reply text. Raises VoiceReasonerError on any API-layer failure
        or an empty/refused response — callers (session.py) must never
        treat a caught VoiceReasonerError as "Claude said nothing", only
        as "the call failed"."""
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=SYSTEM_PROMPT,
                output_config={"effort": self._effort},
                messages=messages,
            )
        except anthropic.APIConnectionError as e:
            raise VoiceReasonerError(f"network error calling Claude API: {e}") from e
        except anthropic.RateLimitError as e:
            raise VoiceReasonerError(f"Claude API rate limited: {e}") from e
        except anthropic.APIStatusError as e:
            raise VoiceReasonerError(f"Claude API error (HTTP {e.status_code}): {e.message}") from e
        except anthropic.AnthropicError as e:
            raise VoiceReasonerError(f"Claude API request failed: {e}") from e

        if response.stop_reason == "refusal":
            raise VoiceReasonerError("Claude declined to respond (safety refusal)")

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()
        if not text:
            raise VoiceReasonerError(f"Claude returned no text content (stop_reason={response.stop_reason!r})")

        return text
