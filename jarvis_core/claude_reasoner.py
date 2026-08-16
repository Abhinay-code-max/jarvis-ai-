"""
jarvis_core/claude_reasoner.py
================================
Claude API integration for B.2.1. Loads the JARVIS architecture
document once at construction time and uses it verbatim as the system
prompt — see docs/jarvis_architecture.md's own header for what that
document currently contains and the assumption it stands in for (no
prior architecture document existed anywhere in this codebase).

Structured output is enforced by forcing a tool call (`tool_choice`
pinned to the one `record_decision` tool) rather than asking for JSON
in prose and regex-parsing it out — this is what B.2.1's "do not allow
malformed or ambiguous Claude responses to silently continue execution"
is enforced by at the transport level. Claude can still return a
malformed *value* for a field (an out-of-range confidence, an
unrecognized decision string) inside a syntactically valid tool call —
jarvis_core.decision_schema.parse_decision() is what catches that;
ClaudeReasoner.reason() lets DecisionValidationError propagate
unchanged so callers only need to handle it in one place.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import anthropic

from jarvis_core.decision_schema import DECISION_TYPES, ReasoningDecision, parse_decision

_log = logging.getLogger("jarvis_core.claude_reasoner")

_TOOL_NAME = "record_decision"

_TOOL_SCHEMA = {
    "name": _TOOL_NAME,
    "description": (
        "Record the structured routing decision for this queue item. "
        "Always call this exactly once — never respond with plain text "
        "instead of calling it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "enum": list(DECISION_TYPES),
                "description": "The routing outcome for this queue item.",
            },
            "reason": {
                "type": "string",
                "description": (
                    "Why this decision is correct for this ticket. Written "
                    "for the human reviewer / audit log, not for yourself."
                ),
            },
            "action": {
                "type": "object",
                "description": (
                    "Enough information for the executor to perform the "
                    "selected action safely. Include at least `type` and, "
                    "for NEEDS_CODE/MARKETING_ACTION, `instructions`."
                ),
                "properties": {
                    "type": {"type": "string"},
                    "instructions": {"type": "string"},
                },
            },
            "confidence": {
                "type": "number",
                "description": "Genuine confidence in this decision, from 0.0 to 1.0.",
            },
        },
        "required": ["decision", "reason", "confidence"],
        "additionalProperties": False,
    },
}


class ClaudeReasonerError(Exception):
    """Base class for reasoning failures: network errors, timeouts,
    rate limits, non-2xx API responses, safety refusals, or a response
    that never produced the required tool call. Never carries the raw
    queue item payload in its message (ticket content isn't known to be
    free of sensitive data)."""


class ClaudeReasoner:
    """Wraps one Claude client + one loaded architecture doc. Stateless
    across calls — safe to reuse for every queue item in a process's
    lifetime."""

    def __init__(
        self,
        architecture_doc_path: Path,
        model: str,
        max_tokens: int,
        client: Any | None = None,
    ):
        self._system_prompt = architecture_doc_path.read_text(encoding="utf-8")
        self._model = model
        self._max_tokens = max_tokens
        self._client = client or anthropic.Anthropic()

    def reason(self, queue_item_context: dict[str, Any]) -> ReasoningDecision:
        """Calls Claude with the queue item's payload/metadata as the user
        message, forces the structured-decision tool call, and returns a
        validated ReasoningDecision.

        Raises ClaudeReasonerError for any API-layer failure (network,
        timeout, rate limit, non-2xx, safety refusal, missing tool call)
        and DecisionValidationError (from decision_schema) for a
        syntactically-present but semantically invalid decision value.
        Callers (engine.py) are expected to catch both and record the
        queue item as failed — never to catch neither and let a bad
        response propagate as a crash, and never to catch broadly and
        silently continue."""
        user_message = (
            "A queue item requires a routing decision. Item context "
            "(JSON):\n\n" + json.dumps(queue_item_context, indent=2, default=str)
        )

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=self._system_prompt,
                tools=[_TOOL_SCHEMA],
                tool_choice={"type": "tool", "name": _TOOL_NAME},
                messages=[{"role": "user", "content": user_message}],
            )
        except anthropic.APIConnectionError as e:
            raise ClaudeReasonerError(f"network error calling Claude API: {e}") from e
        except anthropic.RateLimitError as e:
            raise ClaudeReasonerError(f"Claude API rate limited: {e}") from e
        except anthropic.APIStatusError as e:
            raise ClaudeReasonerError(f"Claude API error (HTTP {e.status_code}): {e.message}") from e
        except anthropic.AnthropicError as e:
            raise ClaudeReasonerError(f"Claude API request failed: {e}") from e

        if response.stop_reason == "refusal":
            raise ClaudeReasonerError("Claude declined to reason over this item (safety refusal)")

        tool_use = next(
            (block for block in response.content if getattr(block, "type", None) == "tool_use" and block.name == _TOOL_NAME),
            None,
        )
        if tool_use is None:
            raise ClaudeReasonerError(
                f"Claude did not return the required tool call (stop_reason={response.stop_reason!r})"
            )

        return parse_decision(tool_use.input)
