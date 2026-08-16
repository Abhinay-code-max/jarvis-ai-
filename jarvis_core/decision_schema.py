"""
jarvis_core/decision_schema.py
================================
The structured decision Claude must return for every queue item
(B.2.1). Validation is intentionally strict: a response that doesn't
parse into exactly this shape raises DecisionValidationError rather
than being coerced or guessed at. See engine.py for what happens on
that error — the decision row is recorded as `failed`, never silently
treated as e.g. RESOLVED.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DECISION_TYPES = ("NEEDS_CODE", "MARKETING_ACTION", "RESOLVED", "NEEDS_APPROVAL")


class DecisionValidationError(Exception):
    """Raised when Claude's response is missing a required field, has an
    unrecognized `decision` value, or `confidence` is out of range."""


@dataclass(frozen=True)
class ReasoningDecision:
    decision: str
    reason: str
    action_type: str | None
    action_payload: dict[str, Any]
    confidence: float
    raw: dict[str, Any]


def parse_decision(raw: Any) -> ReasoningDecision:
    """Validates and parses a raw dict (Claude's tool-call `input`, or an
    already-decoded JSON object) into a ReasoningDecision. Raises
    DecisionValidationError on any deviation from the required shape —
    never fills in a default or best-guesses a missing/invalid field."""
    if not isinstance(raw, dict):
        raise DecisionValidationError(f"expected a JSON object, got {type(raw).__name__}")

    decision = raw.get("decision")
    if decision not in DECISION_TYPES:
        raise DecisionValidationError(
            f"'decision' must be one of {DECISION_TYPES}, got {decision!r}"
        )

    reason = raw.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise DecisionValidationError("'reason' must be a non-empty string")

    confidence = raw.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise DecisionValidationError("'confidence' must be a number")
    confidence = float(confidence)
    if not (0.0 <= confidence <= 1.0):
        raise DecisionValidationError(f"'confidence' must be between 0 and 1, got {confidence}")

    action = raw.get("action") or {}
    if not isinstance(action, dict):
        raise DecisionValidationError("'action' must be an object if present")
    action_type = action.get("type")
    if action_type is not None and not isinstance(action_type, str):
        raise DecisionValidationError("'action.type' must be a string if present")

    return ReasoningDecision(
        decision=decision,
        reason=reason,
        action_type=action_type,
        action_payload=action,
        confidence=confidence,
        raw=raw,
    )
