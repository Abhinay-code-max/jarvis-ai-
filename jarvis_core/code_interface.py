"""
jarvis_core/code_interface.py
===============================
B.2.2's "Invoke Claude Code through the B.3 interface" — implemented
here as the smallest clean abstraction, because a standalone B.3
component does not exist yet anywhere in this codebase.

CodeInterface is the seam: engine.py calls only submit() on it and never
implements code changes itself (per B.2.2's "The reasoning core should
not itself implement arbitrary code changes"). When a real, dedicated
B.3 component exists (a queue, a sandboxed runner, a PR-creation flow),
swap this class for a client of that component — nothing in engine.py
needs to change, since it only depends on the submit() -> ExecutionResult
contract below.

ASSUMPTION: the concrete implementation shells out to the local `claude`
CLI (Claude Code) in headless/print mode (`claude -p <prompt>`). This is
a real, working caller — not a stub that always succeeds — but it is
still standing in for whatever B.3's eventual dedicated interface looks
like. If the `claude` CLI isn't installed/authenticated on this machine,
submit() reports that as ExecutionResult(success=False, ...) rather than
raising, since a missing B.3 dependency is a routing-time failure to
record, not a reason to crash the reasoning core.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

_log = logging.getLogger("jarvis_core.code_interface")

DEFAULT_TIMEOUT_SEC = 600.0


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    detail: str


class CodeInterface:
    def __init__(self, claude_cli_path: str | None = None, timeout_sec: float = DEFAULT_TIMEOUT_SEC):
        self._cli = claude_cli_path or shutil.which("claude")
        self._timeout_sec = timeout_sec

    def submit(self, instructions: str, context: dict[str, Any]) -> ExecutionResult:
        """Hands `instructions` (plus the queue item's context) to Claude
        Code. Never raises for an ordinary failed/timed-out run — those
        come back as ExecutionResult(success=False, detail=<why>) so the
        caller can record the real failure reason instead of a stack
        trace, and never silently reports success for a run that
        actually failed."""
        if not self._cli:
            _log.error("B.3 interface unavailable: no 'claude' CLI found on PATH")
            return ExecutionResult(
                success=False,
                detail="B.3 interface unavailable: no 'claude' CLI found on PATH",
            )

        prompt = self._build_prompt(instructions, context)
        try:
            proc = subprocess.run(
                [self._cli, "-p", prompt],
                capture_output=True,
                text=True,
                timeout=self._timeout_sec,
            )
        except subprocess.TimeoutExpired:
            _log.error("Claude Code timed out after %.0fs", self._timeout_sec)
            return ExecutionResult(success=False, detail=f"Claude Code timed out after {self._timeout_sec:.0f}s")
        except OSError as e:
            _log.error("Failed to invoke Claude Code: %s", e)
            return ExecutionResult(success=False, detail=f"failed to invoke Claude Code: {e}")

        if proc.returncode != 0:
            detail = f"Claude Code exited {proc.returncode}: {proc.stderr.strip()[:2000]}"
            _log.error(detail)
            return ExecutionResult(success=False, detail=detail)

        return ExecutionResult(success=True, detail=proc.stdout.strip()[:4000])

    @staticmethod
    def _build_prompt(instructions: str, context: dict[str, Any]) -> str:
        return (
            "JARVIS's reasoning core has determined this queue item needs "
            "a code change. Implement exactly what is described below.\n\n"
            f"Instructions:\n{instructions}\n\n"
            f"Queue item context (JSON):\n{json.dumps(context, indent=2, default=str)}"
        )
