"""
tests/test_code_interface.py
==============================
CodeInterface.submit() exercised against the REAL local `claude` CLI via
a real subprocess.run() call — not a mock, not a stub, matching this
project's standing preference for real local resources over mocks (see
eyv_poller's real ThreadingHTTPServer tests, voice's real-local-HTTP-
server TTS smoke test, and tests/test_live_smoke.py for the same idea
applied to the Claude API itself). tests/test_engine.py already covers
the submit() -> ExecutionResult *contract* against a hand-written stub
(_StubCodeInterface); this file is what actually exercises the
subprocess/CLI plumbing inside submit() itself, which nothing else in
this suite touches.

Gated the same way tests/test_live_smoke.py is gated: skipped by
default, opt in with JARVIS_LIVE_TEST=1. It's the same category of test
(depends on a real external dependency, costs a real Claude API call
under the hood since `claude -p` itself calls the Claude API) so it
gets the same opt-in gate, not just a shutil.which() check — a missing
`claude` CLI is *also* handled gracefully (skipped with a clear reason)
in case JARVIS_LIVE_TEST=1 is set on a machine that doesn't have the
CLI installed.

Every subprocess call in this file runs with the process's cwd
temporarily pointed at a throwaway temp directory (see setUp/tearDown)
rather than this repo — submit()'s real prompt explicitly tells Claude
Code not to use any tools, and the one successful-call test confirms
that instruction was followed (empty temp dir afterward), but there is
no reason for a real agentic subprocess invocation in a test to run
with this repo as its working directory regardless.

On the "malformed prompt" coverage bullet: CodeInterface._build_prompt()
always wraps `instructions` in fixed, non-empty framing text before it
ever reaches `claude -p`, so there is no way to make submit() hand the
real CLI a genuinely empty/malformed prompt (confirmed empirically: an
empty prompt IS rejected by the real CLI with a clean "Error: Input
must be provided..." and exit code 1, but that's not reachable through
submit()'s fixed prompt shape). The closest real, non-mocked equivalent
reachable through submit() itself is an oversized prompt that exceeds
the OS's command-line length limit — subprocess.run() raises a real
OSError (WinError 206 on Windows) before the CLI process even starts,
which is exactly the failure class submit()'s `except OSError` clause
exists to catch. See test_oversized_prompt_is_rejected_without_crashing.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis_core.code_interface import CodeInterface  # noqa: E402


@unittest.skipUnless(
    os.environ.get("JARVIS_LIVE_TEST") == "1",
    "real `claude` CLI subprocess test — set JARVIS_LIVE_TEST=1 to run (costs a real API call)",
)
class CodeInterfaceLiveSubprocessTest(unittest.TestCase):
    def setUp(self):
        if shutil.which("claude") is None:
            self.skipTest("the 'claude' CLI is not installed/on PATH on this machine")

        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self._orig_cwd = os.getcwd()
        os.chdir(self._tmpdir.name)
        self.addCleanup(os.chdir, self._orig_cwd)

    # -- 1. real successful headless call --------------------------------

    def test_successful_call_returns_execution_result_with_real_cli_output(self):
        code = CodeInterface(timeout_sec=90)
        instructions = (
            "There is no code change required for this test. Do not read, "
            "write, create, or modify any files, and do not run any shell "
            "commands or use any tools of any kind. Simply output the exact "
            "text: PONG"
        )

        t0 = time.time()
        result = code.submit(instructions, {"queue_item_id": "test-code-interface-live", "payload": {}})
        elapsed = time.time() - t0

        print(f"\n[live CodeInterface] elapsed={elapsed:.1f}s success={result.success} detail={result.detail!r}")

        self.assertTrue(result.success, f"expected a successful real CLI call, got: {result.detail}")
        self.assertIn("PONG", result.detail.upper())
        # The instructions explicitly forbid any file activity — confirm
        # Claude Code actually left the (throwaway) working directory alone.
        self.assertEqual(os.listdir(self._tmpdir.name), [])

    # -- 2. a call the real CLI/OS rejects, never a crash -----------------

    def test_oversized_prompt_is_rejected_without_crashing(self):
        code = CodeInterface(timeout_sec=30)
        # Comfortably past Windows' ~32K total command-line length limit;
        # see module docstring for why this, not a semantically "malformed"
        # prompt, is the real rejection path reachable through submit().
        oversized_instructions = "x" * 300_000

        result = code.submit(oversized_instructions, {})

        self.assertFalse(result.success)
        self.assertIn("failed to invoke Claude Code", result.detail)

    # -- 3. missing/uninstalled CLI never crashes --------------------------

    def test_missing_cli_binary_returns_failure_not_crash(self):
        code = CodeInterface(claude_cli_path="definitely-not-a-real-claude-binary-xyz", timeout_sec=10)

        result = code.submit("do nothing", {})

        self.assertFalse(result.success)
        self.assertIn("failed to invoke Claude Code", result.detail)

    # -- 4. timeout is caught, never an uncaught TimeoutExpired ------------

    def test_timeout_returns_failure_not_uncaught_timeoutexpired(self):
        code = CodeInterface(timeout_sec=0.01)

        result = code.submit("reply with PONG", {})

        self.assertFalse(result.success)
        self.assertIn("timed out", result.detail)


if __name__ == "__main__":
    unittest.main(verbosity=2)
