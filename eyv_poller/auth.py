"""
eyv_poller/auth.py
====================
Storage for the single JARVIS-only bearer token this service sends to
EYV's /jarvis/* endpoints. Same convention JARVIS-XL used for its GitHub
PAT (core/github_ci_auth.py) and its service:support Hermes token
(core/hermes_auth.py): one local file under ~/.jarvis/, permissions
0600, never hard-coded, never logged.

The token is a shared secret this service and EYV must agree on — there
is no OAuth round-trip and no third party minting it, so either side can
originate the value:

  * `python -m eyv_poller.auth generate` creates a fresh cryptographically
    random token, stores it locally, and prints it once so it can be
    pasted into EYV's Railway environment configuration. Prefer this for
    the first-ever setup — it needs nothing from EYV first.
  * `python -m eyv_poller.auth set` reads a token via getpass (hidden,
    not echoed to the terminal) for the case where the value instead
    originates on EYV's side and needs to be copied over here.
  * `python -m eyv_poller.auth show` re-prints the currently stored
    token once, for re-pasting into Railway (e.g. after a token was
    generated here but Railway's config was never updated, or needs
    re-entering after a redeploy wiped it).

get_token() re-reads the file on every call rather than caching it in a
module-level variable, so a rotated token takes effect on the very next
poll without a process restart.
"""
from __future__ import annotations

import getpass
import logging
import secrets
import sys
from pathlib import Path

from eyv_poller.config import DEFAULT_TOKEN_PATH

_log = logging.getLogger("eyv_poller.auth")

_TOKEN_BYTES = 32  # secrets.token_urlsafe(32) -> 256 bits of entropy


class TokenError(Exception):
    """Raised when the stored token can't be read/written. Never carries
    the token value itself in its message."""


def get_token(token_path: Path | None = None) -> str | None:
    """Returns the stored bearer token, or None if setup hasn't been done
    yet. Never logs or raises with the token value itself."""
    path = token_path or DEFAULT_TOKEN_PATH
    if not path.exists():
        return None
    try:
        token = path.read_text(encoding="utf-8").strip()
        return token or None
    except OSError:
        _log.warning("Could not read stored EYV token at %s", path, exc_info=True)
        return None


def _store_token(token: str, token_path: Path) -> None:
    if not token:
        raise TokenError("refusing to store an empty token")
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(token, encoding="utf-8")
    try:
        token_path.chmod(0o600)  # owner read/write only
    except OSError:
        # Best-effort on platforms (e.g. some Windows filesystems) where
        # chmod doesn't fully apply POSIX bits — same tolerance
        # JARVIS-XL's own token-storage modules use.
        pass


def generate_token(token_path: Path | None = None) -> str:
    """Generates a new random token, overwrites any previously stored
    one (revoke-and-reissue — no grace period), and returns it so the
    caller can display it exactly once."""
    path = token_path or DEFAULT_TOKEN_PATH
    token = secrets.token_urlsafe(_TOKEN_BYTES)
    _store_token(token, path)
    return token


def set_token(token: str, token_path: Path | None = None) -> None:
    """Stores an externally-provided token (see run_set_flow() for the
    interactive getpass entry point)."""
    _store_token(token.strip(), token_path or DEFAULT_TOKEN_PATH)


def run_generate_flow(token_path: Path | None = None) -> None:
    path = token_path or DEFAULT_TOKEN_PATH
    token = generate_token(path)
    print(f"Generated a new EYV queue token and stored it at {path}.")
    print(f"Token (copy this now into EYV's Railway env config — it will not be shown again here): {token}")


def run_set_flow(token_path: Path | None = None) -> None:
    path = token_path or DEFAULT_TOKEN_PATH
    token = getpass.getpass("Paste the EYV queue bearer token (input hidden): ").strip()
    if not token:
        raise TokenError("No token entered.")
    set_token(token, path)
    print(f"Token saved to {path}.")


def run_show_flow(token_path: Path | None = None) -> None:
    path = token_path or DEFAULT_TOKEN_PATH
    token = get_token(path)
    if not token:
        print(f"No token stored yet at {path}. Run 'generate' or 'set' first.")
        return
    print(f"Stored token (from {path} — copy into EYV's Railway env config as needed): {token}")


def _main(argv: list[str]) -> int:
    if len(argv) != 1 or argv[0] not in ("generate", "set", "show"):
        print("Usage: python -m eyv_poller.auth {generate|set|show}")
        return 1
    {"generate": run_generate_flow, "set": run_set_flow, "show": run_show_flow}[argv[0]]()
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
