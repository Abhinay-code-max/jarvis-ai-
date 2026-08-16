"""
jarvis_core/voice/auth.py
============================
Storage for the ElevenLabs API key used for text-to-speech (the voice
layer). Same convention eyv_poller/auth.py uses: one local file under
~/.jarvis/, permissions 0600, never hard-coded, never logged.

Unlike eyv_poller's EYV queue token — a shared secret this side and EYV
can either one originate — an ElevenLabs API key is minted by
ElevenLabs itself when you sign up. There's nothing to cryptographically
`generate` locally, so this module only offers:

  * `python -m jarvis_core.voice.auth set` — paste a key you already
    have (via getpass, hidden, not echoed to the terminal).
  * `python -m jarvis_core.voice.auth show` — re-print the stored key
    once, for reference.

get_key() re-reads the file on every call rather than caching it, so a
rotated key takes effect on the next TTS call without a process
restart — same reasoning as eyv_poller.auth.get_token().
"""
from __future__ import annotations

import getpass
import logging
import sys
from pathlib import Path

from jarvis_core.voice.config import DEFAULT_ELEVENLABS_KEY_PATH

_log = logging.getLogger("jarvis_core.voice.auth")


class VoiceAuthError(Exception):
    """Raised when the stored key can't be read/written. Never carries
    the key value itself in its message."""


def get_key(key_path: Path | None = None) -> str | None:
    """Returns the stored ElevenLabs API key, or None if setup hasn't
    been done yet. Never logs or raises with the key value itself."""
    path = key_path or DEFAULT_ELEVENLABS_KEY_PATH
    if not path.exists():
        return None
    try:
        key = path.read_text(encoding="utf-8").strip()
        return key or None
    except OSError:
        _log.warning("Could not read stored ElevenLabs key at %s", path, exc_info=True)
        return None


def _store_key(key: str, key_path: Path) -> None:
    if not key:
        raise VoiceAuthError("refusing to store an empty ElevenLabs API key")
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(key, encoding="utf-8")
    try:
        key_path.chmod(0o600)
    except OSError:
        # Best-effort on platforms where chmod doesn't fully apply
        # POSIX bits (e.g. some Windows filesystems) — same tolerance
        # eyv_poller.auth._store_token() uses.
        pass


def set_key(key: str, key_path: Path | None = None) -> None:
    """Stores an externally-provided key (see run_set_flow() for the
    interactive getpass entry point)."""
    _store_key(key.strip(), key_path or DEFAULT_ELEVENLABS_KEY_PATH)


def run_set_flow(key_path: Path | None = None) -> None:
    path = key_path or DEFAULT_ELEVENLABS_KEY_PATH
    key = getpass.getpass("Paste your ElevenLabs API key (input hidden): ").strip()
    if not key:
        raise VoiceAuthError("No key entered.")
    set_key(key, path)
    print(f"ElevenLabs API key saved to {path}.")


def run_show_flow(key_path: Path | None = None) -> None:
    path = key_path or DEFAULT_ELEVENLABS_KEY_PATH
    key = get_key(path)
    if not key:
        print(f"No ElevenLabs key stored yet at {path}. Run 'set' first.")
        return
    print(f"Stored ElevenLabs key (from {path}): {key}")


def _main(argv: list[str]) -> int:
    if len(argv) != 1 or argv[0] not in ("set", "show"):
        print("Usage: python -m jarvis_core.voice.auth {set|show}")
        return 1
    try:
        {"set": run_set_flow, "show": run_show_flow}[argv[0]]()
    except VoiceAuthError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
