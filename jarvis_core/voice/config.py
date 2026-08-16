"""
jarvis_core/voice/config.py
=============================
Voice-layer-specific tunables, sourced from environment variables
(optionally backfilled from a local ".env" file — reuses
eyv_poller.config's _load_dotenv(), same as jarvis_core/config.py, so
one ".env" covers all three processes).

Deliberately its own, separate config from jarvis_core/config.py's
JarvisConfig — the voice layer doesn't touch the decision database's
required fields (architecture doc, confidence threshold, /jarvis/decisions
path) and has settings JarvisConfig has no business knowing about
(ElevenLabs voice/model, Whisper model size). The two do share
ANTHROPIC_API_KEY (same Claude account) and the local decision database
file (via jarvis_core.config.resolve_db_path() — see voice/store.py),
reused directly rather than duplicated.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from eyv_poller.config import _load_dotenv
from jarvis_core.config import JARVIS_DIR

DEFAULT_ELEVENLABS_KEY_PATH = JARVIS_DIR / "jarvis_core" / "voice" / "elevenlabs_key.txt"

# Face-recognition enrollment data (see face_id.py) — one .npy encoding
# per enrolled person, same 0600-local-file family as the ElevenLabs key.
DEFAULT_FACES_DIR = JARVIS_DIR / "jarvis_core" / "voice" / "faces"

# claude-opus-5 per this project's standing rule: default to the most
# capable model unless the user says otherwise. Latency is controlled
# via `effort`, not by swapping to a cheaper model — see
# voice_reasoner.py's module docstring.
DEFAULT_CLAUDE_MODEL = "claude-opus-5"
DEFAULT_CLAUDE_MAX_TOKENS = 1024
DEFAULT_CLAUDE_EFFORT = "low"

DEFAULT_WHISPER_MODEL_SIZE = "base"
DEFAULT_WHISPER_DEVICE = "cpu"
DEFAULT_WHISPER_COMPUTE_TYPE = "int8"
DEFAULT_MIN_UTTERANCE_SEC = 0.4
DEFAULT_NO_SPEECH_PROB_THRESHOLD = 0.6
# Greedy decoding by default — see stt.py's own comment on why beam
# search's accuracy gain isn't worth its CPU cost for a voice assistant.
DEFAULT_WHISPER_BEAM_SIZE = 1

DEFAULT_SAMPLE_RATE = 16000

# ElevenLabs' well-known default premade voice ("Rachel") — a sane
# technical default, freely overridable; not a business decision the
# way eyv_poller's poll interval was, so it's fine to default it.
DEFAULT_ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
DEFAULT_ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"
DEFAULT_ELEVENLABS_OUTPUT_FORMAT = "pcm_16000"
DEFAULT_REQUEST_TIMEOUT_SEC = 15.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_BASE_SEC = 1.5


class VoiceConfigError(Exception):
    """Raised for missing/invalid required voice-layer configuration.
    Always caught at the top of ui.py — never meant to surface as a bare
    traceback to an operator."""


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise VoiceConfigError(f"{name}={raw!r} is not a valid number") from None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise VoiceConfigError(f"{name}={raw!r} is not a valid integer") from None


@dataclass(frozen=True)
class VoiceConfig:
    claude_model: str
    claude_max_tokens: int
    claude_effort: str
    whisper_model_size: str
    whisper_device: str
    whisper_compute_type: str
    min_utterance_sec: float
    no_speech_prob_threshold: float
    whisper_beam_size: int
    sample_rate: int
    elevenlabs_key_path: Path
    faces_dir: Path
    elevenlabs_voice_id: str
    elevenlabs_model_id: str
    elevenlabs_output_format: str
    request_timeout_sec: float
    max_retries: int
    retry_backoff_base_sec: float

    @classmethod
    def load(cls, dotenv_path: Path | None = None) -> "VoiceConfig":
        _load_dotenv(dotenv_path or Path.cwd() / ".env")

        if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            raise VoiceConfigError(
                "ANTHROPIC_API_KEY is not set. It's required for voice_reasoner.py's "
                "conversational Claude calls. Set it in the environment or .env."
            )

        key_path_raw = os.environ.get("JARVIS_VOICE_ELEVENLABS_KEY_PATH", "").strip()
        elevenlabs_key_path = Path(key_path_raw) if key_path_raw else DEFAULT_ELEVENLABS_KEY_PATH

        faces_dir_raw = os.environ.get("JARVIS_VOICE_FACES_DIR", "").strip()
        faces_dir = Path(faces_dir_raw) if faces_dir_raw else DEFAULT_FACES_DIR

        return cls(
            claude_model=os.environ.get("JARVIS_VOICE_CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL),
            claude_max_tokens=_env_int("JARVIS_VOICE_CLAUDE_MAX_TOKENS", DEFAULT_CLAUDE_MAX_TOKENS),
            claude_effort=os.environ.get("JARVIS_VOICE_CLAUDE_EFFORT", DEFAULT_CLAUDE_EFFORT),
            whisper_model_size=os.environ.get("JARVIS_VOICE_WHISPER_MODEL", DEFAULT_WHISPER_MODEL_SIZE),
            whisper_device=os.environ.get("JARVIS_VOICE_WHISPER_DEVICE", DEFAULT_WHISPER_DEVICE),
            whisper_compute_type=os.environ.get("JARVIS_VOICE_WHISPER_COMPUTE_TYPE", DEFAULT_WHISPER_COMPUTE_TYPE),
            min_utterance_sec=_env_float("JARVIS_VOICE_MIN_UTTERANCE_SEC", DEFAULT_MIN_UTTERANCE_SEC),
            no_speech_prob_threshold=_env_float(
                "JARVIS_VOICE_NO_SPEECH_PROB_THRESHOLD", DEFAULT_NO_SPEECH_PROB_THRESHOLD
            ),
            whisper_beam_size=_env_int("JARVIS_VOICE_WHISPER_BEAM_SIZE", DEFAULT_WHISPER_BEAM_SIZE),
            sample_rate=_env_int("JARVIS_VOICE_SAMPLE_RATE", DEFAULT_SAMPLE_RATE),
            elevenlabs_key_path=elevenlabs_key_path,
            faces_dir=faces_dir,
            elevenlabs_voice_id=os.environ.get("JARVIS_VOICE_ELEVENLABS_VOICE_ID", DEFAULT_ELEVENLABS_VOICE_ID),
            elevenlabs_model_id=os.environ.get("JARVIS_VOICE_ELEVENLABS_MODEL_ID", DEFAULT_ELEVENLABS_MODEL_ID),
            elevenlabs_output_format=os.environ.get(
                "JARVIS_VOICE_ELEVENLABS_OUTPUT_FORMAT", DEFAULT_ELEVENLABS_OUTPUT_FORMAT
            ),
            request_timeout_sec=_env_float("JARVIS_VOICE_REQUEST_TIMEOUT_SEC", DEFAULT_REQUEST_TIMEOUT_SEC),
            max_retries=_env_int("JARVIS_VOICE_MAX_RETRIES", DEFAULT_MAX_RETRIES),
            retry_backoff_base_sec=_env_float(
                "JARVIS_VOICE_RETRY_BACKOFF_BASE_SEC", DEFAULT_RETRY_BACKOFF_BASE_SEC
            ),
        )
