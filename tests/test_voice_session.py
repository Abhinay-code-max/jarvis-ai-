"""
tests/test_voice_session.py
==============================
jarvis_core.voice.session.VoiceSession exercised against hand-written
stubs for every collaborator — recorder, player, transcriber, reasoner,
tts, store. Never a real microphone, real Whisper model, real Claude
API, or real ElevenLabs call. Same "hand-written stub over mocking
framework" convention as tests/test_engine.py.

Covers the misfire-resilience contract (a None transcript never reaches
Claude), every external failure mode stopping the turn at the right
point without crashing, the state-transition sequence, interrupt/barge-in,
and that a VoiceStore write failure is logged but never blocks the rest
of a turn.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from jarvis_core.voice.audio_io import AudioIOError  # noqa: E402
from jarvis_core.voice.session import VoiceSession, VoiceState  # noqa: E402
from jarvis_core.voice.stt import SttError  # noqa: E402
from jarvis_core.voice.store import VoiceStoreError  # noqa: E402
from jarvis_core.voice.tts import ElevenLabsError  # noqa: E402
from jarvis_core.voice.voice_reasoner import VoiceReasonerError  # noqa: E402

_AUDIO = np.zeros(16000, dtype=np.float32)


class _StubRecorder:
    sample_rate = 16000

    def __init__(self, audio=_AUDIO, start_error=None, stop_error=None):
        self._audio = audio
        self._start_error = start_error
        self._stop_error = stop_error
        self.start_calls = 0
        self.stop_calls = 0

    def start(self):
        self.start_calls += 1
        if self._start_error:
            raise self._start_error

    def stop(self):
        self.stop_calls += 1
        if self._stop_error:
            raise self._stop_error
        return self._audio


class _StubPlayer:
    def __init__(self, play_error=None):
        self._play_error = play_error
        self.play_calls = []
        self.stop_calls = 0

    def play(self, pcm, sample_rate):
        self.play_calls.append((pcm, sample_rate))
        if self._play_error:
            raise self._play_error

    def stop(self):
        self.stop_calls += 1


class _StubTranscriber:
    def __init__(self, result="hello jarvis"):
        self._result = result
        self.calls = []

    def transcribe(self, audio, sample_rate):
        self.calls.append((audio, sample_rate))
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _StubReasoner:
    def __init__(self, result="hello there"):
        self._result = result
        self.calls = []
        self.user_names: list[str | None] = []

    def respond(self, messages, user_name=None):
        self.calls.append(list(messages))
        self.user_names.append(user_name)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _StubTTS:
    def __init__(self, result=(b"\x00\x01", 16000)):
        self._result = result
        self.calls = []

    def synthesize(self, text):
        self.calls.append(text)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _StubStore:
    def __init__(self, record_error=None):
        self._record_error = record_error
        self.turns = []

    def record_turn(self, session_id, role, text):
        if self._record_error:
            raise self._record_error
        self.turns.append((session_id, role, text))


def _session(recorder=None, player=None, transcriber=None, reasoner=None, tts=None, store=None):
    states, transcripts, errors = [], [], []
    session = VoiceSession(
        recorder=recorder or _StubRecorder(),
        player=player or _StubPlayer(),
        transcriber=transcriber or _StubTranscriber(),
        reasoner=reasoner or _StubReasoner(),
        tts=tts or _StubTTS(),
        store=store or _StubStore(),
        on_state_changed=states.append,
        on_transcript=lambda role, text: transcripts.append((role, text)),
        on_error=errors.append,
    )
    return session, states, transcripts, errors


class VoiceSessionTest(unittest.TestCase):
    def _run_full_turn(self, session):
        session.start_listening()
        session.stop_listening_and_respond()

    # -- happy path -------------------------------------------------------

    def test_full_turn_transcribes_reasons_and_plays_response(self):
        recorder, player = _StubRecorder(), _StubPlayer()
        transcriber, reasoner, tts, store = _StubTranscriber("hello jarvis"), _StubReasoner("hello there"), _StubTTS((b"pcm-bytes", 16000)), _StubStore()
        session, states, transcripts, errors = _session(recorder, player, transcriber, reasoner, tts, store)

        self._run_full_turn(session)

        self.assertEqual(states, [VoiceState.LISTENING, VoiceState.THINKING, VoiceState.SPEAKING, VoiceState.IDLE])
        self.assertEqual(transcripts, [("user", "hello jarvis"), ("assistant", "hello there")])
        self.assertEqual(errors, [])
        self.assertEqual(player.play_calls, [(b"pcm-bytes", 16000)])
        self.assertEqual(store.turns, [
            (session.session_id, "user", "hello jarvis"),
            (session.session_id, "assistant", "hello there"),
        ])
        self.assertEqual(session.state, VoiceState.IDLE)

    def test_set_user_name_is_passed_to_the_reasoner(self):
        transcriber = _StubTranscriber("hello jarvis")
        reasoner = _StubReasoner("hello there")
        session, *_ = _session(transcriber=transcriber, reasoner=reasoner)

        session.set_user_name("Abhinay")
        self._run_full_turn(session)

        self.assertEqual(reasoner.user_names, ["Abhinay"])

    def test_no_user_name_by_default(self):
        transcriber = _StubTranscriber("hello jarvis")
        reasoner = _StubReasoner("hello there")
        session, *_ = _session(transcriber=transcriber, reasoner=reasoner)

        self._run_full_turn(session)

        self.assertEqual(reasoner.user_names, [None])

    def test_conversation_history_accumulates_across_turns(self):
        transcriber = _StubTranscriber("first question")
        reasoner = _StubReasoner("first answer")
        session, *_ = _session(transcriber=transcriber, reasoner=reasoner)

        self._run_full_turn(session)

        transcriber._result = "second question"
        reasoner._result = "second answer"
        self._run_full_turn(session)

        self.assertEqual(reasoner.calls[1], [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
        ])

    # -- misfire resilience: the core requirement --------------------------

    def test_silent_utterance_never_reaches_claude_or_tts(self):
        transcriber = _StubTranscriber(None)  # simulates STT's own misfire guard
        reasoner, tts, player = _StubReasoner(), _StubTTS(), _StubPlayer()
        session, states, transcripts, errors = _session(transcriber=transcriber, reasoner=reasoner, tts=tts, player=player)

        self._run_full_turn(session)

        self.assertEqual(reasoner.calls, [])
        self.assertEqual(tts.calls, [])
        self.assertEqual(player.play_calls, [])
        self.assertEqual(transcripts, [])
        self.assertEqual(errors, [])  # silence is not an error
        self.assertEqual(session.state, VoiceState.IDLE)

    # -- external failures: each stops the turn at the right point ---------

    def test_recorder_start_failure_emits_error_and_stays_idle(self):
        recorder = _StubRecorder(start_error=AudioIOError("no microphone"))
        session, states, _transcripts, errors = _session(recorder=recorder)

        session.start_listening()

        self.assertEqual(session.state, VoiceState.IDLE)
        self.assertEqual(len(errors), 1)
        self.assertIn("no microphone", errors[0])

    def test_recorder_stop_failure_emits_error(self):
        recorder = _StubRecorder(stop_error=AudioIOError("device unplugged"))
        session, _states, _transcripts, errors = _session(recorder=recorder)

        session.start_listening()
        session.stop_listening_and_respond()

        self.assertEqual(session.state, VoiceState.IDLE)
        self.assertIn("device unplugged", errors[0])

    def test_stt_failure_emits_error_and_never_calls_claude(self):
        transcriber = _StubTranscriber(SttError("whisper crashed"))
        reasoner = _StubReasoner()
        session, _states, _transcripts, errors = _session(transcriber=transcriber, reasoner=reasoner)

        self._run_full_turn(session)

        self.assertEqual(reasoner.calls, [])
        self.assertIn("whisper crashed", errors[0])
        self.assertEqual(session.state, VoiceState.IDLE)

    def test_claude_failure_emits_error_and_never_calls_tts(self):
        reasoner = _StubReasoner(VoiceReasonerError("rate limited"))
        tts, store = _StubTTS(), _StubStore()
        session, _states, _transcripts, errors = _session(reasoner=reasoner, tts=tts, store=store)

        self._run_full_turn(session)

        self.assertEqual(tts.calls, [])
        self.assertIn("rate limited", errors[0])
        # The user's turn was still captured before the Claude call failed.
        self.assertEqual(store.turns, [(session.session_id, "user", "hello jarvis")])

    def test_tts_failure_still_records_assistant_turn_but_never_plays(self):
        tts = _StubTTS(ElevenLabsError("synthesis failed"))
        player, store = _StubPlayer(), _StubStore()
        session, _states, _transcripts, errors = _session(tts=tts, player=player, store=store)

        self._run_full_turn(session)

        self.assertEqual(player.play_calls, [])
        self.assertIn("synthesis failed", errors[0])
        self.assertEqual(
            [t[1] for t in store.turns], ["user", "assistant"],
            "Claude's reply is still worth persisting even if speaking it aloud fails",
        )

    def test_playback_failure_emits_error_but_turn_still_reaches_idle(self):
        player = _StubPlayer(play_error=AudioIOError("speaker error"))
        session, _states, _transcripts, errors = _session(player=player)

        self._run_full_turn(session)

        self.assertIn("speaker error", errors[0])
        self.assertEqual(session.state, VoiceState.IDLE)

    # -- persistence failures are logged, never turn-blocking ---------------

    def test_store_failure_does_not_block_the_rest_of_the_turn(self):
        store = _StubStore(record_error=VoiceStoreError("disk full"))
        player = _StubPlayer()
        session, _states, _transcripts, errors = _session(store=store, player=player)

        self._run_full_turn(session)

        # The conversation still completes end to end...
        self.assertEqual(player.play_calls, [(b"\x00\x01", 16000)])
        self.assertEqual(session.state, VoiceState.IDLE)
        # ...and a persistence failure is not surfaced as a user-facing
        # conversational error (it's logged instead — see session.py).
        self.assertEqual(errors, [])

    # -- interrupt / barge-in ----------------------------------------------

    def test_interrupt_stops_player_and_resets_to_idle(self):
        player = _StubPlayer()
        session, *_ = _session(player=player)
        session.interrupt()
        self.assertEqual(player.stop_calls, 1)
        self.assertEqual(session.state, VoiceState.IDLE)

    def test_interrupt_while_listening_discards_the_recording(self):
        recorder = _StubRecorder()
        session, *_ = _session(recorder=recorder)

        session.start_listening()
        self.assertEqual(session.state, VoiceState.LISTENING)

        session.interrupt()

        self.assertEqual(recorder.stop_calls, 1)
        self.assertEqual(session.state, VoiceState.IDLE)

    def test_interrupt_during_listening_does_not_transcribe_the_discarded_audio(self):
        recorder = _StubRecorder()
        transcriber = _StubTranscriber("should never be transcribed")
        session, *_ = _session(recorder=recorder, transcriber=transcriber)

        session.start_listening()
        session.interrupt()

        self.assertEqual(transcriber.calls, [])

    # -- guard against overlapping calls ------------------------------------

    def test_stop_listening_while_idle_is_a_noop(self):
        session, states, _transcripts, errors = _session()
        session.stop_listening_and_respond()  # never started listening
        self.assertEqual(states, [])
        self.assertEqual(errors, [])

    def test_start_listening_while_already_listening_is_ignored(self):
        recorder = _StubRecorder()
        session, *_ = _session(recorder=recorder)
        session.start_listening()
        session.start_listening()
        self.assertEqual(recorder.start_calls, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
