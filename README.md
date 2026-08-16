# eyv_poller (B.1) + jarvis_core (B.2)

This repo holds two small, standalone Python processes for JARVIS:

* **`eyv_poller` (B.1)** — polls EYV's `GET /jarvis/queue` endpoint on
  an interval and hands whatever it finds off to a processing step.
* **`jarvis_core` (B.2)** — the JARVIS Reasoning Core. Reuses B.1's
  queue polling wholesale and adds the actual decision-making: for
  every queue item, it calls Claude, validates a structured decision,
  routes it (to Claude Code, to Bob via `/jarvis/decisions`, to a local
  approval record, or to nothing at all if resolved), and records every
  decision durably in a local SQLite database. See "jarvis_core — B.2
  JARVIS Reasoning Core" below.

Neither is part of, or depends on, JARVIS-XL or Hermes — no shared
database, no shared auth, no shared process. They run on their own.

## ⚠️ Open decision: polling interval

**`EYV_POLL_INTERVAL_SEC` has no built-in default.** The service refuses
to start without it set explicitly (`Config.load()` raises a clear
`ConfigError`). This is deliberate, not an oversight: "a minute or two"
has been floated as a plausible interval for ticket escalations, but
that has not been confirmed as the right value, and this service should
not silently bake in a guess that then becomes the de facto answer by
default. **Confirm the interval (with whoever owns EYV's ticket
escalation SLAs) before running this in production**, then set it via
`EYV_POLL_INTERVAL_SEC` in the environment or `.env`.

## ⚠️ `/jarvis/queue` does not exist on EYV yet

Per the current state of things, EYV has no `/jarvis/queue` endpoint and
no JARVIS-only bearer token wired up yet — both need to be built/
confirmed on EYV's side (Railway deployment) before this service can be
live-tested end-to-end. Until then:

* `eyv_poller/schema.py` isolates response parsing behind
  `parse_queue_response()` so the real schema can be dropped in later
  without touching the client, poller, or processor. It currently
  accepts a bare JSON list or an object with an `items`/`queue`/
  `results` list — a guess, not a contract.
* Everything else (auth, retry logic, the poll loop, logging) is
  schema-independent and has been exercised against a real local HTTP
  server in the test suite (see "Testing" below) rather than against
  the real EYV deployment.

## Project structure

```
eyv_poller/
  __init__.py
  __main__.py       entry point for `python -m eyv_poller`
  main.py           wiring: config -> logging -> auth -> client -> poller loop
  config.py         all tunables, sourced from env / .env
  auth.py           local token storage (generate / set / show), 0600 file
  client.py         HTTP transport to EYV: auth header, retry/backoff, timeouts
  schema.py         response parsing — isolated pending EYV schema confirmation
  processor.py      extensibility point for what happens to queue items
  poller.py         the sleep/poll/process loop, one failure-contained tick at a time
  logging_setup.py  console + rotating local file logging
jarvis_core/
  __init__.py
  __main__.py       entry point for `python -m jarvis_core`
  main.py           wiring: eyv_poller's Config/auth/client/PollerLoop -> JarvisEngine
  config.py         B.2-specific tunables, sourced from env / .env
  db.py             durable local SQLite decision log (B.2.3)
  decision_schema.py structured-decision validation (B.2.1)
  claude_reasoner.py Claude API integration — forced tool call, one call per queue item
  engine.py         reasoning + routing orchestration (B.2.1 + B.2.2)
  code_interface.py  B.3 seam (Claude Code) — see its module docstring for the assumption
  marketing_client.py POST /jarvis/decisions client (Bob)
  approval_interface.py B.4 seam (approvals) — approve()/reject()/list_pending()
  approvals_cli.py  terminal front end for approval_interface.py
  dashboard.py      PyQt6 front end for approval_interface.py (B.4 Approval UI)
  logging_setup.py  console + rotating local file logging
  voice/            two-way voice interaction — see "Voice" section below
    __init__.py
    __main__.py      entry point for `python -m jarvis_core.voice`
    config.py         voice-layer tunables, sourced from env / .env
    auth.py           ElevenLabs API key storage (set/show), 0600 file
    audio_io.py        mic capture (push-to-talk) + speaker playback (sounddevice)
    stt.py             local speech-to-text (faster-whisper), misfire-resilient
    voice_reasoner.py  conversational Claude calls — separate from claude_reasoner.py,
                       no forced tool_choice, not routed through decision_schema.py
    tts.py             ElevenLabs text-to-speech client
    store.py           its own sqlite3 connection/DDL for a `voice_turns` table —
                       same JARVIS_DB_PATH file, but never touches db.py's tables
    session.py         VoiceSession orchestrator — state machine, interrupt/barge-in
    ui.py              PyQt6 window: transcript, state indicator, push-to-talk, stop
docs/
  jarvis_architecture.md  Claude's system prompt for B.2 reasoning — see its own header
tests/
  test_auth.py                  token storage
  test_config.py                 required-vs-defaulted config, incl. the interval rule above
  test_schema.py                  response-shape parsing, malformed inputs
  test_client_resilience.py      retry/backoff/auth-error behavior against a real local server
  test_poller.py                  loop survives failing ticks, stops promptly
  test_jarvis_config.py           required-vs-defaulted B.2 config
  test_decision_schema.py         structured-decision validation, malformed inputs
  test_db.py                      decision log CRUD, idempotency constraint, approvals
  test_claude_reasoner.py         Claude integration against a fake client (never the real API)
  test_marketing_client.py        retry/backoff/auth-error behavior against a real local server
  test_engine.py                  all four decision paths + every failure mode (see below)
  test_dashboard_model.py          dashboard.py's data/action layer, incl. CLI-vs-dashboard equivalence
  test_voice_stt.py                misfire resilience (silence/noise never reaches Claude)
  test_voice_tts.py                ElevenLabs client, incl. a real local HTTP server smoke test
  test_voice_reasoner.py           conversational Claude path, mocked, no tool_choice
  test_voice_store.py              voice_turns table, incl. same-DB-file isolation from decisions/approvals
  test_voice_auth.py               ElevenLabs key storage (set/show only)
  test_voice_session.py            orchestration: happy path, every failure mode, interrupt/barge-in
requirements.txt
.env.example
```

## Setup

```
pip install -r requirements.txt
```

### 1. Generate the JARVIS-only bearer token

This service and EYV need to agree on one shared bearer token for the
`/jarvis/*` endpoints. Since there's no third party minting it, generate
it locally:

```
python -m eyv_poller.auth generate
```

This creates a cryptographically random token, stores it at
`~/.jarvis/eyv_poller/queue_token.txt` with `0600` permissions (owner
read/write only), and **prints it exactly once**. Copy that value
immediately — nothing else in this codebase will show it to you again
except `python -m eyv_poller.auth show` (see below).

If instead the token is going to originate on EYV's side, use:

```
python -m eyv_poller.auth set
```

which prompts via `getpass` (input hidden, not echoed to the terminal)
for a token to paste in.

**Manual transfer to EYV (Railway) — do this carefully:**

1. Read the token from the local file — either the one-time output of
   `generate`, or later via `python -m eyv_poller.auth show`.
2. Paste it directly into Railway's environment configuration for EYV
   (as whatever env var EYV's `/jarvis/*` auth checks against).
3. Don't create intermediate copies: no pasting into a scratch file, no
   putting it in shell history (avoid typing it as a bare command-line
   argument), no committing it anywhere, no writing it into a doc.

The token is never hard-coded, never logged, and never committed — it
lives only in that one local file and in Railway's own secret storage.

### 2. Configure the service

Copy `.env.example` to `.env` (or set real environment variables) and
fill in at minimum:

```
EYV_BASE_URL=https://<your-eyv-railway-app>.up.railway.app
EYV_POLL_INTERVAL_SEC=<confirmed interval — see "Open decision" above>
```

All other settings have defaults — see `.env.example` for the full list
(`EYV_QUEUE_PATH`, `EYV_TOKEN_PATH`, `EYV_REQUEST_TIMEOUT_SEC`,
`EYV_MAX_RETRIES`, `EYV_RETRY_BACKOFF_BASE_SEC`, `EYV_LOG_DIR`,
`EYV_LOG_LEVEL`).

### 3. Run it

```
python -m eyv_poller
```

Logs go to stdout and to a rotating file under `EYV_LOG_DIR` (default
`~/.jarvis/eyv_poller/logs/eyv_poller.log`). Stop with Ctrl+C, or send
`SIGTERM` — both trigger a clean shutdown after the in-flight poll
finishes.

## Polling flow

1. `main.py` loads config, sets up logging, and confirms a token is
   stored (refuses to start otherwise, with a clear message pointing at
   `auth generate`/`auth set`).
2. `poller.PollerLoop.run()` loops: poll, sleep for
   `EYV_POLL_INTERVAL_SEC` (interruptible — shutdown doesn't wait out a
   long sleep), repeat.
3. Each poll calls `client.EYVQueueClient.fetch_queue()`, which:
   - Reads the current token fresh off disk (so a rotated token takes
     effect on the very next poll, no restart needed) and attaches it
     as `Authorization: Bearer <token>`.
   - Retries connection errors, timeouts, and 5xx responses up to
     `EYV_MAX_RETRIES` times with exponential backoff
     (`EYV_RETRY_BACKOFF_BASE_SEC * 2^attempt`). Does **not** retry 401/
     403 (bad token) or other 4xx (e.g. wrong path) — those are
     configuration problems, not transient ones.
   - Parses the JSON body via `schema.parse_queue_response()`, isolated
     specifically so the eventual real schema only touches this one
     function.
4. On success, `poller.py` hands the parsed `QueueSnapshot` to a
   `process` callback — by default `processor.process_queue_snapshot()`,
   which only logs the item count and ids. `jarvis_core` (B.2, below)
   reuses this same `PollerLoop` but swaps in its own callback
   (`JarvisEngine.process_queue_snapshot`) instead of running alongside
   `processor.py` — see jarvis_core/main.py.
5. Any failure at any of these steps (network, auth, malformed
   response, or an exception from the processor itself) is caught,
   logged locally, and the loop continues to the next tick — one bad
   poll never terminates the process.

## Logging

Every poll attempt, success (with item count), retry, and failure is
logged locally (console + rotating file). Bearer tokens and
`Authorization` headers are never included in any log line — call sites
in this codebase pass counts, status codes, and URLs, never headers or
credentials.

## Testing

```
python -m unittest discover -s tests -v
```

Following this codebase's existing convention (see JARVIS-XL's
`tests/test_hermes_client.py`) of exercising real components over
mocks where practical:

- `test_client_resilience.py` runs a real `ThreadingHTTPServer` on
  `127.0.0.1` and drives actual retry/backoff, timeout, and auth-error
  behavior over real sockets — including a genuine unreachable-port
  connection failure.
- `test_auth.py` writes to a real temp-directory file and checks the
  actual permission bits (skipped on Windows, where POSIX bits aren't
  meaningful).
- `test_poller.py` and `test_config.py` use small hand-written stubs
  (not a mocking framework) for the one or two collaborators each
  doesn't need to be real to test its own contract.

---

# jarvis_core — B.2 JARVIS Reasoning Core

A standalone process that consumes EYV queue items (reusing B.1's
polling wholesale), reasons over each one using Claude, determines the
required action, executes or delegates that action, and records every
decision durably in a local SQLite database.

## How it fits together

```
eyv_poller.client.EYVQueueClient  (B.1, unmodified)
              |
eyv_poller.poller.PollerLoop      (B.1, unmodified — just given a
              |                    different `process` callback)
              v
jarvis_core.engine.JarvisEngine.process_queue_snapshot
              |
              v
jarvis_core.claude_reasoner.ClaudeReasoner   -- Claude API, forced tool call
              |
              v
      structured decision (jarvis_core.decision_schema.ReasoningDecision)
              |
     +--------+--------+--------------------+
     |        |        |                    |
NEEDS_CODE  MARKETING  RESOLVED         NEEDS_APPROVAL
     |      _ACTION       |                    |
     v        v           v                    v
CodeInterface  POST     mark row       ApprovalInterface
(B.3 seam)   /jarvis/   resolved       (B.4 seam) -> local
              decisions                 pending approval;
              (Bob)                     STOPS here until an
                                         operator resolves it
              |
              v
   jarvis_core.db.DecisionStore (SQLite) records every step of the
   above, for every item, regardless of outcome.
```

## Running it

```
pip install -r requirements.txt
python -m eyv_poller.auth generate   # if not already done for B.1
```

Set the required environment variables (see `.env.example`):

```
EYV_BASE_URL=https://<your-eyv-railway-app>.up.railway.app
EYV_POLL_INTERVAL_SEC=<confirmed interval — see B.1's "Open decision">
ANTHROPIC_API_KEY=sk-ant-...
```

Everything else has a default — see `.env.example` for the full list
(`JARVIS_ARCHITECTURE_DOC_PATH`, `JARVIS_CLAUDE_MODEL`,
`JARVIS_CLAUDE_MAX_TOKENS`, `JARVIS_CONFIDENCE_THRESHOLD`,
`JARVIS_DB_PATH`, `JARVIS_DECISIONS_PATH`, `JARVIS_REQUEST_TIMEOUT_SEC`,
`JARVIS_MAX_RETRIES`, `JARVIS_RETRY_BACKOFF_BASE_SEC`, `JARVIS_LOG_DIR`,
`JARVIS_LOG_LEVEL`). Then:

```
python -m jarvis_core
```

Logs go to stdout and to `EYV_LOG_DIR`'s sibling
`~/.jarvis/jarvis_core/logs/jarvis_core.log` by default. The SQLite
decision log lives at `~/.jarvis/jarvis_core/jarvis.db` by default and
is created (schema included) automatically on first run — there's no
separate migration step to run; see `db.py`'s module docstring for why
`CREATE TABLE IF NOT EXISTS` is sufficient here. Stop with Ctrl+C or
`SIGTERM`, same clean-shutdown behavior as B.1.

### Resolving a pending approval

`NEEDS_APPROVAL` decisions (including ones the confidence gate forced
into this path) stop at a locally-recorded approval and are never
auto-executed. Resolve them either from the terminal:

```
python -m jarvis_core.approvals_cli list
python -m jarvis_core.approvals_cli approve <approval_id>
python -m jarvis_core.approvals_cli reject <approval_id>
```

or from the dashboard (below). An approved decision resumes
automatically on the next poll (`JarvisEngine.resume_waiting_approvals()`,
called before any new items are processed each cycle) — resolving it
here or in the dashboard only flips the approval's own status; neither
front end executes anything itself.

## Dashboard — B.4 Approval UI

A local PyQt6 GUI over the same decision database and approval
interface `approvals_cli.py` uses — a viewer for recent decisions plus
an approve/reject front end for pending approvals. Runs as a completely
separate process from `python -m jarvis_core`'s polling/reasoning loop
(same separation JARVIS-XL keeps between its UI and its background
execution) — start/stop it independently, any time, without affecting
the main process.

```
pip install -r requirements.txt   # includes PyQt6
python -m jarvis_core.dashboard
```

It only needs `JARVIS_DB_PATH` (or its default,
`~/.jarvis/jarvis_core/jarvis.db`) — no `ANTHROPIC_API_KEY` and no
architecture document required, since it never calls Claude.

**Decisions tab** — the `decisions` table, most recent first, refreshed
on a timer (every 3s by default). Rows whose status is `in_progress`,
`waiting_approval`, or `failed` are highlighted, since those are the
ones worth an operator's attention; `in_progress` in particular covers
both a genuinely mid-flight item in a live `jarvis_core` process and a
crash orphan (see `engine.py`'s `recover_orphaned_decisions()`) — the
database doesn't carry a separate flag distinguishing the two, so the
dashboard (deliberately, to avoid touching `engine.py`'s own
bookkeeping) treats any persistent `in_progress` row as worth a look.

**Pending Approvals tab** — every approval still `status='pending'`,
with the originating queue item, JARVIS's full reasoning, its
confidence (looked up from the decision row the approval was created
for), and the proposed action. Approve/Reject buttons call
`ApprovalInterface.approve()`/`reject()` directly — the exact same
calls `approvals_cli.py` makes — so there is one place, not two, that
defines what "approve" or "reject" does to the database. Selecting a
row shows the full proposed-action and context JSON below the table.

**Read-only safety, by construction:** `dashboard.py` never imports
`claude_reasoner.py`, `code_interface.py`, `marketing_client.py`,
`engine.py`, or `eyv_poller` — there is no code path in this file that
can call the Claude API, execute a code change, POST to
`/jarvis/decisions`, or trigger a queue poll. The only writes it can
ever make are `approve()`/`reject()` on an *existing* pending approval;
`engine.py`'s decision/confidence logic is untouched and unreachable
from here.

## Voice — two-way voice interaction

A separate PyQt6 UI (`jarvis_core/voice/`) for talking to JARVIS out
loud: push-to-talk → local speech-to-text → a conversational Claude
call → ElevenLabs text-to-speech → playback, with a live transcript and
a state indicator (idle/listening/thinking/speaking). It does not touch
the dashboard, the decisions/approvals tables, or `engine.py`'s
routing/confidence logic — it's a third, independent process, run
alongside (not inside) `jarvis_core.main` and `jarvis_core.dashboard`.

**Why these choices** (see `jarvis_voice_ui_prompt.md`'s tradeoff
discussion for the full reasoning):

- **PyQt6**, matching `dashboard.py`'s stack — keeps mic capture, STT,
  TTS playback, and UI state in one process, which is what makes
  barge-in a direct function call instead of a network hop.
- **Local speech-to-text** via `faster-whisper` — no cloud STT
  credential/cost/network-failure surface to add, and voice audio never
  leaves the machine.
- **Push-to-talk** for v1 — simpler to ship than wake-word, and it
  solves most of the misfire-resilience requirement by construction:
  only audio the user deliberately captured is ever transcribed.
- **A new `jarvis_core/voice/` subpackage** — the first subpackage in
  this codebase, since voice is a clearly separable concern with
  several new files; `jarvis_core/claude_reasoner.py`, `db.py`, and
  `engine.py`'s internals are untouched (voice_reasoner.py is a
  deliberately separate, non-tool-forced Claude call path; store.py is
  its own sqlite3 connection/table, not an extension of `db.py`).

### Setup

```
pip install -r requirements.txt   # includes faster-whisper, sounddevice
python -m jarvis_core.voice.auth set    # paste your ElevenLabs API key (hidden input)
```

Also requires `ANTHROPIC_API_KEY` (same one B.2 uses). Everything else
has a default — see `.env.example` for the full list
(`JARVIS_VOICE_CLAUDE_MODEL`, `JARVIS_VOICE_WHISPER_MODEL`,
`JARVIS_VOICE_ELEVENLABS_VOICE_ID`, etc.).

### Run

```
python -m jarvis_core.voice
```

Hold **Hold to Talk** to record, release to send. **Stop** interrupts
JARVIS mid-response (or discards an in-progress recording) — barge-in,
via a direct call into `AudioPlayer.stop()`/`MicRecorder.stop()`, not
true audio-level interruption. **New Session** clears the transcript
and conversation history (a fresh `session_id`) without restarting the
process or reloading the Whisper model.

### Misfire resilience

`stt.py`'s `WhisperTranscriber.transcribe()` returns `None` — never an
empty or garbage string — for audio too short to plausibly contain
speech, for a high Whisper no-speech probability, or for empty text
after stripping. `session.py` treats a `None` transcript as "nothing
happened," not an error: no Claude call, no TTS call, straight back to
idle. This is the mechanism, not a UI-level filter — it can't be
bypassed by a caller forgetting to check something.

### Transcript persistence

Every turn (both roles) is written to a `voice_turns` table via
`voice/store.py` — its own `sqlite3` connection and DDL, sharing
`JARVIS_DB_PATH`'s file but never `db.py`'s `DecisionStore` object or
its `decisions`/`approvals` tables. Raw audio is never persisted, only
transcript text, per this feature's scope. A write failure here is
logged (full traceback) but does not interrupt the live conversation —
see `session.py`'s module docstring.

### Testing

```
python -m unittest discover -s tests -v
```

Same rule as the rest of `jarvis_core`: Claude is mocked in every test
(`test_voice_reasoner.py`, hand-written fake client). `faster-whisper`
is also mocked (`test_voice_stt.py`, a fake model object — no real
model load or download in tests). `test_voice_tts.py` is the "at least
one real local smoke test path" this feature's spec asked for: a real
`ThreadingHTTPServer` standing in for ElevenLabs, exercising the same
retry/auth/timeout behavior as `test_client_resilience.py`/
`test_marketing_client.py` did for their own HTTP clients — the actual
network code, over real sockets, never mocked. `test_voice_session.py`
covers the full orchestration (happy path, every external failure mode,
interrupt/barge-in, and that a persistence failure never blocks a live
turn) against hand-written stubs for every collaborator.

## The decision schema (B.2.1)

Every queue item gets exactly one Claude API call. The architecture
document at `docs/jarvis_architecture.md` (or wherever
`JARVIS_ARCHITECTURE_DOC_PATH` points) is used verbatim as the system
prompt — see that file's own header for the assumption it stands in
for. Claude is forced (via `tool_choice`) to call a single
`record_decision` tool rather than asked to produce JSON in prose, so a
malformed response can only ever be a malformed *value* inside an
otherwise well-formed tool call — never free text needing to be parsed.
`jarvis_core.decision_schema.parse_decision()` validates that value
strictly; any failure raises `DecisionValidationError`, which
`engine.py` always turns into a `failed` decision row — never a silent
best-effort continuation.

```json
{
  "decision": "NEEDS_CODE",
  "reason": "Explanation of why this action is required",
  "action": {"type": "code", "instructions": "What Claude Code should implement"},
  "confidence": 0.95
}
```

**Low-confidence override:** any decision with `confidence` below
`JARVIS_CONFIDENCE_THRESHOLD` (default `0.7`) is forced to
`NEEDS_APPROVAL` regardless of what Claude originally classified it as
— the original decision and confidence are preserved in the recorded
`reason`, never dropped. `NEEDS_APPROVAL` decisions are never
downgraded by this gate (they're already the most conservative path).

## Decision routing (B.2.2)

| Decision | Route |
|---|---|
| `NEEDS_CODE` | `CodeInterface.submit()` — the B.3 seam (see its module docstring for the current implementation and its assumption) |
| `MARKETING_ACTION` | `MarketingDecisionClient.post_decision()` — `POST /jarvis/decisions`, retried on network/5xx, never on 4xx (see its module docstring for the endpoint-doesn't-exist-yet assumption) |
| `RESOLVED` | Decision row marked `resolved`. No external call. |
| `NEEDS_APPROVAL` | `ApprovalInterface.create_approval()` — the B.4 seam (see its module docstring — **this is not JARVIS-XL's `task_approval.py`**, a different component in a different codebase). Processing of that item stops until an operator resolves the approval. |

## Durable local state (B.2.3)

`jarvis_core/db.py` — plain stdlib SQLite, one `decisions` row per
queue item plus an `approvals` table for the B.4 placeholder. Answers
every question B.2.3 asks for: what did JARVIS decide
(`decision`/`reason`/`confidence`), which queue item caused it
(`queue_item_id`/`source_item`), when (`created_at`/`updated_at`/
`completed_at`), what action was attempted (`action_type`/
`action_payload`), did it succeed/fail/why (`status`/`outcome`/
`error`), and whether it's waiting for approval
(`status='waiting_approval'`/`approval_id`).

**Idempotency:** `queue_item_id` is `UNIQUE`. `engine.py` checks for an
existing row before reasoning over an item at all — an item that
already has a decision row is never reasoned over or routed again.

**Crash recovery:** a row left `status='in_progress'` means a prior
process crashed after routing began but before it finished — the
action may or may not have completed. `JarvisEngine.recover_orphaned_decisions()`
runs once at startup and logs each one distinctly (`ORPHANED DECISION`,
at warning level); it deliberately does **not** auto-retry or
auto-resolve these, since guessing either way risks exactly the
duplicate execution this requirement exists to prevent. An operator has
to look and update the row by hand.

## Error handling (reliability requirements)

Every category B.2 calls out is caught and recorded, never silently
swallowed:

- **Claude API failures / timeouts / rate limits** → `ClaudeReasonerError`, recorded as `failed`.
- **Malformed Claude responses** → `DecisionValidationError`, recorded as `failed`.
- **Queue-item failures** (any unexpected exception while processing one item) → caught per-item in `engine.py`, logged with a full traceback, recorded as `failed` where a row already exists; never allowed to stop the rest of the batch or the poll loop (B.1's own poller.py is the outer safety net; `engine.py` adds an inner one per item).
- **HTTP failures from `/jarvis/decisions`** → `MarketingClientError` (auth vs. unavailable, same retry policy as B.1's own EYV client), recorded as `failed`.
- **Claude Code (B.3) failures** → `ExecutionResult(success=False, ...)`, recorded as `failed` with the real failure detail — never reported as a success that didn't happen.
- **Approval creation failures (B.4)** → `ApprovalError`, recorded as `failed`.
- **SQLite failures** → `DecisionStoreError`, always logged with a full traceback; a failure to *write* an outcome is logged but never crashes the process (the action already happened or didn't — losing the write is a visibility gap to fix, not a reason to also lose the poll loop).

## Assumptions and gaps (dependent components not yet available)

Reported per this task's own closing instruction — these are real
limitations of what B.2 alone can deliver, not implementation bugs:

1. **No prior JARVIS architecture document existed** anywhere in this
   repo, the EYV website repo, or elsewhere on disk. `docs/jarvis_architecture.md`
   was authored as the smallest artifact needed to satisfy B.2.1's "use
   the previously created JARVIS architecture document as the system
   prompt" — see that file's own header. If a fuller one exists or gets
   written elsewhere, point `JARVIS_ARCHITECTURE_DOC_PATH` at it.
2. **`POST /jarvis/decisions` does not exist on EYV's backend yet** — as
   of this writing, `eyv-website-main/backend/internal_jarvis_api.py`
   only implements `GET /queue`. `marketing_client.py` is built against
   the contract B.2.2 itself describes and assumes the real endpoint
   will sit under the same `/jarvis/*` prefix and bearer-token gate as
   `/jarvis/queue` — untestable end-to-end until that endpoint exists.
3. **B.3 (Claude Code as its own component) does not exist yet** —
   `code_interface.py` shells out to the local `claude` CLI in headless
   mode as a real, working stand-in. When a dedicated B.3 component
   exists, only that one file needs to change.
4. **B.4 (a real approval service) does not exist yet** —
   `approval_interface.py` persists approvals in this process's own
   SQLite database and ships a minimal CLI (`approvals_cli.py`) for a
   human to resolve them. This is explicitly not JARVIS-XL's
   `task_approval.py`, which is a different approval model for a
   different, unrelated codebase.
5. **The queue is scoped to code + marketing work only** (EYV tickets
   are `kind: "bug"|"feature"`, per `internal_jarvis_api.py`'s own
   schema decision) — the decision schema therefore only routes to
   `NEEDS_CODE`/`MARKETING_ACTION`/`RESOLVED`/`NEEDS_APPROVAL`, per
   B.2.2's spec. Support and analytics tickets are handled by EYV's
   existing human ticket-triage pipeline and are not expected to reach
   this queue; `docs/jarvis_architecture.md` tells Claude this
   explicitly so it doesn't try to invent a `SUPPORT_ACTION`/
   `ANALYTICS_ACTION` routing path that has no destination. If that
   scoping assumption changes, both the architecture doc and the
   decision schema would need new decision types and new routing.

## Testing

```
python -m unittest discover -s tests -v
```

Same conventions as B.1's own tests (see above), plus one added,
explicit rule for this codebase: **the Claude API is mocked in every
test, with a hand-written fake client — never a real API call.** See
`test_claude_reasoner.py`'s module docstring. `test_marketing_client.py`
follows B.1's `test_client_resilience.py` pattern exactly (a real local
`ThreadingHTTPServer`, not a mocked `requests`) since that's a plain
HTTP client with no LLM involved. `test_engine.py` exercises all four
decision paths (`NEEDS_CODE`, `MARKETING_ACTION`, `RESOLVED`,
`NEEDS_APPROVAL`) plus malformed-response handling, every external
failure mode, the low-confidence override, idempotency, and orphan
recovery — all against hand-written stubs for Claude/B.3/Bob/B.4, per
the rule above.
