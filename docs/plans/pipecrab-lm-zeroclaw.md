# ZeroClaw daemon LM adapter

## Goal

Add `pipecrab-lm-zeroclaw`, a native adapter that implements
`pipecrab_lm::LanguageModel` as a JSON-RPC peer of a running ZeroClaw daemon,
so a ZeroClaw agent stands in as the pipeline's LM stage and the voice
pipeline becomes just another native interaction mechanism in ZeroClaw.

The conversation is a first-class daemon session. The ZeroClaw TUI — another
RPC peer of the same daemon — can list it, open it, and read its transcript;
the voice pipeline and the TUI are two clients of one conversation.

Tool calling is internalized: the daemon's agent loop executes tools inline
during a turn, and asynchronous work goes through ZeroClaw's `delegate` tool
with `background: true` — its delegation system — instead of PipeCrab's
`dispatch_task`/`update_task` round trip. `DispatchEgress` and `DispatchSink`
are not used. `DispatchIngress` survives as the re-entry path: a new
`ZeroclawDelegateSource` watches the session workspace's delegation results
and emits `DispatchEvent`s so a finished task wakes the voice loop.

This supersedes `pipecrab-dispatch-zeroclaw` for the voice topology. The
webhook adapter remains the right tool when ZeroClaw is a remote gateway
behind some other LM.

## Background

Two integration shapes were considered. Embedding a ZeroClaw `Agent` in the
pipeline process was rejected: the conversation would live only in PipeCrab's
process memory, invisible to any daemon and therefore to the TUI, and it would
drag ZeroClaw's entire native tree in as path dependencies on a
`publish = false` workspace. Speaking the daemon's RPC protocol instead makes
the conversation durable and shared, and requires **no ZeroClaw crate
dependency at all** — the protocol is newline-delimited JSON-RPC 2.0 over a
local socket.

Verified protocol facts (ZeroClaw `crates/zeroclaw-runtime/src/rpc/`):

- Transport: unix domain socket, path from `$ZEROCLAW_SOCKET` or the default
  under the daemon's data dir; one JSON-RPC message per line.
- `initialize { protocol_version: 1, env?, clientCapabilities?, tui_id?,
  tui_sig? }` → `{ protocol_version, server_version, tui_id, .. }`. The
  daemon registers the client in its TUI registry; the returned id and HMAC
  signature authenticate reconnects.
- `session/new { agent_alias, cwd?, session_id?, exclude_memory?, chat_mode }`
  → `{ session_id, agent_alias, message_count, workspace_dir }`. The caller
  may supply a stable `session_id` to reattach; a nonzero `message_count`
  signals rehydration. `workspace_dir` locates the delegation results.
- `session/prompt { session_id, prompt }` runs one full agent turn. Progress
  streams back as `session/update` notifications on the issuing connection;
  the terminal event is `turn_complete { outcome, content }` with outcome
  `completed | cancelled | failed`.
- `session/cancel { session_id }` aborts the in-flight turn; the partial
  assistant text is preserved in session history.
- `session/update` events are a tagged enum (`"type"`, snake_case):
  `agent_message_chunk`, `agent_thought_chunk`, `tool_call`, `tool_result`,
  `approval_request`, `context_usage`, `plan`, `turn_complete`,
  `history_trimmed`.

Streaming granularity is inherited from the session's provider. The daemon
streams chunk events only when the active provider supports streaming — and,
with tools registered, streaming tool events. OpenRouter, Anthropic, OpenAI,
and the OpenAI-compatible family (llama.cpp server, LM Studio, vLLM) with
native tool calling all qualify; native `ollama` does not. A non-qualifying
profile still works but delivers the whole reply as one terminal chunk, which
destroys time-to-first-audio.

## Scope

This change includes:

- A publishable `crates/adapters/pipecrab-lm-zeroclaw` crate, native-only,
  with no ZeroClaw crate dependencies.
- Hand-mirrored serde types for the protocol subset above.
- `ZeroclawLm`, a connection-actor handle implementing `LanguageModel`.
- `ZeroclawDelegateSource`, a `DispatchSource` that discovers background
  delegation results by polling `{workspace_dir}/delegate_results/*.json`.
- A `connect` constructor performing the handshake and session bootstrap and
  returning both halves wired to one runtime and one turn-settled notifier.
- Deterministic tests against an in-process mock daemon and temp-dir result
  files.
- An ignored integration test against a live daemon.
- An `examples/e2e-voice-agent-zeroclaw` app modeled on
  `e2e-voice-agent-dispatch` with the egress stage removed.

This change does not include:

- Any change to `pipecrab-lm`, `pipecrab-dispatch`, or `LmStage`.
- Live token-level mirroring of voice turns in the TUI. Today the daemon
  streams `session/update` only to the connection that issued the prompt; the
  TUI sees the voice session at turn granularity via `session/messages`. The
  enabling change is a daemon-side per-session fanout (see Deferred).
- ZeroClaw-side changes of any kind (delegation push hook, update fanout,
  approval routing).
- Mid-turn steering (the RPC surface has no steering method; barge-in is
  cancel-and-reprompt).
- `Accepted` dispatch events (requires correlating `tool_call` update events
  with minted task ids; deferred).
- The wss transport for a remote daemon, and Windows named pipes; the local
  unix socket comes first.
- A native voice channel inside the daemon (option B in the design
  discussion; a later consolidation this adapter does not preclude).
- WebAssembly support.

## Repository placement

The adapter belongs under `crates/adapters` and declares the adapter layer:

```toml
[package.metadata.pipecrab]
layer = "adapter"
```

The expected files are:

```text
crates/adapters/pipecrab-lm-zeroclaw/
├── Cargo.toml
├── src/
│   ├── client.rs      # line-framed JSON-RPC over UnixStream, id matching
│   ├── config.rs
│   ├── lib.rs
│   ├── protocol.rs    # mirrored wire types
│   ├── render.rs      # last-message extraction and event rendering
│   ├── source.rs      # ZeroclawDelegateSource + poller
│   └── worker.rs      # connection actor, turn routing, cancellation
└── tests/
    ├── model.rs
    └── source.rs
```

Dependencies: `pipecrab-core`, `pipecrab-lm`, `pipecrab-dispatch` (for
`DispatchSource`), `tokio` (net, io-util, sync, time, rt), `serde`,
`serde_json`, `futures`, `async-trait`. Because the wire protocol replaces
the library dependency, the crate is publishable like the other adapters; add
it to `[workspace.dependencies]` with `path` + `version`. It stays out of the
wasm CI matrix, like the other native adapters.

Mirroring the protocol instead of importing it trades compile-time coupling
for wire-compatibility risk; the ignored integration test is the tripwire for
drift, and the mirrored subset is small (two requests, one notification enum).

## Pipeline topology

```text
ResamplerStage(16 kHz)
VadStage
SttStage
UserTurnGate                          (drops empty finals — required; the
                                       daemon rejects blank prompts)
DispatchIngress<ZeroclawDelegateSource>
LmStage<ZeroclawLm>                   (constructed WITHOUT tools)
SentenceChunker
TtsStage
ResamplerStage(device rate)
```

No `DispatchEgress`, no `DispatchSink`. `ModelFrame::ToolCall` frames emitted
for observability flow downstream untranslated.

Alongside the pipeline: `zeroclaw daemon` runs independently, and the ZeroClaw
TUI attaches to the same daemon whenever the user wants to watch or type into
the conversation.

## Public API

The crate exports:

```rust
pub struct ZeroclawLm;                 // Clone + Send + Sync handle
pub struct ZeroclawDelegateSource;     // implements DispatchSource
pub struct ZeroclawLmConfig;
pub struct PollConfig;
pub enum ZeroclawLmBuildError;

pub async fn connect(
    config: ZeroclawLmConfig,
) -> Result<(ZeroclawLm, ZeroclawDelegateSource), ZeroclawLmBuildError>;
```

`connect` dials the socket, performs `initialize`, issues `session/new`, and
returns both halves. The `SessionNewResult` supplies the `workspace_dir` the
poller watches — the pipeline never duplicates ZeroClaw workspace
configuration. `ZeroclawLmBuildError` distinguishes socket dial failure,
handshake or protocol errors, and an unknown agent alias.

## Configuration

```rust
pub struct ZeroclawLmConfig {
    pub socket_path: Option<PathBuf>,  // default: $ZEROCLAW_SOCKET, else the
                                       //   daemon's default endpoint path
    pub agent_alias: String,
    pub session_id: Option<Arc<str>>,  // stable id to reattach across runs;
                                       //   default: mint "pc-voice-{uuid}"
    pub exclude_memory: bool,          // default false
    pub surface_tool_calls: bool,      // default true
    pub poll: PollConfig,
}

pub struct PollConfig {
    pub interval: Duration,            // default 500 ms
    pub settle_backoff: Duration,      // default 2 s after 30 s pending
    pub stale_after: Duration,         // default 15 min
}
```

The agent profile is part of the configuration surface even though it lives
in ZeroClaw's own config:

- The provider must stream with tool events (see Background), or every reply
  arrives as one terminal chunk.
- The tool registry should be trimmed for voice: inline tools stall speech
  for their full duration; `spawn_subagent` blocks the whole turn and should
  be excluded; long work belongs in `delegate` with `background: true`.
- `delegation_policy` must allow the target aliases, and no registered tool
  may be approval-gated — an `approval_request` cannot be answered from the
  voice loop and stalls the turn until its timeout.
- The system prompt should explain that bracketed `[dispatch/completion]`
  messages are background-task results to relay, not user speech.

The poller requires the pipeline process to share a filesystem with the
daemon (it reads `workspace_dir` directly). That holds for the local-voice
topology this plan targets; a remote daemon needs the wss transport and a
different re-entry mechanism, both deferred.

## `ZeroclawLm`

### Connection actor

`ZeroclawLm` holds a command sender, a cancellation epoch, and a worker
handle. The worker is one named thread running a current-thread tokio runtime
that owns the `UnixStream`: it writes requests, matches responses by id,
routes `session/update` notifications for the bootstrapped session to the
active turn (notifications for other sessions are ignored), and fires the
turn-settled notifier shared with the poller after every terminal event.
Owning a private runtime keeps the pipeline free to be driven by any
executor. Dropping the last handle closes the socket and joins the thread.

Turns are strictly serialized: a `Generate` arriving while a cancelled turn
is still awaiting its `turn_complete { outcome: cancelled }` waits for that
terminal event first, so a barge-in followed by a fast next utterance cannot
interleave two prompts.

If the socket drops, the in-flight turn fails with a recoverable
`LmError::Engine`, and the worker reconnects with capped exponential backoff:
`initialize` with the saved `tui_id`/`tui_sig`, then `session/new` with the
same `session_id` — the daemon rehydrates the session from its store, so the
conversation survives both daemon and pipeline restarts.

### `generate`

`generate(&self, conversation, params, tools)`:

1. If `tools` is non-empty, return `Err(LmError::Engine(..))` — tool
   definitions are managed by ZeroClaw; `LmStage` must be constructed without
   tools. Failing loudly per generation beats silently ignoring a
   misconfigured pipeline.
2. Extract the **last** message of `conversation` and render it (see
   Conversation ownership). A whitespace-only rendering returns an
   immediately empty stream rather than tripping the daemon's blank-prompt
   rejection.
3. Snapshot the cancellation epoch, create the delta channel, send the turn
   command, and return the receiver boxed as the `ModelStream`.
4. `GenParams` is ignored (sampling and budgets are governed by the agent
   profile; `grammar` has no ZeroClaw equivalent). A non-default `GenParams`
   logs one warning per adapter lifetime.

`save_state` / `load_state` are `Ok` stubs — session durability is the
daemon's job.

### Event mapping

| `session/update` event | Mapping |
| --- | --- |
| `agent_message_chunk { text }` | `ModelDelta::Text(text)` |
| `agent_thought_chunk` | dropped — never spoken |
| `tool_call { tool_call_id, name, raw_input }` | `ModelDelta::tool_call(..)` when `surface_tool_calls` and `raw_input` is an object, else dropped |
| `tool_result` | dropped (internalized; logged at debug) |
| `plan` | dropped |
| `approval_request` | logged at warn — the profile must preclude these |
| `history_trimmed` | logged at warn |
| `context_usage` | logged at debug |
| `turn_complete { outcome: completed }` | end of stream |
| `turn_complete { outcome: cancelled }` | end of stream, silently |
| `turn_complete { outcome: failed, content }` | `Err(LmError::Engine(content))` |

Surfaced tool calls give downstream stages the hook the dispatch example uses
for "Let me check that…" filler audio. `LmStage` also records them in its own
conversation, which is dead weight (see below) but harmless.

A turn whose only text arrives in the terminal event (the non-streaming
provider case) logs one warning identifying the turn as non-streamed, so a
misconfigured profile is diagnosable rather than silently slow.

### Cancellation

`cancel()` is synchronous, idempotent, and non-blocking: it bumps the epoch
and nudges the worker over a watch channel; the worker fires
`session/cancel` and keeps draining until the terminal event arrives. Deltas
tagged with a stale epoch are discarded, so a cancel racing with in-flight
notifications cannot leak text into the next turn. `LmStage` calls `cancel()`
from `decide_system` on every `Interrupt`; the perform future is dropped by
the runtime, so send failures on the abandoned delta channel are ignored.

## Conversation ownership

The daemon session's history is the source of truth. `LmStage` still
accumulates its own `Conversation`, but the adapter reads only its final
message per generation:

| Last message | Rendering sent as the prompt |
| --- | --- |
| `Message::User { content }` | `content` as-is |
| `Message::Event { source, kind, content }` | `[{source}/{kind}] {content}` |
| `Message::ToolResult { name, content, .. }` | `[{name}] {content}` |
| anything else | `Err(LmError::Engine(..))` — protocol violation |

The `Event` arm is the delegation re-entry path: `DispatchIngress` projects a
`Completion` to `ModelInput::Respond(Event { source: "dispatch", kind:
"completion", .. })`, `LmStage` appends it and emits `Generate`, and the
rendering above becomes the next daemon turn — which the TUI sees as an
ordinary exchange in the session.

Two accepted divergences, both deliberate:

- On barge-in, PipeCrab's convention commits no assistant turn; the daemon
  keeps the partial text (`turn_complete { outcome: cancelled, content }`).
  The agent remembering what the user actually heard is the better behavior
  for voice, so the daemon's semantics win.
- `LmStage`'s conversation grows without ever being read beyond its tail. It
  is bounded dead weight, documented in the crate docs.

The daemon timestamps user messages and may inject a memory preamble; both
are accepted as-is.

## `ZeroclawDelegateSource`

### Contract

Implements `DispatchSource`: `next_event` is a channel `recv` —
cancellation-safe by construction, `None` on close — and `cancel()` stops the
poller and drops the sender. Idempotent, non-blocking.

### Discovery

Background delegations write `{workspace_dir}/delegate_results/{task_id}.json`
atomically (temp file + rename): a `running` record before the task spawns
and a terminal record when it settles:

```json
{
  "task_id": "…", "agent": "…",
  "status": "running | completed | failed | cancelled",
  "output": "…", "error": "…",
  "started_at": "rfc3339", "finished_at": "rfc3339"
}
```

The poller scans the directory rather than tracking ids from tool-call
events — scanning also catches tasks whose spawning turn was cancelled before
its `tool_result` update was observed. Filters:

- `started_at >= connect time` — files from earlier sessions are ignored.
- A seen-set keyed by `task_id` ensures each terminal state is emitted once.
- Unparseable files are skipped and retried next scan.

The daemon's own TUI/webhook/channel traffic can also create delegations in
the same workspace; the time filter does not distinguish provenance. A
workspace (or agent alias) dedicated to the voice session is recommended and
documented.

### Schedule

The poller sleeps until the turn-settled notifier fires, scans, and keeps
polling at `interval` only while `running` tasks are pending, relaxing to
`settle_backoff` after 30 seconds. Idle cost is zero.

### Event mapping

| Observed terminal state | Emitted `DispatchEvent` |
| --- | --- |
| `completed` | `Completion { task_id, message }` |
| `failed` | `Failure { task_id, message, retryable: false }` |
| `cancelled` | none — the agent cancelled it itself via `cancel_task` |

`message` is composed as `task {task_id} (agent {agent}): {output-or-error}`
because the ingress projection keeps only message text — the task id must
ride inside it for the agent to correlate against its own history.

### Staleness

The daemon's control plane records background tasks and its reaper recovers
crashed ones, but nothing guarantees the *result file* leaves `running` on
every failure path (kill, panic, failed terminal write). Any task still
non-terminal `stale_after` after first seen emits
`Failure { retryable: false }` with a message saying the task produced no
terminal status, and joins the seen-set. A later genuine terminal write for
that id is then ignored — acceptable, and noted in the docs.

## TUI visibility

What works the day this ships: the voice conversation appears in
`session/list` on any TUI attached to the daemon; opening it shows the full
transcript (`session/messages`), updating at turn granularity; typing into
the same session from the TUI is an ordinary `session/prompt` whose reply the
voice pipeline does **not** speak (updates stream to the prompting
connection), which is the correct default for a text interjection.

Live token-by-token mirroring of voice turns in the TUI requires the daemon
to fan `session/update` out to every connection subscribed to the session,
not only the prompting one. That is a contained ZeroClaw enhancement —
notification construction is already centralized — and is deferred.

## Tests

### Model tests (`tests/model.rs`)

Built against an in-process mock daemon: a unix-socket JSON-RPC server
scripted per test (handshake, canned `session/new`, then per-prompt
notification scripts):

- `connect` performs `initialize` + `session/new` and surfaces the returned
  `workspace_dir`; a JSON-RPC error on either fails construction.
- Streamed `agent_message_chunk`s arrive as incremental `ModelDelta::Text`.
- Thought, tool-result, plan, and usage events produce no deltas.
- `tool_call` surfaces as `ModelDelta::ToolCall` with id, name, and object
  arguments; non-object `raw_input` and `surface_tool_calls = false` suppress
  it.
- Updates for a foreign `session_id` are ignored.
- `turn_complete { completed }` ends the stream; `{ failed }` yields
  `Err(LmError::Engine)`; a terminal-only turn logs the non-streamed warning.
- Non-empty `tools` fails the generation with `LmError::Engine`.
- A whitespace-only rendered input yields an empty stream without a
  `session/prompt`.
- Last-message rendering: user text verbatim, dispatch event bracketed, a
  trailing assistant message is a protocol error.
- `cancel()` issues `session/cancel`, discards stale-epoch deltas, and a
  second `cancel()` is a no-op; a `generate` issued before the cancelled
  turn's terminal event waits for it.
- A dropped socket fails the in-flight turn recoverably; the actor
  reconnects, re-initializes with the saved identity, reissues `session/new`
  with the same session id, and the next turn succeeds.
- Dropping the last handle closes the socket and joins the worker thread.

### Source tests (`tests/source.rs`)

Temp-dir driven, no daemon required:

- A `running` file followed by an atomic rewrite to `completed` emits one
  `Completion` whose message contains the task id and agent alias.
- `failed` emits `Failure { retryable: false }`; `cancelled` emits nothing.
- Files with `started_at` before connect time are ignored.
- Each terminal state is emitted exactly once across repeated scans.
- Unparseable JSON is skipped, then picked up once valid.
- A file stuck at `running` past `stale_after` emits the staleness
  `Failure`; a terminal write afterwards is ignored.
- `cancel()` closes the event channel (`next_event` returns `Ok(None)`).

### Integration test

An ignored test reads a socket path and agent alias from environment
variables, connects to a live daemon, runs one turn, and — when the profile
allows delegation — requests a background task and drains a `Completion`
from the source. Default workspace validation requires neither a daemon nor
network access.

## Verification

Run:

```console
cargo fmt --all -- --check
cargo test -p pipecrab-lm-zeroclaw
cargo test -p pipecrab-arch --test layering
cargo clippy -p pipecrab-lm-zeroclaw --all-targets -- -D warnings
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

The layering gate must accept the crate as an ordinary adapter with no
exception. The wasm CI matrix is unchanged.

## Acceptance criteria

- `ZeroclawLm` implements `LanguageModel`; `LmStage<ZeroclawLm>` slots into
  the pipeline with no framework changes and no ZeroClaw crate dependency.
- The conversation is a daemon session: it survives pipeline and daemon
  restarts, appears in `session/list`, and its transcript is readable from
  the ZeroClaw TUI at turn granularity while voice is running.
- Provider streaming is inherited from the agent profile; a non-streaming
  profile degrades to one terminal delta and is logged, never silent.
- `cancel()` is synchronous and idempotent; barge-in cancels the daemon turn,
  preserves the partial in session history, and never blocks the pipeline
  thread; stale deltas cannot leak into the next turn.
- Tool calls surface as `ModelFrame::ToolCall` for observability only; no
  dispatch command ever leaves the pipeline.
- A background delegation's completion re-enters the conversation through
  `DispatchIngress`, triggers a spoken response, and lands in the session
  transcript the TUI reads, with the task id in the message text.
- Stuck-`running` tasks resolve to a `Failure` after `stale_after`; idle
  polling cost is zero.
- All non-integration tests run without a daemon, a provider, or network
  access.

## Deferred

- Daemon-side `session/update` fanout to all connections attached to a
  session, enabling live token mirroring in the TUI (ZeroClaw change).
- A delegation-completion push channel replacing the poller internals behind
  the unchanged `DispatchSource` (ZeroClaw change).
- Speaking TUI-initiated turns through the voice pipeline (needs the same
  fanout).
- `Accepted` dispatch events correlated from `tool_call` updates.
- The wss transport for a remote daemon, and Windows named-pipe support.
- A native voice channel inside the daemon (pipecrab as a ZeroClaw
  dependency) as a later consolidation.
