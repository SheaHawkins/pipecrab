# End-to-end voice agent on a ZeroClaw daemon

The [local voice pipeline](../e2e-voice-agent) with a **ZeroClaw daemon as the
brain**. The pipeline is an RPC peer of `zeroclaw daemon`: each utterance
becomes a `session/prompt`, streamed reply chunks become speech, and the
conversation is a first-class daemon session — open ZeroClaw's TUI against the
same daemon and watch the transcript update turn by turn.

```text
CpalSource (mic)
    ▼
ResamplerStage (16 kHz mono)
    ▼
VadStage<SherpaVad> ──▶ SttStage<OfflineSherpaStt> ──▶ UserTurnGate
    ▼
DispatchIngress<ZeroclawDelegateSource>   (delegation results re-enter here)
    ▼
LmStage<ZeroclawLm>                       (no tools: the daemon owns them)
    ▼
ToolCallEcho ──▶ DispatchEcho
    ▼
SentenceChunker ──▶ AgentEcho ──▶ TtsStage<SherpaTts>
    ▼
ResamplerStage (device rate) ──▶ CpalSink (speaker)
```

Unlike [`e2e-voice-agent-dispatch`](../e2e-voice-agent-dispatch), there is no
dispatch egress and no local GGUF: tool calls never leave the pipeline. The
daemon executes tools inline during the turn, long work goes through
ZeroClaw's `delegate` tool with `background: true`, and the
`ZeroclawDelegateSource` watches the session workspace's `delegate_results/`
so a finished task re-enters as a `[dispatch/completion]` turn the agent
speaks — minutes later, mid-conversation, without blocking anything.

**Use headphones** — over speakers the microphone re-captures the agent's own
voice and it talks to itself.

## Requirements

- Rust 1.88 or newer; macOS or Linux (unix domain sockets) with a working
  microphone and output device.
- The VAD, ASR, and TTS models from
  [`e2e-voice-agent`](../e2e-voice-agent#download-the-models) — follow that
  README's download steps first. No GGUF is needed here; the language model
  lives behind the daemon's provider.
- A ZeroClaw checkout or installed `zeroclaw` binary (0.7 or newer — the
  daemon must speak the `session/*` RPC protocol).
- A model-provider credential the daemon can use (the walkthrough below uses
  OpenRouter).

## Step 1 — install ZeroClaw

From a checkout:

```console
cd ../zeroclaw          # wherever your zeroclaw checkout lives
cargo install --path .
```

If this is a brand-new install, run `zeroclaw` once and let quickstart create
`~/.zeroclaw/config.toml`, then layer the config below on top. (The daemon
refuses to serve until setup has completed at least once.)

## Step 2 — configure a voice-friendly agent

The agent profile is part of this example's contract: it decides streaming,
tool latency, and whether background delegation works at all. Add the
following to `~/.zeroclaw/config.toml` (or configure the same thing through
the gateway dashboard / zerocode):

```toml
# ── The models ──────────────────────────────────────────────────────────
# Both must stream WITH tool events, or every reply arrives as one buffered
# chunk and time-to-first-audio dies. OpenRouter, Anthropic, and OpenAI
# qualify, as does an OpenAI-compatible local server (llamacpp / lmstudio /
# vllm slots) running a model with native tool calling. Native `ollama`
# does NOT stream — do not use it here.
#
# Two aliases because the two agents have opposite needs: the voice agent
# is on the latency path of every spoken reply, so it gets a fast model;
# the worker runs in the background where nobody is waiting on first-token
# latency, so it gets a smart one.
[providers.models.openrouter.fast]
api_key = "sk-or-…"
model   = "anthropic/claude-haiku-4.5"

[providers.models.openrouter.smart]
api_key = "sk-or-…"
model   = "anthropic/claude-sonnet-4.5"

# ── The agent the pipeline talks to ────────────────────────────────────
[agents.voice]
model_provider  = "openrouter.fast"
risk_profile    = "voice"
runtime_profile = "voice"
# `independent` is load-bearing: a `bounded` delegate's tools are capped by
# the CALLER's registry, and the voice agent's registry is delegate-only —
# a bounded worker would arrive with no tools at all.
delegates = [ { agent = "research", mode = "independent" } ]

# ── The background worker it delegates to ──────────────────────────────
[agents.research]
model_provider  = "openrouter.smart"
risk_profile    = "research"
runtime_profile = "research"

# ── Risk profiles ───────────────────────────────────────────────────────
[risk_profiles.voice]
# "full" = autonomous within policy bounds. A voice loop cannot answer an
# approval prompt; anything gated on approval stalls the turn to timeout.
level             = "full"
# Inline tools stall speech for their full duration, so the speaking agent
# gets exactly one: delegate. (A non-empty list also keeps spawn_subagent
# out, which would block the whole turn.)
allowed_tools     = ["delegate"]
delegation_policy = { mode = "allow" }

[risk_profiles.research]
level         = "full"
allowed_tools = ["web_search_tool", "web_fetch", "http_request", "file_write"]

# ── Runtime profiles ────────────────────────────────────────────────────
[runtime_profiles.voice]
agentic              = true
max_tool_iterations  = 3
max_delegation_depth = 1

[runtime_profiles.research]
agentic             = true
max_tool_iterations = 10
```

Some research tools need their own credentials (a search-provider key for
`web_search_tool`, for example); see ZeroClaw's tools documentation. The
worker still functions without them via `web_fetch`/`http_request`.

### The persona

ZeroClaw injects workspace identity files (`SOUL.md`, `IDENTITY.md`, …) into
the system prompt. Create the voice agent's:

```console
mkdir -p ~/.zeroclaw/agents/voice/workspace
cat > ~/.zeroclaw/agents/voice/workspace/SOUL.md <<'EOF'
You are a friendly voice assistant. Every word you write is spoken aloud,
so answer in one or two short sentences of plain prose — no markup, no
lists, no code, and no emoji or other symbols that cannot be spoken.

Answer from your own knowledge whatever you can. Anything that needs the
web, the world outside this machine, or more than a few seconds of work is
an errand: call `delegate` with agent "research" and background=true.

Acknowledge an errand exactly once. Call the tool immediately, writing no
text in the same step as the tool call — do not announce what you are
about to do. After the tool returns, say in one short sentence that the
errand is underway, then stop. Never confirm the same errand twice, and
never run long work inline.

Messages arriving in the form `[dispatch/completion] task <id> …` are
background-task results, not user speech. Relay the result in the same
short spoken style. `[dispatch/failure]` means the task failed; say so
briefly.
EOF
```

## Step 3 — start the daemon

```console
zeroclaw daemon
```

The daemon binds its RPC socket at `~/.zeroclaw/data/daemon.sock` (or
`$ZEROCLAW_SOCKET` if set). The example resolves the same path automatically;
pass `--zeroclaw-socket <path>` only if yours lives elsewhere.

## Step 4 — run

With the speech models downloaded per
[`e2e-voice-agent`](../e2e-voice-agent#download-the-models):

```console
ASR=./models/sherpa-onnx-moonshine-base-en-quantized-2026-02-27
TTS=./models/kokoro-en-v0_19

cargo run -p e2e-voice-agent-zeroclaw -- \
  --vad-model ./models/silero_vad.onnx \
  --encoder "$ASR/encoder_model.ort" \
  --merged-decoder "$ASR/decoder_model_merged.ort" \
  --tokens "$ASR/tokens.txt" \
  --tts-model "$TTS/model.onnx" \
  --tts-voices "$TTS/voices.bin" \
  --tts-tokens "$TTS/tokens.txt" \
  --tts-data-dir "$TTS/espeak-ng-data" \
  --agent voice
```

Add `--session-id my-kitchen` to keep one durable conversation across
restarts; without it a fresh `pc-voice-…` session is minted per run.

## Step 5 — talk

Say **"what's the capital of Australia?"** — a plain turn, answered and
spoken as it streams. Then say **"find me a good crab cake recipe"**:

```text
e2e-voice-agent-zeroclaw: agent "voice", session pc-voice-8edbbeeb… — open the
    ZeroClaw TUI to watch this conversation
e2e-voice-agent-zeroclaw: watching ~/.zeroclaw/agents/voice/workspace/delegate_results
    for background delegations
SpeechStarted
SpeechStopped (2.10 s)
You: find me a good crab cake recipe
[tool] delegate {"agent":"research","prompt":"find a good crab cake recipe","background":true}
Agent: I've sent that off — I'll let you know when it's back.
  ⏱ first speech 1.42 s after silence
[task] completed: task 4f6f…-… (agent research): Here's a classic crab cake recipe …
Agent: That recipe came back: lump crab, a light bread-crumb bind, …
```

Both `Agent:` lines are spoken. The second one arrives whenever the worker
finishes — the conversation stays usable in between, so ask something else
while it runs.

## Step 6 — watch it in the TUI

Open ZeroClaw's TUI against the same daemon and the session from the startup
line (`pc-voice-…`, or your `--session-id`) is in the session list like any
other conversation — full transcript, delegations included. It updates at
**turn granularity**: the daemon streams tokens only to the connection that
prompted, so live typing in the TUI while voice drives is a daemon-side
fanout feature that does not exist yet. Typing into the session *from* the
TUI works, and its replies stay in the TUI (they are not spoken) — the right
default for a text interjection.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `dialing the zeroclaw daemon socket failed` | The daemon is not running, or its socket is not at `~/.zeroclaw/data/daemon.sock`. Start `zeroclaw daemon`; pass `--zeroclaw-socket` or set `$ZEROCLAW_SOCKET` if relocated. |
| `zeroclaw session bootstrap failed: … unknown agent alias` | `--agent` does not match an `[agents.<alias>]` block in the daemon's config. |
| `turn was not streamed by the provider` on every turn | The agent's provider fails the streaming gate (streaming + tool events). Switch to OpenRouter/Anthropic/OpenAI or an OpenAI-compatible local server with native tool calling; native `ollama` never streams. Replies still work — they just arrive as one buffered delta. |
| `approval_request for tool …` warnings and long stalls | The risk profile routes a tool through approval, which a voice loop cannot answer. Use `level = "full"` and keep gated tools out of `allowed_tools`. |
| The agent says it started a task but nothing ever comes back | Check `delegation_policy = { mode = "allow" }`, the `delegates` roster (with `mode = "independent"`), and that the model actually passed `background=true` (the `[tool]` line shows the arguments). Result files land in the workspace's `delegate_results/`; a task stuck at `running` is reported as failed after 15 minutes. |
| The agent narrates raw markup or reads lists aloud | That is the persona's job — tighten `SOUL.md`; the pipeline speaks every token the daemon streams. |
| The agent confirms the same errand twice | The model is acking on both sides of the tool round: once alongside the tool call, once after the result. The persona above pins one side down ("writing no text in the same step as the tool call … acknowledge exactly once"); if your model still pre-announces, strengthen that paragraph — this is prompt-shaped behavior, not pipeline ordering. |
| Stray emoji or symbols get "spoken" | Same category: the persona must forbid them ("no emoji or other symbols that cannot be spoken"); TTS pronounces or stumbles over whatever the model emits. |
| First words of a reply are clipped or delayed | The first sentence waits on `SentenceChunker` + TTS; the `⏱ first speech` line measures the whole path. A faster provider/model is the biggest lever. |

## Tests

The adapter's own tests need no daemon: `cargo test -p pipecrab-lm-zeroclaw`.
The wire-compatibility tripwire against a live daemon is:

```console
ZEROCLAW_LIVE_AGENT=voice cargo test -p pipecrab-lm-zeroclaw --test live -- --ignored
```

Run it after upgrading ZeroClaw — the RPC protocol is mirrored, not imported,
so this is where drift shows up.

## Flags

| Flag | Meaning |
|---|---|
| `--agent <alias>` | ZeroClaw agent alias to run the session as (required) |
| `--zeroclaw-socket <path>` | Daemon RPC socket (default: `$ZEROCLAW_SOCKET`, else `~/.zeroclaw/data/daemon.sock`) |
| `--session-id <id>` | Stable session id to reattach across runs |
| `--vad-model`, `--encoder`, `--merged-decoder`, `--tokens` | VAD + ASR models |
| `--tts-model`, `--tts-voices`, `--tts-tokens`, `--tts-data-dir` | Kokoro TTS assets |
| `--speaker <n>`, `--speed <x>` | TTS voice and rate |
| `--stt-threads <n>` | ASR compute threads (default 2) |
| `--seconds <n>` | Stop after n seconds instead of Ctrl-C |
