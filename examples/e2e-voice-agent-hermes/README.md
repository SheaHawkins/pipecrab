# End-to-end voice agent with Hermes dispatch

The [local voice agent](../e2e-voice-agent) plus a **Hermes Agent** transport
behind `pipecrab-dispatch`. Speak an errand and it runs as a Hermes run in the
background; the conversation continues meanwhile, and the agent speaks the
result whenever it lands.

```text
CpalSource (mic)
    │ device sample rate, mono
    ▼
ResamplerStage (16 kHz mono)
    ▼
VadStage<SherpaVad>
    ▼
SttStage<OfflineSherpaStt>
    ▼
UserTurnGate            (prints "You: …", drops empty finals)
    ▼
DispatchIngress         (injects Hermes events into the idle pipeline)
    ▼
LmStage<LlamaCpp>       (dispatch tools configured; speaks completions)
    ▼
DispatchEgress          (ToolCall → DispatchCommand → HermesSink → POST /v1/runs)
    ▼
DispatchAck             (speaks "Let me check the weather in Denver.")
    ▼
DispatchEcho            (prints "[task] …")
    ▼
SentenceChunker
    ▼
AgentEcho               (prints "Agent: …", forwards it)
    ▼
TtsStage<SherpaTts>
    ▼
ResamplerStage (device rate)
    ▼
CpalSink (speaker)
```

Ingress sits **above** the model so an event arriving while nothing is flowing
still reaches it; egress sits **below** so the model's tool calls leave through
the transport. The Hermes poller runs on its own tokio task, which is what lets
a task finishing during silence wake the pipeline and speak.

## The model decides

Nothing here recognises a trigger word. `LmStage::with_tools` hands the local
model `dispatch_task`, `pipecrab-lm-llamacpp` renders it into the prompt and
reads calls back out, and `DispatchEgress` translates the call into the Hermes
run. Whether an errand is worth delegating is the model's judgement, shaped by
the system prompt and the tool description.

`dispatch_task` is the only tool in scope. `pipecrab-dispatch` also defines
`update_task`, for following up on an accepted task, but a small model does not
reliably hold the two apart: given both, a 0.5B reaches for `update_task` and
invents a `task_id` rather than starting anything. Egress translates both
regardless — give a larger model `dispatch_tool_definitions()` and the pair is
back.

That needs a GGUF the adapter's default `ChatMlXml` dialect covers — Qwen 2.5/3
or Nous Hermes, which the `qwen3-4b-q4_k_m.gguf` below is. A Llama 3.x,
Mistral, or Gemma GGUF fails the dialect check with a message naming the
mismatch; see [`lm-llamacpp`](../lm-llamacpp#tool-calling-and-the-model-you-pick).

### Size decides whether it dispatches

Deciding to reach for a tool is the part small models are worst at, and the
failure is one-sided: they under-dispatch, refusing ("I can't check the
weather") rather than calling. Measured over eight utterances × three seeds,
against this example's prompt and tool description:

| Model | Correct |
|---|---|
| Qwen 3 1.7B | 13/24 |
| Qwen 3 4B | 23/24 |

Neither ever dispatched something it should have answered itself, so the cost of
a smaller model is errands silently declined, not spurious tasks. 4B is what
this example documents; the 1.7B is usable if you phrase errands as errands
("kick off a task to …"), which is the phrasing it does recognise.

## Thinking is suppressed

Qwen 3 reasons before it answers, and this pipeline speaks every token that is
not tool-call syntax — so its reasoning would be read aloud. Qwen 3's chat
template hides it behind an `enable_thinking=false` kwarg, but the legacy
`llama_chat_apply_template` that `llama-cpp-2` wraps takes no kwargs, so that
branch never fires. The adapter applies it by hand instead:
`LlamaCppConfig::with_assistant_prefix` prefills the assistant turn with the
empty `<think></think>` block that branch emits, which is what this example does
by default. Pass `--thinking` to leave reasoning on and hear it.

The prefix is prompt, not generation, so none of it reaches a transcript. It is
also model-specific: Qwen 2.5 never saw those tokens, so use `--thinking` with a
2.5 GGUF.

## Latency

Startup decodes one throwaway token before it starts listening. The system
message and tool declarations are a constant prefix of every turn, and the
adapter reuses the longest prefix it has already decoded, so paying for them at
startup takes them off the first thing you say. On an M-series laptop with
`gpu_layers: 0` and Qwen 3 4B, time from your first token to the model's:

| | First question |
|---|---|
| Cold | 6.2 s |
| Warmed | 2.0 s |

The ~4 s moves to startup, where nothing is waiting on it. Every later turn was
already warm.

Each reply prints how long it took, measured from the end of your speech to the
first synthesised audio:

```text
SpeechStopped (1.74 s)
  ⏱ first speech 2.31 s after silence
```

That interval covers transcription, the model, and the first sentence of
synthesis — the whole pipeline, not just the model. `--stt-threads` and
`LlamaCppConfig::with_gpu_layers` are the two biggest levers on it.

## Requirements

- Rust 1.88 or newer.
- macOS, Windows, or Linux with a working microphone and output device.
- The VAD, ASR, and TTS models from
  [`e2e-voice-agent`](../e2e-voice-agent#download-the-models) — follow that
  README's download steps first.
- A chat GGUF. This example documents Qwen 3 4B rather than the 0.5B that
  README downloads, because dispatching well means judging which errands leave
  the machine:

  ```console
  curl -L \
    https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf \
    -o models/qwen3-4b-q4_k_m.gguf
  ```

- A reachable `hermes gateway` and its `API_SERVER_KEY`.

The example is verified on macOS. Its unit tests are excluded from CI's test
run: on x86-64 Linux this package's test binary — static sherpa and llama.cpp
linked together — aborts with a C++ `std::bad_alloc`, the same way
[`lm-llamacpp`](../lm-llamacpp#requirements) does. Run them locally with
`cargo test -p e2e-voice-agent-hermes`; they need no models.

## The gateway key

The key is the gateway's `API_SERVER_KEY`, sent as a bearer token. Give it
either way:

```console
HERMES_API_KEY=… cargo run -p e2e-voice-agent-hermes -- …   # preferred
cargo run -p e2e-voice-agent-hermes -- … --hermes-key …     # flag wins if both
```

Prefer the environment variable — the flag lands in your shell history. With
neither, startup fails with `--hermes-key is required (or set HERMES_API_KEY)`.
A *wrong* key is not a startup failure: the gateway refuses the run and you hear
the agent relay it, because the refusal arrives as a `Rejected` event.

## Run

The env var is set inline below; export it instead if you prefer.

```console
ASR=./models/sherpa-onnx-moonshine-base-en-quantized-2026-02-27
TTS=./models/kokoro-en-v0_19

HERMES_API_KEY=… cargo run -p e2e-voice-agent-hermes -- \
  --vad-model ./models/silero_vad.onnx \
  --encoder "$ASR/encoder_model.ort" \
  --merged-decoder "$ASR/decoder_model_merged.ort" \
  --tokens "$ASR/tokens.txt" \
  --lm-model ./models/qwen3-4b-q4_k_m.gguf \
  --tts-model "$TTS/model.onnx" \
  --tts-voices "$TTS/voices.bin" \
  --tts-tokens "$TTS/tokens.txt" \
  --tts-data-dir "$TTS/espeak-ng-data" \
  --hermes-url http://127.0.0.1:8642
```

Then say **“go find me a crab recipe”**. Expected output resembles:

```text
e2e-voice-agent-hermes: thinking = suppressed
e2e-voice-agent-hermes: hermes = http://127.0.0.1:8642
e2e-voice-agent-hermes: ask for an errand and the model dispatches it
e2e-voice-agent-hermes: warming the model … 4.1 s
e2e-voice-agent-hermes: listening until Ctrl-C
SpeechStarted
SpeechStopped (2.10 s)
You: go find me a crab recipe
[task] dispatching: find a crab recipe
Agent: Let me find a crab recipe.
  ⏱ first speech 2.31 s after silence
[task] accepted, task pc-8edbbeeb062344ca8606a5424b654d57
[task] progress: running
[task] completed: Here is a simple crab cake recipe …
Agent: The errand came back with a crab cake recipe.
```

…with both `Agent:` lines spoken aloud. The first comes from `DispatchAck`, not
the model: a tool-calling turn is the call and nothing else, so without that
stage the user hears silence from the moment they finish speaking until the task
reports back. The second is the model relaying the completion. The conversation
stays usable throughout — ask something else while the errand runs and the agent
answers it locally.

The acknowledgement names the errand because the system prompt asks for `task`
as the phrase completing "Let me …" — so "what's the weather like in Denver?"
dispatches as `check the weather in Denver`, which is both a better task
description and a sentence. A model that writes a question there anyway falls
back to a generic line rather than reading it out as one.

## Flags

| Flag | Default | Purpose |
|---|---|---|
| `--hermes-url` | `http://127.0.0.1:8642` | Gateway base URL. |
| `--hermes-key` | `$HERMES_API_KEY` | The gateway's `API_SERVER_KEY` — see [above](#the-gateway-key). |
| `--thinking` | off | Leave the model's reasoning on, and hear it — see [above](#thinking-is-suppressed). Use it with a Qwen 2.5 GGUF. |

The inherited flags (`--speaker`, `--speed`, `--system-prompt`, `--seconds`,
`--stt-threads`) behave as in [`e2e-voice-agent`](../e2e-voice-agent#run).

**Use headphones** — over speakers the microphone re-captures the agent's own
voice and it talks to itself.

## What this demonstrates

- **Delegation is a tool call.** A model running locally decides which errands
  leave the machine, from the same `ToolDefinition`s a hosted adapter would
  receive.
- **The app speaks what the model cannot.** A tool call arrives with no text
  beside it, so `DispatchAck` supplies the acknowledgement rather than the
  prompt asking the model for speech it does not emit on that turn — shaping the
  tool *argument* into something speakable instead.
- **A task outlives the turn.** The errand runs as a Hermes run under its own
  `session_id` and reports back whenever it lands, with no turn held open
  waiting for it. (Chaining a *follow-up* onto an accepted task is `update_task`,
  which this example leaves out of scope — see above.)
- **Events enter an idle pipeline.** `DispatchIngress` is an *active* stage: it
  polls the transport alongside the pipeline lanes, so a completion arriving
  during silence still produces speech.
- **Failure is conversational.** A gateway that is down, or a bad key, produces
  a `Rejected` event the model can relay — not a pipeline error.
