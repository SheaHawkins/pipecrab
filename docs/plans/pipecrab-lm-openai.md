# OpenAI-compatible LM adapter

## Goal

Add `pipecrab-lm-openai`, an adapter that implements `pipecrab_lm::LanguageModel`
over the OpenAI Chat Completions wire format, so `LmStage` drives any
OpenAI-compatible host: OpenAI itself, OpenRouter, Together, vLLM, a llama.cpp
`llama-server`, LM Studio, Ollama's compat endpoint, or a Hermes proxy.

Tool calling is first class. The stage's `ToolDefinition`s go out as `tools`,
streamed `tool_calls` fragments come back as `ModelDelta::ToolCall`, and
`Message::ToolResult` / `Message::Event` render into valid provider history — so
the dispatch round-trip (`ToolCall` → transport → `ModelInput` → next turn)
closes without any dispatch-specific code in the adapter.

## Scope

This change includes:

- A publishable `crates/adapters/pipecrab-lm-openai` crate.
- A `POST {base}/chat/completions` client with SSE streaming.
- A pure SSE decoder and tool-call assembler, separate from the HTTP edge.
- Full `Conversation` → OpenAI `messages` rendering for every `Message` variant.
- `tools` request rendering from `ToolDefinition`.
- Epoch-based `cancel` that drops the in-flight response.
- Host-compatibility escape hatches: extra headers, extra body fields, an
  optional non-streaming mode.
- Deterministic tests over `wiremock` plus inline decoder unit tests.
- An `#[ignore]` live test gated on environment variables.

This change does not include:

- An example app or an e2e wiring change. `examples/lm-openai` and swapping
  `e2e-voice-agent-hermes` onto a hosted model are separate, follow-up PRs.
- The Responses API, the Assistants API, or Azure's deployment-scoped URL shape
  (reachable through `extra_headers` / `base_url`, not modelled).
- Multimodal content parts, audio input, `n > 1`, or logprobs.
- Prompt caching directives, usage accounting, or cost reporting.
- Retries, backoff, or failover between hosts.
- `wasm32` support. Like `pipecrab-audio-cpal` and `pipecrab-dispatch-hermes`,
  this crate is native-only and outside the wasm portability matrix.

## Repository placement

```toml
[package.metadata.pipecrab]
layer = "adapter"
```

```text
crates/adapters/pipecrab-lm-openai/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs      crate docs, re-exports
│   ├── config.rs   OpenAiConfig, OpenAiBuildError, redacted Debug
│   ├── wire.rs     serde request/response types; Conversation → messages
│   ├── stream.rs   SSE line decoder + tool-call accumulator
│   ├── client.rs   the reqwest edge: headers, non-2xx mapping
│   └── model.rs    OpenAi handle; LanguageModel impl; cancel epoch
└── tests/
    ├── generate.rs wiremock-backed streaming, tools, errors
    └── live.rs     #[ignore], env-gated
```

Register in `[workspace.dependencies]`:

```toml
pipecrab-lm-openai = { path = "crates/adapters/pipecrab-lm-openai", version = "0.6.0" }
```

The `crates/adapters/*` glob makes it a workspace member and subjects it to the
layering gate with no exception. It needs no entry in CI's wasm check list —
adapters are exempt.

Dependencies: `pipecrab-lm`, `async-trait`, `futures`, `serde`, `serde_json`,
`thiserror`, `url`, `uuid`, and `reqwest` (`rustls-tls`, no `json` feature
needed beyond request bodies). Dev: `tokio`, `wiremock`.

## Runtime requirement

`reqwest`'s native client needs a tokio reactor, so an application using this
adapter drives its pipeline on a tokio runtime rather than
`futures::executor::block_on` — the same constraint `pipecrab-dispatch-hermes`
already imposes, and stated in the crate docs.

Unlike `pipecrab-lm-llamacpp`, there is **no worker thread**. HTTP is already
async and already cancellable by dropping the response, so `generate` returns a
stream that maps the response body directly. Nothing owns mutable decode state.

## Public API

```rust
pub struct OpenAi;               // cheap cloneable handle
pub struct OpenAiConfig;         // builder
pub enum OpenAiBuildError;       // construction-time failure

impl OpenAi {
    pub fn new(config: OpenAiConfig) -> Result<Self, OpenAiBuildError>;
}

impl LanguageModel for OpenAi { /* generate, cancel, save_state, load_state */ }
```

`OpenAi` holds an `Arc<Inner>` of the reqwest `Client`, the resolved endpoint
`Url`, the rendered static headers, and an `Arc<AtomicU64>` epoch.

## Configuration

```rust
OpenAiConfig::new(model, api_key)
    .with_base_url(url)                     // default https://api.openai.com/v1
    .with_connect_timeout(Duration)         // default 10s
    .with_read_timeout(Duration)            // default 30s
    .with_header(name, value)               // repeatable
    .with_extra_body(serde_json::Map)       // merged into the request root
    .with_streaming(bool)                   // default true
```

| Setting | Default |
| --- | --- |
| base URL | `https://api.openai.com/v1` |
| connect timeout | 10s |
| read timeout (idle between body chunks) | 30s |
| streaming | enabled |
| extra headers | none |
| extra body | none |

**No total request timeout.** A whole-request deadline would abort a long but
healthy generation. The timeouts bound *connecting* and *silence between
chunks*, which is what actually indicates a dead host.

**The key is a secret.** `api_key` is `Arc<str>`, redacted in `Debug` exactly as
`HermesConfig` redacts its token, and it appears only in the `Authorization`
header — never in a URL, a body, or an error string. An **empty key omits the
header**, which is how a local `llama-server` or Ollama endpoint is reached.

**The base URL is taken literally.** It should include the version segment
(`https://openrouter.ai/api/v1`, `http://127.0.0.1:8080/v1`); the adapter
appends `chat/completions` with the same `pop_if_empty` segment handling the
Hermes client uses, so a trailing slash is harmless. No `/v1` is inferred.

**`extra_headers` and `extra_body` are the compatibility surface.** They are why
one adapter covers every host without a per-provider enum: OpenRouter's
`HTTP-Referer` / `X-Title`, Azure's `api-key`, a proxy's tenant header;
`top_p`, `provider`, `reasoning`, or a host that demands
`max_completion_tokens` instead of `max_tokens`. `extra_body` merges into the
request root and may overwrite an adapter-set field, so it can also correct one.

`OpenAiBuildError` covers an empty model name, a `base_url` that cannot be a
base, an invalid header name or value, and a reqwest client that failed to
build.

## Request

```json
{
  "model": "...",
  "messages": [...],
  "stream": true,
  "tools": [{"type": "function", "function": {"name", "description", "parameters"}}],
  "max_tokens": 256,
  "temperature": 0.7
}
```

`tools` is omitted entirely when the stage passes none — a host that rejects an
empty array stays happy. `tool_choice` is never sent; every host defaults to
auto, and forcing a choice is an `extra_body` concern.

`max_tokens` and `temperature` are sent only when `GenParams` carries them, so
the host applies its own defaults otherwise.

`GenParams::grammar`, when set, is parsed as a JSON Schema document and sent as
`response_format: {"type": "json_schema", "json_schema": {"name": "response",
"schema": <parsed>}}`. Text that is not valid JSON is `LmError::Engine` rather
than a silently dropped constraint.

### Message rendering

| `Message` | OpenAI |
| --- | --- |
| `System { content }` | `{"role": "system", "content": ...}` |
| `User { content }` | `{"role": "user", "content": ...}` |
| `Assistant { content, tool_calls }` | `{"role": "assistant", "content": ... or null, "tool_calls": [...]}` |
| `ToolResult { tool_call_id, content, .. }` | `{"role": "tool", "tool_call_id": ..., "content": ...}` |
| `Event { source, kind, content }` | `{"role": "user", "content": "[source/kind] content"}` |

`content` is `null`, not `""`, on a tool-call-only assistant turn: several hosts
reject an assistant message with both empty content and tool calls.

An assistant turn's `tool_calls` are replayed as
`{"id", "type": "function", "function": {"name", "arguments"}}`, arguments as
the JSON *text* `ToolCall` already carries — no re-encoding.

`ToolResult`'s `name` is dropped. The field is deprecated on OpenAI's tool
message and some strict hosts reject unknown keys; `tool_call_id` is the
correlator that matters.

`Event` has no OpenAI role. It renders as a labelled user turn, matching the
llama.cpp adapter's `[{source}/{kind}] {content}` so a conversation reads the
same across both adapters.

## Streaming and tool-call assembly

`stream.rs` is a pure state machine over response bytes. It never touches
reqwest, so nearly all of the protocol is unit-tested without a server.

**Framing.** Accumulate bytes; split on `\n`, tolerating `\r\n`. Ignore comment
lines (`:`) and non-`data` fields (`event:`, `id:`, `retry:`). Per SSE, join
consecutive `data:` lines with `\n` and dispatch the event on a blank line, plus
a flush at EOF. `data: [DONE]` ends the stream.

**Per chunk**, read `choices[0].delta`:

- `content` → `ModelDelta::Text`. Empty strings are skipped, so no empty partial
  transcript reaches the pipeline.
- `reasoning_content` / `reasoning` → **dropped**. Hosts like DeepSeek and
  OpenRouter stream chain-of-thought in these fields; forwarded, TTS would speak
  it aloud.
- `tool_calls[]` → fed to the accumulator.
- A chunk shaped `{"error": ...}` → `LmError::ProviderStream` with the message.

**Accumulator.** Keyed by `index`. The first non-empty `id` and
`function.name` win; `function.arguments` fragments append. A call is flushed
when a fragment for a higher index arrives, and every open call is flushed on
`finish_reason` or at stream end — so a turn's calls emit in index order and a
parallel-call turn does not stall behind the last one.

**Flush** parses the accumulated argument text and builds the delta through
`ModelDelta::tool_call`, which enforces the object shape and re-serializes:

- Empty argument text means `{}`. Zero-argument tools stream `""` on several
  hosts, and failing them would be a false negative.
- A missing `id` is minted as `call_{uuid}`, matching the llama.cpp adapter's
  `mint_call_id`. Local hosts omit it, and it only has to be unique within the
  turn for the tool result to correlate.
- Unparsable or non-object arguments become `LmError::InvalidToolArguments` on
  the stream. `LmStage` turns that into a `StageError` and the turn is dropped —
  better than pushing half a call into dispatch.

**Non-streaming mode** (`with_streaming(false)`) posts without `stream` and
reads `choices[0].message`: `content` as one `Text` delta, `tool_calls` through
the same flush path already complete. It exists because a few compat hosts
implement SSE tool calls incorrectly while their blocking endpoint is fine. The
cost is one turn's latency, and it is off by default.

## Cancellation

`cancel` is synchronous and non-blocking: it bumps the shared epoch. `generate`
captures the epoch it started under, and the stream checks it before yielding
each item; a mismatch ends the stream, which drops the reqwest response and
closes the connection so the host stops billing tokens.

That covers the same races as the llama.cpp adapter: cancel before the response
arrives, cancel between chunks, and cancel after a chunk is decoded but before
`LmStage` observes it. `LmStage` separately drops the in-flight `perform`, so
both halves stop.

Nothing is retried after a cancel, and no request is issued to abort remotely —
closing the body is the abort.

## Session state

`save_state` returns an empty blob; `load_state` accepts an empty blob and
rejects a non-empty one with `LmError::Engine`. A hosted endpoint holds no
per-session decode state — the conversation lives in `LmStage` and is replayed
in full on every call — so there is genuinely nothing to checkpoint, and saying
so is better than pretending to serialize.

## Error mapping

| Condition | `LmError` |
| --- | --- |
| connect failure, DNS, TLS, read timeout | `Engine` |
| non-2xx response | `Engine`, with status and a 200-char body excerpt |
| stream ends mid-chunk, undecodable JSON | `ProviderStream` |
| `{"error": ...}` chunk | `ProviderStream` |
| bad tool arguments | `InvalidToolArguments` |

Status is preserved in the message so a 401, a 429, and a 500 are
distinguishable in a log without a variant per code. No error string ever
contains the API key.

## Tests

### Decoder unit tests (in `stream.rs` / `wire.rs`, no HTTP)

- Split events arriving across arbitrary byte boundaries mid-JSON.
- `\r\n` framing, comment lines, `event:` / `id:` fields, multi-line `data:`.
- `[DONE]` terminates; trailing bytes after it are ignored.
- Text deltas concatenate; empty and null `content` emit nothing.
- `reasoning_content` and `reasoning` emit nothing.
- One tool call assembled from fragmented `arguments`.
- Two calls flushed in index order; the first flushes on the second's arrival.
- A call arriving whole in one chunk.
- Missing `id` is minted; missing `name` is an error.
- Empty arguments become `{}`; non-object arguments are
  `InvalidToolArguments`.
- `finish_reason` flushes an open call; so does EOF without `[DONE]`.
- An `{"error": ...}` chunk becomes `ProviderStream`.
- Every `Message` variant renders to the table above, including the `null`
  content on a tool-call-only assistant turn and the dropped `ToolResult` name.
- `ToolDefinition` renders to the `tools` shape; an empty set omits the field.
- `extra_body` merges into the root and can overwrite an adapter field.
- `OpenAiConfig`'s `Debug` redacts the key.
- An empty key omits the `Authorization` header.

### wiremock integration tests

- A text-only turn yields the expected `Text` deltas in order.
- A tool-calling turn yields exactly one `ToolCall` and no stray text.
- A request body assertion: `model`, `messages`, `tools`, `stream`, and the
  bearer header are what the config implies.
- 401 and 500 map to `Engine` carrying the status; the body excerpt is present
  and the key is not.
- A body that ends mid-event maps to `ProviderStream`.
- Non-streaming mode produces the same deltas as streaming for the same logical
  reply.
- `cancel` during a delayed response terminates the stream without an error.
- A `ToolResult` and an `Event` round-trip: a second generation's request body
  carries the `tool` message and the labelled user turn.

### Live test

`tests/live.rs`, `#[ignore]`, reading `PIPECRAB_OPENAI_BASE_URL`,
`PIPECRAB_OPENAI_API_KEY`, and `PIPECRAB_OPENAI_MODEL`. It runs one text turn
and one tool-calling turn against the real host. Default workspace tests need no
network and no credentials.

## Verification

```console
cargo fmt --all -- --check
cargo test -p pipecrab-lm-openai
cargo test -p pipecrab-arch --test layering
cargo clippy -p pipecrab-lm-openai --all-targets -- -D warnings
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo tree -d
```

The layering test must accept the crate with no exception, and `cargo tree -d`
must not gain a duplicate `reqwest` or `rustls` — the versions match
`pipecrab-dispatch-hermes`.

## Acceptance criteria

- `OpenAi` implements `LanguageModel` and drives `LmStage` unchanged.
- Stage `ToolDefinition`s reach the host as `tools`; streamed `tool_calls`
  fragments reassemble into complete `ModelDelta::ToolCall`s.
- Tool results and external events render into history the host accepts, so the
  dispatch round-trip closes without adapter changes.
- No tool-call syntax, JSON, or reasoning text ever reaches a transcript.
- `cancel` is synchronous, drops the in-flight response, and cannot let a stale
  delta surface after it.
- A run against OpenAI, OpenRouter, and a local `llama-server` differs only by
  `base_url`, `model`, `api_key`, and optional extra headers.
- The API key appears in no `Debug` output, log line, or error message.
- The adapter owns no conversation state; `save_state` is honestly empty.
- Every wire behavior above is covered by tests that need neither network nor
  credentials.
