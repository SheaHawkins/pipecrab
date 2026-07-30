# pipecrab-lm-openai

This crate adapts any host speaking the OpenAI Chat Completions format to
PipeCrab's `LanguageModel` capability. `OpenAi` is a cloneable handle over one
endpoint; `LmStage` drives it like any other model.

One adapter covers OpenAI, OpenRouter, Together, vLLM, a llama.cpp
`llama-server`, LM Studio, Ollama's compat endpoint, and proxies. A host is not
a variant here — it is a configuration.

```rust,no_run
use pipecrab_lm::LmStage;
use pipecrab_lm_openai::{OpenAi, OpenAiConfig};

let config = OpenAiConfig::new("gpt-5", std::env::var("OPENAI_API_KEY")?);
let stage = LmStage::new(OpenAi::new(config)?, "You are a helpful assistant.");
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Pointing at another host

`base_url` is taken literally and should carry the version segment; the adapter
appends `chat/completions` to it and infers no `/v1`. An **empty API key omits
the `Authorization` header**, which is how a local server that wants none is
reached.

```rust,no_run
# use pipecrab_lm_openai::{OpenAi, OpenAiConfig};
# use url::Url;
// OpenRouter, which also wants attribution headers.
let openrouter = OpenAiConfig::new("anthropic/claude-sonnet-4.5", std::env::var("OPENROUTER_API_KEY")?)
    .with_base_url(Url::parse("https://openrouter.ai/api/v1")?)
    .with_header("HTTP-Referer", "https://example.com")
    .with_header("X-Title", "my voice agent");

// A local llama-server: no key at all.
let local = OpenAiConfig::new("qwen3", "")
    .with_base_url(Url::parse("http://127.0.0.1:8080/v1")?);
# let _ = (openrouter, local);
# Ok::<(), Box<dyn std::error::Error>>(())
```

`with_header` and `with_extra_body` are the whole compatibility surface.
`extra_body` merges into the request root **last**, so it can add a host's own
field (`provider`, `top_p`, `reasoning`) or correct one the adapter set — a host
that wants `max_completion_tokens` instead of `max_tokens` is a config change,
not a code change.

## Configuration

| Setting | Default |
| --- | --- |
| base URL | `https://api.openai.com/v1` |
| connect timeout | 10s |
| read timeout (silence between body chunks) | 30s |
| streaming | enabled |
| extra headers | none |
| extra body | none |

There is deliberately **no total request timeout**: a whole-request deadline
would abort a long but healthy generation. What is bounded is connecting, and
silence — which is what actually indicates a dead host.

`with_streaming(false)` posts a blocking request instead, for a host whose SSE
tool calls are broken while its blocking endpoint is fine. It costs a turn's
latency and is off by default.

The API key is a secret: `Debug` redacts it, it is set as a sensitive header
value, and it reaches no URL, body, or error message.

## Tool calling

Stage tools go out as `tools`, and a host streams a call back in fragments — an
id and name in one chunk, its `arguments` JSON split across the next several. The
decoder accumulates by `index` and emits a call only once it is complete, so the
stage never sees half a call and no call syntax can reach a transcript.

Tool results and external events render back into provider history as a `tool`
message and a labelled `[source/kind]` user turn, which is what closes the
dispatch round-trip — this crate needs no dispatch-specific code for it.

Two host behaviors are accommodated because failing them would be a false
negative: a call streamed **without an id** gets one minted, and **empty
argument text** means `{}`. A call whose arguments never parse is an error, not
a half-formed call pushed downstream.

Reasoning fields (`reasoning`, `reasoning_content`) are dropped. They carry
chain-of-thought that a TTS stage would otherwise speak aloud.

## Cancellation and state

`cancel` bumps an epoch; the stream ends at its next item and dropping the
response closes the connection, so the host stops generating. There is no worker
thread — HTTP is already async and already cancellable.

`save_state` returns an empty blob and `load_state` accepts only an empty one. A
hosted endpoint holds no per-session decode state: `LmStage` owns the
conversation and replays it in full every turn.

## Runtime

`reqwest`'s native client needs a tokio reactor, so drive the pipeline on a tokio
runtime rather than `futures::executor::block_on`. Like `pipecrab-audio-cpal` and
`pipecrab-dispatch-hermes`, this crate is native-only and outside the wasm
portability matrix.

## Tests

`cargo test -p pipecrab-lm-openai` needs no network and no credentials — the
protocol is covered by unit tests over the decoder and by `wiremock`. The live
test is ignored by default; see `tests/live.rs` for the environment variables it
reads.
