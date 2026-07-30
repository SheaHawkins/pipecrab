//! OpenAI-compatible implementation of Pipecrab's language-model capability.
//!
//! [`OpenAi`] is a cheap cloneable handle over one Chat Completions endpoint. It
//! posts the running [`Conversation`](pipecrab_lm::Conversation) to
//! `{base_url}/chat/completions` and decodes the SSE response into Pipecrab's
//! [`ModelStream`](pipecrab_lm::ModelStream): visible text as
//! [`ModelDelta::Text`](pipecrab_lm::ModelDelta::Text), reassembled `tool_calls`
//! as [`ModelDelta::ToolCall`](pipecrab_lm::ModelDelta::ToolCall).
//!
//! One adapter serves every host that speaks the format — OpenAI, OpenRouter,
//! Together, vLLM, a llama.cpp `llama-server`, LM Studio, Ollama's compat
//! endpoint, a proxy. They differ only by [`OpenAiConfig`]: base URL, model, key,
//! and the [`with_header`](OpenAiConfig::with_header) /
//! [`with_extra_body`](OpenAiConfig::with_extra_body) escape hatches, which carry
//! a host's own headers and request fields without a provider enum here.
//!
//! # Tool calling
//!
//! The stage's [`ToolDefinition`](pipecrab_lm::ToolDefinition)s go out as
//! `tools`, and a provider streams a call back in fragments: an id and name in
//! one chunk, its `arguments` JSON split across the next several. The decoder
//! accumulates a call by `index` and emits it only once complete, so the stage
//! never sees half a call and no call syntax reaches a transcript.
//!
//! Tool results and external events render back into provider history — a `tool`
//! message and a labelled user turn — which is what closes the dispatch
//! round-trip without any dispatch-specific code here.
//!
//! A host's reasoning fields (`reasoning`, `reasoning_content`) are dropped: they
//! carry chain-of-thought a TTS stage would otherwise speak aloud.
//!
//! # No worker thread
//!
//! Unlike a native engine, nothing here owns mutable decode state — HTTP is
//! already async, and already cancellable by dropping the response.
//! [`cancel`](pipecrab_lm::LanguageModel::cancel) bumps an epoch; the stream ends
//! at its next item and the dropped response closes the connection, so the host
//! stops generating.
//!
//! # Native only
//!
//! Like `pipecrab-audio-cpal` and `pipecrab-dispatch-hermes`, this crate depends
//! on `reqwest`, whose native client needs a tokio reactor. Applications drive
//! the pipeline on a tokio runtime rather than `futures::executor::block_on`, and
//! the crate is outside the wasm portability matrix.
//!
//! # Wiring
//!
//! ```no_run
//! use pipecrab_lm::LmStage;
//! use pipecrab_lm_openai::{OpenAi, OpenAiConfig};
//!
//! # fn wire() -> Result<(), Box<dyn std::error::Error>> {
//! let config = OpenAiConfig::new("gpt-5", std::env::var("OPENAI_API_KEY")?);
//! let stage = LmStage::new(OpenAi::new(config)?, "You are a helpful assistant.");
//! # let _ = stage;
//! # Ok(())
//! # }
//! ```
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod client;
mod config;
mod model;
mod stream;
mod wire;

pub use config::{OpenAiBuildError, OpenAiConfig};
pub use model::OpenAi;
