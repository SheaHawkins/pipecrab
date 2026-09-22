//! pipecrab-lm-web: a browser language model for pipecrab, over Transformers.js.
//!
//! [`TransformersLm`] runs a Transformers.js `text-generation` pipeline — any
//! instruction-tuned model with an ONNX export and a chat template on the Hub —
//! and implements [`LanguageModel`](pipecrab_lm::LanguageModel), streaming the
//! reply a word at a time into [`LmStage`](pipecrab_lm::LmStage), exactly as the
//! native llama.cpp adapter does:
//!
//! ```ignore
//! let lm = TransformersLm::load(&transformers, TransformersLmConfig::new(model)).await?;
//! let stage = LmStage::new(lm, system_prompt);
//! ```
//!
//! # Streaming without a callback into Rust
//!
//! The engine's streamer pushes each piece of text into a queue on the JS side,
//! and the [`ModelStream`](pipecrab_lm::ModelStream) pulls from it one promise
//! at a time. JS never holds a Rust closure, so a generation that outlives its
//! stream — the stage was dropped mid-reply — has nothing freed to call.
//! Dropping the stream interrupts the generation at its next token, as does
//! [`cancel`](pipecrab_lm::LanguageModel::cancel).
//!
//! # The engine comes from the page
//!
//! [`load`](TransformersLm::load) takes the Transformers.js namespace the page
//! already imported, rather than importing one itself. So the crate pins no CDN
//! and no bundler layout, and where the engine puts its own work — a Web Worker,
//! WebGPU, the main thread — stays the application's call.
//!
//! # wasm only
//!
//! This crate is the browser engine adapter, so it is empty on every other
//! target; the native path is `pipecrab-lm-llamacpp`.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

#[cfg(target_arch = "wasm32")]
mod config;
#[cfg(target_arch = "wasm32")]
mod model;

#[cfg(target_arch = "wasm32")]
pub use config::TransformersLmConfig;
#[cfg(target_arch = "wasm32")]
pub use model::TransformersLm;
