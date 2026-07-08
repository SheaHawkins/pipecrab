//! pipecrab-stt-moonshine: the Moonshine STT model, wired into pipecrab's
//! [`Transcriber`] trait.
//!
//! Moonshine is a fast, streaming-friendly speech-to-text model. This crate
//! implements [`pipecrab_stt::Transcriber`] on top of a Moonshine engine,
//! picking the backend by target: the native `ort`-hosted engine
//! ([`moonshine-ort`]) on the host, and the browser onnxruntime-web engine
//! ([`moonshine-web`]) on `wasm32`. The pipeline above depends only on the
//! interface, never on this crate — so swapping Moonshine for another model
//! touches nothing upstream.
//!
//! Scaffold: the crate and its place in the workspace and release graph are set
//! up, and the interface types are re-exported for convenience; the concrete
//! `Transcriber` impl and its `moonshine-web` / `moonshine-ort` engine
//! dependencies land in follow-up work.
//!
//! [`moonshine-ort`]: https://docs.rs/moonshine-ort
//! [`moonshine-web`]: https://docs.rs/moonshine-web
#![forbid(unsafe_code)]

#[doc(no_inline)]
pub use pipecrab_stt::{SttError, Transcriber};
