//! pipecrab-stt: the speech-to-text interface.
//!
//! [`StreamingTranscriber`] is the STT capability the conversation pipeline
//! drives: `f32` audio a window at a time in, [`SttEvent`]s out — partial
//! hypotheses while the user is still speaking, then a final transcript, which
//! is what a low-latency conversation loop needs. Concrete models stay behind
//! it, so the pipeline never names one.
//!
//! An engine reaches that interface one of two ways:
//!
//! * A native streaming engine (e.g. a streaming Zipformer) implements
//!   [`StreamingTranscriber`] directly, emitting real partials.
//! * A chunk-final engine (e.g. Moonshine) implements the simpler one-shot
//!   [`Transcriber`] — `f32` samples in, one transcript out, no partials — and
//!   the [`Buffered`] adapter lifts it to [`StreamingTranscriber`] by
//!   accumulating the utterance and transcribing it once at the end. So a
//!   partial-less engine still plugs into the same streaming interface, without
//!   the pipeline knowing the difference.
//!
//! [`SttStage`] adapts a [`StreamingTranscriber`] into a pipeline
//! [`Stage`](pipecrab_runtime::Stage) as a **stateless protocol adapter**.
//!
//! # Format authority
//!
//! Format authority flows from the party that *knows* the requirement, not from
//! whatever the wire happens to carry. Stages reject fatally mismatched formats.
//!
//! Platform-neutral and `wasm32`-checkable: the concrete engines live elsewhere
//! (native `ort`, browser Transformers.js in a Web Worker), each behind these
//! traits, so the interface itself carries no backend dependency and compiles for
//! both the host and `wasm32-unknown-unknown`.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod stage;
mod streaming;
mod transcriber;

pub use stage::{SttEffect, SttStage};
pub use streaming::{Buffered, StreamingTranscriber, SttEvent};
pub use transcriber::{SttError, Transcriber};
