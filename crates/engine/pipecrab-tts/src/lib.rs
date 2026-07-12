//! pipecrab-tts: the text-to-speech interface.
//!
//! [`Synthesizer`] is the swappable TTS capability the conversation loop drives:
//! text in, audio out *incrementally* — every stream item is a preemption point,
//! so a barge-in [`Interrupt`](pipecrab_core::SystemFrame::Interrupt) can stop
//! playback within a single chunk. Concrete engines stay behind it, so the
//! pipeline never names one.
//!
//! [`TtsStage`] adapts any [`Synthesizer`] into a pipeline
//! [`Stage`](pipecrab_runtime::Stage): on a final agent
//! [`Transcript`](pipecrab_core::Transcript) it synthesizes the text and streams
//! [`Audio`](pipecrab_core::DataFrame::Audio) chunks in its place; every other
//! frame passes through untouched.
//!
//! [`SentenceChunker`] is the low-latency feeder that sits *upstream* of
//! [`TtsStage`]: it splits a streaming agent generation into one final agent
//! [`Transcript`](pipecrab_core::Transcript) per sentence, so synthesis of the
//! first sentence can begin before the model has finished the last. This is why
//! a [`Final`](pipecrab_core::Finality::Final) agent transcript is documented as
//! a *generation* unit that may in practice be a single sentence.
//!
//! Platform-neutral and `wasm32`-checkable: the concrete engines live elsewhere
//! (a native engine, a browser engine in a Web Worker), each behind these
//! traits, so the interface itself carries no backend dependency and compiles for
//! both the host and `wasm32-unknown-unknown`.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod chunker;
mod stage;
mod synthesizer;

pub use chunker::{EmitSentence, SentenceChunker};
pub use stage::{Speak, TtsStage};
pub use synthesizer::{Synthesizer, TtsAudioStream, TtsError};
