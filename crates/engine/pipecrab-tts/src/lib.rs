//! Text-to-speech interfaces and pipeline integration.
//!
//! [`Synthesizer`] streams audio from text. Chunked output gives the pipeline
//! await points where an interrupt can stop synthesis promptly.
//!
//! [`TtsStage`] converts final agent [`Transcript`](pipecrab_core::Transcript)s
//! into [`Audio`](pipecrab_core::DataFrame::Audio) frames.
//!
//! Place [`SentenceChunker`] upstream to start speaking complete sentences
//! before the language model finishes its response.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod chunker;
mod stage;
mod synthesizer;

pub use chunker::{EmitSentence, SentenceChunker};
pub use stage::{Speak, TtsStage};
pub use synthesizer::{Synthesizer, TtsAudioStream, TtsError};
