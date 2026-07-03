//! pipecrab-vad: the voice-activity-detection seam.
//!
//! One trait — [`VoiceActivityDetector`] — is the swappable VAD capability:
//! `f32` samples in, a speech-probability out. Concrete detectors live elsewhere
//! (e.g. Silero on native `ort` or in the browser, behind
//! `pipecrab-vad-silero`), each implementing this one trait so the pipeline
//! above it never names a concrete model.
//!
//! Like [`pipecrab-stt`](https://docs.rs/pipecrab-stt), the seam itself carries
//! no backend dependency: it is platform-neutral and compiles for both the host
//! and `wasm32-unknown-unknown`.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod detector;

pub use detector::{VadError, VoiceActivityDetector};
