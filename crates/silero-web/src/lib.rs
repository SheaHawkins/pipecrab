//! silero-web: browser inference engine for the Silero VAD model.
//!
//! Placeholder crate. This will host the WebAssembly engine that runs Silero
//! voice-activity detection in the browser (in a Web Worker), awaiting the
//! worker so the pipeline thread never blocks. It plugs in behind the
//! `VoiceActivityDetector` seam (`pipecrab-vad`) via `pipecrab-vad-silero`.
#![forbid(unsafe_code)]
