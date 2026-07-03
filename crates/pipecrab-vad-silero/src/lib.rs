//! pipecrab-vad-silero: Silero voice-activity detection behind the
//! `pipecrab-vad` seam.
//!
//! Placeholder crate. This will implement
//! [`VoiceActivityDetector`](pipecrab_vad::VoiceActivityDetector) for the Silero
//! model, wiring the seam to a target-selected engine: native ONNX Runtime
//! (`silero-ort`) on the host, a WebAssembly worker (`silero-web`) in the
//! browser. The pipeline above it names only the seam, never the model.
#![forbid(unsafe_code)]
