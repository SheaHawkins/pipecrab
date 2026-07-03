//! silero-ort: native ONNX Runtime engine for the Silero VAD model.
//!
//! Placeholder crate. This will host the `ort`-backed (ONNX Runtime) inference
//! engine that runs Silero voice-activity detection on native targets,
//! offloading the forward pass off the pipeline thread. It plugs in behind the
//! `VoiceActivityDetector` seam (`pipecrab-vad`) via `pipecrab-vad-silero`.
#![forbid(unsafe_code)]
