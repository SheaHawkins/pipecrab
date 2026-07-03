//! moonshine-ort: native ONNX Runtime engine for the Moonshine STT model.
//!
//! Placeholder crate. This will host the `ort`-backed (ONNX Runtime) inference
//! engine that runs Moonshine speech-to-text on native targets, offloading the
//! forward pass off the pipeline thread. It plugs in behind the `Transcriber`
//! seam (`pipecrab-stt`) via `pipecrab-stt-moonshine`.
#![forbid(unsafe_code)]
