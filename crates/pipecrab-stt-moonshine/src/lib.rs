//! pipecrab-stt-moonshine: Moonshine speech-to-text behind the `pipecrab-stt`
//! seam.
//!
//! Placeholder crate. This will implement
//! [`Transcriber`](pipecrab_stt::Transcriber) for the Moonshine model, wiring
//! the seam to a target-selected engine: native ONNX Runtime (`moonshine-ort`)
//! on the host, a WebAssembly worker (`moonshine-web`) in the browser. The
//! pipeline above it names only the seam, never the model.
#![forbid(unsafe_code)]
