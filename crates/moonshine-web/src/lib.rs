//! moonshine-web: browser inference engine for the Moonshine STT model.
//!
//! Placeholder crate. This will host the WebAssembly engine that runs Moonshine
//! speech-to-text in the browser (Transformers.js in a Web Worker), awaiting the
//! worker so the pipeline thread never blocks. It plugs in behind the
//! `Transcriber` seam (`pipecrab-stt`) via `pipecrab-stt-moonshine`.
#![forbid(unsafe_code)]
