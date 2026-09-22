//! pipecrab-tts-web: Kokoro text-to-speech in the browser, over kokoro-js.
//!
//! [`KokoroTts`] runs a Kokoro ONNX export on [kokoro-js] and implements
//! [`Synthesizer`](pipecrab_tts::Synthesizer): text in, 24 kHz mono audio out,
//! a clause at a time. Feed it through
//! [`SentenceChunker`](pipecrab_tts::SentenceChunker) into
//! [`TtsStage`](pipecrab_tts::TtsStage), exactly as the native Kokoro adapter
//! is fed:
//!
//! ```ignore
//! let tts = KokoroTts::load(&kokoro, KokoroTtsConfig::new(model)).await?;
//! let stage = TtsStage::new(tts);
//! ```
//!
//! # The engine comes from the page
//!
//! [`load`](KokoroTts::load) takes the kokoro-js namespace the page already
//! imported, rather than importing one itself. So the crate pins no CDN and no
//! bundler layout. kokoro-js runs on Transformers.js, and which copy it
//! resolves — the page's own, through an import map — stays the application's
//! call.
//!
//! # wasm only
//!
//! This crate is the browser engine adapter, so it is empty on every other
//! target; the native Kokoro path is `pipecrab-tts-sherpa`.
//!
//! [kokoro-js]: https://www.npmjs.com/package/kokoro-js
#![forbid(unsafe_code)]
#![warn(missing_docs)]

// Pure text handling, so its tests also run on the host.
#[cfg(any(target_arch = "wasm32", test))]
mod clauses;
#[cfg(target_arch = "wasm32")]
mod config;
#[cfg(target_arch = "wasm32")]
mod synthesizer;

#[cfg(target_arch = "wasm32")]
pub use config::KokoroTtsConfig;
#[cfg(target_arch = "wasm32")]
pub use synthesizer::KokoroTts;
