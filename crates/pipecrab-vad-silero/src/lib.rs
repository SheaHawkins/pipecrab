//! pipecrab-vad-silero: the Silero VAD, wired into pipecrab's
//! [`VoiceActivityDetector`] trait.
//!
//! Silero is a small, fast voice-activity model. This crate implements
//! [`pipecrab_vad::VoiceActivityDetector`] on top of a Silero engine, picking
//! the backend by target: the native `ort`-hosted engine ([`silero-vad-ort`])
//! on the host, and the browser transformers.js engine ([`silero-vad-web`]) on
//! `wasm32` — both built on the shared, backend-free [`silero-vad-core`] crate.
//! The pipeline above depends only on the interface, never on this crate — so
//! swapping Silero for another detector touches nothing upstream.
//!
//! # Design (locked in `docs/plans/silero-vad.md`, §4.4)
//!
//! Scaffold: the crate and its place in the workspace and release graph are set
//! up, and the interface types are re-exported for convenience; the concrete
//! `VoiceActivityDetector` impl lands in follow-up work. The locked shape:
//!
//! - `SileroDetector` holds a `Mutex<engine>`: the trait takes `&self`, while
//!   the engine's recurrent state needs `&mut`, so interior mutability bridges
//!   them (the `VadState` precedent in `pipecrab-vad`'s stage). The standalone
//!   engine crates keep an honest `&mut`; the `Mutex` lives only here.
//! - `detect` rejects any format other than 16 kHz mono with
//!   `VadError::UnsupportedFormat` (it never resamples — trait contract) and
//!   requires exact `frame_len()` frames (`VadStage` supplies them; plan §4.5),
//!   thresholds the probability into `VadVerdict::is_speech`, and leaves edge
//!   debouncing to `VadStage` (no hysteresis here, or it would double-debounce).
//! - Native inference runs inside `offload(…)` so a barge-in can preempt it;
//!   wasm runs inline until a Web Worker offload lands in `pipecrab-runtime`
//!   (one frame is ~1 ms).
//! - The engine dependency is the workspace's first target-cfg split
//!   (`silero-vad-ort` on native, `silero-vad-web` on `wasm32`); it lands with
//!   the impl.
//!
//! [`silero-vad-core`]: https://docs.rs/silero-vad-core
//! [`silero-vad-ort`]: https://docs.rs/silero-vad-ort
//! [`silero-vad-web`]: https://docs.rs/silero-vad-web
#![forbid(unsafe_code)]

#[doc(no_inline)]
pub use pipecrab_vad::{VadError, VadVerdict, VoiceActivityDetector};
