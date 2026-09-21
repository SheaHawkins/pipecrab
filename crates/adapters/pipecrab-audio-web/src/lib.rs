//! pipecrab-audio-web: the browser audio backend for pipecrab.
//!
//! [`WebAudioSource`] captures the microphone through the Web Audio graph and
//! implements the platform-neutral [`AudioSource`](pipecrab_audio::AudioSource)
//! trait — so a pipeline runs in the browser without naming Web Audio, exactly
//! as `pipecrab-audio-cpal` lets it run on the desktop without naming cpal.
//!
//! Capture only. Playback is the browser half of a TTS path that does not exist
//! yet, so there is no `AudioSink` here to go stale.
//!
//! # The real-time boundary
//!
//! Web Audio's counterpart to cpal's real-time callback is the
//! [`AudioWorkletProcessor`], which runs on the audio rendering thread and must
//! not block. The worklet downmixes to mono, accumulates a fixed-size chunk, and
//! transfers it to the main thread; the async side pops from a bounded queue.
//! A full queue drops the chunk and counts an **overrun**
//! ([`WebAudioSource::overruns`]), the same contract as the cpal ring.
//!
//! Unlike cpal there is no `!Send` handle to park on its own thread: wasm32 is
//! single-threaded and `MaybeSend` is vacuous there, so the JS handles live
//! directly on the source.
//!
//! # Format
//!
//! The context runs at the browser's own rate (typically 48 kHz), reported by
//! [`format`](pipecrab_audio::AudioSource::format) as mono. Feed a
//! [`ResamplerStage`](pipecrab_audio::ResamplerStage) at the head of the
//! pipeline to reach an engine's rate.
//!
//! # wasm only
//!
//! This crate is the browser backend, so it is empty on every other target: the
//! host build of the workspace compiles it to nothing rather than excluding it.
//!
//! [`AudioWorkletProcessor`]: https://developer.mozilla.org/docs/Web/API/AudioWorkletProcessor
#![forbid(unsafe_code)]
#![warn(missing_docs)]

#[cfg(target_arch = "wasm32")]
mod config;
#[cfg(target_arch = "wasm32")]
mod source;

#[cfg(target_arch = "wasm32")]
pub use config::WebAudioConfig;
#[cfg(target_arch = "wasm32")]
pub use source::WebAudioSource;
