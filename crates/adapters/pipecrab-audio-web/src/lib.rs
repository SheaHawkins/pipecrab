//! pipecrab-audio-web: the browser audio backend for pipecrab.
//!
//! [`WebAudioSource`] captures the microphone and [`WebAudioSink`] plays to the
//! speakers, through the Web Audio graph and behind the platform-neutral
//! [`AudioSource`](pipecrab_audio::AudioSource) and
//! [`AudioSink`](pipecrab_audio::AudioSink) traits — so a pipeline runs in the
//! browser without naming Web Audio, exactly as `pipecrab-audio-cpal` lets it
//! run on the desktop without naming cpal.
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
//! Playback needs no worklet. Each chunk is an `AudioBufferSourceNode` scheduled
//! on the context's clock behind the last, and the rendering thread plays it
//! from there — so a main thread held up by inference cannot starve the output.
//! A barge-in [`cancel`](pipecrab_audio::AudioSink::cancel) stops every
//! scheduled node.
//!
//! # Format
//!
//! Both contexts run at the browser's own rate (typically 48 kHz), reported by
//! `format` as mono. Put a
//! [`ResamplerStage`](pipecrab_audio::ResamplerStage) at the head of the
//! pipeline to reach an engine's rate, and another at the tail to reach the
//! sink's.
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
mod sink;
#[cfg(target_arch = "wasm32")]
mod source;

#[cfg(target_arch = "wasm32")]
pub use config::WebAudioConfig;
#[cfg(target_arch = "wasm32")]
pub use sink::WebAudioSink;
#[cfg(target_arch = "wasm32")]
pub use source::WebAudioSource;
