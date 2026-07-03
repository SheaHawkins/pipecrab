//! pipecrab-audio-cpal: the [`cpal`]-backed audio backend for pipecrab.
//!
//! [`CpalSource`] captures from an input device and [`CpalSink`] plays to an
//! output device, both implementing the platform-neutral
//! [`AudioSource`](pipecrab_audio::AudioSource) /
//! [`AudioSink`](pipecrab_audio::AudioSink) traits — so a pipeline drives audio
//! I/O without ever naming cpal. Both are opened from one shared [`CpalConfig`]
//! (which device per side, plus chunk/buffer sizing); [`input_device_names`] /
//! [`output_device_names`] enumerate the choices for
//! [`DeviceSelection::Name`].
//!
//! Available wherever cpal is — macOS, Windows, Linux, iOS, Android, and the
//! browser (wasm) — so there is no platform gate.
//!
//! # The real-time boundary
//!
//! cpal's device callbacks run on a real-time thread that must never block,
//! allocate, or lock. Each callback only moves `f32` samples across a lock-free
//! [`rtrb`] ring and wakes a `futures::task::AtomicWaker`; the async side
//! (`next_chunk` / `play`) polls the ring and registers that waker. Waking from
//! the callback is a pragmatic simplification — glitch-free at these ~20 ms
//! buffer sizes; a strict wait-free bridge is deferred.
//!
//! The [`AudioSource`](pipecrab_audio::AudioSource) /
//! [`AudioSink`](pipecrab_audio::AudioSink) seam is `Send` on native, but a
//! `cpal::Stream` is `!Send`. So on native the stream is built and parked on a
//! dedicated owning thread (see `stream`), and [`CpalSource`] / [`CpalSink`]
//! hold only the `Send` ring end — a server can spawn one pump per session. On
//! `wasm32` the seam bound is vacuous and cpal runs on the single main thread,
//! so the stream is held inline.
//!
//! # Format & timing
//!
//! Capture and playback run at their device's default sample rate (no
//! resampling — a shared-clock same-device setup keeps the rates matched), mono
//! end to end: multi-channel input is downmixed, mono output is duplicated
//! across the device's channels. Chunk size and ring depth come from
//! [`CpalConfig`].
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod bridge;
mod config;
mod sink;
mod source;
#[cfg(not(target_arch = "wasm32"))]
mod stream;

pub use config::{CpalConfig, DeviceSelection};
pub use sink::{output_device_names, CpalSink};
pub use source::{input_device_names, CpalSource};
