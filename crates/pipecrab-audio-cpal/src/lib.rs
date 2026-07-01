//! pipecrab-audio-cpal: the desktop (cpal + rtrb) backend for pipecrab audio I/O.
//!
//! [`CpalSource`] captures from the default input device and [`CpalSink`] plays
//! to the default output device, both implementing the platform-neutral
//! [`AudioSource`](pipecrab_audio::AudioSource) /
//! [`AudioSink`](pipecrab_audio::AudioSink) traits — so a pipeline drives audio
//! I/O without ever naming cpal.
//!
//! # The real-time boundary
//!
//! cpal's device callbacks run on a real-time OS thread that must never block,
//! allocate, or lock. Each callback therefore only moves `f32` samples across a
//! lock-free [`rtrb`] ring buffer and wakes a [`futures::task::AtomicWaker`];
//! the async side (`next_chunk` / `play`) polls the ring and registers that
//! waker, `.await`ing for data (capture) or for room (playback backpressure).
//! Waking an `AtomicWaker` from the callback is a pragmatic desktop
//! simplification — glitch-free at these ~20 ms buffer sizes; a strict
//! wait-free bridge is deferred.
//!
//! # Format & timing
//!
//! Capture and playback both run at their device's default sample rate (no
//! resampling — on a shared-clock same-device setup the two rates match), and
//! mono end to end: multi-channel input is downmixed to mono, mono output is
//! duplicated across the device's channels. Chunks are ~20 ms
//! (`rate * 20 / 1000` frames; 960 @ 48 kHz).
#![forbid(unsafe_code)]
#![warn(missing_docs)]

#[cfg(any(target_os = "macos", target_os = "windows", target_os = "linux"))]
mod desktop;

#[cfg(any(target_os = "macos", target_os = "windows", target_os = "linux"))]
pub use desktop::{CpalSink, CpalSource};
