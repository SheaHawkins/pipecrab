//! [`cpal`]-backed audio input and output.
//!
//! [`CpalSource`] and [`CpalSink`] implement the platform-neutral
//! [`AudioSource`](pipecrab_audio::AudioSource) and
//! [`AudioSink`](pipecrab_audio::AudioSink) traits. [`CpalConfig`] selects
//! devices and buffer sizing.
//!
//! This backend is native-only.
//!
//! # The real-time boundary
//!
//! Device callbacks do not block, allocate, or lock. They move `f32` samples
//! through [`rtrb`] rings and wake the async side.
//!
//! Because `cpal::Stream` is `!Send`, a dedicated thread owns each stream while
//! the public source or sink holds the `Send` ring endpoint.
//!
//! # Format & timing
//!
//! Capture and playback use the device's default rate without resampling. Input
//! is downmixed to mono; output mono is duplicated across device channels.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod bridge;
mod config;
mod sink;
mod source;
mod stream;

pub use config::{CpalConfig, DeviceSelection};
pub use sink::{output_device_names, CpalSink};
pub use source::{input_device_names, CpalSource};
