//! [`CpalSink`]: playback over the [`AudioSink`] trait.
//!
//! The async [`play`](AudioSink::play) pushes a chunk's samples into a
//! [`bridge`](crate::bridge) ring, awaiting room when full; the cpal output
//! callback pops one mono sample per frame, duplicates it across the device's
//! channels, and outputs silence on underrun.

use std::sync::Arc;

use async_trait::async_trait;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample};
use rtrb::{Consumer, RingBuffer};

use pipecrab_audio::{AudioChunk, AudioError, AudioFormat, AudioSink};

use crate::bridge::{PlaybackRing, Signal};
use crate::config::{CpalConfig, DeviceSelection};

/// Plays mono `f32` audio to an output device.
pub struct CpalSink {
    name: String,
    ring: PlaybackRing,
    /// Keeps the playback stream alive. The audio interface is `Send` but a
    /// `cpal::Stream` is not, so the stream is parked on its own thread and only
    /// this `Send` handle is held here.
    _thread: crate::stream::StreamThread,
}

impl CpalSink {
    /// Open the output device named by `config.sink_device` and start playback
    /// at its default sample rate, chunked per `config`.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::Device`] if the device is missing, its config can't
    /// be read, its sample format is unsupported, or the stream fails to build
    /// or start.
    pub fn new(config: &CpalConfig) -> Result<Self, AudioError> {
        // The interface is `Send` but a `cpal::Stream` is not, so build and park the
        // stream on its own thread, keeping only the `Send` ring end here.
        let config = config.clone();
        let ((ring, name), thread) =
            crate::stream::spawn_stream(move || build_playback(&config).map(|(s, r, n)| (s, (r, n))))?;
        Ok(Self { name, ring, _thread: thread })
    }

    /// The name of the output device audio is being played to.
    pub fn device_name(&self) -> &str {
        &self.name
    }
}

#[async_trait]
impl AudioSink for CpalSink {
    fn format(&self) -> AudioFormat {
        self.ring.format()
    }

    async fn play(&mut self, chunk: AudioChunk) -> Result<(), AudioError> {
        self.ring.play(chunk).await
    }
}

/// Open the output device named by `config`, build and start its playback
/// stream, and return the (`!Send`) stream alongside the `Send` async end (a
/// [`PlaybackRing`]) and the device name. Split out so it can run on the
/// stream-owning thread (see [`CpalSink::new`] and [`crate::stream`]).
fn build_playback(config: &CpalConfig) -> Result<(cpal::Stream, PlaybackRing, String), AudioError> {
    let host = cpal::default_host();
    let device = find_output_device(&host, &config.sink_device)?;
    let name = device.name().unwrap_or_else(|_| "<unknown output>".into());
    let supported = device
        .default_output_config()
        .map_err(|e| AudioError::Device(format!("default output config: {e}")))?;
    let sample_format = supported.sample_format();
    let stream_config: cpal::StreamConfig = supported.into();
    let sample_rate = stream_config.sample_rate.0;
    let channels = stream_config.channels as usize;

    let (producer, consumer) = RingBuffer::<f32>::new(config.ring_capacity(sample_rate));
    let signal = Signal::new();

    // Only one match arm runs, so moving `consumer`/`signal` into each is fine.
    let stream = match sample_format {
        SampleFormat::F32 => build_playback_stream::<f32>(
            &device, &stream_config, consumer, signal.clone(), channels,
        ),
        SampleFormat::I16 => build_playback_stream::<i16>(
            &device, &stream_config, consumer, signal.clone(), channels,
        ),
        SampleFormat::U16 => build_playback_stream::<u16>(
            &device, &stream_config, consumer, signal.clone(), channels,
        ),
        other => {
            return Err(AudioError::Device(format!("unsupported output sample format: {other:?}")))
        }
    }
    .map_err(|e| AudioError::Device(format!("build output stream: {e}")))?;
    stream.play().map_err(|e| AudioError::Device(format!("start output stream: {e}")))?;

    let ring = PlaybackRing::new(producer, signal, AudioFormat::new(sample_rate, 1));
    Ok((stream, ring, name))
}

/// Names of the available output (playback) devices, for building a
/// [`DeviceSelection::Name`].
pub fn output_device_names() -> Result<Vec<String>, AudioError> {
    Ok(cpal::default_host()
        .output_devices()
        .map_err(|e| AudioError::Device(format!("enumerate output devices: {e}")))?
        .filter_map(|d| d.name().ok())
        .collect())
}

/// Resolve a [`DeviceSelection`] against the host's output devices.
fn find_output_device(host: &cpal::Host, selection: &DeviceSelection) -> Result<cpal::Device, AudioError> {
    match selection {
        DeviceSelection::Default => host
            .default_output_device()
            .ok_or_else(|| AudioError::Device("no default output device".into())),
        DeviceSelection::Name(name) => host
            .output_devices()
            .map_err(|e| AudioError::Device(format!("enumerate output devices: {e}")))?
            .find(|d| d.name().map(|n| &n == name).unwrap_or(false))
            .ok_or_else(|| AudioError::Device(format!("no output device named {name:?}"))),
    }
}

/// Build the output stream for a device whose native sample type is `T`,
/// duplicating each mono `f32` sample across the device's channels.
fn build_playback_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    mut consumer: Consumer<f32>,
    signal: Arc<Signal>,
    channels: usize,
) -> Result<cpal::Stream, cpal::BuildStreamError>
where
    T: SizedSample + FromSample<f32>,
{
    let err_signal = signal.clone();
    device.build_output_stream::<T, _, _>(
        config,
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            for frame in data.chunks_mut(channels) {
                let mono = consumer.pop().unwrap_or(0.0); // silence on underrun.
                let value = T::from_sample(mono);
                for out in frame.iter_mut() {
                    *out = value;
                }
            }
            signal.wake(); // freed room -> wake a play() blocked on backpressure.
        },
        move |err| {
            eprintln!("pipecrab-audio-cpal: output stream error: {err}");
            err_signal.fail(); // set closed AND wake, so play can't hang.
        },
        None,
    )
}
