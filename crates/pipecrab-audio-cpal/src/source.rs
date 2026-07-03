//! [`CpalSource`]: capture over the [`AudioSource`] trait.
//!
//! The cpal input callback downmixes each frame to mono and pushes it into a
//! [`bridge`](crate::bridge) ring; the async [`next_chunk`](AudioSource::next_chunk)
//! pops a chunk once one is buffered, or parks until the callback signals more
//! (or the stream fails).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample};
use rtrb::{Producer, RingBuffer};

use pipecrab_audio::{AudioChunk, AudioError, AudioFormat, AudioSource};

use crate::bridge::{CaptureRing, Signal};
use crate::config::{CpalConfig, DeviceSelection};

/// Captures mono `f32` audio from an input device.
pub struct CpalSource {
    name: String,
    ring: CaptureRing,
    /// Keeps the capture stream alive. The audio seam is `Send` but a
    /// `cpal::Stream` is not, so the stream is parked on its own thread and only
    /// this `Send` handle is held here.
    _thread: crate::stream::StreamThread,
}

impl CpalSource {
    /// Open the input device named by `config.source_device` and start capturing
    /// at its default sample rate, chunked per `config`.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::Device`] if the device is missing, its config can't
    /// be read, its sample format is unsupported, or the stream fails to build
    /// or start.
    pub fn new(config: &CpalConfig) -> Result<Self, AudioError> {
        // The seam is `Send` but a `cpal::Stream` is not, so build and park the
        // stream on its own thread, keeping only the `Send` ring end here.
        let config = config.clone();
        let ((ring, name), thread) =
            crate::stream::spawn_stream(move || build_capture(&config).map(|(s, r, n)| (s, (r, n))))?;
        Ok(Self { name, ring, _thread: thread })
    }

    /// The name of the input device audio is being captured from.
    pub fn device_name(&self) -> &str {
        &self.name
    }

    /// Number of frames (samples per channel) in each chunk this source yields.
    pub fn chunk_frames(&self) -> usize {
        self.ring.chunk_frames()
    }

    /// Samples dropped so far because the capture ring was full (the async side
    /// was not popping fast enough). Monotonic; a healthy stream stays at 0.
    pub fn overruns(&self) -> usize {
        self.ring.overruns()
    }
}

#[async_trait]
impl AudioSource for CpalSource {
    fn format(&self) -> AudioFormat {
        self.ring.format()
    }

    async fn next_chunk(&mut self) -> Result<Option<AudioChunk>, AudioError> {
        self.ring.next_chunk().await
    }
}

/// Open the input device named by `config`, build and start its capture stream,
/// and return the (`!Send`) stream alongside the `Send` async end (a
/// [`CaptureRing`]) and the device name. Split out so it can run on the
/// stream-owning thread (see [`CpalSource::new`] and [`crate::stream`]).
fn build_capture(config: &CpalConfig) -> Result<(cpal::Stream, CaptureRing, String), AudioError> {
    let host = cpal::default_host();
    let device = find_input_device(&host, &config.source_device)?;
    let name = device.name().unwrap_or_else(|_| "<unknown input>".into());
    let supported = device
        .default_input_config()
        .map_err(|e| AudioError::Device(format!("default input config: {e}")))?;
    let sample_format = supported.sample_format();
    let stream_config: cpal::StreamConfig = supported.into();
    let sample_rate = stream_config.sample_rate.0;
    let channels = stream_config.channels as usize;
    let chunk_frames = config.chunk_frames(sample_rate);

    let (producer, consumer) = RingBuffer::<f32>::new(config.ring_capacity(sample_rate));
    let signal = Signal::new();
    let overruns = Arc::new(AtomicUsize::new(0));

    // Only one match arm runs, so moving `producer`/the Arcs into each is fine.
    let stream = match sample_format {
        SampleFormat::F32 => build_capture_stream::<f32>(
            &device, &stream_config, producer, signal.clone(), overruns.clone(), channels,
        ),
        SampleFormat::I16 => build_capture_stream::<i16>(
            &device, &stream_config, producer, signal.clone(), overruns.clone(), channels,
        ),
        SampleFormat::U16 => build_capture_stream::<u16>(
            &device, &stream_config, producer, signal.clone(), overruns.clone(), channels,
        ),
        other => {
            return Err(AudioError::Device(format!("unsupported input sample format: {other:?}")))
        }
    }
    .map_err(|e| AudioError::Device(format!("build input stream: {e}")))?;
    stream.play().map_err(|e| AudioError::Device(format!("start input stream: {e}")))?;

    let ring =
        CaptureRing::new(consumer, signal, overruns, chunk_frames, AudioFormat::new(sample_rate, 1));
    Ok((stream, ring, name))
}

/// Names of the available input (capture) devices, for building a
/// [`DeviceSelection::Name`].
pub fn input_device_names() -> Result<Vec<String>, AudioError> {
    Ok(cpal::default_host()
        .input_devices()
        .map_err(|e| AudioError::Device(format!("enumerate input devices: {e}")))?
        .filter_map(|d| d.name().ok())
        .collect())
}

/// Resolve a [`DeviceSelection`] against the host's input devices.
fn find_input_device(host: &cpal::Host, selection: &DeviceSelection) -> Result<cpal::Device, AudioError> {
    match selection {
        DeviceSelection::Default => host
            .default_input_device()
            .ok_or_else(|| AudioError::Device("no default input device".into())),
        DeviceSelection::Name(name) => host
            .input_devices()
            .map_err(|e| AudioError::Device(format!("enumerate input devices: {e}")))?
            .find(|d| d.name().map(|n| &n == name).unwrap_or(false))
            .ok_or_else(|| AudioError::Device(format!("no input device named {name:?}"))),
    }
}

/// Build the input stream for a device whose native sample type is `T`.
fn build_capture_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    mut producer: Producer<f32>,
    signal: Arc<Signal>,
    overruns: Arc<AtomicUsize>,
    channels: usize,
) -> Result<cpal::Stream, cpal::BuildStreamError>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let err_signal = signal.clone();
    device.build_input_stream::<T, _, _>(
        config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            capture_write(data, channels, &mut producer, &overruns);
            signal.wake();
        },
        move |err| {
            eprintln!("pipecrab-audio-cpal: input stream error: {err}");
            err_signal.fail(); // set closed AND wake, so next_chunk can't hang.
        },
        None,
    )
}

/// The input callback's per-buffer work: downmix each interleaved frame of the
/// device's native sample type `T` to mono `f32`, push it into the ring, and
/// count samples dropped when the ring is full (an overrun the async side
/// observes via [`CpalSource::overruns`]). This is the one cpal-coupled piece of
/// the capture path; the ring itself ([`crate::bridge`]) is backend-agnostic.
fn capture_write<T>(data: &[T], channels: usize, producer: &mut Producer<f32>, overruns: &AtomicUsize)
where
    T: Sample,
    f32: FromSample<T>,
{
    for frame in data.chunks(channels) {
        let mut acc = 0.0f32;
        for &s in frame {
            acc += f32::from_sample(s);
        }
        let mono = acc / channels as f32;
        if producer.push(mono).is_err() {
            overruns.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The input callback counts the samples it drops when the ring is full.
    #[test]
    fn capture_write_counts_overruns_when_full() {
        let (mut producer, _consumer) = RingBuffer::<f32>::new(4); // holds 4 samples.
        let overruns = AtomicUsize::new(0);

        // 10 mono frames (channels = 1); only 4 fit, so 6 must overrun.
        capture_write(&[0.0f32; 10], 1, &mut producer, &overruns);

        assert_eq!(overruns.load(Ordering::Relaxed), 6, "6 dropped on a 4-slot ring");
    }
}
