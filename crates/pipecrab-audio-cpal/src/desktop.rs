//! The desktop implementation, gated to macOS/Windows/Linux by [`crate`].
//!
//! One [`rtrb`] ring per direction bridges cpal's real-time callback thread and
//! the async pipeline. The callback is the ring's *producer* on capture and its
//! *consumer* on playback; it only pushes/pops `f32` and wakes an
//! [`AtomicWaker`], never blocking or allocating. `CpalSource`/`CpalSink` own
//! the opposite end plus the live [`cpal::Stream`] (dropping the stream stops
//! the device).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::Poll;

use async_trait::async_trait;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample};
use futures::future::poll_fn;
use futures::task::AtomicWaker;
use rtrb::{Consumer, Producer, RingBuffer};

use pipecrab_audio::{AudioChunk, AudioError, AudioFormat, AudioSink, AudioSource};

/// Chunk duration at the device rate: `rate * CHUNK_MS / 1000` frames (~20 ms).
const CHUNK_MS: u32 = 20;
/// Ring capacity in chunks — enough to absorb scheduling jitter while bounding
/// latency (`RING_CHUNKS * CHUNK_MS` ms worst case).
const RING_CHUNKS: usize = 8;

/// Captures mono `f32` audio from the default input device.
///
/// The cpal input callback downmixes each frame to mono and pushes it into a
/// ring; [`next_chunk`](AudioSource::next_chunk) pops a ~20 ms chunk once one is
/// buffered, otherwise parks on the waker until the callback signals more.
pub struct CpalSource {
    name: String,
    format: AudioFormat,
    chunk_frames: usize,
    consumer: Consumer<f32>,
    waker: Arc<AtomicWaker>,
    closed: Arc<AtomicBool>,
    _stream: cpal::Stream,
}

impl CpalSource {
    /// Open the default input device and start capturing at its default rate.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::Device`] if there is no default input device, its
    /// config can't be read, its sample format is unsupported, or the stream
    /// fails to build or start.
    pub fn new() -> Result<Self, AudioError> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| AudioError::Device("no default input device".into()))?;
        let name = device.name().unwrap_or_else(|_| "<unknown input>".into());
        let supported = device
            .default_input_config()
            .map_err(|e| AudioError::Device(format!("default input config: {e}")))?;
        let sample_format = supported.sample_format();
        let config: cpal::StreamConfig = supported.into();
        let sample_rate = config.sample_rate.0;
        let channels = config.channels as usize;
        let chunk_frames = (sample_rate * CHUNK_MS / 1000) as usize;

        let (producer, consumer) = RingBuffer::<f32>::new(chunk_frames * RING_CHUNKS);
        let waker = Arc::new(AtomicWaker::new());
        let closed = Arc::new(AtomicBool::new(false));

        // Only one match arm runs, so moving `producer` into each is fine.
        let stream = match sample_format {
            SampleFormat::F32 => build_capture_stream::<f32>(
                &device, &config, producer, waker.clone(), closed.clone(), channels,
            ),
            SampleFormat::I16 => build_capture_stream::<i16>(
                &device, &config, producer, waker.clone(), closed.clone(), channels,
            ),
            SampleFormat::U16 => build_capture_stream::<u16>(
                &device, &config, producer, waker.clone(), closed.clone(), channels,
            ),
            other => {
                return Err(AudioError::Device(format!("unsupported input sample format: {other:?}")))
            }
        }
        .map_err(|e| AudioError::Device(format!("build input stream: {e}")))?;
        stream.play().map_err(|e| AudioError::Device(format!("start input stream: {e}")))?;

        Ok(Self {
            name,
            format: AudioFormat::new(sample_rate, 1),
            chunk_frames,
            consumer,
            waker,
            closed,
            _stream: stream,
        })
    }

    /// The name of the input device audio is being captured from.
    pub fn device_name(&self) -> &str {
        &self.name
    }

    /// Number of frames (samples per channel) in each chunk this source yields.
    pub fn chunk_frames(&self) -> usize {
        self.chunk_frames
    }
}

#[async_trait(?Send)]
impl AudioSource for CpalSource {
    fn format(&self) -> AudioFormat {
        self.format
    }

    async fn next_chunk(&mut self) -> Option<AudioChunk> {
        poll_fn(move |cx| {
            let n = self.chunk_frames;
            if self.consumer.slots() >= n {
                return Poll::Ready(Some(collect_chunk(&mut self.consumer, n, self.format)));
            }
            // Arm the waker, then re-check: the callback may have filled the ring
            // between the check above and this registration (avoids a lost wakeup).
            self.waker.register(cx.waker());
            if self.consumer.slots() >= n {
                return Poll::Ready(Some(collect_chunk(&mut self.consumer, n, self.format)));
            }
            if self.closed.load(Ordering::Acquire) {
                return Poll::Ready(None); // stream died; a partial chunk is dropped.
            }
            Poll::Pending
        })
        .await
    }
}

/// Plays mono `f32` audio to the default output device.
///
/// [`play`](AudioSink::play) pushes a chunk's samples into a ring, `.await`ing
/// for room when it is full (backpressure paces the caller to real time); the
/// cpal output callback pops one mono sample per frame, duplicates it across the
/// device's channels, and outputs silence on underrun.
pub struct CpalSink {
    name: String,
    format: AudioFormat,
    producer: Producer<f32>,
    waker: Arc<AtomicWaker>,
    closed: Arc<AtomicBool>,
    _stream: cpal::Stream,
}

impl CpalSink {
    /// Open the default output device and start playback at its default rate.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::Device`] if there is no default output device, its
    /// config can't be read, its sample format is unsupported, or the stream
    /// fails to build or start.
    pub fn new() -> Result<Self, AudioError> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| AudioError::Device("no default output device".into()))?;
        let name = device.name().unwrap_or_else(|_| "<unknown output>".into());
        let supported = device
            .default_output_config()
            .map_err(|e| AudioError::Device(format!("default output config: {e}")))?;
        let sample_format = supported.sample_format();
        let config: cpal::StreamConfig = supported.into();
        let sample_rate = config.sample_rate.0;
        let channels = config.channels as usize;
        let chunk_frames = (sample_rate * CHUNK_MS / 1000) as usize;

        let (producer, consumer) = RingBuffer::<f32>::new(chunk_frames * RING_CHUNKS);
        let waker = Arc::new(AtomicWaker::new());
        let closed = Arc::new(AtomicBool::new(false));

        // Only one match arm runs, so moving `consumer` into each is fine.
        let stream = match sample_format {
            SampleFormat::F32 => build_playback_stream::<f32>(
                &device, &config, consumer, waker.clone(), closed.clone(), channels,
            ),
            SampleFormat::I16 => build_playback_stream::<i16>(
                &device, &config, consumer, waker.clone(), closed.clone(), channels,
            ),
            SampleFormat::U16 => build_playback_stream::<u16>(
                &device, &config, consumer, waker.clone(), closed.clone(), channels,
            ),
            other => {
                return Err(AudioError::Device(format!("unsupported output sample format: {other:?}")))
            }
        }
        .map_err(|e| AudioError::Device(format!("build output stream: {e}")))?;
        stream.play().map_err(|e| AudioError::Device(format!("start output stream: {e}")))?;

        Ok(Self {
            name,
            format: AudioFormat::new(sample_rate, 1),
            producer,
            waker,
            closed,
            _stream: stream,
        })
    }

    /// The name of the output device audio is being played to.
    pub fn device_name(&self) -> &str {
        &self.name
    }
}

#[async_trait(?Send)]
impl AudioSink for CpalSink {
    fn format(&self) -> AudioFormat {
        self.format
    }

    async fn play(&mut self, chunk: AudioChunk) -> Result<(), AudioError> {
        let samples = chunk.samples;
        let mut offset = 0usize;
        poll_fn(move |cx| {
            if self.closed.load(Ordering::Acquire) {
                return Poll::Ready(Err(AudioError::Closed));
            }
            push_available(&mut self.producer, &samples, &mut offset);
            if offset >= samples.len() {
                return Poll::Ready(Ok(()));
            }
            // Ring full: arm the waker, then retry once in case the callback
            // drained after the push loop above (avoids a lost wakeup).
            self.waker.register(cx.waker());
            push_available(&mut self.producer, &samples, &mut offset);
            if offset >= samples.len() {
                Poll::Ready(Ok(()))
            } else if self.closed.load(Ordering::Acquire) {
                Poll::Ready(Err(AudioError::Closed))
            } else {
                Poll::Pending
            }
        })
        .await
    }
}

/// Pop up to `frames` mono samples out of the ring into a fresh chunk. The
/// caller has already checked that at least `frames` are available.
fn collect_chunk(consumer: &mut Consumer<f32>, frames: usize, format: AudioFormat) -> AudioChunk {
    let mut samples = Vec::with_capacity(frames);
    for _ in 0..frames {
        match consumer.pop() {
            Ok(s) => samples.push(s),
            Err(_) => break,
        }
    }
    AudioChunk::new(Arc::from(samples), format)
}

/// Push `samples[offset..]` into the ring until it is full, advancing `offset`.
fn push_available(producer: &mut Producer<f32>, samples: &[f32], offset: &mut usize) {
    while *offset < samples.len() {
        match producer.push(samples[*offset]) {
            Ok(()) => *offset += 1,
            Err(_) => break, // full
        }
    }
}

/// Build the input stream for a device whose native sample type is `T`,
/// downmixing to mono `f32` in the callback.
fn build_capture_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    mut producer: Producer<f32>,
    waker: Arc<AtomicWaker>,
    closed: Arc<AtomicBool>,
    channels: usize,
) -> Result<cpal::Stream, cpal::BuildStreamError>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    device.build_input_stream::<T, _, _>(
        config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            for frame in data.chunks(channels) {
                let mut acc = 0.0f32;
                for &s in frame {
                    acc += f32::from_sample(s);
                }
                let mono = acc / channels as f32;
                let _ = producer.push(mono); // drop on overrun: the RT thread never blocks.
            }
            waker.wake();
        },
        move |err| {
            eprintln!("pipecrab-audio-cpal: input stream error: {err}");
            closed.store(true, Ordering::Release);
        },
        None,
    )
}

/// Build the output stream for a device whose native sample type is `T`,
/// duplicating each mono `f32` sample across the device's channels.
fn build_playback_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    mut consumer: Consumer<f32>,
    waker: Arc<AtomicWaker>,
    closed: Arc<AtomicBool>,
    channels: usize,
) -> Result<cpal::Stream, cpal::BuildStreamError>
where
    T: SizedSample + FromSample<f32>,
{
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
            waker.wake(); // freed room -> wake a play() blocked on backpressure.
        },
        move |err| {
            eprintln!("pipecrab-audio-cpal: output stream error: {err}");
            closed.store(true, Ordering::Release);
        },
        None,
    )
}
