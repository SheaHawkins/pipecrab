//! The lock-free real-time ↔ async boundary, deliberately independent of cpal's
//! `Stream`/`Device` so it can be unit-tested without any audio hardware.
//!
//! An [`rtrb`] ring carries `f32` samples one way; a [`Signal`] carries wakeups
//! and a "the stream died" flag the other way. The cpal callback owns the RT end
//! of both; [`CaptureRing`] / [`PlaybackRing`] own the async end and the stream
//! shell ([`crate::desktop`]) wires them to a real device.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use cpal::{FromSample, Sample};
use futures::future::poll_fn;
use futures::task::AtomicWaker;
use rtrb::{Consumer, Producer};

use pipecrab_audio::{AudioChunk, AudioError, AudioFormat};

/// Wake + close signal shared between an RT callback and the async side.
///
/// The async side [`register`](Self::register)s its task waker; the RT side
/// [`wake`](Self::wake)s after moving samples, or [`fail`](Self::fail)s when the
/// stream errors — which both flips the closed flag *and* wakes, so a parked
/// task can never hang after a device failure.
pub(crate) struct Signal {
    waker: AtomicWaker,
    closed: AtomicBool,
}

impl Signal {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self { waker: AtomicWaker::new(), closed: AtomicBool::new(false) })
    }

    /// Async side: arm the waker to be notified of the next `wake`/`fail`.
    pub(crate) fn register(&self, waker: &Waker) {
        self.waker.register(waker);
    }

    /// RT side: notify the async task that progress may be possible.
    pub(crate) fn wake(&self) {
        self.waker.wake();
    }

    /// RT side: mark the stream failed and wake, so a parked task resolves to an
    /// error instead of hanging forever.
    pub(crate) fn fail(&self) {
        self.closed.store(true, Ordering::Release);
        self.waker.wake();
    }

    pub(crate) fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }
}

/// The async/consumer half of the capture bridge: pops fixed-size mono chunks
/// out of the ring the input callback fills.
pub(crate) struct CaptureRing {
    consumer: Consumer<f32>,
    signal: Arc<Signal>,
    overruns: Arc<AtomicUsize>,
    chunk_frames: usize,
    format: AudioFormat,
}

impl CaptureRing {
    pub(crate) fn new(
        consumer: Consumer<f32>,
        signal: Arc<Signal>,
        overruns: Arc<AtomicUsize>,
        chunk_frames: usize,
        format: AudioFormat,
    ) -> Self {
        Self { consumer, signal, overruns, chunk_frames, format }
    }

    pub(crate) fn format(&self) -> AudioFormat {
        self.format
    }

    pub(crate) fn chunk_frames(&self) -> usize {
        self.chunk_frames
    }

    /// Samples the input callback has dropped because the ring was full.
    pub(crate) fn overruns(&self) -> usize {
        self.overruns.load(Ordering::Relaxed)
    }

    pub(crate) async fn next_chunk(&mut self) -> Result<Option<AudioChunk>, AudioError> {
        poll_fn(|cx| self.poll_next_chunk(cx)).await
    }

    /// `Ready(Ok(Some))` once a whole chunk is buffered, `Ready(Err(Closed))` if
    /// the stream has failed, else `Pending` with the waker armed.
    fn poll_next_chunk(&mut self, cx: &mut Context<'_>) -> Poll<Result<Option<AudioChunk>, AudioError>> {
        if self.consumer.slots() >= self.chunk_frames {
            return Poll::Ready(Ok(Some(self.collect())));
        }
        // Arm the waker, then re-check: the callback may have filled the ring
        // between the check above and this registration (avoids a lost wakeup).
        self.signal.register(cx.waker());
        if self.consumer.slots() >= self.chunk_frames {
            return Poll::Ready(Ok(Some(self.collect())));
        }
        if self.signal.is_closed() {
            // A live device never ends gracefully, so a closed stream is an
            // error, not `Ok(None)`; any partial chunk in the ring is dropped.
            return Poll::Ready(Err(AudioError::Closed));
        }
        Poll::Pending
    }

    fn collect(&mut self) -> AudioChunk {
        let mut samples = Vec::with_capacity(self.chunk_frames);
        for _ in 0..self.chunk_frames {
            match self.consumer.pop() {
                Ok(s) => samples.push(s),
                Err(_) => break,
            }
        }
        AudioChunk::new(Arc::from(samples), self.format)
    }
}

/// The async/producer half of the playback bridge: pushes a chunk's samples into
/// the ring the output callback drains, applying backpressure when it is full.
pub(crate) struct PlaybackRing {
    producer: Producer<f32>,
    signal: Arc<Signal>,
    format: AudioFormat,
}

impl PlaybackRing {
    pub(crate) fn new(producer: Producer<f32>, signal: Arc<Signal>, format: AudioFormat) -> Self {
        Self { producer, signal, format }
    }

    pub(crate) fn format(&self) -> AudioFormat {
        self.format
    }

    pub(crate) async fn play(&mut self, chunk: AudioChunk) -> Result<(), AudioError> {
        // Reject mismatched audio up front: this sink does not resample, so
        // playing a chunk in the wrong rate/channel count would be a silent
        // pitch/speed bug. The caller must resample to the sink's format first.
        if chunk.format != self.format {
            return Err(AudioError::FormatMismatch { expected: self.format, got: chunk.format });
        }
        let samples = chunk.samples;
        let mut offset = 0usize;
        poll_fn(|cx| self.poll_push(&samples, &mut offset, cx)).await
    }

    /// Push `samples[*offset..]` into the ring; `Ready(Ok)` once all are in,
    /// `Ready(Err(Closed))` if the stream failed, else `Pending` (backpressure)
    /// with the waker armed for the output callback's next drain.
    fn poll_push(&mut self, samples: &[f32], offset: &mut usize, cx: &mut Context<'_>) -> Poll<Result<(), AudioError>> {
        if self.signal.is_closed() {
            return Poll::Ready(Err(AudioError::Closed));
        }
        push_available(&mut self.producer, samples, offset);
        if *offset >= samples.len() {
            return Poll::Ready(Ok(()));
        }
        // Ring full: arm the waker, then retry once in case the callback drained
        // after the push above (avoids a lost wakeup).
        self.signal.register(cx.waker());
        push_available(&mut self.producer, samples, offset);
        if *offset >= samples.len() {
            Poll::Ready(Ok(()))
        } else if self.signal.is_closed() {
            Poll::Ready(Err(AudioError::Closed))
        } else {
            Poll::Pending
        }
    }
}

/// Push `samples[*offset..]` into the ring until it is full, advancing `*offset`.
fn push_available(producer: &mut Producer<f32>, samples: &[f32], offset: &mut usize) {
    while *offset < samples.len() {
        match producer.push(samples[*offset]) {
            Ok(()) => *offset += 1,
            Err(_) => break, // full
        }
    }
}

/// The input callback's per-buffer work: downmix each interleaved frame to mono
/// `f32`, push it into the ring, and count samples dropped when the ring is full
/// (an overrun the async side can observe via [`CaptureRing::overruns`]).
pub(crate) fn capture_write<T>(
    data: &[T],
    channels: usize,
    producer: &mut Producer<f32>,
    overruns: &AtomicUsize,
) where
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
    use futures::executor::block_on;
    use futures::task::{waker, ArcWake};
    use rtrb::RingBuffer;

    /// A `Waker` that counts how many times it is woken.
    struct CountingWaker(AtomicUsize);
    impl CountingWaker {
        fn new() -> Arc<Self> {
            Arc::new(Self(AtomicUsize::new(0)))
        }
        fn count(&self) -> usize {
            self.0.load(Ordering::SeqCst)
        }
    }
    impl ArcWake for CountingWaker {
        fn wake_by_ref(arc: &Arc<Self>) {
            arc.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    const FMT: AudioFormat = AudioFormat { sample_rate: 48_000, channels: 1 };

    // #1 (capture): a stream error must wake a parked `next_chunk`, and the next
    // poll must resolve to an error rather than hang.
    #[test]
    fn capture_error_wakes_parked_next_chunk() {
        let (_producer, consumer) = RingBuffer::<f32>::new(64);
        let signal = Signal::new();
        let overruns = Arc::new(AtomicUsize::new(0));
        let mut ring = CaptureRing::new(consumer, signal.clone(), overruns, 8, FMT);

        let cw = CountingWaker::new();
        let w = waker(cw.clone());
        let mut cx = Context::from_waker(&w);

        // Empty ring, not closed -> parks and arms the waker.
        assert!(matches!(ring.poll_next_chunk(&mut cx), Poll::Pending));
        assert_eq!(cw.count(), 0);

        // Simulate the cpal error callback.
        signal.fail();
        assert_eq!(cw.count(), 1, "a stream error must wake the parked next_chunk");

        assert!(matches!(
            ring.poll_next_chunk(&mut cx),
            Poll::Ready(Err(AudioError::Closed))
        ));
    }

    // #1 (playback): a stream error must wake a `play` parked on backpressure,
    // and the next poll must resolve to an error rather than hang.
    #[test]
    fn playback_error_wakes_parked_play() {
        let (producer, _consumer) = RingBuffer::<f32>::new(2); // tiny: fills fast.
        let signal = Signal::new();
        let mut ring = PlaybackRing::new(producer, signal.clone(), FMT);

        let samples = [0.0f32; 4]; // more than the ring holds.
        let mut offset = 0;
        let cw = CountingWaker::new();
        let w = waker(cw.clone());
        let mut cx = Context::from_waker(&w);

        // Pushes 2, then parks on a full ring with the waker armed.
        assert!(matches!(ring.poll_push(&samples, &mut offset, &mut cx), Poll::Pending));
        assert_eq!(offset, 2);

        signal.fail();
        assert!(cw.count() >= 1, "a stream error must wake the parked play");

        assert!(matches!(
            ring.poll_push(&samples, &mut offset, &mut cx),
            Poll::Ready(Err(AudioError::Closed))
        ));
    }

    // #2: play rejects a chunk whose format differs from the sink's, and accepts
    // one that matches.
    #[test]
    fn play_rejects_format_mismatch() {
        let (producer, _consumer) = RingBuffer::<f32>::new(64);
        let mut ring = PlaybackRing::new(producer, Signal::new(), FMT);

        let wrong_fmt = AudioFormat::new(16_000, 1);
        let wrong = AudioChunk::new(Arc::from(&[0.0f32; 4][..]), wrong_fmt);
        assert_eq!(
            block_on(ring.play(wrong)),
            Err(AudioError::FormatMismatch { expected: FMT, got: wrong_fmt }),
        );

        // Matching format is accepted (fits comfortably in the ring).
        let right = AudioChunk::new(Arc::from(&[0.1f32; 4][..]), FMT);
        assert_eq!(block_on(ring.play(right)), Ok(()));
    }

    // #3: the input callback counts samples it drops when the ring is full.
    #[test]
    fn capture_write_counts_overruns_when_full() {
        let (mut producer, _consumer) = RingBuffer::<f32>::new(4); // holds 4 samples.
        let overruns = Arc::new(AtomicUsize::new(0));

        // 10 mono frames (channels = 1); only 4 fit, so 6 must overrun.
        let data = [0.0f32; 10];
        capture_write(&data, 1, &mut producer, &overruns);

        assert_eq!(overruns.load(Ordering::Relaxed), 6, "6 dropped samples on a 4-slot ring");
    }
}
