//! Backend-independent audio resampling and its pipeline stage adapter.

use std::fmt;
use std::sync::Mutex;

use pipecrab_core::{
    AudioChunk, AudioFormat, DataFrame, Decision, Direction, Processor, SystemFrame,
};
use pipecrab_runtime::{MaybeSend, Outbound, Stage, StageError, maybe_async_trait};

use crate::rubato_sinc::RubatoSincResampler;

/// A synchronous, stateful audio-to-audio format converter.
///
/// Implementations own their streaming state. One input chunk may produce one
/// output chunk or no output yet when the implementation must buffer more
/// samples. Every produced chunk must use [`output_format`](Self::output_format).
///
/// [`reset`](Self::reset) is a non-blocking, infallible control call. It must
/// discard buffered input and filter history so audio processed afterward is
/// independent of audio processed before the reset.
pub trait Resampler: MaybeSend {
    /// The fixed format attached to every output chunk.
    fn output_format(&self) -> AudioFormat;

    /// Convert one input chunk, or return `None` while buffering it.
    fn resample(&mut self, input: &AudioChunk) -> Result<Option<AudioChunk>, ResamplerError>;

    /// Clear all streaming state.
    fn reset(&mut self);
}

/// Adapts any [`Resampler`] into a pipeline [`Stage`].
///
/// `ResamplerStage<R>` consumes each [`DataFrame::Audio`] and emits the chunk
/// returned by `R`. A buffered input produces no frame. Non-audio data and all
/// system frames pass through unchanged; [`SystemFrame::Interrupt`] also resets
/// the backend before it is forwarded.
///
/// `R` defaults to [`RubatoSincResampler`], so
/// [`ResamplerStage::new`](ResamplerStage::new) is the ergonomic production
/// constructor. Use [`with_resampler`](Self::with_resampler) to inject another
/// implementation.
///
/// # Orchestrator occupancy
///
/// Resampling happens synchronously in [`Processor::decide_data`]. This keeps
/// all mutable DSP state in the non-cancellable half of a stage, but it also
/// means one conversion occupies the orchestrator until it finishes. The
/// default backend uses bounded internal blocks and reusable buffers, and the
/// crate benchmark asserts an upper bound relative to audio duration.
/// TODO: Add an asynchronous resampler adapter backed by a shared worker so DSP
/// does not occupy the orchestrator thread.
pub struct ResamplerStage<R: Resampler = RubatoSincResampler> {
    output_format: AudioFormat,
    // A backend only needs Send, while Stage requires Sync because perform
    // borrows &self. decide_data has exclusive access and uses get_mut, so the
    // synchronous hot path never acquires the mutex.
    resampler: Mutex<R>,
}

impl ResamplerStage<RubatoSincResampler> {
    /// Create a stage using the bundled windowed-sinc implementation.
    pub fn new(output_format: AudioFormat) -> Result<Self, ResamplerError> {
        Self::with_resampler(RubatoSincResampler::new(output_format)?)
    }
}

impl<R: Resampler> ResamplerStage<R> {
    /// Create a stage using `resampler` as its audio conversion backend.
    pub fn with_resampler(resampler: R) -> Result<Self, ResamplerError> {
        let output_format = resampler.output_format();
        validate_format(output_format)?;
        Ok(Self {
            output_format,
            resampler: Mutex::new(resampler),
        })
    }

    /// The fixed format attached to every output chunk.
    pub fn output_format(&self) -> AudioFormat {
        self.output_format
    }
}

/// A conversion command emitted by [`ResamplerStage`].
///
/// Its payload is private because only the stage constructs and interprets it.
pub struct ResamplerEffect(Result<AudioChunk, ResamplerError>);

impl<R: Resampler> Processor for ResamplerStage<R> {
    type Effect = ResamplerEffect;

    fn decide_data(&mut self, frame: &DataFrame) -> Decision<Self::Effect> {
        let DataFrame::Audio(chunk) = frame else {
            return Decision::forward();
        };
        let result = self
            .resampler
            .get_mut()
            .expect("resampler mutex poisoned")
            .resample(chunk);
        match result {
            Ok(Some(chunk)) => Decision::drop().emit(ResamplerEffect(Ok(chunk))),
            Ok(None) => Decision::drop(),
            Err(error) => Decision::drop().emit(ResamplerEffect(Err(error))),
        }
    }

    fn decide_system(
        &mut self,
        _direction: Direction,
        frame: &SystemFrame,
    ) -> Decision<Self::Effect> {
        if matches!(frame, SystemFrame::Interrupt) {
            self.resampler
                .get_mut()
                .expect("resampler mutex poisoned")
                .reset();
        }
        Decision::forward()
    }
}

maybe_async_trait! {
    impl<R: Resampler> Stage for ResamplerStage<R> {
        async fn perform(
            &self,
            ResamplerEffect(result): ResamplerEffect,
            out: &Outbound,
        ) -> Result<(), StageError> {
            let chunk = result.map_err(|error| StageError::fatal(error.to_string()))?;
            let _ = out.send_data(DataFrame::Audio(chunk)).await;
            Ok(())
        }
    }
}

/// Why an audio chunk could not be converted by a [`Resampler`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResamplerError {
    /// A sample rate or channel count was zero.
    InvalidFormat {
        /// The unusable format.
        format: AudioFormat,
    },
    /// An interleaved buffer did not contain whole audio frames.
    MisalignedSamples {
        /// Number of samples in the buffer.
        samples: usize,
        /// Number of interleaved channels declared by the chunk.
        channels: u16,
    },
    /// A resampling implementation rejected an operation.
    Resampling(String),
}

impl fmt::Display for ResamplerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFormat { format } => write!(
                formatter,
                "invalid audio format: sample rate and channels must be non-zero, got {} Hz/{} ch",
                format.sample_rate, format.channels
            ),
            Self::MisalignedSamples { samples, channels } => write!(
                formatter,
                "audio buffer has {samples} samples, which is not divisible by {channels} channels"
            ),
            Self::Resampling(message) => write!(formatter, "audio resampling failed: {message}"),
        }
    }
}

impl std::error::Error for ResamplerError {}

pub(crate) fn validate_format(format: AudioFormat) -> Result<(), ResamplerError> {
    if format.sample_rate == 0 || format.channels == 0 {
        Err(ResamplerError::InvalidFormat { format })
    } else {
        Ok(())
    }
}

pub(crate) fn validate_chunk(chunk: &AudioChunk) -> Result<(), ResamplerError> {
    validate_format(chunk.format)?;
    if chunk.samples.len() % usize::from(chunk.format.channels) != 0 {
        return Err(ResamplerError::MisalignedSamples {
            samples: chunk.samples.len(),
            channels: chunk.format.channels,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use pipecrab_core::Disposition;

    use super::*;

    struct Gain {
        output_format: AudioFormat,
        resets: Arc<AtomicUsize>,
    }

    impl Resampler for Gain {
        fn output_format(&self) -> AudioFormat {
            self.output_format
        }

        fn resample(&mut self, input: &AudioChunk) -> Result<Option<AudioChunk>, ResamplerError> {
            let samples: Arc<[f32]> = input.samples.iter().map(|sample| sample * 2.0).collect();
            Ok(Some(AudioChunk::new(samples, self.output_format)))
        }

        fn reset(&mut self) {
            self.resets.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn custom_resampler_drives_the_generic_stage() {
        let output_format = AudioFormat::new(24_000, 1);
        let resets = Arc::new(AtomicUsize::new(0));
        let mut stage = ResamplerStage::with_resampler(Gain {
            output_format,
            resets: resets.clone(),
        })
        .unwrap();
        let frame = DataFrame::Audio(AudioChunk::new(
            Arc::from([0.25, -0.5]),
            AudioFormat::new(48_000, 1),
        ));

        let decision = stage.decide_data(&frame);
        assert_eq!(decision.disposition, Disposition::Drop);
        let chunk = decision.effects.into_iter().next().unwrap().0.unwrap();
        assert_eq!(chunk.format, output_format);
        assert_eq!(&*chunk.samples, &[0.5, -1.0]);

        stage.decide_system(Direction::Down, &SystemFrame::Interrupt);
        assert_eq!(resets.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn rejects_a_custom_resampler_with_an_invalid_output_format() {
        let result = ResamplerStage::with_resampler(Gain {
            output_format: AudioFormat::new(0, 1),
            resets: Arc::new(AtomicUsize::new(0)),
        });
        assert!(matches!(result, Err(ResamplerError::InvalidFormat { .. })));
    }
}
