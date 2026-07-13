//! A streaming sample-rate and channel-count conversion stage.

use std::collections::VecDeque;
use std::fmt;
use std::sync::Mutex;

use pipecrab_core::{
    AudioChunk, AudioFormat, DataFrame, Decision, Direction, Processor, SystemFrame,
};
use pipecrab_runtime::{maybe_async_trait, Outbound, Stage, StageError};
use rubato::{
    calculate_cutoff, Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};

const BLOCK_FRAMES: usize = 64;
const SINC_LENGTH: usize = 128;
const OVERSAMPLING_FACTOR: usize = 128;

/// Converts every [`DataFrame::Audio`] to one fixed [`AudioFormat`].
///
/// The stage is transparent to non-audio data and to the system lane. Audio
/// already in `output_format` is forwarded without copying its sample buffer.
/// Other audio is consumed and replaced by a converted [`AudioChunk`].
///
/// Conversion is continuous across chunks with the same input format: the
/// windowed-sinc filter retains its delay line and fractional position. An
/// input-format change starts a new stream, and [`SystemFrame::Interrupt`]
/// resets the current stream so pre-interrupt samples cannot bleed into the
/// next utterance.
///
/// Sample-rate conversion advances in 64-frame blocks. A smaller input chunk
/// can therefore be consumed without immediately producing an output frame;
/// its samples remain buffered and are emitted when later audio completes the
/// block. A stream reset discards that incomplete block together with the sinc
/// filter history.
///
/// # Channel mapping
///
/// Equal channel counts are resampled independently. When the counts differ,
/// input channels are averaged to mono before resampling and the result is
/// replicated to every output channel. [`AudioFormat`] carries a channel count
/// but no speaker layout, so a more specific multichannel matrix cannot be
/// inferred safely.
///
/// # Orchestrator occupancy
///
/// Resampling happens synchronously in [`Processor::decide_data`]. This keeps
/// all mutable DSP state in the non-cancellable half of a stage, but it also
/// means one conversion occupies the orchestrator until it finishes. Work is
/// split into fixed-size internal blocks, buffers and filter state are reused,
/// and the crate benchmark asserts an upper bound relative to audio duration.
pub struct ResamplerStage {
    output_format: AudioFormat,
    current_input: Option<AudioFormat>,
    // Rubato's resampler is Send but not Sync. Stage requires Sync because
    // perform borrows &self, although only decide_data touches this state. The
    // exclusive &mut self there lets us use Mutex::get_mut without locking.
    rate_state: Mutex<Option<RateState>>,
}

impl ResamplerStage {
    /// Create a stage which emits audio in `output_format`.
    ///
    /// A zero sample rate or channel count is rejected because neither denotes
    /// a usable PCM format.
    pub fn new(output_format: AudioFormat) -> Result<Self, ResamplerError> {
        validate_format(output_format)?;
        Ok(Self {
            output_format,
            current_input: None,
            rate_state: Mutex::new(None),
        })
    }

    /// The fixed format attached to every converted output chunk.
    pub fn output_format(&self) -> AudioFormat {
        self.output_format
    }

    fn reset_stream(&mut self) {
        self.current_input = None;
        if let Some(state) = self
            .rate_state
            .get_mut()
            .expect("resampler state mutex poisoned")
        {
            state.reset();
        }
    }

    fn convert(&mut self, chunk: &AudioChunk) -> Result<AudioChunk, ResamplerError> {
        validate_chunk(chunk)?;

        let input_format = chunk.format;
        if self.current_input != Some(input_format) {
            self.current_input = Some(input_format);
            if let Some(state) = self
                .rate_state
                .get_mut()
                .expect("resampler state mutex poisoned")
            {
                state.reset();
            }
        }

        if input_format.sample_rate == self.output_format.sample_rate {
            let samples = remix_channels(
                &chunk.samples,
                input_format.channels,
                self.output_format.channels,
            );
            return Ok(AudioChunk::new(samples.into(), self.output_format));
        }

        let working_channels = if input_format.channels == self.output_format.channels {
            input_format.channels
        } else {
            1
        };
        let rate_state = self
            .rate_state
            .get_mut()
            .expect("resampler state mutex poisoned");
        let state_matches = rate_state.as_ref().is_some_and(|state| {
            state.input_rate == input_format.sample_rate
                && state.working_channels == working_channels
        });
        if !state_matches {
            *rate_state = Some(RateState::new(
                input_format.sample_rate,
                self.output_format.sample_rate,
                working_channels,
            )?);
        }

        let state = rate_state
            .as_mut()
            .expect("rate state was constructed above");
        let input_channels = usize::from(input_format.channels);
        let output_channels = usize::from(self.output_format.channels);
        let working_channels = usize::from(working_channels);
        let input_frames = chunk.samples.len() / input_channels;
        let ratio = f64::from(self.output_format.sample_rate) / f64::from(input_format.sample_rate);
        let estimated_frames = ((input_frames + BLOCK_FRAMES) as f64 * ratio).ceil() as usize + 16;
        let mut converted = Vec::with_capacity(estimated_frames * output_channels);

        if working_channels == input_channels {
            for channel in 0..working_channels {
                for frame in 0..input_frames {
                    state.pending[channel]
                        .push_back(chunk.samples[frame * input_channels + channel]);
                }
            }
        } else {
            let scale = 1.0 / input_channels as f32;
            for frame in 0..input_frames {
                let offset = frame * input_channels;
                let mono = chunk.samples[offset..offset + input_channels]
                    .iter()
                    .copied()
                    .sum::<f32>()
                    * scale;
                state.pending[0].push_back(mono);
            }
        }

        while state.pending[0].len() >= BLOCK_FRAMES {
            for channel in 0..working_channels {
                for frame in 0..BLOCK_FRAMES {
                    state.input[channel][frame] = state.pending[channel]
                        .pop_front()
                        .expect("all working channels have equal length");
                }
            }

            let (_, output_frames) = state
                .resampler
                .process_into_buffer(&state.input, &mut state.output, None)
                .map_err(ResamplerError::from_rubato)?;

            if working_channels == output_channels {
                for frame in 0..output_frames {
                    for channel in 0..output_channels {
                        converted.push(state.output[channel][frame]);
                    }
                }
            } else {
                for frame in 0..output_frames {
                    converted
                        .extend(std::iter::repeat(state.output[0][frame]).take(output_channels));
                }
            }
        }

        Ok(AudioChunk::new(converted.into(), self.output_format))
    }
}

struct RateState {
    input_rate: u32,
    working_channels: u16,
    resampler: SincFixedIn<f32>,
    input: Vec<Vec<f32>>,
    output: Vec<Vec<f32>>,
    pending: Vec<VecDeque<f32>>,
}

impl RateState {
    fn new(
        input_rate: u32,
        output_rate: u32,
        working_channels: u16,
    ) -> Result<Self, ResamplerError> {
        let window = WindowFunction::Blackman2;
        let parameters = SincInterpolationParameters {
            sinc_len: SINC_LENGTH,
            f_cutoff: calculate_cutoff(SINC_LENGTH, window),
            oversampling_factor: OVERSAMPLING_FACTOR,
            interpolation: SincInterpolationType::Linear,
            window,
        };
        let ratio = f64::from(output_rate) / f64::from(input_rate);
        let resampler = SincFixedIn::new(
            ratio,
            1.0,
            parameters,
            BLOCK_FRAMES,
            usize::from(working_channels),
        )
        .map_err(ResamplerError::from_rubato)?;
        let input = resampler.input_buffer_allocate(true);
        let output = resampler.output_buffer_allocate(true);
        let pending = (0..usize::from(working_channels))
            .map(|_| VecDeque::with_capacity(BLOCK_FRAMES * 2))
            .collect();
        Ok(Self {
            input_rate,
            working_channels,
            resampler,
            input,
            output,
            pending,
        })
    }

    fn reset(&mut self) {
        self.resampler.reset();
        for channel in &mut self.pending {
            channel.clear();
        }
    }
}

/// A conversion command emitted by [`ResamplerStage`].
///
/// Its payload is private because only the stage constructs and interprets it.
pub struct ResamplerEffect(Result<AudioChunk, ResamplerError>);

impl Processor for ResamplerStage {
    type Effect = ResamplerEffect;

    fn decide_data(&mut self, frame: &DataFrame) -> Decision<Self::Effect> {
        match frame {
            DataFrame::Audio(chunk) if chunk.format == self.output_format => {
                if let Err(error) = validate_chunk(chunk) {
                    return Decision::drop().emit(ResamplerEffect(Err(error)));
                }
                if self.current_input != Some(chunk.format) {
                    self.reset_stream();
                    self.current_input = Some(chunk.format);
                }
                Decision::forward()
            }
            DataFrame::Audio(chunk) => match self.convert(chunk) {
                Ok(chunk) if chunk.samples.is_empty() => Decision::drop(),
                Ok(chunk) => Decision::drop().emit(ResamplerEffect(Ok(chunk))),
                Err(error) => Decision::drop().emit(ResamplerEffect(Err(error))),
            },
            _ => Decision::forward(),
        }
    }

    fn decide_system(
        &mut self,
        _direction: Direction,
        frame: &SystemFrame,
    ) -> Decision<Self::Effect> {
        if matches!(frame, SystemFrame::Interrupt) {
            self.reset_stream();
        }
        Decision::forward()
    }
}

maybe_async_trait! {
    impl Stage for ResamplerStage {
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

/// Why an audio chunk could not be converted by [`ResamplerStage`].
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
    /// The underlying sample-rate converter rejected an operation.
    Resampling(String),
}

impl ResamplerError {
    fn from_rubato(error: impl fmt::Display) -> Self {
        Self::Resampling(error.to_string())
    }
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

fn validate_format(format: AudioFormat) -> Result<(), ResamplerError> {
    if format.sample_rate == 0 || format.channels == 0 {
        Err(ResamplerError::InvalidFormat { format })
    } else {
        Ok(())
    }
}

fn validate_chunk(chunk: &AudioChunk) -> Result<(), ResamplerError> {
    validate_format(chunk.format)?;
    if chunk.samples.len() % usize::from(chunk.format.channels) != 0 {
        return Err(ResamplerError::MisalignedSamples {
            samples: chunk.samples.len(),
            channels: chunk.format.channels,
        });
    }
    Ok(())
}

fn remix_channels(samples: &[f32], input_channels: u16, output_channels: u16) -> Vec<f32> {
    let input_channels = usize::from(input_channels);
    let output_channels = usize::from(output_channels);
    let frames = samples.len() / input_channels;
    let mut output = Vec::with_capacity(frames * output_channels);
    let scale = 1.0 / input_channels as f32;
    for frame in 0..frames {
        let offset = frame * input_channels;
        let mono = samples[offset..offset + input_channels]
            .iter()
            .copied()
            .sum::<f32>()
            * scale;
        output.extend(std::iter::repeat(mono).take(output_channels));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use pipecrab_core::Disposition;
    use std::sync::Arc;

    fn audio(samples: &[f32], format: AudioFormat) -> DataFrame {
        DataFrame::Audio(AudioChunk::new(Arc::from(samples), format))
    }

    fn converted(decision: Decision<ResamplerEffect>) -> Result<AudioChunk, ResamplerError> {
        assert_eq!(decision.disposition, Disposition::Drop);
        assert_eq!(decision.effects.len(), 1);
        decision.effects.into_iter().next().unwrap().0
    }

    #[test]
    fn rejects_invalid_output_format() {
        assert!(matches!(
            ResamplerStage::new(AudioFormat::new(0, 1)),
            Err(ResamplerError::InvalidFormat { .. })
        ));
        assert!(matches!(
            ResamplerStage::new(AudioFormat::new(48_000, 0)),
            Err(ResamplerError::InvalidFormat { .. })
        ));
    }

    #[test]
    fn equal_format_forwards_the_original_frame() {
        let format = AudioFormat::new(48_000, 2);
        let samples: Arc<[f32]> = Arc::from([0.25, -0.25, 0.5, -0.5]);
        let frame = DataFrame::Audio(AudioChunk::new(samples, format));
        let mut stage = ResamplerStage::new(format).unwrap();

        let decision = stage.decide_data(&frame);

        assert_eq!(decision.disposition, Disposition::Forward);
        assert!(decision.effects.is_empty());
    }

    #[test]
    fn remixes_stereo_to_mono_without_rate_conversion() {
        let input = AudioFormat::new(48_000, 2);
        let output = AudioFormat::new(48_000, 1);
        let mut stage = ResamplerStage::new(output).unwrap();

        let chunk = converted(stage.decide_data(&audio(&[1.0, -1.0, 0.5, 0.25], input))).unwrap();

        assert_eq!(chunk.format, output);
        assert_eq!(&*chunk.samples, &[0.0, 0.375]);
    }

    #[test]
    fn remixes_mono_to_stereo_without_rate_conversion() {
        let input = AudioFormat::new(48_000, 1);
        let output = AudioFormat::new(48_000, 2);
        let mut stage = ResamplerStage::new(output).unwrap();

        let chunk = converted(stage.decide_data(&audio(&[0.25, -0.5], input))).unwrap();

        assert_eq!(&*chunk.samples, &[0.25, 0.25, -0.5, -0.5]);
    }

    #[test]
    fn rejects_misaligned_interleaved_audio() {
        let input = AudioFormat::new(48_000, 2);
        let mut stage = ResamplerStage::new(AudioFormat::new(16_000, 1)).unwrap();

        let error = converted(stage.decide_data(&audio(&[0.0, 1.0, 2.0], input))).unwrap_err();

        assert_eq!(
            error,
            ResamplerError::MisalignedSamples {
                samples: 3,
                channels: 2
            }
        );
    }

    #[test]
    fn rejects_misaligned_audio_on_the_fast_path() {
        let format = AudioFormat::new(48_000, 2);
        let mut stage = ResamplerStage::new(format).unwrap();

        let error = converted(stage.decide_data(&audio(&[0.0, 1.0, 2.0], format))).unwrap_err();

        assert!(matches!(
            error,
            ResamplerError::MisalignedSamples {
                samples: 3,
                channels: 2
            }
        ));
    }

    #[test]
    fn arbitrary_channel_change_uses_mono_as_the_neutral_layout() {
        let input = AudioFormat::new(48_000, 3);
        let output = AudioFormat::new(48_000, 2);
        let mut stage = ResamplerStage::new(output).unwrap();

        let chunk = converted(stage.decide_data(&audio(&[0.0, 0.5, 1.0], input))).unwrap();

        assert_eq!(&*chunk.samples, &[0.5, 0.5]);
    }

    #[test]
    fn sub_block_audio_is_buffered_until_conversion_can_advance() {
        let input = AudioFormat::new(48_000, 1);
        let output = AudioFormat::new(16_000, 1);
        let mut stage = ResamplerStage::new(output).unwrap();
        let short = vec![0.0; BLOCK_FRAMES / 2];

        let first = stage.decide_data(&audio(&short, input));
        assert_eq!(first.disposition, Disposition::Drop);
        assert!(first.effects.is_empty());

        let second = stage.decide_data(&audio(&short, input));
        assert_eq!(second.disposition, Disposition::Drop);
        // The sinc filter has startup delay, so this first complete internal
        // block may still yield no samples. The important contract is that the
        // two input halves are accepted as one continuous block.
        let pending = stage
            .rate_state
            .get_mut()
            .unwrap()
            .as_ref()
            .unwrap()
            .pending[0]
            .len();
        assert_eq!(pending, 0);
    }

    #[test]
    fn input_format_change_starts_a_fresh_stream() {
        let output = AudioFormat::new(16_000, 1);
        let first_input = AudioFormat::new(48_000, 1);
        let second_input = AudioFormat::new(24_000, 1);
        let mut changed = ResamplerStage::new(output).unwrap();
        converted(changed.decide_data(&audio(&vec![1.0; 960], first_input))).unwrap();
        let after_change = converted(changed.decide_data(&audio(&vec![0.0; 480], second_input)))
            .unwrap()
            .samples;

        let mut fresh = ResamplerStage::new(output).unwrap();
        let fresh_second = converted(fresh.decide_data(&audio(&vec![0.0; 480], second_input)))
            .unwrap()
            .samples;

        assert_eq!(after_change, fresh_second);
    }

    #[test]
    fn non_audio_and_system_frames_forward() {
        let mut stage = ResamplerStage::new(AudioFormat::new(16_000, 1)).unwrap();
        assert_eq!(
            stage.decide_data(&DataFrame::SpeechStarted).disposition,
            Disposition::Forward
        );
        assert_eq!(
            stage
                .decide_system(Direction::Down, &SystemFrame::Start)
                .disposition,
            Disposition::Forward
        );
    }

    #[test]
    fn downsampling_tracks_stream_duration_across_chunks() {
        let input = AudioFormat::new(48_000, 1);
        let output = AudioFormat::new(16_000, 1);
        let mut stage = ResamplerStage::new(output).unwrap();
        let samples = vec![0.0; 960];
        let chunks = 50;
        let mut output_frames = 0;

        for _ in 0..chunks {
            let chunk = converted(stage.decide_data(&audio(&samples, input))).unwrap();
            assert_eq!(chunk.format, output);
            output_frames += chunk.samples.len();
        }

        let expected = chunks * samples.len() * usize::from(output.channels)
            / usize::from(input.channels)
            * output.sample_rate as usize
            / input.sample_rate as usize;
        let delay = stage
            .rate_state
            .get_mut()
            .unwrap()
            .as_ref()
            .unwrap()
            .resampler
            .output_delay();
        assert!(output_frames.abs_diff(expected) <= delay + 2);
    }

    #[test]
    fn chunk_boundaries_do_not_change_the_stream() {
        let input = AudioFormat::new(48_000, 1);
        let output = AudioFormat::new(16_000, 1);
        let samples: Vec<f32> = (0..4_800)
            .map(|frame| {
                (std::f32::consts::TAU * 440.0 * frame as f32 / input.sample_rate as f32).sin()
            })
            .collect();

        let mut whole_stage = ResamplerStage::new(output).unwrap();
        let whole = converted(whole_stage.decide_data(&audio(&samples, input)))
            .unwrap()
            .samples;

        let mut split_stage = ResamplerStage::new(output).unwrap();
        let mut split = Vec::new();
        for part in samples.chunks(960) {
            split.extend(
                converted(split_stage.decide_data(&audio(part, input)))
                    .unwrap()
                    .samples
                    .iter()
                    .copied(),
            );
        }

        assert_eq!(split.len(), whole.len());
        for (left, right) in split.iter().zip(whole.iter()) {
            assert!((left - right).abs() < 1e-6, "{left} != {right}");
        }
    }

    #[test]
    fn sinc_filter_attenuates_frequencies_above_output_nyquist() {
        let input = AudioFormat::new(48_000, 1);
        let output = AudioFormat::new(16_000, 1);
        let low = resample_sine(input, output, 1_000.0);
        let high = resample_sine(input, output, 12_000.0);

        let low_rms = rms(&low[200..]);
        let high_rms = rms(&high[200..]);
        assert!(
            high_rms < low_rms * 0.05,
            "out-of-band RMS {high_rms} was not sufficiently below pass-band RMS {low_rms}"
        );
    }

    #[test]
    fn interrupt_resets_filter_history() {
        let input = AudioFormat::new(48_000, 1);
        let output = AudioFormat::new(16_000, 1);
        let mut interrupted = ResamplerStage::new(output).unwrap();
        let loud = vec![1.0; 960];
        let silence = vec![0.0; 960];
        converted(interrupted.decide_data(&audio(&loud, input))).unwrap();

        interrupted.decide_system(Direction::Down, &SystemFrame::Interrupt);
        let after_interrupt = converted(interrupted.decide_data(&audio(&silence, input)))
            .unwrap()
            .samples;

        let mut fresh = ResamplerStage::new(output).unwrap();
        let fresh_silence = converted(fresh.decide_data(&audio(&silence, input)))
            .unwrap()
            .samples;
        assert_eq!(after_interrupt, fresh_silence);
    }

    fn resample_sine(input: AudioFormat, output: AudioFormat, frequency: f32) -> Vec<f32> {
        let mut stage = ResamplerStage::new(output).unwrap();
        let mut converted_samples = Vec::new();
        for chunk_index in 0..10 {
            let samples: Vec<f32> = (0..960)
                .map(|frame| {
                    let frame = chunk_index * 960 + frame;
                    (std::f32::consts::TAU * frequency * frame as f32 / input.sample_rate as f32)
                        .sin()
                })
                .collect();
            converted_samples.extend(
                converted(stage.decide_data(&audio(&samples, input)))
                    .unwrap()
                    .samples
                    .iter()
                    .copied(),
            );
        }
        converted_samples
    }

    fn rms(samples: &[f32]) -> f32 {
        (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32).sqrt()
    }
}
