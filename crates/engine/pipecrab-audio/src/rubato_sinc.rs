//! The bundled windowed-sinc [`Resampler`](crate::Resampler) implementation.

use std::collections::VecDeque;

use pipecrab_core::{AudioChunk, AudioFormat};
use rubato::{
    Resampler as _, SincFixedIn, SincInterpolationParameters, SincInterpolationType,
    WindowFunction, calculate_cutoff,
};

use crate::resampler::{Resampler, ResamplerError, validate_chunk, validate_format};

const BLOCK_FRAMES: usize = 64;
const SINC_LENGTH: usize = 128;
const OVERSAMPLING_FACTOR: usize = 128;

/// Streaming windowed-sinc sample-rate and channel-count conversion.
///
/// Equal channel counts are resampled independently. When the counts differ,
/// input channels are averaged to mono before resampling and the result is
/// replicated to every output channel. [`AudioFormat`] carries a channel count
/// but no speaker layout, so a more specific multichannel matrix cannot be
/// inferred safely.
///
/// Sample-rate conversion advances in 64-frame blocks. A smaller input chunk
/// can therefore return `None`; its samples remain buffered until later audio
/// completes the block. [`reset`](Resampler::reset) discards that incomplete
/// block together with the sinc filter history.
pub struct RubatoSincResampler {
    output_format: AudioFormat,
    current_input: Option<AudioFormat>,
    rate_state: Option<RateState>,
}

impl RubatoSincResampler {
    /// Create a converter with one fixed output format.
    pub fn new(output_format: AudioFormat) -> Result<Self, ResamplerError> {
        validate_format(output_format)?;
        Ok(Self {
            output_format,
            current_input: None,
            rate_state: None,
        })
    }

    fn convert(&mut self, chunk: &AudioChunk) -> Result<AudioChunk, ResamplerError> {
        let input_format = chunk.format;
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
        let state_matches = self.rate_state.as_ref().is_some_and(|state| {
            state.input_rate == input_format.sample_rate
                && state.working_channels == working_channels
        });
        if !state_matches {
            self.rate_state = Some(RateState::new(
                input_format.sample_rate,
                self.output_format.sample_rate,
                working_channels,
            )?);
        }

        let state = self
            .rate_state
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
                .map_err(from_rubato)?;

            if working_channels == output_channels {
                for frame in 0..output_frames {
                    for channel in 0..output_channels {
                        converted.push(state.output[channel][frame]);
                    }
                }
            } else {
                for frame in 0..output_frames {
                    converted.extend(std::iter::repeat_n(state.output[0][frame], output_channels));
                }
            }
        }

        Ok(AudioChunk::new(converted.into(), self.output_format))
    }
}

impl Resampler for RubatoSincResampler {
    fn output_format(&self) -> AudioFormat {
        self.output_format
    }

    fn resample(&mut self, input: &AudioChunk) -> Result<Option<AudioChunk>, ResamplerError> {
        validate_chunk(input)?;
        if self.current_input != Some(input.format) {
            self.reset();
            self.current_input = Some(input.format);
        }
        if input.format == self.output_format {
            return Ok(Some(input.clone()));
        }
        let chunk = self.convert(input)?;
        Ok((!chunk.samples.is_empty()).then_some(chunk))
    }

    fn reset(&mut self) {
        self.current_input = None;
        if let Some(state) = &mut self.rate_state {
            state.reset();
        }
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
        .map_err(from_rubato)?;
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

fn from_rubato(error: impl std::fmt::Display) -> ResamplerError {
    ResamplerError::Resampling(error.to_string())
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
        output.extend(std::iter::repeat_n(mono, output_channels));
    }
    output
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn audio(samples: &[f32], format: AudioFormat) -> AudioChunk {
        AudioChunk::new(Arc::from(samples), format)
    }

    fn converted(
        resampler: &mut RubatoSincResampler,
        samples: &[f32],
        format: AudioFormat,
    ) -> Result<AudioChunk, ResamplerError> {
        resampler
            .resample(&audio(samples, format))
            .map(|chunk| chunk.expect("expected converted audio"))
    }

    #[test]
    fn rejects_invalid_output_format() {
        assert!(matches!(
            RubatoSincResampler::new(AudioFormat::new(0, 1)),
            Err(ResamplerError::InvalidFormat { .. })
        ));
        assert!(matches!(
            RubatoSincResampler::new(AudioFormat::new(48_000, 0)),
            Err(ResamplerError::InvalidFormat { .. })
        ));
    }

    #[test]
    fn equal_format_reuses_the_sample_buffer() {
        let format = AudioFormat::new(48_000, 2);
        let samples: Arc<[f32]> = Arc::from([0.25, -0.25, 0.5, -0.5]);
        let retained = samples.clone();
        let mut resampler = RubatoSincResampler::new(format).unwrap();

        let output = resampler
            .resample(&AudioChunk::new(samples, format))
            .unwrap()
            .unwrap();

        assert!(Arc::ptr_eq(&retained, &output.samples));
    }

    #[test]
    fn remixes_stereo_to_mono_without_rate_conversion() {
        let input = AudioFormat::new(48_000, 2);
        let output = AudioFormat::new(48_000, 1);
        let mut resampler = RubatoSincResampler::new(output).unwrap();

        let chunk = converted(&mut resampler, &[1.0, -1.0, 0.5, 0.25], input).unwrap();

        assert_eq!(chunk.format, output);
        assert_eq!(&*chunk.samples, &[0.0, 0.375]);
    }

    #[test]
    fn remixes_mono_to_stereo_without_rate_conversion() {
        let input = AudioFormat::new(48_000, 1);
        let output = AudioFormat::new(48_000, 2);
        let mut resampler = RubatoSincResampler::new(output).unwrap();

        let chunk = converted(&mut resampler, &[0.25, -0.5], input).unwrap();

        assert_eq!(&*chunk.samples, &[0.25, 0.25, -0.5, -0.5]);
    }

    #[test]
    fn rejects_misaligned_interleaved_audio() {
        let format = AudioFormat::new(48_000, 2);
        let mut resampler = RubatoSincResampler::new(format).unwrap();

        let error = resampler
            .resample(&audio(&[0.0, 1.0, 2.0], format))
            .unwrap_err();

        assert_eq!(
            error,
            ResamplerError::MisalignedSamples {
                samples: 3,
                channels: 2,
            }
        );
    }

    #[test]
    fn arbitrary_channel_change_uses_mono_as_the_neutral_layout() {
        let input = AudioFormat::new(48_000, 3);
        let output = AudioFormat::new(48_000, 2);
        let mut resampler = RubatoSincResampler::new(output).unwrap();

        let chunk = converted(&mut resampler, &[0.0, 0.5, 1.0], input).unwrap();

        assert_eq!(&*chunk.samples, &[0.5, 0.5]);
    }

    #[test]
    fn sub_block_audio_is_buffered_until_conversion_can_advance() {
        let input = AudioFormat::new(48_000, 1);
        let output = AudioFormat::new(16_000, 1);
        let mut resampler = RubatoSincResampler::new(output).unwrap();
        let short = vec![0.0; BLOCK_FRAMES / 2];

        assert!(resampler.resample(&audio(&short, input)).unwrap().is_none());
        let _ = resampler.resample(&audio(&short, input)).unwrap();

        assert_eq!(resampler.rate_state.as_ref().unwrap().pending[0].len(), 0);
    }

    #[test]
    fn input_format_change_starts_a_fresh_stream() {
        let output = AudioFormat::new(16_000, 1);
        let first_input = AudioFormat::new(48_000, 1);
        let second_input = AudioFormat::new(24_000, 1);
        let mut changed = RubatoSincResampler::new(output).unwrap();
        converted(&mut changed, &vec![1.0; 960], first_input).unwrap();
        let after_change = converted(&mut changed, &vec![0.0; 480], second_input)
            .unwrap()
            .samples;

        let mut fresh = RubatoSincResampler::new(output).unwrap();
        let fresh_second = converted(&mut fresh, &vec![0.0; 480], second_input)
            .unwrap()
            .samples;

        assert_eq!(after_change, fresh_second);
    }

    #[test]
    fn downsampling_tracks_stream_duration_across_chunks() {
        let input = AudioFormat::new(48_000, 1);
        let output = AudioFormat::new(16_000, 1);
        let mut resampler = RubatoSincResampler::new(output).unwrap();
        let samples = vec![0.0; 960];
        let chunks = 50;
        let mut output_frames = 0;

        for _ in 0..chunks {
            let chunk = converted(&mut resampler, &samples, input).unwrap();
            output_frames += chunk.samples.len();
        }

        let expected =
            chunks * samples.len() * output.sample_rate as usize / input.sample_rate as usize;
        let delay = resampler
            .rate_state
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

        let mut whole_resampler = RubatoSincResampler::new(output).unwrap();
        let whole = converted(&mut whole_resampler, &samples, input)
            .unwrap()
            .samples;

        let mut split_resampler = RubatoSincResampler::new(output).unwrap();
        let mut split = Vec::new();
        for part in samples.chunks(960) {
            split.extend(
                converted(&mut split_resampler, part, input)
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
    fn reset_clears_filter_history() {
        let input = AudioFormat::new(48_000, 1);
        let output = AudioFormat::new(16_000, 1);
        let mut reset = RubatoSincResampler::new(output).unwrap();
        converted(&mut reset, &vec![1.0; 960], input).unwrap();
        reset.reset();
        let after_reset = converted(&mut reset, &vec![0.0; 960], input)
            .unwrap()
            .samples;

        let mut fresh = RubatoSincResampler::new(output).unwrap();
        let fresh_silence = converted(&mut fresh, &vec![0.0; 960], input)
            .unwrap()
            .samples;

        assert_eq!(after_reset, fresh_silence);
    }

    fn resample_sine(input: AudioFormat, output: AudioFormat, frequency: f32) -> Vec<f32> {
        let mut resampler = RubatoSincResampler::new(output).unwrap();
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
                converted(&mut resampler, &samples, input)
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
