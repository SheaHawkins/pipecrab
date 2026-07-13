use std::fmt;
use std::path::PathBuf;

use sherpa_onnx::{SileroVadModelConfig, VadModelConfig};

pub(crate) const SAMPLE_RATE: u32 = 16_000;
pub(crate) const WINDOW_SAMPLES: usize = 512;

/// Configuration for [`SherpaVad`](crate::SherpaVad).
///
/// The adapter fixes the engine to 16 kHz mono, 512-sample windows, the CPU
/// provider, and one compute thread. The fields here tune Sherpa's segmenting
/// policy and internal result-buffer duration.
#[derive(Clone, Debug)]
pub struct SherpaVadConfig {
    /// Path to a Sherpa-compatible Silero VAD ONNX model.
    pub model: PathBuf,
    /// Speech probability threshold.
    pub threshold: f32,
    /// Silence required to close an utterance, in seconds.
    pub min_silence_duration: f32,
    /// Speech required to open an utterance, in seconds.
    pub min_speech_duration: f32,
    /// Maximum duration of one speech segment, in seconds.
    pub max_speech_duration: f32,
    /// Capacity of Sherpa's internal result buffer, in seconds.
    pub buffer_size: f32,
    /// Enable Sherpa model diagnostics.
    pub debug: bool,
}

impl SherpaVadConfig {
    /// Create a CPU Silero configuration with production defaults.
    pub fn new(model: impl Into<PathBuf>) -> Self {
        Self {
            model: model.into(),
            threshold: 0.5,
            min_silence_duration: 0.25,
            min_speech_duration: 0.25,
            max_speech_duration: 5.0,
            buffer_size: 30.0,
            debug: false,
        }
    }

    pub(crate) fn into_sherpa(self) -> Result<(VadModelConfig, f32), SherpaVadBuildError> {
        self.validate()?;
        let model = self
            .model
            .to_str()
            .ok_or_else(|| {
                SherpaVadBuildError::InvalidConfig(
                    "model path must contain valid UTF-8 for Sherpa".into(),
                )
            })?
            .to_owned();
        let silero_vad = SileroVadModelConfig {
            model: Some(model),
            threshold: self.threshold,
            min_silence_duration: self.min_silence_duration,
            min_speech_duration: self.min_speech_duration,
            window_size: WINDOW_SAMPLES as i32,
            max_speech_duration: self.max_speech_duration,
        };
        Ok((
            VadModelConfig {
                silero_vad,
                ten_vad: Default::default(),
                sample_rate: SAMPLE_RATE as i32,
                num_threads: 1,
                provider: Some("cpu".into()),
                debug: self.debug,
            },
            self.buffer_size,
        ))
    }

    fn validate(&self) -> Result<(), SherpaVadBuildError> {
        if !self.model.is_file() {
            return Err(SherpaVadBuildError::InvalidConfig(format!(
                "Silero VAD model does not exist or is not a file: {}",
                self.model.display()
            )));
        }
        validate_range("threshold", self.threshold, 0.0, 1.0)?;
        validate_positive("min_silence_duration", self.min_silence_duration)?;
        validate_positive("min_speech_duration", self.min_speech_duration)?;
        validate_positive("max_speech_duration", self.max_speech_duration)?;
        validate_positive("buffer_size", self.buffer_size)?;
        if self.buffer_size < self.max_speech_duration {
            return Err(SherpaVadBuildError::InvalidConfig(format!(
                "buffer_size ({}) must be at least max_speech_duration ({})",
                self.buffer_size, self.max_speech_duration
            )));
        }
        Ok(())
    }
}

fn validate_range(
    name: &str,
    value: f32,
    minimum: f32,
    maximum: f32,
) -> Result<(), SherpaVadBuildError> {
    if value.is_finite() && (minimum..=maximum).contains(&value) {
        Ok(())
    } else {
        Err(SherpaVadBuildError::InvalidConfig(format!(
            "{name} must be finite and in [{minimum}, {maximum}], got {value}"
        )))
    }
}

fn validate_positive(name: &str, value: f32) -> Result<(), SherpaVadBuildError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(SherpaVadBuildError::InvalidConfig(format!(
            "{name} must be finite and positive, got {value}"
        )))
    }
}

/// Why a Sherpa VAD worker could not be constructed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SherpaVadBuildError {
    /// A configuration field or model path is unusable.
    InvalidConfig(String),
    /// Sherpa rejected the model configuration.
    CreateDetector(String),
    /// The actor thread could not start or exited during setup.
    Worker(String),
}

impl fmt::Display for SherpaVadBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => {
                write!(formatter, "invalid Sherpa VAD config: {message}")
            }
            Self::CreateDetector(message) => {
                write!(formatter, "create Sherpa VAD detector: {message}")
            }
            Self::Worker(message) => write!(formatter, "Sherpa VAD worker: {message}"),
        }
    }
}

impl std::error::Error for SherpaVadBuildError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_a_missing_model_before_starting_a_worker() {
        let error = SherpaVadConfig::new("definitely-not-a-silero-model.onnx")
            .into_sherpa()
            .unwrap_err();
        assert!(matches!(error, SherpaVadBuildError::InvalidConfig(_)));
    }
}
