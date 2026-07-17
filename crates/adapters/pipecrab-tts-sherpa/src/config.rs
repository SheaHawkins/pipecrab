use std::fmt;
use std::path::{Path, PathBuf};

use sherpa_onnx::{GenerationConfig, OfflineTtsConfig};

/// Configuration for a Kokoro [`SherpaTts`](crate::SherpaTts).
///
/// Kokoro is an offline multi-speaker model distributed as one ONNX model, a
/// packed voice-embedding file, a token table, and an espeak-ng data
/// directory. Synthesis runs on the CPU provider; the engine reports its own
/// output sample rate (24 kHz for the published Kokoro models).
#[derive(Clone, Debug)]
pub struct KokoroConfig {
    /// Path to the Kokoro ONNX model.
    pub model: PathBuf,
    /// Path to the packed voice-embedding file (`voices.bin`).
    pub voices: PathBuf,
    /// Path to the model token table.
    pub tokens: PathBuf,
    /// Path to the espeak-ng data directory shipped with the model.
    pub data_dir: PathBuf,
    /// Optional jieba dictionary directory (multi-language Kokoro models).
    pub dict_dir: Option<PathBuf>,
    /// Optional comma-separated lexicon files (multi-language Kokoro models).
    pub lexicon: Option<String>,
    /// Optional language hint (e.g. `"en-us"`) for multi-language models.
    pub lang: Option<String>,
    /// Built-in speaker to voice, in `0..num_speakers` for the model.
    pub speaker: i32,
    /// Speaking-rate multiplier; `1.0` is the model's natural pace.
    pub speed: f32,
    /// ONNX Runtime compute threads used by the engine.
    pub num_threads: i32,
    /// Enable Sherpa model diagnostics.
    pub debug: bool,
}

impl KokoroConfig {
    /// Create a CPU Kokoro configuration with two compute threads, the first
    /// built-in speaker, and the natural speaking rate.
    pub fn new(
        model: impl Into<PathBuf>,
        voices: impl Into<PathBuf>,
        tokens: impl Into<PathBuf>,
        data_dir: impl Into<PathBuf>,
    ) -> Self {
        Self {
            model: model.into(),
            voices: voices.into(),
            tokens: tokens.into(),
            data_dir: data_dir.into(),
            dict_dir: None,
            lexicon: None,
            lang: None,
            speaker: 0,
            speed: 1.0,
            num_threads: 2,
            debug: false,
        }
    }

    pub(crate) fn into_sherpa(
        self,
    ) -> Result<(OfflineTtsConfig, GenerationConfig), SherpaTtsBuildError> {
        self.validate()?;

        let mut config = OfflineTtsConfig::default();
        config.model.kokoro.model = Some(path_string("model", &self.model)?);
        config.model.kokoro.voices = Some(path_string("voices", &self.voices)?);
        config.model.kokoro.tokens = Some(path_string("tokens", &self.tokens)?);
        config.model.kokoro.data_dir = Some(path_string("data_dir", &self.data_dir)?);
        config.model.kokoro.dict_dir = self
            .dict_dir
            .as_deref()
            .map(|dir| path_string("dict_dir", dir))
            .transpose()?;
        config.model.kokoro.lexicon = self.lexicon.clone();
        config.model.kokoro.lang = self.lang.clone();
        config.model.num_threads = self.num_threads;
        config.model.provider = Some("cpu".into());
        config.model.debug = self.debug;

        let generation = GenerationConfig {
            sid: self.speaker,
            speed: self.speed,
            ..GenerationConfig::default()
        };
        Ok((config, generation))
    }

    pub(crate) fn validate(&self) -> Result<(), SherpaTtsBuildError> {
        validate_file("model", &self.model)?;
        validate_file("voices", &self.voices)?;
        validate_file("tokens", &self.tokens)?;
        validate_dir("data_dir", &self.data_dir)?;
        if let Some(dict_dir) = &self.dict_dir {
            validate_dir("dict_dir", dict_dir)?;
        }
        if let Some(lexicon) = &self.lexicon {
            for entry in lexicon.split(',') {
                validate_file("lexicon", Path::new(entry.trim()))?;
            }
        }
        if self.speaker < 0 {
            return Err(SherpaTtsBuildError::InvalidConfig(format!(
                "speaker must be non-negative, got {}",
                self.speaker
            )));
        }
        if !self.speed.is_finite() || self.speed <= 0.0 {
            return Err(SherpaTtsBuildError::InvalidConfig(format!(
                "speed must be finite and positive, got {}",
                self.speed
            )));
        }
        validate_threads(self.num_threads)
    }
}

fn validate_file(name: &str, path: &Path) -> Result<(), SherpaTtsBuildError> {
    if path.is_file() {
        Ok(())
    } else {
        Err(SherpaTtsBuildError::InvalidConfig(format!(
            "{name} does not exist or is not a file: {}",
            path.display()
        )))
    }
}

fn validate_dir(name: &str, path: &Path) -> Result<(), SherpaTtsBuildError> {
    if path.is_dir() {
        Ok(())
    } else {
        Err(SherpaTtsBuildError::InvalidConfig(format!(
            "{name} does not exist or is not a directory: {}",
            path.display()
        )))
    }
}

fn validate_threads(num_threads: i32) -> Result<(), SherpaTtsBuildError> {
    if num_threads <= 0 {
        return Err(SherpaTtsBuildError::InvalidConfig(format!(
            "num_threads must be positive, got {num_threads}"
        )));
    }
    Ok(())
}

fn path_string(name: &str, path: &Path) -> Result<String, SherpaTtsBuildError> {
    path.to_str().map(str::to_owned).ok_or_else(|| {
        SherpaTtsBuildError::InvalidConfig(format!(
            "{name} path must contain valid UTF-8 for Sherpa"
        ))
    })
}

/// Why a Sherpa TTS worker could not be constructed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SherpaTtsBuildError {
    /// A configuration field or model path is unusable.
    InvalidConfig(String),
    /// Sherpa rejected the offline TTS configuration.
    CreateEngine(String),
    /// The actor thread could not start or exited during setup.
    Worker(String),
}

impl fmt::Display for SherpaTtsBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => {
                write!(formatter, "invalid Sherpa TTS config: {message}")
            }
            Self::CreateEngine(message) => {
                write!(formatter, "create Sherpa offline TTS engine: {message}")
            }
            Self::Worker(message) => write!(formatter, "Sherpa TTS worker: {message}"),
        }
    }
}

impl std::error::Error for SherpaTtsBuildError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_config() -> KokoroConfig {
        let file = std::env::current_exe().expect("test executable has a path");
        let dir = file.parent().expect("test executable has a directory");
        KokoroConfig::new(&file, &file, &file, dir)
    }

    fn invalid_config_message(config: KokoroConfig) -> String {
        match config.into_sherpa().unwrap_err() {
            SherpaTtsBuildError::InvalidConfig(message) => message,
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn translates_fixed_kokoro_configuration() {
        let (config, generation) = valid_config().into_sherpa().unwrap();

        assert!(config.model.kokoro.model.is_some());
        assert!(config.model.kokoro.voices.is_some());
        assert!(config.model.kokoro.tokens.is_some());
        assert!(config.model.kokoro.data_dir.is_some());
        assert!(config.model.kokoro.dict_dir.is_none());
        assert!(config.model.kokoro.lexicon.is_none());
        assert_eq!(config.model.num_threads, 2);
        assert_eq!(config.model.provider.as_deref(), Some("cpu"));
        assert_eq!(generation.sid, 0);
        assert_eq!(generation.speed, 1.0);
    }

    #[test]
    fn retains_speaker_and_speed() {
        let mut config = valid_config();
        config.speaker = 7;
        config.speed = 1.25;

        let (_, generation) = config.into_sherpa().unwrap();

        assert_eq!(generation.sid, 7);
        assert_eq!(generation.speed, 1.25);
    }

    #[test]
    fn rejects_each_missing_model_path() {
        for (name, invalidate) in [
            (
                "model",
                (|config| config.model = "missing.onnx".into()) as fn(&mut KokoroConfig),
            ),
            ("voices", |config| config.voices = "missing.bin".into()),
            ("tokens", |config| config.tokens = "missing.txt".into()),
            ("data_dir", |config| config.data_dir = "missing-dir".into()),
            ("dict_dir", |config| {
                config.dict_dir = Some("missing-dict".into());
            }),
            ("lexicon", |config| {
                config.lexicon = Some("missing-lexicon.txt".into());
            }),
        ] {
            let mut config = valid_config();
            invalidate(&mut config);
            assert!(
                invalid_config_message(config).starts_with(name),
                "expected {name} to be rejected"
            );
        }
    }

    #[test]
    fn rejects_a_negative_speaker() {
        let mut config = valid_config();
        config.speaker = -1;
        assert_eq!(
            invalid_config_message(config),
            "speaker must be non-negative, got -1"
        );
    }

    #[test]
    fn rejects_non_positive_and_non_finite_speeds() {
        for speed in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let mut config = valid_config();
            config.speed = speed;
            assert!(
                invalid_config_message(config).starts_with("speed must be finite and positive")
            );
        }
    }

    #[test]
    fn rejects_non_positive_thread_counts() {
        for threads in [0, -1] {
            let mut config = valid_config();
            config.num_threads = threads;
            assert_eq!(
                invalid_config_message(config),
                format!("num_threads must be positive, got {threads}")
            );
        }
    }
}
