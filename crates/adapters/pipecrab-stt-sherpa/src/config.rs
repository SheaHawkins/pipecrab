use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use sherpa_onnx::{OfflineRecognizerConfig, OnlineRecognizerConfig};

pub(crate) const SAMPLE_RATE: u32 = 16_000;
pub(crate) const DEFAULT_INITIAL_CONTEXT: Duration = Duration::from_secs(1);
pub(crate) const DEFAULT_FINAL_CONTEXT: Duration = Duration::from_millis(300);
pub(crate) const DEFAULT_MOONSHINE_CHUNK_DURATION: Duration = Duration::from_secs(8);
pub(crate) const DEFAULT_MOONSHINE_CHUNK_OVERLAP: Duration = Duration::from_millis(500);
pub(crate) const MAX_MOONSHINE_CHUNK_DURATION: Duration = Duration::from_secs(9);

/// Configuration for [`OnlineSherpaStt`](crate::OnlineSherpaStt).
///
/// The adapter fixes recognition to 16 kHz mono audio, the CPU provider,
/// greedy search, and external PipeCrab utterance boundaries. The model paths
/// identify one Sherpa streaming transducer.
#[derive(Clone, Debug)]
pub struct OnlineSherpaSttConfig {
    /// Path to the streaming transducer encoder model.
    pub encoder: PathBuf,
    /// Path to the streaming transducer decoder model.
    pub decoder: PathBuf,
    /// Path to the streaming transducer joiner model.
    pub joiner: PathBuf,
    /// Path to the model token table.
    pub tokens: PathBuf,
    /// ONNX Runtime compute threads used by the recognizer.
    pub num_threads: i32,
    /// Zero-valued audio decoded before each utterance to prime model context.
    /// Set to [`Duration::ZERO`] to disable initial padding.
    pub initial_context: Duration,
    /// Zero-valued audio appended before finishing each utterance.
    /// Set to [`Duration::ZERO`] to disable final padding.
    pub final_context: Duration,
    /// Enable Sherpa model diagnostics.
    pub debug: bool,
}

impl OnlineSherpaSttConfig {
    /// Create a CPU streaming-transducer configuration with two compute
    /// threads, greedy search, and endpoint detection disabled.
    pub fn new(
        encoder: impl Into<PathBuf>,
        decoder: impl Into<PathBuf>,
        joiner: impl Into<PathBuf>,
        tokens: impl Into<PathBuf>,
    ) -> Self {
        Self {
            encoder: encoder.into(),
            decoder: decoder.into(),
            joiner: joiner.into(),
            tokens: tokens.into(),
            num_threads: 2,
            initial_context: DEFAULT_INITIAL_CONTEXT,
            final_context: DEFAULT_FINAL_CONTEXT,
            debug: false,
        }
    }

    pub(crate) fn into_sherpa(self) -> Result<OnlineRecognizerConfig, SherpaSttBuildError> {
        self.validate()?;

        let mut config = OnlineRecognizerConfig::default();
        config.model_config.transducer.encoder = Some(path_string("encoder", &self.encoder)?);
        config.model_config.transducer.decoder = Some(path_string("decoder", &self.decoder)?);
        config.model_config.transducer.joiner = Some(path_string("joiner", &self.joiner)?);
        config.model_config.tokens = Some(path_string("tokens", &self.tokens)?);
        config.model_config.num_threads = self.num_threads;
        config.model_config.provider = Some("cpu".into());
        config.model_config.debug = self.debug;
        config.feat_config.sample_rate = SAMPLE_RATE as i32;
        config.decoding_method = Some("greedy_search".into());
        config.enable_endpoint = false;
        Ok(config)
    }

    pub(crate) fn validate(&self) -> Result<(), SherpaSttBuildError> {
        validate_file("encoder", &self.encoder)?;
        validate_file("decoder", &self.decoder)?;
        validate_file("joiner", &self.joiner)?;
        validate_file("tokens", &self.tokens)?;
        validate_threads(self.num_threads)
    }
}

/// The online configuration used by the default [`SherpaStt`](crate::SherpaStt)
/// interface.
pub type SherpaSttConfig = OnlineSherpaSttConfig;

/// Configuration for a Moonshine v2 [`OfflineSherpaStt`](crate::OfflineSherpaStt).
///
/// Moonshine v2 is an offline model with an encoder, merged decoder, and token
/// table. Audio is fixed to 16 kHz mono and decoded with CPU greedy search.
#[derive(Clone, Debug)]
pub struct MoonshineV2Config {
    /// Path to the Moonshine v2 encoder model.
    pub encoder: PathBuf,
    /// Path to the Moonshine v2 merged decoder model.
    pub merged_decoder: PathBuf,
    /// Path to the model token table.
    pub tokens: PathBuf,
    /// ONNX Runtime compute threads used by the recognizer.
    pub num_threads: i32,
    /// Maximum audio duration passed to one native Moonshine decode.
    ///
    /// Longer PipeCrab utterances are divided into overlapping windows and
    /// still produce one final transcript.
    pub chunk_duration: Duration,
    /// Audio context repeated between consecutive decode windows.
    pub chunk_overlap: Duration,
    /// Enable Sherpa model diagnostics.
    pub debug: bool,
}

impl MoonshineV2Config {
    /// Create a CPU Moonshine v2 configuration with two compute threads and
    /// greedy search.
    pub fn new(
        encoder: impl Into<PathBuf>,
        merged_decoder: impl Into<PathBuf>,
        tokens: impl Into<PathBuf>,
    ) -> Self {
        Self {
            encoder: encoder.into(),
            merged_decoder: merged_decoder.into(),
            tokens: tokens.into(),
            num_threads: 2,
            chunk_duration: DEFAULT_MOONSHINE_CHUNK_DURATION,
            chunk_overlap: DEFAULT_MOONSHINE_CHUNK_OVERLAP,
            debug: false,
        }
    }

    pub(crate) fn into_sherpa(self) -> Result<OfflineRecognizerConfig, SherpaSttBuildError> {
        self.validate()?;

        let mut config = OfflineRecognizerConfig::default();
        config.model_config.moonshine.encoder = Some(path_string("encoder", &self.encoder)?);
        config.model_config.moonshine.merged_decoder =
            Some(path_string("merged_decoder", &self.merged_decoder)?);
        config.model_config.tokens = Some(path_string("tokens", &self.tokens)?);
        config.model_config.num_threads = self.num_threads;
        config.model_config.provider = Some("cpu".into());
        config.model_config.debug = self.debug;
        config.feat_config.sample_rate = SAMPLE_RATE as i32;
        config.decoding_method = Some("greedy_search".into());
        Ok(config)
    }

    pub(crate) fn validate(&self) -> Result<(), SherpaSttBuildError> {
        validate_file("encoder", &self.encoder)?;
        validate_file("merged_decoder", &self.merged_decoder)?;
        validate_file("tokens", &self.tokens)?;
        validate_threads(self.num_threads)?;
        self.chunk_samples().map(|_| ())
    }

    pub(crate) fn chunk_samples(&self) -> Result<(usize, usize), SherpaSttBuildError> {
        if self.chunk_duration > MAX_MOONSHINE_CHUNK_DURATION {
            return Err(SherpaSttBuildError::InvalidConfig(format!(
                "chunk_duration must be at most {} seconds for Moonshine v2, got {:?}",
                MAX_MOONSHINE_CHUNK_DURATION.as_secs(),
                self.chunk_duration
            )));
        }
        if self.chunk_overlap >= self.chunk_duration {
            return Err(SherpaSttBuildError::InvalidConfig(format!(
                "chunk_overlap ({:?}) must be shorter than chunk_duration ({:?})",
                self.chunk_overlap, self.chunk_duration
            )));
        }

        let chunk = duration_sample_count("chunk_duration", self.chunk_duration)?;
        if chunk == 0 {
            return Err(SherpaSttBuildError::InvalidConfig(
                "chunk_duration must contain at least one 16 kHz sample".into(),
            ));
        }
        let overlap = duration_sample_count("chunk_overlap", self.chunk_overlap)?;
        Ok((chunk, overlap))
    }
}

pub(crate) fn duration_sample_count(
    name: &str,
    duration: Duration,
) -> Result<usize, SherpaSttBuildError> {
    const NANOS_PER_SECOND: u128 = 1_000_000_000;
    let count = duration
        .as_nanos()
        .checked_mul(u128::from(SAMPLE_RATE))
        .ok_or_else(|| {
            SherpaSttBuildError::InvalidConfig(format!(
                "{name} is too large to convert to 16 kHz samples"
            ))
        })?
        / NANOS_PER_SECOND;
    usize::try_from(count).map_err(|_| {
        SherpaSttBuildError::InvalidConfig(format!(
            "{name} is too large to convert to 16 kHz samples"
        ))
    })
}

fn validate_file(name: &str, path: &Path) -> Result<(), SherpaSttBuildError> {
    if path.is_file() {
        Ok(())
    } else {
        Err(SherpaSttBuildError::InvalidConfig(format!(
            "{name} does not exist or is not a file: {}",
            path.display()
        )))
    }
}

fn validate_threads(num_threads: i32) -> Result<(), SherpaSttBuildError> {
    if num_threads <= 0 {
        return Err(SherpaSttBuildError::InvalidConfig(format!(
            "num_threads must be positive, got {num_threads}"
        )));
    }
    Ok(())
}

fn path_string(name: &str, path: &Path) -> Result<String, SherpaSttBuildError> {
    path.to_str().map(str::to_owned).ok_or_else(|| {
        SherpaSttBuildError::InvalidConfig(format!(
            "{name} path must contain valid UTF-8 for Sherpa"
        ))
    })
}

/// Why a Sherpa STT worker could not be constructed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SherpaSttBuildError {
    /// A configuration field or model path is unusable.
    InvalidConfig(String),
    /// Sherpa rejected the online recognizer configuration.
    CreateRecognizer(String),
    /// Sherpa rejected the offline recognizer configuration.
    CreateOfflineRecognizer(String),
    /// The actor thread could not start or exited during setup.
    Worker(String),
}

impl fmt::Display for SherpaSttBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => {
                write!(formatter, "invalid Sherpa STT config: {message}")
            }
            Self::CreateRecognizer(message) => {
                write!(formatter, "create Sherpa online recognizer: {message}")
            }
            Self::CreateOfflineRecognizer(message) => {
                write!(formatter, "create Sherpa offline recognizer: {message}")
            }
            Self::Worker(message) => write!(formatter, "Sherpa STT worker: {message}"),
        }
    }
}

impl std::error::Error for SherpaSttBuildError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_config() -> OnlineSherpaSttConfig {
        let file = std::env::current_exe().expect("test executable has a path");
        OnlineSherpaSttConfig::new(&file, &file, &file, &file)
    }

    fn invalid_config_message(config: OnlineSherpaSttConfig) -> String {
        match config.into_sherpa().unwrap_err() {
            SherpaSttBuildError::InvalidConfig(message) => message,
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    fn assert_missing_file(name: &str, invalidate: impl FnOnce(&mut OnlineSherpaSttConfig)) {
        let mut config = valid_config();
        invalidate(&mut config);
        assert!(invalid_config_message(config).starts_with(name));
    }

    #[test]
    fn translates_fixed_streaming_configuration() {
        let config = valid_config().into_sherpa().unwrap();

        assert_eq!(config.feat_config.sample_rate, SAMPLE_RATE as i32);
        assert_eq!(config.model_config.num_threads, 2);
        assert_eq!(config.model_config.provider.as_deref(), Some("cpu"));
        assert_eq!(config.decoding_method.as_deref(), Some("greedy_search"));
        assert!(!config.enable_endpoint);
        assert!(config.hotwords_file.is_none());
    }

    #[test]
    fn retains_supported_thread_profiles() {
        for threads in [1, 2, 3] {
            let mut config = valid_config();
            config.num_threads = threads;
            assert_eq!(
                config.into_sherpa().unwrap().model_config.num_threads,
                threads
            );
        }
    }

    #[test]
    fn defaults_to_zipformer_boundary_context() {
        let config = valid_config();

        assert_eq!(config.initial_context, Duration::from_secs(1));
        assert_eq!(config.final_context, Duration::from_millis(300));
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

    #[test]
    fn rejects_each_missing_model_file() {
        assert_missing_file("encoder", |config| {
            config.encoder = "missing-encoder.onnx".into();
        });
        assert_missing_file("decoder", |config| {
            config.decoder = "missing-decoder.onnx".into();
        });
        assert_missing_file("joiner", |config| {
            config.joiner = "missing-joiner.onnx".into();
        });
        assert_missing_file("tokens", |config| {
            config.tokens = "missing-tokens.txt".into();
        });
    }

    #[test]
    fn translates_moonshine_v2_configuration() {
        let file = std::env::current_exe().expect("test executable has a path");
        let config = MoonshineV2Config::new(&file, &file, &file)
            .into_sherpa()
            .unwrap();

        assert_eq!(config.feat_config.sample_rate, SAMPLE_RATE as i32);
        assert_eq!(config.model_config.num_threads, 2);
        assert_eq!(config.model_config.provider.as_deref(), Some("cpu"));
        assert_eq!(config.decoding_method.as_deref(), Some("greedy_search"));
        assert!(config.model_config.moonshine.preprocessor.is_none());
        assert!(config.model_config.moonshine.uncached_decoder.is_none());
        assert!(config.model_config.moonshine.cached_decoder.is_none());
        assert!(config.model_config.moonshine.encoder.is_some());
        assert!(config.model_config.moonshine.merged_decoder.is_some());
    }

    #[test]
    fn defaults_to_safe_overlapping_moonshine_windows() {
        let file = std::env::current_exe().expect("test executable has a path");
        let config = MoonshineV2Config::new(&file, &file, &file);

        assert_eq!(config.chunk_duration, Duration::from_secs(8));
        assert_eq!(config.chunk_overlap, Duration::from_millis(500));
        assert_eq!(config.chunk_samples().unwrap(), (128_000, 8_000));
    }

    #[test]
    fn rejects_unsafe_moonshine_window_configuration() {
        let file = std::env::current_exe().expect("test executable has a path");
        let mut config = MoonshineV2Config::new(&file, &file, &file);
        config.chunk_duration = Duration::from_millis(9_001);
        assert!(
            config
                .chunk_samples()
                .unwrap_err()
                .to_string()
                .contains("at most 9")
        );

        config.chunk_duration = Duration::from_secs(8);
        config.chunk_overlap = Duration::from_secs(8);
        assert!(
            config
                .chunk_samples()
                .unwrap_err()
                .to_string()
                .contains("must be shorter")
        );
    }

    #[test]
    fn rejects_incomplete_moonshine_v2_configuration() {
        let file = std::env::current_exe().expect("test executable has a path");
        let mut config = MoonshineV2Config::new(&file, &file, &file);
        config.merged_decoder = "missing-merged-decoder.ort".into();

        let SherpaSttBuildError::InvalidConfig(message) = config.into_sherpa().unwrap_err() else {
            panic!("expected invalid config");
        };
        assert!(message.starts_with("merged_decoder"));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_non_utf8_model_paths() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        assert_eq!(
            path_string(
                "encoder",
                Path::new(&OsString::from_vec(b"encoder-\xff.onnx".to_vec()))
            ),
            Err(SherpaSttBuildError::InvalidConfig(
                "encoder path must contain valid UTF-8 for Sherpa".into()
            ))
        );
    }
}
