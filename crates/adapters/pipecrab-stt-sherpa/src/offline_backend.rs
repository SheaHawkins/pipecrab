use sherpa_onnx::OfflineRecognizer;

use crate::{MoonshineV2Config, SherpaSttBuildError};

/// The utterance-level recognition operation used by the offline Sherpa actor.
///
/// Implementations receive one complete 16 kHz mono utterance. A mutable
/// receiver makes exclusive actor ownership explicit.
pub trait OfflineBackend: Send + 'static {
    /// Decode one complete utterance and return its text, if available.
    fn transcribe(&mut self, samples: &[f32]) -> Option<String>;
}

pub(crate) struct MoonshineV2Backend {
    recognizer: OfflineRecognizer,
}

impl MoonshineV2Backend {
    pub(crate) fn create(config: MoonshineV2Config) -> Result<Self, SherpaSttBuildError> {
        let config = config.into_sherpa()?;
        let recognizer = OfflineRecognizer::create(&config).ok_or_else(|| {
            SherpaSttBuildError::CreateOfflineRecognizer(
                "native constructor returned no recognizer; verify the model and provider".into(),
            )
        })?;
        Ok(Self { recognizer })
    }
}

impl OfflineBackend for MoonshineV2Backend {
    fn transcribe(&mut self, samples: &[f32]) -> Option<String> {
        let stream = self.recognizer.create_stream();
        stream.accept_waveform(crate::config::SAMPLE_RATE as i32, samples);
        self.recognizer.decode(&stream);
        stream.get_result().map(|result| result.text)
    }
}
