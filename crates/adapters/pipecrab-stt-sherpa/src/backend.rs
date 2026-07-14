use sherpa_onnx::{OnlineRecognizer, OnlineStream};

use crate::{OnlineSherpaSttConfig, SherpaSttBuildError};

/// The serialized recognizer operations used by the Sherpa STT actor.
///
/// A mutable receiver makes exclusive actor ownership explicit. The associated
/// stream is created, used, and dropped on that same actor thread.
pub trait OnlineBackend: Send + 'static {
    /// Per-utterance streaming decoder state.
    type Stream: 'static;

    /// Create a clean stream for a new utterance.
    fn create_stream(&mut self) -> Self::Stream;

    /// Append one 16 kHz mono waveform chunk to `stream`.
    fn accept_waveform(&mut self, stream: &mut Self::Stream, samples: &[f32]);

    /// Mark `stream` as having no more input audio.
    fn input_finished(&mut self, stream: &mut Self::Stream);

    /// Whether `stream` has enough input for one decode step.
    fn is_ready(&mut self, stream: &mut Self::Stream) -> bool;

    /// Decode one step from `stream`.
    fn decode(&mut self, stream: &mut Self::Stream);

    /// Return the current text hypothesis, if one is available.
    fn get_result(&mut self, stream: &mut Self::Stream) -> Option<String>;
}

pub(crate) struct OnlineSherpaBackend {
    recognizer: OnlineRecognizer,
}

impl OnlineSherpaBackend {
    pub(crate) fn create(config: OnlineSherpaSttConfig) -> Result<Self, SherpaSttBuildError> {
        let config = config.into_sherpa()?;
        let recognizer = OnlineRecognizer::create(&config).ok_or_else(|| {
            SherpaSttBuildError::CreateRecognizer(
                "native constructor returned no recognizer; verify the model and provider".into(),
            )
        })?;
        Ok(Self { recognizer })
    }
}

impl OnlineBackend for OnlineSherpaBackend {
    type Stream = OnlineStream;

    fn create_stream(&mut self) -> Self::Stream {
        self.recognizer.create_stream()
    }

    fn accept_waveform(&mut self, stream: &mut Self::Stream, samples: &[f32]) {
        stream.accept_waveform(crate::config::SAMPLE_RATE as i32, samples);
    }

    fn input_finished(&mut self, stream: &mut Self::Stream) {
        stream.input_finished();
    }

    fn is_ready(&mut self, stream: &mut Self::Stream) -> bool {
        self.recognizer.is_ready(stream)
    }

    fn decode(&mut self, stream: &mut Self::Stream) {
        self.recognizer.decode(stream);
    }

    fn get_result(&mut self, stream: &mut Self::Stream) -> Option<String> {
        self.recognizer.get_result(stream).map(|result| result.text)
    }
}
