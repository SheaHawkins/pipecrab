use sherpa_onnx::VoiceActivityDetector as SherpaDetector;

use crate::SherpaVadConfig;
use crate::config::SherpaVadBuildError;

/// The serialized engine operations used by the Sherpa VAD actor.
///
/// A mutable receiver makes exclusive actor ownership explicit even when an
/// underlying wrapper exposes shared-reference methods. Implementations are
/// moved into the worker and no reference escapes it.
pub trait Backend: Send + 'static {
    /// Whether the engine currently considers speech active.
    fn detected(&mut self) -> bool;

    /// Feed one exact 512-sample waveform window.
    fn accept_waveform(&mut self, samples: &[f32]);

    /// Whether Sherpa's completed-segment queue is empty.
    fn is_empty(&mut self) -> bool;

    /// Discard the front completed segment.
    fn pop(&mut self);

    /// Return the engine to its idle state.
    fn reset(&mut self);
}

pub(crate) struct SherpaBackend {
    detector: SherpaDetector,
}

impl SherpaBackend {
    pub(crate) fn create(config: SherpaVadConfig) -> Result<Self, SherpaVadBuildError> {
        let (config, buffer_size) = config.into_sherpa()?;
        let detector = SherpaDetector::create(&config, buffer_size).ok_or_else(|| {
            SherpaVadBuildError::CreateDetector(
                "native constructor returned no detector; verify the model and provider".into(),
            )
        })?;
        Ok(Self { detector })
    }
}

impl Backend for SherpaBackend {
    fn detected(&mut self) -> bool {
        self.detector.detected()
    }

    fn accept_waveform(&mut self, samples: &[f32]) {
        self.detector.accept_waveform(samples);
    }

    fn is_empty(&mut self) -> bool {
        self.detector.is_empty()
    }

    fn pop(&mut self) {
        self.detector.pop();
    }

    fn reset(&mut self) {
        self.detector.reset();
    }
}
