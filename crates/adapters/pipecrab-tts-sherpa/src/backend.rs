use sherpa_onnx::{GenerationConfig, OfflineTts};

use crate::{KokoroConfig, SherpaTtsBuildError};

/// The waveform-segment consumer a [`Backend::generate`] call feeds.
///
/// Boxed (rather than a generic or a plain `&mut` closure) because Sherpa's
/// native callback requires an owned `'static` value the backend can move
/// into the engine.
pub type Emit = Box<dyn FnMut(&[f32]) -> bool>;

/// The serialized synthesis operation used by the Sherpa TTS actor.
///
/// A mutable receiver makes exclusive actor ownership explicit even when an
/// underlying wrapper exposes shared-reference methods. Implementations are
/// moved into the worker and no reference escapes it.
pub trait Backend: Send + 'static {
    /// The sample rate of every waveform this engine generates, in Hz.
    fn sample_rate(&mut self) -> u32;

    /// Synthesize `text`, handing each newly generated waveform segment to
    /// `emit` as it is produced. When `emit` returns `false` the engine must
    /// stop generating and return `Ok(())`; the remainder is discarded.
    fn generate(&mut self, text: &str, emit: Emit) -> Result<(), String>;
}

pub(crate) struct KokoroBackend {
    engine: OfflineTts,
    generation: GenerationConfig,
}

impl KokoroBackend {
    pub(crate) fn create(config: KokoroConfig) -> Result<Self, SherpaTtsBuildError> {
        let (config, generation) = config.into_sherpa()?;
        let engine = OfflineTts::create(&config).ok_or_else(|| {
            SherpaTtsBuildError::CreateEngine(
                "native constructor returned no engine; verify the model and provider".into(),
            )
        })?;
        Ok(Self { engine, generation })
    }
}

impl Backend for KokoroBackend {
    fn sample_rate(&mut self) -> u32 {
        self.engine.sample_rate().max(0) as u32
    }

    fn generate(&mut self, text: &str, mut emit: Emit) -> Result<(), String> {
        // Sherpa invokes the callback with each newly generated sentence's
        // samples; returning false stops generation early. The full waveform
        // it returns at the end duplicates what was already streamed.
        self.engine
            .generate_with_config(
                text,
                &self.generation,
                Some(move |samples: &[f32], _progress: f32| emit(samples)),
            )
            .map(|_streamed_already| ())
            .ok_or_else(|| "Kokoro synthesis returned no audio".into())
    }
}
