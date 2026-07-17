use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

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
        let stopped = Arc::new(AtomicBool::new(false));
        let callback_stopped = Arc::clone(&stopped);
        let result = self.engine.generate_with_config(
            text,
            &self.generation,
            Some(move |samples: &[f32], _progress: f32| {
                let keep_going = emit(samples);
                if !keep_going {
                    callback_stopped.store(true, Ordering::Release);
                }
                keep_going
            }),
        );
        // An emit-requested stop is the trait's normal early exit, not an
        // engine failure — report Ok(()) regardless of whether the native call
        // treats a stopped generation as "no audio".
        if stopped.load(Ordering::Acquire) {
            return Ok(());
        }
        result
            .map(|_streamed_already| ())
            .ok_or_else(|| "Kokoro synthesis returned no audio".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The [`Backend`] contract: an emit-requested stop returns `Ok(())`,
    /// whatever the native call makes of a stopped generation.
    #[test]
    #[ignore = "requires SHERPA_KOKORO_MODEL, SHERPA_KOKORO_VOICES, SHERPA_KOKORO_TOKENS, and SHERPA_KOKORO_DATA_DIR"]
    fn an_emit_requested_stop_is_ok_with_the_real_engine() {
        let config = KokoroConfig::new(
            std::env::var("SHERPA_KOKORO_MODEL").expect("set SHERPA_KOKORO_MODEL"),
            std::env::var("SHERPA_KOKORO_VOICES").expect("set SHERPA_KOKORO_VOICES"),
            std::env::var("SHERPA_KOKORO_TOKENS").expect("set SHERPA_KOKORO_TOKENS"),
            std::env::var("SHERPA_KOKORO_DATA_DIR").expect("set SHERPA_KOKORO_DATA_DIR"),
        );
        let mut backend = KokoroBackend::create(config).unwrap();

        let result = backend.generate(
            "One sentence to stop at. A second sentence that is never generated.",
            Box::new(|_samples| false),
        );

        assert_eq!(result, Ok(()));
    }
}
