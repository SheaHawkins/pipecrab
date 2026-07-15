use std::num::{NonZeroU32, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Construction settings for a native llama.cpp worker.
#[derive(Clone, Debug)]
pub struct LlamaCppConfig {
    pub(crate) model_path: PathBuf,
    pub(crate) context_size: NonZeroU32,
    pub(crate) batch_size: NonZeroU32,
    pub(crate) threads: NonZeroUsize,
    pub(crate) threads_batch: NonZeroUsize,
    pub(crate) gpu_layers: u32,
    pub(crate) default_max_tokens: NonZeroU32,
    pub(crate) default_temperature: f32,
    pub(crate) seed: u32,
    pub(crate) chat_template: Option<Arc<str>>,
    pub(crate) logs_enabled: bool,
}

impl LlamaCppConfig {
    /// Create mobile-oriented settings for the GGUF model at `model_path`.
    ///
    /// The default reserves two logical cores for audio and UI work, capped at
    /// six inference threads. It uses a 4096-token context, 512-token batches,
    /// no GPU offload, and at most 256 generated tokens per turn.
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        let available = std::thread::available_parallelism()
            .map(NonZeroUsize::get)
            .unwrap_or(4);
        let threads = available.saturating_sub(2).clamp(1, 6);
        Self {
            model_path: model_path.into(),
            context_size: NonZeroU32::new(4096).expect("4096 is non-zero"),
            batch_size: NonZeroU32::new(512).expect("512 is non-zero"),
            threads: NonZeroUsize::new(threads).expect("thread count is non-zero"),
            threads_batch: NonZeroUsize::new(threads).expect("thread count is non-zero"),
            gpu_layers: 0,
            default_max_tokens: NonZeroU32::new(256).expect("256 is non-zero"),
            default_temperature: 0.7,
            seed: 0xC0FFEE,
            chat_template: None,
            logs_enabled: false,
        }
    }

    /// Path of the GGUF model loaded by the worker.
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Set the maximum context length.
    pub fn with_context_size(mut self, context_size: NonZeroU32) -> Self {
        self.context_size = context_size;
        self
    }

    /// Set the maximum prompt batch size.
    pub fn with_batch_size(mut self, batch_size: NonZeroU32) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set decode and prompt-processing thread counts.
    pub fn with_threads(mut self, threads: NonZeroUsize, threads_batch: NonZeroUsize) -> Self {
        self.threads = threads;
        self.threads_batch = threads_batch;
        self
    }

    /// Request that llama.cpp offload this many model layers to its compiled GPU
    /// backend. Use `u32::MAX` for all layers.
    pub fn with_gpu_layers(mut self, gpu_layers: u32) -> Self {
        self.gpu_layers = gpu_layers;
        self
    }

    /// Set defaults used when [`GenParams`](pipecrab_lm::GenParams) leaves a
    /// value unspecified.
    pub fn with_generation_defaults(
        mut self,
        max_tokens: NonZeroU32,
        temperature: f32,
        seed: u32,
    ) -> Self {
        self.default_max_tokens = max_tokens;
        self.default_temperature = temperature;
        self.seed = seed;
        self
    }

    /// Override the chat template embedded in the GGUF metadata.
    ///
    /// The embedded template is preferred. This override accepts either a
    /// llama.cpp template name or a supported Jinja template.
    pub fn with_chat_template(mut self, chat_template: impl Into<Arc<str>>) -> Self {
        self.chat_template = Some(chat_template.into());
        self
    }

    /// Enable or suppress llama.cpp diagnostic logging.
    pub fn with_logs_enabled(mut self, logs_enabled: bool) -> Self {
        self.logs_enabled = logs_enabled;
        self
    }

    pub(crate) fn validate(&self) -> Result<(), LlamaCppBuildError> {
        if !self.model_path.is_file() {
            return Err(LlamaCppBuildError::ModelNotFound(self.model_path.clone()));
        }
        if self.batch_size.get() > self.context_size.get() {
            return Err(LlamaCppBuildError::InvalidConfig(
                "batch size cannot exceed context size".into(),
            ));
        }
        if !self.default_temperature.is_finite() || self.default_temperature < 0.0 {
            return Err(LlamaCppBuildError::InvalidConfig(
                "temperature must be finite and non-negative".into(),
            ));
        }
        if i32::try_from(self.threads.get()).is_err()
            || i32::try_from(self.threads_batch.get()).is_err()
        {
            return Err(LlamaCppBuildError::InvalidConfig(
                "thread count exceeds llama.cpp's supported range".into(),
            ));
        }
        Ok(())
    }
}

/// Why a llama.cpp worker could not be constructed.
#[derive(Debug, thiserror::Error)]
pub enum LlamaCppBuildError {
    /// The configured model does not exist or is not a regular file.
    #[error("GGUF model not found: {0}")]
    ModelNotFound(PathBuf),
    /// A setting is internally inconsistent or outside llama.cpp's range.
    #[error("invalid llama.cpp configuration: {0}")]
    InvalidConfig(String),
    /// The worker thread could not start or initialize llama.cpp.
    #[error("failed to initialize llama.cpp worker: {0}")]
    Worker(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_model_is_rejected_before_worker_start() {
        let config = LlamaCppConfig::new("definitely-not-a-model.gguf");
        assert!(matches!(
            config.validate(),
            Err(LlamaCppBuildError::ModelNotFound(_))
        ));
    }

    #[test]
    fn batch_cannot_exceed_context() {
        let file = tempfile::NamedTempFile::new().expect("temp file");
        let config = LlamaCppConfig::new(file.path())
            .with_context_size(NonZeroU32::new(128).expect("non-zero"))
            .with_batch_size(NonZeroU32::new(256).expect("non-zero"));
        assert!(matches!(
            config.validate(),
            Err(LlamaCppBuildError::InvalidConfig(_))
        ));
    }
}
