use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc as std_mpsc};
use std::thread::JoinHandle;

use async_trait::async_trait;
use futures::StreamExt;
use futures::channel::{mpsc, oneshot};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use pipecrab_lm::{
    ChatRole, Conversation, GenParams, LanguageModel, LmError, TokenOut, TokenStream,
};

use crate::{LlamaCppBuildError, LlamaCppConfig};

type TokenSender = mpsc::UnboundedSender<Result<TokenOut, LmError>>;

enum Command {
    Generate {
        epoch: u64,
        conversation: Conversation,
        params: GenParams,
        output: TokenSender,
    },
    SaveState(oneshot::Sender<Result<Vec<u8>, LmError>>),
    LoadState {
        blob: Vec<u8>,
        reply: oneshot::Sender<Result<(), LmError>>,
    },
    Shutdown,
}

struct Inner {
    commands: std_mpsc::Sender<Command>,
    generation_epoch: Arc<AtomicU64>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        self.generation_epoch.fetch_add(1, Ordering::AcqRel);
        let _ = self.commands.send(Command::Shutdown);
        if let Some(worker) = self
            .worker
            .lock()
            .expect("llama.cpp worker mutex poisoned")
            .take()
        {
            let _ = worker.join();
        }
    }
}

/// Cloneable handle to a dedicated native llama.cpp inference worker.
#[derive(Clone)]
pub struct LlamaCpp {
    inner: Arc<Inner>,
}

impl LlamaCpp {
    /// Load a GGUF model and start its long-lived worker.
    ///
    /// This call waits for model and context initialization. Applications should
    /// invoke it from their own initialization task rather than a UI thread.
    pub fn load(config: LlamaCppConfig) -> Result<Self, LlamaCppBuildError> {
        config.validate()?;
        let (commands, receiver) = std_mpsc::channel();
        let (ready_tx, ready_rx) = std_mpsc::sync_channel(1);
        let generation_epoch = Arc::new(AtomicU64::new(0));
        let worker_epoch = Arc::clone(&generation_epoch);
        let worker = std::thread::Builder::new()
            .name("pipecrab-llamacpp".into())
            .spawn(move || worker_main(config, receiver, worker_epoch, ready_tx))
            .map_err(|error| LlamaCppBuildError::Worker(error.to_string()))?;

        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                inner: Arc::new(Inner {
                    commands,
                    generation_epoch,
                    worker: Mutex::new(Some(worker)),
                }),
            }),
            Ok(Err(error)) => {
                let _ = worker.join();
                Err(LlamaCppBuildError::Worker(error))
            }
            Err(error) => {
                let _ = worker.join();
                Err(LlamaCppBuildError::Worker(error.to_string()))
            }
        }
    }

    fn send(&self, command: Command) -> Result<(), LmError> {
        self.inner
            .commands
            .send(command)
            .map_err(|_| LmError::Engine("llama.cpp worker stopped".into()))
    }
}

#[async_trait]
impl LanguageModel for LlamaCpp {
    async fn generate(
        &self,
        conversation: &Conversation,
        params: &GenParams,
    ) -> Result<TokenStream, LmError> {
        let (output, receiver) = mpsc::unbounded();
        let epoch = self.inner.generation_epoch.fetch_add(1, Ordering::AcqRel) + 1;
        self.send(Command::Generate {
            epoch,
            conversation: conversation.clone(),
            params: params.clone(),
            output,
        })?;
        Ok(receiver.boxed())
    }

    fn cancel(&self) {
        self.inner.generation_epoch.fetch_add(1, Ordering::AcqRel);
    }

    async fn save_state(&self) -> Result<Vec<u8>, LmError> {
        let (reply, receiver) = oneshot::channel();
        self.send(Command::SaveState(reply))?;
        receiver
            .await
            .map_err(|_| LmError::Engine("llama.cpp worker stopped while saving state".into()))?
    }

    async fn load_state(&self, blob: &[u8]) -> Result<(), LmError> {
        let (reply, receiver) = oneshot::channel();
        self.send(Command::LoadState {
            blob: blob.to_vec(),
            reply,
        })?;
        receiver
            .await
            .map_err(|_| LmError::Engine("llama.cpp worker stopped while loading state".into()))?
    }
}

fn worker_main(
    config: LlamaCppConfig,
    commands: std_mpsc::Receiver<Command>,
    generation_epoch: Arc<AtomicU64>,
    ready: std_mpsc::SyncSender<Result<(), String>>,
) {
    let mut backend = match LlamaBackend::init() {
        Ok(backend) => backend,
        Err(error) => return send_setup_error(ready, error),
    };
    if !config.logs_enabled {
        backend.void_logs();
    }
    if config.gpu_layers > 0 && !backend.supports_gpu_offload() {
        return send_setup_error(ready, "GPU offload was requested but is unavailable");
    }

    let model_params = LlamaModelParams::default().with_n_gpu_layers(config.gpu_layers);
    let model_params = if config.gpu_layers == 0 {
        match model_params.with_devices(&[]) {
            Ok(params) => params,
            Err(error) => return send_setup_error(ready, error),
        }
    } else {
        model_params
    };
    let model = match LlamaModel::load_from_file(&backend, &config.model_path, &model_params) {
        Ok(model) => model,
        Err(error) => return send_setup_error(ready, error),
    };
    let template = match config.chat_template.as_deref() {
        Some(template) => match LlamaChatTemplate::new(template) {
            Ok(template) => template,
            Err(error) => return send_setup_error(ready, error),
        },
        None => match model.chat_template(None) {
            Ok(template) => template,
            Err(error) => return send_setup_error(ready, error),
        },
    };

    let context_params = LlamaContextParams::default()
        .with_n_ctx(Some(config.context_size))
        .with_n_batch(config.batch_size.get())
        .with_n_ubatch(config.batch_size.get().min(512))
        .with_offload_kqv(config.gpu_layers > 0)
        .with_n_threads(i32::try_from(config.threads.get()).expect("validated thread count"))
        .with_n_threads_batch(
            i32::try_from(config.threads_batch.get()).expect("validated thread count"),
        );
    let mut context = match model.new_context(&backend, context_params) {
        Ok(context) => context,
        Err(error) => return send_setup_error(ready, error),
    };
    if ready.send(Ok(())).is_err() {
        return;
    }

    let mut processed_tokens = Vec::new();
    for command in commands {
        match command {
            Command::Generate {
                epoch,
                conversation,
                params,
                output,
            } => {
                if let Err(error) = generate(
                    &model,
                    &template,
                    &mut context,
                    &config,
                    &conversation,
                    &params,
                    &output,
                    &generation_epoch,
                    epoch,
                    &mut processed_tokens,
                ) {
                    let _ = output.unbounded_send(Err(LmError::Engine(error)));
                }
            }
            Command::SaveState(reply) => {
                let _ = reply.send(save_state(&context, &processed_tokens));
            }
            Command::LoadState { blob, reply } => {
                let result = load_state(&mut context, &blob, &mut processed_tokens);
                let _ = reply.send(result);
            }
            Command::Shutdown => break,
        }
    }
}

fn send_setup_error(ready: std_mpsc::SyncSender<Result<(), String>>, error: impl ToString) {
    let _ = ready.send(Err(error.to_string()));
}

#[allow(clippy::too_many_arguments)]
fn generate(
    model: &LlamaModel,
    template: &LlamaChatTemplate,
    context: &mut LlamaContext<'_>,
    config: &LlamaCppConfig,
    conversation: &Conversation,
    params: &GenParams,
    output: &TokenSender,
    generation_epoch: &AtomicU64,
    epoch: u64,
    processed_tokens: &mut Vec<LlamaToken>,
) -> Result<(), String> {
    let messages = conversation
        .messages
        .iter()
        .map(|message| {
            let role = match message.role {
                ChatRole::System => "system",
                ChatRole::User => "user",
                ChatRole::Assistant => "assistant",
            };
            LlamaChatMessage::new(role.into(), message.content.to_string())
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    let prompt = model
        .apply_chat_template(template, &messages, true)
        .map_err(|error| error.to_string())?;
    let prompt_tokens = model
        .str_to_token(&prompt, AddBos::Always)
        .map_err(|error| error.to_string())?;
    if prompt_tokens.is_empty() {
        return Err("chat template produced an empty prompt".into());
    }

    let context_size = context.n_ctx() as usize;
    if prompt_tokens.len() >= context_size {
        return Err(format!(
            "prompt uses {} tokens but context holds {context_size}",
            prompt_tokens.len()
        ));
    }
    let requested = params.max_tokens.unwrap_or(config.default_max_tokens.get()) as usize;
    let max_tokens = requested.min(context_size - prompt_tokens.len());

    context.clear_kv_cache();
    processed_tokens.clear();
    let batch_capacity = config.batch_size.get() as usize;
    let mut batch = LlamaBatch::new(batch_capacity, 1);
    for (chunk_index, chunk) in prompt_tokens.chunks(batch_capacity).enumerate() {
        batch.clear();
        let base = chunk_index * batch_capacity;
        for (index, token) in chunk.iter().enumerate() {
            let absolute = base + index;
            let logits = absolute + 1 == prompt_tokens.len();
            batch
                .add(
                    *token,
                    i32::try_from(absolute).map_err(|error| error.to_string())?,
                    &[0],
                    logits,
                )
                .map_err(|error| error.to_string())?;
        }
        context
            .decode(&mut batch)
            .map_err(|error| error.to_string())?;
    }
    processed_tokens.extend_from_slice(&prompt_tokens);

    let temperature = params.temperature.unwrap_or(config.default_temperature);
    if !temperature.is_finite() || temperature < 0.0 {
        return Err("temperature must be finite and non-negative".into());
    }
    let mut samplers = Vec::new();
    if let Some(grammar) = params.grammar.as_deref() {
        samplers.push(
            LlamaSampler::grammar(model, grammar, "root").map_err(|error| error.to_string())?,
        );
    }
    if temperature == 0.0 {
        samplers.push(LlamaSampler::greedy());
    } else {
        samplers.extend([
            LlamaSampler::top_k(40),
            LlamaSampler::top_p(0.95, 1),
            LlamaSampler::temp(temperature),
            LlamaSampler::dist(config.seed),
        ]);
    }
    let mut sampler = LlamaSampler::chain_simple(samplers);
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    for position in (prompt_tokens.len()..).take(max_tokens) {
        if generation_epoch.load(Ordering::Acquire) != epoch {
            break;
        }
        let token = sampler.sample(context, batch.n_tokens() - 1);
        sampler.accept(token);
        if model.is_eog_token(token) {
            break;
        }
        let delta = model
            .token_to_piece(token, &mut decoder, true, None)
            .map_err(|error| error.to_string())?;
        if !delta.is_empty()
            && output
                .unbounded_send(Ok(TokenOut {
                    delta: Arc::<str>::from(delta),
                }))
                .is_err()
        {
            break;
        }
        if generation_epoch.load(Ordering::Acquire) != epoch {
            break;
        }
        batch.clear();
        batch
            .add(
                token,
                i32::try_from(position).map_err(|error| error.to_string())?,
                &[0],
                true,
            )
            .map_err(|error| error.to_string())?;
        context
            .decode(&mut batch)
            .map_err(|error| error.to_string())?;
        processed_tokens.push(token);
    }
    Ok(())
}

fn save_state(
    context: &LlamaContext<'_>,
    processed_tokens: &[LlamaToken],
) -> Result<Vec<u8>, LmError> {
    let file = tempfile::NamedTempFile::new()
        .map_err(|error| LmError::Engine(format!("create state file: {error}")))?;
    context
        .state_save_file(file.path(), processed_tokens)
        .map_err(|error| LmError::Engine(error.to_string()))?;
    std::fs::read(file.path()).map_err(|error| LmError::Engine(format!("read state file: {error}")))
}

fn load_state(
    context: &mut LlamaContext<'_>,
    blob: &[u8],
    processed_tokens: &mut Vec<LlamaToken>,
) -> Result<(), LmError> {
    let file = tempfile::NamedTempFile::new()
        .map_err(|error| LmError::Engine(format!("create state file: {error}")))?;
    std::fs::write(file.path(), blob)
        .map_err(|error| LmError::Engine(format!("write state file: {error}")))?;
    let tokens = context
        .state_load_file(file.path(), context.n_ctx() as usize)
        .map_err(|error| LmError::Engine(error.to_string()))?;
    *processed_tokens = tokens;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroU32;

    #[test]
    fn build_error_for_missing_model_does_not_start_backend() {
        let error = match LlamaCpp::load(LlamaCppConfig::new("missing.gguf")) {
            Ok(_) => panic!("missing model must fail"),
            Err(error) => error,
        };
        assert!(matches!(error, LlamaCppBuildError::ModelNotFound(_)));
    }

    #[test]
    fn generation_defaults_are_non_zero() {
        let config = LlamaCppConfig::new("missing.gguf");
        assert_eq!(config.context_size, NonZeroU32::new(4096).unwrap());
        assert!(config.default_max_tokens.get() > 0);
    }
}
