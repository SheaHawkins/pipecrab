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
    Conversation, GenParams, LanguageModel, LmError, Message, ModelDelta, ModelStream,
    ToolDefinition,
};

use crate::dialect::{TOOL_CALL_ROOT, regex_escape};
use crate::intercept::{CallInterceptor, Intercepted, mint_call_id};
use crate::{ChatMlXml, LlamaCppBuildError, LlamaCppConfig, ToolDialect};

type TokenSender = mpsc::UnboundedSender<Result<ModelDelta, LmError>>;

enum Command {
    Generate {
        epoch: u64,
        conversation: Conversation,
        params: GenParams,
        tools: Vec<ToolDefinition>,
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
        tools: &[ToolDefinition],
    ) -> Result<ModelStream, LmError> {
        let (output, receiver) = mpsc::unbounded();
        let epoch = self.inner.generation_epoch.fetch_add(1, Ordering::AcqRel) + 1;
        self.send(Command::Generate {
            epoch,
            conversation: conversation.clone(),
            params: params.clone(),
            tools: tools.to_vec(),
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

    let mut session = Session {
        dialect: config
            .tool_dialect
            .clone()
            .unwrap_or_else(|| Arc::new(ChatMlXml)),
        // A dialect the caller named is their assertion that it fits, and a GGUF
        // that declares no name says nothing either way; only a defaulted
        // dialect facing a named model is measured.
        model_name: config
            .tool_dialect
            .is_none()
            .then(|| model.meta_val_str("general.name").ok())
            .flatten(),
        processed_tokens: Vec::new(),
        tools: None,
    };
    for command in commands {
        match command {
            Command::Generate {
                epoch,
                conversation,
                params,
                tools,
                output,
            } => {
                if let Err(error) = generate(
                    &model,
                    &template,
                    &mut context,
                    &config,
                    &conversation,
                    &params,
                    &tools,
                    &output,
                    &generation_epoch,
                    epoch,
                    &mut session,
                ) {
                    let _ = output.unbounded_send(Err(LmError::Engine(error)));
                }
            }
            Command::SaveState(reply) => {
                let _ = reply.send(save_state(&context, &session.processed_tokens));
            }
            Command::LoadState { blob, reply } => {
                let result = load_state(&mut context, &blob, &mut session.processed_tokens);
                let _ = reply.send(result);
            }
            Command::Shutdown => break,
        }
    }
}

/// Render the conversation as the role/content pairs llama.cpp's chat template
/// takes, folding the dialect's tool blocks into roles that template knows.
///
/// llama.cpp's built-in ChatML renderer passes a role through verbatim, so
/// `tool` would emit `<|im_start|>tool` — a sequence no ChatML model was
/// trained on. The dialect names the role each of its blocks belongs under.
fn render_messages(
    conversation: &Conversation,
    dialect: &dyn ToolDialect,
    tools: Option<&ToolPrompt>,
) -> Result<Vec<LlamaChatMessage>, String> {
    let mut turns: Vec<(&str, String)> = Vec::with_capacity(conversation.messages.len() + 1);
    for message in &conversation.messages {
        turns.push(match message {
            Message::System { content } => ("system", content.to_string()),
            Message::User { content } => ("user", content.to_string()),
            // Dropping the calls here would erase them from history, and the
            // model would re-issue what it already asked for.
            Message::Assistant {
                content,
                tool_calls,
            } => (
                "assistant",
                match tools {
                    Some(_) => content.to_string() + &dialect.tool_calls(tool_calls),
                    None => content.to_string(),
                },
            ),
            Message::ToolResult { content, .. } => match tools {
                Some(_) => dialect.tool_result(content),
                None => ("tool", content.to_string()),
            },
            Message::Event {
                source,
                kind,
                content,
            } => ("user", format!("[{source}/{kind}] {content}")),
        });
    }
    if let Some(tools) = tools {
        match turns.iter_mut().find(|(role, _)| *role == "system") {
            Some((_, content)) => content.push_str(&tools.declarations),
            // llama.cpp's ChatML renderer injects no system turn of its own, so
            // without one the declarations would never reach the model.
            None => turns.insert(0, ("system", tools.declarations.trim_start().to_string())),
        }
    }
    turns
        .into_iter()
        .map(|(role, content)| LlamaChatMessage::new(role.into(), content))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())
}

/// Emit one interception, reporting whether the stream should keep going.
fn emit(output: &TokenSender, dialect: &dyn ToolDialect, item: Intercepted) -> bool {
    let delta = match item {
        Intercepted::Text(text) if text.is_empty() => return true,
        Intercepted::Text(text) => Ok(ModelDelta::Text(Arc::from(text))),
        Intercepted::Call(body) => dialect
            .parse(&body)
            .and_then(|(name, arguments)| ModelDelta::tool_call(mint_call_id(), name, arguments)),
    };
    match delta {
        Ok(delta) => output.unbounded_send(Ok(delta)).is_ok(),
        // A body the grammar could not fully constrain: report it and stop
        // rather than pushing half a call down the pipeline.
        Err(error) => {
            let _ = output.unbounded_send(Err(error));
            false
        }
    }
}

/// Worker state that outlives one turn.
struct Session {
    dialect: Arc<dyn ToolDialect>,
    /// The GGUF's `general.name`, present only while the dialect is unverified.
    model_name: Option<String>,
    processed_tokens: Vec<LlamaToken>,
    tools: Option<ToolPrompt>,
}

/// Everything one tool set contributes to a turn.
///
/// Cached because both halves are prompt- and cache-sensitive: the declarations
/// are a prompt prefix, so a byte that moves between turns costs a full
/// re-prefill, and converting the schemas to GBNF crosses into llama.cpp.
struct ToolPrompt {
    tools: Vec<ToolDefinition>,
    declarations: String,
    grammar: String,
    /// The open delimiter as a trigger pattern, sized for `grammar_lazy_patterns`.
    trigger: [String; 1],
}

impl ToolPrompt {
    fn build(dialect: &dyn ToolDialect, tools: &[ToolDefinition]) -> Result<Self, LmError> {
        Ok(Self {
            declarations: dialect.declarations(tools),
            grammar: dialect.grammar(tools)?,
            trigger: [regex_escape(dialect.open())],
            tools: tools.to_vec(),
        })
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
    tools: &[ToolDefinition],
    output: &TokenSender,
    generation_epoch: &AtomicU64,
    epoch: u64,
    session: &mut Session,
) -> Result<(), String> {
    let Session {
        dialect,
        model_name,
        processed_tokens,
        tools: cached_tools,
    } = session;
    let dialect = dialect.as_ref();

    // Without tools nothing here applies: no declarations, no grammar, no
    // interception — the turn renders and streams as it would with no dialect.
    let tool_prompt = if tools.is_empty() {
        None
    } else {
        if let Some(name) = model_name.as_deref().filter(|name| !dialect.supports(name)) {
            return Err(format!(
                "model {name:?} is not one {dialect:?} handles ({}); pass \
                 LlamaCppConfig::with_tool_dialect to override",
                dialect.supported_models().join(", "),
            ));
        }
        if cached_tools
            .as_ref()
            .is_none_or(|cached| cached.tools != tools)
        {
            *cached_tools =
                Some(ToolPrompt::build(dialect, tools).map_err(|error| error.to_string())?);
        }
        cached_tools.as_ref()
    };

    let messages = render_messages(conversation, dialect, tool_prompt)?;
    let mut prompt = model
        .apply_chat_template(template, &messages, true)
        .map_err(|error| error.to_string())?;
    // After the generation prompt, so the model continues from it rather than
    // producing it. `llama_chat_apply_template` takes no template kwargs, so a
    // family whose non-thinking mode is a kwarg gets it applied here instead.
    if let Some(assistant_prefix) = config.assistant_prefix.as_deref() {
        prompt.push_str(assistant_prefix);
    }
    // `AddBos::Always` maps to llama.cpp's `add_special`, which adds BOS only
    // when the vocab metadata asks for one — so BOS-less chat vocabs are
    // already handled. The residual case is a vocab that wants BOS *and* a
    // template whose rendering also begins with it (Llama-3-style
    // `<|begin_of_text|>`): that yields a doubled leading BOS, which
    // llama.cpp's own examples detect the same way and warn about. Collapse it.
    let mut prompt_tokens = model
        .str_to_token(&prompt, AddBos::Always)
        .map_err(|error| error.to_string())?;
    let bos = model.token_bos();
    if prompt_tokens.len() >= 2 && prompt_tokens[0] == bos && prompt_tokens[1] == bos {
        prompt_tokens.remove(0);
    }
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

    // Reuse the KV cache across turns: the conversation is append-mostly, so
    // the previous prompt plus its reply is usually a prefix of this prompt.
    // Keep the shared prefix, drop the cache past it, and decode only the new
    // suffix — per-turn prefill cost is then proportional to the new tokens,
    // not the whole conversation. At least the final prompt token is always
    // re-decoded so sampling has fresh logits for it.
    let shared = processed_tokens
        .iter()
        .zip(&prompt_tokens)
        .take_while(|(cached, incoming)| cached == incoming)
        .count()
        .min(prompt_tokens.len() - 1);
    let shared = match context.clear_kv_cache_seq(
        Some(0),
        Some(u32::try_from(shared).map_err(|error| error.to_string())?),
        None,
    ) {
        Ok(true) => shared,
        // Partial removal is unsupported (e.g. non-transformer memory): fall
        // back to a full re-prefill.
        Ok(false) | Err(_) => {
            context.clear_kv_cache();
            0
        }
    };
    processed_tokens.truncate(shared);

    let batch_capacity = config.batch_size.get() as usize;
    let mut batch = LlamaBatch::new(batch_capacity, 1);
    for (chunk_index, chunk) in prompt_tokens[shared..].chunks(batch_capacity).enumerate() {
        // A barge-in can land mid-prefill; checking between decode chunks
        // keeps cancellation bounded by one decode step here too.
        if generation_epoch.load(Ordering::Acquire) != epoch {
            return Ok(());
        }
        batch.clear();
        let base = shared + chunk_index * batch_capacity;
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
        // Record per chunk, so an early return above leaves this vector
        // agreeing with what the KV cache actually holds.
        processed_tokens.extend_from_slice(chunk);
    }

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
    // Lazy, never eager: an eager grammar would make *every* turn a tool call,
    // so the agent could never speak an ordinary sentence again. Until the open
    // delimiter appears the sampler is a no-op and text is unconstrained.
    if let Some(tools) = tool_prompt {
        samplers.push(
            LlamaSampler::grammar_lazy_patterns(
                model,
                &tools.grammar,
                TOOL_CALL_ROOT,
                &tools.trigger,
                &[],
            )
            .map_err(|error| error.to_string())?,
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
    let mut interceptor =
        tool_prompt.map(|_| CallInterceptor::new(dialect.open(), dialect.close()));
    let mut intercepted = Vec::new();
    // Set where the turn ends without reaching its own end: a barge-in, or a
    // receiver that went away. Neither leaves a tail worth flushing.
    let mut stopped = false;
    for position in (prompt_tokens.len()..).take(max_tokens) {
        if generation_epoch.load(Ordering::Acquire) != epoch {
            stopped = true;
            break;
        }
        // `sample` accepts the token into the sampler chain itself; accepting it
        // again here would advance the grammar twice per token.
        let token = sampler.sample(context, batch.n_tokens() - 1);
        if model.is_eog_token(token) {
            break;
        }
        let piece = model
            .token_to_piece(token, &mut decoder, true, None)
            .map_err(|error| error.to_string())?;
        match interceptor.as_mut() {
            Some(interceptor) => {
                intercepted.clear();
                interceptor.push(&piece, &mut intercepted);
                stopped = !intercepted
                    .drain(..)
                    .all(|item| emit(output, dialect, item));
            }
            None => stopped = !emit(output, dialect, Intercepted::Text(piece)),
        }
        if stopped || generation_epoch.load(Ordering::Acquire) != epoch {
            stopped = true;
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
    if let Some(interceptor) = interceptor.as_mut().filter(|_| !stopped) {
        match interceptor.finish() {
            Ok(Some(tail)) => {
                emit(output, dialect, tail);
            }
            Ok(None) => {}
            Err(error) => {
                let _ = output.unbounded_send(Err(error));
            }
        }
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
