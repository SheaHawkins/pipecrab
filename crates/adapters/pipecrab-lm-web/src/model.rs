//! [`TransformersLm`]: one Transformers.js text-generation pipeline behind
//! [`LanguageModel`].

use std::sync::Arc;

use async_trait::async_trait;
use js_sys::{Array, Object};
use pipecrab_lm::{
    Conversation, GenParams, LanguageModel, LmError, Message, ModelDelta, ModelStream,
    ToolDefinition,
};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use crate::TransformersLmConfig;
use crate::config::set;

#[wasm_bindgen(module = "/js/generate.js")]
extern "C" {
    type Generation;

    #[wasm_bindgen(catch, js_name = loadGenerator)]
    fn load_generator(
        transformers: &JsValue,
        model: &str,
        options: &Object,
    ) -> Result<js_sys::Promise, JsValue>;

    #[wasm_bindgen(catch, js_name = startGeneration)]
    fn start_generation(
        handle: &JsValue,
        messages: &Array,
        options: &Object,
    ) -> Result<Generation, JsValue>;

    #[wasm_bindgen(js_name = interruptCurrent)]
    fn interrupt_current(handle: &JsValue);

    #[wasm_bindgen(method)]
    fn next(this: &Generation) -> js_sys::Promise;

    #[wasm_bindgen(method)]
    fn interrupt(this: &Generation);
}

/// Streams chat replies from a Transformers.js text-generation pipeline.
///
/// Text only: a generation carrying tools or a grammar is rejected, since
/// Transformers.js has no constrained decoding to hold a tool call to its
/// schema. Stateless between calls — every generation re-reads the whole
/// conversation — so [`save_state`](LanguageModel::save_state) has nothing to
/// checkpoint.
pub struct TransformersLm {
    /// The JS-side pipeline and the generation running on it.
    handle: JsValue,
    max_new_tokens: u32,
}

impl TransformersLm {
    /// Fetch and warm the pipeline named by `config` on the page's Transformers.js.
    ///
    /// `transformers` is the namespace the page imported. The model download and
    /// its compilation both happen here, so the wait — hundreds of megabytes on
    /// a cold cache — is paid at construction rather than on the first reply.
    ///
    /// # Errors
    ///
    /// Returns [`LmError::Engine`] if the model cannot be fetched, or if the
    /// engine rejects the requested device or quantization.
    pub async fn load(
        transformers: &JsValue,
        config: TransformersLmConfig,
    ) -> Result<Self, LmError> {
        let promise = load_generator(transformers, &config.model, &config.pipeline_options())
            .map_err(|error| LmError::Engine(message(error)))?;
        let handle = JsFuture::from(promise)
            .await
            .map_err(|error| LmError::Engine(message(error)))?;
        Ok(Self {
            handle,
            max_new_tokens: config.max_new_tokens,
        })
    }

    /// Options for one call: how long the reply may run and how it samples.
    fn generate_options(&self, params: &GenParams) -> Result<Object, LmError> {
        let options = Object::new();
        let max_new_tokens = params.max_tokens.unwrap_or(self.max_new_tokens);
        set(&options, "max_new_tokens", Some(max_new_tokens));
        if let Some(temperature) = params.temperature {
            if !temperature.is_finite() || temperature < 0.0 {
                return Err(LmError::Engine(
                    "temperature must be finite and non-negative".into(),
                ));
            }
            set(&options, "do_sample", Some(temperature > 0.0));
            set(&options, "temperature", Some(temperature));
        }
        Ok(options)
    }
}

/// One generation's text, pulled a piece at a time. Dropping it — the stage
/// stopped reading — interrupts the generation, so a torn-down pipeline does
/// not leave the model decoding a reply nobody will hear.
struct Pull(Generation);

impl Drop for Pull {
    fn drop(&mut self) {
        self.0.interrupt();
    }
}

// `?Send` unconditionally: this crate exists only on wasm32, where the trait is
// declared exactly this way.
#[async_trait(?Send)]
impl LanguageModel for TransformersLm {
    async fn generate(
        &self,
        conversation: &Conversation,
        params: &GenParams,
        tools: &[ToolDefinition],
    ) -> Result<ModelStream, LmError> {
        if !tools.is_empty() {
            return Err(LmError::Engine(
                "TransformersLm does not support tool calling".into(),
            ));
        }
        if params.grammar.is_some() {
            return Err(LmError::Engine(
                "TransformersLm does not support grammar-constrained generation".into(),
            ));
        }
        let options = self.generate_options(params)?;
        let generation = start_generation(&self.handle, &messages(conversation), &options)
            .map_err(|error| LmError::Engine(message(error)))?;

        let stream = futures::stream::try_unfold(Pull(generation), |pull| async move {
            loop {
                let text = JsFuture::from(pull.0.next())
                    .await
                    .map_err(|error| LmError::ProviderStream(message(error)))?;
                if text.is_undefined() {
                    return Ok(None);
                }
                let text = text.as_string().ok_or_else(|| {
                    LmError::ProviderStream(format!("streamer yielded {text:?}, not a string"))
                })?;
                // The streamer flushes on word boundaries and may flush nothing.
                if !text.is_empty() {
                    return Ok(Some((ModelDelta::Text(Arc::from(text)), pull)));
                }
            }
        });
        Ok(Box::pin(stream))
    }

    fn cancel(&self) {
        interrupt_current(&self.handle);
    }

    async fn save_state(&self) -> Result<Vec<u8>, LmError> {
        Ok(Vec::new())
    }

    async fn load_state(&self, _blob: &[u8]) -> Result<(), LmError> {
        Ok(())
    }
}

/// The conversation as chat-template messages: `{ role, content }` objects.
///
/// A tool result goes under the `tool` role and an external event is spoken as
/// a user turn tagged with its source, the same rendering the native llama.cpp
/// adapter uses without tools.
fn messages(conversation: &Conversation) -> Array {
    conversation
        .messages
        .iter()
        .map(|message| {
            let (role, content) = match message {
                Message::System { content } => ("system", content.to_string()),
                Message::User { content } => ("user", content.to_string()),
                Message::Assistant { content, .. } => ("assistant", content.to_string()),
                Message::ToolResult { content, .. } => ("tool", content.to_string()),
                Message::Event {
                    source,
                    kind,
                    content,
                } => ("user", format!("[{source}/{kind}] {content}")),
            };
            let object = Object::new();
            set(&object, "role", Some(role));
            set(&object, "content", Some(content));
            object
        })
        .collect()
}

/// Render a rejected promise or a thrown error as text.
fn message(error: JsValue) -> String {
    error
        .as_string()
        .or_else(|| {
            js_sys::Reflect::get(&error, &JsValue::from_str("message"))
                .ok()
                .and_then(|m| m.as_string())
        })
        .unwrap_or_else(|| format!("{error:?}"))
}
