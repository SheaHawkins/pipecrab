//! [`OpenAi`]: the handle that implements [`LanguageModel`] over an
//! OpenAI-compatible Chat Completions endpoint.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use pipecrab_lm::{Conversation, GenParams, LanguageModel, LmError, ModelStream, ToolDefinition};

use crate::client::OpenAiClient;
use crate::config::{OpenAiBuildError, OpenAiConfig};
use crate::stream;

/// The epoch one generation captured, against the shared counter that
/// [`OpenAi::cancel`] and every new generation bump.
#[derive(Clone)]
pub(crate) struct Epoch {
    value: u64,
    shared: Arc<AtomicU64>,
}

impl Epoch {
    /// Whether the generation holding this epoch is still the current one.
    pub(crate) fn is_current(&self) -> bool {
        self.shared.load(Ordering::Acquire) == self.value
    }
}

/// What the handle shares between its clones.
struct Inner {
    client: OpenAiClient,
    epoch: Arc<AtomicU64>,
}

/// A cloneable handle to one OpenAI-compatible Chat Completions endpoint.
///
/// Holds no conversation and no decode state: [`LmStage`](pipecrab_lm::LmStage)
/// owns the conversation and replays it in full each turn, and the endpoint
/// keeps nothing between requests.
#[derive(Clone)]
pub struct OpenAi {
    inner: Arc<Inner>,
}

impl OpenAi {
    /// Build a handle from `config`, validating the endpoint and headers.
    /// Issues no request, so a reachable host is not required here.
    pub fn new(config: OpenAiConfig) -> Result<Self, OpenAiBuildError> {
        Ok(Self {
            inner: Arc::new(Inner {
                client: OpenAiClient::new(&config)?,
                epoch: Arc::new(AtomicU64::new(0)),
            }),
        })
    }
}

#[async_trait]
impl LanguageModel for OpenAi {
    async fn generate(
        &self,
        conversation: &Conversation,
        params: &GenParams,
        tools: &[ToolDefinition],
    ) -> Result<ModelStream, LmError> {
        // Starting a generation supersedes any earlier one, so a stream left
        // behind by a dropped turn cannot yield into this one.
        let value = self.inner.epoch.fetch_add(1, Ordering::AcqRel) + 1;
        let epoch = Epoch {
            value,
            shared: Arc::clone(&self.inner.epoch),
        };

        let body = self
            .inner
            .client
            .request_body(conversation, params, tools)?;
        let response = self.inner.client.send(body).await?;

        if self.inner.client.streaming() {
            return Ok(stream::sse_stream(response, epoch));
        }
        let body = response
            .bytes()
            .await
            .map_err(|error| LmError::ProviderStream(format!("completion body failed: {error}")))?;
        Ok(stream::decoded_stream(
            stream::decode_response(&body),
            epoch,
        ))
    }

    fn cancel(&self) {
        // Moving the epoch ends the in-flight stream at its next item; dropping
        // it closes the connection, which is how the host learns to stop.
        self.inner.epoch.fetch_add(1, Ordering::AcqRel);
    }

    async fn save_state(&self) -> Result<Vec<u8>, LmError> {
        // Honestly empty: a hosted endpoint holds no per-session decode state.
        Ok(Vec::new())
    }

    async fn load_state(&self, blob: &[u8]) -> Result<(), LmError> {
        if blob.is_empty() {
            return Ok(());
        }
        Err(LmError::Engine(
            "an OpenAI-compatible endpoint has no session state to restore".into(),
        ))
    }
}
