//! Multi-turn generation against a real model, exercising the worker's
//! incremental prefill: the second turn's prompt shares a prefix with the
//! processed tokens, so only the new suffix is decoded.
//!
//! A separate integration-test file from `model.rs` on purpose: llama.cpp's
//! backend is a process-global singleton, and each test file runs as its own
//! process.

use std::num::NonZeroU32;

use futures::{StreamExt, executor::block_on};
use pipecrab_lm::{Conversation, GenParams, LanguageModel, Message, ModelDelta};
use pipecrab_lm_llamacpp::{LlamaCpp, LlamaCppConfig};

#[test]
#[ignore = "requires PIPECRAB_LLAMA_MODEL to point to a chat GGUF"]
fn second_turn_generates_from_a_reused_cache() {
    let path = std::env::var_os("PIPECRAB_LLAMA_MODEL")
        .expect("PIPECRAB_LLAMA_MODEL must point to a chat GGUF");
    let model = LlamaCpp::load(LlamaCppConfig::new(path).with_generation_defaults(
        NonZeroU32::new(16).expect("non-zero"),
        0.0,
        42,
    ))
    .expect("load model");

    // Turn one primes the KV cache with the prompt and the reply.
    let mut conversation = Conversation {
        messages: vec![
            Message::system("Answer briefly."),
            Message::user("Name one primary color."),
        ],
    };
    let first = collect_reply(&model, &conversation);
    assert!(!first.trim().is_empty(), "first turn returned no text");

    // Turn two extends the conversation, so the worker diffs the new prompt
    // against the processed tokens and decodes only the suffix.
    conversation.messages.push(Message::assistant(first));
    conversation.messages.push(Message::user("Name another."));
    let second = collect_reply(&model, &conversation);
    assert!(
        !second.trim().is_empty(),
        "second turn returned no text after cache reuse"
    );
}

fn collect_reply(model: &LlamaCpp, conversation: &Conversation) -> String {
    let deltas = block_on(async {
        model
            .generate(conversation, &GenParams::default(), &[])
            .await
            .expect("start generation")
            .collect::<Vec<_>>()
            .await
    });
    deltas.into_iter().fold(String::new(), |mut text, delta| {
        match delta.expect("decode token") {
            ModelDelta::Text(piece) => text.push_str(&piece),
            ModelDelta::ToolCall(_) => panic!("native adapter emits only text"),
        }
        text
    })
}
