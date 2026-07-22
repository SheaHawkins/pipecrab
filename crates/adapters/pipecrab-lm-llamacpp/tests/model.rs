use std::num::NonZeroU32;

use futures::{StreamExt, executor::block_on};
use pipecrab_lm::{Conversation, GenParams, LanguageModel, Message, ModelDelta};
use pipecrab_lm_llamacpp::{LlamaCpp, LlamaCppConfig};

#[test]
#[ignore = "requires PIPECRAB_LLAMA_MODEL to point to a chat GGUF"]
fn streams_text_from_a_real_model() {
    let path = std::env::var_os("PIPECRAB_LLAMA_MODEL")
        .expect("PIPECRAB_LLAMA_MODEL must point to a chat GGUF");
    let model = LlamaCpp::load(LlamaCppConfig::new(path).with_generation_defaults(
        NonZeroU32::new(16).expect("non-zero"),
        0.0,
        42,
    ))
    .expect("load model");
    let conversation = Conversation {
        messages: vec![
            Message::system("Answer briefly."),
            Message::user("Say hello in one sentence."),
        ],
    };

    let deltas = block_on(async {
        model
            .generate(&conversation, &GenParams::default(), &[])
            .await
            .expect("start generation")
            .collect::<Vec<_>>()
            .await
    });
    let text = deltas.into_iter().fold(String::new(), |mut text, delta| {
        match delta.expect("decode token") {
            ModelDelta::Text(piece) => text.push_str(&piece),
            ModelDelta::ToolCall(_) => panic!("native adapter emits only text"),
        }
        text
    });
    assert!(!text.trim().is_empty(), "model returned no text");

    let state = block_on(model.save_state()).expect("save state");
    assert!(!state.is_empty(), "saved state is empty");
    block_on(model.load_state(&state)).expect("restore state");
}
