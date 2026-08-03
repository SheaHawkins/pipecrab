//! The wire-compatibility tripwire: one real turn against a live daemon.
//!
//! Ignored by default — the mirrored protocol in this crate is validated
//! against ZeroClaw here, so run it after upgrading the daemon:
//!
//! ```console
//! ZEROCLAW_LIVE_AGENT=<alias> [ZEROCLAW_SOCKET=<path>] \
//!     cargo test -p pipecrab-lm-zeroclaw --test live -- --ignored
//! ```
//!
//! The agent profile decides everything else (provider, tools, delegation);
//! this test only asserts the turn round-trips and yields spoken text.

use std::time::Duration;

use futures::StreamExt;
use pipecrab_lm::{Conversation, GenParams, LanguageModel, Message, ModelDelta};
use pipecrab_lm_zeroclaw::{ZeroclawLmConfig, connect};

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires a running zeroclaw daemon; set ZEROCLAW_LIVE_AGENT"]
async fn one_real_turn_round_trips() {
    let agent = std::env::var("ZEROCLAW_LIVE_AGENT")
        .expect("set ZEROCLAW_LIVE_AGENT to a configured agent alias");
    let mut config = ZeroclawLmConfig::new(agent.as_str());
    if let Ok(socket) = std::env::var("ZEROCLAW_SOCKET") {
        config = config.with_socket_path(socket);
    }

    let (lm, _source) = connect(config).await.expect("connect to the live daemon");
    println!(
        "live: session {} in workspace {}",
        lm.session_id(),
        lm.workspace_dir().display()
    );

    let conversation = Conversation {
        messages: vec![Message::user(
            "Reply with one short sentence: what model are you running on?",
        )],
    };
    let mut stream = lm
        .generate(&conversation, &GenParams::default(), &[])
        .await
        .expect("generate");

    let mut text = String::new();
    loop {
        match tokio::time::timeout(Duration::from_secs(120), stream.next()).await {
            Ok(Some(Ok(ModelDelta::Text(delta)))) => text.push_str(&delta),
            Ok(Some(Ok(ModelDelta::ToolCall(call)))) => {
                println!("live: tool call {} {}", call.name, call.arguments_json);
            }
            Ok(Some(Err(error))) => panic!("turn failed: {error}"),
            Ok(None) => break,
            Err(_) => panic!("timed out waiting for the turn; partial: {text:?}"),
        }
    }
    println!("live: agent said {text:?}");
    assert!(
        !text.trim().is_empty(),
        "a live turn must produce spoken text"
    );
}
