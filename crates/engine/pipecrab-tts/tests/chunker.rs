//! `SentenceChunker` splits a streaming agent generation into one final agent
//! transcript per sentence, flushes the trailing remainder on the generation's
//! own final, forwards non-agent frames, and resets on barge-in.
//!
//! Deterministic and tokio-free (`block_on`), so it rides the default
//! `cargo test --workspace` path.

use futures::executor::block_on;
use pipecrab_core::{DataFrame, Direction, Finality, Role, SystemFrame, Transcript};
use pipecrab_runtime::{PipelineBuilder, Received};
use pipecrab_tts::SentenceChunker;

/// Collect the text of every agent-final transcript the chunker emits for a
/// scripted sequence of input frames, closing the pipeline afterward.
async fn run(inputs: Vec<DataFrame>) -> Vec<String> {
    let (ends, driver) = PipelineBuilder::new().stage(SentenceChunker::new()).build().start();
    let input = ends.input;
    let mut output = ends.output;

    let feed = async move {
        let _ = input.send_system(Direction::Down, SystemFrame::Start).await;
        for frame in inputs {
            let _ = input.send_data(frame).await;
        }
        // Returning drops `input`, cascading shutdown.
    };

    let drain = async move {
        let mut sentences = Vec::new();
        while let Some(received) = output.recv().await {
            if let Received::Data(DataFrame::Transcript(t)) = received {
                assert_eq!(t.role, Role::Agent, "the chunker only emits agent speech");
                assert_eq!(t.finality, Finality::Final, "each emitted sentence is final");
                sentences.push(t.text.to_string());
            }
        }
        sentences
    };

    let (_, sentences, _) = futures::join!(feed, drain, driver);
    sentences
}

/// Append-only agent partials, each carrying the full text so far.
fn partial(text: &str) -> DataFrame {
    Transcript::agent_partial(text).into()
}

fn agent_final(text: &str) -> DataFrame {
    Transcript::agent_final(text).into()
}

#[test]
fn emits_a_sentence_as_soon_as_it_completes() {
    let sentences = block_on(run(vec![
        // The first sentence is only known complete once the whitespace after
        // "one." arrives — a partial ending exactly at "one." holds no boundary.
        partial("one."),
        partial("one. tw"),
        partial("one. two."),
        partial("one. two. three"),
        agent_final("one. two. three"),
    ]));
    assert_eq!(sentences, vec!["one.", "two.", "three"]);
}

#[test]
fn final_flushes_the_trailing_remainder_without_punctuation() {
    // A generation that ends mid-thought: the tail past the last boundary is
    // flushed as a final sentence even though it has no terminator.
    let sentences = block_on(run(vec![partial("Hello world. And"), agent_final("Hello world. And more")]));
    assert_eq!(sentences, vec!["Hello world.", "And more"]);
}

#[test]
fn does_not_split_mid_number_or_mid_token() {
    // "3." with no following whitespace is not a boundary; the whole thing is
    // one sentence flushed at the final.
    let sentences = block_on(run(vec![partial("pi is 3."), partial("pi is 3.14"), agent_final("pi is 3.14")]));
    assert_eq!(sentences, vec!["pi is 3.14"]);
}

#[test]
fn does_not_split_after_an_abbreviation() {
    // "Dr." must not end a sentence: the title and the name stay together, and
    // the real boundary only lands after "today".
    let sentences = block_on(run(vec![
        partial("Dr. Smith called."),
        partial("Dr. Smith called. Come"),
        agent_final("Dr. Smith called. Come in today"),
    ]));
    assert_eq!(sentences, vec!["Dr. Smith called.", "Come in today"]);
}

#[test]
fn forwards_non_agent_frames_untouched() {
    // A user transcript is not the chunker's to touch: it forwards unchanged and
    // keeps its user/final identity.
    let (ends, driver) = PipelineBuilder::new().stage(SentenceChunker::new()).build().start();
    let input = ends.input;
    let mut output = ends.output;

    block_on(async {
        let feed = async move {
            let _ = input.send_data(Transcript::user_final("a question").into()).await;
        };
        let drain = async move {
            let mut got = None;
            while let Some(received) = output.recv().await {
                if let Received::Data(DataFrame::Transcript(t)) = received {
                    got = Some((t.role, t.text.to_string()));
                }
            }
            got
        };
        let (_, got, _) = futures::join!(feed, drain, driver);
        assert_eq!(got, Some((Role::User, "a question".to_string())));
    });
}

#[test]
fn interrupt_forwards_through_the_chunker() {
    // The chunker owns no engine, so an Interrupt just resets its offset and
    // forwards downstream — it must not be dropped or panic.
    let (ends, driver) = PipelineBuilder::new().stage(SentenceChunker::new()).build().start();
    let input = ends.input;
    let mut output = ends.output;

    block_on(async {
        let feed = async move {
            let _ = input.send_system(Direction::Down, SystemFrame::Interrupt).await;
        };
        let drain = async move {
            let mut saw_interrupt = false;
            while let Some(received) = output.recv().await {
                if let Received::Sys(Direction::Down, SystemFrame::Interrupt) = received {
                    saw_interrupt = true;
                }
            }
            saw_interrupt
        };
        let (_, saw_interrupt, _) = futures::join!(feed, drain, driver);
        assert!(saw_interrupt, "the chunker forwards the Interrupt downstream");
    });
}
