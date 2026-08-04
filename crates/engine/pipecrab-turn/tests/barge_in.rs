//! `BargeInStage` consumes the speech-onset edge and emits a downstream
//! Interrupt ahead of it; everything else forwards untouched.

use std::sync::Arc;

use futures::FutureExt;
use futures::executor::block_on;
use pipecrab_core::{
    AudioChunk, AudioFormat, DataFrame, Direction, Disposition, Processor, SystemFrame, Transcript,
};
use pipecrab_runtime::{Received, Stage, link};
use pipecrab_test_util::allocs;
use pipecrab_turn::BargeInStage;

#[test]
fn speech_started_emits_the_effect_and_consumes_the_edge() {
    let mut stage = BargeInStage::new();
    let decision = stage.decide_data(&DataFrame::SpeechStarted);
    assert_eq!(decision.disposition, Disposition::Drop);
    assert_eq!(decision.effects.len(), 1);
}

#[test]
fn every_other_frame_forwards() {
    let mut stage = BargeInStage::new();
    let audio = DataFrame::Audio(AudioChunk::new(
        Arc::from(&[0.0f32][..]),
        AudioFormat::new(16_000, 1),
    ));
    for frame in [
        audio,
        DataFrame::SpeechStopped,
        Transcript::user_final("hi").into(),
    ] {
        let decision = stage.decide_data(&frame);
        assert_eq!(decision.disposition, Disposition::Forward);
        assert!(decision.effects.is_empty());
    }
    let sys = stage.decide_system(Direction::Down, &SystemFrame::Interrupt);
    assert_eq!(sys.disposition, Disposition::Forward);
    assert!(sys.effects.is_empty());
}

#[test]
fn perform_sends_the_interrupt_before_the_edge() {
    block_on(async {
        let stage = BargeInStage::new();
        let (out, mut inb) = link(8);

        let mut decision = BargeInStage::new().decide_data(&DataFrame::SpeechStarted);
        let effect = decision.effects.pop().expect("one effect");
        stage.perform(effect, &out).await.unwrap();

        // The Interrupt arrives first (sys lane preempts)…
        match inb.recv().await {
            Some(Received::Sys(Direction::Down, SystemFrame::Interrupt)) => {}
            other => panic!("expected the Interrupt first, got {other:?}"),
        }
        // …and the re-emitted edge is stamped *after* it: the causal flush a
        // downstream stage runs on this Interrupt must keep the edge.
        let kept = inb.flush_data();
        assert!(
            matches!(kept.as_slice(), [DataFrame::SpeechStarted]),
            "the re-emitted edge must survive its own interrupt's flush, got {kept:?}",
        );
    });
}

#[test]
fn decide_data_on_speech_started_allocates_nothing() {
    let mut stage = BargeInStage::new();
    let frame = DataFrame::SpeechStarted;
    let n = allocs(|| {
        let decision = stage.decide_data(&frame);
        std::hint::black_box(&decision);
        // The effect is a ZST, so even the effects Vec never allocates.
        assert_eq!(decision.effects.len(), 1);
    });
    assert_eq!(n, 0, "the barge-in decide path must not allocate, got {n}");
}

#[test]
fn perform_survives_a_closed_link() {
    // Shutdown race: the downstream stage is gone. perform must not error the
    // pipeline over it — sends fail silently like the run loop's own forwards.
    block_on(async {
        let stage = BargeInStage::new();
        let (out, inb) = link(8);
        drop(inb);

        let mut decision = BargeInStage::new().decide_data(&DataFrame::SpeechStarted);
        let effect = decision.effects.pop().expect("one effect");
        assert!(stage.perform(effect, &out).now_or_never().unwrap().is_ok());
    });
}
