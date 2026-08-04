//! Tests for Inbound::flush_data: causal interrupt-flush of the data lane.
//!
//! The flush floor is the stamp of the most recently received system frame:
//! frames queued before it are kept only if `survives_flush()`; frames queued
//! at or after it are always kept.

use std::sync::Arc;

use futures::FutureExt;
use pipecrab_core::{
    AudioChunk, AudioFormat, DataFrame, Direction, DispatchEvent, DispatchFrame, SystemFrame,
    Transcript,
};
use pipecrab_runtime::{Inbound, Outbound, Received, link};

/// A frame that survives a flush on its own: every dispatch frame is durable.
fn survivor(task_id: &str) -> DataFrame {
    DataFrame::Dispatch(DispatchFrame::from(DispatchEvent::Progress {
        task_id: Arc::from(task_id),
        message: Arc::from("keep"),
    }))
}

fn audio() -> DataFrame {
    DataFrame::Audio(AudioChunk::new(
        Arc::from(&[0.0f32][..]),
        AudioFormat::new(48_000, 1),
    ))
}

/// Capacity is ample in these tests, so sends resolve immediately.
fn send_data(out: &Outbound, frame: DataFrame) {
    out.send_data(frame)
        .now_or_never()
        .expect("send resolves immediately")
        .unwrap();
}

fn send_interrupt(out: &Outbound) {
    out.send_system(Direction::Down, SystemFrame::Interrupt)
        .now_or_never()
        .expect("send resolves immediately")
        .unwrap();
}

/// Receive the buffered Interrupt, moving the flush floor onto it.
fn recv_interrupt(inb: &mut Inbound) {
    match inb.recv().now_or_never() {
        Some(Some(Received::Sys(Direction::Down, SystemFrame::Interrupt))) => {}
        other => panic!("expected the buffered Interrupt, got {other:?}"),
    }
}

#[test]
fn flush_selective_drops_unmarked_keeps_survivors_in_order() {
    let (out, mut inb) = link(16);
    send_data(&out, Transcript::user_final("A").into());
    send_data(&out, survivor("S1"));
    send_data(&out, audio()); // B
    send_data(&out, survivor("S2"));
    send_interrupt(&out);
    recv_interrupt(&mut inb);

    let kept = inb.flush_data();
    assert_eq!(kept.len(), 2);
    assert!(matches!(kept[0], DataFrame::Dispatch(_)));
    assert!(matches!(kept[1], DataFrame::Dispatch(_)));
}

#[test]
fn flush_empty_lane_returns_empty() {
    let (out, mut inb) = link(16);
    send_interrupt(&out);
    recv_interrupt(&mut inb);
    assert!(inb.flush_data().is_empty());
}

#[test]
fn flush_all_unmarked_returns_empty() {
    let (out, mut inb) = link(16);
    send_data(&out, Transcript::user_final("x").into());
    send_data(&out, Transcript::user_final("y").into());
    send_interrupt(&out);
    recv_interrupt(&mut inb);
    assert!(inb.flush_data().is_empty());
}

#[test]
fn flush_all_marked_returns_all_in_order() {
    let (out, mut inb) = link(16);
    send_data(&out, survivor("S1"));
    send_data(&out, survivor("S2"));
    send_data(&out, survivor("S3"));
    send_interrupt(&out);
    recv_interrupt(&mut inb);
    let kept = inb.flush_data();
    assert_eq!(kept.len(), 3);
    assert!(kept.iter().all(|f| matches!(f, DataFrame::Dispatch(_))));
}

#[test]
fn flush_keeps_frames_queued_after_the_interrupt() {
    let (out, mut inb) = link(16);
    send_data(&out, Transcript::user_final("stale").into());
    send_interrupt(&out);
    // The barge-in utterance's own frames, queued behind the Interrupt: kept
    // even though a Transcript does not survive a flush on its own.
    send_data(&out, Transcript::user_final("fresh").into());
    recv_interrupt(&mut inb);

    let kept = inb.flush_data();
    assert_eq!(kept.len(), 1);
    match &kept[0] {
        DataFrame::Transcript(t) => assert_eq!(t.text, "fresh".into()),
        other => panic!("expected the post-interrupt transcript, got {other:?}"),
    }
}

#[test]
fn flush_does_not_touch_sys_lane() {
    let (out, mut inb) = link(16);
    send_data(&out, Transcript::user_final("drop me").into());
    send_interrupt(&out);
    recv_interrupt(&mut inb);
    // A second Interrupt buffered behind the one being flushed against.
    send_interrupt(&out);

    let kept = inb.flush_data();
    assert!(kept.is_empty());
    // `futures`' Receiver has no `len()`; prove the lane is untouched by
    // pulling the frame back out — it must still be the buffered Interrupt.
    match inb.recv().now_or_never() {
        Some(Some(Received::Sys(Direction::Down, SystemFrame::Interrupt))) => {}
        other => panic!("sys lane must be untouched by flush_data, got {other:?}"),
    }
}
