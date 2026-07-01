//! Tests for Inbound::flush_data: selective interrupt-flush of the data lane.

use std::sync::Arc;

use futures::channel::mpsc;
use pipecrab_core::{AudioChunk, AudioFormat, DataFrame, Direction, SystemFrame};
use pipecrab_runtime::Inbound;

fn lanes() -> (mpsc::Sender<(Direction, SystemFrame)>, mpsc::Sender<DataFrame>, Inbound) {
    let (sys_tx, sys) = mpsc::channel(16);
    let (data_tx, data) = mpsc::channel(16);
    (sys_tx, data_tx, Inbound { sys, data })
}

fn input_audio() -> DataFrame {
    DataFrame::InputAudio { bytes: Arc::from(&[0u8; 4][..]), sample_rate: 16_000, num_channels: 1 }
}

fn audio() -> DataFrame {
    DataFrame::Audio(AudioChunk::new(Arc::from(&[0.0f32][..]), AudioFormat::new(48_000, 1)))
}

#[test]
fn flush_selective_drops_unmarked_keeps_input_audio_in_order() {
    let (_, mut data_tx, mut inb) = lanes();
    data_tx.try_send(DataFrame::Transcript("A".into())).unwrap();
    data_tx.try_send(input_audio()).unwrap(); // IN1
    data_tx.try_send(audio()).unwrap(); // B
    data_tx.try_send(input_audio()).unwrap(); // IN2

    let kept = inb.flush_data();
    assert_eq!(kept.len(), 2);
    assert!(matches!(kept[0], DataFrame::InputAudio { .. }));
    assert!(matches!(kept[1], DataFrame::InputAudio { .. }));
}

#[test]
fn flush_empty_lane_returns_empty() {
    let (_, _, mut inb) = lanes();
    assert!(inb.flush_data().is_empty());
}

#[test]
fn flush_all_unmarked_returns_empty() {
    let (_, mut data_tx, mut inb) = lanes();
    data_tx.try_send(DataFrame::Transcript("x".into())).unwrap();
    data_tx.try_send(DataFrame::Transcript("y".into())).unwrap();
    assert!(inb.flush_data().is_empty());
}

#[test]
fn flush_all_marked_returns_all_in_order() {
    let (_, mut data_tx, mut inb) = lanes();
    data_tx.try_send(input_audio()).unwrap();
    data_tx.try_send(input_audio()).unwrap();
    data_tx.try_send(input_audio()).unwrap();
    let kept = inb.flush_data();
    assert_eq!(kept.len(), 3);
    assert!(kept.iter().all(|f| matches!(f, DataFrame::InputAudio { .. })));
}

#[test]
fn flush_does_not_touch_sys_lane() {
    let (mut sys_tx, mut data_tx, mut inb) = lanes();
    sys_tx.try_send((Direction::Down, SystemFrame::Interrupt)).unwrap();
    data_tx.try_send(DataFrame::Transcript("drop me".into())).unwrap();

    let kept = inb.flush_data();
    assert!(kept.is_empty());
    // `futures`' Receiver has no `len()`; prove the lane is untouched by
    // pulling the frame back out — it must still be the buffered Interrupt.
    assert!(
        matches!(inb.sys.try_recv(), Ok((Direction::Down, SystemFrame::Interrupt))),
        "sys lane must be untouched by flush_data",
    );
}
