//! Tests for Component 1, the priority mailbox.
//!
//! Contract: FIFO within a lane; `sys` preempts a backed-up `data` lane; the
//! direction tag is carried through untouched; both lanes closed => `None`. The
//! preemption is exercised in both directions a system frame travels — an
//! `Interrupt` going down and an `Error` going up — since fast upstream failure
//! depends on the error jumping the data backlog.

use pipecrab_core::{Direction, Frame};
use pipecrab_runtime::Inbound;
use tokio::sync::mpsc;

type Sender = mpsc::Sender<(Direction, Frame)>;

fn lanes() -> (Sender, Sender, Inbound) {
    let (sys_tx, sys) = mpsc::channel(16);
    let (data_tx, data) = mpsc::channel(16);
    (sys_tx, data_tx, Inbound { sys, data })
}

#[tokio::test]
async fn interrupt_preempts_backed_up_data() {
    let (sys_tx, data_tx, mut inb) = lanes();
    for i in 0..8 {
        data_tx.send((Direction::Down, Frame::Transcript(i.to_string().into()))).await.unwrap();
    }
    sys_tx.send((Direction::Down, Frame::Interrupt)).await.unwrap();

    let (dir, frame) = inb.recv().await.unwrap();
    assert!(matches!(frame, Frame::Interrupt), "interrupt must jump the backlog, got {frame:?}");
    assert_eq!(dir, Direction::Down);
}

#[tokio::test]
async fn fatal_error_propagates_upstream_ahead_of_data() {
    let (sys_tx, data_tx, mut inb) = lanes();
    for i in 0..8 {
        data_tx.send((Direction::Down, Frame::Transcript(i.to_string().into()))).await.unwrap();
    }
    sys_tx.send((Direction::Up, Frame::Error("inference exploded".into()))).await.unwrap();

    match inb.recv().await.unwrap() {
        (Direction::Up, Frame::Error(msg)) => assert_eq!(msg, "inference exploded".into()),
        other => panic!("expected (Up, Error), got {other:?}"),
    }
}

#[tokio::test]
async fn data_lane_is_fifo() {
    let (_sys_tx, data_tx, mut inb) = lanes();
    for i in 0..4 {
        data_tx.send((Direction::Down, Frame::Transcript(i.to_string().into()))).await.unwrap();
    }
    for i in 0..4 {
        match inb.recv().await.unwrap() {
            (Direction::Down, Frame::Transcript(s)) => assert_eq!(s, i.to_string().into()),
            other => panic!("expected Transcript({i}) Down, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn direction_tag_is_carried_through() {
    let (_sys_tx, data_tx, mut inb) = lanes();
    data_tx.send((Direction::Up, Frame::Transcript("u".into()))).await.unwrap();
    data_tx.send((Direction::Down, Frame::Transcript("d".into()))).await.unwrap();
    assert_eq!(inb.recv().await.unwrap().0, Direction::Up);
    assert_eq!(inb.recv().await.unwrap().0, Direction::Down);
}

#[tokio::test]
async fn both_lanes_closed_yields_none() {
    let (sys_tx, data_tx, mut inb) = lanes();
    drop(sys_tx);
    drop(data_tx);
    assert!(inb.recv().await.is_none(), "closed lanes must signal shutdown via None");
}
