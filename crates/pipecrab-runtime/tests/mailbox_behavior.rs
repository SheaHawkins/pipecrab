//! Tests for the priority mailbox.
//!
//! Contract: FIFO within a lane; `sys` preempts a backed-up `data` lane; the
//! sys direction tag is carried through untouched; data lane always yields
//! `Received::Data`; both lanes closed => `None`. The preemption is exercised
//! in both directions a system frame travels — an `Interrupt` going down and an
//! `Error` going up — since fast upstream failure depends on the error jumping
//! the data backlog.

use pipecrab_core::{DataFrame, Direction, SystemFrame};
use pipecrab_runtime::{Inbound, Received};
use tokio::sync::mpsc;

fn lanes() -> (mpsc::Sender<(Direction, SystemFrame)>, mpsc::Sender<DataFrame>, Inbound) {
    let (sys_tx, sys) = mpsc::channel(16);
    let (data_tx, data) = mpsc::channel(16);
    (sys_tx, data_tx, Inbound { sys, data })
}

#[tokio::test]
async fn interrupt_preempts_backed_up_data() {
    let (sys_tx, data_tx, mut inb) = lanes();
    for i in 0..8 {
        data_tx.send(DataFrame::Transcript(i.to_string().into())).await.unwrap();
    }
    sys_tx.send((Direction::Down, SystemFrame::Interrupt)).await.unwrap();

    let r = inb.recv().await.unwrap();
    assert!(
        matches!(r, Received::Sys(Direction::Down, SystemFrame::Interrupt)),
        "interrupt must jump the backlog, got {r:?}",
    );
}

#[tokio::test]
async fn fatal_error_propagates_upstream_ahead_of_data() {
    let (sys_tx, data_tx, mut inb) = lanes();
    for i in 0..8 {
        data_tx.send(DataFrame::Transcript(i.to_string().into())).await.unwrap();
    }
    sys_tx
        .send((Direction::Up, SystemFrame::Error { message: "inference exploded".into(), fatal: true }))
        .await
        .unwrap();

    match inb.recv().await.unwrap() {
        Received::Sys(Direction::Up, SystemFrame::Error { message, .. }) => {
            assert_eq!(message, "inference exploded".into());
        }
        other => panic!("expected Sys(Up, Error), got {other:?}"),
    }
}

#[tokio::test]
async fn data_lane_is_fifo() {
    let (_sys_tx, data_tx, mut inb) = lanes();
    for i in 0..4 {
        data_tx.send(DataFrame::Transcript(i.to_string().into())).await.unwrap();
    }
    for i in 0..4 {
        match inb.recv().await.unwrap() {
            Received::Data(DataFrame::Transcript(s)) => assert_eq!(s, i.to_string().into()),
            other => panic!("expected Data(Transcript({i})), got {other:?}"),
        }
    }
}

#[tokio::test]
async fn data_lane_is_always_downstream() {
    let (_sys_tx, data_tx, mut inb) = lanes();
    data_tx.send(DataFrame::Transcript("a".into())).await.unwrap();
    data_tx.send(DataFrame::Transcript("b".into())).await.unwrap();
    assert!(matches!(inb.recv().await.unwrap(), Received::Data(_)));
    assert!(matches!(inb.recv().await.unwrap(), Received::Data(_)));
}

#[tokio::test]
async fn both_lanes_closed_yields_none() {
    let (sys_tx, data_tx, mut inb) = lanes();
    drop(sys_tx);
    drop(data_tx);
    assert!(inb.recv().await.is_none(), "closed lanes must signal shutdown via None");
}
