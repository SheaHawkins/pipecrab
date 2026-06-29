//! Tests for the priority mailbox.
//!
//! Contract: FIFO within a lane; `sys` preempts a backed-up `data` lane; the
//! sys direction tag is carried through untouched; data lane always yields
//! `Received::Data`; both lanes closed => `None`. The preemption is exercised
//! in both directions a system frame travels — an `Interrupt` going down and an
//! `Error` going up — since fast upstream failure depends on the error jumping
//! the data backlog.

use futures::channel::mpsc;
use futures::executor::block_on;
use futures::sink::SinkExt;
use futures::FutureExt;
use pipecrab_core::{DataFrame, Direction, SystemFrame};
use pipecrab_runtime::{Inbound, Received};

fn lanes() -> (mpsc::Sender<(Direction, SystemFrame)>, mpsc::Sender<DataFrame>, Inbound) {
    let (sys_tx, sys) = mpsc::channel(16);
    let (data_tx, data) = mpsc::channel(16);
    (sys_tx, data_tx, Inbound { sys, data })
}

#[test]
fn interrupt_preempts_backed_up_data() {
    block_on(async {
        let (mut sys_tx, mut data_tx, mut inb) = lanes();
        for i in 0..8 {
            data_tx.send(DataFrame::Transcript(i.to_string().into())).await.unwrap();
        }
        sys_tx.send((Direction::Down, SystemFrame::Interrupt)).await.unwrap();

        let r = inb.recv().await.unwrap();
        assert!(
            matches!(r, Received::Sys(Direction::Down, SystemFrame::Interrupt)),
            "interrupt must jump the backlog, got {r:?}",
        );
    });
}

#[test]
fn fatal_error_propagates_upstream_ahead_of_data() {
    block_on(async {
        let (mut sys_tx, mut data_tx, mut inb) = lanes();
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
    });
}

#[test]
fn data_lane_is_fifo() {
    block_on(async {
        let (_sys_tx, mut data_tx, mut inb) = lanes();
        for i in 0..4 {
            data_tx.send(DataFrame::Transcript(i.to_string().into())).await.unwrap();
        }
        for i in 0..4 {
            match inb.recv().await.unwrap() {
                Received::Data(DataFrame::Transcript(s)) => assert_eq!(s, i.to_string().into()),
                other => panic!("expected Data(Transcript({i})), got {other:?}"),
            }
        }
    });
}

#[test]
fn data_lane_is_always_downstream() {
    block_on(async {
        let (_sys_tx, mut data_tx, mut inb) = lanes();
        data_tx.send(DataFrame::Transcript("a".into())).await.unwrap();
        data_tx.send(DataFrame::Transcript("b".into())).await.unwrap();
        assert!(matches!(inb.recv().await.unwrap(), Received::Data(_)));
        assert!(matches!(inb.recv().await.unwrap(), Received::Data(_)));
    });
}

#[test]
fn both_lanes_closed_yields_none() {
    block_on(async {
        let (sys_tx, data_tx, mut inb) = lanes();
        drop(sys_tx);
        drop(data_tx);
        assert!(inb.recv().await.is_none(), "closed lanes must signal shutdown via None");
    });
}

#[test]
fn one_closed_lane_does_not_signal_shutdown() {
    block_on(async {
        let (sys_tx, data_tx, mut inb) = lanes();
        // Data lane closes while sys is still open but empty.
        drop(data_tx);
        // recv must NOT resolve to None — a single closed lane is not shutdown.
        // `now_or_never` yields `None` when the future is still pending.
        assert!(
            inb.recv().now_or_never().is_none(),
            "a still-open sys lane must keep recv pending, not report shutdown",
        );

        // The other lane closing too is what finally yields `None`.
        drop(sys_tx);
        assert!(
            matches!(inb.recv().now_or_never(), Some(None)),
            "both lanes closed must resolve immediately to None",
        );
    });
}

#[test]
fn closed_sys_lane_still_serves_buffered_data() {
    block_on(async {
        let (sys_tx, mut data_tx, mut inb) = lanes();
        data_tx.send(DataFrame::Transcript("after sys closed".into())).await.unwrap();
        // Sys lane closes, but a buffered data frame must still be delivered.
        drop(sys_tx);

        match inb.recv().await.unwrap() {
            Received::Data(DataFrame::Transcript(s)) => assert_eq!(s, "after sys closed".into()),
            other => panic!("closed sys lane must not block the data lane, got {other:?}"),
        }
    });
}
