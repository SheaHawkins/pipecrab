//! Tests for the priority mailbox.
//!
//! Contract: FIFO within a lane; `sys` preempts a backed-up `data` lane; the
//! sys direction tag is carried through untouched; data lane always yields
//! `Received::Data`. The preemption is exercised in both directions a system
//! frame travels — an `Interrupt` going down and an `Error` going up — since
//! fast upstream failure depends on the error jumping the data backlog.
//!
//! Lane-close semantics (one lane closed, both closed) live as unit tests in
//! `src/inbound.rs`: a single `Outbound` owns both senders, so `link` cannot
//! close one lane on its own.

use futures::executor::block_on;
use pipecrab_core::{DataFrame, Direction, SystemFrame, Transcript};
use pipecrab_runtime::{Received, link};

#[test]
fn interrupt_preempts_backed_up_data() {
    block_on(async {
        let (out, mut inb) = link(16);
        for i in 0..8 {
            out.send_data(Transcript::user_final(i.to_string()).into())
                .await
                .unwrap();
        }
        out.send_system(Direction::Down, SystemFrame::Interrupt)
            .await
            .unwrap();

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
        let (out, mut inb) = link(16);
        for i in 0..8 {
            out.send_data(Transcript::user_final(i.to_string()).into())
                .await
                .unwrap();
        }
        out.send_system(
            Direction::Up,
            SystemFrame::Error {
                message: "inference exploded".into(),
                fatal: true,
            },
        )
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
        let (out, mut inb) = link(16);
        for i in 0..4 {
            out.send_data(Transcript::user_final(i.to_string()).into())
                .await
                .unwrap();
        }
        for i in 0..4 {
            match inb.recv().await.unwrap() {
                Received::Data(DataFrame::Transcript(s)) => {
                    assert_eq!(s.text, i.to_string().into())
                }
                other => panic!("expected Data(Transcript({i})), got {other:?}"),
            }
        }
    });
}

#[test]
fn data_lane_is_always_downstream() {
    block_on(async {
        let (out, mut inb) = link(16);
        out.send_data(Transcript::user_final("a").into())
            .await
            .unwrap();
        out.send_data(Transcript::user_final("b").into())
            .await
            .unwrap();
        assert!(matches!(inb.recv().await.unwrap(), Received::Data(_)));
        assert!(matches!(inb.recv().await.unwrap(), Received::Data(_)));
    });
}

#[test]
fn dropping_the_outbound_closes_both_lanes() {
    block_on(async {
        let (out, mut inb) = link(16);
        out.send_data(Transcript::user_final("buffered").into())
            .await
            .unwrap();
        drop(out);

        // Buffered frames are still served after the sender side is gone…
        match inb.recv().await.unwrap() {
            Received::Data(DataFrame::Transcript(s)) => {
                assert_eq!(s.text, "buffered".into())
            }
            other => panic!("buffered frame must survive sender drop, got {other:?}"),
        }
        // …and only then does recv signal shutdown.
        assert!(
            inb.recv().await.is_none(),
            "closed lanes must signal shutdown via None"
        );
    });
}
