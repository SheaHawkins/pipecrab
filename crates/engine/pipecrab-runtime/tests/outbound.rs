use futures::executor::block_on;
use pipecrab_core::{DataFrame, Direction, SystemFrame, Transcript};
use pipecrab_runtime::{Received, link};

#[test]
fn send_data_delivers_frame() {
    block_on(async {
        let (outb, mut inb) = link(8);

        outb.send_data(Transcript::user_final("hello").into())
            .await
            .unwrap();

        match inb.recv().await.unwrap() {
            Received::Data(DataFrame::Transcript(s)) => assert_eq!(s.text, "hello".into()),
            other => panic!("unexpected {other:?}"),
        }
    });
}

#[test]
fn send_system_preserves_direction() {
    block_on(async {
        let (outb, mut inb) = link(8);

        outb.send_system(
            Direction::Up,
            SystemFrame::Error {
                message: "boom".into(),
                fatal: false,
            },
        )
        .await
        .unwrap();

        match inb.recv().await.unwrap() {
            Received::Sys(Direction::Up, SystemFrame::Error { message, .. }) => {
                assert_eq!(message, "boom".into())
            }
            other => panic!("unexpected {other:?}"),
        }
    });
}

#[test]
fn send_data_to_closed_channel_returns_err() {
    block_on(async {
        let (outb, inb) = link(8);
        drop(inb);

        assert!(
            outb.send_data(Transcript::user_final("x").into())
                .await
                .is_err()
        );
    });
}

#[test]
fn send_system_to_closed_channel_returns_err() {
    block_on(async {
        let (outb, inb) = link(8);
        drop(inb);

        assert!(
            outb.send_system(Direction::Down, SystemFrame::Stop)
                .await
                .is_err()
        );
    });
}
