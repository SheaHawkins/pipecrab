use pipecrab_core::{DataFrame, Direction, SystemFrame};
use pipecrab_runtime::Outbound;
use tokio::sync::mpsc;

#[tokio::test]
async fn send_data_delivers_frame() {
    let (data_tx, mut data_rx) = mpsc::channel(8);
    let (sys_tx, _sys_rx) = mpsc::channel(8);
    let outb = Outbound { data: data_tx, sys: sys_tx };

    outb.send_data(DataFrame::Transcript("hello".into())).await.unwrap();

    match data_rx.recv().await.unwrap() {
        DataFrame::Transcript(s) => assert_eq!(s, "hello".into()),
        other => panic!("unexpected {other:?}"),
    }
}

#[tokio::test]
async fn send_system_preserves_direction() {
    let (data_tx, _data_rx) = mpsc::channel::<DataFrame>(8);
    let (sys_tx, mut sys_rx) = mpsc::channel(8);
    let outb = Outbound { data: data_tx, sys: sys_tx };

    outb.send_system(Direction::Up, SystemFrame::Error { message: "boom".into(), fatal: false })
        .await
        .unwrap();

    match sys_rx.recv().await.unwrap() {
        (Direction::Up, SystemFrame::Error { message, .. }) => assert_eq!(message, "boom".into()),
        other => panic!("unexpected {other:?}"),
    }
}

#[tokio::test]
async fn send_data_to_closed_channel_returns_err() {
    let (data_tx, data_rx) = mpsc::channel(8);
    let (sys_tx, _sys_rx) = mpsc::channel(8);
    let outb = Outbound { data: data_tx, sys: sys_tx };
    drop(data_rx);

    assert!(outb.send_data(DataFrame::Transcript("x".into())).await.is_err());
}

#[tokio::test]
async fn send_system_to_closed_channel_returns_err() {
    let (data_tx, _data_rx) = mpsc::channel::<DataFrame>(8);
    let (sys_tx, sys_rx) = mpsc::channel(8);
    let outb = Outbound { data: data_tx, sys: sys_tx };
    drop(sys_rx);

    assert!(outb.send_system(Direction::Down, SystemFrame::Stop).await.is_err());
}
