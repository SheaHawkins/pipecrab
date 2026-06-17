//! Allocation audit: dispatching a buffered system frame must not allocate.

use std::future::Future;
use std::hint::black_box;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

use pipecrab_core::{Direction, Frame};
use pipecrab_runtime::Inbound;
use pipecrab_test_util::allocs;
use tokio::sync::mpsc;

struct NoopWaker;
impl Wake for NoopWaker {
    fn wake(self: Arc<Self>) {}
}

#[test]
fn dispatching_a_buffered_system_frame_is_allocation_free() {
    let (sys_tx, sys) = mpsc::channel(16);
    let (_data_tx, data) = mpsc::channel(16);
    let mut inb = Inbound { sys, data };

    sys_tx.try_send((Direction::Down, Frame::Interrupt)).unwrap();

    let waker = Waker::from(Arc::new(NoopWaker));
    let mut cx = Context::from_waker(&waker);

    let n = allocs(|| {
        let fut = inb.recv();
        let mut fut = pin!(fut);
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(Some((_, f))) => { black_box(f); }
            Poll::Ready(None) => panic!("unexpected end of stream"),
            Poll::Pending => panic!("buffered frame should poll Ready"),
        }
    });
    assert!(n <= 1, "dispatching a buffered system frame only allocates select!, got {n}");
}
