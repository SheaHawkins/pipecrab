//! Allocation audit: dispatching a buffered system frame must not allocate.

use std::future::Future;
use std::hint::black_box;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

use futures::channel::mpsc;
use pipecrab_core::{DataFrame, Direction, SystemFrame};
use pipecrab_runtime::Inbound;
use pipecrab_test_util::allocs;

struct NoopWaker;
impl Wake for NoopWaker {
    fn wake(self: Arc<Self>) {}
}

#[test]
fn dispatching_a_buffered_system_frame_is_allocation_free() {
    let (mut sys_tx, sys) = mpsc::channel(16);
    let (_data_tx, data) = mpsc::channel::<DataFrame>(16);
    let mut inb = Inbound { sys, data };

    sys_tx.try_send((Direction::Down, SystemFrame::Interrupt)).unwrap();

    let waker = Waker::from(Arc::new(NoopWaker));
    let mut cx = Context::from_waker(&waker);

    let n = allocs(|| {
        let fut = inb.recv();
        let mut fut = pin!(fut);
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(Some(r)) => { black_box(r); }
            Poll::Ready(None) => panic!("unexpected end of stream"),
            Poll::Pending => panic!("buffered frame should poll Ready"),
        }
    });
    assert!(n <= 1, "dispatching a buffered system frame only allocates select!, got {n}");
}
