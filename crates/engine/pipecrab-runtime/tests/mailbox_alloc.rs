//! Allocation audit: dispatching a buffered system frame must not allocate.

use std::future::Future;
use std::hint::black_box;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

use futures::FutureExt;
use pipecrab_core::{Direction, SystemFrame};
use pipecrab_runtime::link;
use pipecrab_test_util::allocs;

struct NoopWaker;
impl Wake for NoopWaker {
    fn wake(self: Arc<Self>) {}
}

#[test]
fn dispatching_a_buffered_system_frame_is_allocation_free() {
    let (out, mut inb) = link(16);

    out.send_system(Direction::Down, SystemFrame::Interrupt)
        .now_or_never()
        .expect("send resolves immediately")
        .unwrap();

    let waker = Waker::from(Arc::new(NoopWaker));
    let mut cx = Context::from_waker(&waker);

    let n = allocs(|| {
        let fut = inb.recv();
        let mut fut = pin!(fut);
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(Some(r)) => {
                black_box(r);
            }
            Poll::Ready(None) => panic!("unexpected end of stream"),
            Poll::Pending => panic!("buffered frame should poll Ready"),
        }
    });
    assert!(
        n <= 1,
        "dispatching a buffered system frame only allocates select!, got {n}"
    );
}
