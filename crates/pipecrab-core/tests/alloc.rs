//! Allocation audits: deterministic counts asserted as hard limits.
//! A regression that adds an allocation to a hot path fails the build.

use pipecrab_core::{Direction, Frame, Processor};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::sync::Arc;

// Per-thread counter, so audits stay correct even when tests run in parallel.
// const-init TLS is allocation-free, so reading it inside `alloc` can't recurse.
thread_local! {
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
}

struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.with(|c| c.set(c.get() + 1));
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
}
#[global_allocator]
static GA: Counting = Counting;

/// Allocations that happen on this thread while `f` runs.
fn allocs(f: impl FnOnce()) -> u64 {
    let start = ALLOCS.with(Cell::get);
    f();
    ALLOCS.with(Cell::get) - start
}

// --- a tiny processor to exercise the hot path ---
enum Cmd {
    Speak(Arc<str>),
    Forward(Direction, Frame),
}
#[derive(Default)]
struct Say;
impl Processor for Say {
    type Effect = Cmd;
    fn decide(&mut self, dir: Direction, f: &Frame) -> Vec<Cmd> {
        match (dir, f) {
            (_, Frame::Interrupt) => vec![],
            (Direction::Down, Frame::Transcript(t)) => vec![Cmd::Speak(t.clone())],
            (d, f) => vec![Cmd::Forward(d, f.clone())],
        }
    }
}

#[test]
fn interrupt_path_allocates_nothing() {
    let mut s = Say;
    let frame = Frame::Interrupt;
    let n = allocs(|| {
        black_box(s.decide(black_box(Direction::Down), black_box(&frame)));
    });
    assert_eq!(n, 0, "the interrupt/reset path must not allocate");
}

#[test]
fn transcript_path_within_budget() {
    let mut s = Say;
    let frame = Frame::Transcript("hello".into());
    let n = allocs(|| {
        black_box(s.decide(black_box(Direction::Down), black_box(&frame)));
    });
    // Was 2 (Vec + String copy). Arc<str> made the payload clone free.
    assert_eq!(n, 1, "Transcript decide should allocate exactly 1 (the Vec), got {n}");
}