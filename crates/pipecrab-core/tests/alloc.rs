//! Allocation audits: deterministic counts asserted as hard limits.
//! A regression that adds an allocation to a hot path fails the build.

use pipecrab_core::{Direction, Frame, Processor};
use pipecrab_test_util::allocs;
use std::hint::black_box;
use std::sync::Arc;

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
