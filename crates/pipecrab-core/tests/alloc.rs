//! Allocation audits: deterministic counts asserted as hard limits.
//! A regression that adds an allocation to a hot path fails the build.

use pipecrab_core::{DataFrame, Direction, Processor, SystemFrame};
use pipecrab_test_util::allocs;
use std::hint::black_box;

// --- a tiny processor to exercise the hot path ---
enum Cmd { Speak, Forward }
#[derive(Default)]
struct Say;
impl Processor for Say {
    type Effect = Cmd;
    fn decide_data(&mut self, f: &DataFrame) -> Vec<Cmd> {
        match f {
            DataFrame::Transcript(_) => vec![Cmd::Speak],
            _ => vec![Cmd::Forward],
        }
    }
    fn decide_system(&mut self, _dir: Direction, f: &SystemFrame) -> Vec<Cmd> {
        match f {
            SystemFrame::Interrupt => vec![],
            _ => vec![Cmd::Forward],
        }
    }
}

#[test]
fn interrupt_path_allocates_nothing() {
    let mut s = Say;
    let frame = SystemFrame::Interrupt;
    let n = allocs(|| {
        black_box(s.decide_system(black_box(Direction::Down), black_box(&frame)));
    });
    assert_eq!(n, 0, "the interrupt/reset path must not allocate");
}

#[test]
fn transcript_path_within_budget() {
    let mut s = Say;
    let frame = DataFrame::Transcript("hello".into());
    let n = allocs(|| {
        black_box(s.decide_data(black_box(&frame)));
    });
    // Was 2 (Vec + String copy). Arc<str> made the payload clone free.
    assert_eq!(n, 1, "Transcript decide should allocate exactly 1 (the Vec), got {n}");
}
