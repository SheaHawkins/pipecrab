//! Allocation audits: deterministic counts asserted as hard limits.
//! A regression that adds an allocation to a hot path fails the build.

use pipecrab_core::{DataFrame, Decision, Direction, Processor, SystemFrame};
use pipecrab_test_util::allocs;
use std::hint::black_box;

// --- a tiny processor to exercise the hot path ---
enum Cmd { Speak }
#[derive(Default)]
struct Say;
impl Processor for Say {
    type Effect = Cmd;
    fn decide_data(&mut self, f: &DataFrame) -> Decision<Cmd> {
        match f {
            DataFrame::Transcript(_) => Decision::drop().emit(Cmd::Speak),
            _ => Decision::forward(),
        }
    }
    fn decide_system(&mut self, _dir: Direction, f: &SystemFrame) -> Decision<Cmd> {
        match f {
            SystemFrame::Interrupt => Decision::drop(),
            _ => Decision::forward(),
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
    // Cmd is a ZST (single unit variant), so Vec push allocates nothing.
    assert_eq!(n, 0, "Transcript decide with a ZST effect should not allocate, got {n}");
}
