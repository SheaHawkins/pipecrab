use crate::Frame;
use crate::frame::Direction;
/// The pure core of a stage.
///
/// `decide` maps an incoming `(direction, frame)` to a list of commands. It is
/// synchronous and total — no `.await`, no I/O — so it cannot be cancelled
/// mid-step. It also sees system frames (notably `Interrupt`), which is how a
/// stage resets its own state when the user barges in.
///
/// `Effect` is the stage's OWN command vocabulary: plain data, defined here in
/// the core. `decide` emits commands describing *what should happen*; the
/// runtime's `perform` interprets them, does the I/O, and mints the output
/// frames (e.g. the audio a TTS call returns).
pub trait Processor {
    /// The command vocabulary this stage can emit; plain data, no I/O.
    type Effect;
    /// Maps an incoming frame to zero or more effects describing what should happen next.
    fn decide(&mut self, dir: Direction, frame: &Frame) -> Vec<Self::Effect>;
}
