use crate::frame::{DataFrame, Direction, SystemFrame};

/// The pure core of a stage.
///
/// Both methods are synchronous and total — no `.await`, no I/O — so they
/// cannot be cancelled mid-step. All state mutation happens here; `perform`
/// (in the runtime) is `&self` so a dropped future can't tear state.
///
/// `Effect` is the stage's own command vocabulary: plain data defined in core.
/// The methods emit commands describing *what should happen*; the runtime's
/// `perform` interprets them and does the I/O.
///
/// # Defaults
///
/// Both methods default to returning an empty `Vec`, which the run loop treats
/// as "pass this frame on unchanged." Override only the variants you care about.
pub trait Processor {
    /// The command vocabulary this stage can emit; plain data, no I/O.
    type Effect;

    /// Maps an incoming data frame to zero or more effects.
    ///
    /// Called for every [`DataFrame`] arriving on the data lane. The default
    /// passes the frame through (returns an empty `Vec`).
    fn decide_data(&mut self, _frame: &DataFrame) -> Vec<Self::Effect> {
        vec![]
    }

    /// Maps an incoming system frame to zero or more effects.
    ///
    /// Called for every [`SystemFrame`] arriving on the system lane, along with
    /// the [`Direction`] it is travelling. The default passes the frame through.
    fn decide_system(&mut self, _dir: Direction, _frame: &SystemFrame) -> Vec<Self::Effect> {
        vec![]
    }
}
