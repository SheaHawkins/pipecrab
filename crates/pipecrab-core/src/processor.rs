use crate::frame::{DataFrame, Direction, SystemFrame};

/// What happens to the frame that [`Processor::decide_data`] or
/// [`Processor::decide_system`] just received.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Disposition {
    /// Pass the input onward — data downstream, system in its travel direction.
    Forward,
    /// Consume the input; it does not propagate.
    Drop,
}

/// Outcome of [`Processor::decide_data`] / [`Processor::decide_system`]:
/// a disposition for the input frame plus zero or more effects to emit.
///
/// # Four forms
///
/// ```
/// use pipecrab_core::{Decision, Disposition};
///
/// // 1. Pass-through: forward the input, emit nothing.
/// let d: Decision<u32> = Decision::forward();
/// assert_eq!(d.disposition, Disposition::Forward);
/// assert!(d.effects.is_empty());
///
/// // 2. Transform (drop-on-emit): consume the input, emit a replacement.
/// let d: Decision<u32> = Decision::drop().emit(42);
/// assert_eq!(d.disposition, Disposition::Drop);
/// assert_eq!(d.effects, [42]);
///
/// // 3. Observe-and-pass: forward the input *and* emit a derived value.
/// let d: Decision<u32> = Decision::forward().emit(42);
/// assert_eq!(d.disposition, Disposition::Forward);
/// assert_eq!(d.effects, [42]);
///
/// // 4. Silent drop: consume the input, emit nothing.
/// let d: Decision<u32> = Decision::drop();
/// assert_eq!(d.disposition, Disposition::Drop);
/// assert!(d.effects.is_empty());
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decision<E> {
    /// What happens to the input frame.
    pub disposition: Disposition,
    /// Effects to perform, in order, after the disposition is acted on.
    pub effects: Vec<E>,
}

impl<E> Decision<E> {
    /// Forward the input, emit nothing. Allocates nothing.
    pub fn forward() -> Self {
        Self { disposition: Disposition::Forward, effects: Vec::new() }
    }

    /// Drop the input, emit nothing. Allocates nothing.
    pub fn drop() -> Self {
        Self { disposition: Disposition::Drop, effects: Vec::new() }
    }

    /// Append an effect, leaving the disposition unchanged. Chainable.
    pub fn emit(mut self, effect: E) -> Self {
        self.effects.push(effect);
        self
    }
}

impl<E> Default for Decision<E> {
    fn default() -> Self {
        Self::forward()
    }
}

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
/// # Control calls
///
/// "No I/O" has one carve-out. `decide_*` may issue *control calls*:
/// synchronous, non-blocking, idempotent, infallible operations on owned
/// engines. The canonical example is `cancel()`, which flips an atomic flag an
/// engine's worker observes; it cannot block, fail, allocate unboundedly, or
/// tear state, so invoking it from the synchronous, `&mut self` decide step is
/// sound. Anything that can block, fail, allocate unboundedly, or perform real
/// I/O is *not* a control call — it remains an [`Effect`](Processor::Effect)
/// for `perform` to carry out.
///
/// # Defaults
///
/// Both methods default to [`Decision::forward()`]: an un-overridden stage is a
/// transparent pass-through. Override only the variants you care about.
pub trait Processor {
    /// The command vocabulary this stage can emit; plain data, no I/O.
    type Effect;

    /// Maps an incoming data frame to a disposition and zero or more effects.
    ///
    /// Called for every [`DataFrame`] arriving on the data lane. The default
    /// forwards the frame unchanged.
    fn decide_data(&mut self, _frame: &DataFrame) -> Decision<Self::Effect> {
        Decision::forward()
    }

    /// Maps an incoming system frame to a disposition and zero or more effects.
    ///
    /// Called for every [`SystemFrame`] arriving on the system lane, along with
    /// the [`Direction`] it is travelling. The default forwards the frame.
    fn decide_system(&mut self, _dir: Direction, _frame: &SystemFrame) -> Decision<Self::Effect> {
        Decision::forward()
    }
}
