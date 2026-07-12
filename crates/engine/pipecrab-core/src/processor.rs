use crate::frame::{DataFrame, Direction, SystemFrame};

/// How a [`Processor`] handles its input frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Disposition {
    /// Pass the input onward.
    Forward,
    /// Consume the input; it does not propagate.
    Drop,
}

/// A frame [`Disposition`] and zero or more effects.
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
    /// Effects to perform in order.
    pub effects: Vec<E>,
}

impl<E> Decision<E> {
    /// Forwards the input without effects.
    pub fn forward() -> Self {
        Self {
            disposition: Disposition::Forward,
            effects: Vec::new(),
        }
    }

    /// Drops the input without effects.
    pub fn drop() -> Self {
        Self {
            disposition: Disposition::Drop,
            effects: Vec::new(),
        }
    }

    /// Appends an effect without changing the disposition.
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

/// The synchronous, stateful half of a stage.
///
/// The `decide_*` methods are sans-I/O: they synchronously update state and
/// describe external work as effects. Because they never await, a transition
/// cannot be cancelled halfway through. The runtime executes effects through a
/// shared reference, so cancelling that work cannot tear processor state.
///
/// [`Effect`](Processor::Effect) describes work for the runtime to perform.
///
/// # Control calls
///
/// `decide_*` may invoke synchronous, non-blocking, idempotent, infallible
/// control operations such as `cancel()`. Other work must be an effect.
///
/// # Defaults
///
/// Both methods default to [`Decision::forward()`].
pub trait Processor {
    /// The commands this processor emits.
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
