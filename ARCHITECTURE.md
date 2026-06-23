We differ from pipecat in a few ways:

1. Pipecat Processor = Pipecrab Stage
1. The pipeline is split into two lanes: Data and Sys (high-priority). Messages in the sys queue are drained first and can interrupt work in progress, as well as travel to stage upstream (such as for configuration requests). The data lane is one-directional and flows downstream only.
1. A Stage can manage internal state via the uninterruptable synchronous `decide` function. By contrast, the async `perform` function can be interrupted but cannot modify state. This prevents broken state 
1. SystemFrames are distinct from DataFrames. In pipecrab, they don't share an inheritance tree so you can't accidentally push SystemFrames into the downstream-only data lane. 
1. `decide` returns a `Decision` (disposition + effects) instead of a bare effect list. `Decision::forward()` is the default, so an un-overridden stage is a transparent pass-through and you never have to re-push frames you don't touch. `Decision::drop().emit(x)` transforms a frame; `Decision::forward().emit(x)` taps it without consuming it. See the `Decision` rustdoc for all four forms.