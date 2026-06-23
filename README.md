# PipeCrab
🦀 

## Writing a stage

A stage implements `Processor`. Both `decide_data` and `decide_system` return a `Decision` — which answers two questions at once: *does the incoming frame keep moving downstream?* and *what should this stage emit?*

| You return | Input frame | Emits |
|---|---|---|
| `Decision::forward()` | forwarded downstream | nothing |
| `Decision::drop()` | consumed | nothing |
| `Decision::drop().emit(x)` | consumed | `x` |
| `Decision::forward().emit(x)` | forwarded downstream | `x` |

**Transform** (e.g. STT, redactor): `drop().emit(output)` — the input never reaches downstream, only the replacement does.

**Tap** (e.g. VAD, logger): `forward().emit(derived)` — the original frame passes through and is followed by the derived one.

**Pass-through**: don't override `decide_data` / `decide_system` — the default is `Decision::forward()`, so every frame on an ignored lane flows on unchanged.

```rust
fn decide_data(&mut self, frame: &DataFrame) -> Decision<Self::Effect> {
    match frame {
        DataFrame::Audio(a) => Decision::drop().emit(Effect::Transcript(self.stt(a))),
        _ => Decision::forward(),
    }
}
```
