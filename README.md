```
██████  ██ ██████  ███████  ██████ ██████   █████  ██████  
██   ██ ██ ██   ██ ██      ██      ██   ██ ██   ██ ██   ██ 
██████  ██ ██████  █████   ██      ██████  ███████ ██████  
██      ██ ██      ██      ██      ██   ██ ██   ██ ██   ██ 
██      ██ ██      ███████  ██████ ██   ██ ██   ██ ██████                                     
```
Pipecrab is a thoughtful grounds-up rewrite of `pipecat` but in Rust. It is a cross-platform voice agent pipeline for low-latency local inference.

## Local Inference Runs On

| | VAD | STT | LM | TTS |
|---|---|---|---|---|
| macOS | ✅ | ✅ | ✅ | ✅ |
| iOS | ✅ | ✅ | ✅ | ❌ |
| Android | ✅ | ✅ | ✅ | ❌ |
| Linux | ❓ | ❓ | ❓ | ❌ |
| Windows | ❓ | ❓ | ❓ | ❌ |

❓ = expected to work, not yet verified. ❌ = not yet implemented.

## Running the examples

Six runnable examples live under [`examples/`](./examples), smallest first.
Each has its own README with full model-download and setup steps.

| Example | What it shows | Setup |
|---|---|---|
| [`echo`](./examples/echo) | Capture → playback: the shortest end-to-end path | none |
| [`vad-sherpa`](./examples/vad-sherpa) | Sherpa Silero VAD printing speech edges | 1 model file |
| [`stt-sherpa`](./examples/stt-sherpa) | VAD + streaming Zipformer transcription | VAD + ASR models |
| [`stt-sherpa-moonshine`](./examples/stt-sherpa-moonshine) | VAD + offline Moonshine v2 transcription | VAD + ASR models |
| [`lm-llamacpp`](./examples/lm-llamacpp) | VAD + STT + a local llama.cpp chat model streaming replies | VAD + ASR models + chat GGUF |
| [`e2e-voice-agent`](./examples/e2e-voice-agent) | The full loop: VAD + STT + LM + Kokoro TTS speaking replies | VAD + ASR models + chat GGUF + TTS model |

**Use headphones** 

### Transcription — `stt-sherpa` and `stt-sherpa-moonshine`

Both add an STT stage after the VAD gate: `stt-sherpa` uses a streaming
Zipformer, `stt-sherpa-moonshine` an offline Moonshine v2 model. They need
several model files — see each example's README
([`stt-sherpa`](./examples/stt-sherpa/README.md),
[`stt-sherpa-moonshine`](./examples/stt-sherpa-moonshine/README.md)) for the
download commands and the full flag list.

## Writing a pipeline

A pipeline is an ordered list of stages built with `PipelineBuilder`. Stages run
head-first in the order you add them, and each stage's emitted frames become the
next stage's input. `build().start()` wires the pipeline and hands back its two
ends plus a driver future.

```rust
use pipecrab::{DataFrame, Direction, PipelineBuilder, Received, SystemFrame};

let (ends, driver) = PipelineBuilder::new()
    .stage(ResamplerStage::new(SHERPA_FORMAT)?)  // capture rate → 16 kHz mono
    .stage(VadStage::with_config(detector, cfg)) // gate: emit only utterances
    .stage(SttStage::new(transcriber))           // Audio → Transcript
    .build()
    .start();
let input = ends.input;        // Outbound — feed the head
let mut output = ends.output;  // Inbound  — read past the tail
```

Send frames into `ends.input` and read results from `ends.output`. Open the run with a `Start`
system frame, then push data frames. Dropping `input` closes the head and
cascades a clean shutdown downstream.

```rust
let pump_in = async move {
    input.send_system(Direction::Down, SystemFrame::Start).await.ok();
    while let Ok(Some(chunk)) = source.next_chunk().await {
        if input.send_data(DataFrame::Audio(chunk)).await.is_err() {
            break; // downstream gone
        }
    }
    // `input` dropped here → the pipeline shuts down
};

let drain = async move {
    while let Some(received) = output.recv().await {
        if let Received::Data(DataFrame::Transcript(t)) = received {
            println!("{}", t.text);
        }
    }
};
```

Drive the driver and both pumps together on one thread — pipecrab bakes in no
executor, so the caller runs the future (`block_on` natively, `spawn_local` in the
browser):

```rust
block_on(async { futures::join!(driver, pump_in, drain) });
```

A `Pipeline` is itself a `Stage`, so a whole pipeline can be passed to `.stage(..)`
to nest it inside another, and `PipelineBuilder::capacity(n)` sets the per-lane
buffer depth (backpressure). See [`examples/stt-sherpa`](./examples/stt-sherpa)
for the full version of the pipeline above, and
[ARCHITECTURE.md](./ARCHITECTURE.md#writing-a-stage) for how to write the stages
that go in it.

## Contributing
See [CONTRIBUTING.md](./CONTRIBUTING.md)
