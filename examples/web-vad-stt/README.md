# Browser VAD + STT example

This example captures the microphone with the Web Audio API, resamples its audio
to 16 kHz mono, gates it on Silero voice activity running under
[onnxruntime-web], and transcribes each utterance with a
[Transformers.js] speech-to-text model. It prints speech edges and transcripts
into the page.

It is the same pipeline as [`stt-sherpa`](../stt-sherpa), stage for stage:

```rust
PipelineBuilder::new()
    .stage(ResamplerStage::new(ENGINE_FORMAT)?)      // capture rate → 16 kHz mono
    .stage(VadStage::new(Debounced::new(scorer)))    // gate: emit only utterances
    .stage(SttStage::new(Buffered::new(transcriber)))// Audio → Transcript
```

Only the backends differ — `WebAudioSource` instead of `CpalSource`,
`SileroScorer` instead of `SherpaVad`, `TransformersStt` instead of `SherpaStt`.
Each sits behind the same capability trait, so the stages above never learn which
one they got.

## Requirements

- Rust 1.88 or newer, with the `wasm32-unknown-unknown` target.
- `wasm-bindgen-cli`, matching the `wasm-bindgen` version in `Cargo.lock`.
- Python 3 (or any static file server) to serve the page.
- A browser with `AudioWorklet` and WebAssembly: current Chrome, Firefox, Safari,
  or Edge.
- A microphone, and a **secure context** to reach it — `http://localhost` counts,
  a plain-http LAN address does not.

```console
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.126
```

## Download the VAD model

The page fetches the Silero model from its own origin, so it has to be served
alongside it. From the repository root:

```console
mkdir -p examples/web-vad-stt/web/models

curl -L \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx \
  -o examples/web-vad-stt/web/models/silero_vad.onnx
```

The speech-to-text model needs no setup: Transformers.js fetches it from the
Hugging Face Hub on the first run and caches it in the browser.

## Build

```console
cargo build -p web-vad-stt --target wasm32-unknown-unknown --release

wasm-bindgen target/wasm32-unknown-unknown/release/web_vad_stt.wasm \
  --out-dir examples/web-vad-stt/web/pkg --target web
```

`wasm-bindgen` emits the JS bindings next to the `.wasm`, plus a `snippets/`
directory holding the browser-side glue each adapter crate ships — the capture
worklet, the Silero session, the Transformers.js pipeline.

## Run

```console
python3 -m http.server -d examples/web-vad-stt/web 8000
```

Open <http://localhost:8000>, press **Start listening**, and allow microphone
access. The first run downloads the speech-to-text model — tens of megabytes,
so give it a moment; the browser caches it afterwards. Then speak, pause, and
speak again. Expected output resembles:

```text
listening — capturing at 48000 Hz, processing at 16000 Hz mono
— started —
— stopped —
After early nightfall, the yellow lamps would light up here and there.
```

**Stop** releases the microphone and shuts the pipeline down; the indicator in
the browser's tab clears with it.

## Choosing a model

The **Speech-to-text model** field takes any Hugging Face id with an ONNX export
Transformers.js can run. Decoding the 6.6-second
[`test-resources`](../../test-resources/audio) fixture, single-threaded on wasm:

| Model | Load | Decode | Notes |
|---|---|---|---|
| `onnx-community/moonshine-tiny-ONNX` | ~2 s | ~0.3 s | The default: built for short utterances, and the quickest of the three by a wide margin. |
| `Xenova/whisper-tiny.en` | ~4 s | ~1.3 s | Whisper at its smallest. |
| `onnx-community/whisper-base.en` | ~4 s | ~3.9 s | The most accurate here, and the slowest. |

English-only models (`.en`) reject a language option; pass one through
`TransformersSttConfig::language` only for a multilingual model.

The page pins Transformers.js 3.8.1. Its 4.x default exports currently fail to
create a session under onnxruntime-web (`TransposeDQWeightsForMatMulNBits
Missing required scale`) unless every module's `dtype` is forced to `fp32`,
which costs both the download size and the speed the table above reports.

## Where the work happens

The pipeline itself is one `!Send` task driven by `spawn_local` — no executor is
baked into pipecrab, so the same stages that `block_on` natively just get driven
by the browser's event loop instead. Inference does not run on that task: both
engines are JS, and each `.await` hands control back to the page while
onnxruntime-web works. That is the browser's answer to native's `offload`, and it
is why the adapters take an already-loaded engine namespace rather than importing
one — where the engine runs, and on what backend, stays the page's call.

Everything runs single-threaded: multi-threaded ONNX needs `SharedArrayBuffer`,
which needs COOP/COEP response headers that `python3 -m http.server` does not
send. Serving the page cross-origin-isolated and raising
`ort.env.wasm.numThreads` in `web/main.js` is the upgrade path.

[onnxruntime-web]: https://onnxruntime.ai/docs/tutorials/web/
[Transformers.js]: https://huggingface.co/docs/transformers.js
