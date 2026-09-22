# Browser voice agent example

Talk to a fully local voice agent in a browser tab: Silero VAD brackets each
utterance from the microphone, a [Transformers.js] Moonshine model transcribes
it, a Transformers.js chat model streams a reply, and Kokoro speaks that reply
through [kokoro-js]. The network is used only to download the models.

It is the same pipeline as [`e2e-voice-agent`](../e2e-voice-agent), stage for
stage:

```rust
PipelineBuilder::new()
    .stage(ResamplerStage::new(ENGINE_FORMAT)?)       // capture rate → 16 kHz mono
    .stage(VadStage::with_config(Debounced::with_config(scorer, DEBOUNCE), gate))
    .stage(SttStage::new(Buffered::new(transcriber))) // Audio → Transcript
    .stage(BargeInStage::new())                       // speech onset ⇒ Interrupt
    .stage(UserTurnGate { on_event })                 // shows "You", drops empty finals
    .stage(LmStage::new(lm, SYSTEM_PROMPT))           // streams agent partials + a final
    .stage(SentenceChunker::new())                    // one final per sentence
    .stage(AgentEcho { on_event })                    // shows "Agent"
    .stage(TtsStage::new(tts))                        // 24 kHz audio per clause
    .stage(ResamplerStage::new(sink.format())?)       // 24 kHz → output rate
```

Only the backends differ — `WebAudioSource`/`WebAudioSink` instead of
`CpalSource`/`CpalSink`, `SileroScorer` instead of `SherpaVad`,
`TransformersStt` instead of `OfflineSherpaStt`, `TransformersLm` instead of
`LlamaCpp`, `KokoroTts` instead of `SherpaTts`. Each sits behind the same
capability trait, so the stages above never learn which one they got.

**Use headphones** — over speakers the microphone re-captures the agent's own
voice and it talks to itself. Barge-in is wired in: speak over the agent and the
reply stops, and your new utterance is answered instead.

## Requirements

- Rust 1.88 or newer, with the `wasm32-unknown-unknown` target.
- `wasm-bindgen-cli`, matching the `wasm-bindgen` version in `Cargo.lock`.
- Python 3 (or any static file server) to serve the page.
- A browser with `AudioWorklet` and WebAssembly: current Chrome, Firefox, Safari,
  or Edge. WebGPU is optional but makes replies several times faster.
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
mkdir -p examples/web-e2e/web/models

curl -L \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx \
  -o examples/web-e2e/web/models/silero_vad.onnx
```

The other three models need no setup: the engines fetch them from the Hugging
Face Hub on the first run and the browser caches them. That first run downloads
about 630 MB on WebGPU and 490 MB on WebAssembly.

## Build

```console
cargo build -p web-e2e --target wasm32-unknown-unknown --release

wasm-bindgen target/wasm32-unknown-unknown/release/web_e2e.wasm \
  --out-dir examples/web-e2e/web/pkg --target web
```

`wasm-bindgen` emits the JS bindings next to the `.wasm`, plus a `snippets/`
directory holding the browser-side glue each adapter crate ships — capture and
playback, the Silero session, the Transformers.js pipelines, the Kokoro model.

## Run

```console
python3 -m http.server -d examples/web-e2e/web 8000
```

Open <http://localhost:8000>, press **Start talking**, and allow microphone
access. The page runs the models on WebGPU when the browser has an adapter and
on WebAssembly otherwise; **Run on** switches before you start. Once the status
line reads `listening`, speak, pause, and listen. Expected output resembles:

```text
listening — capturing at 48000 Hz, speaking at 48000 Hz, Kokoro at 24000 Hz

YOU
After early nightfall the yellow lamps would light up here and there, the squalid quarter of the brothels.
AGENT
That sounds like a lively scene. I'd say it's a great example of the vibrant atmosphere of the brothels.
first audio 1.3 s after you stopped
```

…followed by the reply spoken aloud. The latency is measured from the moment
the VAD closes your turn — half a second of silence after your last word — to
the reply's first audio reaching the speakers. **Stop** releases the microphone,
silences the agent, and shuts the pipeline down.

## Choosing a model

The **Language model** field takes any Hugging Face id with an ONNX export and a
chat template that Transformers.js can run. Answering "What is a crab?" under
the example's system prompt, on an Apple M4 in Chrome 153:

| Model | Download (WebGPU / wasm) | WebGPU: first words / reply | wasm: first words / reply | Notes |
|---|---|---|---|---|
| `HuggingFaceTB/SmolLM2-360M-Instruct` | 273 / 365 MB | 0.2 s / 2.0 s | 0.8 s / 3.4 s | The default: two plain sentences, as the prompt asks. |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | 118 / 137 MB | 0.1 s / 2.5 s | 0.3 s / 2.6 s | Smallest; runs past the two-sentence ask and invents facts. |
| `onnx-community/Qwen2.5-0.5B-Instruct` | 483 / 512 MB | — | 1.1 s / 5.3 s | The desktop example's model. Its `q4f16` export repeats itself into garbage on WebGPU, so run it on WebAssembly. |

`web/main.js` picks each model's quantization per backend: `q4f16` for the LM
and `fp32` for Kokoro on WebGPU, `q8` for both on WebAssembly.

The **Voice** field takes any Kokoro v1.0 voice — `af_heart`, `af_bella`,
`am_adam`, `bf_emma`, `bm_george`, and more. A name the model does not have
fails the load with the full list.

The page pins Transformers.js 3.8.1 for the same reason `web-vad-stt` does, and
maps kokoro-js's own `@huggingface/transformers` import to that copy through the
import map in `web/index.html`, so all three Transformers.js models share one
ONNX runtime.

## Where the work happens

The pipeline is one `!Send` task driven by `spawn_local`, exactly as in
`web-vad-stt`. What differs is how long the engines hold the thread. A
WebAssembly generation would hold the main thread for its whole length — no
microphone chunk, and so no barge-in, gets through until it ends — so
`web/main.js` sets `env.backends.onnx.wasm.proxy`, and onnxruntime-web runs the
STT, LM, and TTS in its own worker. On WebGPU the work is on the GPU anyway.
Neither choice touches an adapter or a stage.

Playback does not depend on the main thread either: `WebAudioSink` schedules
each synthesized clause on the audio clock, and the rendering thread plays it
from there. `play` therefore never waits, so the output pump sees a barge-in
`Interrupt` as soon as it arrives and needs no race against an in-flight push
the way the desktop pump does.

Kokoro renders its whole input in one inference run, and nothing can stop a
run part-way. So `KokoroTts` splits each sentence at its commas, semicolons,
colons, and dashes, speaks it a clause at a time, and trims the silence Kokoro
pads each run with where two clauses meet. Speech starts once the first clause
is synthesized: for "The lamps would be lit up, casting a warm glow on the
streets, as the night air filled with the sounds of the city." that takes 0.4 s
on WebGPU and 2.7 s on single-threaded WebAssembly, against 1.2 s and 10.6 s
for the whole sentence. A barge-in waits out at most the clause in flight —
the next reply's first clause queues behind it if it is still running — and
the rest of the sentence never starts.

Kokoro's speed still matters. On WebGPU it synthesizes several times faster
than it speaks, so clauses play back to back. Single-threaded WebAssembly is
slower than real time — 3.8 s of audio takes 6.0 s — so the speakers run dry
between clauses, leaving pauses of about 1.7 s mid-sentence in our runs.
Serving the page cross-origin-isolated — `Cross-Origin-Opener-Policy:
same-origin` and `Cross-Origin-Embedder-Policy: require-corp` — lets
`web/main.js` give WebAssembly four threads, which brings that 6.0 s to 3.4 s;
the CDNs and the Hub both serve what an isolated page needs. `python3 -m
http.server` cannot send those headers.

[Transformers.js]: https://huggingface.co/docs/transformers.js
[kokoro-js]: https://www.npmjs.com/package/kokoro-js
