# End-to-end voice agent example

Talk to a fully local voice agent: Sherpa VAD brackets each utterance from the
microphone, Moonshine v2 transcribes it, a llama.cpp chat model streams a
reply, and Kokoro speaks that reply through the default output device.

```text
CpalSource (mic)
    │ device sample rate, mono
    ▼
ResamplerStage (16 kHz mono)
    ▼
VadStage<SherpaVad>
    ▼
SttStage<OfflineSherpaStt>
    ▼
UserTurnGate            (prints "You: …", drops empty finals)
    ▼
LmStage<LlamaCpp>       (streams agent partials + one final)
    ▼
SentenceChunker         (one final agent transcript per sentence)
    ▼
AgentEcho               (prints "Agent: …", forwards it)
    ▼
TtsStage<SherpaTts>     (24 kHz mono audio per sentence)
    ▼
ResamplerStage (device rate)
    ▼
CpalSink (speaker)
```

The `SentenceChunker` is why the agent starts talking before the model has
finished writing: each completed sentence of the streaming generation becomes
its own final agent transcript, Kokoro synthesizes it while the next sentence
is still being generated, and Sherpa's own per-sentence progress callback
streams the audio on down the pipeline.

**Use headphones** — over speakers the microphone re-captures the agent's own
voice, the VAD opens, and the agent starts talking to itself. Barge-in (a
speech-start interrupt that stops an in-flight reply) is not wired into this
example; if you speak over the agent, the new reply queues behind the current
one.

## Requirements

- Rust 1.86 or newer.
- macOS, Windows, or Linux with a working microphone and output device.
- Microphone permission for the terminal or application running Cargo.
- A Sherpa-compatible 16 kHz Silero VAD model.
- A Sherpa Moonshine v2 model.
- A llama.cpp-compatible chat model in GGUF format.
- A Sherpa Kokoro model.

## Download the models

From the repository root:

```console
mkdir -p models

curl -L \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx \
  -o models/silero_vad.onnx

curl -L \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-base-en-quantized-2026-02-27.tar.bz2 \
  -o models/sherpa-onnx-moonshine-base-en-quantized-2026-02-27.tar.bz2

tar xvf models/sherpa-onnx-moonshine-base-en-quantized-2026-02-27.tar.bz2 \
  -C models

curl -L \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -o models/qwen2.5-0.5b-instruct-q4_k_m.gguf

curl -L \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2 \
  -o models/kokoro-en-v0_19.tar.bz2

tar xvf models/kokoro-en-v0_19.tar.bz2 -C models
```

The repository ignores the `models/` directory, so downloaded model artifacts
do not appear as source changes.

The Kokoro package contains `model.onnx`, `voices.bin`, `tokens.txt`, and an
`espeak-ng-data` directory. The English v0.19 model bundles eleven speakers
(0–10); `--speaker` selects one.

## Run

```console
ASR=./models/sherpa-onnx-moonshine-base-en-quantized-2026-02-27
TTS=./models/kokoro-en-v0_19

cargo run -p e2e-voice-agent -- \
  --vad-model ./models/silero_vad.onnx \
  --encoder "$ASR/encoder_model.ort" \
  --merged-decoder "$ASR/decoder_model_merged.ort" \
  --tokens "$ASR/tokens.txt" \
  --lm-model ./models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --tts-model "$TTS/model.onnx" \
  --tts-voices "$TTS/voices.bin" \
  --tts-tokens "$TTS/tokens.txt" \
  --tts-data-dir "$TTS/espeak-ng-data"
```

Then speak. Expected output resembles:

```text
e2e-voice-agent: loading ./models/qwen2.5-0.5b-instruct-q4_k_m.gguf …
e2e-voice-agent: input = MacBook Pro Microphone @ 48000 Hz mono
e2e-voice-agent: output = MacBook Pro Speakers @ 48000 Hz
e2e-voice-agent: kokoro @ 24000 Hz mono, speaker 0, speed 1
e2e-voice-agent: listening until Ctrl-C
SpeechStarted
SpeechStopped (1.92 s)
You: what is a crab
Agent: A crab is a crustacean with a hard shell and ten legs.
```

…followed by the same sentence spoken aloud.

Useful flags: `--speaker 8` picks another Kokoro voice, `--speed 1.2` talks
faster, `--system-prompt "…"` changes the agent's instructions,
`--seconds 30` bounds the run, and `--stt-threads 1` is a low-power profile.

The first Sherpa build may download matching native libraries. Set
`SHERPA_ONNX_LIB_DIR` before running Cargo to use an existing compatible Sherpa
installation instead.

## Model-backed integration test

The TTS adapter's integration test synthesizes real speech with the downloaded
Kokoro model. Cargo runs test binaries with the package directory as their
working directory, so pass the model paths absolute (here via `$PWD`, from the
repository root):

```console
TTS="$PWD/models/kokoro-en-v0_19"

SHERPA_KOKORO_MODEL="$TTS/model.onnx" \
SHERPA_KOKORO_VOICES="$TTS/voices.bin" \
SHERPA_KOKORO_TOKENS="$TTS/tokens.txt" \
SHERPA_KOKORO_DATA_DIR="$TTS/espeak-ng-data" \
cargo test -p pipecrab-tts-sherpa --test synthesizer -- --ignored --nocapture
```

The test is compiled but ignored in ordinary `cargo test` and CI runs because
the model artifacts are large and are not checked into the repository.
