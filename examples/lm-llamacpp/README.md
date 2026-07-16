# llama.cpp voice-to-reply example

This example runs the full input side of a voice agent: it captures the
default microphone, resamples to 16 kHz mono, uses Sherpa Silero VAD to
bracket each utterance, decodes the completed utterance with Moonshine v2,
and hands the transcript to a local llama.cpp chat model, which streams its
reply to the console token by token.

```text
CpalSource
    │ device sample rate, mono
    ▼
ResamplerStage (16 kHz mono)
    ▼
VadStage<SherpaVad>
    ▼
SttStage<OfflineSherpaStt>
    ▼
UserTurnGate (prints "You: …", drops empty finals)
    ▼
LmStage<LlamaCpp>
    ▼
streamed agent reply
```

The VAD and STT configuration matches the
[`stt-sherpa-moonshine`](../stt-sherpa-moonshine) example: a 0.35 speech
threshold, 100 ms minimum speech, one second of pre-roll, 500 ms of trailing
silence, and a 30-second utterance ceiling. As there, Moonshine v2 runs as an
offline recognizer over the VAD-bracketed utterance, so there are no partial
user hypotheses — one final transcript per utterance.

`LmStage` accumulates the running conversation — the system prompt, every
completed user utterance, and every generated reply — so successive turns see
the whole history. It consumes user transcripts, so the example inserts a
small `UserTurnGate` stage above it that prints each completed utterance
(the last place the user side of the conversation can be observed) and drops
empty finals so a noise trigger never wakes the language model. The stage is
also a compact template for writing your own: state changes and frame
routing in `decide_data`, I/O in `perform`.

The reply streams as append-only partial transcripts followed by one final,
which is what lets a later barge-in stop a reply within one token — see the
`LmStage` documentation.

## Requirements

- Rust 1.86 or newer.
- macOS, Windows, or Linux with a working microphone.
- Microphone permission for the terminal or application running Cargo.
- A Sherpa-compatible 16 kHz Silero VAD model.
- A Sherpa Moonshine v2 model.
- A chat GGUF model with an embedded chat template.

The first build compiles llama.cpp from source, which takes a few minutes.

## Download the models

From the repository root, download Silero VAD, the quantized Moonshine v2
base English model, and a small chat GGUF (Qwen2.5 0.5B Instruct, ~400 MB):

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
```

The tiny English Moonshine model (~43 MB of model files) also works — see the
[`stt-sherpa-moonshine` README](../stt-sherpa-moonshine/README.md) for its
download command.

The repository ignores the `models/` directory, so downloaded model artifacts
do not appear as source changes.

Any chat GGUF works in place of Qwen as long as its metadata embeds a chat
template (instruct-tuned releases do); `LlamaCppConfig::with_chat_template`
can supply one otherwise.

## Run

```console
MODEL=./models/sherpa-onnx-moonshine-base-en-quantized-2026-02-27

cargo run -p lm-llamacpp -- \
  --vad-model ./models/silero_vad.onnx \
  --encoder "$MODEL/encoder_model.ort" \
  --merged-decoder "$MODEL/decoder_model_merged.ort" \
  --tokens "$MODEL/tokens.txt" \
  --lm-model ./models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

Use `--seconds 60` for a bounded run, `--stt-threads 1` for a low-power
profile, or `--system-prompt "…"` to replace the default prompt (a friendly
assistant asked to answer in one or two spoken-style sentences).

Speak, then pause long enough for VAD to close the utterance. Output
resembles:

```text
lm-llamacpp: loading ./models/qwen2.5-0.5b-instruct-q4_k_m.gguf …
lm-llamacpp: input = Default Microphone @ 48000 Hz mono
lm-llamacpp: processing @ 16000 Hz mono
lm-llamacpp: STT compute threads = 2
lm-llamacpp: VAD threshold = 0.35, minimum speech = 100 ms, pre-roll = 1000 ms, trailing silence = 500 ms
lm-llamacpp: listening until Ctrl-C
SpeechStarted
SpeechStopped (1.92 s)
You: what is the capital of france
Agent: The capital of France is Paris.
SpeechStarted
SpeechStopped (1.34 s)
You: and how many people live there
Agent: About two million people live in Paris itself.
```

The `Agent:` line appears incrementally as the model decodes; the second
answer shows the conversation history working — "there" resolves against the
previous turn.

The model runs on CPU by default with llama.cpp's mobile-oriented defaults
(4096-token context, at most 256 generated tokens per turn). The first reply
of a session includes the prompt prefill, so it starts noticeably slower than
later ones.

The first Sherpa build may download matching native libraries. Set
`SHERPA_ONNX_LIB_DIR` before running Cargo to use an existing compatible
Sherpa installation instead.

## Model-backed integration test

The integration test drives a committed 48 kHz WAV through the real
resampler, VAD, STT, and llama.cpp stages and asserts that a reply streams
back. The WAV is under `test-resources/audio`; tests do not depend on files
outside the repository other than the downloaded models.

Cargo runs test binaries with the package directory as their working
directory, so pass the model paths absolute (here via `$PWD`, from the
repository root):

```console
MODEL="$PWD/models/sherpa-onnx-moonshine-base-en-quantized-2026-02-27"

SHERPA_VAD_MODEL="$PWD/models/silero_vad.onnx" \
SHERPA_MOONSHINE_ENCODER="$MODEL/encoder_model.ort" \
SHERPA_MOONSHINE_MERGED_DECODER="$MODEL/decoder_model_merged.ort" \
SHERPA_MOONSHINE_TOKENS="$MODEL/tokens.txt" \
PIPECRAB_LLAMA_MODEL="$PWD/models/qwen2.5-0.5b-instruct-q4_k_m.gguf" \
cargo test -p lm-llamacpp --test model_pipeline -- --ignored --nocapture
```

The test is compiled but ignored in ordinary `cargo test` and CI runs because
the model artifacts are large and are not checked into the repository.
