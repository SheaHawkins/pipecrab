# Sherpa Moonshine v2 STT example

This example captures the default microphone, resamples it to 16 kHz mono,
uses Sherpa Silero VAD to bracket each utterance, and decodes the completed
utterance with Moonshine v2.

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
one final transcript per utterance
```

Moonshine v2 is exposed by Sherpa's `OfflineRecognizer`, not its
`OnlineRecognizer`. Sherpa's Moonshine v2 “simulated streaming” applications
run an offline recognizer over bounded audio windows. PipeCrab instead treats
its VAD edges as the utterance authority: audio is accumulated between
`SpeechStarted` and `SpeechStopped`, then decoded once. The example therefore
prints final transcripts but no partial hypotheses.

The VAD profile matches the streaming Sherpa example: a 0.35 threshold, 100 ms
minimum speech, one second of pre-roll, 500 ms trailing silence, and a
30-second utterance ceiling. The pre-roll preserves audio that arrived before
the VAD start decision.

## Requirements

- Rust 1.86 or newer.
- macOS, Windows, or Linux with a working microphone.
- Microphone permission for the terminal or application running Cargo.
- A Sherpa-compatible 16 kHz Silero VAD model.
- A Sherpa Moonshine v2 model.

## Download the models

From the repository root:

```console
curl -L \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx \
  -o silero_vad.onnx

curl -L \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-base-en-quantized-2026-02-27.tar.bz2 \
  -o sherpa-onnx-moonshine-base-en-quantized-2026-02-27.tar.bz2

tar xvf sherpa-onnx-moonshine-base-en-quantized-2026-02-27.tar.bz2
```

Moonshine v2 packages contain `encoder_model.ort`,
`decoder_model_merged.ort`, and `tokens.txt`. They do not use the four-model
Moonshine v1 layout.

## Run

```console
MODEL=./sherpa-onnx-moonshine-base-en-quantized-2026-02-27

cargo run -p stt-sherpa-moonshine -- \
  --vad-model ./silero_vad.onnx \
  --encoder "$MODEL/encoder_model.ort" \
  --merged-decoder "$MODEL/decoder_model_merged.ort" \
  --tokens "$MODEL/tokens.txt"
```

Use `--seconds 30` for a bounded run or `--stt-threads 1` for a low-power
profile. The default is two STT compute threads.

### Run with the tiny English model

Sherpa also publishes a quantized Moonshine v2 tiny English model. Its model
files total approximately 43 MB, compared with approximately 135 MB for the
base English model, making it the lighter starting point for constrained
devices.

Download and extract it from the repository root:

```console
curl -L \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2 \
  -o sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2

tar xvf sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2
```

Run it with the same example and flags:

```console
MODEL=./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27

cargo run -p stt-sherpa-moonshine -- \
  --vad-model ./silero_vad.onnx \
  --encoder "$MODEL/encoder_model.ort" \
  --merged-decoder "$MODEL/decoder_model_merged.ort" \
  --tokens "$MODEL/tokens.txt"
```

Expected output resembles:

```text
stt-sherpa-moonshine: input = Default Microphone @ 48000 Hz mono
stt-sherpa-moonshine: processing @ 16000 Hz mono
stt-sherpa-moonshine: STT compute threads = 2
SpeechStarted
SpeechStopped (1.84 s)
Final: hello world
```

The first Sherpa build may download matching native libraries. Set
`SHERPA_ONNX_LIB_DIR` before running Cargo to use an existing compatible Sherpa
installation instead.

## Model-backed integration test

The integration test drives a committed 48 kHz WAV through the real resampler,
VAD, and Moonshine v2 adapter. The WAV is under `test-resources/audio`; tests do
not depend on files outside the repository other than the downloaded models.

```console
MODEL=./sherpa-onnx-moonshine-base-en-quantized-2026-02-27

SHERPA_VAD_MODEL=./silero_vad.onnx \
SHERPA_MOONSHINE_ENCODER="$MODEL/encoder_model.ort" \
SHERPA_MOONSHINE_MERGED_DECODER="$MODEL/decoder_model_merged.ort" \
SHERPA_MOONSHINE_TOKENS="$MODEL/tokens.txt" \
cargo test -p stt-sherpa-moonshine --test model_pipeline -- --ignored --nocapture
```

The test is compiled but ignored in ordinary `cargo test` and CI runs because
the model artifacts are large and are not checked into the repository.
