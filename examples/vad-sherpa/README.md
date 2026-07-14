# Sherpa VAD example

This example captures the default microphone with cpal, resamples its audio to
16 kHz mono, and runs Sherpa ONNX's Silero voice-activity detector. It prints a
`SpeechStarted` or `SpeechStopped` line whenever the detector changes state.

## Requirements

- Rust 1.86 or newer.
- macOS, Windows, or Linux with a working microphone.
- Microphone permission for the terminal or application running Cargo.
- A Sherpa-compatible 16 kHz Silero VAD ONNX model.

## Download the model

From the repository root:

```console
curl -L \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx \
  -o silero_vad.onnx
```

## Run

Listen until Ctrl-C:

```console
cargo run -p vad-sherpa -- --model ./silero_vad.onnx
```

Use `--seconds` for a bounded run:

```console
cargo run -p vad-sherpa -- --model ./silero_vad.onnx --seconds 10
```

Speak, pause for at least a quarter second, and speak again. Expected output
resembles:

```text
vad-sherpa: input = Default Microphone @ 48000 Hz mono
vad-sherpa: processing @ 16000 Hz mono
SpeechStarted
SpeechStopped
```

The first Sherpa build may download matching native libraries. Set
`SHERPA_ONNX_LIB_DIR` before running Cargo to use an existing compatible Sherpa
installation instead.
