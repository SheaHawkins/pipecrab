# Sherpa streaming STT example

This example captures the default microphone, resamples its audio to 16 kHz
mono, uses Sherpa Silero VAD to bracket each utterance, and feeds the gated
audio to Sherpa's streaming recognizer. It prints changing partial hypotheses
and one final transcript for each completed utterance.

```text
CpalSource
    │ device sample rate, mono
    ▼
ResamplerStage (16 kHz mono)
    ▼
VadStage<SherpaVad>
    ▼
SttStage<SherpaStt>
    ▼
partial and final transcripts
```

The example uses a short-utterance VAD profile: a 0.35 speech threshold, 100 ms
minimum speech, one second of pre-roll, 500 ms of trailing silence, and a
30-second utterance ceiling. Sherpa endpoint detection remains disabled; VAD is
the only boundary authority.

With its default configuration, `SherpaStt` also primes each recognizer stream
with one second of decoded silence and appends 300 ms of silence when the
utterance ends. Both durations are fields on `SherpaSttConfig`. The VAD pre-roll
retains microphone audio before detection; the recognizer padding supplies
model context so opening words and short utterances are not lost at stream
edges.

## Requirements

- Rust 1.86 or newer.
- macOS, Windows, or Linux with a working microphone.
- Microphone permission for the terminal or application running Cargo.
- A Sherpa-compatible 16 kHz Silero VAD model.
- A Sherpa streaming transducer model.

## Download the models

From the repository root, download Silero VAD and Sherpa's small English
streaming Zipformer:

```console
curl -L \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx \
  -o silero_vad.onnx

curl -L \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2 \
  -o sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2

tar xvf sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2
```

The example below uses the int8 encoder and joiner with the floating-point
decoder.

## Run

```console
MODEL=./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17

cargo run -p stt-sherpa -- \
  --vad-model ./silero_vad.onnx \
  --encoder "$MODEL/encoder-epoch-99-avg-1.int8.onnx" \
  --decoder "$MODEL/decoder-epoch-99-avg-1.onnx" \
  --joiner "$MODEL/joiner-epoch-99-avg-1.int8.onnx" \
  --tokens "$MODEL/tokens.txt"
```

Use `--seconds` for a bounded run or `--stt-threads 1` for a low-power profile:

```console
MODEL=./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17

cargo run -p stt-sherpa -- \
  --vad-model ./silero_vad.onnx \
  --encoder "$MODEL/encoder-epoch-99-avg-1.int8.onnx" \
  --decoder "$MODEL/decoder-epoch-99-avg-1.onnx" \
  --joiner "$MODEL/joiner-epoch-99-avg-1.int8.onnx" \
  --tokens "$MODEL/tokens.txt" \
  --stt-threads 1 \
  --seconds 30
```

Speak and then pause long enough for VAD to close the utterance. Output
resembles:

```text
stt-sherpa: input = Default Microphone @ 48000 Hz mono
stt-sherpa: processing @ 16000 Hz mono
stt-sherpa: STT compute threads = 2
stt-sherpa: VAD threshold = 0.35, minimum speech = 100 ms, pre-roll = 1000 ms, trailing silence = 500 ms
SpeechStarted
Partial: hello
Partial: hello world
SpeechStopped (1.84 s)
Final: hello world
```

An empty result is printed as `Final: <no speech recognized>`. The duration on
the preceding `SpeechStopped` line helps distinguish a short noise trigger from
a real utterance that the model failed to recognize.

The 20M model is a latency and memory baseline, not a short-keyword model. An
isolated syllable such as "hi" can produce an empty hypothesis even when the
complete waveform is fed directly to the recognizer without VAD. Stream
padding prevents boundary clipping, but it cannot recover a token the acoustic
model scores as blank. Applications that require reliable isolated keywords
should benchmark a larger streaming English model or use contextual biasing.

The first Sherpa build may download matching native libraries. Set
`SHERPA_ONNX_LIB_DIR` before running Cargo to use an existing compatible Sherpa
installation instead.
