# pipecrab-vad-sherpa

This crate adapts Sherpa ONNX's Silero VAD to PipeCrab's
`VoiceActivityDetector` trait. Sherpa runs on a dedicated actor thread so one
owner serializes every operation on the native detector.

## Components

```text
VadStage
   │ VoiceActivityDetector
   ▼
SherpaVad
   ├── reset generation (atomic)
   └── WorkerHandle
          ├── Command sender ───────────────┐
          └── actor thread join handle      │
                                            ▼
                                      worker thread
                                            │ owns
                                            ▼
                                      Box<dyn Backend>
                                            │ production implementation
                                            ▼
                                      SherpaBackend
                                            │ owns exactly one
                                            ▼
                         sherpa_onnx::VoiceActivityDetector
```

`Backend` is the small internal-engine boundary used by the worker. Its methods
mirror the Sherpa operations needed for VAD: inspect detection state, accept a
waveform, drain completed segments, and reset. Mutable receivers express that
the actor has exclusive access. Tests provide scripted `Backend`
implementations without loading an ONNX model.

`SherpaBackend` is the production `Backend`. It contains exactly one
`sherpa_onnx::VoiceActivityDetector`. That detector encapsulates the loaded
model and mutable VAD state; it is the Sherpa session for this adapter. It is
constructed, accessed, reset, and dropped on the worker thread. It is not
shared with another VAD or a future Sherpa STT recognizer.

`SherpaVad` is the public, inexpensive handle used by `VadStage`.
`WorkerHandle` keeps the command sender and thread join handle together. When
`SherpaVad` is dropped, `WorkerHandle` closes the command channel and
joins the actor, so the Sherpa detector is destroyed on its owning thread.

## Request flow

The private `Command` protocol currently has one operation:

```text
Process { samples: Arc<[f32]>, generation, reply }
```

1. `SherpaVad::process` snapshots the reset generation and sends the shared
   sample buffer with a one-shot reply channel.
2. The worker retains any partial input and feeds the backend exact 512-sample
   windows.
3. It compares `detected()` before and after each window to produce
   `SpeechStarted` and `SpeechStopped` edges.
4. It pops Sherpa's completed-segment queue because `VadStage` owns and forwards
   the original audio.
5. The worker replies with all edges produced by the command.

`SherpaVad::reset` only increments the atomic generation, so it is synchronous
and non-blocking. The worker checks that generation between windows and before
replying. A changed generation resets the backend, clears the partial window,
and discards stale events.

See [`examples/vad-sherpa`](../../../examples/vad-sherpa) for a live microphone
pipeline.
