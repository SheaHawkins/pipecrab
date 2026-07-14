# Audio test resources

`sherpa-zipformer-en-20m-0.wav` is the `test_wavs/0.wav` sample distributed in
Sherpa ONNX's `sherpa-onnx-streaming-zipformer-en-20M-2023-02-17` model archive:

<https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2>

The spoken reference text supplied with that archive is:

> AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE
> SQUALID QUARTER OF THE BROTHELS

`sherpa-zipformer-en-20m-0-48khz.wav` is the same fixture resampled to 48 kHz
mono PCM with FFmpeg. It exercises the microphone example's 48→16 kHz
resampling path before VAD and STT.

`sherpa-zipformer-en-20m-0-short-48khz.wav` is the first 800 milliseconds of
that 48 kHz fixture. It contains roughly 400 milliseconds of speech and covers
the short-utterance VAD→STT path.

The WAV is committed so model-backed tests use one stable input. Those tests
remain ignored by default because their ONNX model files are external.
