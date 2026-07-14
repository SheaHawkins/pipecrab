# Audio test resources

`sherpa-zipformer-en-20m-0.wav` is the `test_wavs/0.wav` sample distributed in
Sherpa ONNX's `sherpa-onnx-streaming-zipformer-en-20M-2023-02-17` model archive:

<https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2>

The spoken reference text supplied with that archive is:

> AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE
> SQUALID QUARTER OF THE BROTHELS

The WAV is committed so model-backed tests use one stable input. Those tests
remain ignored by default because their ONNX model files are external.
