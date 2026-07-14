use std::sync::Arc;
use std::time::Duration;

use futures::executor::block_on;
use pipecrab::{AudioChunk, AudioFormat, DataFrame, PipelineBuilder, Received};
use pipecrab_audio::ResamplerStage;
use pipecrab_stt::SttStage;
use pipecrab_stt_sherpa::{MoonshineV2Config, OfflineSherpaStt};
use pipecrab_vad::{GateConfig, VadStage};
use pipecrab_vad_sherpa::{SherpaVad, SherpaVadConfig};

#[test]
#[ignore = "requires SHERPA_VAD_MODEL and the three SHERPA_MOONSHINE_* model paths"]
fn vad_gated_wave_produces_a_moonshine_transcript() {
    block_on(async {
        let wave_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test-resources/audio/sherpa-zipformer-en-20m-0-48khz.wav");
        let wave = sherpa_onnx::Wave::read(wave_path.to_str().expect("UTF-8 test resource path"))
            .expect("read test speech resource");
        assert_eq!(wave.sample_rate(), 48_000, "test wave must be 48 kHz");

        let mut vad_config = SherpaVadConfig::new(required("SHERPA_VAD_MODEL"));
        vad_config.threshold = 0.35;
        vad_config.min_speech_duration = 0.1;
        vad_config.min_silence_duration = 0.5;
        vad_config.max_speech_duration = 30.0;
        let detector = SherpaVad::new(vad_config).expect("create VAD");
        let transcriber = OfflineSherpaStt::new(MoonshineV2Config::new(
            required("SHERPA_MOONSHINE_ENCODER"),
            required("SHERPA_MOONSHINE_MERGED_DECODER"),
            required("SHERPA_MOONSHINE_TOKENS"),
        ))
        .expect("create Moonshine v2 STT");

        let (ends, driver) = PipelineBuilder::new()
            .stage(ResamplerStage::new(AudioFormat::new(16_000, 1)).expect("create resampler"))
            .stage(VadStage::with_config(
                detector,
                GateConfig {
                    preroll: Duration::from_secs(1),
                },
            ))
            .stage(SttStage::new(transcriber))
            .build()
            .start();
        let input = ends.input;
        let mut output = ends.output;

        let pump = async move {
            for samples in wave.samples().chunks(960) {
                input
                    .send_data(DataFrame::Audio(AudioChunk::new(
                        Arc::from(samples),
                        AudioFormat::new(48_000, 1),
                    )))
                    .await
                    .expect("pipeline input remains open");
            }
            for _ in 0..50 {
                input
                    .send_data(DataFrame::Audio(AudioChunk::new(
                        Arc::from([0.0; 960]),
                        AudioFormat::new(48_000, 1),
                    )))
                    .await
                    .expect("pipeline input remains open");
            }
        };
        let drain = async move {
            let mut finals = Vec::new();
            while let Some(received) = output.recv().await {
                if let Received::Data(DataFrame::Transcript(transcript)) = received {
                    if transcript.finality == pipecrab::Finality::Final {
                        finals.push(transcript.text);
                    }
                }
            }
            finals
        };

        let (_, _, finals) = futures::join!(driver, pump, drain);
        assert!(
            finals.iter().any(|text| !text.is_empty()),
            "VAD-gated known speech produced no Moonshine transcript: {finals:?}"
        );
        println!("{finals:?}");
    });
}

fn required(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("set {name}"))
}
