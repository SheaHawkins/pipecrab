use std::sync::Arc;

use async_trait::async_trait;
use pipecrab_core::AudioFormat;
use pipecrab_runtime::PipelineBuilder;
use pipecrab_tts::{Synthesizer, TtsAudioStream, TtsError, TtsStage};
use pipecrab_vad::{VadError, VadEvent, VadStage, VoiceActivityDetector};

struct Detector;

#[async_trait]
impl VoiceActivityDetector for Detector {
    fn input_format(&self) -> AudioFormat {
        AudioFormat::new(16_000, 1)
    }

    async fn process(&self, _samples: Arc<[f32]>) -> Result<Vec<VadEvent>, VadError> {
        Ok(Vec::new())
    }

    fn reset(&self) {}
}

struct SynthesizerDouble;

#[async_trait]
impl Synthesizer for SynthesizerDouble {
    fn output_format(&self) -> AudioFormat {
        AudioFormat::new(24_000, 1)
    }

    async fn synthesize(&self, _text: &str) -> Result<TtsAudioStream, TtsError> {
        Ok(Box::pin(futures::stream::empty()))
    }

    fn cancel(&self) {}
}

#[test]
fn vad_and_tts_stages_compose_in_a_pipeline() {
    let _pipeline = PipelineBuilder::new()
        .stage(VadStage::new(Detector))
        .stage(TtsStage::new(SynthesizerDouble))
        .build();
}
