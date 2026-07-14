use std::sync::Arc;

use pipecrab_audio::{AudioChunk, AudioFormat, Resampler, RubatoSincResampler};

#[test]
fn microphone_resampling_preserves_the_speech_fixture() {
    let resources =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-resources/audio");
    let source = sherpa_onnx::Wave::read(
        resources
            .join("sherpa-zipformer-en-20m-0.wav")
            .to_str()
            .expect("UTF-8 fixture path"),
    )
    .expect("read 16 kHz fixture");
    let microphone = sherpa_onnx::Wave::read(
        resources
            .join("sherpa-zipformer-en-20m-0-48khz.wav")
            .to_str()
            .expect("UTF-8 fixture path"),
    )
    .expect("read 48 kHz fixture");

    let mut resampler = RubatoSincResampler::new(AudioFormat::new(16_000, 1)).unwrap();
    let mut converted = Vec::new();
    for samples in microphone.samples().chunks(960) {
        if let Some(chunk) = resampler
            .resample(&AudioChunk::new(
                Arc::from(samples),
                AudioFormat::new(48_000, 1),
            ))
            .unwrap()
        {
            converted.extend_from_slice(&chunk.samples);
        }
    }

    let source_rms = rms(source.samples());
    let converted_rms = rms(&converted);
    let (lag, error) = (-128..=128)
        .map(|lag| (lag, normalized_error(source.samples(), &converted, lag)))
        .min_by(|left, right| left.1.total_cmp(&right.1))
        .unwrap();

    println!(
        "source={} converted={} rms={source_rms:.6}/{converted_rms:.6} lag={lag} error={error:.6}",
        source.samples().len(),
        converted.len(),
    );
    assert!(lag.abs() <= 2, "resampled waveform lag is {lag} samples");
    assert!(error < 0.02, "resampled waveform error is {error}");
    assert!(
        (converted_rms / source_rms - 1.0).abs() < 0.01,
        "resampled RMS changed from {source_rms} to {converted_rms}"
    );
}

fn rms(samples: &[f32]) -> f32 {
    (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32).sqrt()
}

fn normalized_error(reference: &[f32], converted: &[f32], lag: i32) -> f32 {
    let reference_start = lag.max(0) as usize;
    let converted_start = (-lag).max(0) as usize;
    let len = (reference.len() - reference_start).min(converted.len() - converted_start);
    let squared_error = reference[reference_start..reference_start + len]
        .iter()
        .zip(&converted[converted_start..converted_start + len])
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f32>();
    squared_error / len as f32 / rms(reference).powi(2)
}
