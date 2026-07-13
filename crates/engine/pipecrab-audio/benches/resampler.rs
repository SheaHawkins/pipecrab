//! Asserted orchestrator-occupancy benchmark for synchronous resampling.

use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use pipecrab_audio::{AudioChunk, AudioFormat, ResamplerStage};
use pipecrab_core::{DataFrame, Processor};

const STEADY_STATE_FRACTION: f64 = 0.25;
const COLD_START_FRACTION: f64 = 0.25;
const ITERATIONS: u32 = 500;

struct Case {
    name: &'static str,
    input: AudioFormat,
    output: AudioFormat,
    frames: usize,
}

fn main() {
    for case in [
        Case {
            name: "48k stereo -> 16k mono",
            input: AudioFormat::new(48_000, 2),
            output: AudioFormat::new(16_000, 1),
            frames: 960,
        },
        Case {
            name: "24k mono -> 48k mono",
            input: AudioFormat::new(24_000, 1),
            output: AudioFormat::new(48_000, 1),
            frames: 480,
        },
        Case {
            name: "44.1k stereo -> 48k stereo",
            input: AudioFormat::new(44_100, 2),
            output: AudioFormat::new(48_000, 2),
            frames: 882,
        },
    ] {
        run(case);
    }
}

fn run(case: Case) {
    let frame = sine_chunk(case.input, case.frames);
    let chunk_duration =
        Duration::from_secs_f64(case.frames as f64 / case.input.sample_rate as f64);
    let mut stage = ResamplerStage::new(case.output).unwrap();

    let started = Instant::now();
    black_box(stage.decide_data(black_box(&frame)));
    let cold = started.elapsed();
    assert_budget(
        case.name,
        "cold start",
        cold,
        chunk_duration,
        COLD_START_FRACTION,
    );

    for _ in 0..8 {
        black_box(stage.decide_data(black_box(&frame)));
    }
    let started = Instant::now();
    for _ in 0..ITERATIONS {
        black_box(stage.decide_data(black_box(&frame)));
    }
    let elapsed = started.elapsed();
    let audio_duration = chunk_duration * ITERATIONS;
    assert_budget(
        case.name,
        "steady state",
        elapsed,
        audio_duration,
        STEADY_STATE_FRACTION,
    );
}

fn assert_budget(
    case: &str,
    phase: &str,
    elapsed: Duration,
    audio_duration: Duration,
    maximum_fraction: f64,
) {
    let fraction = elapsed.as_secs_f64() / audio_duration.as_secs_f64();
    println!(
        "{case} ({phase}): {:.3} ms processing / {:.3} ms audio = {:.2}%",
        elapsed.as_secs_f64() * 1_000.0,
        audio_duration.as_secs_f64() * 1_000.0,
        fraction * 100.0,
    );
    assert!(
        fraction <= maximum_fraction,
        "{case} {phase} occupied {:.2}% of its audio duration; budget is {:.2}%",
        fraction * 100.0,
        maximum_fraction * 100.0,
    );
}

fn sine_chunk(format: AudioFormat, frames: usize) -> DataFrame {
    let channels = usize::from(format.channels);
    let mut samples = Vec::with_capacity(frames * channels);
    for frame in 0..frames {
        let sample =
            (std::f32::consts::TAU * 440.0 * frame as f32 / format.sample_rate as f32).sin();
        for channel in 0..channels {
            samples.push(if channel % 2 == 0 { sample } else { -sample });
        }
    }
    DataFrame::Audio(AudioChunk::new(Arc::from(samples), format))
}
