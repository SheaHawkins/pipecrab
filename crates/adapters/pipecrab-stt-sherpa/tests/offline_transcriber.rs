use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::ThreadId;

use futures::executor::block_on;
use pipecrab_core::AudioFormat;
use pipecrab_stt::{StreamingTranscriber, SttEvent};
use pipecrab_stt_sherpa::{MoonshineV2Config, OfflineBackend, OfflineSherpaStt};

#[derive(Default)]
struct Probe {
    utterances: Vec<Vec<f32>>,
    threads: Vec<ThreadId>,
}

struct ScriptedBackend {
    results: Vec<String>,
    probe: Arc<Mutex<Probe>>,
}

impl OfflineBackend for ScriptedBackend {
    fn transcribe(&mut self, samples: &[f32]) -> Option<String> {
        let mut probe = self.probe.lock().unwrap();
        probe.utterances.push(samples.to_vec());
        probe.threads.push(std::thread::current().id());
        drop(probe);
        Some(self.results.remove(0))
    }
}

#[test]
fn buffers_chunks_and_emits_only_a_final_result() {
    block_on(async {
        let probe = Arc::new(Mutex::new(Probe::default()));
        let transcriber = OfflineSherpaStt::with_backend(ScriptedBackend {
            results: vec!["hello moonshine".into(), "again".into()],
            probe: probe.clone(),
        })
        .unwrap();

        assert_eq!(transcriber.input_format(), AudioFormat::new(16_000, 1));
        transcriber.begin_utterance().await.unwrap();
        assert!(
            transcriber
                .feed(Arc::from([0.1, 0.2]))
                .await
                .unwrap()
                .is_empty()
        );
        assert!(transcriber.feed(Arc::from([0.3])).await.unwrap().is_empty());
        assert_eq!(
            transcriber.end_utterance().await.unwrap(),
            vec![SttEvent::Final("hello moonshine".into())]
        );

        transcriber.begin_utterance().await.unwrap();
        transcriber.feed(Arc::from([0.4])).await.unwrap();
        assert_eq!(
            transcriber.end_utterance().await.unwrap(),
            vec![SttEvent::Final("again".into())]
        );

        let probe = probe.lock().unwrap();
        assert_eq!(probe.utterances, [vec![0.1, 0.2, 0.3], vec![0.4]]);
        assert!(
            probe
                .threads
                .iter()
                .all(|thread| *thread == probe.threads[0])
        );
        assert_ne!(probe.threads[0], std::thread::current().id());
    });
}

#[test]
fn chunks_a_long_utterance_and_merges_overlapping_text() {
    block_on(async {
        let probe = Arc::new(Mutex::new(Probe::default()));
        let transcriber = OfflineSherpaStt::with_backend(ScriptedBackend {
            results: vec![
                "hello brave new world".into(),
                "new world this is a test".into(),
            ],
            probe: probe.clone(),
        })
        .unwrap();
        let samples: Vec<f32> = (0..140_000).map(|sample| sample as f32).collect();

        transcriber.begin_utterance().await.unwrap();
        transcriber.feed(Arc::from(samples)).await.unwrap();
        assert_eq!(
            transcriber.end_utterance().await.unwrap(),
            vec![SttEvent::Final(
                "hello brave new world this is a test".into()
            )]
        );

        let probe = probe.lock().unwrap();
        assert_eq!(probe.utterances.len(), 2);
        assert_eq!(probe.utterances[0].len(), 128_000);
        assert_eq!(probe.utterances[1].len(), 20_000);
        assert_eq!(probe.utterances[1][0], 120_000.0);
    });
}

#[test]
fn protocol_violations_are_reported() {
    block_on(async {
        let transcriber = OfflineSherpaStt::with_backend(ScriptedBackend {
            results: Vec::new(),
            probe: Arc::new(Mutex::new(Probe::default())),
        })
        .unwrap();

        assert!(
            transcriber
                .feed(Arc::from([]))
                .await
                .unwrap_err()
                .to_string()
                .contains("without an active utterance")
        );
        assert!(
            transcriber
                .end_utterance()
                .await
                .unwrap_err()
                .to_string()
                .contains("without an active utterance")
        );
        transcriber.begin_utterance().await.unwrap();
        assert!(
            transcriber
                .begin_utterance()
                .await
                .unwrap_err()
                .to_string()
                .contains("already active")
        );
    });
}

#[test]
fn cancellation_discards_buffered_audio_before_the_next_utterance() {
    block_on(async {
        let probe = Arc::new(Mutex::new(Probe::default()));
        let transcriber = OfflineSherpaStt::with_backend(ScriptedBackend {
            results: vec!["current".into()],
            probe: probe.clone(),
        })
        .unwrap();

        transcriber.begin_utterance().await.unwrap();
        transcriber.feed(Arc::from([0.1, 0.2])).await.unwrap();
        transcriber.cancel();

        transcriber.begin_utterance().await.unwrap();
        transcriber.feed(Arc::from([0.3])).await.unwrap();
        assert_eq!(
            transcriber.end_utterance().await.unwrap(),
            vec![SttEvent::Final("current".into())]
        );
        assert_eq!(probe.lock().unwrap().utterances, [vec![0.3]]);
    });
}

struct BlockingBackend {
    reached: mpsc::Sender<()>,
    release: mpsc::Receiver<()>,
    calls: Arc<AtomicUsize>,
}

impl OfflineBackend for BlockingBackend {
    fn transcribe(&mut self, _samples: &[f32]) -> Option<String> {
        if self.calls.fetch_add(1, Ordering::AcqRel) == 0 {
            self.reached.send(()).unwrap();
            self.release.recv().unwrap();
        }
        Some("stale text".into())
    }
}

#[test]
fn cancellation_suppresses_an_in_flight_offline_result() {
    block_on(async {
        let (reached_tx, reached_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let calls = Arc::new(AtomicUsize::new(0));
        let transcriber = OfflineSherpaStt::with_backend(BlockingBackend {
            reached: reached_tx,
            release: release_rx,
            calls: calls.clone(),
        })
        .unwrap();

        transcriber.begin_utterance().await.unwrap();
        transcriber
            .feed(Arc::from(vec![0.1; 140_000]))
            .await
            .unwrap();
        let mut end = Box::pin(transcriber.end_utterance());
        assert!(futures::poll!(end.as_mut()).is_pending());
        reached_rx.recv().unwrap();
        transcriber.cancel();
        release_tx.send(()).unwrap();

        assert!(end.await.unwrap().is_empty());
        assert_eq!(calls.load(Ordering::Acquire), 1);
    });
}

#[test]
#[ignore = "requires SHERPA_MOONSHINE_ENCODER, SHERPA_MOONSHINE_MERGED_DECODER, and SHERPA_MOONSHINE_TOKENS"]
fn moonshine_v2_transcribes_a_wave_file() {
    block_on(async {
        let config = MoonshineV2Config::new(
            std::env::var("SHERPA_MOONSHINE_ENCODER").expect("set SHERPA_MOONSHINE_ENCODER"),
            std::env::var("SHERPA_MOONSHINE_MERGED_DECODER")
                .expect("set SHERPA_MOONSHINE_MERGED_DECODER"),
            std::env::var("SHERPA_MOONSHINE_TOKENS").expect("set SHERPA_MOONSHINE_TOKENS"),
        );
        let wave_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../test-resources/audio/sherpa-zipformer-en-20m-0.wav");
        let wave = sherpa_onnx::Wave::read(wave_path.to_str().expect("UTF-8 test resource path"))
            .expect("read test speech resource");
        assert_eq!(wave.sample_rate(), 16_000, "test wave must be 16 kHz");

        let transcriber = OfflineSherpaStt::new(config).unwrap();
        transcriber.begin_utterance().await.unwrap();
        for samples in wave.samples().chunks(512) {
            transcriber.feed(Arc::from(samples)).await.unwrap();
        }
        let events = transcriber.end_utterance().await.unwrap();
        let [SttEvent::Final(text)] = events.as_slice() else {
            panic!("expected one final transcript, got {events:?}");
        };
        assert!(!text.is_empty(), "known speech wave produced no transcript");
        println!("{text}");
    });
}

#[test]
#[ignore = "requires SHERPA_MOONSHINE_ENCODER, SHERPA_MOONSHINE_MERGED_DECODER, and SHERPA_MOONSHINE_TOKENS"]
fn moonshine_v2_chunks_a_long_wave_file() {
    block_on(async {
        let config = MoonshineV2Config::new(
            std::env::var("SHERPA_MOONSHINE_ENCODER").expect("set SHERPA_MOONSHINE_ENCODER"),
            std::env::var("SHERPA_MOONSHINE_MERGED_DECODER")
                .expect("set SHERPA_MOONSHINE_MERGED_DECODER"),
            std::env::var("SHERPA_MOONSHINE_TOKENS").expect("set SHERPA_MOONSHINE_TOKENS"),
        );
        let wave_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../test-resources/audio/sherpa-zipformer-en-20m-0.wav");
        let wave = sherpa_onnx::Wave::read(wave_path.to_str().expect("UTF-8 test resource path"))
            .expect("read test speech resource");
        let samples: Vec<f32> = wave
            .samples()
            .iter()
            .copied()
            .cycle()
            .take(13 * 16_000)
            .collect();

        let transcriber = OfflineSherpaStt::new(config).unwrap();
        transcriber.begin_utterance().await.unwrap();
        for samples in samples.chunks(512) {
            transcriber.feed(Arc::from(samples)).await.unwrap();
        }
        let events = transcriber.end_utterance().await.unwrap();
        let [SttEvent::Final(text)] = events.as_slice() else {
            panic!("expected one final transcript, got {events:?}");
        };
        assert!(
            !text.is_empty(),
            "chunked long speech produced no transcript"
        );
        println!("{text}");
    });
}
