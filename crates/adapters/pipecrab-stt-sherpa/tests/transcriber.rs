use std::collections::VecDeque;
use std::sync::{Arc, Mutex, mpsc};
use std::thread::ThreadId;

use futures::executor::block_on;
use pipecrab_core::AudioFormat;
use pipecrab_stt::{StreamingTranscriber, SttEvent};
use pipecrab_stt_sherpa::{Backend, SherpaStt, SherpaSttConfig};

#[derive(Default)]
struct Probe {
    streams_created: usize,
    streams_dropped: usize,
    accepted: Vec<(usize, usize, bool)>,
    input_finished: usize,
    readiness_checks: usize,
    decodes: usize,
    result_reads: usize,
    threads: Vec<ThreadId>,
}

struct ScriptedStream {
    pending: VecDeque<Option<String>>,
    current: Option<String>,
    probe: Arc<Mutex<Probe>>,
}

impl Drop for ScriptedStream {
    fn drop(&mut self) {
        self.probe.lock().unwrap().streams_dropped += 1;
    }
}

struct ScriptedBackend {
    feed_steps: VecDeque<Vec<Option<String>>>,
    end_steps: Vec<Option<String>>,
    probe: Arc<Mutex<Probe>>,
}

impl ScriptedBackend {
    fn new(
        feed_steps: impl IntoIterator<Item = Vec<Option<&'static str>>>,
        end_steps: impl IntoIterator<Item = Option<&'static str>>,
    ) -> (Self, Arc<Mutex<Probe>>) {
        let probe = Arc::new(Mutex::new(Probe::default()));
        (
            Self {
                feed_steps: feed_steps
                    .into_iter()
                    .map(|steps| {
                        steps
                            .into_iter()
                            .map(|text| text.map(str::to_owned))
                            .collect()
                    })
                    .collect(),
                end_steps: end_steps
                    .into_iter()
                    .map(|text| text.map(str::to_owned))
                    .collect(),
                probe: probe.clone(),
            },
            probe,
        )
    }

    fn record_thread(probe: &mut Probe) {
        probe.threads.push(std::thread::current().id());
    }
}

impl Backend for ScriptedBackend {
    type Stream = ScriptedStream;

    fn create_stream(&mut self) -> Self::Stream {
        let mut probe = self.probe.lock().unwrap();
        Self::record_thread(&mut probe);
        probe.streams_created += 1;
        ScriptedStream {
            pending: VecDeque::new(),
            current: None,
            probe: self.probe.clone(),
        }
    }

    fn accept_waveform(&mut self, stream: &mut Self::Stream, samples: &[f32]) {
        let is_padding =
            matches!(samples.len(), 16_000 | 4_800) && samples.iter().all(|sample| *sample == 0.0);
        let mut probe = self.probe.lock().unwrap();
        Self::record_thread(&mut probe);
        probe.accepted.push((
            samples.as_ptr() as usize,
            samples.len(),
            samples.iter().all(|sample| *sample == 0.0),
        ));
        drop(probe);
        if !is_padding {
            stream
                .pending
                .extend(self.feed_steps.pop_front().unwrap_or_default());
        }
    }

    fn input_finished(&mut self, stream: &mut Self::Stream) {
        let mut probe = self.probe.lock().unwrap();
        Self::record_thread(&mut probe);
        probe.input_finished += 1;
        drop(probe);
        stream.pending.extend(std::mem::take(&mut self.end_steps));
    }

    fn is_ready(&mut self, stream: &mut Self::Stream) -> bool {
        let mut probe = self.probe.lock().unwrap();
        Self::record_thread(&mut probe);
        probe.readiness_checks += 1;
        !stream.pending.is_empty()
    }

    fn decode(&mut self, stream: &mut Self::Stream) {
        let mut probe = self.probe.lock().unwrap();
        Self::record_thread(&mut probe);
        probe.decodes += 1;
        drop(probe);
        stream.current = stream.pending.pop_front().expect("ready decode step");
    }

    fn get_result(&mut self, stream: &mut Self::Stream) -> Option<String> {
        let mut probe = self.probe.lock().unwrap();
        Self::record_thread(&mut probe);
        probe.result_reads += 1;
        stream.current.clone()
    }
}

fn partial(text: &str) -> Vec<SttEvent> {
    vec![SttEvent::Partial {
        text: text.into(),
        stable: 0,
    }]
}

#[test]
fn streams_changed_unstable_hypotheses_and_a_final_result() {
    block_on(async {
        let (backend, probe) = ScriptedBackend::new(
            [
                vec![Some("hel")],
                vec![Some("hello")],
                vec![Some("hello")],
                vec![Some("")],
            ],
            [Some("hello world")],
        );
        let transcriber = SherpaStt::with_backend(backend).unwrap();
        assert_eq!(transcriber.input_format(), AudioFormat::new(16_000, 1));

        transcriber.begin_utterance().await.unwrap();

        let first: Arc<[f32]> = Arc::from(vec![0.1; 512]);
        let first_pointer = first.as_ptr() as usize;
        assert_eq!(transcriber.feed(first).await.unwrap(), partial("hel"));
        assert_eq!(
            transcriber.feed(Arc::from(vec![0.2; 512])).await.unwrap(),
            partial("hello")
        );
        assert!(
            transcriber
                .feed(Arc::from(vec![0.3; 512]))
                .await
                .unwrap()
                .is_empty(),
            "an unchanged hypothesis must not be repeated"
        );
        assert_eq!(
            transcriber.feed(Arc::from(vec![0.4; 512])).await.unwrap(),
            partial("")
        );
        assert_eq!(
            transcriber.end_utterance().await.unwrap(),
            vec![SttEvent::Final("hello world".into())]
        );

        let probe = probe.lock().unwrap();
        assert_eq!(probe.streams_created, 1);
        assert_eq!(probe.streams_dropped, 1);
        assert_eq!(probe.input_finished, 1);
        assert_eq!(probe.decodes, 5);
        assert_eq!(probe.accepted[0].1, 16_000);
        assert!(probe.accepted[0].2);
        assert_eq!(probe.accepted[1], (first_pointer, 512, false));
        let (_, padding_len, padding_is_silent) = probe.accepted.last().unwrap();
        assert_eq!(*padding_len, 4_800);
        assert!(*padding_is_silent);
    });
}

#[test]
fn drains_every_ready_step_and_uses_the_latest_end_result() {
    block_on(async {
        let (backend, probe) = ScriptedBackend::new(
            [vec![Some("a"), Some("ab"), Some("abc")]],
            [Some("abcd"), Some("abcde")],
        );
        let transcriber = SherpaStt::with_backend(backend).unwrap();

        transcriber.begin_utterance().await.unwrap();
        assert_eq!(
            transcriber.feed(Arc::from(vec![0.0; 512])).await.unwrap(),
            partial("abc")
        );
        assert_eq!(
            transcriber.end_utterance().await.unwrap(),
            vec![SttEvent::Final("abcde".into())]
        );

        assert_eq!(probe.lock().unwrap().decodes, 5);
    });
}

#[test]
fn empty_engine_result_still_finishes_the_utterance() {
    block_on(async {
        let (backend, _) = ScriptedBackend::new([Vec::new()], []);
        let transcriber = SherpaStt::with_backend(backend).unwrap();

        transcriber.begin_utterance().await.unwrap();
        assert!(
            transcriber
                .feed(Arc::from(Vec::new()))
                .await
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            transcriber.end_utterance().await.unwrap(),
            vec![SttEvent::Final("".into())]
        );
    });
}

#[test]
fn protocol_violations_are_reported_and_a_completed_stream_can_restart() {
    block_on(async {
        let (backend, probe) = ScriptedBackend::new([Vec::new()], []);
        let transcriber = SherpaStt::with_backend(backend).unwrap();

        assert!(
            transcriber
                .feed(Arc::from(Vec::new()))
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
        transcriber.end_utterance().await.unwrap();
        transcriber.begin_utterance().await.unwrap();
        transcriber.cancel();

        drop(transcriber);
        let probe = probe.lock().unwrap();
        assert_eq!(probe.streams_created, 2);
        assert_eq!(probe.streams_dropped, 2);
    });
}

#[test]
fn every_backend_operation_runs_on_one_actor_thread() {
    block_on(async {
        let caller = std::thread::current().id();
        let (backend, probe) = ScriptedBackend::new([vec![Some("one")]], [Some("done")]);
        let transcriber = SherpaStt::with_backend(backend).unwrap();

        transcriber.begin_utterance().await.unwrap();
        transcriber.feed(Arc::from(vec![0.0; 512])).await.unwrap();
        transcriber.end_utterance().await.unwrap();

        let probe = probe.lock().unwrap();
        let actor = *probe.threads.first().expect("backend was called");
        assert_ne!(actor, caller);
        assert!(probe.threads.iter().all(|thread| *thread == actor));
    });
}

#[derive(Default)]
struct BlockingProbe {
    streams_created: usize,
    streams_dropped: usize,
    accepted_lengths: Vec<usize>,
    decodes: usize,
}

struct BlockingStream {
    ready: bool,
    text: String,
    probe: Arc<Mutex<BlockingProbe>>,
}

impl Drop for BlockingStream {
    fn drop(&mut self) {
        self.probe.lock().unwrap().streams_dropped += 1;
    }
}

struct BlockingBackend {
    reached: mpsc::Sender<()>,
    release: mpsc::Receiver<()>,
    block_next_decode: bool,
    probe: Arc<Mutex<BlockingProbe>>,
}

impl Backend for BlockingBackend {
    type Stream = BlockingStream;

    fn create_stream(&mut self) -> Self::Stream {
        self.probe.lock().unwrap().streams_created += 1;
        BlockingStream {
            ready: false,
            text: String::new(),
            probe: self.probe.clone(),
        }
    }

    fn accept_waveform(&mut self, stream: &mut Self::Stream, samples: &[f32]) {
        self.probe
            .lock()
            .unwrap()
            .accepted_lengths
            .push(samples.len());
        if samples.len() != 16_000 {
            stream.ready = true;
        }
    }

    fn input_finished(&mut self, _stream: &mut Self::Stream) {}

    fn is_ready(&mut self, stream: &mut Self::Stream) -> bool {
        stream.ready
    }

    fn decode(&mut self, stream: &mut Self::Stream) {
        self.probe.lock().unwrap().decodes += 1;
        if self.block_next_decode {
            self.block_next_decode = false;
            self.reached.send(()).unwrap();
            self.release.recv().unwrap();
        }
        stream.ready = false;
        stream.text = "stale text".into();
    }

    fn get_result(&mut self, stream: &mut Self::Stream) -> Option<String> {
        Some(stream.text.clone())
    }
}

#[test]
fn cancellation_drops_in_flight_and_queued_work_before_a_new_begin() {
    block_on(async {
        let (reached_tx, reached_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let probe = Arc::new(Mutex::new(BlockingProbe::default()));
        let transcriber = SherpaStt::with_backend(BlockingBackend {
            reached: reached_tx,
            release: release_rx,
            block_next_decode: true,
            probe: probe.clone(),
        })
        .unwrap();

        transcriber.begin_utterance().await.unwrap();

        let mut in_flight = Box::pin(transcriber.feed(Arc::from(vec![0.0; 512])));
        assert!(futures::poll!(in_flight.as_mut()).is_pending());
        reached_rx.recv().unwrap();

        let mut queued = Box::pin(transcriber.feed(Arc::from(vec![0.0; 256])));
        assert!(futures::poll!(queued.as_mut()).is_pending());

        transcriber.cancel();

        let mut next_begin = Box::pin(transcriber.begin_utterance());
        assert!(futures::poll!(next_begin.as_mut()).is_pending());
        release_tx.send(()).unwrap();

        assert!(in_flight.await.unwrap().is_empty());
        assert!(queued.await.unwrap().is_empty());
        next_begin.await.unwrap();

        let probe = probe.lock().unwrap();
        assert_eq!(probe.accepted_lengths, vec![16_000, 512, 16_000]);
        assert_eq!(probe.decodes, 1);
        assert_eq!(probe.streams_created, 2);
        assert_eq!(probe.streams_dropped, 1);
    });
}

struct PanickingBackend;

impl Backend for PanickingBackend {
    type Stream = ();

    fn create_stream(&mut self) -> Self::Stream {}

    fn accept_waveform(&mut self, _stream: &mut Self::Stream, samples: &[f32]) {
        if samples.len() != 16_000 {
            panic!("scripted worker failure");
        }
    }

    fn input_finished(&mut self, _stream: &mut Self::Stream) {}

    fn is_ready(&mut self, _stream: &mut Self::Stream) -> bool {
        false
    }

    fn decode(&mut self, _stream: &mut Self::Stream) {}

    fn get_result(&mut self, _stream: &mut Self::Stream) -> Option<String> {
        None
    }
}

#[test]
fn worker_failure_is_reported_as_an_stt_error() {
    block_on(async {
        let transcriber = SherpaStt::with_backend(PanickingBackend).unwrap();
        transcriber.begin_utterance().await.unwrap();
        let error = transcriber
            .feed(Arc::from(vec![0.0; 512]))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("dropped its response"));
    });
}

#[test]
#[ignore = "requires SHERPA_STT_ENCODER, SHERPA_STT_DECODER, SHERPA_STT_JOINER, and SHERPA_STT_TOKENS"]
fn production_backend_transcribes_a_wave_file() {
    block_on(async {
        let config = SherpaSttConfig::new(
            std::env::var("SHERPA_STT_ENCODER").expect("set SHERPA_STT_ENCODER"),
            std::env::var("SHERPA_STT_DECODER").expect("set SHERPA_STT_DECODER"),
            std::env::var("SHERPA_STT_JOINER").expect("set SHERPA_STT_JOINER"),
            std::env::var("SHERPA_STT_TOKENS").expect("set SHERPA_STT_TOKENS"),
        );
        let wave_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../test-resources/audio/sherpa-zipformer-en-20m-0.wav");
        let wave = sherpa_onnx::Wave::read(wave_path.to_str().expect("UTF-8 test resource path"))
            .expect("read test speech resource");
        assert_eq!(wave.sample_rate(), 16_000, "test wave must be 16 kHz");
        let transcriber = SherpaStt::new(config).unwrap();
        transcriber.begin_utterance().await.unwrap();
        for samples in wave.samples().chunks(512) {
            transcriber
                .feed(Arc::from(samples))
                .await
                .expect("feed test wave");
        }
        let events = transcriber.end_utterance().await.unwrap();
        let [SttEvent::Final(text)] = events.as_slice() else {
            panic!("expected one final transcript, got {events:?}");
        };
        assert!(!text.is_empty(), "known speech wave produced no transcript");
        assert!(
            text.starts_with("AFTER EARLY NIGHTFALL"),
            "known speech wave lost its opening words: {text}"
        );
        println!("{text}");
    });
}
