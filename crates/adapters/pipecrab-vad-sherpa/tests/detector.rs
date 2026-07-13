use std::collections::VecDeque;
use std::sync::{Arc, Mutex, mpsc};
use std::thread::ThreadId;

use futures::executor::block_on;
use pipecrab_core::AudioFormat;
use pipecrab_vad::{VadEvent, VoiceActivityDetector};
use pipecrab_vad_sherpa::{Backend, SherpaVad, SherpaVadConfig};

#[derive(Default)]
struct Probe {
    accepted_lengths: Vec<usize>,
    transitions: VecDeque<bool>,
    detected: bool,
    queued_segments: usize,
    pops: usize,
    resets: usize,
    threads: Vec<ThreadId>,
}

struct ScriptedBackend {
    probe: Arc<Mutex<Probe>>,
}

impl ScriptedBackend {
    fn new(transitions: impl IntoIterator<Item = bool>) -> (Self, Arc<Mutex<Probe>>) {
        let probe = Arc::new(Mutex::new(Probe {
            transitions: transitions.into_iter().collect(),
            ..Probe::default()
        }));
        (
            Self {
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
    fn detected(&mut self) -> bool {
        let mut probe = self.probe.lock().unwrap();
        Self::record_thread(&mut probe);
        probe.detected
    }

    fn accept_waveform(&mut self, samples: &[f32]) {
        let mut probe = self.probe.lock().unwrap();
        Self::record_thread(&mut probe);
        probe.accepted_lengths.push(samples.len());
        if let Some(next) = probe.transitions.pop_front() {
            if probe.detected && !next {
                probe.queued_segments += 1;
            }
            probe.detected = next;
        }
    }

    fn is_empty(&mut self) -> bool {
        let mut probe = self.probe.lock().unwrap();
        Self::record_thread(&mut probe);
        probe.queued_segments == 0
    }

    fn pop(&mut self) {
        let mut probe = self.probe.lock().unwrap();
        Self::record_thread(&mut probe);
        probe.queued_segments -= 1;
        probe.pops += 1;
    }

    fn reset(&mut self) {
        let mut probe = self.probe.lock().unwrap();
        Self::record_thread(&mut probe);
        probe.detected = false;
        probe.queued_segments = 0;
        probe.resets += 1;
    }
}

#[test]
fn windows_arbitrary_input_and_returns_every_transition() {
    block_on(async {
        let (backend, probe) = ScriptedBackend::new([true, false, true]);
        let detector = SherpaVad::with_backend(backend).unwrap();

        assert!(
            detector
                .process(Arc::from(vec![0.0; 300]))
                .await
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            detector.process(Arc::from(vec![0.0; 212])).await.unwrap(),
            vec![VadEvent::SpeechStarted]
        );
        assert_eq!(
            detector.process(Arc::from(vec![0.0; 1024])).await.unwrap(),
            vec![VadEvent::SpeechStopped, VadEvent::SpeechStarted]
        );

        let probe = probe.lock().unwrap();
        assert_eq!(probe.accepted_lengths, vec![512, 512, 512]);
        assert_eq!(probe.pops, 1, "completed Sherpa segments are discarded");
        assert_eq!(probe.queued_segments, 0);
    });
}

#[test]
fn reset_clears_the_partial_window_and_backend_state() {
    block_on(async {
        let (backend, probe) = ScriptedBackend::new([true]);
        let detector = SherpaVad::with_backend(backend).unwrap();

        detector.process(Arc::from(vec![0.0; 300])).await.unwrap();
        detector.reset();

        assert!(
            detector
                .process(Arc::from(vec![0.0; 212]))
                .await
                .unwrap()
                .is_empty()
        );
        assert!(probe.lock().unwrap().accepted_lengths.is_empty());

        assert_eq!(
            detector.process(Arc::from(vec![0.0; 300])).await.unwrap(),
            vec![VadEvent::SpeechStarted]
        );
        let probe = probe.lock().unwrap();
        assert_eq!(probe.accepted_lengths, vec![512]);
        assert_eq!(probe.resets, 1);
    });
}

#[test]
fn every_backend_operation_runs_on_one_actor_thread() {
    block_on(async {
        let caller = std::thread::current().id();
        let (backend, probe) = ScriptedBackend::new([true, false]);
        let detector = SherpaVad::with_backend(backend).unwrap();
        detector.process(Arc::from(vec![0.0; 1024])).await.unwrap();
        detector.reset();
        detector.process(Arc::from(Vec::new())).await.unwrap();

        let probe = probe.lock().unwrap();
        let worker = *probe.threads.first().expect("backend was called");
        assert_ne!(worker, caller);
        assert!(probe.threads.iter().all(|thread| *thread == worker));
    });
}

struct BlockingBackend {
    reached: mpsc::Sender<()>,
    release: mpsc::Receiver<()>,
    probe: Arc<Mutex<Probe>>,
    detected: bool,
}

impl Backend for BlockingBackend {
    fn detected(&mut self) -> bool {
        self.detected
    }

    fn accept_waveform(&mut self, _samples: &[f32]) {
        self.reached.send(()).unwrap();
        self.release.recv().unwrap();
        self.detected = true;
    }

    fn is_empty(&mut self) -> bool {
        true
    }

    fn pop(&mut self) {}

    fn reset(&mut self) {
        self.detected = false;
        self.probe.lock().unwrap().resets += 1;
    }
}

#[test]
fn reset_during_a_command_discards_its_events_and_state() {
    let (reached_tx, reached_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let probe = Arc::new(Mutex::new(Probe::default()));
    let detector = Arc::new(
        SherpaVad::with_backend(BlockingBackend {
            reached: reached_tx,
            release: release_rx,
            probe: probe.clone(),
            detected: false,
        })
        .unwrap(),
    );

    let processing = {
        let detector = detector.clone();
        std::thread::spawn(move || block_on(detector.process(Arc::from(vec![0.0; 512]))).unwrap())
    };
    reached_rx.recv().unwrap();
    detector.reset();
    release_tx.send(()).unwrap();

    assert!(processing.join().unwrap().is_empty());
    assert_eq!(probe.lock().unwrap().resets, 1);
}

struct PanickingBackend;

impl Backend for PanickingBackend {
    fn detected(&mut self) -> bool {
        false
    }

    fn accept_waveform(&mut self, _samples: &[f32]) {
        panic!("scripted worker failure");
    }

    fn is_empty(&mut self) -> bool {
        true
    }

    fn pop(&mut self) {}

    fn reset(&mut self) {}
}

#[test]
fn worker_failure_is_reported_as_a_vad_error() {
    block_on(async {
        let detector = SherpaVad::with_backend(PanickingBackend).unwrap();
        let error = detector
            .process(Arc::from(vec![0.0; 512]))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("dropped its response"));
    });
}

#[test]
fn declares_silero_input_format() {
    let (backend, _) = ScriptedBackend::new([]);
    let detector = SherpaVad::with_backend(backend).unwrap();
    assert_eq!(detector.input_format(), AudioFormat::new(16_000, 1));
}

#[test]
#[ignore = "requires SHERPA_VAD_MODEL to point to a Silero ONNX model"]
fn production_backend_processes_one_silero_window() {
    block_on(async {
        let model = std::env::var("SHERPA_VAD_MODEL").expect("set SHERPA_VAD_MODEL");
        let detector = SherpaVad::new(SherpaVadConfig::new(model)).unwrap();
        let events = detector.process(Arc::from(vec![0.0; 512])).await.unwrap();
        assert!(events.is_empty(), "silence should not start speech");
    });
}
