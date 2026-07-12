//! Owns each `!Send` cpal stream on a dedicated thread.
//!
//! The public audio interfaces are `Send`, but `cpal::Stream` is not. The owning
//! thread builds, runs, and drops the stream; the source or sink retains a
//! [`StreamThread`] and a `Send` ring endpoint.

use std::sync::mpsc;
use std::thread;

use pipecrab_audio::AudioError;

/// Owns the thread that keeps a cpal `Stream` alive; dropping it stops the stream.
pub(crate) struct StreamThread {
    /// Dropping the sender unblocks the thread's `recv`, its cue to drop the stream.
    shutdown: Option<mpsc::Sender<()>>,
    handle: Option<thread::JoinHandle<()>>,
}

impl Drop for StreamThread {
    fn drop(&mut self) {
        // Drop the sender *before* joining, or the thread's `recv` never returns.
        drop(self.shutdown.take());
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Builds and retains a cpal stream on a dedicated thread.
///
/// `build` creates the `!Send` objects on that thread and returns a `Send`
/// handle. This function blocks until setup completes. Dropping the returned
/// [`StreamThread`] stops the stream and joins its thread.
pub(crate) fn spawn_stream<T, F>(build: F) -> Result<(T, StreamThread), AudioError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<(cpal::Stream, T), AudioError> + Send + 'static,
{
    let (setup_tx, setup_rx) = mpsc::channel::<Result<T, AudioError>>();
    let (shutdown_tx, shutdown_rx) = mpsc::channel::<()>();

    let handle = thread::Builder::new()
        .name("pipecrab-audio-cpal".into())
        .spawn(move || match build() {
            Ok((stream, handles)) => {
                // Hand the Send end back. If the caller is already gone, there is
                // nothing left to keep the stream alive for.
                if setup_tx.send(Ok(handles)).is_err() {
                    return;
                }
                // Park holding the stream. `recv` returns `Err` once the owning
                // `StreamThread` drops its `shutdown` sender — our cue to stop.
                let _ = shutdown_rx.recv();
                drop(stream);
            }
            Err(e) => {
                let _ = setup_tx.send(Err(e));
            }
        })
        .map_err(|e| AudioError::Device(format!("spawn audio thread: {e}")))?;

    match setup_rx.recv() {
        Ok(Ok(handles)) => Ok((
            handles,
            StreamThread {
                shutdown: Some(shutdown_tx),
                handle: Some(handle),
            },
        )),
        Ok(Err(e)) => {
            let _ = handle.join();
            Err(e)
        }
        // The thread ended (e.g. panicked) before reporting — surface it as a
        // device error rather than leaving the caller to hang.
        Err(_) => {
            let _ = handle.join();
            Err(AudioError::Device(
                "audio thread exited before setup".into(),
            ))
        }
    }
}
