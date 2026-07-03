//! Keeping a `!Send` cpal `Stream` alive on its own thread.
//!
//! The audio seam is `MaybeSend`, so a [`CpalSource`](crate::CpalSource) /
//! [`CpalSink`](crate::CpalSink) must be `Send` — that is what lets a server
//! spawn one capture/playback pump per session. But `cpal::Stream` is `!Send`,
//! so it cannot be a field. Instead a dedicated thread builds the stream, starts
//! it, and parks holding it alive; the constructor keeps only the ring end
//! (which *is* `Send`) plus a [`StreamThread`] handle. Dropping the handle tells
//! the thread to drop the stream and joins it.

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

/// Build a cpal stream on a dedicated thread and keep it alive there.
///
/// `build` runs on the new thread — cpal's `Host`/`Device`/`Stream` are `!Send`
/// and so must be created (and later dropped) there, never moved across the
/// boundary. It returns the started stream paired with a `Send` handle (`T`: the
/// ring end plus device name) to hand back to the caller. This function blocks
/// until `build` reports success or failure, so construction is still
/// synchronous from the caller's view; on success the stream then stays parked
/// on the thread until the returned [`StreamThread`] is dropped.
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
        Ok(Ok(handles)) => {
            Ok((handles, StreamThread { shutdown: Some(shutdown_tx), handle: Some(handle) }))
        }
        Ok(Err(e)) => {
            let _ = handle.join();
            Err(e)
        }
        // The thread ended (e.g. panicked) before reporting — surface it as a
        // device error rather than leaving the caller to hang.
        Err(_) => {
            let _ = handle.join();
            Err(AudioError::Device("audio thread exited before setup".into()))
        }
    }
}
