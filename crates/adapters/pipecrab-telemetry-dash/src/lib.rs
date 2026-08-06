//! Live local dashboard for pipecrab telemetry.
//!
//! [`DashboardSink`] serves a single self-contained page on localhost and
//! streams every finished [`TurnRecord`] to it over Server-Sent Events — no
//! collector, no database, no external assets. Open the printed URL while the
//! agent runs and watch latencies, stage spans, barge-ins, and the transcript
//! land turn by turn; a late-joining browser is caught up from an in-memory
//! backlog of recent turns.
//!
//! ```no_run
//! use pipecrab_telemetry::Telemetry;
//! use pipecrab_telemetry_dash::DashboardSink;
//!
//! let telemetry = Telemetry::builder()
//!     .sink(DashboardSink::serve("127.0.0.1:7878").unwrap())
//!     .build();
//! // http://127.0.0.1:7878 now shows the session live.
//! ```
//!
//! The sink runs entirely on its own threads (one listener, one per
//! connection); `record` only serializes and fans out over non-blocking
//! channels, so the telemetry worker never waits on a slow browser — a client
//! that falls more than a channel's depth behind is dropped and can simply
//! reload.
//!
//! Prefer the terminal? The same endpoint feeds the bundled TUI viewer —
//! run it in its own terminal beside the agent:
//!
//! ```text
//! cargo run -p pipecrab-telemetry-dash --bin pipecrab-dash-tui
//! ```
#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex};

use pipecrab_telemetry::{SinkError, TelemetrySink, TurnRecord};

/// The dashboard page, embedded so the sink serves with zero external assets.
const PAGE: &str = include_str!("dashboard.html");

/// Turns kept for catching up a late-joining browser.
const BACKLOG: usize = 512;

/// Per-client channel depth; a browser this far behind is dropped.
const CLIENT_DEPTH: usize = 256;

/// Comment-ping cadence on an idle SSE stream.
const KEEPALIVE: std::time::Duration = std::time::Duration::from_secs(15);

struct Shared {
    backlog: Mutex<VecDeque<Arc<str>>>,
    clients: Mutex<Vec<SyncSender<Arc<str>>>>,
    shutdown: AtomicBool,
}

/// A [`TelemetrySink`] that serves the live dashboard; see the crate docs.
pub struct DashboardSink {
    shared: Arc<Shared>,
    addr: SocketAddr,
}

impl DashboardSink {
    /// Bind `addr` (e.g. `"127.0.0.1:7878"`, or port 0 for an ephemeral one)
    /// and start serving. The bound address is available via
    /// [`addr`](Self::addr).
    pub fn serve(addr: impl ToSocketAddrs) -> std::io::Result<Self> {
        let listener = TcpListener::bind(addr)?;
        let addr = listener.local_addr()?;
        let shared = Arc::new(Shared {
            backlog: Mutex::new(VecDeque::new()),
            clients: Mutex::new(Vec::new()),
            shutdown: AtomicBool::new(false),
        });
        let accept_shared = shared.clone();
        std::thread::Builder::new()
            .name("pipecrab-dash".into())
            .spawn(move || {
                for stream in listener.incoming() {
                    if accept_shared.shutdown.load(Ordering::Relaxed) {
                        break;
                    }
                    let Ok(stream) = stream else { continue };
                    let conn_shared = accept_shared.clone();
                    let _ = std::thread::Builder::new()
                        .name("pipecrab-dash-conn".into())
                        .spawn(move || serve_connection(stream, &conn_shared));
                }
            })?;
        Ok(Self { shared, addr })
    }

    /// The address the dashboard is being served on.
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// The dashboard's URL.
    pub fn url(&self) -> String {
        format!("http://{}", self.addr)
    }
}

impl TelemetrySink for DashboardSink {
    fn record(&mut self, turn: &TurnRecord) -> Result<(), SinkError> {
        let line: Arc<str> = Arc::from(
            serde_json::to_string(turn).map_err(|e| SinkError::new(format!("encode: {e}")))?,
        );
        {
            let mut backlog = self.shared.backlog.lock().unwrap();
            if backlog.len() == BACKLOG {
                backlog.pop_front();
            }
            backlog.push_back(line.clone());
        }
        // Fan out without blocking: a full or hung-up client is dropped.
        self.shared.clients.lock().unwrap().retain(|client| {
            !matches!(
                client.try_send(line.clone()),
                Err(TrySendError::Disconnected(_)) | Err(TrySendError::Full(_))
            )
        });
        Ok(())
    }
}

impl Drop for DashboardSink {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Relaxed);
        // Senders drop with the clients vec, ending every SSE stream; a
        // throwaway connection unblocks the accept loop so it can observe
        // the flag and exit.
        self.shared.clients.lock().unwrap().clear();
        let _ = TcpStream::connect(self.addr);
    }
}

fn serve_connection(stream: TcpStream, shared: &Shared) {
    let mut reader = BufReader::new(match stream.try_clone() {
        Ok(clone) => clone,
        Err(_) => return,
    });
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return;
    }
    // Drain the headers; this server routes on the request line alone.
    let mut header = String::new();
    while reader.read_line(&mut header).is_ok() && header.trim() != "" {
        header.clear();
    }
    let path = request_line.split_whitespace().nth(1).unwrap_or("/");
    match path {
        "/" | "/index.html" => respond(stream, "200 OK", "text/html; charset=utf-8", PAGE),
        "/events" => serve_events(stream, shared),
        _ => respond(stream, "404 Not Found", "text/plain", "not found"),
    }
}

fn respond(mut stream: TcpStream, status: &str, content_type: &str, body: &str) {
    let _ = write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len(),
    );
}

/// Stream the backlog, then every new record, as SSE `data:` events until the
/// client hangs up or the sink is dropped (which drops our sender).
fn serve_events(mut stream: TcpStream, shared: &Shared) {
    if write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n\r\n"
    )
    .is_err()
    {
        return;
    }
    // Body bytes straight away: browsers only commit an EventSource's `open`
    // once the stream produces something, so an empty session would otherwise
    // sit at "connecting". The retry field tunes reconnect while we're at it.
    if write!(stream, "retry: 2000\n\n: connected\n\n").is_err() {
        return;
    }
    let rx: Receiver<Arc<str>> = {
        // Register before snapshotting the backlog so no record can fall
        // between them; the channel buffers anything recorded meanwhile.
        let (tx, rx) = sync_channel(CLIENT_DEPTH);
        shared.clients.lock().unwrap().push(tx);
        rx
    };
    let backlog: Vec<Arc<str>> = shared.backlog.lock().unwrap().iter().cloned().collect();
    let mut seen = std::collections::HashSet::new();
    for line in &backlog {
        seen.insert(Arc::as_ptr(line).cast::<u8>() as usize);
        if write!(stream, "data: {line}\n\n").is_err() {
            return;
        }
    }
    loop {
        let line = match rx.recv_timeout(KEEPALIVE) {
            Ok(line) => line,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                // Comment ping: keeps intermediaries from timing the stream
                // out and detects a gone client between turns.
                if write!(stream, ": ping\n\n").is_err() {
                    return;
                }
                continue;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return,
        };
        // A record fanned out between registration and the snapshot appears
        // in both; skip the copy already sent.
        if seen
            .take(&(Arc::as_ptr(&line).cast::<u8>() as usize))
            .is_some()
        {
            continue;
        }
        if write!(stream, "data: {line}\n\n").is_err() {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::sync::Arc;

    use pipecrab_telemetry::{SessionInfo, TurnEnd, TurnOrigin, TurnTimings};

    fn record(turn: u64) -> TurnRecord {
        TurnRecord {
            session: SessionInfo {
                id: Arc::from("dash"),
                started_unix_ms: 0,
            },
            turn,
            origin: TurnOrigin::Speech,
            end: TurnEnd::NextTurn,
            started_ms: 0.0,
            ended_ms: 1_000.0,
            user_text: Some(Arc::from("hi")),
            agent_text: Some(Arc::from("hello")),
            tool_calls: Vec::new(),
            interrupted_at_ms: None,
            timings: TurnTimings::default(),
            stages: Vec::new(),
            errors: Vec::new(),
            lost_events: 0,
        }
    }

    fn get(addr: SocketAddr, path: &str, read_until: &str) -> String {
        let mut stream = TcpStream::connect(addr).unwrap();
        write!(stream, "GET {path} HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
        let mut out = String::new();
        let mut buf = [0u8; 4096];
        while !out.contains(read_until) {
            match stream.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => out.push_str(&String::from_utf8_lossy(&buf[..n])),
                Err(_) => break,
            }
        }
        out
    }

    #[test]
    fn serves_the_page_and_streams_backlog_and_live_records() {
        let mut sink = DashboardSink::serve("127.0.0.1:0").unwrap();
        let addr = sink.addr();

        let page = get(addr, "/", "</html>");
        assert!(page.starts_with("HTTP/1.1 200"), "got: {page:.100}");
        assert!(page.contains("pipecrab"), "page must be the dashboard");

        // A record written before the client connects arrives as backlog.
        sink.record(&record(0)).unwrap();
        let events = get(addr, "/events", "\"turn\":0");
        assert!(events.contains("text/event-stream"));
        assert!(events.contains("data: {"));

        // A live client receives records recorded after it connected.
        let mut stream = TcpStream::connect(addr).unwrap();
        write!(stream, "GET /events HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
        let mut header = [0u8; 512];
        let _ = stream.read(&mut header).unwrap();
        sink.record(&record(1)).unwrap();
        let mut out = String::new();
        let mut buf = [0u8; 4096];
        while !out.contains("\"turn\":1") {
            let n = stream.read(&mut buf).unwrap();
            assert!(n > 0, "stream ended before the live record arrived");
            out.push_str(&String::from_utf8_lossy(&buf[..n]));
        }
    }
}
