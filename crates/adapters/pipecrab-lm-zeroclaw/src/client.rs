//! One live connection to the daemon: a written half the actor sends requests
//! on, and a reader task that classifies every incoming line into
//! [`Incoming`] on an internal channel.
//!
//! The reader task exists so the actor can `select!` socket traffic against
//! its command and cancel channels without owning a split read half — and so
//! a dropped connection surfaces as one final [`Incoming::Closed`] rather
//! than an error the actor must remember to poll for.

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::net::unix::OwnedWriteHalf;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::protocol::{RpcErrorObject, SessionUpdate, method};

/// One classified incoming message.
#[derive(Debug)]
pub(crate) enum Incoming {
    /// A response to a request this client sent, matched later by id.
    Response {
        id: u64,
        result: Result<Value, RpcErrorObject>,
    },
    /// A `session/update` notification.
    Update(SessionUpdate),
    /// The connection ended: EOF, an I/O error, or an unparseable stream.
    Closed(String),
}

/// A live daemon connection: the write half plus the reader task feeding
/// [`Incoming`] messages. Dropping it aborts the reader and closes the socket.
pub(crate) struct Connection {
    writer: OwnedWriteHalf,
    pub(crate) incoming: mpsc::UnboundedReceiver<Incoming>,
    reader: JoinHandle<()>,
    next_id: u64,
}

impl Drop for Connection {
    fn drop(&mut self) {
        self.reader.abort();
    }
}

impl Connection {
    /// Dial the daemon socket and start the reader task. Must be called from
    /// within a tokio runtime (the worker's).
    pub(crate) async fn dial(path: &std::path::Path) -> std::io::Result<Self> {
        let stream = UnixStream::connect(path).await?;
        let (read_half, writer) = stream.into_split();
        let (incoming_tx, incoming) = mpsc::unbounded_channel();

        let reader = tokio::spawn(async move {
            let mut lines = BufReader::new(read_half).lines();
            loop {
                match lines.next_line().await {
                    Ok(Some(line)) => {
                        let line = line.trim();
                        if line.is_empty() {
                            continue;
                        }
                        if let Some(message) = classify(line)
                            && incoming_tx.send(message).is_err()
                        {
                            return; // the actor is gone
                        }
                    }
                    Ok(None) => {
                        let _ = incoming_tx
                            .send(Incoming::Closed("daemon closed the connection".into()));
                        return;
                    }
                    Err(error) => {
                        let _ = incoming_tx
                            .send(Incoming::Closed(format!("socket read failed: {error}")));
                        return;
                    }
                }
            }
        });

        Ok(Self {
            writer,
            incoming,
            reader,
            next_id: 0,
        })
    }

    /// Send one JSON-RPC request; returns the id to match its response by.
    pub(crate) async fn send_request(
        &mut self,
        method: &str,
        params: impl serde::Serialize,
    ) -> std::io::Result<u64> {
        self.next_id += 1;
        let id = self.next_id;
        let request = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": serde_json::to_value(params).map_err(std::io::Error::other)?,
        });
        let mut line = request.to_string();
        line.push('\n');
        self.writer.write_all(line.as_bytes()).await?;
        Ok(id)
    }

    /// Await the response to request `id`, discarding session updates in the
    /// meantime. Handshake-only: during a turn the actor routes updates
    /// itself and never uses this.
    pub(crate) async fn await_response(&mut self, id: u64) -> Result<Value, String> {
        loop {
            match self.incoming.recv().await {
                Some(Incoming::Response { id: got, result }) if got == id => {
                    return result.map_err(|error| error.to_string());
                }
                Some(Incoming::Response { .. }) | Some(Incoming::Update(_)) => continue,
                Some(Incoming::Closed(reason)) => return Err(reason),
                None => return Err("connection reader ended".into()),
            }
        }
    }
}

/// Classify one line of daemon output. Lines that are neither a response nor
/// a `session/update` (other notifications, malformed JSON) yield `None` and
/// are ignored — an unknown broadcast must never kill the connection.
fn classify(line: &str) -> Option<Incoming> {
    let value: Value = serde_json::from_str(line).ok()?;

    if let Some(method_name) = value.get("method").and_then(Value::as_str) {
        if method_name == method::SESSION_UPDATE {
            let params = value.get("params").cloned().unwrap_or(Value::Null);
            return Some(Incoming::Update(SessionUpdate::from_params(&params)));
        }
        return None;
    }

    let id = value.get("id").and_then(Value::as_u64)?;
    if let Some(error) = value.get("error") {
        let error: RpcErrorObject =
            serde_json::from_value(error.clone()).unwrap_or(RpcErrorObject {
                code: 0,
                message: "unparseable error object".into(),
            });
        return Some(Incoming::Response {
            id,
            result: Err(error),
        });
    }
    let result = value.get("result").cloned().unwrap_or(Value::Null);
    Some(Incoming::Response {
        id,
        result: Ok(result),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_responses_updates_and_noise() {
        let ok = classify(r#"{"jsonrpc":"2.0","id":3,"result":{"x":1}}"#).unwrap();
        assert!(matches!(
            ok,
            Incoming::Response {
                id: 3,
                result: Ok(_)
            }
        ));

        let err = classify(r#"{"jsonrpc":"2.0","id":4,"error":{"code":-32000,"message":"gone"}}"#)
            .unwrap();
        let Incoming::Response {
            id: 4,
            result: Err(error),
        } = err
        else {
            panic!("expected error response, got {err:?}");
        };
        assert_eq!(error.code, -32000);

        let update = classify(
            r#"{"jsonrpc":"2.0","method":"session/update","params":{"type":"plan","session_id":"s"}}"#,
        )
        .unwrap();
        assert!(matches!(
            update,
            Incoming::Update(SessionUpdate::Plan { .. })
        ));

        assert!(classify(r#"{"jsonrpc":"2.0","method":"logs/event","params":{}}"#).is_none());
        assert!(classify("not json at all").is_none());
    }
}
