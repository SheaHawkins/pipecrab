//! Terminal dashboard for pipecrab telemetry.
//!
//! Connects to a running [`DashboardSink`](pipecrab_telemetry_dash::DashboardSink)'s
//! SSE endpoint and renders the same live view in the terminal: KPIs, latency
//! sparklines, per-stage busy time, and the recent turns. Run it in its own
//! terminal beside the agent — the agent keeps its stdout, the dashboard keeps
//! this one:
//!
//! ```text
//! cargo run -p pipecrab-telemetry-dash --bin pipecrab-dash-tui            # 127.0.0.1:7878
//! cargo run -p pipecrab-telemetry-dash --bin pipecrab-dash-tui -- <addr>
//! ```
//!
//! Rendering is plain in-place ANSI (no raw mode, no alternate screen), so
//! Ctrl-C exits cleanly and the last frame stays on the scroll-back. The
//! viewer reconnects while the agent is down and catches up from the sink's
//! backlog when it returns.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::time::Duration;

use pipecrab_telemetry::{TurnEnd, TurnOrigin, TurnRecord};

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const BLUE: &str = "\x1b[34m"; // response latency · decide
const YELLOW: &str = "\x1b[33m"; // time to first speech · perform
const GREEN: &str = "\x1b[32m"; // LM first token · live dot
const RED: &str = "\x1b[31m"; // interrupted · errors

/// Turns kept for the charts and feed.
const KEEP: usize = 512;

fn main() {
    let addr = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "127.0.0.1:7878".to_string());
    let mut app = App::default();
    let mut painter = Painter::default();
    loop {
        app.connected = false;
        painter.paint(&app, &addr);
        match stream_events(&addr, &mut app, &mut painter) {
            Ok(()) | Err(_) => {
                app.connected = false;
                painter.paint(&app, &addr);
                std::thread::sleep(Duration::from_secs(2));
            }
        }
    }
}

#[derive(Default)]
struct App {
    turns: Vec<TurnRecord>,
    connected: bool,
    lost: u64,
}

/// Connect, replay the backlog, then follow the live stream; returns when the
/// connection drops.
fn stream_events(addr: &str, app: &mut App, painter: &mut Painter) -> std::io::Result<()> {
    let stream = TcpStream::connect(addr)?;
    let mut out = stream.try_clone()?;
    write!(
        out,
        "GET /events HTTP/1.1\r\nHost: {addr}\r\nAccept: text/event-stream\r\n\r\n"
    )?;
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    // Status line + headers.
    reader.read_line(&mut line)?;
    if !line.contains("200") {
        return Err(std::io::Error::other(format!("bad response: {line}")));
    }
    while {
        line.clear();
        reader.read_line(&mut line)? > 0 && line.trim() != ""
    } {}
    // A fresh connection replays the backlog; drop stale local state so a
    // restarted agent (new session) is not mixed with the old one.
    app.turns.clear();
    app.lost = 0;
    app.connected = true;
    painter.paint(app, addr);
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Ok(()); // server closed
        }
        if let Some(json) = line.trim_end().strip_prefix("data: ") {
            match serde_json::from_str::<TurnRecord>(json) {
                Ok(turn) => {
                    app.lost += turn.lost_events;
                    app.turns.push(turn);
                    if app.turns.len() > KEEP {
                        app.turns.remove(0);
                    }
                    painter.paint(app, addr);
                }
                Err(error) => eprintln!("pipecrab-dash-tui: bad record: {error}"),
            }
        }
        // Comment keepalives and blank event delimiters need no action.
    }
}

/// In-place frame painter: rewinds over the previous frame and repaints, so
/// the dashboard updates without raw mode or an alternate screen.
#[derive(Default)]
struct Painter {
    painted_lines: usize,
}

impl Painter {
    fn paint(&mut self, app: &App, addr: &str) {
        let width = std::env::var("COLUMNS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(110)
            .clamp(60, 200);
        let frame = render(app, addr, width);
        let mut out = std::io::stdout().lock();
        if self.painted_lines > 0 {
            // Rewind to the top of the previous frame.
            let _ = write!(out, "\x1b[{}F", self.painted_lines);
        }
        for line in &frame {
            let _ = writeln!(out, "\x1b[2K{line}");
        }
        // A shrinking frame leaves stale lines below; clear them and return.
        let extra = self.painted_lines.saturating_sub(frame.len());
        if extra > 0 {
            for _ in 0..extra {
                let _ = writeln!(out, "\x1b[2K");
            }
            let _ = write!(out, "\x1b[{extra}F");
        }
        let _ = out.flush();
        self.painted_lines = frame.len();
    }
}

fn render(app: &App, addr: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let status = if app.connected {
        format!("{GREEN}●{RESET} live")
    } else {
        format!("{DIM}○ connecting to {addr}…{RESET}")
    };
    let session = app
        .turns
        .last()
        .map(|t| format!(" · session {}", t.session.id))
        .unwrap_or_default();
    lines.push(format!(
        "{BOLD}pipecrab telemetry{RESET}{DIM}{session}{RESET}  {status}"
    ));
    lines.push(String::new());

    if app.turns.is_empty() {
        lines.push(format!("{DIM}waiting for the first turn…{RESET}"));
        return lines;
    }

    // --- KPIs ---
    let recent: Vec<&TurnRecord> = app.turns.iter().rev().take(20).collect();
    let med = |f: &dyn Fn(&TurnRecord) -> Option<f64>| median(recent.iter().filter_map(|t| f(t)));
    let interrupted: Vec<&TurnRecord> = app
        .turns
        .iter()
        .filter(|t| t.end == TurnEnd::Interrupted)
        .collect();
    let tools: usize = app.turns.iter().map(|t| t.tool_calls.len()).sum();
    lines.push(format!(
        "turns {BOLD}{}{RESET} · response {BOLD}{}{RESET} · first speech {BOLD}{}{RESET} · \
         LM first token {BOLD}{}{RESET} · barge-ins {BOLD}{}{RESET} · tools {BOLD}{tools}{RESET}",
        app.turns.len(),
        fmt_ms(med(&|t| t.timings.response_latency_ms)),
        fmt_ms(med(&|t| t.timings.time_to_first_speech_ms)),
        fmt_ms(med(&|t| t.timings.lm_ttft_ms)),
        interrupted.len(),
    ));
    if app.lost > 0 {
        lines.push(format!(
            "{YELLOW}⚠ {} telemetry events dropped under load — some records may be incomplete{RESET}",
            app.lost
        ));
    }
    lines.push(String::new());

    // --- sparklines: medians in the KPI row above, per-turn shape here ---
    type Metric<'a> = (&'a str, &'a str, &'a dyn Fn(&TurnRecord) -> Option<f64>);
    let spark_w = width.saturating_sub(34).min(60);
    let series: [Metric; 3] = [
        ("response", BLUE, &|t| t.timings.response_latency_ms),
        ("first speech", YELLOW, &|t| {
            t.timings.time_to_first_speech_ms
        }),
        ("LM first token", GREEN, &|t| t.timings.lm_ttft_ms),
    ];
    let window: Vec<&TurnRecord> = app.turns.iter().rev().take(spark_w).rev().collect();
    let max = window
        .iter()
        .flat_map(|t| series.iter().filter_map(|(_, _, f)| f(t)))
        .fold(1.0_f64, f64::max);
    for (name, color, f) in series {
        let values: Vec<Option<f64>> = window.iter().map(|t| f(t)).collect();
        let latest = values.iter().rev().flatten().next().copied();
        lines.push(format!(
            "{name:>14}  {color}{}{RESET} {}",
            sparkline(&values, max),
            fmt_ms(latest),
        ));
    }
    lines.push(String::new());

    // --- per-stage busy time (whole session) ---
    let mut stages: std::collections::HashMap<&str, (String, f64, f64)> =
        std::collections::HashMap::new();
    for turn in &app.turns {
        for stage in &turn.stages {
            let entry = stages
                .entry(&stage.path)
                .or_insert_with(|| (shorten(&stage.name), 0.0, 0.0));
            entry.1 += stage.decide_ms;
            entry.2 += stage.perform_ms;
        }
    }
    let mut rows: Vec<(String, f64, f64)> = stages.into_values().collect();
    rows.retain(|(_, decide, perform)| decide + perform > 0.05);
    rows.sort_by(|a, b| (b.1 + b.2).total_cmp(&(a.1 + a.2)));
    rows.truncate(6);
    if !rows.is_empty() {
        lines.push(format!(
            "{DIM}stage busy time{RESET}   {BLUE}█{RESET} decide  {YELLOW}█{RESET} perform"
        ));
        let bar_w = width.saturating_sub(40).min(50);
        let max = rows.iter().map(|(_, d, p)| d + p).fold(1.0, f64::max);
        for (name, decide, perform) in &rows {
            let cells_d = (decide / max * bar_w as f64).round() as usize;
            let cells_p = (perform / max * bar_w as f64).round() as usize;
            lines.push(format!(
                "{name:>18}  {BLUE}{}{RESET}{YELLOW}{}{RESET} {}",
                "█".repeat(cells_d),
                "█".repeat(cells_p),
                fmt_ms(Some(decide + perform)),
            ));
        }
        lines.push(String::new());
    }

    // --- recent turns ---
    lines.push(format!(
        "{DIM}{:>4}  {:<13} {:>9} {:>9}  transcript{RESET}",
        "#", "end", "response", "1st spch"
    ));
    for turn in app.turns.iter().rev().take(8) {
        let end = match (turn.end, turn.errors.is_empty()) {
            (TurnEnd::Interrupted, _) => format!("{RED}⚠ interrupted{RESET}"),
            (_, false) => format!("{RED}✕ error{RESET}      "),
            _ if turn.origin == TurnOrigin::Model => "◆ model turn ".to_string(),
            _ => format!("{DIM}done{RESET}         "),
        };
        let mut text = String::new();
        if let Some(user) = &turn.user_text {
            text.push_str(&format!("{DIM}you:{RESET} {user} "));
        }
        if let Some(agent) = &turn.agent_text {
            text.push_str(&format!("{DIM}agent:{RESET} {agent}"));
        }
        for call in &turn.tool_calls {
            text.push_str(&format!(" {DIM}[{}]{RESET}", call.name));
        }
        let line = format!(
            "{:>4}  {end} {:>9} {:>9}  {text}",
            turn.turn,
            fmt_ms(turn.timings.response_latency_ms),
            fmt_ms(turn.timings.time_to_first_speech_ms),
        );
        lines.push(truncate_ansi(&line, width));
    }
    lines
}

/// Sparkline over optional values; a gap renders as a space.
fn sparkline(values: &[Option<f64>], max: f64) -> String {
    const BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    values
        .iter()
        .map(|value| match value {
            Some(v) => BLOCKS[(((v / max) * 7.0).round() as usize).min(7)],
            None => ' ',
        })
        .collect()
}

fn median(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut sorted: Vec<f64> = values.collect();
    if sorted.is_empty() {
        return None;
    }
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    Some(if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    })
}

fn fmt_ms(value: Option<f64>) -> String {
    match value {
        None => "–".to_string(),
        Some(v) if v >= 10_000.0 => format!("{:.1} s", v / 1_000.0),
        Some(v) => format!("{} ms", v.round() as i64),
    }
}

/// Type names arrive fully qualified with generics; keep the leaf.
fn shorten(name: &str) -> String {
    let base = name.split('<').next().unwrap_or(name);
    base.rsplit("::").next().unwrap_or(base).to_string()
}

/// Truncate to `width` visible columns, ignoring ANSI escapes (kept intact)
/// and counting other chars as one column each; appends a reset so an escape
/// cut mid-line cannot bleed color.
fn truncate_ansi(line: &str, width: usize) -> String {
    let mut out = String::new();
    let mut visible = 0;
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            out.push(c);
            for c in chars.by_ref() {
                out.push(c);
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
            continue;
        }
        if visible == width.saturating_sub(1) {
            out.push('…');
            break;
        }
        out.push(c);
        visible += 1;
    }
    out.push_str(RESET);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparkline_scales_and_gaps() {
        let s = sparkline(&[Some(0.0), Some(50.0), None, Some(100.0)], 100.0);
        assert_eq!(s, "▁▅ █");
    }

    #[test]
    fn median_of_odd_and_even() {
        assert_eq!(median([3.0, 1.0, 2.0].into_iter()), Some(2.0));
        assert_eq!(median([1.0, 2.0, 3.0, 4.0].into_iter()), Some(2.5));
        assert_eq!(median(std::iter::empty()), None);
    }

    #[test]
    fn shorten_keeps_the_leaf_type() {
        assert_eq!(
            shorten("pipecrab_stt::stage::SttStage<sherpa::Offline>"),
            "SttStage"
        );
        assert_eq!(shorten("tail"), "tail");
    }

    #[test]
    fn truncate_counts_visible_columns_not_escapes() {
        let colored = format!("{BLUE}abcdef{RESET}");
        let cut = truncate_ansi(&colored, 4);
        assert!(cut.contains("abc…"), "got {cut:?}");
        assert!(cut.ends_with(RESET));
        // Short lines pass through with a trailing reset.
        assert_eq!(truncate_ansi("ab", 10), format!("ab{RESET}"));
    }
}
