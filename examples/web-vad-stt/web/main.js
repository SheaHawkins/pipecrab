// Loads the two engines, hands them to the wasm pipeline, and prints what comes
// back. Everything below the `start` call is Rust — the same stages the desktop
// examples build, compiled to wasm32.

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.wasm.bundle.min.mjs";
import * as transformers from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js";
import init, { start } from "./pkg/web_vad_stt.js";

// Served next to this page; see the README for the one curl that puts it there.
const VAD_MODEL = "./models/silero_vad.onnx";

// Multi-threaded ONNX needs SharedArrayBuffer, which needs the COOP/COEP headers
// a plain static file server does not send. One thread keeps the example
// runnable from `python3 -m http.server`.
ort.env.wasm.numThreads = 1;
transformers.env.backends.onnx.wasm.numThreads = 1;

const controls = document.getElementById("controls");
const modelInput = document.getElementById("model");
const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");
const statusLine = document.getElementById("status");
const log = document.getElementById("log");

let session = null;
// What the pipeline is doing, and the model files it is fetching for it.
let doing = "";
const files = new Map();

function setStatus(text, kind = "status") {
  statusLine.textContent = text;
  statusLine.dataset.kind = kind;
}

// "loading X" plus how much of it has arrived, while a download is in flight.
function showProgress() {
  let loaded = 0;
  let total = 0;
  for (const file of files.values()) {
    loaded += file.loaded;
    total += file.total;
  }
  // Below a megabyte it is a config or a tokenizer: not worth a size.
  const megabytes = (bytes) => Math.round(bytes / 1e6);
  setStatus(total >= 1e6 ? `${doing} — ${megabytes(loaded)} of ${megabytes(total)} MB` : doing);
}

// The engine reports one of these per file it fetches. A file already in the
// browser's cache reports itself complete at once.
function onProgress(event) {
  if (!event?.total) {
    return;
  }
  const loaded = event.status === "done" ? event.total : (event.loaded ?? 0);
  files.set(event.file, { loaded, total: event.total });
  showProgress();
}

function append(kind, text) {
  const line = document.createElement("li");
  line.dataset.kind = kind;
  line.textContent = kind === "speech" ? `— ${text} —` : text;
  log.append(line);
  line.scrollIntoView({ block: "nearest" });
}

// The one callback the Rust side reports through.
function onEvent(kind, text) {
  if (kind === "status") {
    doing = text;
    files.clear();
    setStatus(text);
    return;
  }
  if (kind === "error") {
    setStatus(text, "error");
  }
  if (kind === "partial") {
    // Replace the previous partial rather than stacking hypotheses.
    const last = log.lastElementChild;
    if (last?.dataset.kind === "partial") {
      last.textContent = text;
      return;
    }
  }
  append(kind, text);
}

await init();
setStatus("Ready. Allow microphone access when the browser asks.");

controls.addEventListener("submit", async (event) => {
  event.preventDefault();
  startButton.disabled = true;
  modelInput.disabled = true;
  try {
    // Called straight out of the click: the microphone prompt and the
    // AudioContext both need a user gesture behind them.
    session = await start(
      ort,
      transformers,
      VAD_MODEL,
      modelInput.value.trim(),
      onEvent,
      onProgress,
    );
    stopButton.disabled = false;
  } catch (error) {
    setStatus(error?.message ?? String(error), "error");
    startButton.disabled = false;
    modelInput.disabled = false;
  }
});

stopButton.addEventListener("click", () => {
  stopButton.disabled = true;
  session?.stop();
  session = null;
  startButton.disabled = false;
  modelInput.disabled = false;
});
