// Loads the three engines, hands them to the wasm pipeline, and shows the
// conversation that comes back. Everything below the `start` call is Rust — the
// same stages the desktop e2e-voice-agent example builds, compiled to wasm32.

import * as ort from "onnxruntime-web";
import * as transformers from "@huggingface/transformers";
import * as kokoro from "kokoro-js";
import init, { start } from "./pkg/web_e2e.js";

// Served next to this page; see the README for the one curl that puts it there.
const VAD_MODEL = "./models/silero_vad.onnx";
const STT_MODEL = "onnx-community/moonshine-tiny-ONNX";
const TTS_MODEL = "onnx-community/Kokoro-82M-v1.0-ONNX";

// Quantization per backend: WebGPU runs half- and full-precision kernels well,
// WebAssembly is quickest on int8.
const DTYPES = {
  webgpu: { lm: "q4f16", tts: "fp32" },
  wasm: { lm: "q8", tts: "q8" },
};

// Multi-threaded ONNX needs SharedArrayBuffer, which needs the COOP/COEP headers
// a plain static file server does not send. Served that way, the WebAssembly
// models get four threads; from `python3 -m http.server`, one.
ort.env.wasm.numThreads = 1;
transformers.env.backends.onnx.wasm.numThreads = self.crossOriginIsolated
  ? Math.min(4, navigator.hardwareConcurrency)
  : 1;
// Run Transformers.js's WebAssembly inference in a worker. On the main thread a
// reply holds the event loop for as long as it takes to generate, and the
// microphone's chunks — and a barge-in with them — would wait behind it.
transformers.env.backends.onnx.wasm.proxy = true;

const controls = document.getElementById("controls");
const fields = ["lmModel", "voice", "device"].map((id) => document.getElementById(id));
const [lmModel, voice, device] = fields;
const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");
const statusLine = document.getElementById("status");
const hearing = document.getElementById("hearing");
const log = document.getElementById("log");

let session = null;
// The agent's reply in progress: sentences arrive one at a time.
let reply = null;
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

// The engines report one of these per file they fetch. A file already in the
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
  line.textContent = text;
  log.append(line);
  line.scrollIntoView({ block: "nearest" });
  return line;
}

// Settings are fixed while a session loads or runs.
function lock(locked) {
  startButton.disabled = locked;
  for (const field of fields) {
    field.disabled = locked;
  }
}

// The one callback the Rust side reports through.
function onEvent(kind, text) {
  switch (kind) {
    case "status":
      doing = text;
      files.clear();
      setStatus(text);
      break;
    case "error":
      setStatus(text, "error");
      append("error", text);
      break;
    case "speech":
      hearing.dataset.speech = text;
      hearing.textContent = text === "started" ? "Hearing you" : "Listening";
      break;
    case "user":
      reply = null;
      append("user", text);
      break;
    case "agent":
      if (reply) {
        reply.textContent += ` ${text}`;
      } else {
        reply = append("agent", text);
      }
      break;
    case "latency":
      if (reply) {
        reply.dataset.latency = `first audio ${(Number(text) / 1000).toFixed(1)} s after you stopped`;
      }
      break;
  }
}

await init();

// WebGPU where there is an adapter to run it; WebAssembly everywhere.
const adapter = await navigator.gpu?.requestAdapter().catch(() => null);
if (!adapter) {
  device.value = "wasm";
  device.querySelector('option[value="webgpu"]').disabled = true;
}
setStatus("Ready. Allow microphone access when the browser asks.");

controls.addEventListener("submit", async (event) => {
  event.preventDefault();
  lock(true);
  try {
    // Called straight out of the click: the microphone prompt and both
    // AudioContexts need a user gesture behind them.
    session = await start(
      ort,
      transformers,
      kokoro,
      {
        vadModel: VAD_MODEL,
        sttModel: STT_MODEL,
        lmModel: lmModel.value.trim(),
        ttsModel: TTS_MODEL,
        voice: voice.value.trim(),
        device: device.value,
        lmDtype: DTYPES[device.value].lm,
        ttsDtype: DTYPES[device.value].tts,
      },
      onEvent,
      onProgress,
    );
    stopButton.disabled = false;
    hearing.hidden = false;
  } catch (error) {
    setStatus(error?.message ?? String(error), "error");
    lock(false);
  }
});

stopButton.addEventListener("click", () => {
  stopButton.disabled = true;
  hearing.hidden = true;
  session?.stop();
  session = null;
  reply = null;
  lock(false);
});
