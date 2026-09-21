// Silero VAD on onnxruntime-web, behind pipecrab's SpeechScorer.
//
// The page supplies the loaded `ort` namespace rather than this module importing
// one, so the crate pins no CDN, no bundler layout, and no execution provider.
//
// Every name below is read off the session instead of assumed, because the
// exports in circulation disagree on all of them. v5/v6 carries one `state`
// tensor and calls its waveform `input`; the k2-fsa v4 re-export the desktop
// examples use carries `h`/`c`, calls its waveform `x`, and has no `sr` input at
// all — its 16 kHz branch is baked in.

const UNIFIED_STATE_OUT = ["stateN", "state_out", "state"];
const HIDDEN_OUT = ["hn", "new_h", "h"];
const CELL_OUT = ["cn", "new_c", "c"];

function pick(names, candidates) {
  return candidates.find((candidate) => names.includes(candidate));
}

export async function loadSilero(ort, modelUrl, sampleRate) {
  const session = await ort.InferenceSession.create(modelUrl);
  const inputs = session.inputNames;
  const outputs = session.outputNames;

  const unified = inputs.includes("state");
  if (!unified && !(inputs.includes("h") && inputs.includes("c"))) {
    throw new Error(`not a Silero VAD export: its inputs are ${inputs.join(", ")}`);
  }

  // Each recurrent *input* paired with the output that produces its next value.
  const carried = unified
    ? { state: pick(outputs, UNIFIED_STATE_OUT) }
    : { h: pick(outputs, HIDDEN_OUT), c: pick(outputs, CELL_OUT) };
  for (const [input, output] of Object.entries(carried)) {
    if (!output) {
      throw new Error(
        `model takes a "${input}" input with no matching recurrent output ` +
          `(outputs: ${outputs.join(", ")})`,
      );
    }
  }

  // The waveform is the input that is neither the sample rate nor state; the
  // probability is the output that is not state.
  const waveform = inputs.find((name) => name !== "sr" && !(name in carried));
  const probability = outputs.find((name) => !Object.values(carried).includes(name));
  if (!waveform || !probability) {
    throw new Error(
      `cannot tell which tensor is which (inputs: ${inputs.join(", ")}; ` +
        `outputs: ${outputs.join(", ")})`,
    );
  }

  return {
    ort,
    session,
    carried,
    waveform,
    probability,
    rate: inputs.includes("sr")
      ? new ort.Tensor("int64", BigInt64Array.of(BigInt(sampleRate)), [1])
      : null,
    state: zeroState(ort, unified),
  };
}

function zeroState(ort, unified) {
  return unified
    ? { state: new ort.Tensor("float32", new Float32Array(2 * 1 * 128), [2, 1, 128]) }
    : {
        h: new ort.Tensor("float32", new Float32Array(2 * 1 * 64), [2, 1, 64]),
        c: new ort.Tensor("float32", new Float32Array(2 * 1 * 64), [2, 1, 64]),
      };
}

export async function scoreSilero(handle, window) {
  const { ort, session, carried } = handle;
  const feeds = {
    [handle.waveform]: new ort.Tensor("float32", window, [1, window.length]),
    ...handle.state,
  };
  if (handle.rate) {
    feeds.sr = handle.rate;
  }

  const results = await session.run(feeds);

  const next = {};
  for (const [input, output] of Object.entries(carried)) {
    next[input] = results[output];
  }
  handle.state = next;

  return results[handle.probability].data[0];
}
