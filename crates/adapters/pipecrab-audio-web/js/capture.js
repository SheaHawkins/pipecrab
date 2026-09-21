// Microphone capture for pipecrab-audio-web.
//
// getUserMedia feeds an AudioContext whose AudioWorklet downmixes to mono and
// posts fixed-size chunks back to the main thread. The worklet source is inlined
// and loaded from a blob URL, so the crate ships as a single JS snippet with no
// second file for the page to serve.

const WORKLET_SOURCE = `
class PipecrabCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.size = options.processorOptions.chunkFrames;
    this.chunk = new Float32Array(this.size);
    this.filled = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) {
      return true;
    }
    const channels = input.length;
    const frames = input[0].length;
    for (let frame = 0; frame < frames; frame += 1) {
      let sum = 0;
      for (let channel = 0; channel < channels; channel += 1) {
        sum += input[channel][frame];
      }
      this.chunk[this.filled] = sum / channels;
      this.filled += 1;
      if (this.filled === this.size) {
        const chunk = this.chunk.slice();
        this.port.postMessage(chunk, [chunk.buffer]);
        this.filled = 0;
      }
    }
    return true;
  }
}

registerProcessor("pipecrab-capture", PipecrabCapture);
`;

// One live capture graph. Held by the Rust side for the source's lifetime.
export class Capture {
  constructor(stream, context, worklet, nodes) {
    this.stream = stream;
    this.context = context;
    this.worklet = worklet;
    this.nodes = nodes;
  }

  get sampleRate() {
    return this.context.sampleRate;
  }

  close() {
    // Detach the handler first, and synchronously: the Rust callback it lands in
    // is freed the instant the source drops, while tearing the graph down is
    // asynchronous enough for the worklet to post again on the way out.
    this.worklet.port.onmessage = null;
    for (const node of this.nodes) {
      node.disconnect();
    }
    for (const track of this.stream.getTracks()) {
      track.stop();
    }
    this.context.close();
  }
}

export async function openCapture(chunkMs, onChunk) {
  const media = globalThis.navigator?.mediaDevices;
  if (!media?.getUserMedia) {
    throw new Error(
      "getUserMedia is unavailable — a microphone needs a secure context (https, or localhost)",
    );
  }
  const stream = await media.getUserMedia({ audio: true, video: false });

  const AudioContextClass = globalThis.AudioContext ?? globalThis.webkitAudioContext;
  const context = new AudioContextClass();
  // A context created before a user gesture starts suspended; capture must be
  // opened from a click handler for this to succeed.
  await context.resume();

  const url = URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: "text/javascript" }));
  try {
    await context.audioWorklet.addModule(url);
  } finally {
    URL.revokeObjectURL(url);
  }

  const worklet = new AudioWorkletNode(context, "pipecrab-capture", {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [1],
    processorOptions: {
      chunkFrames: Math.max(128, Math.round((context.sampleRate * chunkMs) / 1000)),
    },
  });
  worklet.port.onmessage = (event) => onChunk(event.data);

  const source = context.createMediaStreamSource(stream);
  // A node is only pulled while it reaches the destination, so the worklet is
  // routed onward through a silent gain rather than left dangling — which keeps
  // the graph running without echoing the microphone back out of the speakers.
  const mute = context.createGain();
  mute.gain.value = 0;
  source.connect(worklet);
  worklet.connect(mute);
  mute.connect(context.destination);

  return new Capture(stream, context, worklet, [source, worklet, mute]);
}
