// Speaker playback for pipecrab-audio-web.
//
// Each chunk becomes an AudioBufferSourceNode scheduled on the context's clock,
// back to back behind the last. Scheduled audio plays on the rendering thread,
// so a main thread busy with inference cannot starve it.

// Headroom between scheduling a chunk and its first sample, so an idle sink's
// first chunk is not already late by the time the rendering thread sees it.
const LEAD_SECONDS = 0.02;

// One live output graph. Held by the Rust side for the sink's lifetime.
export class Playback {
  constructor(context) {
    this.context = context;
    this.next = 0;
    this.sources = new Set();
  }

  get sampleRate() {
    return this.context.sampleRate;
  }

  enqueue(samples) {
    const buffer = this.context.createBuffer(1, samples.length, this.context.sampleRate);
    buffer.copyToChannel(samples, 0);
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.context.destination);
    source.onended = () => {
      source.disconnect();
      this.sources.delete(source);
    };
    const start = Math.max(this.next, this.context.currentTime + LEAD_SECONDS);
    source.start(start);
    this.next = start + buffer.duration;
    this.sources.add(source);
  }

  cancel() {
    for (const source of this.sources) {
      source.onended = null;
      source.stop();
      source.disconnect();
    }
    this.sources.clear();
    this.next = 0;
  }

  close() {
    this.cancel();
    this.context.close();
  }
}

export async function openPlayback() {
  const AudioContextClass = globalThis.AudioContext ?? globalThis.webkitAudioContext;
  const context = new AudioContextClass();
  // A context created outside a user gesture starts suspended; playback must be
  // opened from a click handler for this to succeed.
  await context.resume();
  return new Playback(context);
}
