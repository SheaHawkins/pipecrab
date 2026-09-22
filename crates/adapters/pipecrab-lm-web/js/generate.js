// Transformers.js text-generation, behind pipecrab's LanguageModel.
//
// The page supplies the loaded Transformers.js namespace rather than this module
// importing one, so the crate pins no CDN and no bundler layout. Which model is
// fetched, in what quantization, on WASM or WebGPU, all stay the page's call.

export async function loadGenerator(transformers, model, options) {
  const generator = await transformers.pipeline("text-generation", model, options ?? {});
  // `tail` settles once the last generation has; `current` is the one running.
  return { transformers, generator, tail: Promise.resolve(), current: null };
}

// One generation's text, queued on the JS side and pulled with `next()`. The
// streamer pushes here rather than into Rust, so a generation that outlives the
// Rust stream reading it has nothing freed to call back into.
class Generation {
  constructor(stop) {
    this.stop = stop;
    this.queue = [];
    this.waiter = null;
    this.done = false;
    this.error = null;
  }

  push(text) {
    this.queue.push(text);
    this.wake();
  }

  finish(error) {
    this.done = true;
    this.error = error ?? null;
    this.wake();
  }

  wake() {
    const waiter = this.waiter;
    this.waiter = null;
    waiter?.();
  }

  // The next piece of text, or `undefined` once the generation has ended.
  async next() {
    while (this.queue.length === 0 && !this.done) {
      await new Promise((resolve) => {
        this.waiter = resolve;
      });
    }
    if (this.queue.length > 0) {
      return this.queue.shift();
    }
    if (this.error) {
      throw this.error;
    }
    return undefined;
  }

  // Stop at the next token. Idempotent.
  interrupt() {
    this.stop.interrupt();
  }
}

export function startGeneration(handle, messages, options) {
  const { transformers, generator } = handle;
  const generation = new Generation(new transformers.InterruptableStoppingCriteria());
  const streamer = new transformers.TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (text) => generation.push(text),
  });
  // One generation at a time on the model: an interrupted one only stops at its
  // next token, so the next waits for it rather than racing it. One interrupted
  // while still waiting never starts.
  const run = handle.tail.then(() => {
    if (generation.stop.interrupted) {
      return undefined;
    }
    return generator(messages, { ...options, streamer, stopping_criteria: generation.stop });
  });
  handle.tail = run.then(
    () => generation.finish(),
    (error) => generation.finish(error),
  );
  handle.current = generation;
  return generation;
}

export function interruptCurrent(handle) {
  handle.current?.interrupt();
}
