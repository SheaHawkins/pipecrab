// Kokoro text-to-speech on kokoro-js, behind pipecrab's Synthesizer.
//
// The page supplies the loaded kokoro-js namespace rather than this module
// importing one, so the crate pins no CDN and no bundler layout. kokoro-js runs
// on Transformers.js; which copy it resolves, and on what backend, stays the
// page's call.

export async function loadKokoro(kokoro, model, voice, options) {
  const tts = await kokoro.KokoroTTS.from_pretrained(model, options ?? {});
  // Checked here so a typo fails the load rather than the first reply.
  if (!(voice in tts.voices)) {
    throw new Error(`no voice "${voice}" (voices: ${Object.keys(tts.voices).join(", ")})`);
  }
  // `tail` settles once the last synthesis has.
  return { tts, tail: Promise.resolve() };
}

export function synthesize(handle, text, options) {
  // One synthesis at a time on the model. A synthesis cannot be stopped once
  // started, so one abandoned on a barge-in runs out and the next waits for it.
  const run = handle.tail.then(() => handle.tts.generate(text, options));
  handle.tail = run.then(
    () => undefined,
    () => undefined,
  );
  return run.then((audio) => ({ samples: audio.audio, sampleRate: audio.sampling_rate }));
}
