// Transformers.js automatic-speech-recognition, behind pipecrab's Transcriber.
//
// The page supplies the loaded Transformers.js namespace rather than this module
// importing one, so the crate pins no CDN and no bundler layout. Which model is
// fetched, in what quantization, on WASM or WebGPU, all stay the page's call.

export async function loadTranscriber(transformers, model, options) {
  const transcribe = await transformers.pipeline(
    "automatic-speech-recognition",
    model,
    options ?? {},
  );
  return { transcribe };
}

export async function transcribe(handle, samples, options) {
  const result = await handle.transcribe(samples, options ?? {});
  // Chunked long-form decoding answers with an array of segments; a single pass
  // answers with one object. Both carry `text`.
  const text = Array.isArray(result)
    ? result.map((segment) => segment?.text ?? "").join(" ")
    : (result?.text ?? "");
  return text.trim();
}
