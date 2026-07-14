# Echo example

This example captures audio from the default microphone, sends it through a
PipeCrab pipeline, and plays it through the default output device.

## Requirements

- Rust 1.86 or newer.
- macOS, Windows, or Linux with working input and output audio devices.
- Microphone permission for the terminal or application running Cargo.
- Headphones. Using speakers can create a loud feedback loop.

## Run

From the repository root, start a live monitor:

```console
cargo run -p echo
```

Add a delay to make the returned audio sound like an echo:

```console
cargo run -p echo -- --delay-ms 400
```

Use `--seconds` for a bounded run:

```console
cargo run -p echo -- --seconds 5
cargo run -p echo -- --delay-ms 400 --seconds 5
```

Without `--seconds`, stop the example with Ctrl-C.

The example prints the selected input and output devices, audio format, chunk
size, and configured delay before audio starts flowing.
