# deckengine

**A low-latency, real-time-safe two-deck audio engine in Rust, for Linux and Windows.**

Playback, gapless seeking, sample-accurate loops, DJ crossfades, pitch/tempo, stem isolation,
loudness normalisation and a brick-wall limiter, in one library that apps in any language
can depend on through its C API.

- **Target: under 5 ms output latency.** Measured **3.7 ms** (callback → DAC, incl. limiter)
  on Linux, see [Latency](#latency).
- **Real-time safe:** the audio callback never allocates, locks or does I/O. A test with a
  counting allocator enforces this.
- **Light:** about 20–40 µs per 128-frame callback with two decks, stems and master chain
  (under 2 % of one core).
- **Portable:** Linux (ALSA, PipeWire, JACK, PulseAudio), Windows (own `IAudioClient3` WASAPI
  backend, ASIO) and macOS (CoreAudio).

> Status: 0.1, early. APIs may still change.

---

## Features

| | |
|---|---|
| **Decks** | Two independent decks, each with its own background decoder thread |
| **Formats** | MP3, AAC/M4A, FLAC, Ogg Vorbis, WAV, AIFF, ALAC (via [symphonia](https://github.com/pdeljanov/Symphonia)). Custom sources through the `FrameSource` trait (network streams, other decoders, synths) |
| **Seeking** | Sample-exact (verified against linear decoding for MP3, FLAC, Vorbis, AAC). Old audio keeps playing until the new position is decoded, then a ~2 ms micro-crossfade, so seeks never click and never play silence |
| **Loops** | Seamless, sample-accurate, any length |
| **Gapless** | Encoder delay and padding trimmed (MP3 LAME/Xing, Vorbis, ...) |
| **Tempo / pitch** | Varispeed 0.25×–4× through a 16-tap windowed-sinc resampler (also does sample-rate conversion, bit-transparent at 1:1) |
| **Stems** | Bass / vocal / drums isolator: phase-coherent Linkwitz-Riley 3-band split, centre-channel extraction for vocals (0 = instrumental, 2 = a cappella) |
| **Mixer** | Crossfader (equal-power, linear, additive), per-deck volume and trim |
| **Scheduling** | `play_at`, `crossfade`, stem changes at an exact frame of the engine clock, for beat-synced transitions |
| **Master** | Real-time loudness normaliser (BS.1770 short-term, target LUFS), look-ahead brick-wall limiter (default 1 ms, −1 dBFS) |
| **Analysis** | Offline EBU R128: integrated loudness, true peak, loudness range, and a gain to hit a target |
| **Telemetry** | Callback time (avg/max), DSP load, deadline misses, xruns, device latency, levels, all lock-free |
| **Bindings** | Rust API, C ABI (`include/deckengine.h`), `cdylib` + `staticlib` for C/C++, Kotlin/Java (JNA/Panama), C#, Python, Dart/Flutter, ... |

## Latency

Measured with `examples/play.rs`, 48 kHz, two decks loaded, full master chain on.
*Output latency* = time from the audio callback to the DAC (as the backend reports it, never
less than one buffer) + limiter look-ahead (1 ms).

| System | Backend | Period | Output latency | Xruns |
|---|---|---|---|---|
| Linux 7.1, Intel HDA (ALC662) | ALSA `hw:` direct, `realtime` feature | 128 fr (2.67 ms) | **3.67 ms** | 0 |
| Linux 7.1, Intel HDA (ALC662) | ALSA `hw:` direct | 128 fr (2.67 ms) | **4.35 ms** | 0 |
| Linux 7.1, Intel HDA (ALC662) | ALSA `hw:` direct, `realtime` feature | 64 fr (1.33 ms) | 2.33 ms | some¹ |
| Linux 7.1, USB DAC | PipeWire (native), stock config | 128 fr | 11.7 ms² | 0 |
| Linux 7.1, USB DAC | ALSA → PipeWire plugin, stock config | 128 fr | 14.2 ms² | 0 |
| Windows 10/11 | WASAPI `IAudioClient3` | driver minimum (typically 128 fr) | expected ≈ 4–5 ms³ | – |

¹ The engine never missed a deadline (max 63 µs per 1.33 ms block). The xruns come from thread
wake-up jitter in the OS. Use 128 frames unless the system is tuned for real-time audio.
² Dominated by PipeWire's buffering for USB devices (`api.alsa.headroom = 256`,
`period-size = 256`), not by the engine. See [Linux tuning](#linux-under-5-ms-with-pipewire).
³ Not measured on hardware yet. Reports welcome.

Engine CPU (release build, 64-frame blocks, two decks + stems + normaliser + limiter): average
**42 µs**, worst **120 µs** per block, i.e. **3 % of the real-time budget**.

## Quick start (Rust)

```toml
[dependencies]
deckengine = { git = "https://github.com/<you>/deckengine" }
```

```rust
use deckengine::{DeckId, Engine, EngineConfig, OutputConfig, StemLevels, When};

let engine = Engine::start(OutputConfig::default(), EngineConfig::default())?;
println!("{:?}", engine.output_info());

engine.load_file(DeckId::A, "track-a.flac")?;
engine.load_file(DeckId::B, "track-b.mp3")?;
engine.play(DeckId::A)?;

// Level-match B to A using offline analysis.
let b = deckengine::analyze_file("track-b.mp3")?;
engine.set_trim_db(DeckId::B, b.gain_to_target(-14.0, -1.0))?;

// Exactly 2 s from now: start B, kill its bass, and fade over 8 s.
let at = engine.clock() + engine.seconds_to_frames(2.0);
engine.play_at(DeckId::B, When::At(at))?;
engine.set_stems_at(DeckId::B, StemLevels { bass: 0.0, ..Default::default() }, When::At(at))?;
engine.crossfade(1.0, 8.0, When::At(at))?;

engine.set_loop(DeckId::A, 30.0, 32.0)?; // seamless 2 s loop
println!("{:?}", engine.stats());
```

Without an audio device (tests, rendering to a file, your own backend):

```rust
let (handle, mut processor) = deckengine::engine::create(48_000, &Default::default());
handle.load_file(DeckId::A, "song.wav")?;
handle.play(DeckId::A)?;
let mut block = vec![0.0f32; 256 * 2];
processor.render(&mut block); // interleaved stereo
```

## Using it from other languages (C ABI)

Build `libdeckengine.so` / `deckengine.dll` / `libdeckengine.dylib` with `cargo build --release`
and include [`include/deckengine.h`](include/deckengine.h).

```c
DeOutputConfig cfg = {0};          /* zero = defaults: 128 frames, default device */
DeEngine *e = de_engine_start(&cfg);
de_deck_load(e, DE_DECK_A, "song.mp3");
de_deck_play(e, DE_DECK_A);
de_deck_set_stems(e, DE_DECK_A, 1.0f, 0.0f, 1.0f);   /* instrumental */
/* ... */
de_engine_free(e);
```

See [`examples/c/minimal.c`](examples/c/minimal.c).

**Kotlin / JVM (JNA)**:

```kotlin
interface DeckEngine : Library {
    fun de_engine_start(config: Pointer?): Pointer?
    fun de_engine_free(engine: Pointer)
    fun de_deck_load(engine: Pointer, deck: Int, path: String): Int
    fun de_deck_play(engine: Pointer, deck: Int): Int
    fun de_set_crossfader(engine: Pointer, position: Float): Int
    fun de_last_error(): String
    companion object { val lib: DeckEngine = Native.load("deckengine", DeckEngine::class.java) }
}

val e = DeckEngine.lib.de_engine_start(null) ?: error(DeckEngine.lib.de_last_error())
DeckEngine.lib.de_deck_load(e, 0, "/music/song.flac")
DeckEngine.lib.de_deck_play(e, 0)
```

## Architecture

```text
 app / UI thread                         audio thread (device callback)
 ───────────────                         ─────────────────────────────
 EngineHandle ──commands (lock-free)──▶  Processor::process()
      ▲                                    │  deck A: resample → stems → gain ─┐
      └──── telemetry (atomics) ◀──────────┤  deck B: resample → stems → gain ─┴▶ crossfader
                                           │        ▶ normaliser ▶ limiter ▶ device
 decoder threads ──decoded chunks (lock-free rings)──▶ decks
```

- **No locks, allocations or I/O on the audio thread.** Commands go through a wait-free ring.
  Buffers are preallocated. Freed resources are handed to a janitor thread.
- **Generations instead of flushes.** Seeks and loop changes start a new *generation* in the
  decoder. The audio thread switches once the new data is there, either sample-aligned
  (loops) or with a micro-crossfade (seeks).
- **Everything is timed on the engine clock** (frames rendered), so scheduled actions land on
  an exact sample even inside a callback.

## Build

```sh
cargo build --release                  # library + cdylib/staticlib
cargo test --release                   # 28 tests, no audio device needed
cargo run --release --example play -- song.flac --buffer 128
cargo run --release --example play -- --list          # hosts and device ids
```

### Cargo features

| Feature | Default | What it does |
|---|---|---|
| `decode` | ✓ | File decoding via symphonia |
| `output` | ✓ | Audio device output via cpal (+ Windows `IAudioClient3` backend) |
| `pipewire` | | Native PipeWire host on Linux (needs `libpipewire-0.3` dev files and clang) |
| `jack` | | JACK host |
| `realtime` | | Real-time priority for the audio thread on Linux (RTKit/rlimits). Recommended |
| `asio` | | ASIO on Windows (needs the Steinberg ASIO SDK, see the cpal docs) |

Linux build dependency: `libasound2-dev` (Debian/Ubuntu) or `alsa-lib` (Arch/Fedora).

## Platform notes

### Linux: under 5 ms with PipeWire

The engine asks PipeWire for a 128-frame quantum (2.67 ms), and PipeWire grants it. The rest
is PipeWire's own buffering of the hardware device, which is large for USB interfaces by
default. To lower it, add a WirePlumber rule, for example
`~/.config/wireplumber/wireplumber.conf.d/51-low-latency.conf`:

```
monitor.alsa.rules = [
  {
    matches = [ { node.name = "~alsa_output.usb-*" } ]
    actions = {
      update-props = {
        api.alsa.period-size   = 128
        api.alsa.headroom      = 0
        api.alsa.disable-batch = true
      }
    }
  }
]
```

then `systemctl --user restart wireplumber pipewire`. Lower values reduce latency but raise
the risk of dropouts. Alternatively open the ALSA `hw:` device directly
(`OutputConfig { host: Some("alsa".into()), device: Some("hw:CARD=0,DEV=0".into()), .. }`),
which bypasses PipeWire but takes exclusive access to the card.

### Windows

The default backend is WASAPI shared mode through `IAudioClient3`, which gets the driver's
minimum engine period (128 frames = 2.67 ms on the standard Microsoft HD Audio driver)
without taking exclusive access. On Windows 8.1 and older it falls back to classic WASAPI
(10 ms period). For pro interfaces use `host: Some("asio".into())` with the `asio` feature.

## Limitations

- Speed changes tempo and pitch together (like a turntable); time-stretching is not done yet.
- Stems are DSP isolation, not machine-learning separation.
- M4A files without iTunes gapless metadata keep the AAC encoder delay.

## License

Dual-licensed: [AGPL-3.0-or-later](LICENSE), or a commercial license for closed-source or hosted use.
See [LICENSING.md](LICENSING.md).

© 2026 Jason-Marshall Fastner
