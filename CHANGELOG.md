# Changelog

## 0.1.0 — unreleased

First version.

- Two decks with real-time-safe streaming from a background decoder (lock-free rings).
- Decoding of MP3, AAC/M4A, FLAC, Ogg Vorbis, WAV, AIFF, ALAC via symphonia, with
  sample-exact seeking (verified for MP3, FLAC, Vorbis, AAC) and gapless trimming.
- Seamless, sample-accurate loops; click-free seeks (~2 ms micro-crossfade).
- Variable-ratio windowed-sinc resampling (sample-rate conversion + varispeed).
- Three-band stem isolator (bass / vocal / drums), Linkwitz-Riley crossovers.
- Crossfader with equal-power, linear and additive curves; sample-accurate scheduling of
  play, pause, stems and crossfades on the engine clock.
- Master bus: real-time loudness normaliser (BS.1770 short-term), look-ahead limiter.
- Offline EBU R128 analysis (integrated loudness, true peak, LRA).
- Outputs: ALSA / PipeWire / JACK / PulseAudio / CoreAudio / ASIO via cpal, plus a native
  Windows `IAudioClient3` low-latency WASAPI backend.
- C ABI (`include/deckengine.h`), `cdylib` and `staticlib`.

### Known limitations

- M4A files written without iTunes gapless info keep the AAC encoder delay (symphonia does
  not apply MP4 edit lists).
- Speed changes pitch and tempo together (varispeed); no time-stretching yet.
- Stems are spectral/spatial isolation, not ML source separation.
