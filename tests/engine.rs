// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Jason-Marshall Fastner <jasonfastner@protonmail.com>

//! Offline end-to-end tests of the engine (no audio device needed).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::time::{Duration, Instant};

use deckengine::engine::create;
use deckengine::{CrossfadeCurve, DeckId, EngineConfig, PcmSource, StemLevels, When};

// ── Allocation guard: counts allocations made by the current thread while armed. ──

struct CountingAlloc;

thread_local! {
    static ARMED: Cell<bool> = const { Cell::new(false) };
    static COUNT: Cell<u64> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ARMED.try_with(|a| {
            if a.get() {
                COUNT.with(|c| c.set(c.get() + 1));
            }
        });
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let _ = ARMED.try_with(|a| {
            if a.get() {
                COUNT.with(|c| c.set(c.get() + 1));
            }
        });
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn count_allocs<R>(f: impl FnOnce() -> R) -> (R, u64) {
    COUNT.with(|c| c.set(0));
    ARMED.with(|a| a.set(true));
    let r = f();
    ARMED.with(|a| a.set(false));
    (r, COUNT.with(|c| c.get()))
}

// ── Helpers ──

fn sine(freq: f32, sr: u32, seconds: f32, amp: f32) -> PcmSource {
    let n = (sr as f32 * seconds) as usize;
    let mono: Vec<f32> =
        (0..n).map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin() * amp).collect();
    PcmSource::from_mono(&mono, sr)
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

/// Renders `frames` stereo frames in blocks of `block`, retrying briefly while the streamer
/// thread fills the rings.
fn render(p: &mut deckengine::Processor, frames: usize, block: usize) -> Vec<f32> {
    let mut out = vec![0.0; frames * 2];
    for chunk in out.chunks_mut(block * 2) {
        p.render(chunk);
    }
    out
}

/// Zero-crossing frequency estimate of the left channel.
fn estimate_freq(stereo: &[f32], sr: u32) -> f32 {
    let left: Vec<f32> = stereo.iter().step_by(2).copied().collect();
    let crossings = left.windows(2).filter(|w| w[0] <= 0.0 && w[1] > 0.0).count();
    crossings as f32 * sr as f32 / left.len() as f32
}

fn wait_for_buffer() {
    // Give the streamer thread time to decode ahead.
    std::thread::sleep(Duration::from_millis(60));
}

#[test]
fn plays_and_resamples_44k1_to_48k() {
    let (h, mut p) = create(48_000, &EngineConfig::default());
    h.load_source(DeckId::A, Box::new(sine(1000.0, 44_100, 2.0, 0.5))).unwrap();
    h.play(DeckId::A).unwrap();
    wait_for_buffer();
    let out = render(&mut p, 48_000, 128);
    let steady = &out[9600..];
    let f = estimate_freq(steady, 48_000);
    assert!((f - 1000.0).abs() < 3.0, "pitch after resampling: {f} Hz");
    let level = rms(steady);
    assert!((level - 0.5 / 2f32.sqrt()).abs() < 0.01, "level {level}");
    let st = h.deck_status(DeckId::A);
    assert!(st.playing);
    assert!((st.position_seconds - 1.0).abs() < 0.01, "{st:?}");
}

#[test]
fn speed_changes_pitch() {
    let (h, mut p) = create(48_000, &EngineConfig::default());
    h.load_source(DeckId::A, Box::new(sine(1000.0, 48_000, 3.0, 0.5))).unwrap();
    h.set_speed(DeckId::A, 1.08).unwrap();
    h.play(DeckId::A).unwrap();
    wait_for_buffer();
    let out = render(&mut p, 48_000, 256);
    let f = estimate_freq(&out[24_000..], 48_000);
    assert!((f - 1080.0).abs() < 4.0, "{f}");
}

#[test]
fn scheduled_crossfade_is_sample_accurate() {
    let (h, mut p) = create(48_000, &EngineConfig::default());
    h.set_crossfade_curve(CrossfadeCurve::Linear).unwrap();
    h.set_limiter(false).unwrap();
    // Deck A: DC-ish constant 0.25, deck B: constant 0.5 → the mix reveals the fader.
    let a = PcmSource::new(vec![0.25f32; 48_000 * 2 * 2], 48_000);
    let b = PcmSource::new(vec![0.5f32; 48_000 * 2 * 2], 48_000);
    h.load_source(DeckId::A, Box::new(a)).unwrap();
    h.load_source(DeckId::B, Box::new(b)).unwrap();
    h.play(DeckId::A).unwrap();
    h.play(DeckId::B).unwrap();
    // Instant cut to B at exactly frame 10_007.
    h.crossfade(1.0, 0.0, When::At(10_007)).unwrap();
    wait_for_buffer();
    let out = render(&mut p, 20_000, 128);
    let left: Vec<f32> = out.iter().step_by(2).copied().collect();
    assert!((left[10_006] - 0.25).abs() < 1e-3, "{}", left[10_006]);
    assert!((left[10_007] - 0.5).abs() < 1e-3, "{}", left[10_007]);
}

#[test]
fn stems_bass_kill() {
    let (h, mut p) = create(48_000, &EngineConfig::default());
    h.load_source(DeckId::A, Box::new(sine(50.0, 48_000, 3.0, 0.5))).unwrap();
    h.set_stems(DeckId::A, StemLevels { bass: 0.0, ..Default::default() }).unwrap();
    h.play(DeckId::A).unwrap();
    wait_for_buffer();
    let out = render(&mut p, 48_000, 128);
    assert!(rms(&out[48_000..]) < 0.02, "{}", rms(&out[48_000..]));
}

#[test]
fn limiter_holds_ceiling() {
    let (h, mut p) = create(48_000, &EngineConfig::default());
    h.load_source(DeckId::A, Box::new(sine(200.0, 48_000, 2.0, 1.0))).unwrap();
    h.set_volume(DeckId::A, 3.0).unwrap(); // +9.5 dB into the limiter
    h.play(DeckId::A).unwrap();
    wait_for_buffer();
    let out = render(&mut p, 48_000, 128);
    let peak = out.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(peak <= 0.8913 + 1e-4, "peak {peak}");
}

#[test]
fn track_end_is_reported() {
    let (h, mut p) = create(48_000, &EngineConfig::default());
    h.load_source(DeckId::A, Box::new(sine(440.0, 48_000, 0.1, 0.5))).unwrap();
    h.play(DeckId::A).unwrap();
    wait_for_buffer();
    let _ = render(&mut p, 9_600, 128);
    let st = h.deck_status(DeckId::A);
    assert!(st.ended && !st.playing, "{st:?}");
}

#[test]
fn audio_thread_never_allocates() {
    let (h, mut p) = create(48_000, &EngineConfig::default());
    h.set_normalization(Some(-14.0)).unwrap();
    h.load_source(DeckId::A, Box::new(sine(440.0, 44_100, 5.0, 0.5))).unwrap();
    h.load_source(DeckId::B, Box::new(sine(660.0, 48_000, 5.0, 0.5))).unwrap();
    h.play(DeckId::A).unwrap();
    h.play(DeckId::B).unwrap();
    h.set_stems(DeckId::B, StemLevels { vocal: 0.0, bass: 0.5, drums: 1.5 }).unwrap();
    wait_for_buffer();

    let mut buf = vec![0.0f32; 128 * 2];
    // Warm-up: first block applies the Load commands.
    p.render(&mut buf);

    h.crossfade(1.0, 1.0, When::Now).unwrap();
    h.set_speed(DeckId::A, 1.05).unwrap();
    h.seek(DeckId::B, 2.0).unwrap();
    std::thread::sleep(Duration::from_millis(20));

    let ((), allocs) = count_allocs(|| {
        for _ in 0..2_000 {
            p.render(&mut buf);
        }
    });
    assert_eq!(allocs, 0, "audio thread allocated {allocs} times");
}

#[test]
fn processing_is_far_below_real_time() {
    let (h, mut p) = create(48_000, &EngineConfig::default());
    h.set_normalization(Some(-14.0)).unwrap();
    h.load_source(DeckId::A, Box::new(sine(440.0, 44_100, 10.0, 0.5))).unwrap();
    h.load_source(DeckId::B, Box::new(sine(660.0, 44_100, 10.0, 0.5))).unwrap();
    h.set_speed(DeckId::B, 1.03).unwrap();
    h.set_crossfader(0.5).unwrap();
    h.play(DeckId::A).unwrap();
    h.play(DeckId::B).unwrap();
    wait_for_buffer();

    let mut buf = vec![0.0f32; 64 * 2];
    let mut worst = Duration::ZERO;
    let blocks = 48_000 * 2 / 64; // 2 s of audio in 64-frame (1.33 ms) blocks
    for _ in 0..blocks {
        let t = Instant::now();
        p.render(&mut buf);
        worst = worst.max(t.elapsed());
        // Pace roughly like a device would so the streamer keeps up.
        std::thread::sleep(Duration::from_micros(200));
    }
    let s = h.stats();
    eprintln!(
        "64-frame block: avg {:.1} µs, worst {:?}, dsp load {:.1} %",
        s.process_avg_us,
        worst,
        s.dsp_load * 100.0
    );
    // Budget of a 64-frame block is 1333 µs. Stay far below it even in debug builds.
    assert!(s.process_avg_us < 700.0, "avg {} µs", s.process_avg_us);
}

/// Writes a 16-bit stereo WAV with a rising ramp in the left channel (value encodes the frame
/// index), so the decoded position can be verified sample-exactly.
#[cfg(feature = "decode")]
fn write_ramp_wav(path: &std::path::Path, sr: u32, frames: u32) {
    use std::io::Write;
    let data_len = frames * 4;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    f.write_all(b"RIFF").unwrap();
    f.write_all(&(36 + data_len).to_le_bytes()).unwrap();
    f.write_all(b"WAVEfmt ").unwrap();
    f.write_all(&16u32.to_le_bytes()).unwrap();
    f.write_all(&1u16.to_le_bytes()).unwrap(); // PCM
    f.write_all(&2u16.to_le_bytes()).unwrap();
    f.write_all(&sr.to_le_bytes()).unwrap();
    f.write_all(&(sr * 4).to_le_bytes()).unwrap();
    f.write_all(&4u16.to_le_bytes()).unwrap();
    f.write_all(&16u16.to_le_bytes()).unwrap();
    f.write_all(b"data").unwrap();
    f.write_all(&data_len.to_le_bytes()).unwrap();
    for i in 0..frames {
        let v = (i % 32_768) as i16;
        f.write_all(&v.to_le_bytes()).unwrap();
        f.write_all(&v.to_le_bytes()).unwrap();
    }
}

#[cfg(feature = "decode")]
#[test]
fn decoded_file_seeks_sample_exactly() {
    use deckengine::{DecodedSource, FrameSource};
    let dir = std::env::temp_dir().join(format!("deckengine-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("ramp.wav");
    write_ramp_wav(&path, 44_100, 30_000);

    let mut src = DecodedSource::open(&path).unwrap();
    assert_eq!(src.sample_rate(), 44_100);
    assert_eq!(src.duration_frames(), Some(30_000));
    let mut buf = vec![0.0f32; 64];
    for target in [12_345u64, 5, 29_990, 0] {
        src.seek(target).unwrap();
        let n = src.read(&mut buf).unwrap();
        assert!(n > 0);
        let got = (buf[0] * 32_768.0).round() as u64;
        assert_eq!(got, target % 32_768, "seek to {target}");
    }

    // Through the engine: load, play, analyse.
    let (h, mut p) = create(48_000, &EngineConfig::default());
    let info = h.load_file(DeckId::A, &path).unwrap();
    assert_eq!(info.sample_rate, 44_100);
    h.play(DeckId::A).unwrap();
    wait_for_buffer();
    let _ = render(&mut p, 4_800, 128);
    let st = h.deck_status(DeckId::A);
    assert!((st.position_seconds - 0.1).abs() < 0.002, "{st:?}");
    assert!((st.duration_seconds.unwrap() - 30_000.0 / 44_100.0).abs() < 1e-6);

    let loudness = deckengine::analyze_file(&path).unwrap();
    assert!(loudness.integrated_lufs > -70.0);
    std::fs::remove_dir_all(&dir).ok();
}

/// For every file in `DECKENGINE_TEST_FILES` (paths separated by the OS path separator):
/// after a seek, decoded samples must equal those at the same position of a linear decode.
/// Run with: `DECKENGINE_TEST_FILES=a.mp3:b.flac cargo test --release -- --ignored seek_matches`
#[cfg(feature = "decode")]
#[test]
#[ignore]
fn seek_matches_linear_decode() {
    use deckengine::{DecodedSource, FrameSource};
    let files = std::env::var_os("DECKENGINE_TEST_FILES").expect("set DECKENGINE_TEST_FILES");
    for path in std::env::split_paths(&files) {
        let mut src = DecodedSource::open(&path).unwrap();
        let mut all = Vec::new();
        let mut buf = vec![0.0f32; 4096 * 2];
        loop {
            let n = src.read(&mut buf).unwrap();
            if n == 0 {
                break;
            }
            all.extend_from_slice(&buf[..n * 2]);
        }
        let frames = all.len() / 2;
        let mut worst = 0usize;
        for &frac in &[0.1f64, 0.37, 0.5, 0.81] {
            let target = (frames as f64 * frac) as usize;
            src.seek(target as u64).unwrap();
            let n = src.read(&mut buf[..2048]).unwrap();
            // Find the offset between the seeked data and the linear decode (0 = exact).
            let offset = (-2048i64..=2048)
                .min_by(|&a, &b| {
                    let err = |o: i64| -> f32 {
                        (0..n * 2)
                            .map(|i| {
                                let j = (target as i64 * 2 + o * 2 + i as i64).clamp(0, all.len() as i64 - 1);
                                (buf[i] - all[j as usize]).abs()
                            })
                            .sum()
                    };
                    err(a).total_cmp(&err(b))
                })
                .unwrap();
            worst = worst.max(offset.unsigned_abs() as usize);
        }
        eprintln!("{}: {frames} frames, worst seek offset {worst} frames", path.display());
        assert_eq!(worst, 0, "{}: seek is off by {worst} frames", path.display());
    }
}
