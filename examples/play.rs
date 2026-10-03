// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Jason-Marshall Fastner <jasonfastner@protonmail.com>

//! Plays a file (or a test tone) and prints live latency / load statistics.
//!
//! ```text
//! cargo run --release --example play -- [FILE] [--mix FILE2] [--buffer 128] [--host alsa]
//!                                       [--device NAME] [--rate 48000] [--seconds 10]
//!                                       [--normalize -14] [--mute] [--list]
//! ```
//!
//! With `--mix FILE2` it crossfades into the second file halfway through, beat-accurately
//! scheduled on the engine clock.

use std::time::{Duration, Instant};

use deckengine::{DeckId, Engine, EngineConfig, OutputConfig, PcmSource, When};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut file = None;
    let mut mix = None;
    let mut out = OutputConfig::default();
    let mut seconds = 10.0f64;
    let mut normalize = None;
    let mut mute = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--buffer" => out.buffer_frames = args.next().ok_or("--buffer N")?.parse()?,
            "--host" => out.host = Some(args.next().ok_or("--host NAME")?),
            "--device" => out.device = Some(args.next().ok_or("--device NAME")?),
            "--rate" => out.sample_rate = Some(args.next().ok_or("--rate HZ")?.parse()?),
            "--seconds" => seconds = args.next().ok_or("--seconds S")?.parse()?,
            "--normalize" => normalize = Some(args.next().ok_or("--normalize LUFS")?.parse()?),
            "--mute" => mute = true,
            "--mix" => mix = Some(args.next().ok_or("--mix FILE")?),
            "--list" => {
                for (host, id, name) in deckengine::output::list_devices() {
                    println!("{host:>9}  {id:<45}  {name}");
                }
                return Ok(());
            }
            "-h" | "--help" => {
                println!("see the doc comment at the top of examples/play.rs");
                return Ok(());
            }
            f => file = Some(f.to_string()),
        }
    }

    let engine = Engine::start(out, EngineConfig::default())?;
    let info = engine.output_info().clone();
    println!(
        "output: {} / {} / {}  ·  {} Hz  ·  {} ch  ·  requested period {} frames ({:.2} ms)",
        info.backend,
        info.host,
        info.device,
        info.sample_rate,
        info.channels,
        info.buffer_frames,
        info.buffer_frames as f64 * 1000.0 / info.sample_rate as f64
    );

    match &file {
        Some(f) => {
            let li = engine.load_file(DeckId::A, f)?;
            println!("deck A: {f} ({} Hz, {:.1} s)", li.sample_rate, li.duration_seconds().unwrap_or(0.0));
        }
        None => {
            // 440 Hz test tone with a click every 500 ms.
            let sr = info.sample_rate;
            let n = sr as usize * (seconds as usize + 1);
            let mono: Vec<f32> = (0..n)
                .map(|i| {
                    let t = i as f32 / sr as f32;
                    let click = if i % (sr as usize / 2) < 64 { 0.5 } else { 0.0 };
                    0.2 * (2.0 * std::f32::consts::PI * 440.0 * t).sin() + click
                })
                .collect();
            engine.load_source(DeckId::A, Box::new(PcmSource::from_mono(&mono, sr)))?;
            println!("deck A: test tone");
        }
    }
    if let Some(m) = &mix {
        engine.load_file(DeckId::B, m)?;
        println!("deck B: {m}");
    }
    engine.set_normalization(normalize)?;
    if mute {
        // Full signal path runs, just at zero master volume (latency stats stay valid).
        engine.set_master_volume(0.0)?;
    }
    engine.play(DeckId::A)?;

    let start = Instant::now();
    let mut mixed = false;
    std::thread::sleep(Duration::from_millis(500));
    engine.reset_stats();
    while start.elapsed().as_secs_f64() < seconds {
        std::thread::sleep(Duration::from_secs(1));
        if mix.is_some() && !mixed && start.elapsed().as_secs_f64() > seconds / 2.0 {
            // Start B and fade over 4 s, both exactly 100 ms from now on the engine clock.
            let at = engine.clock() + engine.seconds_to_frames(0.1);
            engine.play_at(DeckId::B, When::At(at))?;
            engine.crossfade(1.0, 4.0, When::At(at))?;
            mixed = true;
        }
        let s = engine.stats();
        let a = engine.deck_status(DeckId::A);
        println!(
            "t={:5.1}s  buf {:>4} fr = {:.2} ms  ·  output latency {:.2} ms (device {:.2} + engine {:.2})  ·  dsp {:.1}% avg {:.0} µs max {:.0} µs  ·  misses {}  xruns {}  underruns {}  ·  peak {:.2}",
            a.position_seconds,
            s.buffer_frames,
            s.buffer_ms,
            s.output_latency_ms,
            s.device_latency_ms,
            s.engine_latency_ms,
            s.dsp_load * 100.0,
            s.process_avg_us,
            s.process_max_us,
            s.deadline_misses,
            s.backend_errors,
            a.underruns,
            s.peak_l.max(s.peak_r),
        );
        if a.ended && !mixed {
            break;
        }
    }
    Ok(())
}
