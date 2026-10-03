// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Jason-Marshall Fastner <jasonfastner@protonmail.com>

//! Checks cue and drop detection on real tracks the way the app does it (byte-range excerpts +
//! envelope) and compares against the fully decoded track.
//! Usage: cargo run --release --example cue_check -- track.mp3 [waveform.json]
use deckengine::{DecodedSource, FrameSource, cue, dj, mp3clip};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let bytes = std::fs::read(path).unwrap();

    let mut src = DecodedSource::open(path).unwrap();
    let rate = src.sample_rate();
    let mut all = Vec::new();
    let mut buf = vec![0.0f32; 8192 * 2];
    loop {
        let n = src.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        all.extend(buf[..n * 2].chunks_exact(2).map(|f| 0.5 * (f[0] + f[1])));
    }
    let dur = all.len() as f64 / rate as f64 * 1000.0;
    let head = mp3clip::parse_head(&bytes[..16384]).unwrap();
    let clip = |from_ms: f64, to_ms: f64| {
        let a = head.byte_at(from_ms.max(0.0)) as usize;
        let b = (head.byte_at(to_ms.min(dur)) as usize + 4 * 418).min(bytes.len());
        mp3clip::decode_clip(&head, &bytes[a..b], a as u64).unwrap()
    };

    // Beat grid as in the app: 30 s from 35 %.
    let start = (dur * 0.35).min(dur - 30_000.0).max(0.0);
    let c = clip(start, start + 30_000.0);
    let grid = dj::analyze(&c.pcm, c.sample_rate, c.start_ms, dur, None).unwrap();
    let period = grid.period_ms();
    let body = Some(cue::bass_level_db(&c.pcm, c.sample_rate));

    // Cue: erste 20 s per Ausschnitt gegen Volldekodierung.
    let intro = clip(0.0, 20_000.0);
    let cue_clip = cue::detect_cue(&intro.pcm, intro.sample_rate, intro.start_ms);
    let cue_full = cue::detect_cue(&all[..(20.0 * rate as f64) as usize], rate, 0.0);

    // Drop: candidate from the waveform, precisely from 24 s around it.
    let env: Option<Vec<f32>> = args.get(2).map(|p| {
        let j = std::fs::read_to_string(p).unwrap();
        let h: f32 = j.split("\"height\":").nth(1).and_then(|s| s.split([',', '}']).next()).and_then(|s| s.trim().parse().ok()).unwrap_or(140.0);
        let list = j.split('[').nth(1).unwrap().split(']').next().unwrap();
        list.split(',').filter_map(|v| v.trim().parse::<f32>().ok()).map(|v| v / h).collect()
    });
    let cands = env.as_deref().map(|e| cue::drop_candidates(e, dur)).unwrap_or_default();
    let mut tried = 0;
    let mut cand = None;
    let drop_clip = cands.iter().find_map(|&t| {
        tried += 1;
        cand = Some(t);
        let d = clip(t - 12_000.0, t + 12_000.0);
        cue::detect_drop(&d.pcm, d.sample_rate, d.start_ms, period, grid.downbeat_ms, body)
    });
    // The app's way: 30 s sections from the start (4 s overlap) until there's a drop – max. 50 % or 150 s.
    let limit = (dur * 0.5).min(150_000.0);
    let (mut at, mut chunks, mut scan) = (0.0, 0, None);
    while at < limit && scan.is_none() {
        let d = clip(at, at + 30_000.0);
        chunks += 1;
        scan = cue::detect_drop(&d.pcm, d.sample_rate, d.start_ms, period, grid.downbeat_ms, body);
        at += 26_000.0;
    }
    println!("    Scan: {} nach {chunks} Abschnitt(en)", scan.map_or("—".into(), |d| format!("{:.2} s (Stärke {:.2})", d.ms / 1000.0, d.strength)));
    // Reference: first drop in the whole track (up to 75 %).
    let upto = (all.len() as f64 * 0.75) as usize;
    let drop_full = cue::detect_drop(&all[..upto], rate, 0.0, period, grid.downbeat_ms, body);

    let s = |v: Option<f64>| v.map_or("—".into(), |ms| format!("{:>2}:{:05.2}", (ms / 60_000.0) as u32, ms / 1000.0 % 60.0));
    println!(
        "{}  {:.0}s  {:.1} BPM | Cue Ausschnitt {} / voll {} | Kandidat {} ({}/{} versucht) → {} (Stärke {:.2}) | voll {}",
        path.rsplit('/').next().unwrap(),
        dur / 1000.0,
        grid.bpm,
        s(cue_clip),
        s(cue_full),
        s(cand),
        tried,
        cands.len(),
        s(drop_clip.map(|d| d.ms)),
        drop_clip.map_or(0.0, |d| d.strength),
        s(drop_full.map(|d| d.ms)),
    );

    // Bass level (< 150 Hz) per second around the found drop: quiet before, loud after?
    if let Some(d) = drop_clip.or(drop_full) {
        let sec = rate as usize;
        let at = (d.ms / 1000.0) as i64;
        let mut line = String::from("    Bass dB um Drop: ");
        for k in -6..=4 {
            let i = at + k;
            if i < 0 || (i as usize + 1) * sec > all.len() {
                continue;
            }
            let w = &all[i as usize * sec..(i as usize + 1) * sec];
            let lp = bass_rms(w, rate);
            line += &format!("{}{:.0} ", if k == 0 { "|" } else { "" }, 20.0 * lp.max(1e-6).log10());
        }
        println!("{line}");
    }
}

/// Rough bass level (first-order low-pass, 2×) of a window.
fn bass_rms(x: &[f32], rate: u32) -> f32 {
    let a = (-std::f32::consts::TAU * 150.0 / rate as f32).exp();
    let (mut y1, mut y2) = (0.0f32, 0.0f32);
    let mut e = 0.0;
    for &v in x {
        y1 = (1.0 - a) * v + a * y1;
        y2 = (1.0 - a) * y1 + a * y2;
        e += y2 * y2;
    }
    (e / x.len() as f32).sqrt()
}
