// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Jason-Marshall Fastner <jasonfastner@protonmail.com>

//! Echolot on real tracks: index all of them, then query degraded 10 s excerpts.
//! Usage: cargo run --release --example echolot_check -- a.mp3 b.mp3 ...
use deckengine::{DecodedSource, FrameSource, echolot};
use std::collections::HashMap;

fn decode(path: &str) -> (Vec<f32>, u32) {
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
    (all, rate)
}

/// "Phone in the room": quieter, 200 Hz high-pass + 3.5 kHz low-pass (first order), noise.
fn degrade(x: &[f32], rate: u32, snr_db: f32, seed: u64) -> Vec<f32> {
    let lp = (-std::f32::consts::TAU * 3500.0 / rate as f32).exp();
    let hp = (-std::f32::consts::TAU * 200.0 / rate as f32).exp();
    let (mut l, mut h, mut prev) = (0.0f32, 0.0f32, 0.0f32);
    let mut y: Vec<f32> = x
        .iter()
        .map(|&v| {
            l = (1.0 - lp) * v + lp * l;
            h = hp * (h + l - prev);
            prev = l;
            0.4 * h
        })
        .collect();
    let p = (y.iter().map(|v| v * v).sum::<f32>() / y.len() as f32).sqrt();
    let amp = p / 10f32.powf(snr_db / 20.0) * 3f32.sqrt();
    let mut s = seed;
    for v in y.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *v += amp * ((s >> 33) as f32 / (1u64 << 31) as f32 - 1.0);
    }
    y
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    let mut tracks = Vec::new();
    // Index: hash → (track, frame), as the server holds it.
    let mut index: HashMap<u32, Vec<(u16, u32)>> = HashMap::new();
    let t = std::time::Instant::now();
    let mut total = 0;
    for (id, p) in paths.iter().enumerate() {
        let (pcm, rate) = decode(p);
        let fp = echolot::fingerprint(&pcm, rate);
        total += fp.len();
        for l in &fp {
            index.entry(l.hash).or_default().push((id as u16, l.frame));
        }
        println!("{:<16} {:>4} s  {:>6} Landmarks ({:.0}/s)", p.rsplit('/').next().unwrap(), pcm.len() / rate as usize, fp.len(), fp.len() as f64 / (pcm.len() as f64 / rate as f64));
        tracks.push((pcm, rate));
    }
    println!("Index: {total} Landmarks, {} Hashes, {:?}\n", index.len(), t.elapsed());

    for snr in [10.0f32, 3.0, 0.0] {
        let (mut ok, mut n, mut margins, mut ms, mut track_ok) = (0, 0, Vec::new(), 0.0, 0);
        for (id, (pcm, rate)) in tracks.iter().enumerate() {
            let secs = pcm.len() / *rate as usize;
            for k in 0..5 {
                let start = (secs.saturating_sub(12)) * (2 * k + 1) / 10;
                let a = start * *rate as usize;
                let q = degrade(&pcm[a..a + 10 * *rate as usize], *rate, snr, (id * 10 + k) as u64 + 1);
                let t = std::time::Instant::now();
                let ql = echolot::fingerprint(&q, *rate);
                let mut votes: HashMap<(u16, i64), u32> = HashMap::new();
                for l in &ql {
                    if let Some(post) = index.get(&l.hash) {
                        for &(tid, f) in post {
                            *votes.entry((tid, f as i64 - l.frame as i64)).or_default() += 1;
                        }
                    }
                }
                // Best hit per track, then the gap to the second-best track.
                let mut best: HashMap<u16, (u32, i64)> = HashMap::new();
                for (&(tid, off), &v) in &votes {
                    let e = best.entry(tid).or_insert((0, 0));
                    if v > e.0 {
                        *e = (v, off);
                    }
                }
                let mut ranked: Vec<(u16, (u32, i64))> = best.into_iter().collect();
                ranked.sort_by(|a, b| b.1.0.cmp(&a.1.0));
                ms += t.elapsed().as_secs_f64() * 1000.0;
                n += 1;
                let (tid, (v, off)) = ranked[0];
                let second = ranked.get(1).map_or(0, |r| r.1.0);
                let off_s = off as f64 * echolot::FRAME_MS / 1000.0;
                if tid as usize == id {
                    track_ok += 1;
                }
                if tid as usize == id && (off_s - start as f64).abs() >= 0.1 && snr == 0.0 {
                    println!("    Track {id} richtig, Stelle {off_s:.2} s statt {start} s ({v} Stimmen)");
                }
                if tid as usize != id && snr == 0.0 {
                    println!("    FALSCH: Track {id} @ {start} s → Track {tid} ({v} Stimmen, eigener: {})", ranked.iter().find(|r| r.0 as usize == id).map_or(0, |r| r.1.0));
                }
                if tid as usize == id && (off_s - start as f64).abs() < 0.1 {
                    ok += 1;
                    margins.push(v as f64 / second.max(1) as f64);
                }
            }
        }
        margins.sort_by(f64::total_cmp);
        println!(
            "SNR {snr:>4} dB: Track {track_ok}/{n}, Track+Stelle {ok}/{n}, Abstand zum 2. Track min {:.1}× median {:.1}×, Ø {:.1} ms je Abfrage (Fingerabdruck + Suche)",
            margins.first().copied().unwrap_or(0.0),
            margins.get(margins.len() / 2).copied().unwrap_or(0.0),
            ms / n as f64
        );
    }
}
