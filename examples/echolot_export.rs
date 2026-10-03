// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Jason-Marshall Fastner <jasonfastner@protonmail.com>

//! Writes Echolot fingerprints as test data for the server:
//! `<id>.ech` (whole track) and `<id>_q<k>_<start>.ech` (noisy 10 s queries, 3 dB SNR).
//! Format: "ECH1", u32 count, then per landmark u32 hash + u32 frame (little endian).
//! Usage: cargo run --release --example echolot_export -- out/ a.mp3 b.mp3 ...
use deckengine::{DecodedSource, FrameSource, echolot};
use std::io::Write;

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

fn noisy(x: &[f32], snr_db: f32, seed: u64) -> Vec<f32> {
    let p = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
    let amp = p / 10f32.powf(snr_db / 20.0) * 3f32.sqrt();
    let mut s = seed;
    x.iter()
        .map(|&v| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            0.5 * v + amp * ((s >> 33) as f32 / (1u64 << 31) as f32 - 1.0)
        })
        .collect()
}

fn write(path: &std::path::Path, fp: &[echolot::Landmark]) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    f.write_all(b"ECH1").unwrap();
    f.write_all(&(fp.len() as u32).to_le_bytes()).unwrap();
    for l in fp {
        f.write_all(&l.hash.to_le_bytes()).unwrap();
        f.write_all(&l.frame.to_le_bytes()).unwrap();
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let out = std::path::PathBuf::from(args.next().expect("Ausgabeordner"));
    std::fs::create_dir_all(&out).unwrap();
    for path in args {
        let id = std::path::Path::new(&path).file_stem().unwrap().to_string_lossy().to_string();
        let (pcm, rate) = decode(&path);
        write(&out.join(format!("{id}.ech")), &echolot::fingerprint(&pcm, rate));
        let secs = pcm.len() / rate as usize;
        for k in 0..3 {
            let start = secs.saturating_sub(12) * (2 * k + 1) / 6;
            let a = start * rate as usize;
            let q = noisy(&pcm[a..a + 10 * rate as usize], 3.0, k as u64 + 11);
            write(&out.join(format!("{id}_q{k}_{start}.ech")), &echolot::fingerprint(&q, rate));
        }
        println!("{id}: {secs} s");
    }
}
