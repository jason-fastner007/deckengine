// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Jason-Marshall Fastner <jasonfastner@protonmail.com>

//! Echolot: audio fingerprint for song recognition (landmark method).
//!
//! Prominent peaks are picked from the spectrogram (11 025 Hz, 93 ms window, 23 ms hop)
//! and connected in pairs: each peak (anchor) forms a pair with up to [`FAN_OUT`] later peaks
//! nearby. A pair yields a 24-bit hash from both frequencies and their distance, plus
//! the anchor's time. Such pairs survive volume, EQ, compression and noise (a microphone in
//! a club) well, and no audio can be recovered from them.
//!
//! Matching: an excerpt matches a track if many of its hashes occur there with the **same**
//! time offset ([`best_offset`]).

use realfft::RealFftPlanner;

/// Sample rate of the analysis.
pub const RATE: u32 = 11_025;
const FFT: usize = 1024;
const HOP: usize = 256;
/// Frequency range used (bins): ~86 Hz … 4.3 kHz.
const BIN_LO: usize = 8;
const BIN_HI: usize = 400;
/// Neighbourhood for peaks: ±bins, ±frames.
const PEAK_DF: usize = 12;
const PEAK_DT: usize = 6;
/// At most this many peaks per second (the strongest).
const PEAKS_PER_SECOND: usize = 30;
/// Target zone of an anchor: 1 … 63 frames later (up to ~1.5 s), ±100 bins.
const DT_MAX: usize = 63;
const DF_MAX: i32 = 100;
const FAN_OUT: usize = 5;

/// Duration of one frame (time resolution of the hashes) in ms.
pub const FRAME_MS: f64 = HOP as f64 * 1000.0 / RATE as f64;

/// A landmark: hash and frame of the anchor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Landmark {
    pub hash: u32,
    pub frame: u32,
}

/// Fingerprint of mono PCM at any sample rate.
pub fn fingerprint(mono: &[f32], sample_rate: u32) -> Vec<Landmark> {
    let x = if sample_rate == RATE {
        std::borrow::Cow::Borrowed(mono)
    } else {
        std::borrow::Cow::Owned(crate::fakeprint::resample(mono, sample_rate, RATE))
    };
    let spec = spectrogram(&x);
    let peaks = pick_peaks(&spec);
    pair(&peaks)
}

/// Log magnitude spectrum per frame (only the range in use).
fn spectrogram(x: &[f32]) -> Vec<Vec<f32>> {
    if x.len() < FFT {
        return Vec::new();
    }
    let frames = (x.len() - FFT) / HOP + 1;
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT);
    let window: Vec<f32> =
        (0..FFT).map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / FFT as f32).cos()).collect();
    let mut buf = fft.make_input_vec();
    let mut out = fft.make_output_vec();
    let mut spec = Vec::with_capacity(frames);
    for f in 0..frames {
        let s = &x[f * HOP..f * HOP + FFT];
        for (b, (v, w)) in buf.iter_mut().zip(s.iter().zip(&window)) {
            *b = v * w;
        }
        fft.process(&mut buf, &mut out).expect("FFT-Größe passt");
        spec.push(out[BIN_LO..BIN_HI].iter().map(|c| (c.norm_sqr() + 1e-10).ln()).collect());
    }
    spec
}

/// Sliding maximum over ±`r` (Van Herk/Gil-Werman would be faster; the windows are small).
fn max_filter(v: &[f32], r: usize) -> Vec<f32> {
    (0..v.len())
        .map(|i| {
            let lo = i.saturating_sub(r);
            let hi = (i + r + 1).min(v.len());
            v[lo..hi].iter().copied().fold(f32::MIN, f32::max)
        })
        .collect()
}

/// Peaks: local maxima in the time-frequency window, clearly above the frame mean,
/// keeping only the strongest per second.
fn pick_peaks(spec: &[Vec<f32>]) -> Vec<(usize, usize)> {
    let n = spec.len();
    if n == 0 {
        return Vec::new();
    }
    let bins = spec[0].len();
    // Maximise along frequency first, then along time (separable).
    let fmax: Vec<Vec<f32>> = spec.iter().map(|row| max_filter(row, PEAK_DF)).collect();
    let mut tmax = vec![vec![0.0f32; bins]; n];
    let mut col = vec![0.0f32; n];
    for b in 0..bins {
        for t in 0..n {
            col[t] = fmax[t][b];
        }
        for (t, v) in max_filter(&col, PEAK_DT).into_iter().enumerate() {
            tmax[t][b] = v;
        }
    }
    let mut candidates: Vec<(usize, usize, f32)> = Vec::new();
    for t in 0..n {
        let mean = spec[t].iter().sum::<f32>() / bins as f32;
        for b in 0..bins {
            let v = spec[t][b];
            // ln magnitude: +2.3 ≈ 10 dB above the frame mean.
            if v >= tmax[t][b] && v > mean + 2.3 {
                candidates.push((t, b, v));
            }
        }
    }
    // Limit density: keep the strongest peaks per second.
    let per_block = (1000.0 / FRAME_MS).round() as usize;
    let mut peaks = Vec::new();
    for block in candidates.chunk_by(|a, b| a.0 / per_block == b.0 / per_block) {
        let mut v = block.to_vec();
        v.sort_by(|a, b| b.2.total_cmp(&a.2));
        peaks.extend(v.into_iter().take(PEAKS_PER_SECOND).map(|(t, b, _)| (t, b)));
    }
    peaks.sort_unstable();
    peaks
}

/// Hash from anchor bin, target bin and frame distance: 9 + 9 + 6 bits.
fn hash(f1: usize, f2: usize, dt: usize) -> u32 {
    ((f1 as u32 & 0x1FF) << 15) | ((f2 as u32 & 0x1FF) << 6) | (dt as u32 & 0x3F)
}

fn pair(peaks: &[(usize, usize)]) -> Vec<Landmark> {
    let mut out = Vec::with_capacity(peaks.len() * FAN_OUT);
    for (i, &(t1, f1)) in peaks.iter().enumerate() {
        let mut fan = 0;
        for &(t2, f2) in &peaks[i + 1..] {
            let dt = t2 - t1;
            if dt > DT_MAX {
                break;
            }
            if dt == 0 || (f2 as i32 - f1 as i32).abs() > DF_MAX {
                continue;
            }
            out.push(Landmark { hash: hash(f1, f2, dt), frame: t1 as u32 });
            fan += 1;
            if fan == FAN_OUT {
                break;
            }
        }
    }
    out
}

/// Matches an excerpt against a reference fingerprint: best time offset (frames) and
/// how many hashes confirm it. The server does the same across all tracks; this is for tests/tools.
pub fn best_offset(reference: &[Landmark], query: &[Landmark]) -> Option<(i64, usize)> {
    use std::collections::HashMap;
    let mut by_hash: HashMap<u32, Vec<u32>> = HashMap::new();
    for l in reference {
        by_hash.entry(l.hash).or_default().push(l.frame);
    }
    let mut votes: HashMap<i64, usize> = HashMap::new();
    for q in query {
        if let Some(frames) = by_hash.get(&q.hash) {
            for &f in frames {
                *votes.entry(f as i64 - q.frame as i64).or_default() += 1;
            }
        }
    }
    votes.into_iter().max_by_key(|&(_, n)| n)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pseudo-music: random tones (with overtones) on an eighth-note grid, deterministic.
    fn music(secs: f32, seed: u64) -> Vec<f32> {
        let mut s = seed;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let n = (secs * RATE as f32) as usize;
        let step = (RATE as f32 * 0.25) as usize;
        let mut x = vec![0.0f32; n];
        for start in (0..n).step_by(step) {
            for _ in 0..3 {
                let hz = 110.0 * 2f32.powf((rnd() % 36) as f32 / 12.0);
                for k in 0..step.min(n - start) {
                    let t = k as f32 / RATE as f32;
                    let env = (-t * 6.0).exp();
                    x[start + k] += 0.2 * env * ((std::f32::consts::TAU * hz * t).sin() + 0.5 * (std::f32::consts::TAU * 2.0 * hz * t).sin());
                }
            }
        }
        x
    }

    fn noise(n: usize, amp: f32, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                amp * ((s >> 33) as f32 / (1u64 << 31) as f32 - 1.0)
            })
            .collect()
    }

    #[test]
    fn excerpt_with_noise_finds_offset() {
        let track = music(60.0, 7);
        let full = fingerprint(&track, RATE);
        assert!(full.len() > 60 * 50, "zu wenige Landmarks: {}", full.len());
        // 10 s from 23.4 s, quieter and with noise (~12 dB SNR).
        let from = (23.4 * RATE as f32) as usize;
        let mut q: Vec<f32> = track[from..from + 10 * RATE as usize].iter().map(|v| v * 0.5).collect();
        let hiss = noise(q.len(), 0.03, 3);
        for (v, n) in q.iter_mut().zip(hiss) {
            *v += n;
        }
        let (off, votes) = best_offset(&full, &fingerprint(&q, RATE)).unwrap();
        let expect = (23.4 * 1000.0 / FRAME_MS).round() as i64;
        assert!((off - expect).abs() <= 1, "Versatz {off} statt {expect}");
        assert!(votes >= 20, "nur {votes} Treffer");
    }

    #[test]
    fn true_match_clearly_beats_other_track() {
        let a_track = music(60.0, 7);
        let a = fingerprint(&a_track, RATE);
        let from = 30 * RATE as usize;
        let own = fingerprint(&a_track[from..from + 10 * RATE as usize], RATE);
        let other = fingerprint(&music(10.0, 99), RATE);
        let (_, hit) = best_offset(&a, &own).unwrap();
        let (_, miss) = best_offset(&a, &other).unwrap_or((0, 0));
        assert!(hit >= 5 * miss.max(1), "Treffer {hit} vs. fremd {miss}");
    }

    #[test]
    fn resampled_input_matches() {
        // The same signal at 44.1 kHz must produce the same timeline.
        let x = music(20.0, 5);
        let hi = crate::fakeprint::resample(&x, RATE, 44_100);
        let (off, votes) = best_offset(&fingerprint(&x, RATE), &fingerprint(&hi, 44_100)).unwrap();
        assert!(off.abs() <= 1 && votes > 100, "Versatz {off}, {votes} Treffer");
    }
}
