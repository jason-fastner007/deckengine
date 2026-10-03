// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Jason-Marshall Fastner <jasonfastner@protonmail.com>

//! The audio fingerprint music generators leave behind, after "A Fourier Explanation of
//! AI-Music Artifacts" (Afchar et al., ISMIR 2025), with the classifier of
//! lofcz/ai-music-detector. Rust port of KittyTune's `AiFakeprint` (PR #43).
//!
//! The decoders of Suno, Udio & co. upsample with transposed convolutions, which leaves peaks at
//! regular frequency intervals in everything they render. The fakeprint isolates them: the
//! song's average spectrum at 16 kHz between 1 and 8 kHz, minus its lower envelope. A logistic
//! regression over its 3585 bins gives the probability that a generator rendered the audio.
//!
//! Every step mirrors the reference inference (torchaudio's resampler and spectrogram, scipy's
//! minimum filter), because the weights only fit features computed exactly that way.
//!
//! The weights in `assets/ai_music_fakeprint.bin` come from lofcz/ai-music-detector:
//! MIT License, Copyright (c) 2026 Matěj Štágl. Permission is hereby granted, free of charge, to
//! any person obtaining a copy of this software and associated documentation files (the
//! "Software"), to deal in the Software without restriction, including without limitation the
//! rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
//! Software, and to permit persons to whom the Software is furnished to do so, subject to the
//! following conditions: The above copyright notice and this permission notice shall be included
//! in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS",
//! WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
//! OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//! THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
//! IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
//! SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use realfft::RealFftPlanner;

pub const SAMPLE_RATE: u32 = 16_000;
const N_FFT: usize = 8192;
const HOP: usize = N_FFT / 2;
const BIN_MIN: usize = 512; // 1000 Hz at 16 kHz / 8192
const BIN_MAX: usize = 4096; // 8000 Hz
pub const FEATURES: usize = BIN_MAX - BIN_MIN + 1;
const HULL_SIZE: usize = 10;
const MIN_DB: f64 = -45.0;
const MAX_DB: f64 = 5.0;
// torchaudio's default resampler: windowed sinc with a Hann window.
const LOWPASS_WIDTH: f64 = 6.0;
const ROLLOFF: f64 = 0.99;

static WEIGHTS: &[u8] = include_bytes!("../assets/ai_music_fakeprint.bin");

/// sigmoid(weights · fakeprint + bias) with the bundled weights.
pub fn probability(fakeprint: &[f32]) -> f64 {
    debug_assert_eq!(WEIGHTS.len(), (FEATURES + 1) * 4);
    let w = |i: usize| f32::from_le_bytes(WEIGHTS[i * 4..i * 4 + 4].try_into().unwrap()) as f64;
    let mut logit = w(FEATURES);
    for (i, &v) in fakeprint.iter().enumerate().take(FEATURES) {
        logit += w(i) * v as f64;
    }
    1.0 / (1.0 + (-logit).exp())
}

pub fn rms(x: &[f32]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / x.len() as f64).sqrt()
}

/// AI probability of mono audio at any rate, or `None` for silence / too little audio.
pub fn detect(mono: &[f32], sample_rate: u32) -> Option<f64> {
    if rms(mono) < 1e-3 || mono.len() < sample_rate as usize {
        return None;
    }
    let x = resample(mono, sample_rate, SAMPLE_RATE);
    Some(probability(&fakeprint(&x)))
}

/// The fakeprint of mono audio already at [`SAMPLE_RATE`].
pub fn fakeprint(audio16k: &[f32]) -> Vec<f32> {
    let mean_db = mean_db_spectrum(audio16k);
    let spectrum = &mean_db[BIN_MIN..=BIN_MAX];
    let mut residue = vec![0.0f64; FEATURES];
    let mut max_residue = 0.0f64;
    for i in 0..FEATURES {
        // scipy's minimum_filter1d(size = 10, mode = "nearest") looks at i - 5 .. i + 4.
        let mut hull = f64::MAX;
        for j in (i as isize - (HULL_SIZE / 2) as isize)..(i as isize - (HULL_SIZE / 2) as isize + HULL_SIZE as isize) {
            hull = hull.min(spectrum[j.clamp(0, FEATURES as isize - 1) as usize]);
        }
        let hull = hull.max(MIN_DB);
        residue[i] = (spectrum[i] - hull).clamp(0.0, MAX_DB);
        max_residue = max_residue.max(residue[i]);
    }
    let scale = max_residue + 1e-6;
    residue.iter().map(|r| (r / scale) as f32).collect()
}

/// torch.stft(n_fft = 8192, hop = 4096, periodic Hann, center = True with reflect padding) as
/// power, in dB per frame, averaged over frames.
pub fn mean_db_spectrum(audio: &[f32]) -> Vec<f64> {
    let pad = N_FFT / 2;
    assert!(audio.len() > pad, "need more than {pad} samples");
    let frames = 1 + audio.len() / HOP;
    let window: Vec<f64> =
        (0..N_FFT).map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos()).collect();
    let bins = N_FFT / 2 + 1;
    let mut sum = vec![0.0f64; bins];
    let mut planner = RealFftPlanner::<f64>::new();
    let r2c = planner.plan_fft_forward(N_FFT);
    let mut input = r2c.make_input_vec();
    let mut spec = r2c.make_output_vec();
    let mut scratch = r2c.make_scratch_vec();
    let last = audio.len() as isize - 1;
    for frame in 0..frames {
        let start = (frame * HOP) as isize - pad as isize;
        for (n, v) in input.iter_mut().enumerate() {
            let mut i = start + n as isize;
            if i < 0 {
                i = -i;
            } else if i > last {
                i = 2 * last - i;
            }
            *v = audio[i as usize] as f64 * window[n];
        }
        r2c.process_with_scratch(&mut input, &mut spec, &mut scratch).expect("fft");
        for (k, s) in sum.iter_mut().enumerate() {
            *s += 10.0 * spec[k].norm_sqr().clamp(1e-10, 1e6).log10();
        }
    }
    sum.iter_mut().for_each(|s| *s /= frames as f64);
    sum
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// `torchaudio.functional.resample` with its defaults (sinc_interp_hann, lowpass width 6,
/// rolloff 0.99). Used for every rate conversion in the analysis, not only here.
pub fn resample(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to {
        return input.to_vec();
    }
    let d = gcd(from, to);
    let orig = (from / d) as usize;
    let new = (to / d) as usize;
    let base_freq = orig.min(new) as f64 * ROLLOFF;
    let width = (LOWPASS_WIDTH * orig as f64 / base_freq).ceil() as usize;
    let taps = 2 * width + orig;

    // One kernel per output phase; outside the sinc's main lobes the Hann window is exactly zero.
    let mut kernels = vec![vec![0.0f64; taps]; new];
    let mut first = vec![0usize; new];
    let mut last = vec![0usize; new];
    for (phase, kernel) in kernels.iter_mut().enumerate() {
        for (k, slot) in kernel.iter_mut().enumerate() {
            let t = (-(phase as f64) / new as f64 + (k as f64 - width as f64) / orig as f64) * base_freq;
            // torchaudio clamps t to ±width, where the window is cos²(±π/2) ≈ 4e-33 instead of
            // an exact zero. Treating it as zero skips ~90 % of the taps and changes nothing.
            if t.abs() >= LOWPASS_WIDTH {
                continue;
            }
            let w = (t * std::f64::consts::PI / LOWPASS_WIDTH / 2.0).cos().powi(2);
            let x = t * std::f64::consts::PI;
            let sinc = if x == 0.0 { 1.0 } else { x.sin() / x };
            *slot = sinc * w * (base_freq / orig as f64);
        }
        first[phase] = kernel.iter().position(|&v| v != 0.0).unwrap_or(0);
        last[phase] = kernel.iter().rposition(|&v| v != 0.0).unwrap_or(0);
    }

    let len = input.len();
    let out_len = ((new * len) as f64 / orig as f64).ceil() as usize;
    let mut out = vec![0.0f32; out_len];
    let blocks = len / orig + 1;
    for block in 0..blocks {
        let base = (block * orig) as isize - width as isize;
        for phase in 0..new {
            let o = block * new + phase;
            if o >= out_len {
                break;
            }
            let kernel = &kernels[phase];
            let mut acc = 0.0f64;
            for (k, &kv) in kernel.iter().enumerate().take(last[phase] + 1).skip(first[phase]) {
                let j = base + k as isize;
                if j >= 0 && (j as usize) < len {
                    acc += kv * input[j as usize] as f64;
                }
            }
            out[o] = acc as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Must match signal() in the reference script (make_reference.py) sample for sample.
    fn signal(rate: u32, seconds: u32) -> Vec<f32> {
        let n = (rate * seconds) as usize;
        let mut state: i64 = 12345;
        (0..n)
            .map(|i| {
                let t = i as f64 / rate as f64;
                let pi = std::f64::consts::PI;
                let tone = 0.3 * (2.0 * pi * 220.0 * t).sin()
                    + 0.2 * (2.0 * pi * 440.0 * t + 0.5).sin()
                    + 0.1 * (2.0 * pi * 1760.5 * t).sin()
                    + 0.05 * (2.0 * pi * 3520.25 * t).sin()
                    + 0.02 * (2.0 * pi * 5000.0 * t).sin() * (2.0 * pi * 0.5 * t).sin();
                state = (state * 1_103_515_245 + 12_345) % 2_147_483_648;
                (tone + 0.05 * (state as f64 / 2_147_483_648.0 - 0.5)) as f32
            })
            .collect()
    }

    fn floats(name: &str) -> Vec<f32> {
        let bytes = std::fs::read(format!("{}/tests/data/fakeprint/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap();
        bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }

    fn check(rate: u32) {
        let resampled = resample(&signal(rate, 8), rate, SAMPLE_RATE);
        assert_eq!(resampled.len(), 128_000);
        let expected = floats(&format!("resampled_{rate}.bin"));
        let err = expected.iter().zip(&resampled).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(err < 1e-5, "resampling differs by {err}");

        let fp = fakeprint(&resampled);
        let expected = floats(&format!("fakeprint_{rate}.bin"));
        assert_eq!(fp.len(), FEATURES);
        let err = expected.iter().zip(&fp).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(err < 1e-3, "fakeprint differs by {err}");
        let (pr, pe) = (probability(&fp), probability(&expected));
        let logit = |p: f64| (p / (1.0 - p)).ln();
        assert!((logit(pr) - logit(pe)).abs() < 0.05, "logit {} vs {}", logit(pr), logit(pe));
    }

    #[test]
    fn matches_python_reference_44k() {
        check(44_100);
    }

    #[test]
    fn matches_python_reference_48k() {
        check(48_000);
    }

    #[test]
    fn silence_is_not_judged() {
        assert_eq!(detect(&vec![0.0; 48_000], 48_000), None);
    }
}
