// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Jason-Marshall Fastner <jasonfastner@protonmail.com>

//! Variable-ratio windowed-sinc interpolator.
//!
//! Used by the decks for both sample-rate conversion (e.g. 44.1 kHz file → 48 kHz device) and
//! varispeed (tempo/pitch fader). 16 taps, 512 table phases with linear interpolation between
//! phases, Kaiser window. At a ratio of exactly 1 it is bit-transparent (phase 0 is a unit
//! impulse), and its group delay is 8 source frames (≈0.18 ms at 44.1 kHz).

use std::sync::OnceLock;

pub const TAPS: usize = 16;
/// Delay of the interpolator in source frames.
pub const HALF: usize = TAPS / 2;
const PHASES: usize = 512;
const KAISER_BETA: f64 = 8.0;

type Table = [[f32; TAPS]; PHASES + 1];

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let y = x * x / 4.0;
    for k in 1..50 {
        term *= y / (k as f64 * k as f64);
        sum += term;
        if term < 1e-12 * sum {
            break;
        }
    }
    sum
}

fn table() -> &'static Table {
    static TABLE: OnceLock<Box<Table>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = Box::new([[0.0f32; TAPS]; PHASES + 1]);
        let i0_beta = bessel_i0(KAISER_BETA);
        for (p, row) in t.iter_mut().enumerate() {
            let frac = p as f64 / PHASES as f64;
            let mut sum = 0.0;
            let mut vals = [0.0f64; TAPS];
            for (j, v) in vals.iter_mut().enumerate() {
                let tau = j as f64 - (HALF as f64 - 1.0) - frac;
                let sinc = if tau.abs() < 1e-12 {
                    1.0
                } else {
                    (std::f64::consts::PI * tau).sin() / (std::f64::consts::PI * tau)
                };
                let r = tau / HALF as f64;
                let win = if r.abs() >= 1.0 {
                    0.0
                } else {
                    bessel_i0(KAISER_BETA * (1.0 - r * r).sqrt()) / i0_beta
                };
                *v = sinc * win;
                sum += *v;
            }
            // Normalise DC gain to exactly 1 for every phase.
            for (dst, v) in row.iter_mut().zip(vals) {
                *dst = (v / sum) as f32;
            }
        }
        t
    })
}

/// Stereo interpolator state. The caller pushes source frames and reads interpolated output
/// at fractional position `frac` in `[0, 1)` between the two centre frames.
#[derive(Clone)]
pub struct Interpolator {
    // Double-written ring so the newest TAPS frames are always contiguous.
    buf: [[f32; 2]; TAPS * 2],
    idx: usize,
    table: &'static Table,
}

impl Default for Interpolator {
    fn default() -> Self {
        Self::new()
    }
}

impl Interpolator {
    pub fn new() -> Self {
        Self { buf: [[0.0; 2]; TAPS * 2], idx: 0, table: table() }
    }

    pub fn reset(&mut self) {
        self.buf = [[0.0; 2]; TAPS * 2];
        self.idx = 0;
    }

    #[inline(always)]
    pub fn push(&mut self, frame: [f32; 2]) {
        self.buf[self.idx] = frame;
        self.buf[self.idx + TAPS] = frame;
        self.idx = (self.idx + 1) % TAPS;
    }

    /// Interpolated frame at `frac` (0..1) past the centre frame.
    #[inline(always)]
    pub fn get(&self, frac: f64) -> [f32; 2] {
        let window = &self.buf[self.idx..self.idx + TAPS];
        let pf = frac * PHASES as f64;
        let p = (pf as usize).min(PHASES - 1);
        let t = (pf - p as f64) as f32;
        if p == 0 && t == 0.0 {
            return window[HALF - 1];
        }
        let (k0, k1) = (&self.table[p], &self.table[p + 1]);
        let (mut l, mut r) = (0.0f32, 0.0f32);
        for j in 0..TAPS {
            let k = k0[j] + (k1[j] - k0[j]) * t;
            l += window[j][0] * k;
            r += window[j][1] * k;
        }
        [l, r]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unity_ratio_is_transparent() {
        let mut ip = Interpolator::new();
        let input: Vec<f32> = (0..100).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut out = Vec::new();
        for &x in &input {
            ip.push([x, -x]);
            out.push(ip.get(0.0)[0]);
        }
        for i in HALF..100 {
            assert_eq!(out[i], input[i - HALF]);
        }
    }

    #[test]
    fn upsampled_sine_is_clean() {
        // 1 kHz at 44.1 kHz → 48 kHz: the error against the ideal sine must be tiny.
        let (src, dst) = (44_100.0f64, 48_000.0f64);
        let step = src / dst;
        let mut ip = Interpolator::new();
        let mut n: i64 = 0;
        let mut frac = 0.0;
        let sig = |i: f64| (2.0 * std::f64::consts::PI * 1000.0 * i / src).sin() as f32;
        for _ in 0..TAPS {
            ip.push([sig(n as f64); 2]);
            n += 1;
        }
        let mut max_err: f32 = 0.0;
        for _ in 0..10_000 {
            let centre = (n - HALF as i64 - 1) as f64;
            let y = ip.get(frac)[0];
            max_err = max_err.max((y - sig(centre + frac)).abs());
            frac += step;
            while frac >= 1.0 {
                frac -= 1.0;
                ip.push([sig(n as f64); 2]);
                n += 1;
            }
        }
        assert!(max_err < 2e-3, "max error {max_err}");
    }
}
