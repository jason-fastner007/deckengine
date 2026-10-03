// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Jason-Marshall Fastner <jasonfastner@protonmail.com>

//! Second-order IIR sections (transposed direct form II), designed with the bilinear transform.

use super::undenormal;
use std::f64::consts::PI;

/// Normalised biquad coefficients (`a0 == 1`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coeffs {
    pub b0: f32,
    pub b1: f32,
    pub b2: f32,
    pub a1: f32,
    pub a2: f32,
}

impl Coeffs {
    pub const IDENTITY: Coeffs = Coeffs { b0: 1.0, b1: 0.0, b2: 0.0, a1: 0.0, a2: 0.0 };

    fn normalise(b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64) -> Self {
        Coeffs {
            b0: (b0 / a0) as f32,
            b1: (b1 / a0) as f32,
            b2: (b2 / a0) as f32,
            a1: (a1 / a0) as f32,
            a2: (a2 / a0) as f32,
        }
    }

    /// Clamps a cutoff frequency into the usable range below Nyquist.
    fn prewarp(freq: f64, sample_rate: f64) -> f64 {
        let f = freq.clamp(1.0, sample_rate * 0.49);
        (PI * f / sample_rate).tan()
    }

    /// Second-order low-pass with quality factor `q`.
    pub fn lowpass(freq: f64, q: f64, sample_rate: f64) -> Self {
        let k = Self::prewarp(freq, sample_rate);
        let k2 = k * k;
        let a0 = 1.0 + k / q + k2;
        Self::normalise(k2, 2.0 * k2, k2, a0, 2.0 * (k2 - 1.0), 1.0 - k / q + k2)
    }

    /// Second-order high-pass with quality factor `q`.
    pub fn highpass(freq: f64, q: f64, sample_rate: f64) -> Self {
        let k = Self::prewarp(freq, sample_rate);
        let k2 = k * k;
        let a0 = 1.0 + k / q + k2;
        Self::normalise(1.0, -2.0, 1.0, a0, 2.0 * (k2 - 1.0), 1.0 - k / q + k2)
    }

    /// Second-order all-pass. With `q = 1/√2` this has exactly the phase response of a
    /// Linkwitz-Riley 4th-order crossover sum (LP + HP), which is what band alignment needs.
    pub fn allpass(freq: f64, q: f64, sample_rate: f64) -> Self {
        let k = Self::prewarp(freq, sample_rate);
        let k2 = k * k;
        let a0 = 1.0 + k / q + k2;
        let a1 = 2.0 * (k2 - 1.0);
        let a2 = 1.0 - k / q + k2;
        Self::normalise(a2, a1, a0, a0, a1, a2)
    }

    /// Magnitude response at `freq` (used by tests).
    pub fn magnitude(&self, freq: f64, sample_rate: f64) -> f64 {
        let w = 2.0 * PI * freq / sample_rate;
        let (c1, s1) = (w.cos(), w.sin());
        let (c2, s2) = ((2.0 * w).cos(), (2.0 * w).sin());
        let (b0, b1, b2) = (self.b0 as f64, self.b1 as f64, self.b2 as f64);
        let (a1, a2) = (self.a1 as f64, self.a2 as f64);
        let nr = b0 + b1 * c1 + b2 * c2;
        let ni = -(b1 * s1 + b2 * s2);
        let dr = 1.0 + a1 * c1 + a2 * c2;
        let di = -(a1 * s1 + a2 * s2);
        ((nr * nr + ni * ni) / (dr * dr + di * di)).sqrt()
    }
}

/// Filter state for one channel.
#[derive(Debug, Clone, Copy, Default)]
pub struct Biquad {
    s1: f32,
    s2: f32,
}

impl Biquad {
    #[inline(always)]
    pub fn process(&mut self, c: &Coeffs, x: f32) -> f32 {
        let y = c.b0 * x + self.s1;
        self.s1 = undenormal(c.b1 * x - c.a1 * y + self.s2);
        self.s2 = undenormal(c.b2 * x - c.a2 * y);
        y
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Butterworth Q, used to build Linkwitz-Riley filters from two cascaded sections.
pub const BUTTERWORTH_Q: f64 = std::f64::consts::FRAC_1_SQRT_2;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowpass_is_minus_3db_at_cutoff() {
        let c = Coeffs::lowpass(1000.0, BUTTERWORTH_Q, 48_000.0);
        let m = c.magnitude(1000.0, 48_000.0);
        assert!((20.0 * m.log10() + 3.01).abs() < 0.05, "{m}");
        assert!((c.magnitude(10.0, 48_000.0) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn highpass_blocks_dc() {
        let c = Coeffs::highpass(200.0, BUTTERWORTH_Q, 48_000.0);
        let mut f = Biquad::default();
        let mut y = 0.0;
        for _ in 0..48_000 {
            y = f.process(&c, 1.0);
        }
        assert!(y.abs() < 1e-4);
    }
}
