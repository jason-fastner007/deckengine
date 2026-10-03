// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Jason-Marshall Fastner <jasonfastner@protonmail.com>

//! Three-band stem isolator.
//!
//! Splits a stereo signal into low / mid / high with phase-coherent Linkwitz-Riley 4th-order
//! crossovers (the three bands sum back to a flat magnitude response), and exposes three
//! DJ-style controls:
//!
//! * **bass**   – level of the low band.
//! * **vocal**  – level of the *centre* (mid) component of the mid band, where lead vocals
//!   usually sit. `0` removes it (karaoke / instrumental), `2` removes everything else
//!   (a cappella-ish).
//! * **drums**  – level of the high band (hats, cymbals, snare crack).
//!
//! This is spectral / spatial isolation, not machine-learning source separation. It is cheap
//! (nine biquads per channel), has zero latency and runs comfortably inside a 64-frame callback.

use super::biquad::{BUTTERWORTH_Q, Biquad, Coeffs};
use super::smooth::Smoothed;

/// Crossover between the "bass" and "mid" bands.
pub const LOW_CROSSOVER_HZ: f64 = 180.0;
/// Crossover between the "mid" and "high" bands.
pub const HIGH_CROSSOVER_HZ: f64 = 4_500.0;

/// Stem levels. `1.0` everywhere is neutral.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StemLevels {
    /// `0` = bass kill, `1` = normal, up to `2` = boost.
    pub bass: f32,
    /// `0` = instrumental, `1` = normal, `2` = a cappella.
    pub vocal: f32,
    /// `0` = no highs/drums, `1` = normal, up to `2` = boost.
    pub drums: f32,
}

impl Default for StemLevels {
    fn default() -> Self {
        Self { bass: 1.0, vocal: 1.0, drums: 1.0 }
    }
}

impl StemLevels {
    pub fn clamped(self) -> Self {
        let c = |v: f32| if v.is_finite() { v.clamp(0.0, 2.0) } else { 1.0 };
        Self { bass: c(self.bass), vocal: c(self.vocal), drums: c(self.drums) }
    }
}

#[derive(Debug, Clone, Copy)]
struct Coefs {
    lp1: Coeffs,
    hp1: Coeffs,
    lp2: Coeffs,
    hp2: Coeffs,
    ap2: Coeffs,
}

#[derive(Debug, Clone, Copy, Default)]
struct ChannelState {
    lp1: [Biquad; 2],
    hp1: [Biquad; 2],
    lp2: [Biquad; 2],
    hp2: [Biquad; 2],
    ap2: Biquad,
}

/// Two identical sections in series (a Linkwitz-Riley 4th-order filter).
#[inline(always)]
fn cascade(stages: &mut [Biquad; 2], c: &Coeffs, x: f32) -> f32 {
    let y = stages[0].process(c, x);
    stages[1].process(c, y)
}

impl ChannelState {
    /// Returns `(low, mid, high)`.
    #[inline(always)]
    fn split(&mut self, c: &Coefs, x: f32) -> (f32, f32, f32) {
        let low = cascade(&mut self.lp1, &c.lp1, x);
        let rest = cascade(&mut self.hp1, &c.hp1, x);
        let mid = cascade(&mut self.lp2, &c.lp2, rest);
        let high = cascade(&mut self.hp2, &c.hp2, rest);
        // Give the low band the same phase shift the upper split applied to mid + high.
        let low = self.ap2.process(&c.ap2, low);
        (low, mid, high)
    }
}

/// Stereo stem isolator.
#[derive(Debug, Clone)]
pub struct StemIsolator {
    coefs: Coefs,
    ch: [ChannelState; 2],
    bass: Smoothed,
    vocal: Smoothed,
    drums: Smoothed,
}

impl StemIsolator {
    pub fn new(sample_rate: f32) -> Self {
        let sr = sample_rate as f64;
        let coefs = Coefs {
            lp1: Coeffs::lowpass(LOW_CROSSOVER_HZ, BUTTERWORTH_Q, sr),
            hp1: Coeffs::highpass(LOW_CROSSOVER_HZ, BUTTERWORTH_Q, sr),
            lp2: Coeffs::lowpass(HIGH_CROSSOVER_HZ, BUTTERWORTH_Q, sr),
            hp2: Coeffs::highpass(HIGH_CROSSOVER_HZ, BUTTERWORTH_Q, sr),
            ap2: Coeffs::allpass(HIGH_CROSSOVER_HZ, BUTTERWORTH_Q, sr),
        };
        Self {
            coefs,
            ch: [ChannelState::default(); 2],
            bass: Smoothed::new(1.0, 8.0, sample_rate),
            vocal: Smoothed::new(1.0, 8.0, sample_rate),
            drums: Smoothed::new(1.0, 8.0, sample_rate),
        }
    }

    pub fn set_levels(&mut self, levels: StemLevels) {
        let l = levels.clamped();
        self.bass.set(l.bass);
        self.vocal.set(l.vocal);
        self.drums.set(l.drums);
    }

    pub fn levels(&self) -> StemLevels {
        StemLevels { bass: self.bass.target(), vocal: self.vocal.target(), drums: self.drums.target() }
    }

    pub fn reset(&mut self) {
        self.ch = [ChannelState::default(); 2];
    }

    /// Processes one stereo frame.
    #[inline(always)]
    pub fn process(&mut self, [l, r]: [f32; 2]) -> [f32; 2] {
        let bass = self.bass.tick();
        let vocal = self.vocal.tick();
        let drums = self.drums.tick();

        let (low_l, mid_l, high_l) = self.ch[0].split(&self.coefs, l);
        let (low_r, mid_r, high_r) = self.ch[1].split(&self.coefs, r);

        // Above 1 the vocal control fades out everything that is *not* centre-mid.
        let acapella = (vocal - 1.0).clamp(0.0, 1.0);
        let centre_gain = vocal.min(1.0);
        let other_gain = 1.0 - acapella;

        let m = 0.5 * (mid_l + mid_r) * centre_gain;
        let s = 0.5 * (mid_l - mid_r) * other_gain;

        let low_g = bass * other_gain;
        let high_g = drums * other_gain;
        [low_l * low_g + (m + s) + high_l * high_g, low_r * low_g + (m - s) + high_r * high_g]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms_after(iso: &mut StemIsolator, freq: f32, stereo_phase_inverted: bool) -> f32 {
        let sr = 48_000.0;
        let mut acc = 0.0;
        let n = 48_000;
        for i in 0..n {
            let x = (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin() * 0.5;
            let [l, _] = iso.process([x, if stereo_phase_inverted { -x } else { x }]);
            if i > n / 2 {
                acc += l * l;
            }
        }
        (acc / (n / 2) as f32).sqrt()
    }

    #[test]
    fn neutral_is_flat() {
        let reference = 0.5 / 2f32.sqrt();
        for f in [50.0, 180.0, 1000.0, 4500.0, 12_000.0] {
            let mut iso = StemIsolator::new(48_000.0);
            let rms = rms_after(&mut iso, f, false);
            assert!((rms / reference - 1.0).abs() < 0.02, "{f} Hz: {rms}");
        }
    }

    #[test]
    fn vocal_zero_removes_centre_mid_but_keeps_side() {
        let mut iso = StemIsolator::new(48_000.0);
        iso.set_levels(StemLevels { vocal: 0.0, ..Default::default() });
        assert!(rms_after(&mut iso, 1000.0, false) < 0.01);

        let mut iso = StemIsolator::new(48_000.0);
        iso.set_levels(StemLevels { vocal: 0.0, ..Default::default() });
        assert!(rms_after(&mut iso, 1000.0, true) > 0.3);
    }

    #[test]
    fn bass_kill() {
        let mut iso = StemIsolator::new(48_000.0);
        iso.set_levels(StemLevels { bass: 0.0, ..Default::default() });
        assert!(rms_after(&mut iso, 40.0, false) < 0.02);
    }
}
