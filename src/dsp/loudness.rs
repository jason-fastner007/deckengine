// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Jason-Marshall Fastner <jasonfastner@protonmail.com>

//! Loudness measurement (EBU R128 / ITU-R BS.1770) and a real-time auto-leveler.

use ebur128::{EbuR128, Mode};

use super::biquad::{Biquad, Coeffs};
use super::db_to_gain;
use super::smooth::Smoothed;

/// Result of measuring a whole programme (e.g. a track).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoudnessInfo {
    /// Integrated loudness in LUFS (−70 for silence).
    pub integrated_lufs: f32,
    /// Maximum true peak across channels in dBTP (−120 for silence).
    pub true_peak_db: f32,
    /// Loudness range in LU.
    pub loudness_range_lu: f32,
}

impl LoudnessInfo {
    /// Gain in dB that brings this programme to `target_lufs` without letting the true peak
    /// exceed `ceiling_db`.
    pub fn gain_to_target(&self, target_lufs: f32, ceiling_db: f32) -> f32 {
        if self.integrated_lufs <= -69.0 {
            return 0.0;
        }
        let mut gain = target_lufs - self.integrated_lufs;
        if self.true_peak_db > -120.0 {
            gain = gain.min(ceiling_db - self.true_peak_db);
        }
        gain.clamp(-24.0, 24.0)
    }
}

/// Offline loudness meter for stereo interleaved `f32` audio.
pub struct LoudnessMeter {
    inner: EbuR128,
}

impl LoudnessMeter {
    pub fn new(sample_rate: u32) -> Result<Self, ebur128::Error> {
        let mode = Mode::I | Mode::LRA | Mode::TRUE_PEAK | Mode::HISTOGRAM;
        Ok(Self { inner: EbuR128::new(2, sample_rate, mode)? })
    }

    /// Feeds interleaved stereo samples.
    pub fn add(&mut self, interleaved: &[f32]) {
        let _ = self.inner.add_frames_f32(interleaved);
    }

    pub fn finish(&self) -> LoudnessInfo {
        let integrated = self.inner.loudness_global().ok().filter(|v| v.is_finite()).unwrap_or(-70.0);
        let lra = self.inner.loudness_range().ok().filter(|v| v.is_finite()).unwrap_or(0.0);
        let mut tp: f64 = 0.0;
        for ch in 0..2 {
            if let Ok(p) = self.inner.true_peak(ch) {
                if p.is_finite() {
                    tp = tp.max(p);
                }
            }
        }
        LoudnessInfo {
            integrated_lufs: integrated as f32,
            true_peak_db: if tp > 0.0 { (20.0 * tp.log10()) as f32 } else { -120.0 },
            loudness_range_lu: lra as f32,
        }
    }
}

/// BS.1770 K-weighting filters (pre-filter shelf + RLB high-pass) for any sample rate.
/// Same design equations as libebur128, so readings match the offline meter.
fn k_weighting(sample_rate: f64) -> (Coeffs, Coeffs) {
    use std::f64::consts::PI;
    let (f0, g, q) = (1681.974450955533, 3.999843853973347, 0.7071752369554196);
    let k = (PI * f0 / sample_rate).tan();
    let vh = 10f64.powf(g / 20.0);
    let vb = vh.powf(0.4996667741545416);
    let a0 = 1.0 + k / q + k * k;
    let pre = Coeffs {
        b0: ((vh + vb * k / q + k * k) / a0) as f32,
        b1: (2.0 * (k * k - vh) / a0) as f32,
        b2: ((vh - vb * k / q + k * k) / a0) as f32,
        a1: (2.0 * (k * k - 1.0) / a0) as f32,
        a2: ((1.0 - k / q + k * k) / a0) as f32,
    };
    let (f0, q) = (38.13547087602444, 0.5003270373238773);
    let k = (PI * f0 / sample_rate).tan();
    let d = 1.0 + k / q + k * k;
    let rlb = Coeffs {
        b0: 1.0,
        b1: -2.0,
        b2: 1.0,
        a1: (2.0 * (k * k - 1.0) / d) as f32,
        a2: ((1.0 - k / q + k * k) / d) as f32,
    };
    (pre, rlb)
}

const ST_BLOCKS: usize = 30; // 30 × 100 ms = 3 s

/// Real-time short-term (3 s) loudness meter with O(1) cost per query.
///
/// Energy is accumulated in 100 ms blocks and kept in a ring of 30 blocks with a running sum,
/// so reading the loudness never scans the window (unlike a full EBU R128 meter).
#[derive(Debug, Clone)]
pub struct ShortTermMeter {
    pre: Coeffs,
    rlb: Coeffs,
    state: [[Biquad; 2]; 2],
    block_len: u32,
    block_pos: u32,
    block_acc: f64,
    ring: [f64; ST_BLOCKS],
    ring_idx: usize,
    ring_sum: f64,
    filled: usize,
}

impl ShortTermMeter {
    pub fn new(sample_rate: u32) -> Self {
        let (pre, rlb) = k_weighting(sample_rate as f64);
        Self {
            pre,
            rlb,
            state: Default::default(),
            block_len: (sample_rate / 10).max(1),
            block_pos: 0,
            block_acc: 0.0,
            ring: [0.0; ST_BLOCKS],
            ring_idx: 0,
            ring_sum: 0.0,
            filled: 0,
        }
    }

    pub fn reset(&mut self) {
        self.state = Default::default();
        self.block_pos = 0;
        self.block_acc = 0.0;
        self.ring = [0.0; ST_BLOCKS];
        self.ring_idx = 0;
        self.ring_sum = 0.0;
        self.filled = 0;
    }

    /// Feeds interleaved stereo samples.
    pub fn add(&mut self, interleaved: &[f32]) {
        for f in interleaved.chunks_exact(2) {
            let mut e = 0.0f32;
            for (ch, &x) in f.iter().enumerate() {
                let [s_pre, s_rlb] = &mut self.state[ch];
                let y = s_rlb.process(&self.rlb, s_pre.process(&self.pre, x));
                e += y * y;
            }
            self.block_acc += e as f64;
            self.block_pos += 1;
            if self.block_pos == self.block_len {
                let energy = self.block_acc / self.block_len as f64;
                self.ring_sum += energy - self.ring[self.ring_idx];
                self.ring[self.ring_idx] = energy;
                self.ring_idx = (self.ring_idx + 1) % ST_BLOCKS;
                if self.ring_idx == 0 {
                    // Re-sum once per window to cancel floating-point drift.
                    self.ring_sum = self.ring.iter().sum();
                }
                self.filled = (self.filled + 1).min(ST_BLOCKS);
                self.block_pos = 0;
                self.block_acc = 0.0;
            }
        }
    }

    /// Short-term loudness in LUFS (−70 when silent or not enough audio yet).
    pub fn loudness(&self) -> f32 {
        if self.filled == 0 {
            return -70.0;
        }
        let energy = self.ring_sum / self.filled as f64;
        if energy <= 1e-10 { -70.0 } else { (-0.691 + 10.0 * energy.log10()) as f32 }
    }
}

/// Real-time loudness normaliser for the master bus.
///
/// Measures short-term loudness (3 s window) and slowly steers a gain towards `target_lufs`.
/// Peaks are handled by the limiter that follows it.
pub struct AutoLeveler {
    meter: ShortTermMeter,
    target_lufs: f32,
    max_boost_db: f32,
    max_cut_db: f32,
    gain_db: Smoothed,
    short_term: f32,
    enabled: bool,
    query_interval: u32,
    since_query: u32,
}

impl AutoLeveler {
    pub fn new(sample_rate: u32, target_lufs: f32) -> Self {
        Self {
            meter: ShortTermMeter::new(sample_rate),
            target_lufs,
            max_boost_db: 12.0,
            max_cut_db: 18.0,
            // Updated once per block; `update` rescales the time constant to the block rate.
            gain_db: Smoothed::new(0.0, 1500.0, 1000.0),
            short_term: -70.0,
            enabled: false,
            query_interval: sample_rate / 10,
            since_query: 0,
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.gain_db.reset(0.0);
            self.meter.reset();
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn set_target(&mut self, target_lufs: f32) {
        self.target_lufs = target_lufs.clamp(-40.0, 0.0);
    }

    pub fn short_term_lufs(&self) -> f32 {
        self.short_term
    }

    pub fn gain_db(&self) -> f32 {
        self.gain_db.current()
    }

    /// Measures an interleaved stereo block and returns the linear gain to apply to it.
    pub fn update(&mut self, interleaved: &[f32], sample_rate: f32) -> f32 {
        if !self.enabled {
            return 1.0;
        }
        let frames = (interleaved.len() / 2).max(1) as u32;
        self.meter.add(interleaved);
        self.since_query += frames;
        if self.since_query >= self.query_interval {
            self.since_query = 0;
            self.short_term = self.meter.loudness();
            // Do not pump up silence or fade-outs.
            if self.short_term > -45.0 {
                let wanted = (self.target_lufs - self.short_term).clamp(-self.max_cut_db, self.max_boost_db);
                self.gain_db.set(wanted);
            }
        }
        self.gain_db.set_time(1500.0, sample_rate / frames as f32);
        db_to_gain(self.gain_db.tick())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_scale_stereo_sine_measures_about_0_lufs() {
        let sr = 48_000;
        let mut m = LoudnessMeter::new(sr).unwrap();
        let buf: Vec<f32> = (0..sr as usize * 5)
            .flat_map(|i| {
                let x = (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sr as f32).sin();
                [x, x]
            })
            .collect();
        m.add(&buf);
        let info = m.finish();
        // BS.1770: a 0 dBFS 1 kHz sine reads −3.01 LUFS per channel; two channels sum to ≈ 0.
        assert!(info.integrated_lufs.abs() < 0.3, "{info:?}");
        assert!(info.true_peak_db.abs() < 0.5, "{info:?}");
    }

    #[test]
    fn realtime_meter_matches_ebur128() {
        let sr = 44_100;
        // Pink-ish test signal: a few sines at different levels, 4 s.
        let buf: Vec<f32> = (0..sr as usize * 4)
            .flat_map(|i| {
                let t = i as f32 / sr as f32;
                let x = 0.3 * (2.0 * std::f32::consts::PI * 60.0 * t).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * 900.0 * t).sin()
                    + 0.1 * (2.0 * std::f32::consts::PI * 6000.0 * t).sin();
                [x, 0.7 * x]
            })
            .collect();
        let mut reference = EbuR128::new(2, sr, Mode::S).unwrap();
        reference.add_frames_f32(&buf).unwrap();
        let expected = reference.loudness_shortterm().unwrap() as f32;
        let mut ours = ShortTermMeter::new(sr);
        ours.add(&buf);
        assert!((ours.loudness() - expected).abs() < 0.1, "ours {} vs ebur128 {expected}", ours.loudness());
    }
}
