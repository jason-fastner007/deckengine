// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Jason-Marshall Fastner <jasonfastner@protonmail.com>

//! Look-ahead brick-wall peak limiter.
//!
//! Algorithm:
//! 1. per frame, the gain needed to keep the (stereo-linked) peak under the threshold,
//! 2. a sliding-window minimum over `lookahead + 1` frames (monotonic deque, O(1) amortised),
//! 3. an exponential release on top of that,
//! 4. a moving average over `lookahead` frames, which turns the step into a smooth fade while
//!    staying below the required gain when the peak leaves the delay line,
//! 5. the audio is delayed by `lookahead` frames and multiplied by the gain.
//!
//! The look-ahead adds latency, so it is kept short (default 1 ms) because the engine targets
//! sub-5 ms output latency.

/// Stereo look-ahead limiter.
#[derive(Debug, Clone)]
pub struct Limiter {
    threshold: f32,
    lookahead: usize,
    release_coeff: f32,

    delay: Vec<[f32; 2]>,
    delay_pos: usize,

    // Monotonic deque for the sliding minimum (ring buffer of capacity lookahead + 1).
    dq_val: Vec<f32>,
    dq_idx: Vec<u64>,
    dq_head: usize,
    dq_len: usize,
    counter: u64,

    held: f32,

    box_buf: Vec<f32>,
    box_pos: usize,
    box_sum: f64,
    since_resum: u32,

    last_gain: f32,
}

impl Limiter {
    /// `threshold` is linear (e.g. `0.891` = −1 dBFS).
    pub fn new(sample_rate: f32, threshold: f32, lookahead_ms: f32, release_ms: f32) -> Self {
        let lookahead = ((lookahead_ms * 0.001 * sample_rate).round() as usize).max(1);
        let window = lookahead + 1;
        let release_samples = (release_ms * 0.001 * sample_rate).max(1.0);
        Self {
            threshold: threshold.clamp(0.01, 1.0),
            lookahead,
            release_coeff: (-1.0 / release_samples).exp(),
            delay: vec![[0.0; 2]; lookahead],
            delay_pos: 0,
            dq_val: vec![1.0; window],
            dq_idx: vec![0; window],
            dq_head: 0,
            dq_len: 0,
            counter: 0,
            held: 1.0,
            box_buf: vec![1.0; lookahead],
            box_pos: 0,
            box_sum: lookahead as f64,
            since_resum: 0,
            last_gain: 1.0,
        }
    }

    /// Latency added by the limiter, in frames.
    pub fn latency_frames(&self) -> usize {
        self.lookahead
    }

    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.clamp(0.01, 1.0);
    }

    /// Current gain reduction as a linear factor (1 = no reduction).
    pub fn gain(&self) -> f32 {
        self.last_gain
    }

    pub fn reset(&mut self) {
        self.delay.iter_mut().for_each(|f| *f = [0.0; 2]);
        self.dq_len = 0;
        self.held = 1.0;
        self.box_buf.iter_mut().for_each(|g| *g = 1.0);
        self.box_sum = self.lookahead as f64;
        self.last_gain = 1.0;
    }

    #[inline(always)]
    fn push_min(&mut self, value: f32) -> f32 {
        let cap = self.dq_val.len();
        let idx = self.counter;
        self.counter += 1;

        // Drop entries from the back that are >= the new value.
        while self.dq_len > 0 {
            let back = (self.dq_head + self.dq_len - 1) % cap;
            if self.dq_val[back] >= value {
                self.dq_len -= 1;
            } else {
                break;
            }
        }
        let slot = (self.dq_head + self.dq_len) % cap;
        self.dq_val[slot] = value;
        self.dq_idx[slot] = idx;
        self.dq_len += 1;

        // Expire entries that fell out of the window.
        let window = cap as u64;
        while idx - self.dq_idx[self.dq_head] >= window {
            self.dq_head = (self.dq_head + 1) % cap;
            self.dq_len -= 1;
        }
        self.dq_val[self.dq_head]
    }

    /// Processes one stereo frame in place.
    #[inline(always)]
    pub fn process(&mut self, l: &mut f32, r: &mut f32) {
        let peak = l.abs().max(r.abs());
        let required = if peak > self.threshold { self.threshold / peak } else { 1.0 };

        let window_min = self.push_min(required);
        self.held = if window_min < self.held {
            window_min
        } else {
            window_min + (self.held - window_min) * self.release_coeff
        };

        self.box_sum += (self.held - self.box_buf[self.box_pos]) as f64;
        self.box_buf[self.box_pos] = self.held;
        self.box_pos = (self.box_pos + 1) % self.lookahead;
        self.since_resum += 1;
        if self.since_resum >= 1 << 16 {
            self.since_resum = 0;
            self.box_sum = self.box_buf.iter().map(|&g| g as f64).sum();
        }
        let gain = (self.box_sum / self.lookahead as f64) as f32;
        self.last_gain = gain;

        let delayed = self.delay[self.delay_pos];
        self.delay[self.delay_pos] = [*l, *r];
        self.delay_pos = (self.delay_pos + 1) % self.lookahead;

        *l = delayed[0] * gain;
        *r = delayed[1] * gain;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn never_exceeds_threshold() {
        let thr = 0.891;
        let mut lim = Limiter::new(48_000.0, thr, 1.0, 50.0);
        let mut max_out: f32 = 0.0;
        let mut seed = 12345u32;
        for i in 0..200_000 {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
            // Loud bursts with sudden transients.
            let amp = if (i / 5000) % 2 == 0 { 4.0 } else { 0.3 };
            let (mut l, mut r) = (noise * amp, -noise * amp * 0.7);
            lim.process(&mut l, &mut r);
            max_out = max_out.max(l.abs()).max(r.abs());
        }
        assert!(max_out <= thr + 1e-4, "{max_out}");
    }

    #[test]
    fn quiet_signal_passes_unchanged_after_delay() {
        let mut lim = Limiter::new(48_000.0, 0.891, 1.0, 50.0);
        let la = lim.latency_frames();
        let input: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let mut out = Vec::new();
        for &x in &input {
            let (mut l, mut r) = (x, x);
            lim.process(&mut l, &mut r);
            out.push(l);
        }
        for i in la..1000 {
            assert!((out[i] - input[i - la]).abs() < 1e-6);
        }
    }
}
