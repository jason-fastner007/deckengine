// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Jason-Marshall Fastner <jasonfastner@protonmail.com>

//! Parameter smoothing, so that gain/speed changes never produce zipper noise or clicks.

/// One-pole exponential smoother.
#[derive(Debug, Clone, Copy)]
pub struct Smoothed {
    current: f32,
    target: f32,
    coeff: f32,
}

impl Smoothed {
    /// `time_ms` is the time constant (≈63 % of the way to the target).
    pub fn new(value: f32, time_ms: f32, sample_rate: f32) -> Self {
        let mut s = Self { current: value, target: value, coeff: 0.0 };
        s.set_time(time_ms, sample_rate);
        s
    }

    pub fn set_time(&mut self, time_ms: f32, sample_rate: f32) {
        let samples = (time_ms * 0.001 * sample_rate).max(1.0);
        self.coeff = (-1.0 / samples).exp();
    }

    #[inline]
    pub fn set(&mut self, target: f32) {
        self.target = target;
    }

    /// Jumps to `value` immediately.
    #[inline]
    pub fn reset(&mut self, value: f32) {
        self.current = value;
        self.target = value;
    }

    #[inline]
    pub fn target(&self) -> f32 {
        self.target
    }

    #[inline]
    pub fn current(&self) -> f32 {
        self.current
    }

    /// True once the smoother is close enough to its target that it can be treated as constant.
    #[inline]
    pub fn is_settled(&self) -> bool {
        (self.current - self.target).abs() < 1e-5
    }

    #[inline(always)]
    pub fn tick(&mut self) -> f32 {
        self.current = self.target + (self.current - self.target) * self.coeff;
        if self.is_settled() {
            self.current = self.target;
        }
        self.current
    }
}

/// Linear ramp that reaches its target in an exact number of samples. Used for sample-accurate
/// automation such as timed crossfades.
#[derive(Debug, Clone, Copy)]
pub struct Ramp {
    current: f32,
    target: f32,
    step: f32,
    remaining: u64,
}

impl Ramp {
    pub fn new(value: f32) -> Self {
        Self { current: value, target: value, step: 0.0, remaining: 0 }
    }

    /// Ramps to `target` over `samples` samples (0 = jump).
    pub fn ramp_to(&mut self, target: f32, samples: u64) {
        self.target = target;
        if samples == 0 {
            self.current = target;
            self.remaining = 0;
            self.step = 0.0;
        } else {
            self.remaining = samples;
            self.step = (target - self.current) / samples as f32;
        }
    }

    #[inline]
    pub fn is_active(&self) -> bool {
        self.remaining > 0
    }

    #[inline]
    pub fn current(&self) -> f32 {
        self.current
    }

    #[inline(always)]
    pub fn tick(&mut self) -> f32 {
        if self.remaining > 0 {
            self.remaining -= 1;
            self.current = if self.remaining == 0 { self.target } else { self.current + self.step };
        }
        self.current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ramp_hits_target_exactly() {
        let mut r = Ramp::new(0.0);
        r.ramp_to(1.0, 100);
        let mut last = 0.0;
        for _ in 0..100 {
            last = r.tick();
        }
        assert_eq!(last, 1.0);
        assert!(!r.is_active());
    }

    #[test]
    fn smoother_converges() {
        let mut s = Smoothed::new(0.0, 5.0, 48_000.0);
        s.set(1.0);
        for _ in 0..48_000 {
            s.tick();
        }
        assert_eq!(s.current(), 1.0);
    }
}
