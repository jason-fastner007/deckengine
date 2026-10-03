// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Jason-Marshall Fastner <jasonfastner@protonmail.com>

//! Lock-free telemetry written by the audio thread and read by the control side.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed};

/// `f32` stored in an `AtomicU32`.
#[derive(Debug, Default)]
pub(crate) struct AtomicF32(AtomicU32);

impl AtomicF32 {
    pub fn new(v: f32) -> Self {
        Self(AtomicU32::new(v.to_bits()))
    }
    #[inline]
    pub fn set(&self, v: f32) {
        self.0.store(v.to_bits(), Relaxed);
    }
    #[inline]
    pub fn get(&self) -> f32 {
        f32::from_bits(self.0.load(Relaxed))
    }
}

#[derive(Debug, Default)]
pub(crate) struct DeckMetrics {
    pub loaded: AtomicBool,
    pub playing: AtomicBool,
    pub ended: AtomicBool,
    pub position: AtomicU64,
    pub source_rate: AtomicU32,
    pub duration: AtomicU64,
    pub has_duration: AtomicBool,
    pub underruns: AtomicU64,
    pub speed: AtomicF32,
    pub peak: AtomicF32,
}

#[derive(Debug)]
pub(crate) struct Metrics {
    pub decks: [DeckMetrics; 2],
    pub sample_rate: AtomicU32,
    pub frames_rendered: AtomicU64,
    pub callbacks: AtomicU64,
    pub last_callback_frames: AtomicU32,
    pub max_callback_frames: AtomicU32,
    pub process_ns_last: AtomicU64,
    pub process_ns_max: AtomicU64,
    pub process_ns_avg: AtomicU64,
    pub deadline_misses: AtomicU64,
    pub device_latency_ns: AtomicU64,
    pub backend_errors: AtomicU64,
    pub limiter_latency_frames: AtomicU32,
    pub peak_l: AtomicF32,
    pub peak_r: AtomicF32,
    pub short_term_lufs: AtomicF32,
    pub leveler_gain_db: AtomicF32,
    pub limiter_gain_db: AtomicF32,
    pub crossfader: AtomicF32,
    pub dropped_garbage: AtomicU64,
}

impl Metrics {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            decks: Default::default(),
            sample_rate: AtomicU32::new(sample_rate),
            frames_rendered: AtomicU64::new(0),
            callbacks: AtomicU64::new(0),
            last_callback_frames: AtomicU32::new(0),
            max_callback_frames: AtomicU32::new(0),
            process_ns_last: AtomicU64::new(0),
            process_ns_max: AtomicU64::new(0),
            process_ns_avg: AtomicU64::new(0),
            deadline_misses: AtomicU64::new(0),
            device_latency_ns: AtomicU64::new(0),
            backend_errors: AtomicU64::new(0),
            limiter_latency_frames: AtomicU32::new(0),
            peak_l: AtomicF32::new(0.0),
            peak_r: AtomicF32::new(0.0),
            short_term_lufs: AtomicF32::new(-70.0),
            leveler_gain_db: AtomicF32::new(0.0),
            limiter_gain_db: AtomicF32::new(0.0),
            crossfader: AtomicF32::new(0.0),
            dropped_garbage: AtomicU64::new(0),
        }
    }

    /// Records the time one callback spent processing `frames` frames.
    pub fn record_callback(&self, frames: usize, elapsed_ns: u64) {
        let sr = self.sample_rate.load(Relaxed).max(1) as u64;
        self.callbacks.fetch_add(1, Relaxed);
        self.last_callback_frames.store(frames as u32, Relaxed);
        if frames as u32 > self.max_callback_frames.load(Relaxed) {
            self.max_callback_frames.store(frames as u32, Relaxed);
        }
        self.process_ns_last.store(elapsed_ns, Relaxed);
        if elapsed_ns > self.process_ns_max.load(Relaxed) {
            self.process_ns_max.store(elapsed_ns, Relaxed);
        }
        let avg = self.process_ns_avg.load(Relaxed);
        let new_avg = if avg == 0 { elapsed_ns } else { avg - avg / 64 + elapsed_ns / 64 };
        self.process_ns_avg.store(new_avg, Relaxed);
        let budget_ns = frames as u64 * 1_000_000_000 / sr;
        if elapsed_ns > budget_ns {
            self.deadline_misses.fetch_add(1, Relaxed);
        }
    }
}
