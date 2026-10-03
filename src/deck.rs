// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Jason-Marshall Fastner <jasonfastner@protonmail.com>

//! One playback deck (audio-thread side).

use std::sync::atomic::Ordering::Relaxed;

use crate::dsp::db_to_gain;
use crate::dsp::isolator::{StemIsolator, StemLevels};
use crate::dsp::resample::{HALF, Interpolator};
use crate::dsp::smooth::{Ramp, Smoothed};
use crate::metrics::DeckMetrics;
use crate::stream::{Feed, Next};

/// Declick fade for play/pause/load, in milliseconds.
const PLAY_FADE_MS: f32 = 4.0;

pub(crate) struct Deck {
    sample_rate: f32,
    pub feed: Option<Box<Feed>>,
    interp: Interpolator,
    frac: f64,
    base_step: f64,
    flushing: usize,

    playing: bool,
    pausing: bool,
    ended: bool,
    play_fade: Ramp,

    speed: Smoothed,
    volume: Smoothed,
    trim: f32,
    isolator: StemIsolator,
}

impl Deck {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            feed: None,
            interp: Interpolator::new(),
            frac: 1.0,
            base_step: 1.0,
            flushing: 0,
            playing: false,
            pausing: false,
            ended: false,
            play_fade: Ramp::new(0.0),
            speed: Smoothed::new(1.0, 30.0, sample_rate),
            volume: Smoothed::new(1.0, 10.0, sample_rate),
            trim: 1.0,
            isolator: StemIsolator::new(sample_rate),
        }
    }

    fn fade_frames(&self) -> u64 {
        (PLAY_FADE_MS * 0.001 * self.sample_rate) as u64
    }

    /// Installs a new stream. Returns the previous one so it can be freed off the audio thread.
    pub fn load(&mut self, feed: Box<Feed>, m: &DeckMetrics) -> Option<Box<Feed>> {
        self.base_step = feed.info.sample_rate as f64 / self.sample_rate as f64;
        m.source_rate.store(feed.info.sample_rate, Relaxed);
        m.has_duration.store(feed.info.duration_frames.is_some(), Relaxed);
        m.duration.store(feed.info.duration_frames.unwrap_or(0), Relaxed);
        m.position.store(0, Relaxed);
        m.loaded.store(true, Relaxed);
        m.ended.store(false, Relaxed);
        self.interp.reset();
        self.isolator.reset();
        self.frac = 1.0;
        self.flushing = 0;
        self.ended = false;
        self.playing = false;
        self.pausing = false;
        self.play_fade.ramp_to(0.0, 0);
        self.feed.replace(feed)
    }

    pub fn unload(&mut self, m: &DeckMetrics) -> Option<Box<Feed>> {
        self.playing = false;
        self.pausing = false;
        m.loaded.store(false, Relaxed);
        m.playing.store(false, Relaxed);
        self.feed.take()
    }

    pub fn play(&mut self) {
        if self.feed.is_none() {
            return;
        }
        if self.ended {
            // Restarting after the end (e.g. after a seek back): clear the end state.
            self.ended = false;
            self.flushing = 0;
        }
        self.playing = true;
        self.pausing = false;
        self.play_fade.ramp_to(1.0, self.fade_frames());
    }

    pub fn pause(&mut self) {
        if self.playing {
            self.pausing = true;
            self.play_fade.ramp_to(0.0, self.fade_frames());
        }
    }

    pub fn set_speed(&mut self, speed: f32) {
        self.speed.set(speed.clamp(0.25, 4.0));
    }

    pub fn set_volume(&mut self, v: f32) {
        self.volume.set(v.clamp(0.0, 4.0));
    }

    pub fn set_trim_db(&mut self, db: f32) {
        self.trim = db_to_gain(db.clamp(-24.0, 24.0));
    }

    pub fn set_stems(&mut self, levels: StemLevels) {
        self.isolator.set_levels(levels);
    }

    /// Renders `out.len()` stereo frames (overwrites `out`).
    pub fn render(&mut self, out: &mut [[f32; 2]], m: &DeckMetrics) {
        let Some(feed) = self.feed.as_deref_mut() else {
            out.fill([0.0; 2]);
            return;
        };
        feed.poll(out.len());

        if !self.playing {
            out.fill([0.0; 2]);
            m.playing.store(false, Relaxed);
            return;
        }

        let mut peak: f32 = 0.0;
        let mut underruns = 0u64;
        for o in out.iter_mut() {
            // Pull source frames until the interpolator is positioned.
            let mut starved = false;
            while self.frac >= 1.0 {
                match feed.next() {
                    Next::Frame(f) => {
                        self.interp.push(f);
                        self.frac -= 1.0;
                    }
                    Next::Underrun => {
                        starved = true;
                        break;
                    }
                    Next::Eos => {
                        // Flush the interpolator's look-ahead with silence, then stop.
                        if self.flushing < HALF + 1 {
                            self.interp.push([0.0; 2]);
                            self.frac -= 1.0;
                            self.flushing += 1;
                        } else {
                            self.ended = true;
                            break;
                        }
                    }
                }
            }
            if self.ended {
                *o = [0.0; 2];
                continue;
            }
            if starved {
                underruns += 1;
                *o = [0.0; 2];
                continue;
            }
            self.flushing = if feed.is_eos() { self.flushing } else { 0 };

            let f = self.interp.get(self.frac);
            self.frac += self.base_step * self.speed.tick() as f64;

            let mut f = self.isolator.process(f);
            let g = self.volume.tick() * self.trim * self.play_fade.tick();
            f[0] *= g;
            f[1] *= g;
            peak = peak.max(f[0].abs()).max(f[1].abs());
            *o = f;
        }

        if self.pausing && !self.play_fade.is_active() {
            self.playing = false;
            self.pausing = false;
        }
        if self.ended {
            self.playing = false;
        }
        m.ended.store(self.ended, Relaxed);
        let pos = feed.position().saturating_sub(HALF as u64 + 1);
        feed.shared.playhead.store(pos, Relaxed);
        m.position.store(pos, Relaxed);
        m.playing.store(self.playing, Relaxed);
        m.speed.set(self.speed.current());
        m.peak.set(peak);
        if underruns > 0 {
            m.underruns.fetch_add(1, Relaxed);
        }
    }
}
