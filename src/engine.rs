// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Jason-Marshall Fastner <jasonfastner@protonmail.com>

//! The engine: control handle ([`EngineHandle`]) and real-time renderer ([`Processor`]).
//!
//! ```text
//!  app / UI thread                      audio thread (device callback)
//!  ───────────────                      ─────────────────────────────
//!  EngineHandle ──commands (lock-free)──▶ Processor::process()
//!       ▲                                   │  deck A ─┐
//!       └──── metrics (atomics) ◀───────────┤  deck B ─┴▶ crossfader ▶ leveler ▶ limiter ▶ out
//!                                           │
//!  streamer threads ──decoded chunks (lock-free rings)──▶ decks
//! ```
//!
//! The audio thread never locks, allocates, frees or does I/O.

use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use rtrb::{Consumer, Producer, RingBuffer};

use crate::Error;
use crate::deck::Deck;
use crate::dsp::isolator::StemLevels;
use crate::dsp::limiter::Limiter;
use crate::dsp::loudness::AutoLeveler;
use crate::dsp::smooth::{Ramp, Smoothed};
use crate::dsp::{db_to_gain, gain_to_db};
use crate::metrics::Metrics;
use crate::source::{FrameSource, SourceInfo};
use crate::stream::{self, Feed, StreamCmd, StreamControl};

/// Largest block rendered in one go. Larger device buffers are split.
pub(crate) const MAX_BLOCK: usize = 512;
const COMMAND_QUEUE: usize = 1024;
const SCHEDULE_SLOTS: usize = 64;

/// One of the two decks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DeckId {
    A = 0,
    B = 1,
}

impl DeckId {
    pub const ALL: [DeckId; 2] = [DeckId::A, DeckId::B];

    #[inline]
    fn idx(self) -> usize {
        self as usize
    }
}

/// Crossfader law.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum CrossfadeCurve {
    /// Constant power (sin/cos). No level dip in the middle. Best for blending.
    #[default]
    EqualPower = 0,
    /// Linear amplitude. −6 dB dip in the middle.
    Linear = 1,
    /// Both decks at full level in the middle (club mixer "additive" curve).
    Additive = 2,
}

impl CrossfadeCurve {
    #[inline(always)]
    fn gains(self, x: f32) -> (f32, f32) {
        let x = x.clamp(0.0, 1.0);
        match self {
            CrossfadeCurve::EqualPower => {
                let a = x * std::f32::consts::FRAC_PI_2;
                (a.cos(), a.sin())
            }
            CrossfadeCurve::Linear => (1.0 - x, x),
            CrossfadeCurve::Additive => (((1.0 - x) * 2.0).min(1.0), (x * 2.0).min(1.0)),
        }
    }
}

/// When a command takes effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum When {
    /// At the start of the next audio block.
    #[default]
    Now,
    /// At an exact frame of the engine clock (see [`EngineHandle::clock`]). Use this for
    /// beat-synchronous starts and transitions.
    At(u64),
}

/// Engine settings.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Look-ahead of the master limiter in ms. Adds to output latency. Default 1 ms.
    pub limiter_lookahead_ms: f32,
    /// Limiter ceiling in dBFS. Default −1.0.
    pub limiter_ceiling_db: f32,
    /// How much audio each deck decodes ahead, in ms. Default 750 ms.
    pub decode_ahead_ms: u32,
    /// Target of the master loudness normaliser in LUFS. Default −14 (streaming standard).
    pub normalize_target_lufs: f32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            limiter_lookahead_ms: 1.0,
            limiter_ceiling_db: -1.0,
            decode_ahead_ms: 750,
            normalize_target_lufs: -14.0,
        }
    }
}

enum Command {
    Load(DeckId, Box<Feed>),
    Unload(DeckId),
    Play(DeckId),
    Pause(DeckId),
    Speed(DeckId, f32),
    Volume(DeckId, f32),
    TrimDb(DeckId, f32),
    Stems(DeckId, StemLevels),
    Crossfader { position: f32, ramp_frames: u64 },
    Curve(CrossfadeCurve),
    MasterVolume(f32),
    Normalize(Option<f32>),
    Limiter(bool),
}

struct Timed {
    at: Option<u64>,
    cmd: Command,
}

/// Real-time renderer. Owned by the audio thread (or by you, for offline rendering).
pub struct Processor {
    sample_rate: u32,
    commands: Consumer<Timed>,
    garbage: Producer<Box<Feed>>,
    schedule: [Option<Timed>; SCHEDULE_SLOTS],
    clock: u64,

    decks: [Deck; 2],
    buf: [Box<[[f32; 2]; MAX_BLOCK]>; 2],
    mix: Box<[[f32; 2]; MAX_BLOCK]>,

    crossfader: Ramp,
    curve: CrossfadeCurve,
    master: Smoothed,
    leveler: AutoLeveler,
    leveler_gain: Smoothed,
    limiter: Limiter,
    limiter_on: bool,

    metrics: Arc<Metrics>,
}

impl Processor {
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Latency added by the engine's own processing (limiter look-ahead), in frames.
    pub fn latency_frames(&self) -> usize {
        if self.limiter_on { self.limiter.latency_frames() } else { 0 }
    }

    /// Reports the device's own latency (callback → DAC). Called by output backends.
    pub fn report_device_latency(&self, latency: Duration) {
        self.metrics.device_latency_ns.store(latency.as_nanos() as u64, Relaxed);
    }

    /// Reports a backend error (xrun, device change, ...).
    pub fn report_backend_error(&self) {
        self.metrics.backend_errors.fetch_add(1, Relaxed);
    }

    fn drop_feed(&mut self, feed: Option<Box<Feed>>) {
        if let Some(f) = feed {
            if let Err(rtrb::PushError::Full(f)) = self.garbage.push(f) {
                // Should never happen (janitor drains continuously). Leaking is the only
                // real-time-safe option left.
                std::mem::forget(f);
                self.metrics.dropped_garbage.fetch_add(1, Relaxed);
            }
        }
    }

    fn apply(&mut self, cmd: Command) {
        match cmd {
            Command::Load(d, feed) => {
                let old = self.decks[d.idx()].load(feed, &self.metrics.decks[d.idx()]);
                self.drop_feed(old);
            }
            Command::Unload(d) => {
                let old = self.decks[d.idx()].unload(&self.metrics.decks[d.idx()]);
                self.drop_feed(old);
            }
            Command::Play(d) => self.decks[d.idx()].play(),
            Command::Pause(d) => self.decks[d.idx()].pause(),
            Command::Speed(d, s) => self.decks[d.idx()].set_speed(s),
            Command::Volume(d, v) => self.decks[d.idx()].set_volume(v),
            Command::TrimDb(d, db) => self.decks[d.idx()].set_trim_db(db),
            Command::Stems(d, l) => self.decks[d.idx()].set_stems(l),
            Command::Crossfader { position, ramp_frames } => {
                self.crossfader.ramp_to(position.clamp(0.0, 1.0), ramp_frames)
            }
            Command::Curve(c) => self.curve = c,
            Command::MasterVolume(v) => self.master.set(v.clamp(0.0, 4.0)),
            Command::Normalize(target) => {
                if let Some(t) = target {
                    self.leveler.set_target(t);
                }
                self.leveler.set_enabled(target.is_some());
            }
            Command::Limiter(on) => {
                if on != self.limiter_on {
                    self.limiter.reset();
                }
                self.limiter_on = on;
                let lat = if on { self.limiter.latency_frames() as u32 } else { 0 };
                self.metrics.limiter_latency_frames.store(lat, Relaxed);
            }
        }
    }

    /// Moves incoming commands into the schedule / applies immediate ones.
    fn drain_commands(&mut self) {
        while let Ok(t) = self.commands.pop() {
            match t.at {
                Some(at) if at > self.clock => {
                    if let Some(slot) = self.schedule.iter_mut().find(|s| s.is_none()) {
                        *slot = Some(t);
                    } else {
                        // Schedule full: run it now rather than lose it.
                        self.apply(t.cmd);
                    }
                }
                _ => self.apply(t.cmd),
            }
        }
    }

    /// Applies scheduled commands due at `self.clock` and returns the next due time.
    fn run_due(&mut self) -> Option<u64> {
        let mut next: Option<u64> = None;
        for i in 0..SCHEDULE_SLOTS {
            let due = matches!(&self.schedule[i], Some(t) if t.at.unwrap_or(0) <= self.clock);
            if due {
                if let Some(t) = self.schedule[i].take() {
                    self.apply(t.cmd);
                }
            } else if let Some(t) = &self.schedule[i] {
                let at = t.at.unwrap_or(0);
                next = Some(next.map_or(at, |n| n.min(at)));
            }
        }
        next
    }

    fn render_block(&mut self, frames: usize) {
        let [bufa, bufb] = &mut self.buf;
        let [deck_a, deck_b] = &mut self.decks;
        deck_a.render(&mut bufa[..frames], &self.metrics.decks[0]);
        deck_b.render(&mut bufb[..frames], &self.metrics.decks[1]);

        let mix = &mut self.mix[..frames];
        if self.crossfader.is_active() {
            for i in 0..frames {
                let (ga, gb) = self.curve.gains(self.crossfader.tick());
                mix[i][0] = bufa[i][0] * ga + bufb[i][0] * gb;
                mix[i][1] = bufa[i][1] * ga + bufb[i][1] * gb;
            }
        } else {
            let (ga, gb) = self.curve.gains(self.crossfader.current());
            for i in 0..frames {
                mix[i][0] = bufa[i][0] * ga + bufb[i][0] * gb;
                mix[i][1] = bufa[i][1] * ga + bufb[i][1] * gb;
            }
        }

        if self.leveler.enabled() {
            let flat = mix.as_flattened();
            let g = self.leveler.update(flat, self.sample_rate as f32);
            self.leveler_gain.set(g);
        } else {
            self.leveler_gain.set(1.0);
        }

        let (mut pl, mut pr) = (0.0f32, 0.0f32);
        for f in mix.iter_mut() {
            let g = self.master.tick() * self.leveler_gain.tick();
            let (mut l, mut r) = (f[0] * g, f[1] * g);
            if self.limiter_on {
                self.limiter.process(&mut l, &mut r);
            }
            // Last line of defence against NaN/inf from a misbehaving source.
            l = if l.is_finite() { l.clamp(-1.0, 1.0) } else { 0.0 };
            r = if r.is_finite() { r.clamp(-1.0, 1.0) } else { 0.0 };
            pl = pl.max(l.abs());
            pr = pr.max(r.abs());
            *f = [l, r];
        }
        self.metrics.peak_l.set(pl);
        self.metrics.peak_r.set(pr);
    }

    /// Renders into an interleaved buffer with `channels` channels (any count ≥ 1).
    /// Channel 0 = left, 1 = right, further channels are silent; mono gets (L+R)/2.
    pub fn process(&mut self, out: &mut [f32], channels: usize) {
        let start = Instant::now();
        let channels = channels.max(1);
        let total = out.len() / channels;

        self.drain_commands();
        let mut done = 0;
        while done < total {
            let next_event = self.run_due();
            let mut n = (total - done).min(MAX_BLOCK);
            if let Some(at) = next_event {
                let until = at.saturating_sub(self.clock) as usize;
                if until > 0 {
                    n = n.min(until);
                }
            }
            self.render_block(n);

            let dst = &mut out[done * channels..(done + n) * channels];
            let src = &self.mix[..n];
            match channels {
                1 => dst.iter_mut().zip(src).for_each(|(d, s)| *d = 0.5 * (s[0] + s[1])),
                2 => dst.copy_from_slice(src.as_flattened()),
                c => {
                    for (d, s) in dst.chunks_exact_mut(c).zip(src) {
                        d[0] = s[0];
                        d[1] = s[1];
                        d[2..].fill(0.0);
                    }
                }
            }
            done += n;
            self.clock += n as u64;
        }

        let m = &self.metrics;
        m.frames_rendered.store(self.clock, Relaxed);
        m.crossfader.set(self.crossfader.current());
        m.short_term_lufs.set(self.leveler.short_term_lufs());
        m.leveler_gain_db.set(self.leveler.gain_db());
        m.limiter_gain_db.set(gain_to_db(self.limiter.gain()));
        m.record_callback(total, start.elapsed().as_nanos() as u64);
    }

    /// Convenience for offline use: renders interleaved stereo.
    pub fn render(&mut self, stereo_out: &mut [f32]) {
        self.process(stereo_out, 2);
    }
}

/// Snapshot of a deck.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DeckStatus {
    pub loaded: bool,
    pub playing: bool,
    /// Reached the end of the track.
    pub ended: bool,
    pub position_seconds: f64,
    pub duration_seconds: Option<f64>,
    /// Current (smoothed) speed ratio.
    pub speed: f32,
    /// Peak level of the last block (after deck gain), linear.
    pub peak: f32,
    /// Number of audio blocks in which the decoder could not keep up.
    pub underruns: u64,
    /// The source reported an I/O or decode error.
    pub failed: bool,
}

/// Snapshot of engine performance and levels.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EngineStats {
    pub sample_rate: u32,
    /// Frames of the most recent device callback.
    pub buffer_frames: u32,
    /// Largest callback seen.
    pub max_buffer_frames: u32,
    pub callbacks: u64,
    /// Average / max time spent in the callback, microseconds.
    pub process_avg_us: f64,
    pub process_max_us: f64,
    /// Average DSP load: processing time / buffer duration (0..1).
    pub dsp_load: f64,
    /// Callbacks that took longer than their real-time budget.
    pub deadline_misses: u64,
    /// Backend-reported errors (xruns, device changes).
    pub backend_errors: u64,
    /// Buffer duration of one callback, ms.
    pub buffer_ms: f64,
    /// Device latency (callback → DAC) as reported by the backend, ms. 0 if unknown.
    pub device_latency_ms: f64,
    /// Engine processing latency (limiter look-ahead), ms.
    pub engine_latency_ms: f64,
    /// Output latency: time from the audio callback to the DAC (at least one buffer) plus the
    /// engine's own look-ahead.
    pub output_latency_ms: f64,
    pub peak_l: f32,
    pub peak_r: f32,
    pub short_term_lufs: f32,
    pub leveler_gain_db: f32,
    pub limiter_gain_db: f32,
    pub crossfader: f32,
}

struct Control {
    commands: Producer<Timed>,
    streams: [Option<StreamControl>; 2],
    loop_region: [Option<(u64, u64)>; 2],
}

/// Thread-safe control handle. Cheap to clone. All methods return immediately.
#[derive(Clone)]
pub struct EngineHandle {
    control: Arc<Mutex<Control>>,
    metrics: Arc<Metrics>,
    sample_rate: u32,
    decode_ahead_ms: u32,
}

/// Creates a connected handle/processor pair for the given output sample rate.
pub fn create(sample_rate: u32, config: &EngineConfig) -> (EngineHandle, Processor) {
    let metrics = Arc::new(Metrics::new(sample_rate));
    let (cmd_tx, cmd_rx) = RingBuffer::new(COMMAND_QUEUE);
    let (garbage_tx, garbage_rx) = RingBuffer::<Box<Feed>>::new(64);
    spawn_janitor(garbage_rx);

    let sr = sample_rate as f32;
    let limiter = Limiter::new(
        sr,
        db_to_gain(config.limiter_ceiling_db.clamp(-20.0, 0.0)),
        config.limiter_lookahead_ms.clamp(0.05, 20.0),
        80.0,
    );
    metrics.limiter_latency_frames.store(limiter.latency_frames() as u32, Relaxed);

    let processor = Processor {
        sample_rate,
        commands: cmd_rx,
        garbage: garbage_tx,
        schedule: std::array::from_fn(|_| None),
        clock: 0,
        decks: [Deck::new(sr), Deck::new(sr)],
        buf: [Box::new([[0.0; 2]; MAX_BLOCK]), Box::new([[0.0; 2]; MAX_BLOCK])],
        mix: Box::new([[0.0; 2]; MAX_BLOCK]),
        crossfader: Ramp::new(0.0),
        curve: CrossfadeCurve::default(),
        master: Smoothed::new(1.0, 10.0, sr),
        leveler: AutoLeveler::new(sample_rate, config.normalize_target_lufs),
        leveler_gain: Smoothed::new(1.0, 50.0, sr),
        limiter,
        limiter_on: true,
        metrics: metrics.clone(),
    };
    let handle = EngineHandle {
        control: Arc::new(Mutex::new(Control {
            commands: cmd_tx,
            streams: [None, None],
            loop_region: [None, None],
        })),
        metrics,
        sample_rate,
        decode_ahead_ms: config.decode_ahead_ms,
    };
    (handle, processor)
}

/// Frees stream feeds that the audio thread handed back, off the audio thread.
fn spawn_janitor(mut rx: Consumer<Box<Feed>>) {
    let _ = std::thread::Builder::new().name("deckengine-janitor".into()).spawn(move || {
        loop {
            while let Ok(feed) = rx.pop() {
                drop(feed);
            }
            if rx.is_abandoned() {
                while let Ok(feed) = rx.pop() {
                    drop(feed);
                }
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    });
}

impl EngineHandle {
    fn lock(&self) -> MutexGuard<'_, Control> {
        // A panic while holding this lock cannot leave Control inconsistent in a way that
        // matters (it only holds queues), so recover from poisoning.
        self.control.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn send_at(&self, when: When, cmd: Command) -> Result<(), Error> {
        let at = match when {
            When::Now => None,
            When::At(f) => Some(f),
        };
        self.lock().commands.push(Timed { at, cmd }).map_err(|_| Error::QueueFull)
    }

    fn send(&self, cmd: Command) -> Result<(), Error> {
        self.send_at(When::Now, cmd)
    }

    #[cfg(feature = "output")]
    pub(crate) fn metrics(&self) -> Arc<Metrics> {
        self.metrics.clone()
    }

    /// Output sample rate of the engine.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Engine clock: frames rendered so far. Use with [`When::At`].
    pub fn clock(&self) -> u64 {
        self.metrics.frames_rendered.load(Relaxed)
    }

    /// Converts seconds to engine-clock frames.
    pub fn seconds_to_frames(&self, seconds: f64) -> u64 {
        (seconds.max(0.0) * self.sample_rate as f64).round() as u64
    }

    /// Loads an arbitrary [`FrameSource`] into a deck (paused, at position 0).
    pub fn load_source(&self, deck: DeckId, source: Box<dyn FrameSource>) -> Result<SourceInfo, Error> {
        let (ctl, feed) = stream::spawn(source, self.decode_ahead_ms)?;
        let info = feed.info;
        let mut c = self.lock();
        c.commands.push(Timed { at: None, cmd: Command::Load(deck, feed) }).map_err(|_| Error::QueueFull)?;
        // The old stream thread is stopped and joined here, on the caller's thread.
        c.streams[deck.idx()] = Some(ctl);
        c.loop_region[deck.idx()] = None;
        Ok(info)
    }

    /// Opens and loads an audio file (MP3, AAC/M4A, FLAC, Ogg Vorbis, WAV, AIFF, ALAC, ...).
    #[cfg(feature = "decode")]
    pub fn load_file(&self, deck: DeckId, path: impl AsRef<std::path::Path>) -> Result<SourceInfo, Error> {
        let src = crate::source::DecodedSource::open(path)?;
        self.load_source(deck, Box::new(src))
    }

    pub fn unload(&self, deck: DeckId) -> Result<(), Error> {
        self.send(Command::Unload(deck))?;
        let mut c = self.lock();
        c.streams[deck.idx()] = None;
        c.loop_region[deck.idx()] = None;
        Ok(())
    }

    pub fn play(&self, deck: DeckId) -> Result<(), Error> {
        self.send(Command::Play(deck))
    }

    /// Starts a deck at an exact engine-clock frame (e.g. on the next downbeat).
    pub fn play_at(&self, deck: DeckId, when: When) -> Result<(), Error> {
        self.send_at(when, Command::Play(deck))
    }

    pub fn pause(&self, deck: DeckId) -> Result<(), Error> {
        self.send(Command::Pause(deck))
    }

    pub fn pause_at(&self, deck: DeckId, when: When) -> Result<(), Error> {
        self.send_at(when, Command::Pause(deck))
    }

    fn source_rate(&self, deck: DeckId) -> Result<u32, Error> {
        let d = &self.metrics.decks[deck.idx()];
        let sr = d.source_rate.load(Relaxed);
        if self.lock().streams[deck.idx()].is_none() || sr == 0 {
            return Err(Error::NoTrack);
        }
        Ok(sr)
    }

    /// Seeks a deck. Playback continues seamlessly (old audio plays until the new position is
    /// decoded, then a ~2 ms crossfade).
    pub fn seek(&self, deck: DeckId, seconds: f64) -> Result<(), Error> {
        let sr = self.source_rate(deck)?;
        let frame = (seconds.max(0.0) * sr as f64).round() as u64;
        let c = self.lock();
        let s = c.streams[deck.idx()].as_ref().ok_or(Error::NoTrack)?;
        s.send(StreamCmd::Seek(frame));
        Ok(())
    }

    /// Sets a seamless, sample-accurate loop (`start..end` in seconds of the track).
    pub fn set_loop(&self, deck: DeckId, start_seconds: f64, end_seconds: f64) -> Result<(), Error> {
        let sr = self.source_rate(deck)? as f64;
        if end_seconds.is_nan() || start_seconds.is_nan() || end_seconds <= start_seconds {
            return Err(Error::InvalidArgument("loop end must be after loop start"));
        }
        let region = ((start_seconds.max(0.0) * sr).round() as u64, (end_seconds * sr).round() as u64);
        let mut c = self.lock();
        c.loop_region[deck.idx()] = Some(region);
        let s = c.streams[deck.idx()].as_ref().ok_or(Error::NoTrack)?;
        s.send(StreamCmd::SetLoop(Some(region)));
        Ok(())
    }

    pub fn clear_loop(&self, deck: DeckId) -> Result<(), Error> {
        let mut c = self.lock();
        c.loop_region[deck.idx()] = None;
        if let Some(s) = c.streams[deck.idx()].as_ref() {
            s.send(StreamCmd::SetLoop(None));
        }
        Ok(())
    }

    /// Speed ratio (1 = normal). Changes tempo and pitch together, like a turntable.
    pub fn set_speed(&self, deck: DeckId, ratio: f32) -> Result<(), Error> {
        self.send(Command::Speed(deck, ratio))
    }

    /// Deck volume / channel fader, linear (1 = unity).
    pub fn set_volume(&self, deck: DeckId, volume: f32) -> Result<(), Error> {
        self.send(Command::Volume(deck, volume))
    }

    /// Per-track gain in dB (e.g. from [`crate::analyze_file`] to match loudness).
    pub fn set_trim_db(&self, deck: DeckId, db: f32) -> Result<(), Error> {
        self.send(Command::TrimDb(deck, db))
    }

    pub fn set_stems(&self, deck: DeckId, levels: StemLevels) -> Result<(), Error> {
        self.send(Command::Stems(deck, levels))
    }

    pub fn set_stems_at(&self, deck: DeckId, levels: StemLevels, when: When) -> Result<(), Error> {
        self.send_at(when, Command::Stems(deck, levels))
    }

    /// Crossfader position: 0 = deck A only, 1 = deck B only.
    pub fn set_crossfader(&self, position: f32) -> Result<(), Error> {
        self.send(Command::Crossfader { position, ramp_frames: 0 })
    }

    /// Moves the crossfader to `position` over `seconds`, starting at `when`. Sample-accurate.
    pub fn crossfade(&self, position: f32, seconds: f64, when: When) -> Result<(), Error> {
        let ramp_frames = self.seconds_to_frames(seconds);
        self.send_at(when, Command::Crossfader { position, ramp_frames })
    }

    pub fn set_crossfade_curve(&self, curve: CrossfadeCurve) -> Result<(), Error> {
        self.send(Command::Curve(curve))
    }

    pub fn set_master_volume(&self, volume: f32) -> Result<(), Error> {
        self.send(Command::MasterVolume(volume))
    }

    /// Enables the real-time loudness normaliser on the master bus (`Some(target LUFS)`), or
    /// disables it (`None`).
    pub fn set_normalization(&self, target_lufs: Option<f32>) -> Result<(), Error> {
        self.send(Command::Normalize(target_lufs))
    }

    /// Enables/disables the master limiter. Disabling removes its look-ahead latency.
    pub fn set_limiter(&self, enabled: bool) -> Result<(), Error> {
        self.send(Command::Limiter(enabled))
    }

    pub fn deck_status(&self, deck: DeckId) -> DeckStatus {
        let d = &self.metrics.decks[deck.idx()];
        let sr = d.source_rate.load(Relaxed).max(1) as f64;
        let failed = self.lock().streams[deck.idx()].as_ref().is_some_and(|s| s.shared.failed.load(Relaxed));
        DeckStatus {
            loaded: d.loaded.load(Relaxed),
            playing: d.playing.load(Relaxed),
            ended: d.ended.load(Relaxed),
            position_seconds: d.position.load(Relaxed) as f64 / sr,
            duration_seconds: d.has_duration.load(Relaxed).then(|| d.duration.load(Relaxed) as f64 / sr),
            speed: d.speed.get(),
            peak: d.peak.get(),
            underruns: d.underruns.load(Relaxed),
            failed,
        }
    }

    pub fn stats(&self) -> EngineStats {
        let m = &self.metrics;
        let sr = self.sample_rate.max(1) as f64;
        let buffer_frames = m.last_callback_frames.load(Relaxed);
        let buffer_ms = buffer_frames as f64 * 1000.0 / sr;
        let avg_us = m.process_ns_avg.load(Relaxed) as f64 / 1000.0;
        let device_latency_ms = m.device_latency_ns.load(Relaxed) as f64 / 1e6;
        let engine_latency_ms = m.limiter_latency_frames.load(Relaxed) as f64 * 1000.0 / sr;
        // Backends report callback→DAC time of the block's first frame. A block can never be
        // audible sooner than its own length, so never report less than one buffer (guards
        // against drivers that report nonsense).
        let output_latency_ms = device_latency_ms.max(buffer_ms) + engine_latency_ms;
        EngineStats {
            sample_rate: self.sample_rate,
            buffer_frames,
            max_buffer_frames: m.max_callback_frames.load(Relaxed),
            callbacks: m.callbacks.load(Relaxed),
            process_avg_us: avg_us,
            process_max_us: m.process_ns_max.load(Relaxed) as f64 / 1000.0,
            dsp_load: if buffer_ms > 0.0 { avg_us / 1000.0 / buffer_ms } else { 0.0 },
            deadline_misses: m.deadline_misses.load(Relaxed),
            backend_errors: m.backend_errors.load(Relaxed),
            buffer_ms,
            device_latency_ms,
            engine_latency_ms,
            output_latency_ms,
            peak_l: m.peak_l.get(),
            peak_r: m.peak_r.get(),
            short_term_lufs: m.short_term_lufs.get(),
            leveler_gain_db: m.leveler_gain_db.get(),
            limiter_gain_db: m.limiter_gain_db.get(),
            crossfader: m.crossfader.get(),
        }
    }

    /// Resets the max-time / miss counters (e.g. after start-up).
    pub fn reset_stats(&self) {
        let m = &self.metrics;
        m.process_ns_max.store(0, Relaxed);
        m.deadline_misses.store(0, Relaxed);
        m.max_callback_frames.store(0, Relaxed);
        m.backend_errors.store(0, Relaxed);
    }
}
