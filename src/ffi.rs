// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Jason-Marshall Fastner <jasonfastner@protonmail.com>

//! C ABI. The matching header is `include/deckengine.h`.
//!
//! Conventions:
//! * functions returning `int32_t` return `0` on success or a negative `DE_ERR_*` code;
//!   `de_last_error()` then returns a message (valid until the next call on the same thread);
//! * strings are NUL-terminated UTF-8;
//! * decks are `0` (A) and `1` (B);
//! * every function is safe to call from any thread; no panic ever crosses the boundary.
//!
//! # Safety (applies to every `unsafe` function here)
//! Pointer arguments must be NULL or valid for the access described in the header: engine
//! pointers must come from `de_engine_start` and not be freed yet, strings must be
//! NUL-terminated, output structs must be writable.
#![allow(clippy::missing_safety_doc)]

use std::cell::RefCell;
#[cfg(any(feature = "output", feature = "decode"))]
use std::ffi::CStr;
use std::ffi::{CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};

#[cfg(feature = "output")]
use crate::DeckId;
use crate::Error;

pub const DE_OK: i32 = 0;
pub const DE_ERR_NULL: i32 = -1;
pub const DE_ERR_IO: i32 = -2;
pub const DE_ERR_DECODE: i32 = -3;
pub const DE_ERR_DEVICE: i32 = -4;
pub const DE_ERR_NO_TRACK: i32 = -5;
pub const DE_ERR_QUEUE_FULL: i32 = -6;
pub const DE_ERR_INVALID: i32 = -7;
pub const DE_ERR_PANIC: i32 = -8;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_error(msg: &str) {
    let c = CString::new(msg.replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = c);
}

fn code(e: &Error) -> i32 {
    match e {
        Error::Io(_) => DE_ERR_IO,
        Error::Decode(_) => DE_ERR_DECODE,
        Error::Device(_) => DE_ERR_DEVICE,
        Error::NoTrack => DE_ERR_NO_TRACK,
        Error::QueueFull => DE_ERR_QUEUE_FULL,
        Error::InvalidArgument(_) => DE_ERR_INVALID,
    }
}

fn guard(f: impl FnOnce() -> Result<(), Error>) -> i32 {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => DE_OK,
        Ok(Err(e)) => {
            set_error(&e.to_string());
            code(&e)
        }
        Err(_) => {
            set_error("internal panic");
            DE_ERR_PANIC
        }
    }
}

#[cfg(feature = "output")]
fn deck(d: u32) -> Result<DeckId, Error> {
    match d {
        0 => Ok(DeckId::A),
        1 => Ok(DeckId::B),
        _ => Err(Error::InvalidArgument("deck must be 0 (A) or 1 (B)")),
    }
}

/// # Safety
/// `s` must be null or a valid NUL-terminated string.
#[cfg(any(feature = "output", feature = "decode"))]
unsafe fn opt_str(s: *const c_char) -> Result<Option<String>, Error> {
    if s.is_null() {
        return Ok(None);
    }
    unsafe { CStr::from_ptr(s) }
        .to_str()
        .map(|s| Some(s.to_owned()))
        .map_err(|_| Error::InvalidArgument("string is not valid UTF-8"))
}

/// Last error message on this thread (never null).
#[unsafe(no_mangle)]
pub extern "C" fn de_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

/// Library version, e.g. "0.1.0".
#[unsafe(no_mangle)]
pub extern "C" fn de_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// Loudness analysis result.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct DeLoudness {
    pub integrated_lufs: f32,
    pub true_peak_db: f32,
    pub loudness_range_lu: f32,
}

/// Measures a file's loudness (blocking; run it on a worker thread).
///
/// # Safety
/// `path` must be a valid string, `out` a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn de_analyze_file(path: *const c_char, out: *mut DeLoudness) -> i32 {
    guard(|| {
        #[cfg(feature = "decode")]
        {
            let path = unsafe { opt_str(path) }?.ok_or(Error::InvalidArgument("path is null"))?;
            let out = unsafe { out.as_mut() }.ok_or(Error::InvalidArgument("out is null"))?;
            let i = crate::analyze_file(path)?;
            *out = DeLoudness {
                integrated_lufs: i.integrated_lufs,
                true_peak_db: i.true_peak_db,
                loudness_range_lu: i.loudness_range_lu,
            };
            Ok(())
        }
        #[cfg(not(feature = "decode"))]
        {
            let _ = (path, out);
            Err(Error::InvalidArgument("built without the `decode` feature"))
        }
    })
}

pub use crate::energy::BeatEnergyAnalysis as DeBeatEnergyAnalysis;

/// Analyzes an amplitude envelope / waveform vector for BPM, downbeat, phrase, energies, and Camelot key.
///
/// # Safety
/// `samples` must point to `count` floats, `out` must be a valid pointer to DeBeatEnergyAnalysis.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn de_analyze_waveform(
    samples: *const f32,
    count: usize,
    duration_ms: i64,
    out: *mut DeBeatEnergyAnalysis,
) -> i32 {
    guard(|| {
        if samples.is_null() || out.is_null() {
            return Err(Error::InvalidArgument("samples or out is null"));
        }
        let slice = unsafe { std::slice::from_raw_parts(samples, count) };
        let res = crate::energy::analyze_waveform(slice, duration_ms)
            .ok_or(Error::InvalidArgument("waveform analysis failed (too short)"))?;
        unsafe { *out = res };
        Ok(())
    })
}

/// Analyzes raw PCM audio frames (interleaved float samples) directly.
///
/// # Safety
/// `pcm` must point to `frames * channels` floats, `out` must be a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn de_analyze_pcm_energies(
    pcm: *const f32,
    frames: usize,
    channels: u32,
    sample_rate: u32,
    out: *mut DeBeatEnergyAnalysis,
) -> i32 {
    guard(|| {
        if pcm.is_null() || out.is_null() {
            return Err(Error::InvalidArgument("pcm or out is null"));
        }
        let total = frames * channels as usize;
        let slice = unsafe { std::slice::from_raw_parts(pcm, total) };
        let res = crate::energy::analyze_pcm_energies(slice, channels as usize, sample_rate)
            .ok_or(Error::InvalidArgument("PCM energy analysis failed"))?;
        unsafe { *out = res };
        Ok(())
    })
}

/// Complete DJ/AI analysis of one track (times in ms on the track's own timeline).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DeTrackAnalysis {
    pub first_beat_ms: f64,
    pub downbeat_ms: f64,
    pub phrase_ms: f64,
    /// < 0 = none.
    pub mix_in_ms: f64,
    /// < 0 = none.
    pub mix_out_ms: f64,
    pub window_start_ms: f64,
    pub analyzed_ms: f64,
    pub bpm: f32,
    pub bpm_confidence: f32,
    pub downbeat_confidence: f32,
    pub phrase_confidence: f32,
    /// 0 = C … 11 = B, -1 = unknown.
    pub key_pitch: i32,
    pub key_minor: i32,
    pub key_confidence: f32,
    pub lufs: f32,
    pub peak_db: f32,
    pub bass: f32,
    pub mid: f32,
    pub high: f32,
    pub onset_density: f32,
    pub energy: f32,
    /// Fakeprint AI probability 0..1, < 0 = not computed.
    pub ai_probability: f32,
    /// Bass level (dBFS, < 150 Hz) of the analysed window – reference for [`de_scan_events`].
    pub bass_body_db: f32,
}

impl DeTrackAnalysis {
    fn from(a: &crate::dj::TrackAnalysis, window_start_ms: f64, ai: Option<f64>, bass_body_db: f32) -> Self {
        Self {
            first_beat_ms: a.first_beat_ms,
            downbeat_ms: a.downbeat_ms,
            phrase_ms: a.phrase_ms,
            mix_in_ms: a.mix_in_ms.unwrap_or(-1.0),
            mix_out_ms: a.mix_out_ms.unwrap_or(-1.0),
            window_start_ms,
            analyzed_ms: a.analyzed_ms,
            bpm: a.bpm,
            bpm_confidence: a.bpm_confidence,
            downbeat_confidence: a.downbeat_confidence,
            phrase_confidence: a.phrase_confidence,
            key_pitch: a.key_pitch.map_or(-1, |k| k as i32),
            key_minor: a.key_minor as i32,
            key_confidence: a.key_confidence,
            lufs: a.lufs,
            peak_db: a.peak_db,
            bass: a.bass,
            mid: a.mid,
            high: a.high,
            onset_density: a.onset_density,
            energy: a.energy,
            ai_probability: ai.map_or(-1.0, |p| p as f32),
            bass_body_db,
        }
    }
}

unsafe fn opt_slice<'a, T>(p: *const T, len: usize) -> Option<&'a [T]> {
    if p.is_null() || len == 0 { None } else { Some(unsafe { std::slice::from_raw_parts(p, len) }) }
}

/// Analyses mono PCM of a window starting at `window_start_ms` in a track of `duration_ms`.
/// `envelope` (optional, may be NULL) is an amplitude envelope of the whole track for mix points.
///
/// # Safety
/// `pcm` must point to `frames` floats, `envelope` to `envelope_len` floats, `out` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn de_analyze_pcm(
    pcm: *const f32,
    frames: usize,
    sample_rate: u32,
    window_start_ms: f64,
    duration_ms: f64,
    envelope: *const f32,
    envelope_len: usize,
    fakeprint: bool,
    out: *mut DeTrackAnalysis,
) -> i32 {
    guard(|| {
        let pcm = unsafe { opt_slice(pcm, frames) }.ok_or(Error::InvalidArgument("pcm is null"))?;
        let out = unsafe { out.as_mut() }.ok_or(Error::InvalidArgument("out is null"))?;
        let env = unsafe { opt_slice(envelope, envelope_len) };
        let a = crate::dj::analyze(pcm, sample_rate, window_start_ms, duration_ms, env)
            .ok_or(Error::InvalidArgument("too little audio or no beat"))?;
        let ai = if fakeprint { crate::fakeprint::detect(pcm, sample_rate) } else { None };
        *out = DeTrackAnalysis::from(&a, window_start_ms, ai, crate::cue::bass_level_db(pcm, sample_rate));
        Ok(())
    })
}

/// AI-generation probability of mono PCM (fakeprint). Writes a value < 0 for silence.
///
/// # Safety
/// `pcm` must point to `frames` floats, `out` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn de_fakeprint(pcm: *const f32, frames: usize, sample_rate: u32, out: *mut f32) -> i32 {
    guard(|| {
        let pcm = unsafe { opt_slice(pcm, frames) }.ok_or(Error::InvalidArgument("pcm is null"))?;
        let out = unsafe { out.as_mut() }.ok_or(Error::InvalidArgument("out is null"))?;
        *out = crate::fakeprint::detect(pcm, sample_rate).map_or(-1.0, |p| p as f32);
        Ok(())
    })
}

/// Stream parameters of a CBR MP3, read from the first bytes of the file.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct DeMp3Head {
    pub frame_bytes: f64,
    pub audio_start: u64,
    /// > 0: the head was too short, fetch at least this many bytes and call again.
    pub needed: u64,
    pub sample_rate: u32,
    pub bitrate_kbps: u32,
    pub samples_per_frame: u32,
    pub trim_samples: u32,
}

/// Parses the head of an MP3 file (ID3v2, Xing/Info, LAME delay).
///
/// # Safety
/// `head` must point to `len` bytes, `out` must be valid.
#[cfg(feature = "clip")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn de_mp3_head(head: *const u8, len: usize, out: *mut DeMp3Head) -> i32 {
    guard(|| {
        let head = unsafe { opt_slice(head, len) }.unwrap_or(&[]);
        let out = unsafe { out.as_mut() }.ok_or(Error::InvalidArgument("out is null"))?;
        *out = match crate::mp3clip::parse_head(head) {
            Ok(h) => DeMp3Head {
                frame_bytes: h.frame_bytes,
                audio_start: h.audio_start,
                needed: 0,
                sample_rate: h.sample_rate,
                bitrate_kbps: h.bitrate_kbps,
                samples_per_frame: h.samples_per_frame,
                trim_samples: h.trim_samples,
            },
            Err(0) => return Err(Error::Decode("not an MP3 stream".into())),
            Err(n) => DeMp3Head { needed: n as u64, ..Default::default() },
        };
        Ok(())
    })
}

/// Decodes an MP3 byte range (cut out of the file at `clip_offset`) and analyses it.
///
/// # Safety
/// Pointers must be valid for their lengths; `envelope` may be NULL.
#[cfg(feature = "clip")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn de_analyze_mp3_clip(
    head: *const u8,
    head_len: usize,
    clip: *const u8,
    clip_len: usize,
    clip_offset: u64,
    duration_ms: f64,
    envelope: *const f32,
    envelope_len: usize,
    fakeprint: bool,
    out: *mut DeTrackAnalysis,
) -> i32 {
    guard(|| {
        let head = unsafe { opt_slice(head, head_len) }.ok_or(Error::InvalidArgument("head is null"))?;
        let clip = unsafe { opt_slice(clip, clip_len) }.ok_or(Error::InvalidArgument("clip is null"))?;
        let out = unsafe { out.as_mut() }.ok_or(Error::InvalidArgument("out is null"))?;
        let env = unsafe { opt_slice(envelope, envelope_len) };
        let h = crate::mp3clip::parse_head(head).map_err(|_| Error::Decode("bad MP3 head".into()))?;
        let c = crate::mp3clip::decode_clip(&h, clip, clip_offset)?;
        let a = crate::dj::analyze(&c.pcm, c.sample_rate, c.start_ms, duration_ms, env)
            .ok_or(Error::InvalidArgument("too little audio or no beat"))?;
        let ai = if fakeprint { crate::fakeprint::detect(&c.pcm, c.sample_rate) } else { None };
        *out = DeTrackAnalysis::from(&a, c.start_ms, ai, crate::cue::bass_level_db(&c.pcm, c.sample_rate));
        Ok(())
    })
}

/// Cue and first drop found in one MP3 clip (times in ms on the track's timeline).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct DeEvents {
    /// First sustained sound after silence, < 0 = none (or not asked for).
    pub cue_ms: f64,
    /// Bass returning after a breakdown, snapped to the bar/beat grid; < 0 = none in this clip.
    pub drop_ms: f64,
    /// 0..1, how clearly the bass returns.
    pub drop_strength: f32,
    pub reserved: f32,
}

/// Scans an MP3 byte-range clip (see [`de_analyze_mp3_clip`]) for the cue point (only when
/// `want_cue`, i.e. the clip starts at the beginning of the file) and the first drop.
/// `period_ms`/`downbeat_ms` come from the track analysis (0 = unknown); `bass_body_db` is its
/// `bass_body_db` (>= 0 = unknown).
///
/// # Safety
/// Pointers must be valid for their lengths.
#[cfg(feature = "clip")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn de_scan_events(
    head: *const u8,
    head_len: usize,
    clip: *const u8,
    clip_len: usize,
    clip_offset: u64,
    period_ms: f64,
    downbeat_ms: f64,
    bass_body_db: f32,
    want_cue: bool,
    out: *mut DeEvents,
) -> i32 {
    guard(|| {
        let head = unsafe { opt_slice(head, head_len) }.ok_or(Error::InvalidArgument("head is null"))?;
        let clip = unsafe { opt_slice(clip, clip_len) }.ok_or(Error::InvalidArgument("clip is null"))?;
        let out = unsafe { out.as_mut() }.ok_or(Error::InvalidArgument("out is null"))?;
        let h = crate::mp3clip::parse_head(head).map_err(|_| Error::Decode("bad MP3 head".into()))?;
        let c = crate::mp3clip::decode_clip(&h, clip, clip_offset)?;
        let body = (bass_body_db < 0.0).then_some(bass_body_db);
        let cue = if want_cue { crate::cue::detect_cue(&c.pcm, c.sample_rate, c.start_ms) } else { None };
        let drop = crate::cue::detect_drop(&c.pcm, c.sample_rate, c.start_ms, period_ms, downbeat_ms, body);
        *out = DeEvents {
            cue_ms: cue.unwrap_or(-1.0),
            drop_ms: drop.map_or(-1.0, |d| d.ms),
            drop_strength: drop.map_or(0.0, |d| d.strength),
            reserved: 0.0,
        };
        Ok(())
    })
}

/// Echolot fingerprint of mono PCM: writes up to `capacity` landmarks as interleaved
/// `(hash, frame)` pairs into `out` (2 × u32 each) and their number into `count`.
/// If `capacity` is too small, nothing is written, `count` gets the required number and the
/// call returns an error. Frames are [`crate::echolot::FRAME_MS`] apart.
///
/// # Safety
/// `pcm` must point to `frames` floats, `out` to `2 * capacity` u32 (may be NULL if capacity is 0).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn de_fingerprint(
    pcm: *const f32,
    frames: usize,
    sample_rate: u32,
    out: *mut u32,
    capacity: usize,
    count: *mut usize,
) -> i32 {
    guard(|| {
        let pcm = unsafe { opt_slice(pcm, frames) }.ok_or(Error::InvalidArgument("pcm is null"))?;
        let count = unsafe { count.as_mut() }.ok_or(Error::InvalidArgument("count is null"))?;
        let fp = crate::echolot::fingerprint(pcm, sample_rate);
        *count = fp.len();
        if fp.len() > capacity || (capacity > 0 && out.is_null()) {
            return Err(Error::InvalidArgument("capacity too small"));
        }
        let dst = unsafe { std::slice::from_raw_parts_mut(out, 2 * fp.len()) };
        for (d, l) in dst.chunks_exact_mut(2).zip(&fp) {
            d[0] = l.hash;
            d[1] = l.frame;
        }
        Ok(())
    })
}

/// Memory for callers that cannot allocate in this module's heap (WebAssembly).
#[unsafe(no_mangle)]
pub extern "C" fn de_alloc(len: usize) -> *mut u8 {
    let mut v = Vec::<u8>::with_capacity(len.max(1));
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// Frees memory from [`de_alloc`].
///
/// # Safety
/// `p` must come from `de_alloc(len)` and not be freed yet.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn de_free(p: *mut u8, len: usize) {
    if !p.is_null() {
        drop(unsafe { Vec::from_raw_parts(p, 0, len.max(1)) });
    }
}

#[cfg(feature = "output")]
pub use engine_api::*;

#[cfg(feature = "output")]
mod engine_api {
    use super::*;
    use crate::{CrossfadeCurve, Engine, EngineConfig, OutputConfig, StemLevels, When};

    /// Opaque engine handle.
    pub struct DeEngine {
        engine: Engine,
    }

    /// Output settings. Zero-initialise for defaults.
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct DeOutputConfig {
        /// Frames per callback, 0 = default (128).
        pub buffer_frames: u32,
        /// Sample rate, 0 = device default.
        pub sample_rate: u32,
        /// Device name substring, or NULL for the default device.
        pub device: *const c_char,
        /// Host name ("alsa", "pipewire", "jack", "wasapi", "asio", ...), or NULL for automatic.
        pub host: *const c_char,
        /// Limiter look-ahead in ms, 0 = default (1 ms).
        pub limiter_lookahead_ms: f32,
    }

    #[repr(C)]
    #[derive(Debug, Default, Clone, Copy)]
    pub struct DeDeckStatus {
        pub loaded: bool,
        pub playing: bool,
        pub ended: bool,
        pub failed: bool,
        pub position_seconds: f64,
        /// Negative when unknown.
        pub duration_seconds: f64,
        pub speed: f32,
        pub peak: f32,
        pub underruns: u64,
    }

    #[repr(C)]
    #[derive(Debug, Default, Clone, Copy)]
    pub struct DeStats {
        pub sample_rate: u32,
        pub buffer_frames: u32,
        pub max_buffer_frames: u32,
        pub callbacks: u64,
        pub process_avg_us: f64,
        pub process_max_us: f64,
        pub dsp_load: f64,
        pub deadline_misses: u64,
        pub backend_errors: u64,
        pub buffer_ms: f64,
        pub device_latency_ms: f64,
        pub engine_latency_ms: f64,
        pub output_latency_ms: f64,
        pub peak_l: f32,
        pub peak_r: f32,
        pub short_term_lufs: f32,
        pub leveler_gain_db: f32,
        pub limiter_gain_db: f32,
        pub crossfader: f32,
    }

    fn with(e: *const DeEngine, f: impl FnOnce(&Engine) -> Result<(), Error>) -> i32 {
        guard(|| {
            let e = unsafe { e.as_ref() }.ok_or(Error::InvalidArgument("engine is null"))?;
            f(&e.engine)
        })
    }

    /// Opens the audio device and starts the engine. Returns NULL on failure.
    ///
    /// # Safety
    /// `config` must be NULL or point to a valid `DeOutputConfig`.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn de_engine_start(config: *const DeOutputConfig) -> *mut DeEngine {
        let mut result: *mut DeEngine = std::ptr::null_mut();
        let rc = guard(|| {
            let mut out = OutputConfig::default();
            let mut eng = EngineConfig::default();
            if let Some(c) = unsafe { config.as_ref() } {
                if c.buffer_frames > 0 {
                    out.buffer_frames = c.buffer_frames;
                }
                if c.sample_rate > 0 {
                    out.sample_rate = Some(c.sample_rate);
                }
                out.device = unsafe { opt_str(c.device) }?;
                out.host = unsafe { opt_str(c.host) }?;
                if c.limiter_lookahead_ms > 0.0 {
                    eng.limiter_lookahead_ms = c.limiter_lookahead_ms;
                }
            }
            let engine = Engine::start(out, eng)?;
            result = Box::into_raw(Box::new(DeEngine { engine }));
            Ok(())
        });
        if rc != DE_OK {
            return std::ptr::null_mut();
        }
        result
    }

    /// Stops audio and frees the engine. NULL is ignored.
    ///
    /// # Safety
    /// `e` must come from `de_engine_start` and not be used afterwards.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn de_engine_free(e: *mut DeEngine) {
        if !e.is_null() {
            let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(e) })));
        }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn de_engine_sample_rate(e: *const DeEngine) -> u32 {
        unsafe { e.as_ref() }.map_or(0, |e| e.engine.sample_rate())
    }

    /// Engine clock in frames (for `*_at` functions).
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn de_engine_clock(e: *const DeEngine) -> u64 {
        unsafe { e.as_ref() }.map_or(0, |e| e.engine.clock())
    }

    /// # Safety
    /// `path` must be a valid string.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn de_deck_load(e: *const DeEngine, d: u32, path: *const c_char) -> i32 {
        with(e, |e| {
            #[cfg(feature = "decode")]
            {
                let path = unsafe { opt_str(path) }?.ok_or(Error::InvalidArgument("path is null"))?;
                e.load_file(deck(d)?, path).map(|_| ())
            }
            #[cfg(not(feature = "decode"))]
            {
                let _ = (e, d, path);
                Err(Error::InvalidArgument("built without the `decode` feature"))
            }
        })
    }

    /// Loads interleaved stereo float PCM (copied).
    ///
    /// # Safety
    /// `samples` must point to `frames * 2` floats.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn de_deck_load_pcm(
        e: *const DeEngine,
        d: u32,
        samples: *const f32,
        frames: u64,
        sample_rate: u32,
    ) -> i32 {
        with(e, |e| {
            if samples.is_null() || sample_rate == 0 {
                return Err(Error::InvalidArgument("samples null or sample_rate 0"));
            }
            let data = unsafe { std::slice::from_raw_parts(samples, frames as usize * 2) };
            let src = crate::PcmSource::new(data.to_vec(), sample_rate);
            e.load_source(deck(d)?, Box::new(src)).map(|_| ())
        })
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn de_deck_unload(e: *const DeEngine, d: u32) -> i32 {
        with(e, |e| e.unload(deck(d)?))
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn de_deck_play(e: *const DeEngine, d: u32) -> i32 {
        with(e, |e| e.play(deck(d)?))
    }

    /// Starts the deck at engine-clock frame `at_frame`.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn de_deck_play_at(e: *const DeEngine, d: u32, at_frame: u64) -> i32 {
        with(e, |e| e.play_at(deck(d)?, When::At(at_frame)))
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn de_deck_pause(e: *const DeEngine, d: u32) -> i32 {
        with(e, |e| e.pause(deck(d)?))
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn de_deck_seek(e: *const DeEngine, d: u32, seconds: f64) -> i32 {
        with(e, |e| e.seek(deck(d)?, seconds))
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn de_deck_set_loop(e: *const DeEngine, d: u32, start_s: f64, end_s: f64) -> i32 {
        with(e, |e| e.set_loop(deck(d)?, start_s, end_s))
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn de_deck_clear_loop(e: *const DeEngine, d: u32) -> i32 {
        with(e, |e| e.clear_loop(deck(d)?))
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn de_deck_set_speed(e: *const DeEngine, d: u32, ratio: f32) -> i32 {
        with(e, |e| e.set_speed(deck(d)?, ratio))
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn de_deck_set_volume(e: *const DeEngine, d: u32, volume: f32) -> i32 {
        with(e, |e| e.set_volume(deck(d)?, volume))
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn de_deck_set_trim_db(e: *const DeEngine, d: u32, db: f32) -> i32 {
        with(e, |e| e.set_trim_db(deck(d)?, db))
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn de_deck_set_stems(
        e: *const DeEngine,
        d: u32,
        bass: f32,
        vocal: f32,
        drums: f32,
    ) -> i32 {
        with(e, |e| e.set_stems(deck(d)?, StemLevels { bass, vocal, drums }))
    }

    /// # Safety
    /// `out` must be a valid pointer.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn de_deck_status(e: *const DeEngine, d: u32, out: *mut DeDeckStatus) -> i32 {
        with(e, |e| {
            let out = unsafe { out.as_mut() }.ok_or(Error::InvalidArgument("out is null"))?;
            let s = e.deck_status(deck(d)?);
            *out = DeDeckStatus {
                loaded: s.loaded,
                playing: s.playing,
                ended: s.ended,
                failed: s.failed,
                position_seconds: s.position_seconds,
                duration_seconds: s.duration_seconds.unwrap_or(-1.0),
                speed: s.speed,
                peak: s.peak,
                underruns: s.underruns,
            };
            Ok(())
        })
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn de_set_crossfader(e: *const DeEngine, position: f32) -> i32 {
        with(e, |e| e.set_crossfader(position))
    }

    /// Crossfades to `position` over `seconds`. `at_frame` = engine clock frame to start at,
    /// or 0 for now.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn de_crossfade(
        e: *const DeEngine,
        position: f32,
        seconds: f64,
        at_frame: u64,
    ) -> i32 {
        let when = if at_frame == 0 { When::Now } else { When::At(at_frame) };
        with(e, |e| e.crossfade(position, seconds, when))
    }

    /// 0 = equal power, 1 = linear, 2 = additive.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn de_set_crossfade_curve(e: *const DeEngine, curve: u32) -> i32 {
        with(e, |e| {
            let c = match curve {
                0 => CrossfadeCurve::EqualPower,
                1 => CrossfadeCurve::Linear,
                2 => CrossfadeCurve::Additive,
                _ => return Err(Error::InvalidArgument("unknown crossfade curve")),
            };
            e.set_crossfade_curve(c)
        })
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn de_set_master_volume(e: *const DeEngine, volume: f32) -> i32 {
        with(e, |e| e.set_master_volume(volume))
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn de_set_normalization(
        e: *const DeEngine,
        enabled: bool,
        target_lufs: f32,
    ) -> i32 {
        with(e, |e| e.set_normalization(enabled.then_some(target_lufs)))
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn de_set_limiter(e: *const DeEngine, enabled: bool) -> i32 {
        with(e, |e| e.set_limiter(enabled))
    }

    /// # Safety
    /// `out` must be a valid pointer.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn de_engine_stats(e: *const DeEngine, out: *mut DeStats) -> i32 {
        with(e, |e| {
            let out = unsafe { out.as_mut() }.ok_or(Error::InvalidArgument("out is null"))?;
            let s = e.stats();
            *out = DeStats {
                sample_rate: s.sample_rate,
                buffer_frames: s.buffer_frames,
                max_buffer_frames: s.max_buffer_frames,
                callbacks: s.callbacks,
                process_avg_us: s.process_avg_us,
                process_max_us: s.process_max_us,
                dsp_load: s.dsp_load,
                deadline_misses: s.deadline_misses,
                backend_errors: s.backend_errors,
                buffer_ms: s.buffer_ms,
                device_latency_ms: s.device_latency_ms,
                engine_latency_ms: s.engine_latency_ms,
                output_latency_ms: s.output_latency_ms,
                peak_l: s.peak_l,
                peak_r: s.peak_r,
                short_term_lufs: s.short_term_lufs,
                leveler_gain_db: s.leveler_gain_db,
                limiter_gain_db: s.limiter_gain_db,
                crossfader: s.crossfader,
            };
            Ok(())
        })
    }
}
