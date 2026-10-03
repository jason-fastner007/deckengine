// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Jason-Marshall Fastner <jasonfastner@protonmail.com>

//! DJ track analysis on decoded PCM: tempo, beat grid down to phrase level, musical key,
//! loudness/energy and mix points.
//!
//! This is the KittyTune `BeatAnalyzer` + `DownbeatTracker` pipeline (classical DSP, no ML):
//! mono PCM → STFT → multi-band spectral flux → autocorrelation tempo → octave resolution →
//! comb-filter phase → downbeat / phrase classification from the 4/4 drum pattern.
//!
//! Additions over the Kotlin version:
//! * **Fine tempo** over the whole window (±2 %, ~0.02 BPM steps) and snapping to whole or half
//!   BPM when the grid fits just as well. DAW exports sit on such values, and 0.1 BPM of error
//!   already drifts ~230 ms over five minutes at 128 BPM.
//! * Onset times are corrected for where the flux actually peaks inside an STFT window
//!   (taking the window start is ~35 ms early, an audible flam when two tracks are layered).
//! * **Key** from a tuned chroma built from spectral peaks, with a separate bass chroma (EDM
//!   basslines carry the tonic), correlated (Pearson) against Krumhansl-Schmuckler and Temperley.
//! * Integrated loudness (ITU-R BS.1770), band balance, onset density and an energy level 1–10.

use realfft::RealFftPlanner;
use realfft::num_complex::Complex;

/// Rate everything is analysed at. 22.05 kHz keeps 11 kHz of bandwidth, plenty for onsets and
/// chroma, at a quarter of the work of 44.1 kHz stereo.
pub const ANALYSIS_RATE: u32 = 22_050;

const FFT: usize = 1024;
const HOP: usize = 256;
/// Where in an STFT window an attack sits when the flux peaks: on the rising flank of the Hann
/// window, ~¾ into it (measured on synthetic kicks). Frame times use this instead of the window
/// start (KittyTune: ~35 ms early) or centre (~11 ms early).
const ONSET_POS: f32 = 0.74;
const KEY_FFT: usize = 4096;
const KEY_HOP: usize = 2048;

const MIN_BPM: f32 = 60.0;
const MAX_BPM: f32 = 180.0;
/// Canonical range tempi are folded into (160 BPM hardtekk stays 160, 80 folds to 160).
const MIN_CANONICAL_BPM: f32 = 90.0;
const MAX_CANONICAL_BPM: f32 = 180.0;
/// Listeners tap along near 120–130 BPM; breaks half/double ties.
const PREFERRED_BPM: f32 = 125.0;
const TEMPO_PRIOR_OCTAVES: f32 = 0.85;

pub const BEATS_PER_BAR: i64 = 4;
pub const BARS_PER_PHRASE: i64 = 4;
pub const BEATS_PER_PHRASE: i64 = BEATS_PER_BAR * BARS_PER_PHRASE;

const MAX_INTRO_SKIP_MS: f64 = 20_000.0;
const MAX_OUTRO_CUT_MS: f64 = 45_000.0;

/// Band edges in Hz: kick body, snare body, snare crack / hats.
const LOW_BAND_HZ: f32 = 150.0;
const MID_BAND_HI_HZ: f32 = 800.0;
const HIGH_BAND_LO_HZ: f32 = 2000.0;
const HIGH_BAND_HI_HZ: f32 = 8000.0;

/// Complete analysis of one track (times in ms on the track's own timeline).
#[derive(Debug, Clone, PartialEq)]
pub struct TrackAnalysis {
    pub bpm: f32,
    pub bpm_confidence: f32,
    pub first_beat_ms: f64,
    pub downbeat_ms: f64,
    pub phrase_ms: f64,
    pub downbeat_confidence: f32,
    pub phrase_confidence: f32,
    /// 0 = C … 11 = B, `None` when there is too little tonal content (drum tracks).
    pub key_pitch: Option<u8>,
    pub key_minor: bool,
    pub key_confidence: f32,
    pub mix_in_ms: Option<f64>,
    pub mix_out_ms: Option<f64>,
    pub lufs: f32,
    pub peak_db: f32,
    /// Share of spectral energy per band (sums to ~1).
    pub bass: f32,
    pub mid: f32,
    pub high: f32,
    /// Onsets per second.
    pub onset_density: f32,
    /// 1 (ambient) … 10 (peak-time).
    pub energy: f32,
    pub analyzed_ms: f64,
}

impl TrackAnalysis {
    pub fn period_ms(&self) -> f64 {
        60_000.0 / self.bpm as f64
    }
}

/// Onset strength per STFT hop, split into the bands that carry rhythmic information.
pub struct BandOnsets {
    pub low: Vec<f32>,
    pub mid: Vec<f32>,
    pub high: Vec<f32>,
    pub full: Vec<f32>,
    pub frame_rate: f32,
    /// Seconds from the first sample to the time frame 0 stands for (see [`ONSET_POS`]).
    pub frame0_s: f32,
}

impl BandOnsets {
    pub fn len(&self) -> usize {
        self.full.len()
    }

    pub fn is_empty(&self) -> bool {
        self.full.is_empty()
    }
}

/// Band energies and loudness gathered along the onset pass.
struct Spectra {
    onsets: BandOnsets,
    band_power: [f64; 3],
}

fn hann(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos()).collect()
}

fn bin_of(hz: f32, rate: u32, fft: usize) -> usize {
    ((hz * fft as f32 / rate as f32) as usize).min(fft / 2 - 1)
}

/// Multi-band half-wave-rectified spectral flux on log-compressed magnitudes.
fn spectra(x: &[f32], rate: u32) -> Spectra {
    let frames = if x.len() >= FFT { (x.len() - FFT) / HOP + 1 } else { 0 };
    let bins = FFT / 2;
    let low_hi = bin_of(LOW_BAND_HZ, rate, FFT);
    let mid_lo = low_hi;
    let mid_hi = bin_of(MID_BAND_HI_HZ, rate, FFT);
    let high_lo = bin_of(HIGH_BAND_LO_HZ, rate, FFT);
    let high_hi = bin_of(HIGH_BAND_HI_HZ, rate, FFT);

    let mut full = vec![0.0f32; frames];
    let mut low = vec![0.0f32; frames];
    let mut mid = vec![0.0f32; frames];
    let mut high = vec![0.0f32; frames];
    let mut band_power = [0.0f64; 3];

    let window = hann(FFT);
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(FFT);
    let mut input = r2c.make_input_vec();
    let mut spec = r2c.make_output_vec();
    let mut scratch = r2c.make_scratch_vec();
    let mut prev = vec![0.0f32; bins];

    for f in 0..frames {
        let off = f * HOP;
        for i in 0..FFT {
            input[i] = x[off + i] * window[i];
        }
        if r2c.process_with_scratch(&mut input, &mut spec, &mut scratch).is_err() {
            continue;
        }
        let (mut sum, mut ls, mut ms, mut hs) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for b in 0..bins {
            let p = spec[b].norm_sqr();
            if b <= low_hi {
                band_power[0] += p as f64;
            } else if b < high_lo {
                band_power[1] += p as f64;
            } else {
                band_power[2] += p as f64;
            }
            let mag = (1.0 + 10.0 * p.sqrt()).ln();
            let diff = mag - prev[b];
            prev[b] = mag;
            if diff <= 0.0 {
                continue;
            }
            sum += diff;
            if b <= low_hi {
                ls += diff;
            }
            if (mid_lo..=mid_hi).contains(&b) {
                ms += diff;
            }
            if (high_lo..=high_hi).contains(&b) {
                hs += diff;
            }
        }
        full[f] = sum;
        low[f] = ls;
        mid[f] = ms;
        high[f] = hs;
    }
    // The very first frame has no predecessor: its "flux" is the whole spectrum.
    for v in [&mut full, &mut low, &mut mid, &mut high] {
        if let Some(first) = v.first_mut() {
            *first = 0.0;
        }
    }

    // Subtract a local mean so autocorrelation sees onsets, not slow dynamics.
    const MEAN_WINDOW: usize = 8;
    Spectra {
        onsets: BandOnsets {
            low: detrend(&low, MEAN_WINDOW),
            mid: detrend(&mid, MEAN_WINDOW),
            high: detrend(&high, MEAN_WINDOW),
            full: detrend(&full, MEAN_WINDOW),
            frame_rate: rate as f32 / HOP as f32,
            frame0_s: ONSET_POS * FFT as f32 / rate as f32,
        },
        band_power,
    }
}

/// `max(0, v[i] - mean(v[i-w ..= i+w]))`, O(n) with a running sum.
fn detrend(v: &[f32], w: usize) -> Vec<f32> {
    let n = v.len();
    let mut prefix = vec![0.0f64; n + 1];
    for i in 0..n {
        prefix[i + 1] = prefix[i] + v[i] as f64;
    }
    (0..n)
        .map(|i| {
            let lo = i.saturating_sub(w);
            let hi = (i + w).min(n - 1);
            let mean = (prefix[hi + 1] - prefix[lo]) / (hi - lo + 1) as f64;
            (v[i] as f64 - mean).max(0.0) as f32
        })
        .collect()
}

/// Relative plausibility of a tempo, peaking at [`PREFERRED_BPM`] and falling off by octaves.
pub fn tempo_prior(bpm: f32) -> f32 {
    if bpm <= 0.0 {
        return 0.0;
    }
    let oct = (bpm / PREFERRED_BPM).log2() / TEMPO_PRIOR_OCTAVES;
    (-0.5 * oct * oct).exp()
}

/// Autocorrelation over the beat-period lag range. Returns (period in frames, confidence),
/// favouring candidates whose double period is also supported.
fn estimate_tempo_period(flux: &[f32], frame_rate: f32) -> Option<(f32, f32)> {
    let min_lag = (frame_rate * 60.0 / MAX_BPM).round() as usize;
    let max_lag = (frame_rate * 60.0 / MIN_BPM).round() as usize;
    if flux.len() < max_lag * 2 || min_lag < 2 {
        return None;
    }
    let mut ac = vec![0.0f32; 2 * max_lag + 2];
    for (lag, slot) in ac.iter_mut().enumerate().take(2 * max_lag + 1).skip(min_lag) {
        if lag >= flux.len() {
            break;
        }
        let mut s = 0.0f32;
        for i in 0..flux.len() - lag {
            s += flux[i] * flux[i + lag];
        }
        *slot = s / (flux.len() - lag) as f32;
    }
    let range = &ac[min_lag..=max_lag];
    let mean = range.iter().sum::<f32>() / range.len() as f32;
    if mean <= 0.0 {
        return None;
    }
    let mut best: Option<usize> = None;
    let mut best_score = 0.0f32;
    for lag in min_lag..=max_lag {
        if lag > min_lag && lag < max_lag && (ac[lag] < ac[lag - 1] || ac[lag] < ac[lag + 1]) {
            continue;
        }
        // True beat vs half-beat: the double lag of the true period correlates too.
        let score = ac[lag] + 0.5 * ac[lag * 2];
        if score > best_score {
            best_score = score;
            best = Some(lag);
        }
    }
    let best = best?;
    let refined = if best > min_lag && best < max_lag {
        let (y0, y1, y2) = (ac[best - 1], ac[best], ac[best + 1]);
        let d = y0 - 2.0 * y1 + y2;
        if d != 0.0 { best as f32 + 0.5 * (y0 - y2) / d } else { best as f32 }
    } else {
        best as f32
    };
    let mut sorted: Vec<f32> = range.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let median = sorted[sorted.len() / 2];
    let conf = if ac[best] > 0.0 { ((ac[best] - median) / ac[best]).clamp(0.0, 1.0) } else { 0.0 };
    Some((refined, conf))
}

/// The strongest autocorrelation peaks (lags in frames), best first.
fn autocorr_peaks(flux: &[f32], frame_rate: f32, max: usize) -> Vec<f32> {
    let min_lag = (frame_rate * 60.0 / MAX_BPM).round() as usize;
    let max_lag = (frame_rate * 60.0 / MIN_BPM).round() as usize;
    if flux.len() < max_lag * 2 || min_lag < 2 {
        return Vec::new();
    }
    let ac: Vec<f32> = (0..=max_lag + 1)
        .map(|lag| {
            if lag < min_lag.saturating_sub(1) {
                return 0.0;
            }
            let mut s = 0.0f32;
            for i in 0..flux.len() - lag {
                s += flux[i] * flux[i + lag];
            }
            s / (flux.len() - lag) as f32
        })
        .collect();
    let mut peaks: Vec<(f32, f32)> = (min_lag..=max_lag)
        .filter(|&l| ac[l] >= ac[l - 1] && ac[l] > ac[l + 1] && ac[l] > 0.0)
        .map(|l| {
            let (y0, y1, y2) = (ac[l - 1], ac[l], ac[l + 1]);
            let d = y0 - 2.0 * y1 + y2;
            (y1, if d != 0.0 { l as f32 + 0.5 * (y0 - y2) / d } else { l as f32 })
        })
        .collect();
    peaks.sort_by(|a, b| b.0.total_cmp(&a.0));
    peaks.into_iter().take(max).map(|p| p.1).collect()
}

/// Picks between a period and its half / double / third using autocorrelation weighted by
/// [`tempo_prior`], instead of blindly folding into a fixed window (which turns a 160 BPM track
/// with a halftime snare into 80 BPM).
fn resolve_tempo_octave(flux: &[f32], frame_rate: f32, period: f32) -> f32 {
    let corr = |lag: f32| -> f32 {
        let l = lag.round() as usize;
        if l < 2 || l >= flux.len() {
            return 0.0;
        }
        let (mut s, mut e) = (0.0f32, 1e-9f32);
        for i in 0..flux.len() - l {
            s += flux[i] * flux[i + l];
            e += flux[i] * flux[i];
        }
        s / e
    };
    const FACTORS: [f32; 9] = [0.25, 1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0, 1.5, 2.0, 3.0, 4.0];
    let mut best = period;
    let mut best_score = f32::NEG_INFINITY;
    for f in FACTORS {
        let cand = period * f;
        let bpm = 60.0 * frame_rate / cand;
        if !(MIN_CANONICAL_BPM..MAX_CANONICAL_BPM).contains(&bpm) {
            continue;
        }
        let s = corr(cand) * tempo_prior(bpm);
        if s > best_score {
            best_score = s;
            best = cand;
        }
    }
    if best_score == f32::NEG_INFINITY {
        let mut p = period;
        let mut bpm = 60.0 * frame_rate / p;
        while bpm < MIN_CANONICAL_BPM {
            bpm *= 2.0;
            p /= 2.0;
        }
        while bpm >= MAX_CANONICAL_BPM {
            bpm /= 2.0;
            p *= 2.0;
        }
        return p;
    }
    best
}

/// Folds the onset envelope onto one beat period. Returns (sharpness, phase in frames):
/// the share of all onset energy in the densest ~1/16 of the period, and where it sits.
fn fold(flux: &[f32], period: f32) -> (f32, f32) {
    const M: usize = 64;
    let mut hist = [0.0f32; M];
    let mut total = 0.0f32;
    let scale = M as f32 / period;
    for (i, &v) in flux.iter().enumerate() {
        if v <= 0.0 {
            continue;
        }
        let pos = (i as f32 % period) * scale;
        let b0 = pos.floor() as usize % M;
        let frac = pos - pos.floor();
        hist[b0] += v * (1.0 - frac);
        hist[(b0 + 1) % M] += v * frac;
        total += v;
    }
    if total <= 0.0 {
        return (0.0, 0.0);
    }
    // Circular smoothing over ±2 bins (≈ ±1/32 beat).
    let mut smooth = [0.0f32; M];
    for (b, s) in smooth.iter_mut().enumerate() {
        *s = hist[(b + M - 2) % M] * 0.1
            + hist[(b + M - 1) % M] * 0.2
            + hist[b] * 0.4
            + hist[(b + 1) % M] * 0.2
            + hist[(b + 2) % M] * 0.1;
    }
    let (mut arg, mut peak) = (0usize, -1.0f32);
    for (b, &s) in smooth.iter().enumerate() {
        if s > peak {
            peak = s;
            arg = b;
        }
    }
    let (y0, y1, y2) = (smooth[(arg + M - 1) % M], smooth[arg], smooth[(arg + 1) % M]);
    let d = y0 - 2.0 * y1 + y2;
    let delta = if d != 0.0 { (0.5 * (y0 - y2) / d).clamp(-0.5, 0.5) } else { 0.0 };
    let phase = ((arg as f32 + delta).rem_euclid(M as f32)) / scale;
    // A perfect pulse train puts 0.4 of the energy into the smoothed peak bin.
    (peak / total / 0.4, phase)
}

/// Searches ±2 % around `period` for the tempo whose grid lines up best over the whole window,
/// then snaps to a whole or half BPM if that fits just as well.
/// Returns (period, phase, sharpness).
fn refine_period(flux: &[f32], frame_rate: f32, period: f32) -> (f32, f32, f32) {
    let mut best = (period, 0.0f32, -1.0f32);
    let steps = 240;
    for s in 0..=steps {
        let p = period * (0.98 + 0.04 * s as f32 / steps as f32);
        let (sharp, phase) = fold(flux, p);
        if sharp > best.2 {
            best = (p, phase, sharp);
        }
    }
    // 30 s cannot tell 0.03 BPM apart (under one histogram bin of drift), five minutes of
    // mixing can. Productions sit on whole (rarely half) BPM, so take those when they fit.
    let bpm = 60.0 * frame_rate / best.0;
    for (snapped, max_off, keep) in [(bpm.round(), 0.15, 0.95), ((bpm * 2.0).round() / 2.0, 0.08, 0.97)] {
        if (snapped - bpm).abs() > max_off || snapped <= 0.0 {
            continue;
        }
        let p = 60.0 * frame_rate / snapped;
        let (sharp, phase) = fold(flux, p);
        if sharp >= best.2 * keep {
            return (p, phase, sharp);
        }
    }
    best
}

/// Which beat of the grid is "the One", and which bar opens a phrase.
pub struct Downbeat {
    pub bar_phase_beats: i64,
    pub phrase_phase_bars: i64,
    pub confidence: f32,
    pub phrase_confidence: f32,
}

const BEAT_TOLERANCE: f32 = 0.12;
const W_KICK: f32 = 1.0;
const W_BACKBEAT: f32 = 1.3;
const W_SNARE_ON_ONE: f32 = 0.8;
const W_NOVELTY: f32 = 0.6;

/// Classifies downbeat and phrase phase from the 4/4 drum template: kick on the one, snare/clap
/// on 2 and 4, arrangement changes on phrase boundaries (port of KittyTune's DownbeatTracker).
pub fn estimate_downbeat(onsets: &BandOnsets, period: f32, beat_phase: f32) -> Downbeat {
    let none = Downbeat { bar_phase_beats: 0, phrase_phase_bars: 0, confidence: 0.0, phrase_confidence: 0.0 };
    let n = onsets.len();
    if period <= 0.0 || n == 0 {
        return none;
    }
    let phase = beat_phase.rem_euclid(period);
    let count = ((n as f32 - phase) / period) as usize;
    let beats: Vec<usize> = (0..count).map(|k| ((phase + k as f32 * period).round() as usize).min(n - 1)).collect();
    if beats.len() < BEATS_PER_BAR as usize * 2 {
        return none;
    }
    let tol = ((period * BEAT_TOLERANCE).round() as usize).max(1);
    let sample = |band: &[f32]| -> Vec<f32> {
        beats
            .iter()
            .map(|&c| {
                let lo = c.saturating_sub(tol);
                let hi = (c + tol).min(band.len() - 1);
                band[lo..=hi].iter().copied().fold(0.0, f32::max)
            })
            .collect()
    };
    let kick = sample(&onsets.low);
    let mids = sample(&onsets.mid);
    let highs = sample(&onsets.high);
    let snare: Vec<f32> = mids.iter().zip(&highs).map(|(m, h)| (m * h).sqrt()).collect();

    let span = (period.round() as usize).max(1);
    let novelty: Vec<f32> = beats
        .iter()
        .map(|&c| {
            let before_lo = c.saturating_sub(span);
            let after_hi = (c + span).min(n - 1);
            if c - before_lo < span / 2 || after_hi - c < span / 2 {
                return 0.0;
            }
            let mut diff = 0.0;
            for band in [&onsets.low, &onsets.mid, &onsets.high] {
                let before: f32 = band[before_lo..c].iter().sum::<f32>() / (c - before_lo).max(1) as f32;
                let after: f32 = band[c..=after_hi].iter().sum::<f32>() / (after_hi - c + 1) as f32;
                diff += (after - before).abs();
            }
            diff
        })
        .collect();

    let bpb = BEATS_PER_BAR as usize;
    let at = |v: &[f32], p: usize, m: usize| -> f32 {
        let (mut s, mut c) = (0.0, 0);
        let mut i = p;
        while i < v.len() {
            s += v[i];
            c += 1;
            i += m;
        }
        if c == 0 { 0.0 } else { s / c as f32 }
    };
    let z_kick = standardize(&(0..bpb).map(|c| at(&kick, c, bpb)).collect::<Vec<_>>());
    let z_snare = standardize(&(0..bpb).map(|c| at(&snare, c, bpb)).collect::<Vec<_>>());
    let z_nov = standardize(&(0..bpb).map(|c| at(&novelty, c, bpb)).collect::<Vec<_>>());
    let scores: Vec<f32> = (0..bpb)
        .map(|c| {
            let backbeat = 0.5 * (z_snare[(c + 1) % bpb] + z_snare[(c + bpb - 1) % bpb]);
            W_KICK * z_kick[c] + W_BACKBEAT * backbeat - W_SNARE_ON_ONE * z_snare[c] + W_NOVELTY * z_nov[c]
        })
        .collect();
    let (bar_phase, bar_conf) = pick(&scores);

    // Phrase: which bar's downbeat accumulates the most novelty (and kick, breaks end on it).
    let bpp = BARS_PER_PHRASE as usize;
    let (mut dn, mut dk) = (Vec::new(), Vec::new());
    let mut b = bar_phase;
    while b < novelty.len() {
        dn.push(novelty[b]);
        dk.push(kick[b]);
        b += bpb;
    }
    let (phrase_phase, phrase_conf) = if dn.len() < bpp * 2 {
        (0, 0.0)
    } else {
        let zn = standardize(&(0..bpp).map(|p| at(&dn, p, bpp)).collect::<Vec<_>>());
        let zk = standardize(&(0..bpp).map(|p| at(&dk, p, bpp)).collect::<Vec<_>>());
        pick(&(0..bpp).map(|p| zn[p] + 0.4 * zk[p]).collect::<Vec<_>>())
    };
    Downbeat {
        bar_phase_beats: bar_phase as i64,
        phrase_phase_bars: phrase_phase as i64,
        confidence: bar_conf,
        phrase_confidence: phrase_conf,
    }
}

/// Winner and its z-scored margin over the runner-up (≈2 is decisive → 1.0).
fn pick(scores: &[f32]) -> (usize, f32) {
    let mut best = 0;
    for (i, &s) in scores.iter().enumerate() {
        if s > scores[best] {
            best = i;
        }
    }
    let runner = scores.iter().enumerate().filter(|&(i, _)| i != best).map(|(_, &s)| s).fold(f32::NEG_INFINITY, f32::max);
    let margin = if runner.is_finite() { ((scores[best] - runner) / 2.0).clamp(0.0, 1.0) } else { 0.0 };
    (best, margin)
}

fn standardize(v: &[f32]) -> Vec<f32> {
    if v.is_empty() {
        return Vec::new();
    }
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / v.len() as f32;
    let sd = var.sqrt();
    if sd <= 1e-9 {
        return vec![0.0; v.len()];
    }
    v.iter().map(|x| (x - mean) / sd).collect()
}

/// Beat grid of the analysed window, resolved up to phrase level (times in ms on the track).
pub struct Grid {
    pub bpm: f32,
    pub period_ms: f64,
    pub confidence: f32,
    pub first_beat_ms: f64,
    pub downbeat_ms: f64,
    pub phrase_ms: f64,
    pub downbeat_confidence: f32,
    pub phrase_confidence: f32,
}

/// Full grid pipeline on onsets of a window that starts at `window_start_ms` in the track.
pub fn compute_grid(onsets: &BandOnsets, window_start_ms: f64) -> Option<Grid> {
    if onsets.len() < 8 {
        return None;
    }
    let fr = onsets.frame_rate;
    let (raw, ac_conf) = estimate_tempo_period(&onsets.full, fr)?;
    let period0 = resolve_tempo_octave(&onsets.full, fr, raw);
    if period0 <= 0.0 {
        return None;
    }
    // The autocorrelation winner can be a subdivision the octave factors don't reach
    // (68 BPM in drum & bass: 170 is ×2.5). Every strong peak × every musical ratio is
    // folded over the whole window; the tempo whose grid gathers the onsets best wins.
    let mut candidates = vec![period0];
    for lag in autocorr_peaks(&onsets.full, fr, 6) {
        for f in [0.25f32, 1.0 / 3.0, 0.4, 0.5, 2.0 / 3.0, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0] {
            let p = lag * f;
            let bpm = 60.0 * fr / p;
            if (MIN_CANONICAL_BPM..MAX_CANONICAL_BPM).contains(&bpm)
                && candidates.iter().all(|&c| (c / p - 1.0).abs() > 0.015)
            {
                candidates.push(p);
            }
        }
    }
    let score = |p: f32, sharp: f32| sharp * tempo_prior(60.0 * fr / p).powf(0.3);
    let (mut period, mut phase, mut sharp) = refine_period(&onsets.full, fr, period0);
    let mut best = score(period, sharp);
    for &c in &candidates[1..] {
        // Cheap screen first; only promising candidates get the fine search.
        let (s0, _) = fold(&onsets.full, c);
        if score(c, s0) < best * 0.8 {
            continue;
        }
        let (p, ph, sh) = refine_period(&onsets.full, fr, c);
        let sc = score(p, sh);
        if sc > best * 1.05 {
            (period, phase, sharp, best) = (p, ph, sh, sc);
        }
    }
    let period_ms = period as f64 / fr as f64 * 1000.0;
    let bpm = (60_000.0 / period_ms) as f32;
    let down = estimate_downbeat(onsets, period, phase);

    // Beat 0 of the window, then extrapolated back to the track start.
    let anchor_ms = window_start_ms + (onsets.frame0_s as f64 + phase as f64 / fr as f64) * 1000.0;
    let first_beat_ms = anchor_ms - (anchor_ms / period_ms).floor() * period_ms;
    let anchor_beat = ((anchor_ms - first_beat_ms) / period_ms).round() as i64;
    let first_down_idx = (anchor_beat + down.bar_phase_beats).rem_euclid(BEATS_PER_BAR);
    let downbeat_ms = first_beat_ms + first_down_idx as f64 * period_ms;
    let bars_from_down = (anchor_beat + down.bar_phase_beats - first_down_idx) / BEATS_PER_BAR;
    let first_phrase_bar = (bars_from_down + down.phrase_phase_bars).rem_euclid(BARS_PER_PHRASE);
    let phrase_ms = downbeat_ms + (first_phrase_bar * BEATS_PER_BAR) as f64 * period_ms;

    // Sharpness ≈ 1 for a perfect pulse train, ~0.04 for noise, 0.3–0.6 for real club music.
    let confidence = (0.5 * ac_conf + 0.5 * ((sharp - 0.08) / 0.4).clamp(0.0, 1.0)).clamp(0.0, 1.0);
    Some(Grid {
        bpm,
        period_ms,
        confidence,
        first_beat_ms: first_beat_ms.max(0.0),
        downbeat_ms: downbeat_ms.max(0.0),
        phrase_ms: phrase_ms.max(0.0),
        downbeat_confidence: down.confidence,
        phrase_confidence: down.phrase_confidence,
    })
}

// ---------- Key ----------

const KS_MAJOR: [f32; 12] = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const KS_MINOR: [f32; 12] = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];
const TEMPERLEY_MAJOR: [f32; 12] = [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0];
const TEMPERLEY_MINOR: [f32; 12] = [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0];

fn pearson(chroma: &[f32; 12], profile: &[f32; 12], tonic: usize) -> f32 {
    let cm = chroma.iter().sum::<f32>() / 12.0;
    let pm = profile.iter().sum::<f32>() / 12.0;
    let (mut num, mut dc, mut dp) = (0.0f32, 0.0f32, 0.0f32);
    for d in 0..12 {
        let c = chroma[(d + tonic) % 12] - cm;
        let p = profile[d] - pm;
        num += c * p;
        dc += c * c;
        dp += p * p;
    }
    if dc <= 1e-12 { 0.0 } else { num / (dc * dp).sqrt() }
}

/// Estimates the key from tuned chroma of spectral peaks. Returns (pitch class, minor, confidence).
pub fn estimate_key(x: &[f32], rate: u32) -> Option<(u8, bool, f32)> {
    if x.len() < KEY_FFT * 4 {
        return None;
    }
    let frames = (x.len() - KEY_FFT) / KEY_HOP + 1;
    let bin_hz = rate as f32 / KEY_FFT as f32;
    let min_bin = ((55.0 / bin_hz) as usize).max(2);
    let bass_hi = (220.0 / bin_hz) as usize;
    let max_bin = ((2000.0 / bin_hz) as usize).min(KEY_FFT / 2 - 2);

    let window = hann(KEY_FFT);
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(KEY_FFT);
    let mut input = r2c.make_input_vec();
    let mut spec: Vec<Complex<f32>> = r2c.make_output_vec();
    let mut scratch = r2c.make_scratch_vec();
    let mut mag = vec![0.0f32; KEY_FFT / 2 + 1];

    // Spectral peaks: (frame, fractional MIDI pitch, weight).
    let mut peaks: Vec<(u32, f32, f32)> = Vec::new();
    let mut frame_weight = vec![0.0f32; frames];
    for f in 0..frames {
        let off = f * KEY_HOP;
        for i in 0..KEY_FFT {
            input[i] = x[off + i] * window[i];
        }
        if r2c.process_with_scratch(&mut input, &mut spec, &mut scratch).is_err() {
            continue;
        }
        let mut fmax = 0.0f32;
        for b in min_bin - 1..=max_bin + 1 {
            mag[b] = spec[b].norm();
            fmax = fmax.max(mag[b]);
        }
        if fmax <= 1e-6 {
            continue;
        }
        let floor = fmax * 0.05;
        let mut sum = 0.0;
        for b in min_bin..=max_bin {
            let m = mag[b];
            if m < floor || m < mag[b - 1] || m <= mag[b + 1] {
                continue;
            }
            // Parabolic peak on log magnitude for a sub-bin frequency.
            let (a, bb, c) = ((mag[b - 1] + 1e-9).ln(), (m + 1e-9).ln(), (mag[b + 1] + 1e-9).ln());
            let d = a - 2.0 * bb + c;
            let delta = if d != 0.0 { (0.5 * (a - c) / d).clamp(-0.5, 0.5) } else { 0.0 };
            let hz = (b as f32 + delta) * bin_hz;
            let midi = 69.0 + 12.0 * (hz / 440.0).log2();
            let w = m.sqrt();
            peaks.push((f as u32, midi, w * if b <= bass_hi { -1.0 } else { 1.0 }));
            sum += w;
        }
        frame_weight[f] = sum;
    }
    if peaks.len() < 64 {
        return None;
    }

    // Tuning: circular mean of the peaks' deviation from equal temperament.
    let (mut sx, mut sy) = (0.0f32, 0.0f32);
    for &(_, midi, w) in &peaks {
        let a = 2.0 * std::f32::consts::PI * midi;
        sx += w.abs() * a.cos();
        sy += w.abs() * a.sin();
    }
    let tuning = sy.atan2(sx) / (2.0 * std::f32::consts::PI);

    let mut treble = [0.0f32; 12];
    let mut bass = [0.0f32; 12];
    for &(f, midi, w) in &peaks {
        // Each frame counts equally, however loud it is.
        let norm = frame_weight[f as usize].max(1e-6);
        let p = midi - tuning;
        let s = p.round();
        let frac = p - s;
        let wt = (std::f32::consts::PI * frac).cos().powi(2) * w.abs() / norm;
        let pc = (s as i32).rem_euclid(12) as usize;
        if w < 0.0 {
            bass[pc] += wt;
        } else {
            treble[pc] += wt;
        }
    }
    let normalize = |c: &mut [f32; 12]| {
        let s: f32 = c.iter().sum();
        if s > 0.0 {
            c.iter_mut().for_each(|v| *v /= s);
        }
    };
    normalize(&mut treble);
    normalize(&mut bass);
    let mut chroma = [0.0f32; 12];
    for i in 0..12 {
        chroma[i] = treble[i] + 0.5 * bass[i];
    }

    let mut scores: Vec<(f32, u8, bool)> = Vec::with_capacity(24);
    for t in 0..12 {
        scores.push((pearson(&chroma, &KS_MAJOR, t) + pearson(&chroma, &TEMPERLEY_MAJOR, t), t as u8, false));
        scores.push((pearson(&chroma, &KS_MINOR, t) + pearson(&chroma, &TEMPERLEY_MINOR, t), t as u8, true));
    }
    scores.sort_by(|a, b| b.0.total_cmp(&a.0));
    let (best, pc, minor) = scores[0];
    if best <= 0.2 {
        return None;
    }
    // Margin over the best key that is not the relative major/minor (those share a Camelot
    // number, so confusing them costs a DJ little).
    let relative = if minor { (pc + 3) % 12 } else { (pc + 9) % 12 };
    let second = scores
        .iter()
        .skip(1)
        .find(|&&(_, p, m)| !(m != minor && p == relative))
        .map(|s| s.0)
        .unwrap_or(0.0);
    let confidence = ((best - second) / 0.5).clamp(0.0, 1.0) * (best / 2.0).clamp(0.0, 1.0);
    Some((pc, minor, confidence))
}

/// Camelot code for a key, e.g. (9, true) → "8A" (A minor).
pub fn camelot(pitch: u8, minor: bool) -> &'static str {
    crate::energy::pitch_to_camelot(pitch as usize, minor)
}

// ---------- Loudness / energy ----------

/// Integrated loudness (BS.1770-4 K-weighting, 400 ms blocks, absolute and relative gates) of a
/// mono downmix, reported as the stereo programme it came from (+3 dB).
pub fn integrated_lufs(x: &[f32], rate: u32) -> f32 {
    use crate::dsp::biquad::{Biquad, Coeffs};
    let fs = rate as f64;
    let shelf = shelf_coeffs(fs);
    let hp = Coeffs::highpass(38.135_47, 0.500_327, fs);
    let (mut s1, mut s2) = (Biquad::default(), Biquad::default());
    let block = (0.4 * fs) as usize;
    let step = block / 4;
    if x.len() < block {
        return -70.0;
    }
    let weighted: Vec<f32> = x.iter().map(|&v| s2.process(&hp, s1.process(&shelf, v))).collect();
    let mut prefix = vec![0.0f64; weighted.len() + 1];
    for (i, v) in weighted.iter().enumerate() {
        prefix[i + 1] = prefix[i] + (*v as f64) * (*v as f64);
    }
    let blocks: Vec<f64> = (0..=(weighted.len() - block) / step)
        .map(|k| (prefix[k * step + block] - prefix[k * step]) / block as f64)
        .collect();
    let lufs = |ms: f64| -0.691 + 10.0 * ms.max(1e-12).log10() + 3.01;
    let gated: Vec<f64> = blocks.iter().copied().filter(|&m| lufs(m) > -70.0).collect();
    if gated.is_empty() {
        return -70.0;
    }
    let rel = lufs(gated.iter().sum::<f64>() / gated.len() as f64) - 10.0;
    let kept: Vec<f64> = gated.into_iter().filter(|&m| lufs(m) > rel).collect();
    if kept.is_empty() {
        return -70.0;
    }
    lufs(kept.iter().sum::<f64>() / kept.len() as f64) as f32
}

/// BS.1770 stage-1 high shelf (+4 dB above ~1.7 kHz) for any sample rate.
fn shelf_coeffs(fs: f64) -> crate::dsp::biquad::Coeffs {
    use std::f64::consts::PI;
    let (f0, g, q) = (1681.974_450_955_533, 3.999_843_853_973_347, 0.707_175_236_955_419_6);
    let k = (PI * f0 / fs).tan();
    let vh = 10f64.powf(g / 20.0);
    let vb = vh.powf(0.499_666_774_155);
    let a0 = 1.0 + k / q + k * k;
    crate::dsp::biquad::Coeffs {
        b0: ((vh + vb * k / q + k * k) / a0) as f32,
        b1: (2.0 * (k * k - vh) / a0) as f32,
        b2: ((vh - vb * k / q + k * k) / a0) as f32,
        a1: (2.0 * (k * k - 1.0) / a0) as f32,
        a2: ((1.0 - k / q + k * k) / a0) as f32,
    }
}

fn onset_density(flux: &[f32], frame_rate: f32) -> f32 {
    if flux.len() < 3 {
        return 0.0;
    }
    let mut sorted = flux.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let thr = sorted[sorted.len() * 3 / 4].max(1e-6) * 1.5;
    let mut count = 0usize;
    let mut last = 0usize;
    for i in 1..flux.len() - 1 {
        if flux[i] > thr && flux[i] >= flux[i - 1] && flux[i] > flux[i + 1] && i - last > 4 {
            count += 1;
            last = i;
        }
    }
    count as f32 / (flux.len() as f32 / frame_rate)
}

fn unit(v: f32, lo: f32, hi: f32) -> f32 {
    ((v - lo) / (hi - lo)).clamp(0.0, 1.0)
}

// ---------- Mix points from the whole-track envelope ----------

fn percentile(v: &[f32], p: f32) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    s[(((s.len() - 1) as f32 * p).round() as usize).min(s.len() - 1)]
}

/// Resamples an amplitude envelope of the whole track into 500 ms blocks.
pub(crate) fn blocks_500ms(env: &[f32], duration_ms: f64) -> Vec<f32> {
    let n = (duration_ms / 500.0).floor() as usize;
    if env.is_empty() || n == 0 {
        return Vec::new();
    }
    (0..n)
        .map(|b| {
            let lo = (b as f64 * 500.0 / duration_ms * env.len() as f64) as usize;
            let hi = (((b + 1) as f64 * 500.0 / duration_ms * env.len() as f64) as usize).clamp(lo + 1, env.len());
            env[lo.min(env.len() - 1)..hi].iter().copied().fold(0.0, f32::max)
        })
        .collect()
}

/// First sustained body-level block, snapped forward onto a 16-beat phrase boundary.
fn detect_mix_in(env: &[f32], phrase_anchor_ms: f64, period_ms: f64) -> Option<f64> {
    if env.len() < 8 || period_ms <= 0.0 {
        return None;
    }
    let r = percentile(env, 0.75);
    if r <= 0.0 {
        return None;
    }
    let cand = (0..env.len() - 4)
        .find(|&i| env[i] >= 0.55 * r && env[i + 1..i + 4].iter().all(|&e| e >= 0.4 * r))?;
    if cand == 0 {
        return None;
    }
    let cand_ms = cand as f64 * 500.0;
    if cand_ms > MAX_INTRO_SKIP_MS {
        return None;
    }
    let phrase = BEATS_PER_PHRASE as f64 * period_ms;
    let k = ((cand_ms - phrase_anchor_ms) / phrase).ceil().max(0.0);
    let drop_in = phrase_anchor_ms + k * phrase;
    Some(if drop_in <= MAX_INTRO_SKIP_MS { drop_in } else { phrase_anchor_ms })
}

/// Last moment at body loudness (the outro starts after it), on a phrase boundary.
fn detect_mix_out(env: &[f32], duration_ms: f64, phrase_anchor_ms: f64, period_ms: f64) -> Option<f64> {
    if env.len() < 8 || duration_ms <= 0.0 {
        return None;
    }
    let r = percentile(env, 0.75);
    if r <= 0.0 {
        return None;
    }
    let last = env.iter().rposition(|&e| e >= 0.5 * r)?;
    let raw = (last + 1) as f64 * 500.0;
    if duration_ms - raw < 3_000.0 {
        return None;
    }
    let out = raw.max(duration_ms - MAX_OUTRO_CUT_MS);
    if period_ms > 0.0 {
        let phrase = BEATS_PER_PHRASE as f64 * period_ms;
        let k = ((out - phrase_anchor_ms) / phrase).floor().max(0.0);
        return Some((phrase_anchor_ms + k * phrase).clamp(phrase_anchor_ms, duration_ms - 3_000.0));
    }
    Some(out)
}

// ---------- Entry point ----------

/// Analyses mono PCM of a window that starts at `window_start_ms` in the track.
///
/// `envelope` is an optional amplitude envelope of the *whole* track (e.g. SoundCloud's waveform)
/// used for mix points; without it only the analysed window is known.
pub fn analyze(
    mono: &[f32],
    sample_rate: u32,
    window_start_ms: f64,
    duration_ms: f64,
    envelope: Option<&[f32]>,
) -> Option<TrackAnalysis> {
    if sample_rate == 0 || mono.len() < sample_rate as usize * 4 {
        return None;
    }
    let x: std::borrow::Cow<[f32]> = if sample_rate == ANALYSIS_RATE {
        std::borrow::Cow::Borrowed(mono)
    } else {
        std::borrow::Cow::Owned(crate::fakeprint::resample(mono, sample_rate, ANALYSIS_RATE))
    };
    let rate = ANALYSIS_RATE;
    let sp = spectra(&x, rate);
    let grid = compute_grid(&sp.onsets, window_start_ms)?;
    let key = estimate_key(&x, rate);

    let total: f64 = sp.band_power.iter().sum::<f64>().max(1e-12);
    let (bass, mid, high) =
        ((sp.band_power[0] / total) as f32, (sp.band_power[1] / total) as f32, (sp.band_power[2] / total) as f32);
    let lufs = integrated_lufs(&x, rate);
    let peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let density = onset_density(&sp.onsets.full, sp.onsets.frame_rate);
    let energy = 1.0
        + 9.0
            * (0.45 * unit(lufs, -18.0, -6.0)
                + 0.25 * unit(density, 1.0, 8.0)
                + 0.15 * unit(grid.bpm, 80.0, 180.0)
                + 0.15 * unit(high, 0.01, 0.12));

    let (mix_in, mix_out) = match envelope {
        Some(env) if duration_ms > 0.0 => {
            let blocks = blocks_500ms(env, duration_ms);
            (
                detect_mix_in(&blocks, grid.phrase_ms, grid.period_ms),
                detect_mix_out(&blocks, duration_ms, grid.phrase_ms, grid.period_ms),
            )
        }
        _ => (None, None),
    };

    Some(TrackAnalysis {
        bpm: grid.bpm,
        bpm_confidence: grid.confidence,
        first_beat_ms: grid.first_beat_ms,
        downbeat_ms: grid.downbeat_ms,
        phrase_ms: grid.phrase_ms,
        downbeat_confidence: grid.downbeat_confidence,
        phrase_confidence: grid.phrase_confidence,
        key_pitch: key.map(|k| k.0),
        key_minor: key.map(|k| k.1).unwrap_or(false),
        key_confidence: key.map(|k| k.2).unwrap_or(0.0),
        mix_in_ms: mix_in,
        mix_out_ms: mix_out,
        lufs,
        peak_db: 20.0 * peak.max(1e-9).log10(),
        bass,
        mid,
        high,
        onset_density: density,
        energy,
        analyzed_ms: x.len() as f64 / rate as f64 * 1000.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic 4/4 loop: kick on every beat, clap on 2 and 4, hats on the off-beats, and a
    /// sustained A-minor chord (A-C-E) with an A bassline.
    pub fn synth(bpm: f64, seconds: f64, offset_ms: f64, rate: u32) -> Vec<f32> {
        let n = (seconds * rate as f64) as usize;
        let period = 60.0 / bpm;
        let mut x = vec![0.0f32; n];
        let mut noise = 12345u32;
        for (i, s) in x.iter_mut().enumerate() {
            let t = i as f64 / rate as f64;
            let tb = t - offset_ms / 1000.0;
            let mut v = 0.0f64;
            if tb >= 0.0 {
                let beat = (tb / period).floor();
                let ph = tb - beat * period;
                let beat_in_bar = (beat as i64).rem_euclid(4);
                // Kick: pitched-down sine burst.
                if ph < 0.12 {
                    let f = 50.0 + 90.0 * (-ph * 30.0).exp();
                    v += 0.8 * (2.0 * std::f64::consts::PI * f * ph).sin() * (-ph * 25.0).exp();
                }
                noise = noise.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                let white = (noise >> 8) as f64 / (1u32 << 24) as f64 - 0.5;
                if (beat_in_bar == 1 || beat_in_bar == 3) && ph < 0.15 {
                    v += 0.5 * white * (-ph * 30.0).exp();
                }
                let off = (ph - period / 2.0).abs();
                if off < 0.03 {
                    v += 0.15 * white * (1.0 - off / 0.03);
                }
            }
            for f in [220.0, 261.63, 329.63, 110.0, 55.0] {
                v += 0.05 * (2.0 * std::f64::consts::PI * f * t).sin();
            }
            *s = v as f32;
        }
        x
    }

    #[test]
    fn detects_tempo_grid_and_key() {
        for &(bpm, offset) in &[(128.0, 137.0), (160.0, 45.0), (174.0, 300.0), (140.0, 0.0)] {
            let x = synth(bpm, 30.0, offset, ANALYSIS_RATE);
            let a = analyze(&x, ANALYSIS_RATE, 0.0, 30_000.0, None).expect("analysis");
            assert!((a.bpm as f64 - bpm).abs() < 0.05, "bpm {} vs {bpm}", a.bpm);
            let period = 60_000.0 / bpm;
            let err = ((a.first_beat_ms - offset).rem_euclid(period)).min((offset - a.first_beat_ms).rem_euclid(period));
            assert!(err < 8.0, "beat phase off by {err} ms at {bpm} BPM");
            // Downbeat: kick without clap sits on beats 0 and 2; the clap marks 1 and 3.
            let db_beat = ((a.downbeat_ms - offset) / period).round() as i64;
            assert_eq!(db_beat.rem_euclid(2), 0, "downbeat on a clap beat at {bpm} BPM");
            assert_eq!(a.key_pitch, Some(9), "key pitch at {bpm} BPM");
            assert!(a.key_minor, "A minor expected");
            assert_eq!(camelot(9, true), "8A");
        }
    }

    #[test]
    fn window_offset_moves_grid() {
        let bpm = 128.0;
        let x = synth(bpm, 25.0, 0.0, ANALYSIS_RATE);
        // Pretend the window started 61.234 s into the track: beat lines must follow.
        let a = analyze(&x, ANALYSIS_RATE, 61_234.0, 240_000.0, None).unwrap();
        let period = 60_000.0 / bpm;
        let err = ((a.first_beat_ms - 61_234.0).rem_euclid(period)).min((61_234.0 - a.first_beat_ms).rem_euclid(period));
        assert!(err < 8.0, "grid off by {err} ms");
        assert!(a.first_beat_ms < period);
    }

    #[test]
    fn loudness_of_full_scale_sine() {
        // 997 Hz sine at −20 dBFS (mono → reported as stereo programme ≈ −20 LUFS).
        let rate = 48_000;
        let x: Vec<f32> = (0..rate * 5)
            .map(|i| 0.1 * (2.0 * std::f32::consts::PI * 997.0 * i as f32 / rate as f32).sin())
            .collect();
        let l = integrated_lufs(&x, rate);
        assert!((l - -20.0).abs() < 0.5, "lufs {l}");
    }
}

/// Debug helper for tuning: tempo candidates of a window (examples only).
#[doc(hidden)]
pub fn debug_tempo(mono: &[f32], sample_rate: u32) {
    let x = crate::fakeprint::resample(mono, sample_rate, ANALYSIS_RATE);
    let sp = spectra(&x, ANALYSIS_RATE);
    let fr = sp.onsets.frame_rate;
    let (raw, conf) = estimate_tempo_period(&sp.onsets.full, fr).unwrap();
    let oct = resolve_tempo_octave(&sp.onsets.full, fr, raw);
    let (p, _, sharp) = refine_period(&sp.onsets.full, fr, oct);
    eprintln!("raw {:.2} BPM (conf {conf:.2}) → octave {:.2} → refined {:.2} (sharp {sharp:.3})", 60.0 * fr / raw, 60.0 * fr / oct, 60.0 * fr / p);
    for bpm in [85.0f32, 113.3, 124.0, 125.0, 136.0, 170.0, 172.0, 174.0] {
        let (s, _) = fold(&sp.onsets.full, 60.0 * fr / bpm);
        eprintln!("  fold {bpm:>6.1}: {s:.3}  prior {:.2}", tempo_prior(bpm));
    }
}
