// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Jason-Marshall Fastner <jasonfastner@protonmail.com>

//! Cue and drop detection for DJ transitions.
//!
//! * **Cue**: the first sustained onset after the silence at the start of a track (where a DJ drops
//!   the needle). Searched in the level curve in 10 ms steps, relative to the loudest part of the excerpt.
//! * **Drop**: the point where the bass returns after a breakdown. Roughly from the envelope of
//!   the whole track ([`drop_candidate`]), precisely from the bass level (< 150 Hz) of an excerpt
//!   around the candidate ([`detect_drop`]), then snapped to the bar.

use crate::dj::blocks_500ms;

/// Level grid.
const HOP_MS: f64 = 10.0;
/// Cut-off frequency for "bass" (kick + sub).
const BASS_HZ: f32 = 150.0;

/// RMS per 10 ms window.
fn rms_frames(x: &[f32], rate: u32) -> Vec<f32> {
    let hop = ((rate as f64 * HOP_MS / 1000.0) as usize).max(1);
    x.chunks(hop)
        .map(|c| (c.iter().map(|v| v * v).sum::<f32>() / c.len() as f32).sqrt())
        .collect()
}

/// 4th-order low-pass (two RBJ biquads, Q = 0.707).
fn lowpass(x: &[f32], rate: u32, fc: f32) -> Vec<f32> {
    let w = std::f32::consts::TAU * fc / rate as f32;
    let (s, c) = w.sin_cos();
    let alpha = s / (2.0 * std::f32::consts::FRAC_1_SQRT_2);
    let a0 = 1.0 + alpha;
    let b0 = (1.0 - c) / 2.0 / a0;
    let b1 = (1.0 - c) / a0;
    let a1 = -2.0 * c / a0;
    let a2 = (1.0 - alpha) / a0;
    let mut y = x.to_vec();
    for _ in 0..2 {
        let (mut x1, mut x2, mut y1, mut y2) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for v in y.iter_mut() {
            let x0 = *v;
            let out = b0 * x0 + b1 * x1 + b0 * x2 - a1 * y1 - a2 * y2;
            x2 = x1;
            x1 = x0;
            y2 = y1;
            y1 = out;
            *v = out;
        }
    }
    y
}

/// Moving average over `n` values (centred).
fn smooth(v: &[f32], n: usize) -> Vec<f32> {
    if n <= 1 || v.is_empty() {
        return v.to_vec();
    }
    let mut prefix = vec![0.0f64; v.len() + 1];
    for (i, x) in v.iter().enumerate() {
        prefix[i + 1] = prefix[i] + *x as f64;
    }
    let half = n / 2;
    (0..v.len())
        .map(|i| {
            let lo = i.saturating_sub(half);
            let hi = (i + half + 1).min(v.len());
            ((prefix[hi] - prefix[lo]) / (hi - lo) as f64) as f32
        })
        .collect()
}

fn mean(v: &[f32]) -> f32 {
    if v.is_empty() { 0.0 } else { v.iter().sum::<f32>() / v.len() as f32 }
}

fn percentile(v: &[f32], p: f32) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    s[(((s.len() - 1) as f32 * p).round() as usize).min(s.len() - 1)]
}

/// First sustained onset after silence, in ms on the track timeline.
///
/// `x` is mono PCM of an excerpt starting at `clip_start_ms` (for the cue: the beginning).
/// `None` if the excerpt is silent throughout.
pub fn detect_cue(x: &[f32], rate: u32, clip_start_ms: f64) -> Option<f64> {
    let f = rms_frames(x, rate);
    if f.len() < 30 {
        return None;
    }
    let loud = percentile(&f, 0.95);
    // Silence: below −50 dBFS or below 3 % (−30 dB) of the loud part – whichever is higher.
    let thr = (loud * 0.03).max(10f32.powf(-50.0 / 20.0));
    if loud <= thr {
        return None;
    }
    // Sustained = 200 ms above the threshold on average (a single click doesn't count).
    let hold = 20;
    let start = (0..f.len().saturating_sub(hold)).find(|&i| f[i] > thr && mean(&f[i..i + hold]) > thr * 2.0)?;
    // Back to the attack: the first window before it that already rises clearly out of the silence.
    let mut i = start;
    while i > 0 && start - i < 10 && f[i - 1] > thr * 0.25 {
        i -= 1;
    }
    Some(clip_start_ms + i as f64 * HOP_MS)
}

/// A detected drop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Drop {
    /// Time on the track timeline (ms), snapped to downbeat or beat where possible.
    pub ms: f64,
    /// 0..1: how clearly the bass comes back (level jump relative to full bass).
    pub strength: f32,
}

fn bass_frames(x: &[f32], rate: u32) -> Vec<f32> {
    smooth(&rms_frames(&lowpass(x, rate, BASS_HZ), rate), 5)
}

/// Bass level (dBFS, < 150 Hz) of the loud part of an excerpt – measured from the middle of the track
/// it's the reference for what "full bass" sounds like in this track.
pub fn bass_level_db(x: &[f32], rate: u32) -> f32 {
    20.0 * percentile(&bass_frames(x, rate), 0.95).max(1e-6).log10()
}

/// Bass return after a breakdown in the excerpt (`clip_start_ms` = start on the track timeline).
///
/// `period_ms`/`downbeat_ms` (beat grid from the analysis, 0 = unknown) are used for snapping.
/// `body_db`: bass level of the track body ([`bass_level_db`]). Without it only the excerpt counts –
/// then a quieter intro that the bass kicks into can look like a drop.
pub fn detect_drop(
    x: &[f32],
    rate: u32,
    clip_start_ms: f64,
    period_ms: f64,
    downbeat_ms: f64,
    body_db: Option<f32>,
) -> Option<Drop> {
    let bass = bass_frames(x, rate);
    if bass.len() < 400 {
        return None;
    }
    let body = body_db.map_or(0.0, |db| 10f32.powf(db / 20.0));
    let full = percentile(&bass, 0.95).max(body);
    if full <= 1e-6 {
        return None;
    }
    // Before the drop ~2 s (at least 4 beats) of little bass, after it ~1 s (at least 2 beats) of full bass.
    let beat_frames = if period_ms > 0.0 { (period_ms / HOP_MS) as usize } else { 50 };
    let pre = (beat_frames * 4).max(200);
    let post = (beat_frames * 2).max(100);
    let mut best: Option<(usize, f32)> = None;
    for i in pre..bass.len().saturating_sub(post) {
        let before = mean(&bass[i - pre..i]);
        let after = mean(&bass[i..i + post]);
        if after < full * 0.55 || before > full * 0.3 {
            if best.is_some() {
                break; // first drop found and passed
            }
            continue;
        }
        let jump = (after - before) / full;
        if best.is_none_or(|(_, j)| jump > j) {
            best = Some((i, jump));
        }
    }
    let (i, jump) = best?;
    // Fine position: steepest rise of the bass level ±200 ms around the hit.
    let lo = i.saturating_sub(20).max(1);
    let hi = (i + 20).min(bass.len() - 1);
    let onset = (lo..=hi).max_by(|&a, &b| (bass[a] - bass[a - 1]).total_cmp(&(bass[b] - bass[b - 1]))).unwrap_or(i);
    let raw = clip_start_ms + onset as f64 * HOP_MS;
    Some(Drop { ms: snap(raw, period_ms, downbeat_ms), strength: jump.clamp(0.0, 1.0) })
}

/// Snaps to the next bar start (if close enough), otherwise to the next beat.
fn snap(t: f64, period_ms: f64, downbeat_ms: f64) -> f64 {
    if period_ms <= 0.0 {
        return t;
    }
    let bar = period_ms * 4.0;
    let on_bar = downbeat_ms + ((t - downbeat_ms) / bar).round() * bar;
    if (on_bar - t).abs() <= period_ms * 0.6 {
        return on_bar.max(0.0);
    }
    (downbeat_ms + ((t - downbeat_ms) / period_ms).round() * period_ms).max(0.0)
}

/// Rough drop positions from the envelope of the whole track (e.g. SoundCloud waveform), earliest first.
///
/// The waveform shows peak levels: in loudly mastered tracks it only drops by 10–25 % in the
/// breakdown. So suggest generously (up to 3 positions, at least 8 s apart) – whether the bass
/// really comes back there is decided by [`detect_drop`] on the excerpt.
pub fn drop_candidates(envelope: &[f32], duration_ms: f64) -> Vec<f64> {
    let b = blocks_500ms(envelope, duration_ms);
    if b.len() < 40 {
        return Vec::new();
    }
    let body = percentile(&b, 0.9);
    if body <= 0.0 {
        return Vec::new();
    }
    // 4 s before versus 4 s after; skip the intro start (4 s) and the last quarter.
    let w = 8;
    let end = b.len() * 3 / 4;
    let scores: Vec<(usize, f32)> = (w..end.saturating_sub(w))
        .filter_map(|i| {
            let before = mean(&b[i - w..i]);
            let after = mean(&b[i..i + w]);
            let jump = (after - before) / body;
            (after >= body * 0.9 && jump >= 0.06).then_some((i, jump))
        })
        .collect();
    let top = scores.iter().map(|s| s.1).fold(0.0f32, f32::max);
    // Per jump only the strongest position (local maximum), then the clear ones, earliest first.
    let mut picks: Vec<(usize, f32)> = Vec::new();
    for &(i, j) in &scores {
        if j < top * 0.5 {
            continue;
        }
        match picks.last_mut() {
            Some(last) if i - last.0 < 16 => {
                if j > last.1 {
                    *last = (i, j);
                }
            }
            _ => picks.push((i, j)),
        }
    }
    picks.into_iter().take(3).map(|(i, _)| i as f64 * 500.0).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 22_050;

    fn tone(hz: f32, secs: f32, amp: f32) -> Vec<f32> {
        (0..(secs * RATE as f32) as usize)
            .map(|n| amp * (std::f32::consts::TAU * hz * n as f32 / RATE as f32).sin())
            .collect()
    }

    #[test]
    fn cue_after_silence() {
        let mut x = vec![0.0; (1.5 * RATE as f32) as usize];
        x.extend(tone(440.0, 5.0, 0.4));
        let cue = detect_cue(&x, RATE, 0.0).unwrap();
        assert!((cue - 1500.0).abs() <= 20.0, "cue {cue}");
    }

    #[test]
    fn cue_ignores_click() {
        let mut x = vec![0.0; RATE as usize];
        x[RATE as usize / 2] = 0.9; // single click
        x.extend(tone(220.0, 4.0, 0.3));
        let cue = detect_cue(&x, RATE, 0.0).unwrap();
        assert!((cue - 1000.0).abs() <= 20.0, "cue {cue}");
    }

    #[test]
    fn cue_none_for_silence() {
        assert_eq!(detect_cue(&vec![0.0; RATE as usize * 3], RATE, 0.0), None);
    }

    #[test]
    fn drop_after_breakdown_snaps_to_bar() {
        // 120 BPM (500 ms/beat): 8 s of highs only (breakdown), then bass + highs.
        let mut x = tone(2000.0, 8.0, 0.2);
        let mut body = tone(55.0, 8.0, 0.6);
        for (v, h) in body.iter_mut().zip(tone(2000.0, 8.0, 0.2)) {
            *v += h;
        }
        x.extend(body);
        // Excerpt starts at 60 s into the track; downbeat grid from 0.
        let d = detect_drop(&x, RATE, 60_000.0, 500.0, 0.0, None).unwrap();
        assert!((d.ms - 68_000.0).abs() < 1.0, "drop {}", d.ms);
        assert!(d.strength > 0.5);
    }

    #[test]
    fn quiet_intro_is_no_drop() {
        // Intro with half bass (−6 dB) after silence: a jump within the excerpt, but not measured against the track body.
        let mut x = tone(2000.0, 4.0, 0.2);
        x.extend(tone(55.0, 6.0, 0.15));
        let body = bass_level_db(&tone(55.0, 4.0, 0.6), RATE);
        assert!(detect_drop(&x, RATE, 0.0, 500.0, 0.0, None).is_some());
        assert_eq!(detect_drop(&x, RATE, 0.0, 500.0, 0.0, Some(body)), None);
    }

    #[test]
    fn no_drop_without_breakdown() {
        let x = tone(55.0, 16.0, 0.6);
        assert_eq!(detect_drop(&x, RATE, 0.0, 500.0, 0.0, None), None);
    }

    #[test]
    fn candidate_from_envelope() {
        // 3 min: 30 s Intro mittel, 60–75 s Breakdown leise, ab 75 s voll.
        let mut env = Vec::new();
        for s in 0..180 {
            let v = match s {
                0..30 => 0.5,
                30..60 => 0.9,
                60..75 => 0.3,
                _ => 0.95,
            };
            env.extend([v; 10]);
        }
        let c = drop_candidates(&env, 180_000.0);
        assert!(c.iter().any(|t| (t - 75_000.0).abs() <= 4_000.0), "candidates {c:?}");
    }
}
