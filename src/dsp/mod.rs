// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Jason-Marshall Fastner <jasonfastner@protonmail.com>

//! Real-time DSP building blocks.
//!
//! Everything in here is allocation-free after construction and safe to call from the audio
//! callback.

pub mod biquad;
pub mod isolator;
pub mod limiter;
pub mod loudness;
pub mod resample;
pub mod smooth;

/// Converts decibels to a linear gain factor.
#[inline]
pub fn db_to_gain(db: f32) -> f32 {
    10.0f32.powf(db / 20.0)
}

/// Converts a linear gain factor to decibels. Returns `-120` for silence.
#[inline]
pub fn gain_to_db(gain: f32) -> f32 {
    if gain <= 1e-6 { -120.0 } else { 20.0 * gain.log10() }
}

/// Flushes denormals to zero. IIR filter states decay into the denormal range on silence,
/// which is dramatically slow on x86 without FTZ/DAZ.
#[inline(always)]
pub(crate) fn undenormal(x: f32) -> f32 {
    if x.abs() < 1e-20 { 0.0 } else { x }
}
