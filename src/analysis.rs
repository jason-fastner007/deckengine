// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Jason-Marshall Fastner <jasonfastner@protonmail.com>

//! Offline track analysis (runs on the caller's thread, not in real time).

use crate::Error;
use crate::dsp::loudness::{LoudnessInfo, LoudnessMeter};
use crate::source::FrameSource;

/// Measures integrated loudness, true peak and loudness range of a whole source.
///
/// Typical use: store the result per track and call
/// `handle.set_trim_db(deck, info.gain_to_target(-14.0, -1.0))` when loading it, so every
/// track plays at the same loudness without any real-time gain riding.
pub fn analyze_source(source: &mut dyn FrameSource) -> Result<LoudnessInfo, Error> {
    let mut meter = LoudnessMeter::new(source.sample_rate()).map_err(|e| Error::Decode(e.to_string()))?;
    let mut buf = vec![0.0f32; 8192 * 2];
    loop {
        let n = source.read(&mut buf)?;
        if n == 0 {
            break;
        }
        meter.add(&buf[..n * 2]);
    }
    Ok(meter.finish())
}

/// Decodes a file completely and measures its loudness.
#[cfg(feature = "decode")]
pub fn analyze_file(path: impl AsRef<std::path::Path>) -> Result<LoudnessInfo, Error> {
    let mut src = crate::source::DecodedSource::open(path)?;
    analyze_source(&mut src)
}
