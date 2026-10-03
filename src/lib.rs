// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Jason-Marshall Fastner <jasonfastner@protonmail.com>

//! # deckengine
//!
//! A low-latency, real-time-safe two-deck audio engine for Linux and Windows: file playback,
//! gapless seeks, sample-accurate loops, DJ crossfades, pitch/tempo (varispeed), three-band
//! stem isolation, EBU R128 loudness normalisation and a brick-wall limiter.
//!
//! Output latency target: **< 5 ms** (128-frame buffers at 48 kHz = 2.7 ms plus device
//! latency). See the README for how each platform gets there.
//!
//! ```no_run
//! use deckengine::{Engine, DeckId, OutputConfig, EngineConfig};
//!
//! let engine = Engine::start(OutputConfig::default(), EngineConfig::default())?;
//! engine.load_file(DeckId::A, "song.flac")?;
//! engine.play(DeckId::A)?;
//! println!("{:?}", engine.stats());
//! # Ok::<(), deckengine::Error>(())
//! ```
//!
//! Without an audio device (tests, rendering to a file, custom backends) use
//! [`engine::create`] and drive [`Processor::process`] yourself.
//!
//! Other languages use the C ABI in [`ffi`] (header: `include/deckengine.h`).

pub mod analysis;
pub mod cue;
pub mod dj;
pub mod dsp;
pub mod echolot;
pub mod energy;
pub mod fakeprint;
#[cfg(feature = "clip")]
pub mod mp3clip;
pub mod engine;
pub mod ffi;
pub mod source;

mod deck;
mod metrics;
mod stream;

#[cfg(feature = "output")]
pub mod output;

#[cfg(feature = "decode")]
pub use analysis::analyze_file;
pub use analysis::analyze_source;
pub use dsp::isolator::StemLevels;
pub use dsp::loudness::LoudnessInfo;
pub use engine::{
    CrossfadeCurve, DeckId, DeckStatus, EngineConfig, EngineHandle, EngineStats, Processor, When,
};
#[cfg(feature = "output")]
pub use output::{Engine, OutputConfig, OutputInfo};
#[cfg(feature = "decode")]
pub use source::DecodedSource;
pub use source::{FrameSource, PcmSource, SourceInfo};

/// Errors returned by the engine.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("decode error: {0}")]
    Decode(String),
    #[error("audio device error: {0}")]
    Device(String),
    #[error("no track loaded on this deck")]
    NoTrack,
    #[error("command queue full (the audio thread is not running?)")]
    QueueFull,
    #[error("invalid argument: {0}")]
    InvalidArgument(&'static str),
}
