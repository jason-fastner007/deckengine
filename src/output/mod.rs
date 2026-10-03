// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Jason-Marshall Fastner <jasonfastner@protonmail.com>

//! Audio device output.
//!
//! | Platform | Default backend | Typical period |
//! |---|---|---|
//! | Linux   | ALSA via cpal (PipeWire/Pulse through their ALSA plugins); native PipeWire or JACK with the `pipewire` / `jack` features | what you ask for, e.g. 128 frames = 2.7 ms |
//! | Windows | own WASAPI `IAudioClient3` low-latency shared stream | the driver minimum, 2.7 ms on the standard Microsoft driver at 48 kHz |
//! | Windows | ASIO via cpal (`asio` feature, `host: Some("asio")`) | whatever the ASIO driver allows |
//! | macOS   | CoreAudio via cpal | what you ask for |
//!
//! cpal's own WASAPI host is only used as a fallback on Windows because it always runs at the
//! 10 ms shared-mode period.

mod cpal_backend;
#[cfg(windows)]
mod wasapi;

use std::ops::Deref;

use crate::Error;
use crate::engine::{self, EngineConfig, EngineHandle};

/// Output device settings.
#[derive(Debug, Clone)]
pub struct OutputConfig {
    /// Requested frames per callback. Default 128 (2.7 ms at 48 kHz). Backends round to what
    /// the hardware supports.
    pub buffer_frames: u32,
    /// Requested sample rate. `None` = device default.
    pub sample_rate: Option<u32>,
    /// Device id as printed by [`list_devices`] (e.g. `hw:CARD=PCH,DEV=0`) or a
    /// case-insensitive substring of the device name. `None` = system default device.
    pub device: Option<String>,
    /// Audio host: `"alsa"`, `"pipewire"`, `"jack"`, `"pulseaudio"`, `"wasapi"`, `"asio"`,
    /// `"coreaudio"`. `None` = best low-latency choice for the platform.
    pub host: Option<String>,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self { buffer_frames: 128, sample_rate: None, device: None, host: None }
    }
}

/// What was actually opened.
#[derive(Debug, Clone, PartialEq)]
pub struct OutputInfo {
    pub backend: &'static str,
    pub host: String,
    pub device: String,
    pub sample_rate: u32,
    pub channels: u16,
    /// Period/buffer size granted by the backend, in frames (0 if unknown until running).
    pub buffer_frames: u32,
}

pub(crate) enum StreamGuard {
    Cpal(#[allow(dead_code)] cpal::Stream),
    #[cfg(windows)]
    Wasapi(#[allow(dead_code)] wasapi::WasapiStream),
}

/// A running engine attached to an audio device. Dereferences to [`EngineHandle`].
/// Dropping it stops the audio.
pub struct Engine {
    handle: EngineHandle,
    info: OutputInfo,
    _stream: StreamGuard,
}

impl Engine {
    /// Opens the output device and starts the engine.
    pub fn start(output: OutputConfig, config: EngineConfig) -> Result<Engine, Error> {
        #[cfg(windows)]
        {
            let host = output.host.as_deref().map(str::to_ascii_lowercase);
            if host.is_none() || host.as_deref() == Some("wasapi-lowlatency") {
                match wasapi::start(&output, &config) {
                    Ok((handle, info, stream)) => {
                        return Ok(Engine { handle, info, _stream: StreamGuard::Wasapi(stream) });
                    }
                    Err(e) if host.is_some() => return Err(e),
                    // e.g. Windows 8.1 or older (no IAudioClient3): fall back to cpal.
                    Err(_) => {}
                }
            }
        }
        let (handle, info, stream) = cpal_backend::start(&output, &config)?;
        Ok(Engine { handle, info, _stream: StreamGuard::Cpal(stream) })
    }

    /// A cloneable handle that stays valid as long as this `Engine` lives.
    pub fn handle(&self) -> EngineHandle {
        self.handle.clone()
    }

    pub fn output_info(&self) -> &OutputInfo {
        &self.info
    }
}

impl Deref for Engine {
    type Target = EngineHandle;
    fn deref(&self) -> &EngineHandle {
        &self.handle
    }
}

/// Lists `(host, device id, device name)` for every output device on every available host.
pub fn list_devices() -> Vec<(String, String, String)> {
    cpal_backend::list_devices()
}

pub(crate) fn create_engine(sample_rate: u32, config: &EngineConfig) -> (EngineHandle, engine::Processor) {
    engine::create(sample_rate, config)
}
