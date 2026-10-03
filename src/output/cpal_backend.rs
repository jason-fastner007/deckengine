// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Jason-Marshall Fastner <jasonfastner@protonmail.com>

//! Output through cpal (ALSA, PipeWire, JACK, PulseAudio, CoreAudio, ASIO, WASAPI fallback).

use std::sync::atomic::Ordering::Relaxed;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, FromSample, SampleFormat, SizedSample, StreamConfig, SupportedBufferSize};

use super::{OutputConfig, OutputInfo, create_engine};
use crate::Error;
use crate::engine::{EngineConfig, EngineHandle, Processor};

fn dev_err(e: impl std::fmt::Display) -> Error {
    Error::Device(e.to_string())
}

fn pick_host(name: Option<&str>) -> Result<cpal::Host, Error> {
    let Some(name) = name else {
        return Ok(cpal::default_host());
    };
    let id = cpal::available_hosts().into_iter().find(|h| h.name().eq_ignore_ascii_case(name)).ok_or_else(
        || {
            let avail: Vec<_> = cpal::available_hosts().iter().map(|h| h.name()).collect();
            Error::Device(format!("audio host '{name}' not available (compiled in: {avail:?})"))
        },
    )?;
    cpal::host_from_id(id).map_err(dev_err)
}

fn device_name(d: &cpal::Device) -> String {
    d.description().map(|x| x.name().to_string()).unwrap_or_else(|_| "unknown".into())
}

fn device_id(d: &cpal::Device) -> String {
    d.id().map(|id| id.to_string()).unwrap_or_default()
}

pub(super) fn list_devices() -> Vec<(String, String, String)> {
    let mut out = Vec::new();
    for id in cpal::available_hosts() {
        let Ok(host) = cpal::host_from_id(id) else { continue };
        if let Ok(devs) = host.output_devices() {
            for d in devs {
                out.push((id.name().to_string(), device_id(&d), device_name(&d)));
            }
        }
    }
    out
}

pub(super) fn start(
    output: &OutputConfig,
    config: &EngineConfig,
) -> Result<(EngineHandle, OutputInfo, cpal::Stream), Error> {
    let host = pick_host(output.host.as_deref())?;
    let device = match &output.device {
        Some(want) => {
            // Exact device id first (e.g. "alsa:hw:CARD=PCH,DEV=0"), then a name substring.
            let devices: Vec<_> = host.output_devices().map_err(dev_err)?.collect();
            let lower = want.to_lowercase();
            devices
                .iter()
                .find(|d| device_id(d) == *want || device_id(d).ends_with(&format!(":{want}")))
                .or_else(|| devices.iter().find(|d| device_name(d).to_lowercase().contains(&lower)))
                .cloned()
                .ok_or_else(|| Error::Device(format!("no output device matching '{want}'")))?
        }
        None => {
            host.default_output_device().ok_or_else(|| Error::Device("no default output device".into()))?
        }
    };

    let default = device.default_output_config().map_err(dev_err)?;
    let supported = match output.sample_rate {
        Some(rate) if rate != default.sample_rate() => device
            .supported_output_configs()
            .map_err(dev_err)?
            .filter(|c| c.min_sample_rate() <= rate && rate <= c.max_sample_rate())
            .max_by_key(|c| (c.sample_format() == SampleFormat::F32, c.channels() == 2))
            .map(|c| c.with_sample_rate(rate))
            .ok_or_else(|| Error::Device(format!("device does not support {rate} Hz")))?,
        _ => default,
    };

    let sample_rate = supported.sample_rate();
    let channels = supported.channels();
    let format = supported.sample_format();
    let frames = match supported.buffer_size() {
        SupportedBufferSize::Range { min, max } => output.buffer_frames.clamp(*min, *max),
        SupportedBufferSize::Unknown => output.buffer_frames,
    };

    let (handle, processor) = create_engine(sample_rate, config);
    let info = OutputInfo {
        backend: "cpal",
        host: host.id().name().to_string(),
        device: device_name(&device),
        sample_rate,
        channels,
        buffer_frames: frames,
    };

    let fixed = StreamConfig { channels, sample_rate, buffer_size: BufferSize::Fixed(frames) };
    let stream = match build(&device, fixed, format, processor, &handle) {
        Ok(s) => s,
        Err(failed) => {
            let (e, processor) = *failed;
            // Some drivers reject explicit sizes; fall back to the device default.
            let dflt = StreamConfig { channels, sample_rate, buffer_size: BufferSize::Default };
            build(&device, dflt, format, processor, &handle)
                .map_err(|f| Error::Device(format!("{e}; with default buffer: {}", f.0)))?
        }
    };
    stream.play().map_err(dev_err)?;
    Ok((handle, info, stream))
}

type BuildResult = Result<cpal::Stream, Box<(Error, Processor)>>;

fn build(
    device: &cpal::Device,
    config: StreamConfig,
    format: SampleFormat,
    processor: Processor,
    handle: &EngineHandle,
) -> BuildResult {
    match format {
        SampleFormat::F32 => build_typed::<f32>(device, config, processor, handle),
        SampleFormat::I16 => build_typed::<i16>(device, config, processor, handle),
        SampleFormat::I32 => build_typed::<i32>(device, config, processor, handle),
        SampleFormat::U16 => build_typed::<u16>(device, config, processor, handle),
        SampleFormat::F64 => build_typed::<f64>(device, config, processor, handle),
        other => Err(Box::new((Error::Device(format!("unsupported sample format {other}")), processor))),
    }
}

fn build_typed<T>(
    device: &cpal::Device,
    config: StreamConfig,
    processor: Processor,
    handle: &EngineHandle,
) -> BuildResult
where
    T: SizedSample + FromSample<f32> + 'static,
{
    let channels = config.channels as usize;
    let err_metrics = handle.metrics();

    // cpal consumes the callback even on failure, so hand the processor over through a slot we
    // can take back if building fails.
    let slot = std::sync::Arc::new(std::sync::Mutex::new(Some(processor)));
    let cb_slot = slot.clone();
    let mut proc_: Option<Processor> = None;
    // Scratch for non-f32 devices, sized generously so the callback never allocates.
    let mut scratch = vec![0.0f32; 16_384 * channels.max(1)];

    let result = device.build_output_stream::<T, _, _>(
        config,
        move |data: &mut [T], info: &cpal::OutputCallbackInfo| {
            let p = match proc_.as_mut() {
                Some(p) => p,
                None => {
                    // First callback: take ownership once. try_lock never blocks.
                    let Ok(mut g) = cb_slot.try_lock() else { return };
                    let Some(p) = g.take() else { return };
                    proc_.insert(p)
                }
            };
            let ts = info.timestamp();
            p.report_device_latency(ts.playback.duration_since(ts.callback));

            if let Some(out) = as_f32_mut(data) {
                p.process(out, channels);
            } else {
                for chunk in data.chunks_mut(scratch.len()) {
                    let tmp = &mut scratch[..chunk.len()];
                    p.process(tmp, channels);
                    for (d, &s) in chunk.iter_mut().zip(tmp.iter()) {
                        *d = T::from_sample(s);
                    }
                }
            }
        },
        move |_err| {
            err_metrics.backend_errors.fetch_add(1, Relaxed);
        },
        None,
    );
    match result {
        Ok(s) => Ok(s),
        Err(e) => {
            let p = slot.lock().unwrap_or_else(|e| e.into_inner()).take();
            match p {
                Some(p) => Err(Box::new((dev_err(e), p))),
                // Cannot happen: the callback never ran.
                None => unreachable!("processor consumed by a stream that failed to build"),
            }
        }
    }
}

/// Reinterprets a sample slice as `f32` when `T` is `f32`.
fn as_f32_mut<T: 'static>(data: &mut [T]) -> Option<&mut [f32]> {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        // SAFETY: T is f32, checked above.
        Some(unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut f32, data.len()) })
    } else {
        None
    }
}
