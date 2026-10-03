// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Jason-Marshall Fastner <jasonfastner@protonmail.com>

//! Windows low-latency output: WASAPI shared mode through `IAudioClient3`.
//!
//! Classic shared-mode WASAPI (what most libraries use) always runs at the audio engine's
//! default 10 ms period. Since Windows 10, `IAudioClient3::InitializeSharedAudioStream` lets a
//! client request a smaller period: the standard Microsoft HD Audio driver allows 128 frames
//! (2.67 ms) at 48 kHz, many USB drivers allow similar. The stream stays in shared mode, so
//! other applications keep playing.
//!
//! The render thread is event driven and registered with MMCSS as "Pro Audio".

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::sync_channel;
use std::thread::JoinHandle;
use std::time::Duration;

use windows::Win32::Devices::FunctionDiscovery::PKEY_Device_FriendlyName;
use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0};
use windows::Win32::Media::Audio::{
    AUDCLNT_BUFFERFLAGS_SILENT, AUDCLNT_STREAMFLAGS_EVENTCALLBACK, DEVICE_STATE_ACTIVE, IAudioClient3,
    IAudioRenderClient, IMMDevice, IMMDeviceEnumerator, MMDeviceEnumerator, WAVEFORMATEX,
    WAVEFORMATEXTENSIBLE, eConsole, eRender,
};
use windows::Win32::Media::Multimedia::KSDATAFORMAT_SUBTYPE_IEEE_FLOAT;
use windows::Win32::System::Com::StructuredStorage::PropVariantToStringAlloc;
use windows::Win32::System::Com::{
    CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx, CoTaskMemFree, CoUninitialize,
    STGM_READ,
};
use windows::Win32::System::Threading::{AvSetMmThreadCharacteristicsW, CreateEventW, WaitForSingleObject};
use windows::core::{PCWSTR, w};

use super::{OutputConfig, OutputInfo, create_engine};
use crate::Error;
use crate::engine::{EngineConfig, EngineHandle, Processor};

const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;

pub(crate) struct WasapiStream {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Drop for WasapiStream {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn werr(e: windows::core::Error) -> String {
    e.to_string()
}

#[derive(Clone, Copy, PartialEq)]
enum SampleKind {
    F32,
    I16,
    I24Packed,
    I32,
}

struct Opened {
    sample_rate: u32,
    channels: u16,
    period: u32,
    device: String,
}

pub(super) fn start(
    output: &OutputConfig,
    config: &EngineConfig,
) -> Result<(EngineHandle, OutputInfo, WasapiStream), Error> {
    let (info_tx, info_rx) = sync_channel::<Result<Opened, String>>(1);
    let (proc_tx, proc_rx) = sync_channel::<Processor>(1);
    let (started_tx, started_rx) = sync_channel::<Result<(), String>>(1);
    let stop = Arc::new(AtomicBool::new(false));

    let out = output.clone();
    let thread_stop = stop.clone();
    let thread = std::thread::Builder::new().name("deckengine-wasapi".into()).spawn(move || {
        // SAFETY: all COM objects are created and used on this thread only.
        unsafe { render_thread(out, info_tx, proc_rx, started_tx, thread_stop) }
    })?;
    let mut stream = WasapiStream { stop, thread: Some(thread) };

    let opened =
        info_rx.recv().map_err(|_| Error::Device("WASAPI thread exited".into()))?.map_err(Error::Device)?;
    let (handle, processor) = create_engine(opened.sample_rate, config);
    proc_tx.send(processor).map_err(|_| Error::Device("WASAPI thread exited".into()))?;
    match started_rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(Error::Device(e)),
        Err(_) => {
            stream.thread.take();
            return Err(Error::Device("WASAPI thread exited".into()));
        }
    }

    let info = OutputInfo {
        backend: "wasapi-iaudioclient3",
        host: "WASAPI".into(),
        device: opened.device,
        sample_rate: opened.sample_rate,
        channels: opened.channels,
        buffer_frames: opened.period,
    };
    Ok((handle, info, stream))
}

unsafe fn friendly_name(dev: &IMMDevice) -> String {
    let get = || -> windows::core::Result<String> {
        let store = unsafe { dev.OpenPropertyStore(STGM_READ) }?;
        let value = unsafe { store.GetValue(&PKEY_Device_FriendlyName) }?;
        let s = unsafe { PropVariantToStringAlloc(&value) }?;
        let name = unsafe { s.to_string() }.unwrap_or_default();
        unsafe { CoTaskMemFree(Some(s.0 as *const _)) };
        Ok(name)
    };
    get().unwrap_or_else(|_| "unknown".into())
}

unsafe fn pick_device(enumerator: &IMMDeviceEnumerator, want: Option<&str>) -> Result<IMMDevice, String> {
    unsafe {
        let Some(want) = want else {
            return enumerator.GetDefaultAudioEndpoint(eRender, eConsole).map_err(werr);
        };
        let want = want.to_lowercase();
        let all = enumerator.EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE).map_err(werr)?;
        for i in 0..all.GetCount().map_err(werr)? {
            let dev = all.Item(i).map_err(werr)?;
            if friendly_name(&dev).to_lowercase().contains(&want) {
                return Ok(dev);
            }
        }
        Err(format!("no output device matching '{want}'"))
    }
}

unsafe fn sample_kind(fmt: *const WAVEFORMATEX) -> Option<SampleKind> {
    let f = unsafe { &*fmt };
    let tag = f.wFormatTag;
    let bits = f.wBitsPerSample;
    let is_float = tag == WAVE_FORMAT_IEEE_FLOAT
        || (tag == WAVE_FORMAT_EXTENSIBLE && {
            let ext = unsafe { &*(fmt as *const WAVEFORMATEXTENSIBLE) };
            let sub = ext.SubFormat;
            sub == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT
        });
    match (is_float, bits) {
        (true, 32) => Some(SampleKind::F32),
        (false, 16) => Some(SampleKind::I16),
        (false, 24) => Some(SampleKind::I24Packed),
        (false, 32) => Some(SampleKind::I32),
        _ => None,
    }
}

/// Picks the smallest engine period ≥ `want` that the driver allows.
fn choose_period(want: u32, default: u32, fundamental: u32, min: u32, max: u32) -> u32 {
    if fundamental == 0 || min == 0 {
        return default;
    }
    let want = want.clamp(min, max.max(min));
    let steps = (want - min).div_ceil(fundamental);
    (min + steps * fundamental).min(max.max(min))
}

unsafe fn render_thread(
    output: OutputConfig,
    info_tx: std::sync::mpsc::SyncSender<Result<Opened, String>>,
    proc_rx: std::sync::mpsc::Receiver<Processor>,
    started_tx: std::sync::mpsc::SyncSender<Result<(), String>>,
    stop: Arc<AtomicBool>,
) {
    unsafe {
        if let Err(e) = CoInitializeEx(None, COINIT_MULTITHREADED).ok() {
            let _ = info_tx.send(Err(werr(e)));
            return;
        }
        let setup = || -> Result<_, String> {
            let enumerator: IMMDeviceEnumerator =
                CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL).map_err(werr)?;
            let device = pick_device(&enumerator, output.device.as_deref())?;
            let name = friendly_name(&device);
            let client: IAudioClient3 = device.Activate(CLSCTX_ALL, None).map_err(werr)?;
            let fmt = client.GetMixFormat().map_err(werr)?;
            let Some(kind) = sample_kind(fmt) else {
                CoTaskMemFree(Some(fmt as *const _));
                return Err("unsupported WASAPI mix format".into());
            };
            let (mut def, mut fund, mut min, mut max) = (0u32, 0u32, 0u32, 0u32);
            client.GetSharedModeEnginePeriod(fmt, &mut def, &mut fund, &mut min, &mut max).map_err(werr)?;
            let period = choose_period(output.buffer_frames, def, fund, min, max);
            client
                .InitializeSharedAudioStream(AUDCLNT_STREAMFLAGS_EVENTCALLBACK, period, fmt, None)
                .map_err(werr)?;
            let event = CreateEventW(None, false, false, PCWSTR::null()).map_err(werr)?;
            client.SetEventHandle(event).map_err(werr)?;
            let buffer_frames = client.GetBufferSize().map_err(werr)?;
            let render: IAudioRenderClient = client.GetService().map_err(werr)?;
            let stream_latency_100ns = client.GetStreamLatency().unwrap_or(0);
            Ok((client, render, fmt, kind, period, event, buffer_frames, stream_latency_100ns, name))
        };

        let (client, render, fmt, kind, period, event, buffer_frames, stream_latency, name) = match setup() {
            Ok(v) => v,
            Err(e) => {
                let _ = info_tx.send(Err(e));
                CoUninitialize();
                return;
            }
        };
        let sample_rate = (*fmt).nSamplesPerSec;
        let channels = (*fmt).nChannels as usize;
        let _ = info_tx.send(Ok(Opened { sample_rate, channels: channels as u16, period, device: name }));

        let Ok(mut processor) = proc_rx.recv() else {
            cleanup(&client, fmt, event);
            return;
        };

        // Pre-roll silence so the first period does not underrun.
        if let Ok(p) = render.GetBuffer(buffer_frames) {
            let _ = p;
            let _ = render.ReleaseBuffer(buffer_frames, AUDCLNT_BUFFERFLAGS_SILENT.0 as u32);
        }

        let mut task_index = 0u32;
        let mmcss = AvSetMmThreadCharacteristicsW(w!("Pro Audio"), &mut task_index);

        if let Err(e) = client.Start() {
            let _ = started_tx.send(Err(werr(e)));
            cleanup(&client, fmt, event);
            return;
        }
        let _ = started_tx.send(Ok(()));

        let mut scratch = vec![0.0f32; buffer_frames as usize * channels];
        let mut consecutive_errors = 0u32;
        let stream_latency_s = stream_latency as f64 * 1e-7;

        while !stop.load(Ordering::Relaxed) {
            if WaitForSingleObject(event, 200) != WAIT_OBJECT_0 {
                continue;
            }
            let result = (|| -> windows::core::Result<()> {
                let padding = client.GetCurrentPadding()?;
                let avail = buffer_frames.saturating_sub(padding);
                if avail == 0 {
                    return Ok(());
                }
                let ptr = render.GetBuffer(avail)?;
                let n = avail as usize * channels;
                match kind {
                    SampleKind::F32 => {
                        let out = std::slice::from_raw_parts_mut(ptr as *mut f32, n);
                        processor.process(out, channels);
                    }
                    SampleKind::I16 => {
                        let tmp = &mut scratch[..n];
                        processor.process(tmp, channels);
                        let out = std::slice::from_raw_parts_mut(ptr as *mut i16, n);
                        for (d, &s) in out.iter_mut().zip(tmp.iter()) {
                            *d = (s * 32767.0) as i16;
                        }
                    }
                    SampleKind::I32 => {
                        let tmp = &mut scratch[..n];
                        processor.process(tmp, channels);
                        let out = std::slice::from_raw_parts_mut(ptr as *mut i32, n);
                        for (d, &s) in out.iter_mut().zip(tmp.iter()) {
                            *d = (s as f64 * 2_147_483_647.0) as i32;
                        }
                    }
                    SampleKind::I24Packed => {
                        let tmp = &mut scratch[..n];
                        processor.process(tmp, channels);
                        let out = std::slice::from_raw_parts_mut(ptr, n * 3);
                        for (d, &s) in out.chunks_exact_mut(3).zip(tmp.iter()) {
                            let v = (s * 8_388_607.0) as i32;
                            d.copy_from_slice(&v.to_le_bytes()[..3]);
                        }
                    }
                }
                render.ReleaseBuffer(avail, 0)?;
                // Audio just written plays after everything queued before it.
                let queued = (padding + avail) as f64 / sample_rate as f64;
                processor.report_device_latency(Duration::from_secs_f64(queued + stream_latency_s));
                Ok(())
            })();
            match result {
                Ok(()) => consecutive_errors = 0,
                Err(_) => {
                    processor.report_backend_error();
                    consecutive_errors += 1;
                    if consecutive_errors > 50 {
                        // Device gone (unplugged / invalidated). Stop cleanly.
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
        }

        if let Ok(h) = mmcss {
            let _ = windows::Win32::System::Threading::AvRevertMmThreadCharacteristics(h);
        }
        cleanup(&client, fmt, event);
    }
}

unsafe fn cleanup(client: &IAudioClient3, fmt: *mut WAVEFORMATEX, event: HANDLE) {
    unsafe {
        let _ = client.Stop();
        let _ = CloseHandle(event);
        CoTaskMemFree(Some(fmt as *const _));
        CoUninitialize();
    }
}

#[cfg(test)]
mod tests {
    use super::choose_period;

    #[test]
    fn period_rounding() {
        // Typical HD Audio: default 480, fundamental 32, min 128, max 480 at 48 kHz.
        assert_eq!(choose_period(128, 480, 32, 128, 480), 128);
        assert_eq!(choose_period(100, 480, 32, 128, 480), 128);
        assert_eq!(choose_period(130, 480, 32, 128, 480), 160);
        assert_eq!(choose_period(4096, 480, 32, 128, 480), 480);
        // Driver without low-latency support: min == max == default.
        assert_eq!(choose_period(128, 480, 480, 480, 480), 480);
    }
}
