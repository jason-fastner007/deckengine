// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Jason-Marshall Fastner <jasonfastner@protonmail.com>

//! Audio sources that feed a deck.
//!
//! A [`FrameSource`] produces interleaved **stereo** `f32` frames at its own sample rate. It is
//! driven from a background streaming thread, never from the audio callback, so it may block,
//! allocate and do I/O. The deck resamples to the device rate.
//!
//! Built-in sources:
//! * [`PcmSource`] – audio already in memory.
//! * [`DecodedSource`] (feature `decode`) – any file/stream symphonia can decode
//!   (MP3, AAC/M4A, FLAC, Ogg Vorbis, WAV, AIFF, ALAC, …).
//!
//! Apps can implement [`FrameSource`] themselves (network streams, other decoders, synths).

use crate::Error;

/// A seekable producer of interleaved stereo `f32` frames.
pub trait FrameSource: Send + 'static {
    /// Native sample rate of the frames this source produces.
    fn sample_rate(&self) -> u32;

    /// Total length in frames, if known.
    fn duration_frames(&self) -> Option<u64>;

    /// Fills `out` (interleaved stereo, so `out.len() / 2` frames) and returns the number of
    /// frames written. `Ok(0)` means end of stream.
    fn read(&mut self, out: &mut [f32]) -> Result<usize, Error>;

    /// Positions the source so that the next `read` starts exactly at `frame`.
    fn seek(&mut self, frame: u64) -> Result<(), Error>;
}

/// Static information about a loaded source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceInfo {
    pub sample_rate: u32,
    pub duration_frames: Option<u64>,
}

impl SourceInfo {
    pub fn duration_seconds(&self) -> Option<f64> {
        self.duration_frames.map(|f| f as f64 / self.sample_rate as f64)
    }
}

/// Audio held in memory (interleaved stereo).
#[derive(Debug, Clone)]
pub struct PcmSource {
    samples: std::sync::Arc<[f32]>,
    sample_rate: u32,
    pos: usize,
}

impl PcmSource {
    /// `interleaved_stereo.len()` must be even.
    pub fn new(interleaved_stereo: impl Into<std::sync::Arc<[f32]>>, sample_rate: u32) -> Self {
        let samples = interleaved_stereo.into();
        assert!(samples.len() % 2 == 0, "PcmSource expects interleaved stereo");
        Self { samples, sample_rate, pos: 0 }
    }

    /// Builds a stereo source from mono samples.
    pub fn from_mono(mono: &[f32], sample_rate: u32) -> Self {
        let stereo: Vec<f32> = mono.iter().flat_map(|&x| [x, x]).collect();
        Self::new(stereo, sample_rate)
    }
}

impl FrameSource for PcmSource {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn duration_frames(&self) -> Option<u64> {
        Some((self.samples.len() / 2) as u64)
    }

    fn read(&mut self, out: &mut [f32]) -> Result<usize, Error> {
        let n = (out.len() / 2).min(self.samples.len() / 2 - self.pos);
        out[..n * 2].copy_from_slice(&self.samples[self.pos * 2..(self.pos + n) * 2]);
        self.pos += n;
        Ok(n)
    }

    fn seek(&mut self, frame: u64) -> Result<(), Error> {
        self.pos = (frame as usize).min(self.samples.len() / 2);
        Ok(())
    }
}

/// Downmixes/upmixes interleaved audio with `channels` channels to stereo, appending to `out`.
#[cfg(any(test, feature = "decode"))]
pub(crate) fn to_stereo(src: &[f32], channels: usize, out: &mut Vec<f32>) {
    match channels {
        0 => {}
        1 => out.extend(src.iter().flat_map(|&x| [x, x])),
        2 => out.extend_from_slice(src),
        n => {
            // FL, FR, FC, LFE, ... (SMPTE / WAVE order). Fold the centre in at −3 dB.
            for frame in src.chunks_exact(n) {
                let c = frame[2] * std::f32::consts::FRAC_1_SQRT_2;
                out.push(frame[0] + c);
                out.push(frame[1] + c);
            }
        }
    }
}

#[cfg(feature = "decode")]
pub use decoded::DecodedSource;

#[cfg(feature = "decode")]
mod decoded {
    use super::{FrameSource, to_stereo};
    use crate::Error;
    use std::path::Path;
    use symphonia::core::codecs::audio::{AudioDecoder, AudioDecoderOptions};
    use symphonia::core::errors::Error as SymError;
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::formats::{FormatOptions, FormatReader, SeekMode, SeekTo, TrackType};
    use symphonia::core::io::{MediaSource, MediaSourceStream};
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::units::{Time, TimeBase, Timestamp};

    /// Frames decoded (and discarded) before a seek target, see `seek`.
    const PREROLL_FRAMES: u64 = 4096;

    /// Decodes a file or byte stream with symphonia.
    pub struct DecodedSource {
        /// Always `Some` except transiently inside `reopen`.
        format: Option<Box<dyn FormatReader>>,
        hint: Hint,
        /// The reader hit the end of the stream. Some readers (MP4) cannot seek afterwards,
        /// so the next seek re-opens the stream first.
        hit_end: bool,
        decoder: Box<dyn AudioDecoder>,
        track_id: u32,
        time_base: Option<TimeBase>,
        sample_rate: u32,
        duration_frames: Option<u64>,
        /// Decoded stereo frames not yet handed out.
        pending: Vec<f32>,
        pending_pos: usize,
        /// Scratch buffer for the decoder's native channel layout.
        scratch: Vec<f32>,
        /// After a seek: the exact frame the next read must start at. Packets are positioned
        /// by their own timestamps, which is exact even for codecs whose first packet after a
        /// decoder reset is shorter (Vorbis, AAC).
        seek_target: Option<u64>,
        /// End frame of the last decoded packet (for end-of-stream detection).
        decoded_until: u64,
    }

    fn sym(e: SymError) -> Error {
        Error::Decode(e.to_string())
    }

    impl DecodedSource {
        /// Opens a file. The extension is used as a format hint.
        pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
            let path = path.as_ref();
            let file = std::fs::File::open(path)?;
            let mut hint = Hint::new();
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                hint.with_extension(ext);
            }
            Self::from_media_source(Box::new(file), hint)
        }

        /// Decodes from any seekable byte source (in-memory buffer, HTTP range reader, ...).
        pub fn from_media_source(source: Box<dyn MediaSource>, hint: Hint) -> Result<Self, Error> {
            let mss = MediaSourceStream::new(source, Default::default());
            let format = symphonia::default::get_probe()
                .probe(&hint, mss, FormatOptions::default(), MetadataOptions::default())
                .map_err(sym)?;

            let track = format
                .default_track(TrackType::Audio)
                .ok_or_else(|| Error::Decode("no audio track".into()))?;
            let track_id = track.id;
            let time_base = track.time_base;
            let params = track
                .codec_params
                .as_ref()
                .and_then(|p| p.audio())
                .ok_or_else(|| Error::Decode("track has no audio codec parameters".into()))?;
            let sample_rate =
                params.sample_rate.ok_or_else(|| Error::Decode("unknown sample rate".into()))?;
            let duration_frames = track.num_frames.or_else(|| {
                let tb = time_base?;
                let d = track.duration?;
                let t = tb.calc_time(Timestamp::new(d.get() as i64))?;
                Some((t.as_secs_f64() * sample_rate as f64).round() as u64)
            });

            let decoder = symphonia::default::get_codecs()
                // Gapless trimming is applied uniformly in `decode_next` (only some symphonia
                // decoders trim on their own), so switch the decoder-side trimming off.
                .make_audio_decoder(params, &AudioDecoderOptions::default().gapless(false))
                .map_err(sym)?;

            Ok(Self {
                format: Some(format),
                hint,
                hit_end: false,
                decoder,
                track_id,
                time_base,
                sample_rate,
                duration_frames,
                pending: Vec::with_capacity(16_384),
                pending_pos: 0,
                scratch: Vec::with_capacity(16_384),
                seek_target: None,
                decoded_until: 0,
            })
        }

        /// Decodes the next packet into `pending`. Returns `false` at end of stream.
        fn decode_next(&mut self) -> Result<bool, Error> {
            loop {
                let at_end = self.at_end();
                let format = self.format.as_mut().ok_or_else(|| Error::Decode("reader lost".into()))?;
                let packet = match format.next_packet() {
                    Ok(Some(p)) => p,
                    Ok(None) => {
                        self.hit_end = true;
                        return Ok(false);
                    }
                    Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                        self.hit_end = true;
                        return Ok(false);
                    }
                    // Some readers (e.g. MP4 with the index at the end) report a malformed
                    // stream instead of a clean end. Accept that only at the actual end.
                    Err(SymError::DecodeError(_)) if at_end => {
                        self.hit_end = true;
                        return Ok(false);
                    }
                    Err(e) => return Err(sym(e)),
                };
                if packet.track_id != self.track_id {
                    continue;
                }
                // Gapless: encoder delay / padding the container asks us to drop.
                let trim_start = self.dur_to_frames(packet.trim_start.get());
                let trim_end = self.dur_to_frames(packet.trim_end.get());
                let packet_start = self.ts_to_frame_signed(packet.pts.get()).map(|f| f + trim_start as i64);
                let buf = match self.decoder.decode(&packet) {
                    Ok(b) => b,
                    // Corrupt packet: skip it, like every player does.
                    Err(SymError::DecodeError(_)) => continue,
                    Err(e) => return Err(sym(e)),
                };
                let channels = buf.spec().channels().count();
                self.scratch.clear();
                self.scratch.resize(buf.samples_interleaved(), 0.0);
                buf.copy_to_slice_interleaved(&mut self.scratch);

                self.pending.clear();
                self.pending_pos = 0;
                to_stereo(&self.scratch, channels, &mut self.pending);
                let total = (self.pending.len() / 2) as u64;
                let keep_end = total.saturating_sub(trim_end);
                self.pending.truncate(keep_end as usize * 2);
                self.pending_pos = (trim_start.min(keep_end) as usize) * 2;
                let have = keep_end.saturating_sub(trim_start);
                if have == 0 {
                    continue;
                }
                // Frames before 0 (possible with negative timestamps) are dropped too.
                let packet_start = packet_start.map(|st| {
                    if st < 0 {
                        self.pending_pos += (-st as usize).min(have as usize) * 2;
                        0
                    } else {
                        st as u64
                    }
                });
                if let Some(start) = packet_start {
                    self.decoded_until = start + have;
                }

                if let Some(target) = self.seek_target {
                    match packet_start {
                        // Entirely before the target: drop it.
                        Some(start) if start + have <= target => continue,
                        // Contains the target: start inside it.
                        Some(start) => {
                            self.pending_pos += target.saturating_sub(start) as usize * 2;
                            self.seek_target = None;
                        }
                        // No timestamp: trust the container's seek.
                        None => self.seek_target = None,
                    }
                }
                if self.pending_pos < self.pending.len() {
                    return Ok(true);
                }
            }
        }

        /// Re-creates the format reader from the start of the same byte stream.
        fn reopen(&mut self) -> Result<(), Error> {
            use std::io::{Seek, SeekFrom};
            let reader = self.format.take().ok_or_else(|| Error::Decode("reader lost".into()))?;
            let mut mss = reader.into_inner();
            mss.seek(SeekFrom::Start(0))?;
            let format = symphonia::default::get_probe()
                .probe(&self.hint, mss, FormatOptions::default(), MetadataOptions::default())
                .map_err(sym)?;
            self.track_id = format
                .default_track(TrackType::Audio)
                .map(|t| t.id)
                .ok_or_else(|| Error::Decode("no audio track after reopen".into()))?;
            self.format = Some(format);
            self.hit_end = false;
            Ok(())
        }

        /// True when decoding has reached the known end of the track (within one packet).
        fn at_end(&self) -> bool {
            match self.duration_frames {
                Some(d) => self.decoded_until + 8192 >= d,
                None => false,
            }
        }

        /// Converts a (possibly negative) timestamp to a signed frame index.
        fn ts_to_frame_signed(&self, ts: i64) -> Option<i64> {
            let t = self.time_base?.calc_time(Timestamp::new(ts))?;
            Some((t.as_secs_f64() * self.sample_rate as f64).round() as i64)
        }

        fn dur_to_frames(&self, dur: u64) -> u64 {
            if dur == 0 {
                return 0;
            }
            self.ts_to_frame_signed(dur as i64).map_or(0, |f| f.max(0) as u64)
        }
    }

    impl FrameSource for DecodedSource {
        fn sample_rate(&self) -> u32 {
            self.sample_rate
        }

        fn duration_frames(&self) -> Option<u64> {
            self.duration_frames
        }

        fn read(&mut self, out: &mut [f32]) -> Result<usize, Error> {
            let want = out.len() / 2;
            let mut written = 0;
            while written < want {
                if self.pending_pos >= self.pending.len() && !self.decode_next()? {
                    break;
                }
                let avail = (self.pending.len() - self.pending_pos) / 2;
                let n = avail.min(want - written);
                out[written * 2..(written + n) * 2]
                    .copy_from_slice(&self.pending[self.pending_pos..self.pending_pos + n * 2]);
                self.pending_pos += n * 2;
                written += n;
            }
            Ok(written)
        }

        fn seek(&mut self, frame: u64) -> Result<(), Error> {
            // Start decoding a little earlier: transform codecs (AAC, Vorbis, Opus, MP3)
            // need the previous frame's overlap to reconstruct the first frame correctly.
            // Everything before `frame` is dropped by timestamp in `decode_next`.
            let secs = frame.saturating_sub(PREROLL_FRAMES) as f64 / self.sample_rate as f64;
            let time = Time::try_from_secs_f64(secs).unwrap_or(Time::ZERO);
            if self.hit_end {
                self.reopen()?;
            }
            let track_id = self.track_id;
            self.format
                .as_mut()
                .ok_or_else(|| Error::Decode("reader lost".into()))?
                .seek(SeekMode::Accurate, SeekTo::Time { time, track_id: Some(track_id) })
                .map_err(sym)?;
            self.decoder.reset();
            self.pending.clear();
            self.pending_pos = 0;
            self.seek_target = Some(frame);
            self.decoded_until = frame;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm_read_and_seek() {
        let mut s = PcmSource::new((0..20).map(|i| i as f32).collect::<Vec<_>>(), 48_000);
        let mut buf = [0.0; 6];
        assert_eq!(s.read(&mut buf).unwrap(), 3);
        assert_eq!(buf, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        s.seek(8).unwrap();
        assert_eq!(s.read(&mut buf).unwrap(), 2);
        assert_eq!(&buf[..4], &[16.0, 17.0, 18.0, 19.0]);
        assert_eq!(s.read(&mut buf).unwrap(), 0);
    }

    #[test]
    fn surround_folds_centre() {
        let mut out = Vec::new();
        to_stereo(&[0.1, 0.2, 1.0, 0.0, 0.0, 0.0], 6, &mut out);
        assert!((out[0] - (0.1 + std::f32::consts::FRAC_1_SQRT_2)).abs() < 1e-6);
    }
}
