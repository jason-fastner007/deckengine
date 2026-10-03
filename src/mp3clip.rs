// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Jason-Marshall Fastner <jasonfastner@protonmail.com>

//! Decodes a byte range cut out of a CBR MP3 (an HTTP `Range` request) onto the track's own
//! timeline, so analyses only have to fetch the seconds they look at.
//!
//! The head of the file (first few KB) says where the audio starts (after ID3v2 and the
//! Xing/Info frame) and how much encoder delay players trim (LAME tag). With that, the time of
//! any frame in the clip follows from its byte offset: CBR frames advance by a fixed number of
//! bytes on average.

use crate::Error;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::units::Timestamp;

/// MP3 decoders emit their output this many samples late (polyphase synthesis).
const DECODER_DELAY: u32 = 529;
/// Frames decoded at the start of a clip before the output is trusted (bit reservoir, overlap).
const PRIMING_FRAMES: usize = 2;

/// Stream parameters read from the head of the file.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mp3Head {
    /// Byte offset of the first audio frame (after ID3v2 and the Xing/Info frame).
    pub audio_start: u64,
    pub sample_rate: u32,
    pub bitrate_kbps: u32,
    pub samples_per_frame: u32,
    /// Average frame length in bytes (CBR: 144000 · kbps / rate for MPEG-1).
    pub frame_bytes: f64,
    /// Samples players drop at the start (LAME encoder delay + decoder delay), 0 without a tag.
    pub trim_samples: u32,
}

impl Mp3Head {
    pub fn bytes_per_second(&self) -> f64 {
        self.bitrate_kbps as f64 * 125.0
    }

    /// Byte offset where audio at `ms` on the player's timeline lives.
    pub fn byte_at(&self, ms: f64) -> u64 {
        let frames = ((ms / 1000.0 * self.sample_rate as f64 + self.trim_samples as f64)
            / self.samples_per_frame as f64)
            .max(0.0);
        self.audio_start + (frames * self.frame_bytes) as u64
    }
}

#[derive(Debug, Clone, Copy)]
struct FrameHeader {
    mpeg1: bool,
    layer3: bool,
    bitrate_kbps: u32,
    sample_rate: u32,
    padding: bool,
    mono: bool,
}

impl FrameHeader {
    fn parse(b: &[u8]) -> Option<Self> {
        if b.len() < 4 || b[0] != 0xFF || b[1] & 0xE0 != 0xE0 {
            return None;
        }
        let version = (b[1] >> 3) & 3; // 3 = MPEG-1, 2 = MPEG-2, 0 = MPEG-2.5
        let layer = (b[1] >> 1) & 3; // 1 = Layer III
        let br_idx = (b[2] >> 4) as usize;
        let sr_idx = ((b[2] >> 2) & 3) as usize;
        if version == 1 || layer == 0 || br_idx == 0 || br_idx == 15 || sr_idx == 3 {
            return None;
        }
        const BR_V1_L3: [u32; 15] = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320];
        const BR_V2_L3: [u32; 15] = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160];
        const SR: [u32; 3] = [44_100, 48_000, 32_000];
        let mpeg1 = version == 3;
        if layer != 1 {
            // Only Layer III is used for music streams.
            return None;
        }
        let div = match version {
            3 => 1,
            2 => 2,
            _ => 4,
        };
        Some(Self {
            mpeg1,
            layer3: true,
            bitrate_kbps: if mpeg1 { BR_V1_L3[br_idx] } else { BR_V2_L3[br_idx] },
            sample_rate: SR[sr_idx] / div,
            padding: b[2] & 2 != 0,
            mono: (b[3] >> 6) == 3,
        })
    }

    fn samples(&self) -> u32 {
        if self.mpeg1 { 1152 } else { 576 }
    }

    fn len(&self) -> usize {
        let k = if self.mpeg1 { 144_000 } else { 72_000 };
        (k * self.bitrate_kbps / self.sample_rate) as usize + self.padding as usize
    }

    fn side_info(&self) -> usize {
        match (self.mpeg1, self.mono) {
            (true, false) => 32,
            (true, true) => 17,
            (false, false) => 17,
            (false, true) => 9,
        }
    }
}

/// Finds a frame at or after `from` that is followed by two more frames of the same stream.
fn find_sync(b: &[u8], from: usize, like: Option<&FrameHeader>) -> Option<(usize, FrameHeader)> {
    let mut i = from;
    while i + 4 <= b.len() {
        if let Some(h) = FrameHeader::parse(&b[i..]) {
            let same = |o: &FrameHeader| o.sample_rate == h.sample_rate && o.mpeg1 == h.mpeg1 && o.layer3;
            if like.is_none_or(|l| l.sample_rate == h.sample_rate) {
                let n1 = i + h.len();
                let ok1 = b.get(n1..).and_then(FrameHeader::parse).is_some_and(|o| same(&o));
                let ok2 = ok1
                    && b.get(n1..).and_then(FrameHeader::parse).is_some_and(|o| {
                        b.get(n1 + o.len()..).and_then(FrameHeader::parse).is_some_and(|p| same(&p))
                    });
                // At the very end of the buffer one confirmation has to do.
                if ok2 || (ok1 && n1 + 4 + 2000 > b.len()) {
                    return Some((i, h));
                }
            }
        }
        i += 1;
    }
    None
}

/// Reads ID3v2, the first frame and its Xing/Info/LAME tag.
///
/// `Err(n)`: the head is too short, fetch at least `n` bytes.
pub fn parse_head(head: &[u8]) -> Result<Mp3Head, usize> {
    let mut pos = 0usize;
    if head.len() < 10 {
        return Err(4096);
    }
    if &head[..3] == b"ID3" {
        let size = ((head[6] as usize & 0x7F) << 21)
            | ((head[7] as usize & 0x7F) << 14)
            | ((head[8] as usize & 0x7F) << 7)
            | (head[9] as usize & 0x7F);
        pos = 10 + size + if head[5] & 0x10 != 0 { 10 } else { 0 };
    }
    if head.len() < pos + 3000 {
        return Err(pos + 4096);
    }
    let (first, h) = find_sync(head, pos, None).ok_or(0usize)?;
    let tag_at = first + 4 + h.side_info();
    let mut audio_start = first as u64;
    let mut trim = 0u32;
    if let Some(tag) = head.get(tag_at..tag_at + 4)
        && (tag == b"Xing" || tag == b"Info")
    {
        // The tag frame carries no audio.
        audio_start = (first + h.len()) as u64;
        let flags = u32::from_be_bytes(head[tag_at + 4..tag_at + 8].try_into().unwrap());
        let mut lame = tag_at + 8;
        for (bit, len) in [(1, 4), (2, 4), (4, 100), (8, 4)] {
            if flags & bit != 0 {
                lame += len;
            }
        }
        // LAME extension: 9-byte version string, then delay/padding at +21 (12 + 12 bits).
        if let Some(d) = head.get(lame + 21..lame + 24)
            && head.get(lame..lame + 4).is_some_and(|s| s == b"LAME" || s == b"Lavc" || s == b"Lavf")
        {
            let delay = ((d[0] as u32) << 4) | (d[1] as u32 >> 4);
            trim = delay + DECODER_DELAY;
        }
    }
    Ok(Mp3Head {
        audio_start,
        sample_rate: h.sample_rate,
        bitrate_kbps: h.bitrate_kbps,
        samples_per_frame: h.samples(),
        frame_bytes: (if h.mpeg1 { 144_000.0 } else { 72_000.0 }) * h.bitrate_kbps as f64 / h.sample_rate as f64,
        trim_samples: trim,
    })
}

/// Decoded clip: mono PCM at the stream's rate and where it starts on the player's timeline.
pub struct Clip {
    pub pcm: Vec<f32>,
    pub sample_rate: u32,
    pub start_ms: f64,
}

fn sym(e: SymError) -> Error {
    Error::Decode(e.to_string())
}

/// Decodes `bytes`, which were cut out of the file starting at `byte_offset`.
pub fn decode_clip(head: &Mp3Head, bytes: &[u8], byte_offset: u64) -> Result<Clip, Error> {
    let probe = FrameHeader {
        mpeg1: head.samples_per_frame == 1152,
        layer3: true,
        bitrate_kbps: head.bitrate_kbps,
        sample_rate: head.sample_rate,
        padding: false,
        mono: false,
    };
    let (sync, _) = find_sync(bytes, 0, Some(&probe)).ok_or_else(|| Error::Decode("no MP3 frame in clip".into()))?;
    let abs = byte_offset + sync as u64;
    if abs < head.audio_start {
        return Err(Error::InvalidArgument("clip starts before the audio"));
    }
    let frame_index = ((abs - head.audio_start) as f64 / head.frame_bytes).round() as i64;

    let src = std::io::Cursor::new(bytes[sync..].to_vec());
    let mss = MediaSourceStream::new(Box::new(src), Default::default());
    let mut hint = Hint::new();
    hint.with_extension("mp3");
    let mut format = symphonia::default::get_probe()
        .probe(&hint, mss, FormatOptions::default(), MetadataOptions::default())
        .map_err(sym)?;
    let track = format.default_track(TrackType::Audio).ok_or_else(|| Error::Decode("no audio track".into()))?;
    let track_id = track.id;
    let time_base = track.time_base;
    let params = track.codec_params.as_ref().and_then(|p| p.audio()).ok_or_else(|| Error::Decode("no codec".into()))?;
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(params, &AudioDecoderOptions::default().gapless(false))
        .map_err(sym)?;

    let rate = head.sample_rate;
    let mut pcm: Vec<f32> = Vec::with_capacity(bytes.len() / head.frame_bytes as usize * head.samples_per_frame as usize);
    let mut scratch: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(SymError::IoError(_)) => break,
            Err(SymError::DecodeError(_)) => continue,
            Err(e) => return Err(sym(e)),
        };
        if packet.track_id != track_id {
            continue;
        }
        // Place every packet by its own timestamp, so a skipped (corrupt) packet leaves a gap
        // instead of shifting everything after it.
        let at = time_base
            .and_then(|tb| tb.calc_time(Timestamp::new(packet.pts.get())))
            .map(|t| (t.as_secs_f64() * rate as f64).round() as usize);
        let buf = match decoder.decode(&packet) {
            Ok(b) => b,
            Err(SymError::DecodeError(_)) => continue,
            Err(e) => return Err(sym(e)),
        };
        let ch = buf.spec().channels().count().max(1);
        scratch.clear();
        scratch.resize(buf.samples_interleaved(), 0.0);
        buf.copy_to_slice_interleaved(&mut scratch);
        let start = at.unwrap_or(pcm.len());
        if pcm.len() < start {
            pcm.resize(start, 0.0);
        }
        pcm.truncate(start);
        pcm.extend(scratch.chunks_exact(ch).map(|f| f.iter().sum::<f32>() / ch as f32));
    }
    let skip = (PRIMING_FRAMES * head.samples_per_frame as usize).min(pcm.len());
    pcm.drain(..skip);
    let first_sample = frame_index * head.samples_per_frame as i64 + skip as i64 - head.trim_samples as i64;
    Ok(Clip { pcm, sample_rate: rate, start_ms: first_sample as f64 * 1000.0 / rate as f64 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_header() {
        // MPEG-1 Layer III, 128 kbps, 44.1 kHz, no padding, joint stereo.
        let h = FrameHeader::parse(&[0xFF, 0xFB, 0x90, 0x40]).unwrap();
        assert_eq!((h.bitrate_kbps, h.sample_rate, h.len(), h.samples()), (128, 44_100, 417, 1152));
    }
}
