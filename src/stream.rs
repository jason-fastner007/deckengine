// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Jason-Marshall Fastner <jasonfastner@protonmail.com>

//! Streaming between a [`FrameSource`] (background thread) and a deck (audio thread).
//!
//! The streamer thread decodes ahead into a lock-free ring of fixed-size [`Chunk`]s. Every
//! chunk carries a *generation*. Seeks and loop changes never flush the ring from the other
//! side (which would need a lock). Instead the streamer bumps the generation, repositions the
//! source and publishes the new generation, and the audio thread switches over as soon as
//! new-generation data is in the ring:
//!
//! * **aligned** switches (loop set/cleared) happen at the exact sample where old and new data
//!   describe the same position, so they are seamless;
//! * **hard** switches (user seeks) use a ~2 ms micro-crossfade, so they never click;
//! * until the new data is there, the old audio keeps playing, so a seek never plays silence.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::thread::JoinHandle;
use std::time::Duration;

use rtrb::{Consumer, Producer, RingBuffer};

use crate::Error;
use crate::source::{FrameSource, SourceInfo};

/// Frames per ring chunk.
pub(crate) const CHUNK_FRAMES: usize = 256;
/// Length of the micro-crossfade used for hard switches, in source frames.
const XFADE_FRAMES: usize = 96;
/// How far ahead of the playhead aligned switches are scheduled, in milliseconds.
const ALIGN_MARGIN_MS: u64 = 40;

#[derive(Clone)]
pub(crate) struct Chunk {
    generation: u32,
    start: u64,
    len: u32,
    eos: bool,
    aligned: bool,
    data: [f32; CHUNK_FRAMES * 2],
}

impl Chunk {
    fn empty() -> Self {
        Self { generation: 0, start: 0, len: 0, eos: false, aligned: false, data: [0.0; CHUNK_FRAMES * 2] }
    }
}

/// State shared between streamer thread, audio thread and control side.
pub(crate) struct StreamShared {
    /// Newest generation the streamer has produced (or is producing).
    latest_gen: AtomicU32,
    /// Source frame the deck is currently playing (written by the audio thread).
    pub playhead: AtomicU64,
    /// Set by the streamer when the source reports an error.
    pub failed: AtomicBool,
}

pub(crate) enum StreamCmd {
    Seek(u64),
    SetLoop(Option<(u64, u64)>),
}

/// Control-side handle of a running streamer thread.
pub(crate) struct StreamControl {
    tx: Sender<StreamCmd>,
    pub shared: Arc<StreamShared>,
    thread: Option<JoinHandle<()>>,
}

impl StreamControl {
    pub fn send(&self, cmd: StreamCmd) {
        let _ = self.tx.send(cmd);
    }
}

impl Drop for StreamControl {
    fn drop(&mut self) {
        // Dropping the sender makes the thread exit on its next command check.
        let (tx, _) = std::sync::mpsc::channel();
        drop(std::mem::replace(&mut self.tx, tx));
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Starts a streamer thread for `source`. Returns the control handle and the audio-thread side.
pub(crate) fn spawn(
    source: Box<dyn FrameSource>,
    buffer_ms: u32,
) -> Result<(StreamControl, Box<Feed>), Error> {
    let info = SourceInfo { sample_rate: source.sample_rate(), duration_frames: source.duration_frames() };
    if info.sample_rate == 0 {
        return Err(Error::Decode("source reports a sample rate of 0".into()));
    }
    let chunks = ((info.sample_rate as u64 * buffer_ms.max(50) as u64 / 1000) as usize / CHUNK_FRAMES).max(8);
    let (producer, consumer) = RingBuffer::new(chunks);
    let shared = Arc::new(StreamShared {
        latest_gen: AtomicU32::new(0),
        playhead: AtomicU64::new(0),
        failed: AtomicBool::new(false),
    });
    let (tx, rx) = std::sync::mpsc::channel();

    let thread_shared = shared.clone();
    let margin = info.sample_rate as u64 * ALIGN_MARGIN_MS / 1000;
    let thread = std::thread::Builder::new()
        .name("deckengine-stream".into())
        .spawn(move || Streamer::new(source, producer, rx, thread_shared, margin).run())?;

    let feed = Box::new(Feed {
        consumer,
        shared: shared.clone(),
        info,
        generation: 0,
        offset: 0,
        pos: 0,
        switch_at: None,
        pending_frames: 0,
        fallback_frames: margin * 3,
        xf_tail: [[0.0; 2]; XFADE_FRAMES],
        xf_len: 0,
        xf_idx: 0,
        eos: false,
    });
    Ok((StreamControl { tx, shared, thread: Some(thread) }, feed))
}

struct Streamer {
    source: Box<dyn FrameSource>,
    producer: Producer<Chunk>,
    rx: Receiver<StreamCmd>,
    shared: Arc<StreamShared>,
    margin: u64,
    generation: u32,
    pos: u64,
    looping: Option<(u64, u64)>,
    wrapped: bool,
    eos: bool,
    next_aligned: bool,
    chunk: Box<Chunk>,
}

impl Streamer {
    fn new(
        source: Box<dyn FrameSource>,
        producer: Producer<Chunk>,
        rx: Receiver<StreamCmd>,
        shared: Arc<StreamShared>,
        margin: u64,
    ) -> Self {
        Self {
            source,
            producer,
            rx,
            shared,
            margin,
            generation: 0,
            pos: 0,
            looping: None,
            wrapped: false,
            eos: false,
            next_aligned: false,
            chunk: Box::new(Chunk::empty()),
        }
    }

    fn run(mut self) {
        loop {
            if self.producer.is_abandoned() {
                return;
            }
            let idle = self.eos || self.producer.is_full();
            let cmd = if idle {
                match self.rx.recv_timeout(Duration::from_millis(if self.eos { 20 } else { 2 })) {
                    Ok(c) => Some(c),
                    Err(RecvTimeoutError::Timeout) => None,
                    Err(RecvTimeoutError::Disconnected) => return,
                }
            } else {
                match self.rx.try_recv() {
                    Ok(c) => Some(c),
                    Err(TryRecvError::Empty) => None,
                    Err(TryRecvError::Disconnected) => return,
                }
            };
            match cmd {
                Some(StreamCmd::Seek(frame)) => self.restart(frame, false),
                Some(StreamCmd::SetLoop(region)) => self.set_loop(region),
                None if !idle => self.fill_one(),
                None => {}
            }
        }
    }

    /// Starts a new generation at `frame`.
    fn restart(&mut self, frame: u64, aligned: bool) {
        self.generation = self.generation.wrapping_add(1);
        self.eos = false;
        self.wrapped = false;
        self.next_aligned = aligned;
        self.pos = frame;
        if let Err(_e) = self.source.seek(frame) {
            self.shared.failed.store(true, Ordering::Relaxed);
            self.eos = true;
            self.push_eos();
        }
        self.shared.latest_gen.store(self.generation, Ordering::Release);
    }

    fn set_loop(&mut self, region: Option<(u64, u64)>) {
        let region = region.filter(|&(a, b)| b > a);
        let playhead = self.shared.playhead.load(Ordering::Relaxed);
        let old = self.looping;
        self.looping = region;

        match (old, region) {
            (None, None) => {}
            // Loop set ahead of the decoder: the buffered audio is still valid.
            (None, Some((_, b))) if self.pos <= b && playhead < b => {}
            // Loop set behind the playhead: jump straight into it.
            (_, Some((a, b))) if playhead >= b => self.restart(a, false),
            // Loop removed/changed before the decoder wrapped: buffered audio still valid,
            // unless the new end lies behind the decode position.
            (Some(_), None) if !self.wrapped => {}
            (Some(_), Some((_, b))) if !self.wrapped && self.pos <= b => {}
            // Buffered audio no longer matches: re-decode from slightly ahead of the playhead
            // and let the deck switch at exactly that sample.
            _ => {
                let mut start = playhead + self.margin;
                if let Some((a, b)) = region {
                    if start >= b {
                        start = a;
                    }
                }
                self.restart(start, true);
            }
        }
    }

    fn fill_one(&mut self) {
        let mut limit = CHUNK_FRAMES;
        if let Some((a, b)) = self.looping {
            if self.pos >= b {
                if self.source.seek(a).is_err() {
                    self.shared.failed.store(true, Ordering::Relaxed);
                    self.eos = true;
                    self.push_eos();
                    return;
                }
                self.pos = a;
                self.wrapped = true;
            }
            limit = limit.min((b - self.pos) as usize);
        }

        let c = &mut *self.chunk;
        let n = match self.source.read(&mut c.data[..limit * 2]) {
            Ok(n) => n,
            Err(_) => {
                self.shared.failed.store(true, Ordering::Relaxed);
                0
            }
        };
        if n == 0 {
            // A loop that runs past the real end of the track wraps instead of ending.
            if let Some((a, _)) = self.looping {
                if self.pos > a && self.source.seek(a).is_ok() {
                    self.pos = a;
                    self.wrapped = true;
                    return;
                }
            }
            self.eos = true;
            self.push_eos();
            return;
        }
        c.generation = self.generation;
        c.start = self.pos;
        c.len = n as u32;
        c.eos = false;
        c.aligned = std::mem::take(&mut self.next_aligned);
        self.pos += n as u64;
        let _ = self.producer.push((*self.chunk).clone());
    }

    fn push_eos(&mut self) {
        let mut c = Chunk::empty();
        c.generation = self.generation;
        c.start = self.pos;
        c.eos = true;
        c.aligned = std::mem::take(&mut self.next_aligned);
        // Wait for room; the consumer is always draining unless paused, in which case the
        // ring is abandoned or the command channel wakes us.
        let mut c = Some(c);
        while let Some(chunk) = c.take() {
            match self.producer.push(chunk) {
                Ok(()) => {}
                Err(rtrb::PushError::Full(chunk)) => {
                    if self.producer.is_abandoned() {
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(2));
                    c = Some(chunk);
                }
            }
        }
    }
}

/// What the feed produced for one frame.
pub(crate) enum Next {
    Frame([f32; 2]),
    /// No data yet (decoder behind). Output silence, do not advance.
    Underrun,
    /// End of stream for the current generation.
    Eos,
}

/// Audio-thread side of a stream. Allocation-free.
pub(crate) struct Feed {
    consumer: Consumer<Chunk>,
    pub shared: Arc<StreamShared>,
    pub info: SourceInfo,
    generation: u32,
    offset: usize,
    /// Position of the next frame to be returned.
    pos: u64,
    /// Aligned switch scheduled for when `pos` reaches this frame.
    switch_at: Option<u64>,
    pending_frames: u64,
    fallback_frames: u64,
    xf_tail: [[f32; 2]; XFADE_FRAMES],
    xf_len: usize,
    xf_idx: usize,
    eos: bool,
}

impl Feed {
    /// Source position of the next frame.
    pub fn position(&self) -> u64 {
        self.pos
    }

    pub fn is_eos(&self) -> bool {
        self.eos
    }

    fn latest(&self) -> u32 {
        self.shared.latest_gen.load(Ordering::Acquire)
    }

    /// Finds the first chunk of generation `g` in the ring: `(index, start, aligned)`.
    fn find_gen(&mut self, g: u32) -> Option<(usize, u64, bool)> {
        let slots = self.consumer.slots();
        let rc = self.consumer.read_chunk(slots).ok()?;
        let (a, b) = rc.as_slices();
        a.iter()
            .chain(b.iter())
            .enumerate()
            .find(|(_, c)| c.generation == g)
            .map(|(i, c)| (i, c.start, c.aligned))
    }

    /// Frames of generation `g` available from ring index `from` on.
    fn frames_of_gen(&mut self, g: u32, from: usize) -> u64 {
        let slots = self.consumer.slots();
        let Ok(rc) = self.consumer.read_chunk(slots) else { return 0 };
        let (a, b) = rc.as_slices();
        a.iter().chain(b.iter()).skip(from).filter(|c| c.generation == g).map(|c| c.len as u64).sum()
    }

    fn discard(&mut self, n: usize) {
        if n > 0 {
            if let Ok(rc) = self.consumer.read_chunk(n) {
                rc.commit_all();
            }
        }
        self.offset = 0;
    }

    /// Called once per audio block. Handles pending generation switches.
    pub fn poll(&mut self, block_frames: usize) {
        let latest = self.latest();
        if latest == self.generation {
            self.switch_at = None;
            self.pending_frames = 0;
            return;
        }
        let Some((idx, start, aligned)) = self.find_gen(latest) else {
            // New data not decoded yet: keep playing the old stream meanwhile.
            self.pending_frames += block_frames as u64;
            return;
        };
        self.pending_frames += block_frames as u64;

        if !aligned || self.pending_frames > self.fallback_frames {
            self.switch_hard(idx, latest);
        } else if self.pos >= start {
            let skip = self.pos - start;
            if self.frames_of_gen(latest, idx) > skip {
                self.switch_seamless(idx, latest, skip);
            } else {
                self.switch_hard(idx, latest);
            }
        } else {
            self.switch_at = Some(start);
        }
    }

    /// Switch now, crossfading from the remaining old audio into the new generation.
    fn switch_hard(&mut self, idx: usize, g: u32) {
        // Capture up to XFADE_FRAMES of old audio to fade out.
        let mut n = 0;
        if let Ok(rc) = self.consumer.read_chunk(idx) {
            let (a, b) = rc.as_slices();
            let mut first = true;
            for c in a.iter().chain(b.iter()) {
                if c.generation != self.generation {
                    first = false;
                    continue;
                }
                let from = if first { self.offset } else { 0 };
                first = false;
                for f in from..c.len as usize {
                    if n == XFADE_FRAMES {
                        break;
                    }
                    self.xf_tail[n] = [c.data[f * 2], c.data[f * 2 + 1]];
                    n += 1;
                }
            }
        }
        self.xf_len = n;
        self.xf_idx = 0;
        self.finish_switch(idx, g, 0);
    }

    fn switch_seamless(&mut self, idx: usize, g: u32, skip: u64) {
        self.xf_len = 0;
        self.xf_idx = 0;
        self.finish_switch(idx, g, skip);
    }

    fn finish_switch(&mut self, idx: usize, g: u32, mut skip: u64) {
        self.discard(idx);
        self.generation = g;
        self.eos = false;
        self.switch_at = None;
        self.pending_frames = 0;
        // Skip `skip` frames into the new generation.
        while let Ok(c) = self.consumer.peek() {
            let (cg, eos, start, len) = (c.generation, c.eos, c.start, c.len as u64);
            self.pos = start;
            if cg != g || eos || skip == 0 {
                break;
            }
            if skip >= len {
                skip -= len;
                self.discard(1);
            } else {
                self.offset = skip as usize;
                self.pos = start + skip;
                break;
            }
        }
    }

    /// Returns the next source frame.
    #[inline]
    pub fn next(&mut self) -> Next {
        if self.switch_at == Some(self.pos) {
            let g = self.latest();
            if let Some((idx, _, _)) = self.find_gen(g) {
                self.switch_seamless(idx, g, 0);
            } else {
                self.switch_at = None;
            }
        }
        loop {
            let (g, eos) = match self.consumer.peek() {
                Ok(c) => (c.generation, c.eos),
                Err(_) => return Next::Underrun,
            };
            if g != self.generation {
                if g == self.latest() {
                    // Old data ran out before the switch point: switch now.
                    self.switch_hard(0, g);
                    continue;
                }
                // Stale generation from an earlier, superseded seek.
                self.discard(1);
                continue;
            }
            if eos {
                self.eos = true;
                return Next::Eos;
            }
            let f = self.offset;
            let (mut frame, start, len) = match self.consumer.peek() {
                Ok(c) => ([c.data[f * 2], c.data[f * 2 + 1]], c.start, c.len as usize),
                Err(_) => return Next::Underrun,
            };
            self.pos = start + f as u64 + 1;
            self.offset += 1;
            if self.offset >= len {
                self.discard(1);
            }
            if self.xf_idx < self.xf_len {
                let t = (self.xf_idx + 1) as f32 / (self.xf_len + 1) as f32;
                let old = self.xf_tail[self.xf_idx];
                frame[0] = old[0] * (1.0 - t) + frame[0] * t;
                frame[1] = old[1] * (1.0 - t) + frame[1] * t;
                self.xf_idx += 1;
            }
            return Next::Frame(frame);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::PcmSource;

    fn ramp_source(frames: usize) -> Box<PcmSource> {
        let v: Vec<f32> = (0..frames).flat_map(|i| [i as f32, i as f32]).collect();
        Box::new(PcmSource::new(v, 48_000))
    }

    fn pull(feed: &mut Feed, n: usize) -> Vec<f32> {
        let mut out = Vec::new();
        let mut spins = 0;
        while out.len() < n {
            feed.poll(1);
            match feed.next() {
                Next::Frame(f) => out.push(f[0]),
                Next::Underrun => {
                    spins += 1;
                    assert!(spins < 2_000_000, "stream stalled");
                    std::thread::yield_now();
                }
                Next::Eos => break,
            }
        }
        out
    }

    #[test]
    fn plays_in_order_to_eos() {
        let (_ctl, mut feed) = spawn(ramp_source(10_000), 100).unwrap();
        let out = pull(&mut feed, 20_000);
        assert_eq!(out.len(), 10_000);
        assert!(out.iter().enumerate().all(|(i, &x)| x == i as f32));
    }

    #[test]
    fn seek_switches_generation() {
        let (ctl, mut feed) = spawn(ramp_source(100_000), 100).unwrap();
        let _ = pull(&mut feed, 1000);
        ctl.send(StreamCmd::Seek(50_000));
        // Wait until the switch happened, then the samples must continue from 50_000 on
        // (after the short crossfade).
        for _ in 0..20_000 {
            let _ = pull(&mut feed, 1);
            if feed.generation != 0 && feed.xf_idx >= feed.xf_len {
                break;
            }
        }
        let after = pull(&mut feed, 2);
        assert!(after[0] >= 50_000.0, "{after:?}");
        assert_eq!(after[1], after[0] + 1.0);
    }

    #[test]
    fn loop_is_sample_exact() {
        let (ctl, mut feed) = spawn(ramp_source(100_000), 100).unwrap();
        ctl.send(StreamCmd::SetLoop(Some((1000, 1100))));
        // Give the streamer a moment; the loop is ahead of the decoder or re-aligned.
        std::thread::sleep(Duration::from_millis(50));
        let out = pull(&mut feed, 5000);
        // Once inside the loop, the sequence must be 1000..1100 repeating with no gaps.
        let first = out.iter().position(|&x| x == 1099.0).unwrap();
        for w in out[first..].windows(2) {
            let (a, b) = (w[0], w[1]);
            assert!(b == a + 1.0 || (a == 1099.0 && b == 1000.0), "{a} -> {b}");
        }
    }
}
