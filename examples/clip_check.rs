//! Checks that an MP3 byte-range clip lands on the same timeline as the fully decoded file,
//! then analyses both. Usage: cargo run --release --example clip_check -- file.mp3 [start_s]
use deckengine::{DecodedSource, FrameSource, dj, fakeprint, mp3clip};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let start_s: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(30.0);
    let bytes = std::fs::read(path).unwrap();

    // Reference: whole file, gapless, as a player sees it.
    let mut src = DecodedSource::open(path).unwrap();
    let rate = src.sample_rate();
    let mut all = Vec::new();
    let mut buf = vec![0.0f32; 8192 * 2];
    loop {
        let n = src.read(&mut buf).unwrap();
        if n == 0 { break; }
        all.extend(buf[..n * 2].chunks_exact(2).map(|f| 0.5 * (f[0] + f[1])));
    }
    let dur_ms = all.len() as f64 / rate as f64 * 1000.0;

    let head = mp3clip::parse_head(&bytes[..8192]).unwrap();
    println!("head {head:?}");
    let from = head.byte_at(start_s * 1000.0) as usize;
    let to = head.byte_at((start_s + 30.0) * 1000.0) as usize;
    let t = std::time::Instant::now();
    let clip = mp3clip::decode_clip(&head, &bytes[from..to.min(bytes.len())], from as u64).unwrap();
    println!("clip {} samples from {:.2} ms (decode {:?})", clip.pcm.len(), clip.start_ms, t.elapsed());

    // Cross-correlate a 1 s piece of the clip against the reference around the claimed time.
    let at = (clip.start_ms / 1000.0 * rate as f64).round() as i64;
    let piece = &clip.pcm[rate as usize..2 * rate as usize];
    let (mut best, mut best_lag) = (f64::MIN, 0i64);
    for lag in -2000i64..=2000 {
        let base = at + rate as i64 + lag;
        if base < 0 { continue; }
        let r: f64 = piece.iter().enumerate().map(|(i, &v)| v as f64 * all[(base as usize) + i] as f64).sum();
        if r > best { best = r; best_lag = lag; }
    }
    println!("timeline error: {} samples = {:.2} ms", best_lag, best_lag as f64 * 1000.0 / rate as f64);

    let t = std::time::Instant::now();
    let a = dj::analyze(&clip.pcm, clip.sample_rate, clip.start_ms, dur_ms, None).unwrap();
    println!("clip analysis ({:?}): {:.2} BPM conf {:.2} first {:.1} down {:.1} phrase {:.1} key {:?}{} ({}) conf {:.2} lufs {:.1} energy {:.1}",
        t.elapsed(), a.bpm, a.bpm_confidence, a.first_beat_ms, a.downbeat_ms, a.phrase_ms, a.key_pitch,
        if a.key_minor {"m"} else {""}, a.key_pitch.map(|k| dj::camelot(k, a.key_minor)).unwrap_or("-"), a.key_confidence, a.lufs, a.energy);
    let t = std::time::Instant::now();
    let f = dj::analyze(&all, rate, 0.0, dur_ms, None).unwrap();
    println!("full analysis ({:?}): {:.2} BPM conf {:.2} first {:.1} down {:.1} phrase {:.1} key {:?}{} ({})",
        t.elapsed(), f.bpm, f.bpm_confidence, f.first_beat_ms, f.downbeat_ms, f.phrase_ms, f.key_pitch,
        if f.key_minor {"m"} else {""}, f.key_pitch.map(|k| dj::camelot(k, f.key_minor)).unwrap_or("-"));
    let t = std::time::Instant::now();
    println!("fakeprint clip {:?} ({:?})", fakeprint::detect(&clip.pcm, clip.sample_rate), t.elapsed());
}
