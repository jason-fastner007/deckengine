//! Prints tempo and key of whole files (full decode). Usage: key_bench a.mp3 b.mp3 ...
use deckengine::{DecodedSource, FrameSource, dj};

fn main() {
    for path in std::env::args().skip(1) {
        let mut src = DecodedSource::open(&path).unwrap();
        let rate = src.sample_rate();
        let mut all = Vec::new();
        let mut buf = vec![0.0f32; 8192 * 2];
        loop {
            let n = src.read(&mut buf).unwrap();
            if n == 0 { break; }
            all.extend(buf[..n * 2].chunks_exact(2).map(|f| 0.5 * (f[0] + f[1])));
        }
        // Like the app: a 30 s window from the middle.
        let mid = all.len() / 2;
        let half = 15 * rate as usize;
        let lo = mid.saturating_sub(half);
        let win = &all[lo..(mid + half).min(all.len())];
        let a = dj::analyze(win, rate, lo as f64 * 1000.0 / rate as f64, 0.0, None);
        match a {
            Some(a) => println!("{path}\t{:.2}\t{}\t{:.2}\t{:.2}", a.bpm,
                a.key_pitch.map(|k| format!("{}{}", ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"][k as usize], if a.key_minor {"m"} else {""})).unwrap_or("-".into()),
                a.key_confidence, a.bpm_confidence),
            None => println!("{path}\tnone"),
        }
    }
}
