use deckengine::{DecodedSource, FrameSource, dj};
fn main() {
    for path in std::env::args().skip(1) {
        let mut src = DecodedSource::open(&path).unwrap();
        let rate = src.sample_rate();
        let mut all = Vec::new();
        let mut buf = vec![0.0f32; 16384];
        loop { let n = src.read(&mut buf).unwrap(); if n == 0 { break; } all.extend(buf[..n * 2].chunks_exact(2).map(|f| 0.5 * (f[0] + f[1]))); }
        let mid = all.len() / 2; let half = 15 * rate as usize;
        eprintln!("{path}");
        dj::debug_tempo(&all[mid - half..mid + half], rate);
    }
}
