# Contributing

Thanks for helping! Bug reports, measurements from real hardware (especially Windows
drivers and USB interfaces) and pull requests are welcome.

## Contributor License Agreement

deckengine is [dual-licensed](LICENSING.md) (AGPL-3.0-or-later and a commercial license).
To keep offering both, every contribution needs a Contributor License Agreement (CLA) that
lets the maintainer distribute your contribution under both licenses. You keep the copyright
to your work. Please say in your pull request that you agree, and the maintainer will send
you the CLA before merging.

## Ground rules for the audio thread

Code reachable from `Processor::process` runs in the real-time audio callback and must never:

- allocate or free memory (`Vec::push`, `Box::new`, `String`, dropping a `Box`, ...),
- take a lock or wait on anything,
- do I/O, log or print,
- panic in normal operation.

The test `audio_thread_never_allocates` enforces the first rule with a counting allocator.
Run it after every change to the audio path.

## Checks

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --release
# Windows cross-check from Linux:
rustup target add x86_64-pc-windows-gnu
cargo clippy --target x86_64-pc-windows-gnu --all-targets -- -D warnings
```

Every source file starts with:

```rust
// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Jason-Marshall Fastner <jasonfastner@protonmail.com>
```
