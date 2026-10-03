# Licensing

deckengine is **dual-licensed**.

## 1. GNU Affero General Public License v3.0 or later (default)

The source code in this repository is licensed under the
[GNU Affero General Public License v3.0 or later](LICENSE) (SPDX: `AGPL-3.0-or-later`).

You may use, modify and redistribute deckengine under the AGPL. In short:

- If you **distribute** an application that links deckengine (statically or dynamically,
  including through the C API, WebAssembly or language bindings), the application as a whole
  must be distributed under the AGPL-3.0 (or the GPL-3.0, which section 13 of both licenses
  allows to be combined with it), with its complete corresponding source code.
- If you **run** deckengine – modified or as part of a larger program – **as a network service**
  (for example a hosted audio-analysis API), you must offer its users the complete
  corresponding source code of that service.

## 2. Commercial license

If you want to ship deckengine in a closed-source product, or run it in a hosted service
without publishing your source code, a commercial license is available from the copyright
holder. It removes the AGPL obligations for the licensed product.

Contact: **Jason-Marshall Fastner — jasonfastner@protonmail.com**

## Contributions

To keep dual licensing possible, contributions are accepted only if the contributor grants
the copyright holder the right to relicense them (see [CONTRIBUTING.md](CONTRIBUTING.md)).

## Third-party components

deckengine links the following libraries. Their licenses permit both AGPL and proprietary
distribution:

| Crate | License |
|---|---|
| [cpal](https://crates.io/crates/cpal) | Apache-2.0 |
| [symphonia](https://crates.io/crates/symphonia) | MPL-2.0 |
| [ebur128](https://crates.io/crates/ebur128) | MIT |
| [rtrb](https://crates.io/crates/rtrb) | MIT OR Apache-2.0 |
| [thiserror](https://crates.io/crates/thiserror) | MIT OR Apache-2.0 |
| [windows](https://crates.io/crates/windows) (Windows only) | MIT OR Apache-2.0 |

MPL-2.0 is a file-level copyleft: modified symphonia source files must stay available under
the MPL, which does not affect the licensing of deckengine or of applications using it.
Optional backends pull in further system libraries (ALSA, PipeWire, JACK, ASIO SDK); check
their terms if you enable those features.
