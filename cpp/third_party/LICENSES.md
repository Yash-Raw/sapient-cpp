# Third-party components of the C++ tree

Every dependency pinned in `cpp/cmake/deps.cmake` (or linked from the system) is listed here with
its licence. All are compatible with SAPIENT's AGPL-3.0-only OR commercial dual licence; none is
copyleft-incompatible or bundles its own model weights. `NOTICE` points here.

## Pinned now (sub-project 0)

| Component | Version / tag | Licence | Used for | Where |
|---|---|---|---|---|
| GoogleTest | v1.15.2 | BSD-3-Clause | unit tests (`SAPIENT_BUILD_TESTS` only, not shipped) | `cmake/deps.cmake` |

## Planned (spec §D4 — add the row when the pin lands, not before)

| Component | Licence | Used for | Sub-project |
|---|---|---|---|
| tl::expected (TartanLlama) | CC0-1.0 | `sapient::Result<T>` until std::expected | 1a |
| nlohmann/json | MIT | config.json / tokenizer.json / HTTP bodies | 1b |
| PCRE2 | BSD-3-Clause | tokenizer pre-tokenizer regexes | 1b |
| minja (ggml-org) | MIT | Jinja chat templates | 1b |
| CLI11 | BSD-3-Clause | CLI parsing | 3 |
| cpp-httplib | MIT | `sapient serve` | 3 |
| libcurl | curl (MIT-style) | HF Hub downloads, self-update | 3 |
| replxx *or* isocline | BSD-3-Clause / MIT | chat line editor (bracketed paste) | 3 |
| md4c | MIT | Markdown parsing for the chat TUI | 3 |
| indicators | MIT | progress bars | 3 |
| miniz-ng | MIT | self-update archives | 3 |
| wgpu-native | MIT OR Apache-2.0 | portable GPU engine (v22 line) | 4 |
| MLX | MIT | Metal engine (macOS) | 4 |
| dr_wav / dr_flac / dr_mp3 | Public domain / MIT-0 | audio decode | 5a |
| stb_vorbis, stb_image | Public domain / MIT | OGG decode; PNG/JPEG decode | 5a / 6 |
| pocketfft | BSD-3-Clause | STFT / iSTFT | 5a |
| miniaudio | Public domain / MIT-0 | mic + speaker | 5c |
| libwebp | BSD-3-Clause | WebP decode | 6 |
| spdlog | MIT | logging | 3 |
| Google Benchmark | Apache-2.0 | benchmarks (not shipped) | 1a |
