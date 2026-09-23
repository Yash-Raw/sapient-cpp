# Sub-project 1a, Plan B: `sapient::io` — GGUF + safetensors loaders — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port `crates/sapient-io` (minus the ONNX reader and the dead `Graph` entry points, which moved to sub-project 8) to a new `sapient::io` C++ library — the whole-file mapping and read facilities, the GGUF header parser, the heap/mmap/metadata-only GGUF loaders with their Q5_0→Q8_0 requantisation, the safetensors loader and the two crate-level helpers — with every Result-path error text byte-identical to the Rust oracle, both Rust unit tests ported by name, synthetic-file tests for every branch, and an env-gated real-file heap-vs-mmap byte check. This is the last plan of sub-project 1a.

**Architecture:** Four `.hpp/.cpp` pairs under `cpp/libs/sapient-io/`. `rust_std` holds the Rust-standard-library text twins the loaders embed in their errors (the `read_exact` EOF text, `Utf8Error`'s Display, `io::Error`'s "`<description> (os error N)`"). `mmap` is the `memmap2::Mmap::map` + `std::fs::read` twin: a refcounted whole-file read-only mapping that reports *which* step (open vs map) failed, because the four Rust call sites wrap those two failures differently. `gguf` is a line-by-line port of `gguf.rs` (a bounds-checked little-endian cursor, `GgufValue` as a `std::variant` of Rust's 16 alternatives, `parse_header`, io's own dequant wrappers over `sapient::core::dequant`'s block functions, `quantize_to_q8_0`, `MmapBuffer`, `make_tensor(_mmap)` and the five `GgufLoader` entry points). `safetensors` decodes the JSON header with nlohmann/json, as strictly as serde did. `io.hpp` is `lib.rs`: re-exports plus `load_gguf`/`load_safetensors`.

**Tech Stack:** C++20, CMake presets from sub-project 0, GoogleTest (incl. death tests), `sapient::core` (plan A: `Tensor`, `Buffer`, `CpuBuffer`, `Error`/`Result`, `f16.hpp`, `dequant.hpp`, `panic.hpp`), POSIX `open`/`fstat`/`read`/`mmap` and Win32 `CreateFileW`/`ReadFile`/`CreateFileMappingW`/`MapViewOfFile`, **nlohmann/json v3.11.3 (new pin, MIT)**. Test-only: `sapient::backends_cpu` (plan D's `quantize_q8_0_block`, the differential oracle for the requantiser).

**Spec:** `docs/superpowers/specs/2026-09-21-cpp-sp1a-core-io-cpu-design.md` (§2.2 entirely, §3 rules 1/4/5/8, §4, §5 row B, §6) under `docs/superpowers/specs/2026-09-20-cpp-rewrite-design.md`. **Porting map — read Part B (§B1–§B8) and §0 before touching a file:** `docs/superpowers/notes/2026-09-21-sp1a-porting-map-core-io.md`. **Rust source (read all of it, it is short):** `crates/sapient-io/src/gguf.rs` (935 lines), `crates/sapient-io/src/safetensors.rs` (149), `crates/sapient-io/src/lib.rs` (43).

## Global Constraints

- **Branch:** `feat/cpp-sp1a` (stacked on `feat/cpp-sp0-scaffold`; plan E is complete at 52bc163). Worktree `.claude/worktrees/feat-cpp-sp0-scaffold`. Commit after every task. Every commit message ends with a blank line followed by exactly this trailer — **copy it byte-for-byte**:
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`
  **Never push** — the user pushes and opens PRs.
- **Compiler/flags (programme spec D1 + spec §3.1):** Clang only; `-ffp-contract=off -fno-fast-math -fno-math-errno` are applied by `cpp/libs/CMakeLists.txt` at directory scope (so the new `sapient-io` subdirectory inherits them automatically); never add `-march=native`/`-ffast-math`. Build and test with `cd cpp && cmake --preset dev && cmake --build --preset dev && ctest --preset dev`. **Never run ctest from the repo root** (it writes a stray `Testing/` directory).
- **Every task's verification step is the FULL `ctest --preset dev`**, never a filtered subset — the suite is 257 registered / 256 run at the start of this plan and every one of them must stay green.
- **Error-text parity rule (the core of this plan).** Every `Result`-path error text is parity-bound and must be byte-identical to the Rust oracle — *including* the Rust-standard-library pieces embedded inside them, because those are deterministic and reproducible:
  - `read_exact` running out of bytes → `failed to fill whole buffer` (`rust_std::READ_EXACT_EOF`);
  - invalid UTF-8 → `invalid utf-8 sequence of {n} bytes from index {i}` or `incomplete utf-8 byte sequence from index {i}` (`rust_std::utf8_error`, a port of Rust's validator — the `n` is how many bytes Rust consumed before rejecting, not the sequence width);
  - an OS failure → `{strerror text} (os error {code})` (`rust_std::os_error_message`);
  - `GgmlType`'s `Debug` names (`Q4_1`, `Q2_K`, `BF16`, …) and `DType`'s lowercase `Display` names (`i32`, `bool`, …).
  - Every expected literal in this plan was **generated from the real Rust code** by a throwaway probe crate on this Mac (2026-09-23), not recalled. The version error contains a **U+2013 en dash** — write it as `"\xE2\x80\x93"` in C++ source, never paste the character.
  - **Exempt (not parity-bound):** the text *after* the SAPIENT-authored prefix when Rust embedded a `serde_json` error (C++ embeds its own/nlohmann's text there — keep the prefix byte-identical), OS error descriptions on **Windows** (FormatMessage wording), and every `sapient::core::panic()` text (plan A ruling: panic texts are not parity-bound, only the fact that both trees abort on the same input).
- **Four different wrappings of one open+map failure** — `MappedFile::open` must report *which* step failed (`MapStage::Open` vs `MapStage::Map`) so each caller can wrap it the Rust way:

  | Rust call site | open fails | map fails |
  |---|---|---|
  | `GgufLoader::load_tensors_mmap` | `Model not found at path '{path}: {os}'` | `GGUF parse error: mmap failed: {os}` |
  | `GgufLoader::parse_metadata_only` | `Model not found at path '{path}: {os}'` | `GGUF parse error: mmap failed for header read: {os}` |
  | `SafetensorsLoader::load` | `Model not found at path '{path}: {os}'` | `Safetensors parse error: {os}` (no prefix) |
  | `GgufLoader::load_tensors_with_metadata` (`std::fs::read`) | `Model not found at path '{path}: {os}'` — **for open AND read failures** | — |

  `{path}` is `display_path(path)` (the `Path::display()` twin — `path::u8string()`, because `path::string()` can throw on Windows).
- **Mirror memmap2 0.9.11, verified from its source.** On POSIX, `Mmap::map` maps `max(len, 1)` bytes with `PROT_READ, MAP_SHARED` and reports `len` — so an **empty regular file maps successfully to an empty span** (and then fails in the *parser* with the parser's own error), while a **directory opens fine and fails at `mmap`** (macOS: `Invalid argument (os error 22)`; Linux: `No such device (os error 19)` — never hard-code the code in a test). **Do not add a POSIX "size == 0 → skip mmap" shortcut** — it would change nothing observable today, but it is a deviation for no reason. `CreateFileMappingW` rejects zero-length files on Windows, so the Windows arm special-cases `len == 0` → an empty mapping (that is what memmap2's Windows arm does too). The spec's §2.2 row says `MAP_PRIVATE`; memmap2 uses `MAP_SHARED`; for a read-only mapping the two are indistinguishable — follow memmap2 and record the as-built note in Task 5.
- **Rust panics become `sapient::core::panic()` (spec §3 rule 8) — do not invent new `Result` texts for them.** The malformed-input panics this crate has, and their C++ twins:
  - `general.alignment == 0` → Rust `u64::div_ceil(0)` panics → `core::panic("attempt to divide by zero")`;
  - a GGUF tensor range whose `start + byte_len` wraps past `usize::MAX` but lands `<=` the file size → Rust's bounds check passes and `&bytes[start..end]` panics → `core::panic("slice index starts at {start} but ends at {end}")`;
  - safetensors `data_offsets` with `start > end` (and `end` in range) → slice panic → same panic text;
  - a safetensors `header_len` so large that `8 + header_len` wraps below 8 → slice panic → same panic text;
  - a `dequantize_*` wrapper handed fewer bytes than its block count needs (unreachable from the loaders — `make_tensor` always passes exactly `tensor_byte_len` bytes — but reachable from the `gguf::detail` test API) → `core::panic("index out of bounds")`, never a C++ out-of-bounds read.
- **One carve-out — never allocate from an untrusted count or length.** Rust's `HashMap::with_capacity(kv_count)`, `Vec::with_capacity(tensor_count)`, `Vec::with_capacity(n_dims)` and `vec![0u8; len]` for a string all *abort the process* on an absurd header value (capacity overflow / allocation failure). In C++ the same `reserve` would throw `std::bad_alloc`/`std::length_error`, and exceptions must not cross the library boundary. So: **no `reserve` sized from a header field**, and `read_gguf_string` checks `len` against the remaining bytes **before** allocating. Result: for every input where Rust returns an error, C++ returns the byte-identical error; for inputs where Rust aborts on allocation, C++ returns Rust's own EOF error instead. Recorded as a deviation in `docs/PARITY.md` (Task 5). Likewise an array whose item type consumes no bytes (item type 9 or ≥ 13) must not loop `count` times doing nothing — skip the loop (behaviour-identical: zero bytes consumed either way; a `count` of 2⁶⁰ would otherwise hang).
- **Reproduce these Rust quirks faithfully** (each gets a pinning test and a Task-5 `PARITY.md` row):
  - a metadata value type ≥ 13 decodes to `Other` and **consumes zero bytes** (cursor desync if a real payload follows);
  - `skip_value` of an unknown or nested-array (9) item type consumes zero bytes;
  - arrays of any item type other than 4 (u32), 6 (f32), 8 (string) decode to `Other` (an `i32` array such as `tokenizer.ggml.token_type` is `Other`);
  - GGUF v1 is accepted and parsed with v2/v3 widths (u64 counts);
  - `general.alignment` stored as `I32` (or anything but `U32`/`U64`) is **ignored** — the default 32 is used;
  - release-profile integer arithmetic wraps (Rust's `[profile.release]` has no overflow checks): `dims` product, `numel * type_size`, `data_start + offset`, `start + byte_len`, `8 + header_len` are all `size_t` wrapping arithmetic — but a wrapped range must never become an out-of-range `subspan` (see the panic list above);
  - `numel = product(dims).max(1)`; a 0-dim tensor gets `Shape{1}`; a zero *dimension* fails `Shape::validate` inside the Tensor constructor → `GGUF parse error: Graph validation failed: Shape has zero dimension at axis {i}`;
  - duplicate metadata keys, duplicate tensor names and duplicate safetensors JSON keys: **last one wins** — use `insert_or_assign`, never `emplace`/`insert` (which keep the first).
- **dequant: io keeps its own semantics over core's block functions.** The Rust io crate carries its own dequantisers whose *whole-tensor* behaviour differs from `sapient::core::dequant::q*` (which panics when `bytes` holds more blocks than `numel`): io's Q4_0/Q8_0 run `bytes.size() / block_bytes` blocks and **silently skip writes past `numel`**; io's K-quants run **`numel / 256`** blocks. So `gguf::detail::dequantize_q*` call core's **per-block** functions (`dequant::q4_0_block`, …) and never the whole-tensor ones. Q5_0 has no core twin and is transcribed locally. **Q5_K uses core's correct per-element form, not io's stale `qh[is/8]` copy** (spec §2.1: "the stale, unreachable Q5_K copy in `sapient-io` is not ported"; `GgmlType::Q5_K` is kept as blocks, so neither loader ever reaches it) — Task-5 `PARITY.md` row.
- **`quantize_to_q8_0` is local to io** (io must not link `sapient::backends_cpu`). `std::fmax` for the NaN-dropping max (Rust `f32::max`), `std::roundf` (half away from zero), then clamp to ±127, then a **NaN → 0** cast (Rust `as i8` saturates NaN to 0; an `inf` input gives `d = inf`, `id = 0`, `inf · 0 = NaN`, and a C++ `static_cast<int8_t>(NaN)` is undefined behaviour). `chunks_exact(32)`: a trailing partial chunk is dropped.
- **Buffer alignment is observable** (`Tensor::buffer().alignment()`), so the constructor choice is load-bearing: GGUF float results use `Tensor::from_f32(span)` (align 64) — **never `from_f32_vec`** (align 4); GGUF kept-type heap tensors use `Tensor::from_quant_bytes` (align 16); `MmapBuffer::alignment()` is the hard-coded `32`; the safetensors F32 copy-once path must also be align 64 and keep `from_f32`'s check order (validate, then count, then compare).
- **Little-endian only** (Rust has no BE branch): every `.cpp` that `memcpy`s LE integers/floats starts with `static_assert(std::endian::native == std::endian::little, "sapient::io assumes a little-endian host (as the Rust crate does)");`.
- **No exceptions across library boundaries.** The only `try`/`catch` in this plan is around `nlohmann::json::parse` in `safetensors.cpp`, caught locally and converted to a `Result`. `std::filesystem` calls in the **library** use no throwing overloads (only `path::c_str()`/`u8string()`); tests use the `std::error_code` overloads throughout (plan E lesson: a throwing overload is one missing directory away from `std::terminate`).
- **Windows really runs ctest** (`cpp-build-windows` runs `ctest --preset ci-windows` — plan E's "compile-only" wording was wrong). The Windows mapping and file-read arms must be real implementations, and every test must pass there: temp dirs unique per process (pid in the name), every `Tensor` and `MappedFile` destroyed **before** its `TempDir` (declare the `TempDir` first so it is destroyed last — Windows cannot delete a mapped file), POSIX-only expectations (`EISDIR`, directory mapping) behind `#if !defined(_WIN32)`, and OS texts compared via `rust_std::os_error_message(code)` rather than literals except inside `#if !defined(_WIN32)` blocks. Code 2 is both `ENOENT` and `ERROR_FILE_NOT_FOUND`, so the missing-file tests are portable when written against `os_error_message(2)`.
- **Warnings are errors** (`-Wall -Wextra -Wpedantic -Wshadow -Werror`); explicit `static_cast` for every narrowing (`bugprone-narrowing-conversions` is part of the CI clang-tidy gate).
- **CI clang-tidy gate** (`.clang-tidy`, `WarningsAsErrors: '*'`, runs in CI only; `just cpp-tidy`): group identical `switch` arms (`bugprone-branch-clone`), take non-trivially-copyable parameters you only read by `const&` and move the ones you store (`performance-unnecessary-value-param`), no `std::move` of a `const` (`performance-move-const-arg`), no dead `using` declarations (`misc-unused-using-decls`), and wrap any function-like macro in `NOLINTBEGIN(bugprone-macro-parentheses)`.
- **Cross-compile every new TU — library AND test — for x86_64** (plan D's rule). The shapes of the commands (run from `cpp/` after a `dev` build so the FetchContent sources exist; `DEPS=build/dev/_deps`):
  - library TU: `clang++ -std=c++20 --target=x86_64-apple-macos -Wall -Wextra -Wpedantic -Wshadow -Werror -ffp-contract=off -fno-math-errno -Ilibs/sapient-core/include -Ilibs/sapient-io/include -isystem $DEPS/tl_expected-src/include -isystem "$(dirname "$(dirname "$(find $DEPS -path '*nlohmann/json.hpp' | head -1)")")" -c libs/sapient-io/src/<file>.cpp -o /dev/null`
  - test TU: the same plus `-Ilibs/sapient-backends-cpu/include -Ilibs/sapient-io/tests -isystem $DEPS/googletest-src/googletest/include` (gtest's includes MUST be `-isystem`, not `-I` — with `-I` they trip `-Werror,-Wcharacter-conversion` in `gtest-printers.h`; plan E lesson).
- **Formatting:** run the CI-pinned clang-format 18 (`.superpowers/tools-venv/bin/clang-format -i`, or `just cpp-fmt`) on every new/changed `.hpp/.cpp` **after `git add`**, then re-add.
- **Naming (spec D3):** target `sapient_io`, alias `sapient::io`; headers `include/sapient/io/{rust_std,mmap,gguf,safetensors,io}.hpp`; sources `src/{rust_std,mmap,gguf,safetensors,io}.cpp`; tests `tests/{rust_std,mmap,gguf,safetensors,real_file}_test.cpp` + the header-only helper `tests/io_test_util.hpp`; test binary `sapient_io_tests`. Namespaces: `sapient::io` (crate: `MappedFile`, `read_file`, `display_path`, `load_gguf`, `load_safetensors`, and the `lib.rs` re-exports), `sapient::io::rust_std`, `sapient::io::gguf` (+ `gguf::detail` for Rust-private items the ported tests call), `sapient::io::safetensors`. The two Rust test names are kept verbatim under gtest suite `Gguf`.
- **SPDX header verbatim** on every new `.hpp/.cpp` (`//` form) and on `CMakeLists.txt`/`deps.cmake` edits (`#` form):
  `// SPDX-License-Identifier: AGPL-3.0-only`
  `// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)`
- **Rust tree frozen — plan B makes no Rust change of any kind** (spec §4: "B — none"; io is gated by 1b's greedy parity; 1a gives it synthetic-file tests and the env-gated real-file check). `cargo test -p sapient-io` must stay `2 passed`; `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets -- -D warnings` stay clean.
- **Docs rule:** Task 5 updates CLAUDE.md, docs/ROADMAP.md, docs/PARITY.md, docs/PROJECT_GUIDE.md (incl. the §6 note the spec asks for), CHANGELOG.md and the spec's §2.2/§5 "as built" notes. CONTRIBUTING/README need no change for a library-internal plan; say so in the commit. `cpp/third_party/LICENSES.md` gains the nlohmann/json row in Task 4.

## Review Focus

The five input classes the spec implies but no ported Rust test exercises, most likely to bite a person first — each pinned by a test in the task named:

1. **An empty file** — every entry point fails with the *parser's* error, not an OS error: `GGUF parse error: failed to fill whole buffer` from all three GGUF entry points (heap, mmap, metadata-only), `Safetensors parse error: file too short` from `SafetensorsLoader::load`. Pinned by `Mmap.empty_file_maps_to_empty_span` (Task 1), `GgufLoader.empty_file_is_eof_on_every_entry_point` (Task 3), `Safetensors.empty_file_is_too_short` (Task 4).
2. **A missing file or a directory** — the exact per-entry-point wrapping from the table above. Pinned by `Mmap.missing_file_fails_at_open_stage`, `Mmap.directory_fails_at_map_stage`, `ReadFile.directory_fails_with_eisdir` (Task 1), `GgufLoader.missing_file_is_model_not_found_everywhere`, `GgufLoader.directory_wrapping_per_entry_point` (Task 3), `Safetensors.missing_file_and_directory` (Task 4).
3. **A truncated or corrupt header** — EOF anywhere, a 2⁶² string length, absurd counts, invalid UTF-8: always a `Result` with Rust's text, never an allocation abort, never a hang. Pinned by `GgufHeader.truncation_at_every_prefix_is_eof`, `GgufHeader.huge_lengths_and_counts_are_eof_not_allocation`, `GgufHeader.invalid_utf8_key_uses_rust_text` (Task 2), `Safetensors.header_errors_match_rust` (Task 4).
4. **A zero-copy tensor outliving its loader** — an mmap-backed tensor stays readable after every loader-local handle is gone, and mutating it panics with Rust's text. Pinned by `GgufLoader.mmap_tensor_outlives_the_loader` and `GgufLoader.mmap_buffer_is_read_only` (Task 3).
5. **The per-dtype fate of a tensor** — GGUF F16/BF16 become F32 while safetensors keeps them raw; Q5_0 becomes Q8_0 only when `numel % 32 == 0` (else F32, with a zero tail); a zero dimension gives the wrapped InvalidGraph text; a 0-dim tensor gets `Shape{1}`; the five kept types are `is_mmap()` on the mmap path and nothing else is. Pinned by `GgufTensors.every_route_materialises_the_fixture` and `GgufTensors.zero_dimension_is_wrapped_invalid_graph` (Task 3), `Safetensors.half_types_stay_raw` (Task 4).

## Execution notes (model selection)

- **Task 1 (target + rust_std + mmap):** standard-tier implementer, mid-tier reviewer. The UTF-8 validator is transcribed code; the reviewer diffs its branches against the Rust validator listing in the task.
- **Task 2 (GGUF header):** standard-tier implementer, mid-tier reviewer.
- **Task 3 (GGUF tensors + loaders + real-file check):** **mid-tier implementer, most-capable-tier reviewer.** It is the plan's parity core. The reviewer's brief must say: *diff every error text against the table in Global Constraints and against `gguf.rs`, confirm no `reserve` is sized from a header field, confirm `from_f32` (not `from_f32_vec`) everywhere a float tensor is built, and confirm the dequant wrappers use core's per-block functions with io's block counts.*
- **Task 4 (safetensors + io.hpp + nlohmann pin):** standard-tier implementer, mid-tier reviewer.
- **Task 5 (docs + PARITY):** cheap-tier implementer, cheap-tier reviewer.

## File structure

```
cpp/cmake/deps.cmake                                        (Task 4: nlohmann/json pin)
cpp/third_party/LICENSES.md                                 (Task 4: nlohmann row moves from Planned to Pinned)
cpp/libs/CMakeLists.txt                                     (Task 1: add_subdirectory(sapient-io))
cpp/libs/sapient-io/CMakeLists.txt                          (Task 1: new; Tasks 2-4 add sources/tests)
cpp/libs/sapient-io/include/sapient/io/rust_std.hpp         (Task 1)
cpp/libs/sapient-io/src/rust_std.cpp                        (Task 1)
cpp/libs/sapient-io/include/sapient/io/mmap.hpp             (Task 1)
cpp/libs/sapient-io/src/mmap.cpp                            (Task 1)
cpp/libs/sapient-io/tests/io_test_util.hpp                  (Task 1: TempDir; Task 2 adds GgufBuilder)
cpp/libs/sapient-io/tests/rust_std_test.cpp                 (Task 1)
cpp/libs/sapient-io/tests/mmap_test.cpp                     (Task 1)
cpp/libs/sapient-io/include/sapient/io/gguf.hpp             (Task 2: header parsing; Task 3: tensors + loaders)
cpp/libs/sapient-io/src/gguf.cpp                            (Task 2; Task 3 appends)
cpp/libs/sapient-io/tests/gguf_test.cpp                     (Task 2: header tests; Task 3 appends)
cpp/libs/sapient-io/tests/real_file_test.cpp                (Task 3: GGUF; Task 4 appends safetensors)
cpp/libs/sapient-io/include/sapient/io/safetensors.hpp      (Task 4)
cpp/libs/sapient-io/src/safetensors.cpp                     (Task 4)
cpp/libs/sapient-io/include/sapient/io/io.hpp               (Task 4)
cpp/libs/sapient-io/src/io.cpp                              (Task 4)
cpp/libs/sapient-io/tests/safetensors_test.cpp              (Task 4)
.github/workflows/ci.yml                                    (Task 3: real-file step in cpp-parity)
docs: CLAUDE.md, docs/ROADMAP.md, docs/PARITY.md, docs/PROJECT_GUIDE.md, CHANGELOG.md, spec §2.2/§5 notes (Task 5)
```

Nothing under `crates/` is touched by any task.

---

### Task 1: the `sapient::io` target, `rust_std` text twins, `mmap` (mapping + whole-file read)

**Files:**
- Create: `cpp/libs/sapient-io/CMakeLists.txt`, `cpp/libs/sapient-io/include/sapient/io/rust_std.hpp`, `cpp/libs/sapient-io/src/rust_std.cpp`, `cpp/libs/sapient-io/include/sapient/io/mmap.hpp`, `cpp/libs/sapient-io/src/mmap.cpp`, `cpp/libs/sapient-io/tests/io_test_util.hpp`, `cpp/libs/sapient-io/tests/rust_std_test.cpp`, `cpp/libs/sapient-io/tests/mmap_test.cpp`
- Modify: `cpp/libs/CMakeLists.txt` (add `add_subdirectory(sapient-io)` as the LAST line, after `add_subdirectory(sapient-backends-cpu)` — so the `sapient::backends_cpu` alias already exists when Task 3's test target links it; the io library itself never depends on backends-cpu)

**Interfaces:**
- Consumes: `sapient::core` (only for the target link; no core types are used by these two modules).
- Produces (namespace `sapient::io::rust_std`): `inline constexpr std::string_view READ_EXACT_EOF`; `std::optional<std::string> utf8_error(std::span<const uint8_t>)`; `std::string os_error_message(int code)`.
- Produces (namespace `sapient::io`): `struct OsError { int code; std::string message; }`; `enum class MapStage { Open, Map }`; `struct MapError { MapStage stage; OsError os; }`; `class MappedFile` with `static tl::expected<std::shared_ptr<const MappedFile>, MapError> open(const std::filesystem::path&)`, `std::span<const uint8_t> bytes() const`, `size_t size() const`; `tl::expected<std::vector<uint8_t>, OsError> read_file(const std::filesystem::path&)`; `std::string display_path(const std::filesystem::path&)`.
- Produces (tests): `tests/io_test_util.hpp` → `namespace sapient::io::test { class TempDir; }` with `explicit TempDir(std::string_view name)`, `const std::filesystem::path& path() const`, `std::filesystem::path write(std::string_view file, std::span<const uint8_t> bytes) const`.

Rust references: `std::io::Read::read_exact`'s EOF error; `core::str::from_utf8`'s `Utf8Error` (the validator below is `core/src/str/validations.rs::run_utf8_validation`, transcribed); `std::io::Error`'s Display for OS errors; memmap2 0.9.11 `src/unix.rs` (`MmapInner::new`, `adjust_mmap_params`: `map_len.max(1)`) and `Mmap::map` (`PROT_READ`, `MAP_SHARED`); `std::fs::read` (open, `fstat` size hint, read-to-end with `EINTR` retry).

- [ ] **Step 1: Write the CMake target and the failing tests**

`cpp/libs/sapient-io/CMakeLists.txt`:

```cmake
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
# Port of crates/sapient-io (sub-project 1a plan B). `mmap` stands in for memmap2 + std::fs::read,
# `rust_std` for the Rust-std error texts the loaders embed. ONNX and the Graph entry points are
# sub-project 8.
add_library(sapient_io STATIC src/rust_std.cpp src/mmap.cpp)
add_library(sapient::io ALIAS sapient_io)
target_include_directories(sapient_io PUBLIC include)
target_link_libraries(sapient_io PUBLIC sapient::core)
sapient_apply_warnings(sapient_io)

if(SAPIENT_BUILD_TESTS)
  add_executable(sapient_io_tests tests/rust_std_test.cpp tests/mmap_test.cpp)
  target_include_directories(sapient_io_tests PRIVATE tests)
  target_link_libraries(sapient_io_tests PRIVATE sapient::io GTest::gtest_main)
  sapient_apply_warnings(sapient_io_tests)
  gtest_discover_tests(sapient_io_tests)
endif()
```

In `cpp/libs/CMakeLists.txt`, append `add_subdirectory(sapient-io)` after `add_subdirectory(sapient-backends-cpu)`.

`cpp/libs/sapient-io/tests/io_test_util.hpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Test helpers for sapient_io_tests. std::filesystem calls use the std::error_code overloads only
// (a throwing overload is one missing directory away from std::terminate — plan E lesson).
// Windows cannot delete a file that is still mapped: declare the TempDir BEFORE any Tensor or
// MappedFile that points into it, so it is destroyed last.

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <system_error>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace sapient::io::test {

inline int current_pid() {
#if defined(_WIN32)
    return _getpid();
#else
    return static_cast<int>(::getpid());
#endif
}

/// A fresh directory unique to this process + name + instance (ctest runs each gtest in its own
/// process, possibly in parallel). Removed, best-effort, on destruction.
class TempDir {
public:
    explicit TempDir(std::string_view name) {
        static std::atomic<unsigned> counter{0};
        std::error_code ec;
        dir_ = std::filesystem::temp_directory_path(ec) /
               ("sapient-io-test-" + std::to_string(current_pid()) + "-" + std::string(name) + "-" +
                std::to_string(counter.fetch_add(1)));
        std::filesystem::remove_all(dir_, ec);
        ec.clear();
        std::filesystem::create_directories(dir_, ec);
        EXPECT_FALSE(ec) << "create_directories " << dir_.string() << ": " << ec.message();
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    const std::filesystem::path& path() const { return dir_; }

    std::filesystem::path write(std::string_view file, std::span<const uint8_t> bytes) const {
        const std::filesystem::path p = dir_ / std::string(file);
        std::ofstream f(p, std::ios::binary | std::ios::trunc);
        EXPECT_TRUE(f.is_open()) << "open " << p.string();
        f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        EXPECT_TRUE(f.good()) << "write " << p.string();
        return p;
    }

private:
    std::filesystem::path dir_;
};

} // namespace sapient::io::test
```

`cpp/libs/sapient-io/tests/rust_std_test.cpp` — every literal below was printed by the real Rust `String::from_utf8(..).unwrap_err()` / `io::Error` on this Mac:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
// The Rust-std text twins. Expected strings are verbatim Rust output (probe crate, 2026-09-23).
#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "sapient/io/rust_std.hpp"

namespace rust_std = sapient::io::rust_std;

namespace {
std::optional<std::string> utf8(std::vector<uint8_t> v) {
    return rust_std::utf8_error(v);
}
} // namespace

TEST(RustStd, read_exact_eof_text) {
    EXPECT_EQ(rust_std::READ_EXACT_EOF, "failed to fill whole buffer");
}

TEST(RustStd, utf8_error_matches_rust_display) {
    EXPECT_EQ(utf8({0xFF}), "invalid utf-8 sequence of 1 bytes from index 0");
    EXPECT_EQ(utf8({'a', 0xE2, 0x82}), "incomplete utf-8 byte sequence from index 1");
    EXPECT_EQ(utf8({0xE0, 0x80}), "invalid utf-8 sequence of 1 bytes from index 0");
    EXPECT_EQ(utf8({0xF0, 0x90, 0x41}), "invalid utf-8 sequence of 2 bytes from index 0");
    EXPECT_EQ(utf8({0xF0, 0x90, 0x80, 0x41}), "invalid utf-8 sequence of 3 bytes from index 0");
    EXPECT_EQ(utf8({0xED, 0xA0, 0x80}), "invalid utf-8 sequence of 1 bytes from index 0"); // surrogate
    EXPECT_EQ(utf8({0xC0, 0x80}), "invalid utf-8 sequence of 1 bytes from index 0");       // overlong
    EXPECT_EQ(utf8({'o', 'k', 0xF4, 0x90, 0x80, 0x80}),
              "invalid utf-8 sequence of 1 bytes from index 2"); // > U+10FFFF
    EXPECT_EQ(utf8({0xF0, 0x9F, 0x98}), "incomplete utf-8 byte sequence from index 0");
}

TEST(RustStd, utf8_error_accepts_valid_text) {
    EXPECT_EQ(utf8({}), std::nullopt);
    EXPECT_EQ(utf8({'a', 'b', 'c'}), std::nullopt);
    EXPECT_EQ(utf8({0xC3, 0xA9}), std::nullopt);             // é
    EXPECT_EQ(utf8({0xE2, 0x82, 0xAC}), std::nullopt);       // €
    EXPECT_EQ(utf8({0xF0, 0x90, 0x8D, 0x88}), std::nullopt); // 𐍈
    EXPECT_EQ(utf8({0xEF, 0xBF, 0xBF}), std::nullopt);       // U+FFFF
    EXPECT_EQ(utf8({0xF4, 0x8F, 0xBF, 0xBF}), std::nullopt); // U+10FFFF
}

TEST(RustStd, os_error_message_format) {
    // Portable: the format is "<system_category text> (os error <code>)" on every OS.
    const std::string m = rust_std::os_error_message(2);
    EXPECT_TRUE(m.ends_with(" (os error 2)")) << m;
#if !defined(_WIN32)
    EXPECT_EQ(rust_std::os_error_message(2), "No such file or directory (os error 2)");
    EXPECT_EQ(rust_std::os_error_message(21), "Is a directory (os error 21)");
    EXPECT_EQ(rust_std::os_error_message(22), "Invalid argument (os error 22)");
#endif
}
```

`cpp/libs/sapient-io/tests/mmap_test.cpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
// MappedFile / read_file / display_path — the memmap2 + std::fs::read twins.
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "io_test_util.hpp"
#include "sapient/io/mmap.hpp"
#include "sapient/io/rust_std.hpp"

using sapient::io::MappedFile;
using sapient::io::MapStage;
using sapient::io::test::TempDir;

namespace {
std::vector<uint8_t> pattern(size_t n) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i)
        v[i] = static_cast<uint8_t>((i * 7 + 3) & 0xFF);
    return v;
}
} // namespace

TEST(Mmap, maps_whole_file_read_only) {
    TempDir dir("maps_whole_file");
    const auto data = pattern(1000);
    const auto p = dir.write("f.bin", data);
    auto m = MappedFile::open(p);
    ASSERT_TRUE(m.has_value()) << m.error().os.message;
    ASSERT_EQ((*m)->size(), data.size());
    EXPECT_TRUE(std::equal(data.begin(), data.end(), (*m)->bytes().begin()));
}

TEST(Mmap, empty_file_maps_to_empty_span) {
    // memmap2 0.9.11 maps max(len,1) bytes and reports len: an empty file is NOT an error here —
    // it fails later, in the parser, with the parser's own text (Review Focus 1).
    TempDir dir("empty_file");
    const auto p = dir.write("empty.bin", {});
    auto m = MappedFile::open(p);
    ASSERT_TRUE(m.has_value()) << m.error().os.message;
    EXPECT_EQ((*m)->size(), 0u);
    EXPECT_TRUE((*m)->bytes().empty());
}

TEST(Mmap, missing_file_fails_at_open_stage) {
    TempDir dir("missing_file");
    auto m = MappedFile::open(dir.path() / "missing.gguf");
    ASSERT_FALSE(m.has_value());
    EXPECT_EQ(m.error().stage, MapStage::Open);
    EXPECT_EQ(m.error().os.code, 2); // ENOENT == ERROR_FILE_NOT_FOUND == 2
    EXPECT_EQ(m.error().os.message, sapient::io::rust_std::os_error_message(2));
}

TEST(Mmap, directory_fails_at_map_stage) {
    TempDir dir("directory");
    auto m = MappedFile::open(dir.path());
    ASSERT_FALSE(m.has_value());
#if defined(_WIN32)
    EXPECT_EQ(m.error().stage, MapStage::Open); // CreateFileW refuses a directory handle
#else
    // POSIX open(2) accepts a directory; mmap(2) then fails (macOS EINVAL, Linux ENODEV).
    EXPECT_EQ(m.error().stage, MapStage::Map);
#endif
    EXPECT_EQ(m.error().os.message, sapient::io::rust_std::os_error_message(m.error().os.code));
}

TEST(Mmap, mapping_outlives_every_other_handle) {
    TempDir dir("outlives");
    const auto data = pattern(4096 + 17);
    std::shared_ptr<const MappedFile> keep;
    {
        auto m = MappedFile::open(dir.write("f.bin", data));
        ASSERT_TRUE(m.has_value());
        keep = *m;
    }
    EXPECT_TRUE(std::equal(data.begin(), data.end(), keep->bytes().begin()));
    keep.reset(); // unmapped before ~TempDir (Windows cannot delete a mapped file)
}

TEST(ReadFile, reads_whole_file) {
    TempDir dir("read_whole");
    const auto data = pattern(200000); // larger than one 64 KiB read chunk
    auto r = sapient::io::read_file(dir.write("f.bin", data));
    ASSERT_TRUE(r.has_value()) << r.error().message;
    EXPECT_EQ(*r, data);
}

TEST(ReadFile, empty_file_reads_empty) {
    TempDir dir("read_empty");
    auto r = sapient::io::read_file(dir.write("e.bin", {}));
    ASSERT_TRUE(r.has_value()) << r.error().message;
    EXPECT_TRUE(r->empty());
}

TEST(ReadFile, missing_file_is_code_2) {
    TempDir dir("read_missing");
    auto r = sapient::io::read_file(dir.path() / "nope.bin");
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, 2);
    EXPECT_EQ(r.error().message, sapient::io::rust_std::os_error_message(2));
}

#if !defined(_WIN32)
TEST(ReadFile, directory_fails_with_eisdir) {
    // Rust std::fs::read: open(2) succeeds on a directory, read(2) fails with EISDIR (21).
    TempDir dir("read_dir");
    auto r = sapient::io::read_file(dir.path());
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, 21);
    EXPECT_EQ(r.error().message, "Is a directory (os error 21)");
}
#endif

TEST(DisplayPath, is_utf8) {
    const std::filesystem::path p =
        std::filesystem::path(u8"models") / std::filesystem::path(u8"héllo.gguf");
    const std::string d = sapient::io::display_path(p);
    EXPECT_TRUE(d.ends_with("h\xC3\xA9llo.gguf")) << d;
}
```

- [ ] **Step 2: Run the build to verify it fails**

Run: `cd cpp && cmake --preset dev && cmake --build --preset dev`
Expected: FAIL — `sapient/io/rust_std.hpp: file not found` (and the two `.cpp` sources missing).

- [ ] **Step 3: Write `rust_std`**

`cpp/libs/sapient-io/include/sapient/io/rust_std.hpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Twins of the Rust standard-library error texts that sapient-io embeds in its Result-path
// errors (and that are therefore parity-bound): `read_exact`'s EOF error, `Utf8Error`'s Display,
// and `io::Error`'s Display for an OS error.

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace sapient::io::rust_std {

/// `std::io::Read::read_exact` on too few bytes (`io::Error::READ_EXACT_EOF`).
inline constexpr std::string_view READ_EXACT_EOF = "failed to fill whole buffer";

/// `std::str::from_utf8(bytes)`: nullopt when valid, else `Utf8Error`'s Display text —
/// "invalid utf-8 sequence of {n} bytes from index {i}" / "incomplete utf-8 byte sequence from
/// index {i}". A transcription of core::str::validations::run_utf8_validation.
std::optional<std::string> utf8_error(std::span<const uint8_t> bytes);

/// `io::Error::from_raw_os_error(code).to_string()`: "{description} (os error {code})". The
/// description is `std::system_category().message(code)` — strerror text on POSIX (identical to
/// Rust's strerror_r), FormatMessage text on Windows (not parity-bound).
std::string os_error_message(int code);

} // namespace sapient::io::rust_std
```

`cpp/libs/sapient-io/src/rust_std.cpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/io/rust_std.hpp"

#include <system_error>

namespace sapient::io::rust_std {

namespace {
// Rust: `next!() as i8 >= -64` rejects — i.e. only 0x80..=0xBF is a continuation byte.
bool is_cont(uint8_t b) {
    return (b & 0xC0u) == 0x80u;
}
// core::str::utf8_char_width.
int char_width(uint8_t first) {
    if (first < 0x80u) return 1;
    if (first >= 0xC2u && first <= 0xDFu) return 2;
    if (first >= 0xE0u && first <= 0xEFu) return 3;
    if (first >= 0xF0u && first <= 0xF4u) return 4;
    return 0; // 0x80..=0xC1 and 0xF5..=0xFF are never valid lead bytes
}
std::string invalid(size_t n, size_t from) {
    return "invalid utf-8 sequence of " + std::to_string(n) + " bytes from index " +
           std::to_string(from);
}
std::string incomplete(size_t from) {
    return "incomplete utf-8 byte sequence from index " + std::to_string(from);
}
} // namespace

std::optional<std::string> utf8_error(std::span<const uint8_t> v) {
    const size_t len = v.size();
    size_t i = 0;
    while (i < len) {
        const size_t start = i; // Rust `old_offset` = Utf8Error::valid_up_to
        const uint8_t first = v[i];
        const int w = char_width(first);
        if (w == 1) {
            ++i;
            continue;
        }
        if (w == 0) return invalid(1, start);
        // Rust `next!()`: advance; running off the end is "incomplete" (error_len None).
        if (++i >= len) return incomplete(start);
        const uint8_t second = v[i];
        if (w == 2) {
            if (!is_cont(second)) return invalid(1, start);
        } else if (w == 3) {
            const bool ok = (first == 0xE0u && second >= 0xA0u && second <= 0xBFu) ||
                            (first >= 0xE1u && first <= 0xECu && is_cont(second)) ||
                            (first == 0xEDu && second >= 0x80u && second <= 0x9Fu) ||
                            (first >= 0xEEu && first <= 0xEFu && is_cont(second));
            if (!ok) return invalid(1, start);
            if (++i >= len) return incomplete(start);
            if (!is_cont(v[i])) return invalid(2, start);
        } else { // w == 4
            const bool ok = (first == 0xF0u && second >= 0x90u && second <= 0xBFu) ||
                            (first >= 0xF1u && first <= 0xF3u && is_cont(second)) ||
                            (first == 0xF4u && second >= 0x80u && second <= 0x8Fu);
            if (!ok) return invalid(1, start);
            if (++i >= len) return incomplete(start);
            if (!is_cont(v[i])) return invalid(2, start);
            if (++i >= len) return incomplete(start);
            if (!is_cont(v[i])) return invalid(3, start);
        }
        ++i;
    }
    return std::nullopt;
}

std::string os_error_message(int code) {
    return std::system_category().message(code) + " (os error " + std::to_string(code) + ")";
}

} // namespace sapient::io::rust_std
```

- [ ] **Step 4: Write `mmap`**

`cpp/libs/sapient-io/include/sapient/io/mmap.hpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// The two file facilities sapient-io uses: `memmap2::Mmap::map(&File)` (a whole-file read-only
// mapping; memmap2 0.9.11 semantics) and `std::fs::read`. Failures carry Rust's io::Error Display
// text; MappedFile also reports WHICH step failed, because the Rust callers wrap an open failure
// and a map failure differently (plan B Global Constraints table).

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <tl/expected.hpp>

namespace sapient::io {

/// A failed OS call. `code` is errno (POSIX) or GetLastError() (Windows); `message` is exactly
/// what Rust's `io::Error` Display prints for it (`rust_std::os_error_message(code)`).
struct OsError {
    int code{0};
    std::string message;
};

/// `File::open` (Open) vs `Mmap::map` (Map — includes the fstat for the length).
enum class MapStage { Open, Map };
struct MapError {
    MapStage stage{MapStage::Open};
    OsError os;
};

/// A whole-file, read-only mapping, shared by every MmapBuffer that points into it (the twin of
/// Rust's `Arc<Mmap>`). An empty file is a valid, empty mapping (memmap2 maps max(len,1) bytes
/// on POSIX and reports len).
class MappedFile {
    struct Private {};

public:
    static tl::expected<std::shared_ptr<const MappedFile>, MapError>
    open(const std::filesystem::path& path);

    std::span<const uint8_t> bytes() const { return {static_cast<const uint8_t*>(ptr_), len_}; }
    size_t size() const { return len_; }

    explicit MappedFile(Private) {}
    ~MappedFile();
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

private:
    const void* ptr_{nullptr};
    size_t len_{0};     // Rust `Mmap::len()`
    size_t map_len_{0}; // bytes actually mapped (POSIX: max(len_, 1)); 0 = nothing to unmap
};

/// `std::fs::read(path)`: open + read to EOF. Every failure (open or read) is one OsError.
tl::expected<std::vector<uint8_t>, OsError> read_file(const std::filesystem::path& path);

/// `Path::display()` for the UTF-8 paths SAPIENT uses. Uses `u8string()` because
/// `path::string()` can throw on Windows.
std::string display_path(const std::filesystem::path& path);

} // namespace sapient::io
```

`cpp/libs/sapient-io/src/mmap.cpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/io/mmap.hpp"

#include <new>
#include <optional>
#include <stdexcept>
#include <utility>

#include "sapient/io/rust_std.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace sapient::io {

namespace {

OsError last_os_error() {
#if defined(_WIN32)
    const int code = static_cast<int>(::GetLastError());
#else
    const int code = errno;
#endif
    return OsError{code, rust_std::os_error_message(code)};
}

tl::unexpected<MapError> map_failure(MapStage stage, OsError os) {
    return tl::unexpected(MapError{stage, std::move(os)});
}

#if !defined(_WIN32)
// Rust's File::open retries open(2) on EINTR (cvt_r). O_CLOEXEC as Rust sets it.
int open_readonly(const std::filesystem::path& path) {
    int fd = -1;
    do {
        fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    } while (fd < 0 && errno == EINTR);
    return fd;
}
#endif

} // namespace

std::string display_path(const std::filesystem::path& path) {
    const std::u8string s = path.u8string();
    return {s.begin(), s.end()};
}

#if defined(_WIN32)

tl::expected<std::shared_ptr<const MappedFile>, MapError>
MappedFile::open(const std::filesystem::path& path) {
    // Rust File::open on Windows: GENERIC_READ, share read|write|delete, OPEN_EXISTING.
    HANDLE file = ::CreateFileW(path.c_str(), GENERIC_READ,
                                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return map_failure(MapStage::Open, last_os_error());
    LARGE_INTEGER size{};
    if (::GetFileSizeEx(file, &size) == 0) {
        OsError err = last_os_error();
        ::CloseHandle(file);
        return map_failure(MapStage::Map, std::move(err));
    }
    auto mf = std::make_shared<MappedFile>(Private{});
    mf->len_ = static_cast<size_t>(size.QuadPart);
    if (mf->len_ == 0) { // CreateFileMappingW rejects empty files; memmap2 returns an empty map
        ::CloseHandle(file);
        return mf;
    }
    HANDLE mapping = ::CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping == nullptr) {
        OsError err = last_os_error();
        ::CloseHandle(file);
        return map_failure(MapStage::Map, std::move(err));
    }
    void* view = ::MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    std::optional<OsError> err;
    if (view == nullptr) err = last_os_error(); // capture before CloseHandle clobbers it
    ::CloseHandle(mapping);                      // the view keeps the section alive
    ::CloseHandle(file);
    if (err) return map_failure(MapStage::Map, std::move(*err));
    mf->ptr_ = view;
    mf->map_len_ = mf->len_;
    return mf;
}

MappedFile::~MappedFile() {
    if (ptr_ != nullptr) ::UnmapViewOfFile(ptr_);
}

tl::expected<std::vector<uint8_t>, OsError> read_file(const std::filesystem::path& path) {
    HANDLE file = ::CreateFileW(path.c_str(), GENERIC_READ,
                                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return tl::unexpected(last_os_error());
    std::vector<uint8_t> out;
    std::vector<uint8_t> chunk(size_t{1} << 16);
    for (;;) {
        DWORD got = 0;
        if (::ReadFile(file, chunk.data(), static_cast<DWORD>(chunk.size()), &got, nullptr) == 0) {
            OsError err = last_os_error();
            ::CloseHandle(file);
            return tl::unexpected(std::move(err));
        }
        if (got == 0) break;
        out.insert(out.end(), chunk.begin(), chunk.begin() + got);
    }
    ::CloseHandle(file);
    return out;
}

#else // POSIX

tl::expected<std::shared_ptr<const MappedFile>, MapError>
MappedFile::open(const std::filesystem::path& path) {
    const int fd = open_readonly(path);
    if (fd < 0) return map_failure(MapStage::Open, last_os_error());
    struct stat st {};
    if (::fstat(fd, &st) != 0) { // memmap2 `file_len` — part of Mmap::map
        OsError err = last_os_error();
        ::close(fd);
        return map_failure(MapStage::Map, std::move(err));
    }
    const auto len = static_cast<size_t>(st.st_size);
    // memmap2 0.9.11 adjust_mmap_params: POSIX rejects a zero-length mmap, so map one byte and
    // keep reporting len 0 (the byte is never exposed). A directory reaches here and fails.
    const size_t map_len = len == 0 ? size_t{1} : len;
    void* p = ::mmap(nullptr, map_len, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
        OsError err = last_os_error();
        ::close(fd);
        return map_failure(MapStage::Map, std::move(err));
    }
    ::close(fd); // the mapping stays valid; Rust drops the File at the end of the loader too
    auto mf = std::make_shared<MappedFile>(Private{});
    mf->ptr_ = p;
    mf->len_ = len;
    mf->map_len_ = map_len;
    return mf;
}

MappedFile::~MappedFile() {
    if (map_len_ != 0) ::munmap(const_cast<void*>(ptr_), map_len_);
}

tl::expected<std::vector<uint8_t>, OsError> read_file(const std::filesystem::path& path) {
    const int fd = open_readonly(path);
    if (fd < 0) return tl::unexpected(last_os_error());
    std::vector<uint8_t> out;
    struct stat st {};
    if (::fstat(fd, &st) == 0 && st.st_size > 0) {
        // Rust fs::read pre-sizes from metadata. A real file's size is not an untrusted header
        // field, but a reservation that cannot be satisfied must still not throw across the
        // library boundary: report it as ENOMEM (not parity-bound; Rust reports its own OOM).
        try {
            out.reserve(static_cast<size_t>(st.st_size));
        } catch (const std::bad_alloc&) {
            ::close(fd);
            return tl::unexpected(OsError{ENOMEM, rust_std::os_error_message(ENOMEM)});
        } catch (const std::length_error&) {
            ::close(fd);
            return tl::unexpected(OsError{ENOMEM, rust_std::os_error_message(ENOMEM)});
        }
    }
    std::vector<uint8_t> chunk(size_t{1} << 16);
    for (;;) {
        const ssize_t n = ::read(fd, chunk.data(), chunk.size());
        if (n < 0) {
            if (errno == EINTR) continue; // Rust read_to_end retries Interrupted
            OsError err = last_os_error();
            ::close(fd);
            return tl::unexpected(std::move(err));
        }
        if (n == 0) break;
        out.insert(out.end(), chunk.begin(), chunk.begin() + n);
    }
    ::close(fd);
    return out;
}

#endif

} // namespace sapient::io
```

- [ ] **Step 5: Build and run the full suite**

Run: `cd cpp && cmake --preset dev && cmake --build --preset dev && ctest --preset dev`
Expected: PASS — 257 → 271 registered (`RustStd.*` ×4, `Mmap.*` ×5, `ReadFile.*` ×4, `DisplayPath.*` ×1; on Windows `ReadFile.directory_fails_with_eisdir` is compiled out), 100% of the run tests passing (the DISABLED spinpool probe stays not-run).

- [ ] **Step 6: Cross-compile the new TUs for x86_64**

From `cpp/`, with `DEPS=build/dev/_deps`, run the library-TU command from Global Constraints for `src/rust_std.cpp` and `src/mmap.cpp`, and the test-TU command for `tests/rust_std_test.cpp` and `tests/mmap_test.cpp`. Expected: all four exit 0 with no output.

- [ ] **Step 7: Format, lint the SPDX headers, commit**

```bash
git add cpp/libs/CMakeLists.txt cpp/libs/sapient-io
.superpowers/tools-venv/bin/clang-format -i $(git diff --cached --name-only -- 'cpp/*.hpp' 'cpp/*.cpp')
git add cpp/libs/sapient-io
python3 cpp/scripts/check_spdx.py cpp crates
git commit -m "$(cat <<'EOF'
cpp(io): sapient::io target — Rust-std error-text twins + memmap2/fs::read port

MappedFile reports the failing stage (open vs map) because the Rust callers wrap
the two differently; an empty file maps to an empty span (memmap2 0.9.11 maps
max(len,1) bytes), a directory opens and fails at mmap. utf8_error transcribes
core::str's validator so Utf8Error texts are byte-identical.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: GGUF header — `GgmlType`, `GgufValue`, the LE cursor, `parse_header`

**Files:**
- Create: `cpp/libs/sapient-io/include/sapient/io/gguf.hpp`, `cpp/libs/sapient-io/src/gguf.cpp`, `cpp/libs/sapient-io/tests/gguf_test.cpp`
- Modify: `cpp/libs/sapient-io/tests/io_test_util.hpp` (add `GgufBuilder` + LE helpers), `cpp/libs/sapient-io/CMakeLists.txt` (add `src/gguf.cpp` to `sapient_io`, `tests/gguf_test.cpp` to `sapient_io_tests`)

**Interfaces:**
- Consumes: `sapient::io::rust_std::{READ_EXACT_EOF, utf8_error}` (Task 1); `sapient::core::{Error, Result, DType}`, `SAPIENT_TRY`/`SAPIENT_TRY_ASSIGN` (plan A), `sapient::core::panic`.
- Produces (namespace `sapient::io::gguf`): `struct GgufOther`; `class GgufValue` (`Storage` variant, `Kind`, `kind()`, `storage()`, `get_if<T>()`, `as_u32/as_u64/as_f32/as_f64/as_bool/as_str`); `using GgufMetadata = std::unordered_map<std::string, GgufValue>`; `using TensorMap = std::unordered_map<std::string, sapient::core::Tensor>`.
- Produces (namespace `sapient::io::gguf::detail`): `GGUF_MAGIC`, `DEFAULT_ALIGNMENT`, `QK_K`; `enum class GgmlType : uint32_t`; `std::optional<GgmlType> ggml_type_from_u32(uint32_t)`; `size_t block_size(GgmlType)`; `size_t type_size(GgmlType)`; `std::optional<core::DType> to_sapient_dtype(GgmlType)`; `std::string_view debug_name(GgmlType)`; `size_t tensor_byte_len(GgmlType, size_t numel)`; `struct GgufTensorInfo { std::string name; std::vector<size_t> dims; GgmlType kind; uint64_t offset; }`; `struct ParsedHeader { GgufMetadata metadata; std::vector<GgufTensorInfo> tensor_infos; size_t data_start; }`; `core::Result<ParsedHeader> parse_header(std::span<const uint8_t>)`.
- Produces (tests): `sapient::io::test::GgufBuilder` and the `le_u16/le_f32/put_*` helpers (Task 3 builds its fixtures with them).

Rust reference: `gguf.rs:80-255` (constants, `GgmlType`, `tensor_byte_len`, `GgufValue`, `GgufTensorInfo`), `:511-578` (`parse_header`), `:780-909` (readers). Porting map §B1.

- [ ] **Step 1: Add the test-side GGUF writer to `io_test_util.hpp`**

Append inside `namespace sapient::io::test` (and add `#include <cstring>`, `#include <initializer_list>`, `#include <optional>`, `#include <utility>`, `#include <vector>` to the include block):

```cpp
// ── Little-endian byte helpers ──────────────────────────────────────────────────────────────
template <class T> void put_le(std::vector<uint8_t>& out, T v) {
    uint8_t b[sizeof(T)];
    std::memcpy(b, &v, sizeof(T)); // hosts are little-endian (sapient::io static_asserts it)
    out.insert(out.end(), b, b + sizeof(T));
}
inline void put_str(std::vector<uint8_t>& out, std::string_view s) {
    put_le<uint64_t>(out, s.size());
    out.insert(out.end(), s.begin(), s.end());
}
inline std::vector<uint8_t> le_u16(std::initializer_list<uint16_t> v) {
    std::vector<uint8_t> out;
    for (const uint16_t x : v)
        put_le(out, x);
    return out;
}
inline std::vector<uint8_t> le_f32(std::initializer_list<float> v) {
    std::vector<uint8_t> out;
    for (const float x : v)
        put_le(out, x);
    return out;
}

/// Writes GGUF bytes the way llama.cpp does: header, KVs, tensor infos, zero padding to
/// `alignment`, then each tensor's data at an `alignment`-aligned offset from data_start.
/// `alignment` here is only the WRITER's padding — tests that set `general.alignment` must set
/// this to the same value.
struct GgufBuilder {
    uint32_t magic = 0x46554747;
    uint32_t version = 3;
    size_t alignment = 32;
    std::vector<uint8_t> kv;
    uint64_t kv_count = 0;
    std::optional<uint64_t> tensor_count_override; // for the absurd-count tests
    std::optional<uint64_t> kv_count_override;

    struct T {
        std::string name;
        std::vector<uint64_t> dims;
        uint32_t kind;
        std::vector<uint8_t> data;
        std::optional<uint64_t> offset_override;
    };
    std::vector<T> tensors;

    GgufBuilder& kv_raw(std::string_view key, uint32_t vtype, std::span<const uint8_t> payload) {
        put_str(kv, key);
        put_le<uint32_t>(kv, vtype);
        kv.insert(kv.end(), payload.begin(), payload.end());
        ++kv_count;
        return *this;
    }
    template <class V> GgufBuilder& kv_scalar(std::string_view key, uint32_t vtype, V v) {
        std::vector<uint8_t> p;
        put_le(p, v);
        return kv_raw(key, vtype, p);
    }
    GgufBuilder& kv_u8(std::string_view k, uint8_t v) { return kv_scalar(k, 0, v); }
    GgufBuilder& kv_i8(std::string_view k, int8_t v) { return kv_scalar(k, 1, v); }
    GgufBuilder& kv_u16(std::string_view k, uint16_t v) { return kv_scalar(k, 2, v); }
    GgufBuilder& kv_i16(std::string_view k, int16_t v) { return kv_scalar(k, 3, v); }
    GgufBuilder& kv_u32(std::string_view k, uint32_t v) { return kv_scalar(k, 4, v); }
    GgufBuilder& kv_i32(std::string_view k, int32_t v) { return kv_scalar(k, 5, v); }
    GgufBuilder& kv_f32(std::string_view k, float v) { return kv_scalar(k, 6, v); }
    GgufBuilder& kv_bool_byte(std::string_view k, uint8_t v) { return kv_scalar(k, 7, v); }
    GgufBuilder& kv_u64(std::string_view k, uint64_t v) { return kv_scalar(k, 10, v); }
    GgufBuilder& kv_i64(std::string_view k, int64_t v) { return kv_scalar(k, 11, v); }
    GgufBuilder& kv_f64(std::string_view k, double v) { return kv_scalar(k, 12, v); }
    GgufBuilder& kv_str(std::string_view k, std::string_view v) {
        std::vector<uint8_t> p;
        put_str(p, v);
        return kv_raw(k, 8, p);
    }
    /// An array KV: item type, element count, then the raw item payload.
    GgufBuilder& kv_array(std::string_view k, uint32_t item_type, uint64_t count,
                          std::span<const uint8_t> items) {
        std::vector<uint8_t> p;
        put_le<uint32_t>(p, item_type);
        put_le<uint64_t>(p, count);
        p.insert(p.end(), items.begin(), items.end());
        return kv_raw(k, 9, p);
    }
    GgufBuilder& kv_arr_str(std::string_view k, std::initializer_list<std::string_view> v) {
        std::vector<uint8_t> items;
        for (const auto s : v)
            put_str(items, s);
        return kv_array(k, 8, v.size(), items);
    }

    GgufBuilder& tensor(std::string name, std::vector<uint64_t> dims, uint32_t kind,
                        std::vector<uint8_t> data) {
        tensors.push_back({std::move(name), std::move(dims), kind, std::move(data), std::nullopt});
        return *this;
    }

    static size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

    /// Relative data offsets: each tensor aligned, unless overridden.
    std::vector<uint64_t> offsets() const {
        std::vector<uint64_t> out;
        size_t off = 0;
        for (const auto& t : tensors) {
            off = align_up(off, alignment);
            out.push_back(t.offset_override.value_or(off));
            off += t.data.size();
        }
        return out;
    }

    /// Everything parse_header reads — no padding, no data.
    std::vector<uint8_t> header_bytes() const {
        std::vector<uint8_t> out;
        put_le<uint32_t>(out, magic);
        put_le<uint32_t>(out, version);
        put_le<uint64_t>(out, tensor_count_override.value_or(tensors.size()));
        put_le<uint64_t>(out, kv_count_override.value_or(kv_count));
        out.insert(out.end(), kv.begin(), kv.end());
        const auto offs = offsets();
        for (size_t i = 0; i < tensors.size(); ++i) {
            put_str(out, tensors[i].name);
            put_le<uint32_t>(out, static_cast<uint32_t>(tensors[i].dims.size()));
            for (const uint64_t d : tensors[i].dims)
                put_le<uint64_t>(out, d);
            put_le<uint32_t>(out, tensors[i].kind);
            put_le<uint64_t>(out, offs[i]);
        }
        return out;
    }

    std::vector<uint8_t> build() const {
        std::vector<uint8_t> out = header_bytes();
        const size_t data_start = align_up(out.size(), alignment);
        out.resize(data_start, 0);
        const auto offs = offsets();
        for (size_t i = 0; i < tensors.size(); ++i) {
            if (tensors[i].offset_override) continue; // deliberately out-of-place: data not written
            out.resize(data_start + offs[i], 0);
            out.insert(out.end(), tensors[i].data.begin(), tensors[i].data.end());
        }
        return out;
    }
};
```

- [ ] **Step 2: Write the failing header tests** — create `cpp/libs/sapient-io/tests/gguf_test.cpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
// Port of crates/sapient-io/src/gguf.rs's behaviour: the two Rust unit tests by name (suite
// `Gguf`, Task 3) plus synthetic-file tests for every branch of the parser and both loaders.
// Every expected error literal is verbatim Rust output (probe crate, 2026-09-23).
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "io_test_util.hpp"
#include "sapient/io/gguf.hpp"

namespace gguf = sapient::io::gguf;
namespace detail = sapient::io::gguf::detail;
using gguf::GgufValue;
using sapient::io::test::GgufBuilder;
using sapient::io::test::put_le;
using sapient::io::test::put_str;

namespace {
std::string err_text(const sapient::core::Result<detail::ParsedHeader>& r) {
    return r.has_value() ? std::string("<ok>") : r.error().to_string();
}
const GgufValue& kv(const detail::ParsedHeader& h, const std::string& key) {
    return h.metadata.at(key); // a missing key throws; gtest reports the uncaught exception
}
} // namespace

// ── GgmlType table (gguf.rs:88-178) ───────────────────────────────────────────────────────────
TEST(GgufHeader, ggml_type_table) {
    using detail::GgmlType;
    struct Row {
        uint32_t code;
        GgmlType t;
        size_t block;
        size_t size;
        const char* name;
    };
    const Row rows[] = {
        {0, GgmlType::F32, 1, 4, "F32"},       {1, GgmlType::F16, 1, 2, "F16"},
        {2, GgmlType::Q4_0, 32, 18, "Q4_0"},   {3, GgmlType::Q4_1, 32, 20, "Q4_1"},
        {6, GgmlType::Q5_0, 32, 22, "Q5_0"},   {7, GgmlType::Q5_1, 32, 24, "Q5_1"},
        {8, GgmlType::Q8_0, 32, 34, "Q8_0"},   {9, GgmlType::Q8_1, 32, 36, "Q8_1"},
        {10, GgmlType::Q2_K, 256, 84, "Q2_K"}, {11, GgmlType::Q3_K, 256, 110, "Q3_K"},
        {12, GgmlType::Q4_K, 256, 144, "Q4_K"}, {13, GgmlType::Q5_K, 256, 176, "Q5_K"},
        {14, GgmlType::Q6_K, 256, 210, "Q6_K"}, {30, GgmlType::BF16, 1, 2, "BF16"},
    };
    for (const auto& r : rows) {
        SCOPED_TRACE(r.name);
        ASSERT_EQ(detail::ggml_type_from_u32(r.code), r.t);
        EXPECT_EQ(detail::block_size(r.t), r.block);
        EXPECT_EQ(detail::type_size(r.t), r.size);
        EXPECT_EQ(detail::debug_name(r.t), r.name);
    }
    for (const uint32_t bad : {4u, 5u, 15u, 16u, 29u, 31u, 99u})
        EXPECT_EQ(detail::ggml_type_from_u32(bad), std::nullopt) << bad;
    using sapient::core::DType;
    EXPECT_EQ(detail::to_sapient_dtype(GgmlType::Q4_0), DType::Q4_0);
    EXPECT_EQ(detail::to_sapient_dtype(GgmlType::Q8_0), DType::Q8_0);
    EXPECT_EQ(detail::to_sapient_dtype(GgmlType::Q4_K), DType::Q4_K);
    EXPECT_EQ(detail::to_sapient_dtype(GgmlType::Q5_K), DType::Q5_K);
    EXPECT_EQ(detail::to_sapient_dtype(GgmlType::Q6_K), DType::Q6_K);
    for (const auto t : {GgmlType::F32, GgmlType::F16, GgmlType::BF16, GgmlType::Q4_1,
                         GgmlType::Q5_0, GgmlType::Q5_1, GgmlType::Q8_1, GgmlType::Q2_K,
                         GgmlType::Q3_K})
        EXPECT_EQ(detail::to_sapient_dtype(t), std::nullopt) << detail::debug_name(t);
    // tensor_byte_len: floats multiply, blocks truncate.
    EXPECT_EQ(detail::tensor_byte_len(GgmlType::F16, 3), 6u);
    EXPECT_EQ(detail::tensor_byte_len(GgmlType::F32, 5), 20u);
    EXPECT_EQ(detail::tensor_byte_len(GgmlType::Q8_0, 64), 68u);
    EXPECT_EQ(detail::tensor_byte_len(GgmlType::Q4_K, 300), 144u);
    EXPECT_EQ(detail::tensor_byte_len(GgmlType::Q5_0, 48), 22u);
}

// ── Scalar KV types (read_value codes 0-8, 10-12) ────────────────────────────────────────────
TEST(GgufHeader, parses_every_scalar_kv_type) {
    GgufBuilder b;
    b.kv_u8("u8", 200).kv_i8("i8", -5).kv_u16("u16", 60000).kv_i16("i16", -30000);
    b.kv_u32("u32", 4000000000u).kv_i32("i32", -7).kv_f32("f32", 1.25f).kv_bool_byte("b", 2);
    b.kv_str("s", "llama").kv_u64("u64", 1ull << 40).kv_i64("i64", -(1ll << 40));
    b.kv_f64("f64", 0.1);
    const auto h = detail::parse_header(b.header_bytes());
    ASSERT_TRUE(h.has_value()) << h.error().to_string();
    EXPECT_EQ(kv(*h, "u8").kind(), GgufValue::Kind::U8);
    EXPECT_EQ(*kv(*h, "u8").get_if<uint8_t>(), 200);
    EXPECT_EQ(*kv(*h, "i8").get_if<int8_t>(), -5);
    EXPECT_EQ(*kv(*h, "u16").get_if<uint16_t>(), 60000);
    EXPECT_EQ(*kv(*h, "i16").get_if<int16_t>(), -30000);
    EXPECT_EQ(*kv(*h, "u32").get_if<uint32_t>(), 4000000000u);
    EXPECT_EQ(*kv(*h, "i32").get_if<int32_t>(), -7);
    EXPECT_EQ(*kv(*h, "f32").get_if<float>(), 1.25f);
    EXPECT_EQ(kv(*h, "b").kind(), GgufValue::Kind::Bool);
    EXPECT_TRUE(*kv(*h, "b").get_if<bool>()); // any non-zero byte is true
    EXPECT_EQ(*kv(*h, "s").get_if<std::string>(), "llama");
    EXPECT_EQ(*kv(*h, "u64").get_if<uint64_t>(), 1ull << 40);
    EXPECT_EQ(*kv(*h, "i64").get_if<int64_t>(), -(1ll << 40));
    EXPECT_EQ(*kv(*h, "f64").get_if<double>(), 0.1);
    EXPECT_EQ(h->metadata.size(), 12u);
}

// ── Accessor widening rules (gguf.rs:203-246) ────────────────────────────────────────────────
TEST(GgufValue, accessor_widening_rules) {
    EXPECT_EQ(GgufValue(uint32_t{7}).as_u32(), 7u);
    EXPECT_EQ(GgufValue(uint64_t{(1ull << 32) + 5}).as_u32(), 5u); // `as u32` truncates
    EXPECT_EQ(GgufValue(int32_t{7}).as_u32(), 7u);
    EXPECT_EQ(GgufValue(int32_t{-1}).as_u32(), std::nullopt);
    EXPECT_EQ(GgufValue(uint16_t{7}).as_u32(), std::nullopt);
    EXPECT_EQ(GgufValue(uint64_t{9}).as_u64(), 9u);
    EXPECT_EQ(GgufValue(uint32_t{9}).as_u64(), 9u);
    EXPECT_EQ(GgufValue(int64_t{9}).as_u64(), std::nullopt);
    EXPECT_EQ(GgufValue(int32_t{9}).as_u64(), std::nullopt);
    EXPECT_EQ(GgufValue(0.1).as_f32(), static_cast<float>(0.1));
    EXPECT_EQ(GgufValue(1.5f).as_f32(), 1.5f);
    EXPECT_EQ(GgufValue(1.5f).as_f64(), 1.5);
    EXPECT_EQ(GgufValue(uint32_t{1}).as_f32(), std::nullopt);
    EXPECT_EQ(GgufValue(true).as_bool(), true);
    EXPECT_EQ(GgufValue(uint8_t{0}).as_bool(), false);
    EXPECT_EQ(GgufValue(uint8_t{3}).as_bool(), true);
    EXPECT_EQ(GgufValue(int8_t{1}).as_bool(), std::nullopt);
    EXPECT_EQ(GgufValue(std::string("x")).as_str(), "x");
    EXPECT_EQ(GgufValue(uint8_t{1}).as_str(), std::nullopt);
    EXPECT_EQ(GgufValue().kind(), GgufValue::Kind::Other);
}

// ── Arrays + the zero-byte skip quirks (read_value code 9, skip_value) ──────────────────────
TEST(GgufHeader, arrays_decode_or_become_other) {
    GgufBuilder b;
    {
        std::vector<uint8_t> items;
        for (const uint32_t v : {1u, 2u, 3u})
            put_le(items, v);
        b.kv_array("u32s", 4, 3, items);
    }
    b.kv_arr_str("strs", {"<s>", "a", "\xC3\xA9"});
    {
        std::vector<uint8_t> items;
        for (const float v : {0.5f, -1.0f})
            put_le(items, v);
        b.kv_array("f32s", 6, 2, items);
    }
    {
        std::vector<uint8_t> items; // item type 5 (i32): skipped 4 bytes each → Other
        for (const int32_t v : {1, -1, 3})
            put_le(items, v);
        b.kv_array("i32s", 5, 3, items);
    }
    {
        std::vector<uint8_t> items; // item type 0 (u8) → Other, 1 byte each
        items = {9, 8};
        b.kv_array("u8s", 0, 2, items);
    }
    b.kv_array("nested", 9, 3, {});        // skip_value(9) consumes NOTHING → no payload
    b.kv_array("unknown_items", 13, 5, {}); // likewise for an unknown item type
    b.kv_u32("after", 42);                  // proves the cursor stayed in sync
    const auto h = detail::parse_header(b.header_bytes());
    ASSERT_TRUE(h.has_value()) << h.error().to_string();
    EXPECT_EQ(*kv(*h, "u32s").get_if<std::vector<uint32_t>>(), (std::vector<uint32_t>{1, 2, 3}));
    EXPECT_EQ(*kv(*h, "strs").get_if<std::vector<std::string>>(),
              (std::vector<std::string>{"<s>", "a", "\xC3\xA9"}));
    EXPECT_EQ(*kv(*h, "f32s").get_if<std::vector<float>>(), (std::vector<float>{0.5f, -1.0f}));
    for (const char* k : {"i32s", "u8s", "nested", "unknown_items"})
        EXPECT_EQ(kv(*h, k).kind(), GgufValue::Kind::Other) << k;
    EXPECT_EQ(kv(*h, "after").as_u32(), 42u);
}

TEST(GgufHeader, unknown_value_type_consumes_nothing) {
    GgufBuilder b;
    b.kv_raw("weird", 13, {}); // type 13: Rust reads no payload → Other
    b.kv_u32("after", 7);
    const auto h = detail::parse_header(b.header_bytes());
    ASSERT_TRUE(h.has_value()) << h.error().to_string();
    EXPECT_EQ(kv(*h, "weird").kind(), GgufValue::Kind::Other);
    EXPECT_EQ(kv(*h, "after").as_u32(), 7u);
}

TEST(GgufHeader, duplicate_keys_last_wins) {
    GgufBuilder b;
    b.kv_u32("k", 1).kv_u32("k", 2);
    const auto h = detail::parse_header(b.header_bytes());
    ASSERT_TRUE(h.has_value());
    EXPECT_EQ(h->metadata.size(), 1u);
    EXPECT_EQ(kv(*h, "k").as_u32(), 2u);
}

// ── Tensor infos, data_start, alignment ─────────────────────────────────────────────────────
TEST(GgufHeader, tensor_infos_keep_gguf_dim_order) {
    GgufBuilder b;
    b.kv_str("general.architecture", "llama");
    b.tensor("w", {64, 32}, 8, std::vector<uint8_t>(64 * 32 / 32 * 34));
    b.tensor("n", {32}, 0, std::vector<uint8_t>(32 * 4));
    const auto hb = b.header_bytes();
    const auto h = detail::parse_header(hb);
    ASSERT_TRUE(h.has_value()) << h.error().to_string();
    ASSERT_EQ(h->tensor_infos.size(), 2u);
    EXPECT_EQ(h->tensor_infos[0].name, "w");
    EXPECT_EQ(h->tensor_infos[0].dims, (std::vector<size_t>{64, 32})); // [in, out] — no flip here
    EXPECT_EQ(h->tensor_infos[0].kind, detail::GgmlType::Q8_0);
    EXPECT_EQ(h->tensor_infos[0].offset, 0u);
    EXPECT_EQ(h->tensor_infos[1].offset, GgufBuilder::align_up(64 * 32 / 32 * 34, 32));
    EXPECT_EQ(h->data_start, GgufBuilder::align_up(hb.size(), 32));
}

TEST(GgufHeader, alignment_from_metadata) {
    for (const bool as_u64 : {false, true}) {
        GgufBuilder b;
        b.alignment = 64;
        if (as_u64)
            b.kv_u64("general.alignment", 64);
        else
            b.kv_u32("general.alignment", 64);
        b.tensor("t", {4}, 0, std::vector<uint8_t>(16));
        const auto hb = b.header_bytes();
        const auto h = detail::parse_header(hb);
        ASSERT_TRUE(h.has_value());
        EXPECT_EQ(h->data_start, GgufBuilder::align_up(hb.size(), 64)) << as_u64;
    }
}

TEST(GgufHeader, alignment_stored_as_i32_is_ignored) {
    GgufBuilder b; // as_u64() of an I32 is None → DEFAULT_ALIGNMENT (faithful Rust quirk)
    b.kv_i32("general.alignment", 64);
    const auto hb = b.header_bytes();
    const auto h = detail::parse_header(hb);
    ASSERT_TRUE(h.has_value());
    EXPECT_EQ(h->data_start, GgufBuilder::align_up(hb.size(), 32));
}

TEST(GgufHeader, zero_alignment_panics_like_rust_div_ceil) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    GgufBuilder b;
    b.kv_u32("general.alignment", 0);
    const auto hb = b.header_bytes();
    EXPECT_DEATH((void)detail::parse_header(hb), "attempt to divide by zero");
}

// ── Header-level errors (texts verbatim from Rust) ───────────────────────────────────────────
TEST(GgufHeader, versions_1_to_3_accepted_others_rejected) {
    for (const uint32_t v : {1u, 2u, 3u}) {
        GgufBuilder b;
        b.version = v;
        EXPECT_TRUE(detail::parse_header(b.header_bytes()).has_value()) << v;
    }
    for (const uint32_t v : {0u, 4u}) {
        GgufBuilder b;
        b.version = v;
        EXPECT_EQ(err_text(detail::parse_header(b.header_bytes())),
                  "GGUF parse error: unsupported GGUF version " + std::to_string(v) +
                      " (expected 1\xE2\x80\x93"
                      "3)");
    }
}

TEST(GgufHeader, bad_magic) {
    GgufBuilder b;
    b.magic = 0x58554747; // "GGUX"
    EXPECT_EQ(err_text(detail::parse_header(b.header_bytes())), "GGUF parse error: bad GGUF magic");
}

TEST(GgufHeader, truncation_at_every_prefix_is_eof) {
    GgufBuilder b;
    b.kv_str("general.architecture", "llama").kv_u32("general.alignment", 32);
    b.kv_arr_str("tokenizer.ggml.tokens", {"a", "bc"});
    b.tensor("w", {32, 2}, 8, std::vector<uint8_t>(68));
    const auto hb = b.header_bytes();
    ASSERT_TRUE(detail::parse_header(hb).has_value());
    for (size_t n = 0; n < hb.size(); ++n)
        ASSERT_EQ(err_text(detail::parse_header(std::span(hb).first(n))),
                  "GGUF parse error: failed to fill whole buffer")
            << "prefix " << n;
}

TEST(GgufHeader, huge_lengths_and_counts_are_eof_not_allocation) {
    const std::string eof = "GGUF parse error: failed to fill whole buffer";
    { // a 2^62-byte key: Rust would try to allocate it; C++ checks the remaining bytes first
        std::vector<uint8_t> v;
        put_le<uint32_t>(v, 0x46554747);
        put_le<uint32_t>(v, 3);
        put_le<uint64_t>(v, 0);
        put_le<uint64_t>(v, 1);
        put_le<uint64_t>(v, 1ull << 62);
        EXPECT_EQ(err_text(detail::parse_header(v)), eof);
    }
    { // 2^60 tensors, 2^60 KVs — no reserve() from header fields
        GgufBuilder b;
        b.tensor_count_override = 1ull << 60;
        EXPECT_EQ(err_text(detail::parse_header(b.header_bytes())), eof);
        GgufBuilder c;
        c.kv_count_override = 1ull << 60;
        EXPECT_EQ(err_text(detail::parse_header(c.header_bytes())), eof);
    }
    { // n_dims = 2^31
        std::vector<uint8_t> v;
        put_le<uint32_t>(v, 0x46554747);
        put_le<uint32_t>(v, 3);
        put_le<uint64_t>(v, 1);
        put_le<uint64_t>(v, 0);
        put_str(v, "w");
        put_le<uint32_t>(v, 1u << 31);
        EXPECT_EQ(err_text(detail::parse_header(v)), eof);
    }
    { // a u32 array claiming 2^60 items with 8 bytes present
        GgufBuilder b;
        const std::vector<uint8_t> items(8, 0);
        b.kv_array("a", 4, 1ull << 60, items);
        EXPECT_EQ(err_text(detail::parse_header(b.header_bytes())), eof);
    }
    { // a zero-width item type claiming 2^60 items: returns Other immediately, never hangs
        GgufBuilder b;
        b.kv_array("a", 9, 1ull << 60, {});
        b.kv_u32("after", 1);
        const auto h = detail::parse_header(b.header_bytes());
        ASSERT_TRUE(h.has_value()) << h.error().to_string();
        EXPECT_EQ(kv(*h, "a").kind(), GgufValue::Kind::Other);
        EXPECT_EQ(kv(*h, "after").as_u32(), 1u);
    }
}

TEST(GgufHeader, invalid_utf8_key_uses_rust_text) {
    GgufBuilder b;
    const std::string bad_key("\xFF", 1);
    b.kv_u32(bad_key, 1);
    EXPECT_EQ(err_text(detail::parse_header(b.header_bytes())),
              "GGUF parse error: invalid utf-8 sequence of 1 bytes from index 0");
}

TEST(GgufHeader, unknown_ggml_type) {
    for (const uint32_t code : {4u, 99u}) {
        GgufBuilder b;
        b.tensor("w", {32}, code, {});
        EXPECT_EQ(err_text(detail::parse_header(b.header_bytes())),
                  "GGUF parse error: unknown ggml type " + std::to_string(code));
    }
}
```

In `cpp/libs/sapient-io/CMakeLists.txt`, add `src/gguf.cpp` to the `add_library(sapient_io …)` list and `tests/gguf_test.cpp` to `add_executable(sapient_io_tests …)`.

- [ ] **Step 3: Run the build to verify it fails**

Run: `cd cpp && cmake --build --preset dev`
Expected: FAIL — `sapient/io/gguf.hpp: file not found`.

- [ ] **Step 4: Write `gguf.hpp` (header-parsing half)**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-io/src/gguf.rs, minus the dead IR entry point `GgufLoader::load -> Graph`
// (sub-project 8). Dims stay in GGUF/ggml order ([in, out] for a linear weight): the flip to HF
// order is sapient-models' job (gguf_weights.rs), exactly as in Rust. Rust-private items that the
// ported tests call live in `gguf::detail`.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "sapient/core/dtype.hpp"
#include "sapient/core/error.hpp"
#include "sapient/core/tensor.hpp"

namespace sapient::io::gguf {

namespace core = sapient::core;

/// Rust `GgufValue::Other` — an unsupported value type or a skipped array.
struct GgufOther {
    bool operator==(const GgufOther&) const = default;
};

/// Rust `enum GgufValue` (gguf.rs:184-201): the 16 alternatives in declaration order.
class GgufValue {
public:
    using Storage =
        std::variant<uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, float, bool, std::string,
                     uint64_t, int64_t, double, std::vector<uint32_t>, std::vector<std::string>,
                     std::vector<float>, GgufOther>;
    enum class Kind : uint8_t {
        U8, I8, U16, I16, U32, I32, F32, Bool, Str, U64, I64, F64, ArrayU32, ArrayStr, ArrayF32,
        Other
    };

    GgufValue() : v_(GgufOther{}) {}
    /// Exact alternatives only — `GgufValue(1)` (an int) must not silently become I64.
    template <class T>
        requires(std::is_same_v<std::remove_cvref_t<T>, uint8_t> ||
                 std::is_same_v<std::remove_cvref_t<T>, int8_t> ||
                 std::is_same_v<std::remove_cvref_t<T>, uint16_t> ||
                 std::is_same_v<std::remove_cvref_t<T>, int16_t> ||
                 std::is_same_v<std::remove_cvref_t<T>, uint32_t> ||
                 std::is_same_v<std::remove_cvref_t<T>, int32_t> ||
                 std::is_same_v<std::remove_cvref_t<T>, float> ||
                 std::is_same_v<std::remove_cvref_t<T>, bool> ||
                 std::is_same_v<std::remove_cvref_t<T>, std::string> ||
                 std::is_same_v<std::remove_cvref_t<T>, uint64_t> ||
                 std::is_same_v<std::remove_cvref_t<T>, int64_t> ||
                 std::is_same_v<std::remove_cvref_t<T>, double> ||
                 std::is_same_v<std::remove_cvref_t<T>, std::vector<uint32_t>> ||
                 std::is_same_v<std::remove_cvref_t<T>, std::vector<std::string>> ||
                 std::is_same_v<std::remove_cvref_t<T>, std::vector<float>> ||
                 std::is_same_v<std::remove_cvref_t<T>, GgufOther>)
    explicit GgufValue(T&& v) : v_(std::forward<T>(v)) {}

    Kind kind() const { return static_cast<Kind>(v_.index()); }
    const Storage& storage() const { return v_; }
    template <class T> const T* get_if() const { return std::get_if<T>(&v_); }
    bool operator==(const GgufValue&) const = default;

    std::optional<uint32_t> as_u32() const;
    std::optional<uint64_t> as_u64() const;
    std::optional<float> as_f32() const;
    std::optional<double> as_f64() const;
    std::optional<bool> as_bool() const;
    std::optional<std::string_view> as_str() const;

private:
    Storage v_;
};

using GgufMetadata = std::unordered_map<std::string, GgufValue>;
using TensorMap = std::unordered_map<std::string, sapient::core::Tensor>;

namespace detail {

inline constexpr uint32_t GGUF_MAGIC = 0x46554747; // "GGUF"
inline constexpr uint64_t DEFAULT_ALIGNMENT = 32;
inline constexpr size_t QK_K = 256;

enum class GgmlType : uint32_t {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
    Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, BF16 = 30
};

std::optional<GgmlType> ggml_type_from_u32(uint32_t v);
size_t block_size(GgmlType t);
size_t type_size(GgmlType t);
/// The five types kept as packed blocks (zero-copy on the mmap path).
std::optional<sapient::core::DType> to_sapient_dtype(GgmlType t);
/// Rust's `{:?}` of the variant ("Q4_1", "BF16", …) — used in an error text.
std::string_view debug_name(GgmlType t);
/// `F32|F16|BF16 → numel * type_size`, else `(numel / block_size) * type_size` (truncating,
/// wrapping like Rust release).
size_t tensor_byte_len(GgmlType kind, size_t numel);

struct GgufTensorInfo {
    std::string name;
    std::vector<size_t> dims; // GGUF order, ne0 fastest-varying
    GgmlType kind{GgmlType::F32};
    uint64_t offset{0}; // relative to data_start
};

struct ParsedHeader {
    GgufMetadata metadata;
    std::vector<GgufTensorInfo> tensor_infos;
    size_t data_start{0}; // alignment-corrected
};

core::Result<ParsedHeader> parse_header(std::span<const uint8_t> bytes);

} // namespace detail
} // namespace sapient::io::gguf
```

- [ ] **Step 5: Write `gguf.cpp` (header-parsing half)**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/io/gguf.hpp"

#include <bit>
#include <cstring>

#include "sapient/core/panic.hpp"
#include "sapient/io/rust_std.hpp"

static_assert(std::endian::native == std::endian::little,
              "sapient::io assumes a little-endian host (as the Rust crate does)");

namespace sapient::io::gguf {

// ── GgufValue accessors (gguf.rs:203-246) ────────────────────────────────────────────────────
std::optional<uint32_t> GgufValue::as_u32() const {
    if (const auto* v = get_if<uint32_t>()) return *v;
    if (const auto* v = get_if<uint64_t>()) return static_cast<uint32_t>(*v); // `as u32`
    if (const auto* v = get_if<int32_t>(); v != nullptr && *v >= 0) return static_cast<uint32_t>(*v);
    return std::nullopt;
}
std::optional<uint64_t> GgufValue::as_u64() const {
    if (const auto* v = get_if<uint64_t>()) return *v;
    if (const auto* v = get_if<uint32_t>()) return static_cast<uint64_t>(*v);
    return std::nullopt;
}
std::optional<float> GgufValue::as_f32() const {
    if (const auto* v = get_if<float>()) return *v;
    if (const auto* v = get_if<double>()) return static_cast<float>(*v);
    return std::nullopt;
}
std::optional<double> GgufValue::as_f64() const {
    if (const auto* v = get_if<double>()) return *v;
    if (const auto* v = get_if<float>()) return static_cast<double>(*v);
    return std::nullopt;
}
std::optional<bool> GgufValue::as_bool() const {
    if (const auto* v = get_if<bool>()) return *v;
    if (const auto* v = get_if<uint8_t>()) return *v != 0;
    return std::nullopt;
}
std::optional<std::string_view> GgufValue::as_str() const {
    if (const auto* v = get_if<std::string>()) return std::string_view(*v);
    return std::nullopt;
}

namespace detail {

// ── GgmlType (gguf.rs:110-178) ───────────────────────────────────────────────────────────────
std::optional<GgmlType> ggml_type_from_u32(uint32_t v) {
    switch (v) {
    case 0: case 1: case 2: case 3: case 6: case 7: case 8: case 9:
    case 10: case 11: case 12: case 13: case 14: case 30:
        return static_cast<GgmlType>(v);
    default:
        return std::nullopt;
    }
}

size_t block_size(GgmlType t) {
    switch (t) {
    case GgmlType::F32: case GgmlType::F16: case GgmlType::BF16:
        return 1;
    case GgmlType::Q4_0: case GgmlType::Q4_1: case GgmlType::Q5_0: case GgmlType::Q5_1:
    case GgmlType::Q8_0: case GgmlType::Q8_1:
        return 32;
    case GgmlType::Q2_K: case GgmlType::Q3_K: case GgmlType::Q4_K: case GgmlType::Q5_K:
    case GgmlType::Q6_K:
        return QK_K;
    }
    core::panic("block_size: invalid GgmlType");
}

size_t type_size(GgmlType t) {
    switch (t) {
    case GgmlType::F32: return 4;
    case GgmlType::F16: case GgmlType::BF16: return 2;
    case GgmlType::Q4_0: return 18;
    case GgmlType::Q4_1: return 20;
    case GgmlType::Q5_0: return 22;
    case GgmlType::Q5_1: return 24;
    case GgmlType::Q8_0: return 34;
    case GgmlType::Q8_1: return 36;
    case GgmlType::Q2_K: return 84;
    case GgmlType::Q3_K: return 110;
    case GgmlType::Q4_K: return 144;
    case GgmlType::Q5_K: return 176;
    case GgmlType::Q6_K: return 210;
    }
    core::panic("type_size: invalid GgmlType");
}

std::optional<core::DType> to_sapient_dtype(GgmlType t) {
    switch (t) {
    case GgmlType::Q4_0: return core::DType::Q4_0;
    case GgmlType::Q8_0: return core::DType::Q8_0;
    case GgmlType::Q4_K: return core::DType::Q4_K;
    case GgmlType::Q5_K: return core::DType::Q5_K;
    case GgmlType::Q6_K: return core::DType::Q6_K;
    default: return std::nullopt;
    }
}

std::string_view debug_name(GgmlType t) {
    switch (t) {
    case GgmlType::F32: return "F32";
    case GgmlType::F16: return "F16";
    case GgmlType::Q4_0: return "Q4_0";
    case GgmlType::Q4_1: return "Q4_1";
    case GgmlType::Q5_0: return "Q5_0";
    case GgmlType::Q5_1: return "Q5_1";
    case GgmlType::Q8_0: return "Q8_0";
    case GgmlType::Q8_1: return "Q8_1";
    case GgmlType::Q2_K: return "Q2_K";
    case GgmlType::Q3_K: return "Q3_K";
    case GgmlType::Q4_K: return "Q4_K";
    case GgmlType::Q5_K: return "Q5_K";
    case GgmlType::Q6_K: return "Q6_K";
    case GgmlType::BF16: return "BF16";
    }
    core::panic("debug_name: invalid GgmlType");
}

size_t tensor_byte_len(GgmlType kind, size_t numel) {
    if (kind == GgmlType::F32 || kind == GgmlType::F16 || kind == GgmlType::BF16)
        return numel * type_size(kind);
    return (numel / block_size(kind)) * type_size(kind);
}

// ── Low-level readers (gguf.rs:780-909) over a bounds-checked cursor ─────────────────────────
namespace {

class Cursor {
public:
    explicit Cursor(std::span<const uint8_t> b) : b_(b) {}
    size_t position() const { return pos_; }
    /// `read_exact`: all `n` bytes or Rust's EOF error. Checked BEFORE any caller allocates.
    core::Result<std::span<const uint8_t>> take(size_t n) {
        if (n > b_.size() - pos_)
            return tl::unexpected(core::Error::gguf_parse(std::string(rust_std::READ_EXACT_EOF)));
        const auto s = b_.subspan(pos_, n);
        pos_ += n;
        return s;
    }

private:
    std::span<const uint8_t> b_;
    size_t pos_{0};
};

template <class T> core::Result<T> read_le(Cursor& c) {
    SAPIENT_TRY_ASSIGN(const auto s, c.take(sizeof(T)));
    T v;
    std::memcpy(&v, s.data(), sizeof(T));
    return v;
}

core::Result<std::string> read_gguf_string(Cursor& c) {
    SAPIENT_TRY_ASSIGN(const uint64_t len, read_le<uint64_t>(c));
    // Hardening (plan B Global Constraints): bounds-check before allocating. Rust allocates
    // `vec![0; len]` first and aborts on an absurd len; for every len it survives, the text is
    // this same EOF error.
    SAPIENT_TRY_ASSIGN(const auto bytes, c.take(static_cast<size_t>(len)));
    if (auto e = rust_std::utf8_error(bytes)) return tl::unexpected(core::Error::gguf_parse(*e));
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

/// skip_value's byte width per item type; 0 = consumes nothing (Rust's `_ => {}` arm, incl. 9).
/// Strings (8) are variable-width and handled separately.
size_t skip_width(uint32_t vtype) {
    switch (vtype) {
    case 0: case 1: case 7: return 1;
    case 2: case 3: return 2;
    case 4: case 5: case 6: return 4;
    case 10: case 11: case 12: return 8;
    default: return 0;
    }
}

core::Result<void> skip_value(Cursor& c, uint32_t vtype) {
    if (vtype == 8) {
        SAPIENT_TRY(read_gguf_string(c));
        return {};
    }
    SAPIENT_TRY(c.take(skip_width(vtype)));
    return {};
}

core::Result<GgufValue> read_array(Cursor& c) {
    SAPIENT_TRY_ASSIGN(const uint32_t item_type, read_le<uint32_t>(c));
    SAPIENT_TRY_ASSIGN(const uint64_t count64, read_le<uint64_t>(c));
    const auto count = static_cast<size_t>(count64);
    switch (item_type) {
    case 4: {
        std::vector<uint32_t> v; // no reserve(count): untrusted
        for (size_t i = 0; i < count; ++i) {
            SAPIENT_TRY_ASSIGN(const uint32_t x, read_le<uint32_t>(c));
            v.push_back(x);
        }
        return GgufValue(std::move(v));
    }
    case 8: {
        std::vector<std::string> v;
        for (size_t i = 0; i < count; ++i) {
            SAPIENT_TRY_ASSIGN(std::string s, read_gguf_string(c));
            v.push_back(std::move(s));
        }
        return GgufValue(std::move(v));
    }
    case 6: {
        std::vector<float> v;
        for (size_t i = 0; i < count; ++i) {
            SAPIENT_TRY_ASSIGN(const float x, read_le<float>(c));
            v.push_back(x);
        }
        return GgufValue(std::move(v));
    }
    default:
        // Rust loops `count` times over skip_value; when the item type consumes nothing that
        // loop is a no-op, so skip it (a 2^60 count would otherwise hang — same bytes consumed).
        if (skip_width(item_type) == 0) return GgufValue(GgufOther{}); // 8 never reaches here
        for (size_t i = 0; i < count; ++i)
            SAPIENT_TRY(skip_value(c, item_type));
        return GgufValue(GgufOther{});
    }
}

core::Result<GgufValue> read_value(Cursor& c, uint32_t vtype) {
    switch (vtype) {
    case 0: {
        SAPIENT_TRY_ASSIGN(const uint8_t v, read_le<uint8_t>(c));
        return GgufValue(v);
    }
    case 1: {
        SAPIENT_TRY_ASSIGN(const uint8_t v, read_le<uint8_t>(c));
        return GgufValue(static_cast<int8_t>(v));
    }
    case 2: {
        SAPIENT_TRY_ASSIGN(const uint16_t v, read_le<uint16_t>(c));
        return GgufValue(v);
    }
    case 3: {
        SAPIENT_TRY_ASSIGN(const uint16_t v, read_le<uint16_t>(c));
        return GgufValue(static_cast<int16_t>(v));
    }
    case 4: {
        SAPIENT_TRY_ASSIGN(const uint32_t v, read_le<uint32_t>(c));
        return GgufValue(v);
    }
    case 5: {
        SAPIENT_TRY_ASSIGN(const int32_t v, read_le<int32_t>(c));
        return GgufValue(v);
    }
    case 6: {
        SAPIENT_TRY_ASSIGN(const float v, read_le<float>(c));
        return GgufValue(v);
    }
    case 7: {
        SAPIENT_TRY_ASSIGN(const uint8_t v, read_le<uint8_t>(c));
        return GgufValue(v != 0);
    }
    case 8: {
        SAPIENT_TRY_ASSIGN(std::string v, read_gguf_string(c));
        return GgufValue(std::move(v));
    }
    case 9:
        return read_array(c);
    case 10: {
        SAPIENT_TRY_ASSIGN(const uint64_t v, read_le<uint64_t>(c));
        return GgufValue(v);
    }
    case 11: {
        SAPIENT_TRY_ASSIGN(const int64_t v, read_le<int64_t>(c));
        return GgufValue(v);
    }
    case 12: {
        SAPIENT_TRY_ASSIGN(const double v, read_le<double>(c));
        return GgufValue(v);
    }
    default:
        return GgufValue(GgufOther{}); // consumes NOTHING (faithful; see Global Constraints)
    }
}

} // namespace

// ── parse_header (gguf.rs:517-578) ───────────────────────────────────────────────────────────
core::Result<ParsedHeader> parse_header(std::span<const uint8_t> bytes) {
    Cursor c(bytes);
    SAPIENT_TRY_ASSIGN(const uint32_t magic, read_le<uint32_t>(c));
    if (magic != GGUF_MAGIC) return tl::unexpected(core::Error::gguf_parse("bad GGUF magic"));
    SAPIENT_TRY_ASSIGN(const uint32_t version, read_le<uint32_t>(c));
    if (version < 1 || version > 3)
        return tl::unexpected(core::Error::gguf_parse("unsupported GGUF version " +
                                                      std::to_string(version) +
                                                      " (expected 1\xE2\x80\x93"
                                                      "3)"));
    SAPIENT_TRY_ASSIGN(const uint64_t tensor_count, read_le<uint64_t>(c));
    SAPIENT_TRY_ASSIGN(const uint64_t kv_count, read_le<uint64_t>(c));

    ParsedHeader h; // NO reserve(kv_count) / reserve(tensor_count): untrusted header fields
    for (uint64_t i = 0; i < kv_count; ++i) {
        SAPIENT_TRY_ASSIGN(std::string key, read_gguf_string(c));
        SAPIENT_TRY_ASSIGN(const uint32_t vtype, read_le<uint32_t>(c));
        SAPIENT_TRY_ASSIGN(GgufValue value, read_value(c, vtype));
        h.metadata.insert_or_assign(std::move(key), std::move(value)); // HashMap::insert: last wins
    }
    for (uint64_t i = 0; i < tensor_count; ++i) {
        SAPIENT_TRY_ASSIGN(std::string name, read_gguf_string(c));
        SAPIENT_TRY_ASSIGN(const uint32_t n_dims, read_le<uint32_t>(c));
        std::vector<size_t> dims; // no reserve(n_dims)
        for (uint32_t d = 0; d < n_dims; ++d) {
            SAPIENT_TRY_ASSIGN(const uint64_t dim, read_le<uint64_t>(c));
            dims.push_back(static_cast<size_t>(dim));
        }
        SAPIENT_TRY_ASSIGN(const uint32_t kind_raw, read_le<uint32_t>(c));
        const auto kind = ggml_type_from_u32(kind_raw);
        if (!kind)
            return tl::unexpected(
                core::Error::gguf_parse("unknown ggml type " + std::to_string(kind_raw)));
        SAPIENT_TRY_ASSIGN(const uint64_t offset, read_le<uint64_t>(c));
        h.tensor_infos.push_back(GgufTensorInfo{std::move(name), std::move(dims), *kind, offset});
    }

    uint64_t alignment = DEFAULT_ALIGNMENT;
    if (const auto it = h.metadata.find("general.alignment"); it != h.metadata.end())
        if (const auto a = it->second.as_u64()) alignment = *a;
    if (alignment == 0) core::panic("attempt to divide by zero"); // Rust u64::div_ceil(0)
    const uint64_t raw_pos = c.position();
    const uint64_t q = raw_pos / alignment + (raw_pos % alignment != 0 ? 1 : 0); // div_ceil
    h.data_start = static_cast<size_t>(q) * static_cast<size_t>(alignment); // wraps like Rust
    return h;
}

} // namespace detail
} // namespace sapient::io::gguf
```

- [ ] **Step 6: Build and run the full suite**

Run: `cd cpp && cmake --build --preset dev && ctest --preset dev`
Expected: PASS — the 16 new `GgufHeader.*`/`GgufValue.*` tests pass, 100% of the run suite green.

- [ ] **Step 7: Cross-compile the new TUs for x86_64** — `src/gguf.cpp` (library command) and `tests/gguf_test.cpp` (test command). Expected: exit 0.

- [ ] **Step 8: Format, SPDX-lint, commit**

```bash
git add cpp/libs/sapient-io
.superpowers/tools-venv/bin/clang-format -i $(git diff --cached --name-only -- 'cpp/*.hpp' 'cpp/*.cpp')
git add cpp/libs/sapient-io
python3 cpp/scripts/check_spdx.py cpp crates
git commit -m "$(cat <<'EOF'
cpp(io): port the GGUF header parser — GgmlType, GgufValue, parse_header

Faithful to gguf.rs incl. its quirks (value types >= 13 and nested-array skips
consume nothing, non-u32/f32/str arrays are Other, I32 alignment ignored, zero
alignment aborts). Never allocates from an untrusted count or length: a string
length is checked against the remaining bytes first, and no reserve() is sized
from a header field, so absurd headers give Rust's EOF text instead of an abort.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: GGUF tensors — dequant wrappers, `quantize_to_q8_0`, `MmapBuffer`, `make_tensor(_mmap)`, the five loaders, the real-file check

**Files:**
- Modify: `cpp/libs/sapient-io/include/sapient/io/gguf.hpp` (append the loader + tensor half), `cpp/libs/sapient-io/src/gguf.cpp` (append), `cpp/libs/sapient-io/tests/gguf_test.cpp` (append), `cpp/libs/sapient-io/CMakeLists.txt` (add `tests/real_file_test.cpp`; link `sapient::backends_cpu` into the **test** target only), `.github/workflows/ci.yml` (real-file step)
- Create: `cpp/libs/sapient-io/tests/real_file_test.cpp`

**Interfaces:**
- Consumes: Task 1 (`MappedFile`, `MapError`, `read_file`, `display_path`); Task 2 (`parse_header`, `GgmlType`, `tensor_byte_len`, `to_sapient_dtype`, `block_size`, `debug_name`, `GgufTensorInfo`, `GgufMetadata`, `TensorMap`); plan A `sapient::core::{Tensor::from_quant_bytes, Tensor::from_f32, Tensor::from_buffer, Buffer, Shape, Error::gguf_parse, Error::model_not_found, f16_le_to_f32, bf16_le_to_f32, f16_to_le, f32_to_f16_bits, dequant::{q4_0_block, q8_0_block, q4_k_block, q5_k_block, q6_k_block}, panic}`. Test-only: `sapient::backends_cpu::kernels::quant::quantize_q8_0_block(std::span<const float>) -> std::array<uint8_t, 34>` (plan D).
- Produces (namespace `sapient::io::gguf`): `class GgufLoader` with `static core::Result<GgufMetadata> parse_metadata_only(const std::filesystem::path&)`, `static core::Result<std::pair<GgufMetadata, TensorMap>> load_tensors_mmap(const std::filesystem::path&)`, `static core::Result<TensorMap> load_tensors(const std::filesystem::path&)`, `static core::Result<std::pair<GgufMetadata, TensorMap>> load_tensors_with_metadata(const std::filesystem::path&)`, `static core::Result<TensorMap> tensors_from_bytes(std::span<const uint8_t>)`.
- Produces (namespace `sapient::io::gguf::detail`): `std::vector<float> dequantize_q4_0/q8_0/q5_0/q4_k/q5_k/q6_k(std::span<const uint8_t> data, size_t numel)`; `std::vector<uint8_t> quantize_to_q8_0(std::span<const float>)`; `core::Result<std::vector<float>> dequantize_to_f32(GgmlType, std::span<const uint8_t>, size_t numel)`; `class MmapBuffer final : public core::Buffer`; `core::Result<core::Tensor> make_tensor(const GgufTensorInfo&, std::span<const uint8_t> bytes, size_t data_start)`; `core::Result<core::Tensor> make_tensor_mmap(const GgufTensorInfo&, const std::shared_ptr<const MappedFile>&, size_t data_start)`. Task 4's `io.hpp` re-exports `GgufLoader`; sub-project 1b's `gguf_weights`/`Pipeline` port consumes all five entry points.

Rust reference: `gguf.rs:30-78` (`MmapBuffer`), `:257-509` (dequant, `quantize_to_q8_0`, `dequantize_to_f32`), `:580-777` (`make_tensor_mmap`, `make_tensor`, the loaders), `:911-935` (the 2 tests). Porting map §B2-§B4, §B7.

- [ ] **Step 1: Write the failing tests** — append to `cpp/libs/sapient-io/tests/gguf_test.cpp` (add `#include <algorithm>`, `#include <array>`, `#include <bit>`, `#include <cmath>`, `#include <cstddef>`, `#include <filesystem>`, `#include <memory>`, `#include <optional>`, `#include <string_view>`, `#include "sapient/backends_cpu/kernels/quant.hpp"`, `#include "sapient/core/dequant.hpp"`, `#include "sapient/core/f16.hpp"`, `#include "sapient/io/mmap.hpp"`, `#include "sapient/io/rust_std.hpp"` to the include block, and `using sapient::io::test::TempDir; using sapient::io::test::le_f32; using sapient::io::test::le_u16; using gguf::GgufLoader;`):

```cpp
// ═════════════════════════════════════════ Task 3 ═══════════════════════════════════════════
namespace {

using sapient::core::DType;
using sapient::core::Tensor;

uint32_t lcg(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return s;
}
/// `n` blocks of `block_bytes` random bytes whose f16 fields at `f16_offsets` hold `f16_bits`
/// (so every scale is finite and the dequantised values stay sane).
std::vector<uint8_t> blocks(size_t n, size_t block_bytes, const std::vector<size_t>& f16_offsets,
                            uint16_t f16_bits, uint32_t seed) {
    std::vector<uint8_t> out(n * block_bytes);
    for (auto& b : out)
        b = static_cast<uint8_t>(lcg(seed) >> 24);
    for (size_t k = 0; k < n; ++k)
        for (const size_t off : f16_offsets)
            sapient::core::f16_to_le(f16_bits, out.data() + k * block_bytes + off);
    return out;
}
/// A Q5_0 block with the given f16 scale and 32-bit high-bit mask; nibble byte j = j | (15-j)<<4.
std::vector<uint8_t> q5_0_block(uint16_t scale_bits, uint32_t qh) {
    std::vector<uint8_t> b;
    put_le<uint16_t>(b, scale_bits);
    put_le<uint32_t>(b, qh);
    for (uint32_t j = 0; j < 16; ++j)
        b.push_back(static_cast<uint8_t>(j | ((15u - j) << 4)));
    return b;
}
/// Closed form of the two q5_0_block variants the tests use (see q5_0_closed_form).
std::vector<float> q5_0_expected(uint32_t qh, float scale) {
    std::vector<float> out(32);
    for (int j = 0; j < 16; ++j) {
        if (qh == 0x0000FFFFu) {
            out[static_cast<size_t>(j)] = static_cast<float>(j) * scale;
            out[static_cast<size_t>(j) + 16] = static_cast<float>(-(j + 1)) * scale;
        } else { // 0xFFFF0000
            out[static_cast<size_t>(j)] = static_cast<float>(j - 16) * scale;
            out[static_cast<size_t>(j) + 16] = static_cast<float>(15 - j) * scale;
        }
    }
    return out;
}
bool same_bits(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::bit_cast<uint32_t>(a[i]) != std::bit_cast<uint32_t>(b[i])) return false;
    return true;
}
std::vector<float> f32_of(const Tensor& t) {
    const auto s = t.f32_slice();
    return {s.begin(), s.end()};
}
std::vector<uint8_t> bytes_of(const Tensor& t) {
    const auto s = t.bytes();
    return {s.begin(), s.end()};
}

} // namespace

// ── The 2 Rust tests (gguf.rs:911-935), names verbatim ──────────────────────────────────────
TEST(Gguf, q8_0_quantize_roundtrips_and_sizes) {
    std::vector<float> data(64);
    for (size_t i = 0; i < 64; ++i)
        data[i] = (static_cast<float>(i) - 32.0f) * 0.1f;
    const auto q = detail::quantize_to_q8_0(data);
    ASSERT_EQ(q.size(), 64u / 32u * 34u) << "Q8_0 = 34 bytes / 32 weights";
    const auto back = detail::dequantize_q8_0(q, 64);
    ASSERT_EQ(back.size(), 64u);
    for (size_t i = 0; i < 64; ++i)
        EXPECT_LT(std::fabs(data[i] - back[i]), 0.03f) << "roundtrip a=" << data[i] << " b=" << back[i];
}

TEST(Gguf, q8_0_quantize_handles_all_zeros) {
    const std::vector<float> zeros(32, 0.0f);
    const auto q = detail::quantize_to_q8_0(zeros);
    ASSERT_EQ(q.size(), 34u);
    for (const float v : detail::dequantize_q8_0(q, 32))
        EXPECT_EQ(v, 0.0f);
}

// ── quantize_to_q8_0 vs plan D's golden-gated quantize_q8_0_block (differential) ─────────────
TEST(GgufQ8, matches_backends_cpu_quantize_q8_0_block) {
    uint32_t seed = 0xC0FFEE;
    std::vector<float> data(5 * 32 + 7); // the 7-element tail is dropped (chunks_exact)
    for (auto& v : data)
        v = (static_cast<float>(lcg(seed) >> 8) / 16777216.0f - 0.5f) * 8.0f;
    data[3] = 0.0f;
    data[40] = -1e-30f;
    const auto q = detail::quantize_to_q8_0(data);
    ASSERT_EQ(q.size(), 5u * 34u);
    for (size_t b = 0; b < 5; ++b) {
        const auto ref = sapient::backends_cpu::kernels::quant::quantize_q8_0_block(
            std::span<const float>(data).subspan(b * 32, 32));
        EXPECT_TRUE(std::equal(ref.begin(), ref.end(), q.begin() + static_cast<std::ptrdiff_t>(b * 34)))
            << "block " << b;
    }
}

TEST(GgufQ8, non_finite_inputs_match_rust_casts) {
    // inf → d = inf, id = 1/inf = 0, inf·0 = NaN → Rust `as i8` = 0 (C++ must not static_cast NaN).
    std::vector<float> a(32, 1.0f);
    a[0] = std::numeric_limits<float>::infinity();
    // NaN is dropped by the f32::max fold; its own quant is NaN·id = NaN → 0.
    std::vector<float> b(32, 2.0f);
    b[5] = std::numeric_limits<float>::quiet_NaN();
    for (const auto* v : {&a, &b}) {
        const auto q = detail::quantize_to_q8_0(*v);
        const auto ref = sapient::backends_cpu::kernels::quant::quantize_q8_0_block(*v);
        EXPECT_TRUE(std::equal(ref.begin(), ref.end(), q.begin()));
    }
    const auto qa = detail::quantize_to_q8_0(a);
    EXPECT_EQ(qa[0], 0x00); // f16 +inf = 0x7C00, little-endian
    EXPECT_EQ(qa[1], 0x7C);
    for (size_t i = 2; i < 34; ++i)
        EXPECT_EQ(qa[i], 0) << i;
}

// ── dequant wrappers ─────────────────────────────────────────────────────────────────────────
TEST(GgufDequant, q5_0_closed_form) {
    auto bytes = q5_0_block(0x3C00, 0x0000FFFFu); // scale 1.0
    const auto second = q5_0_block(0x3800, 0xFFFF0000u); // scale 0.5
    bytes.insert(bytes.end(), second.begin(), second.end());
    const auto out = detail::dequantize_q5_0(bytes, 64);
    auto expected = q5_0_expected(0x0000FFFFu, 1.0f);
    const auto e2 = q5_0_expected(0xFFFF0000u, 0.5f);
    expected.insert(expected.end(), e2.begin(), e2.end());
    EXPECT_TRUE(same_bits(out, expected));
    // numel not a multiple of 32: numel/32 blocks, the tail stays zero.
    const auto tail = detail::dequantize_q5_0(std::span(bytes).first(22), 48);
    ASSERT_EQ(tail.size(), 48u);
    EXPECT_TRUE(same_bits(std::span(tail).first(32), q5_0_expected(0x0000FFFFu, 1.0f)));
    for (size_t i = 32; i < 48; ++i)
        EXPECT_EQ(tail[i], 0.0f);
}

TEST(GgufDequant, kept_types_use_core_blocks_with_io_block_counts) {
    namespace dq = sapient::core::dequant;
    // Q4_0 / Q8_0: bytes/18 (bytes/34) blocks, writes past numel skipped.
    const auto q4 = blocks(2, 18, {0}, 0x3C00, 1);
    const auto out4 = detail::dequantize_q4_0(q4, 40);
    ASSERT_EQ(out4.size(), 40u);
    std::array<float, 32> b0{}, b1{};
    dq::q4_0_block(q4.data(), b0.data());
    dq::q4_0_block(q4.data() + 18, b1.data());
    EXPECT_TRUE(same_bits(std::span(out4).first(32), b0));
    EXPECT_TRUE(same_bits(std::span(out4).subspan(32), std::span(b1).first(8)));
    const auto q8 = blocks(2, 34, {0}, 0x2C00, 2);
    const auto out8 = detail::dequantize_q8_0(q8, 50);
    std::array<float, 32> c1{};
    dq::q8_0_block(q8.data() + 34, c1.data());
    EXPECT_TRUE(same_bits(std::span(out8).subspan(32), std::span(c1).first(18)));
    // K-quants: numel/256 blocks.
    struct K {
        detail::GgmlType t;
        size_t bb;
        std::vector<size_t> f16s; // not initializer_list: its backing array would dangle
        void (*block)(const uint8_t*, float*);
    };
    const K ks[] = {{detail::GgmlType::Q4_K, 144, {0, 2}, dq::q4_k_block},
                    {detail::GgmlType::Q5_K, 176, {0, 2}, dq::q5_k_block}, // core's per-element form
                    {detail::GgmlType::Q6_K, 210, {208}, dq::q6_k_block}};
    for (const auto& k : ks) {
        SCOPED_TRACE(std::string(detail::debug_name(k.t)));
        const auto raw = blocks(2, k.bb, k.f16s, 0x2C00, 3);
        const auto got = detail::dequantize_to_f32(k.t, raw, 512);
        ASSERT_TRUE(got.has_value());
        std::vector<float> want(512);
        k.block(raw.data(), want.data());
        k.block(raw.data() + k.bb, want.data() + 256);
        EXPECT_TRUE(same_bits(*got, want));
    }
}

TEST(GgufDequant, float_types_and_unsupported_types) {
    const auto f16 = le_u16({0x3C00, 0xC000, 0x7BFF});
    const auto a = detail::dequantize_to_f32(detail::GgmlType::F16, f16, 3);
    ASSERT_TRUE(a.has_value());
    EXPECT_EQ(*a, (std::vector<float>{1.0f, -2.0f, 65504.0f}));
    const auto bf16 = le_u16({0x3F80, 0xBFC0});
    const auto b = detail::dequantize_to_f32(detail::GgmlType::BF16, bf16, 2);
    ASSERT_TRUE(b.has_value());
    EXPECT_EQ(*b, (std::vector<float>{1.0f, -1.5f}));
    const auto f32 = le_f32({0.25f, -8.0f});
    const auto c = detail::dequantize_to_f32(detail::GgmlType::F32, f32, 2);
    ASSERT_TRUE(c.has_value());
    EXPECT_EQ(*c, (std::vector<float>{0.25f, -8.0f}));
    for (const auto t : {detail::GgmlType::Q4_1, detail::GgmlType::Q5_1, detail::GgmlType::Q8_1,
                         detail::GgmlType::Q2_K, detail::GgmlType::Q3_K}) {
        const auto r = detail::dequantize_to_f32(t, {}, 0);
        ASSERT_FALSE(r.has_value());
        EXPECT_EQ(r.error().to_string(), "GGUF parse error: unsupported GGUF quantization type " +
                                             std::string(detail::debug_name(t)));
    }
}

// ── Tensor materialisation over all three routes ─────────────────────────────────────────────
namespace {

enum class Route { Heap, Mmap, Bytes };
const char* route_name(Route r) {
    return r == Route::Heap ? "heap" : r == Route::Mmap ? "mmap" : "bytes";
}

GgufBuilder fixture() {
    GgufBuilder b;
    b.kv_str("general.architecture", "llama").kv_u32("llama.block_count", 2);
    b.kv_f32("llama.rope.freq_base", 10000.0f);
    b.kv_arr_str("tokenizer.ggml.tokens", {"<s>", "a", "\xC3\xA9"});
    b.tensor("f32", {3}, 0, le_f32({1.5f, -2.0f, 0.25f}));
    b.tensor("f16", {2, 2}, 1, le_u16({0x3C00, 0xC000, 0x3800, 0x7BFF}));
    b.tensor("bf16", {4}, 30, le_u16({0x3F80, 0xBFC0, 0x3E80, 0x4040}));
    b.tensor("q4_0", {32, 2}, 2, blocks(2, 18, {0}, 0x3C00, 11));
    b.tensor("q8_0", {32}, 8, blocks(1, 34, {0}, 0x2C00, 12));
    b.tensor("q4_k", {256}, 12, blocks(1, 144, {0, 2}, 0x2C00, 13));
    b.tensor("q5_k", {256}, 13, blocks(1, 176, {0, 2}, 0x2C00, 14));
    b.tensor("q6_k", {256, 2}, 14, blocks(2, 210, {208}, 0x2C00, 15));
    auto q5 = q5_0_block(0x3C00, 0x0000FFFFu);
    const auto q5b = q5_0_block(0x3800, 0xFFFF0000u);
    q5.insert(q5.end(), q5b.begin(), q5b.end());
    b.tensor("q5_0", {32, 2}, 6, q5);
    b.tensor("q5_0_tail", {48}, 6, q5_0_block(0x3C00, 0x0000FFFFu)); // 48 % 32 != 0 → F32
    b.tensor("scalar", {}, 0, le_f32({7.0f}));                       // 0-dim → Shape{1}
    return b;
}

sapient::core::Result<gguf::TensorMap> load(Route r, const std::filesystem::path& p,
                                            std::span<const uint8_t> bytes) {
    switch (r) {
    case Route::Heap:
        return GgufLoader::load_tensors(p);
    case Route::Mmap: {
        auto m = GgufLoader::load_tensors_mmap(p);
        if (!m) return tl::unexpected(m.error());
        return std::move(m->second);
    }
    case Route::Bytes:
        return GgufLoader::tensors_from_bytes(bytes);
    }
    return tl::unexpected(sapient::core::Error::internal("route"));
}

} // namespace

TEST(GgufTensors, every_route_materialises_the_fixture) {
    TempDir dir("fixture");
    const GgufBuilder b = fixture();
    const auto bytes = b.build();
    const auto p = dir.write("fixture.gguf", bytes);
    std::vector<float> q5_f32 = q5_0_expected(0x0000FFFFu, 1.0f);
    const auto q5_f32b = q5_0_expected(0xFFFF0000u, 0.5f);
    q5_f32.insert(q5_f32.end(), q5_f32b.begin(), q5_f32b.end());
    for (const Route r : {Route::Heap, Route::Mmap, Route::Bytes}) {
        SCOPED_TRACE(route_name(r));
        auto m = load(r, p, bytes);
        ASSERT_TRUE(m.has_value()) << m.error().to_string();
        ASSERT_EQ(m->size(), 11u);
        const auto& t = [&](const char* name) -> const Tensor& { return m->at(name); };
        // Float sources become F32 (align 64, heap) on every route.
        EXPECT_EQ(t("f32").dtype(), DType::F32);
        EXPECT_EQ(f32_of(t("f32")), (std::vector<float>{1.5f, -2.0f, 0.25f}));
        EXPECT_EQ(t("f16").shape().dims, (std::vector<size_t>{2, 2}));
        EXPECT_EQ(f32_of(t("f16")), (std::vector<float>{1.0f, -2.0f, 0.5f, 65504.0f}));
        EXPECT_EQ(f32_of(t("bf16")), (std::vector<float>{1.0f, -1.5f, 0.25f, 3.0f}));
        for (const char* f : {"f32", "f16", "bf16", "q5_0_tail", "scalar"}) {
            EXPECT_EQ(t(f).dtype(), DType::F32) << f;
            EXPECT_FALSE(t(f).is_mmap()) << f;
            EXPECT_EQ(t(f).buffer().alignment(), 64u) << f << ": from_f32, never from_f32_vec";
        }
        // Q5_0 → re-quantised Q8_0 (numel % 32 == 0); the tail case stays F32 with zeros.
        EXPECT_EQ(t("q5_0").dtype(), DType::Q8_0);
        EXPECT_FALSE(t("q5_0").is_mmap());
        EXPECT_EQ(bytes_of(t("q5_0")), detail::quantize_to_q8_0(q5_f32));
        const auto tail = f32_of(t("q5_0_tail"));
        EXPECT_TRUE(same_bits(std::span(tail).first(32), q5_0_expected(0x0000FFFFu, 1.0f)));
        for (size_t i = 32; i < 48; ++i)
            EXPECT_EQ(tail[i], 0.0f);
        EXPECT_EQ(t("scalar").shape().dims, (std::vector<size_t>{1}));
        // The five kept types: raw bytes, GGUF dim order, zero-copy on the mmap route only.
        struct Kept {
            const char* name;
            DType dtype;
            std::vector<size_t> dims;
        };
        const Kept kept[] = {{"q4_0", DType::Q4_0, {32, 2}},
                             {"q8_0", DType::Q8_0, {32}},
                             {"q4_k", DType::Q4_K, {256}},
                             {"q5_k", DType::Q5_K, {256}},
                             {"q6_k", DType::Q6_K, {256, 2}}};
        for (const auto& k : kept) {
            SCOPED_TRACE(k.name);
            const auto& src = *std::find_if(b.tensors.begin(), b.tensors.end(),
                                            [&](const auto& x) { return x.name == k.name; });
            EXPECT_EQ(t(k.name).dtype(), k.dtype);
            EXPECT_EQ(t(k.name).shape().dims, k.dims);
            EXPECT_EQ(bytes_of(t(k.name)), src.data);
            EXPECT_EQ(t(k.name).is_mmap(), r == Route::Mmap);
            EXPECT_EQ(t(k.name).buffer().alignment(), r == Route::Mmap ? 32u : 16u);
            EXPECT_EQ(t(k.name).buffer().device(), r == Route::Mmap ? "cpu-mmap" : "cpu");
        }
    }
}

TEST(GgufLoader, metadata_agrees_across_entry_points) {
    TempDir dir("metadata");
    const auto p = dir.write("m.gguf", fixture().build());
    auto heap = GgufLoader::load_tensors_with_metadata(p);
    auto mm = GgufLoader::load_tensors_mmap(p);
    auto md = GgufLoader::parse_metadata_only(p);
    ASSERT_TRUE(heap && mm && md);
    EXPECT_EQ(heap->first, mm->first);
    EXPECT_EQ(heap->first, *md);
    EXPECT_EQ(md->at("general.architecture").as_str(), "llama");
    EXPECT_EQ(*md->at("tokenizer.ggml.tokens").get_if<std::vector<std::string>>(),
              (std::vector<std::string>{"<s>", "a", "\xC3\xA9"}));
    auto plain = GgufLoader::load_tensors(p);
    ASSERT_TRUE(plain);
    EXPECT_EQ(plain->size(), heap->second.size());
}

TEST(GgufLoader, duplicate_tensor_names_last_wins) {
    TempDir dir("dup");
    GgufBuilder b;
    b.tensor("w", {1}, 0, le_f32({1.0f}));
    b.tensor("w", {1}, 0, le_f32({2.0f}));
    const auto bytes = b.build();
    const auto p = dir.write("d.gguf", bytes);
    for (const Route r : {Route::Heap, Route::Mmap, Route::Bytes}) {
        auto m = load(r, p, bytes);
        ASSERT_TRUE(m.has_value()) << route_name(r);
        EXPECT_EQ(m->size(), 1u);
        EXPECT_EQ(f32_of(m->at("w")), (std::vector<float>{2.0f})) << route_name(r);
    }
}

TEST(GgufLoader, mmap_tensor_outlives_the_loader) {
    TempDir dir("outlives"); // declared first → destroyed last (Windows cannot delete a mapped file)
    const GgufBuilder b = fixture();
    const auto p = dir.write("o.gguf", b.build());
    std::optional<Tensor> keep;
    {
        auto m = GgufLoader::load_tensors_mmap(p);
        ASSERT_TRUE(m.has_value());
        keep = m->second.at("q6_k");
    } // the map, the metadata and every other tensor are gone; the MappedFile lives on in `keep`
    ASSERT_TRUE(keep->is_mmap());
    EXPECT_EQ(bytes_of(*keep), b.tensors[7].data); // tensors[7] is "q6_k" in fixture()
    keep.reset();
}

TEST(GgufLoader, mmap_buffer_is_read_only) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    TempDir dir("readonly");
    const auto p = dir.write("r.gguf", fixture().build());
    std::optional<Tensor> t;
    {
        auto m = GgufLoader::load_tensors_mmap(p);
        ASSERT_TRUE(m.has_value());
        t = m->second.at("q4_k");
    } // `t` now holds the only handle to its MmapBuffer, so bytes_mut reaches the buffer
    EXPECT_DEATH((void)t->bytes_mut(),
                 "MmapBuffer is read-only \xE2\x80\x94 model weights cannot be mutated in-place");
    t.reset();
}

// ── Materialisation errors (texts verbatim from Rust) ───────────────────────────────────────
TEST(GgufTensors, data_range_past_end_of_file) {
    TempDir dir("range");
    GgufBuilder b;
    b.tensor("w", {4}, 0, {});
    b.tensors[0].offset_override = 100;
    const auto bytes = b.build();
    const auto p = dir.write("r.gguf", bytes);
    const size_t ds = detail::parse_header(bytes)->data_start;
    const std::string want = "GGUF parse error: tensor 'w': data range [" +
                             std::to_string(ds + 100) + ".." + std::to_string(ds + 116) +
                             "] exceeds file size " + std::to_string(bytes.size());
    for (const Route r : {Route::Heap, Route::Mmap, Route::Bytes}) {
        const auto m = load(r, p, bytes);
        ASSERT_FALSE(m.has_value()) << route_name(r);
        EXPECT_EQ(m.error().to_string(), want) << route_name(r);
    }
}

TEST(GgufTensors, unsupported_quant_type_errors_on_every_route) {
    TempDir dir("q4_1");
    GgufBuilder b;
    b.tensor("w", {32}, 3, std::vector<uint8_t>(20)); // Q4_1
    const auto bytes = b.build();
    const auto p = dir.write("q.gguf", bytes);
    for (const Route r : {Route::Heap, Route::Mmap, Route::Bytes}) {
        const auto m = load(r, p, bytes);
        ASSERT_FALSE(m.has_value());
        EXPECT_EQ(m.error().to_string(), "GGUF parse error: unsupported GGUF quantization type Q4_1")
            << route_name(r);
    }
}

TEST(GgufTensors, zero_dimension_is_wrapped_invalid_graph) {
    // numel = max(product, 1) = 1, so the byte range is valid; the Tensor constructor's
    // Shape::validate then rejects axis 0. Two files: the float branch (from_f32) and the kept
    // branch (from_quant_bytes on heap/bytes, from_buffer on mmap) — materialisation stops at the
    // FIRST bad tensor, so each needs its own file.
    struct Case {
        const char* name;
        std::vector<uint64_t> dims;
        uint32_t kind;
        std::vector<uint8_t> data;
    };
    const Case cases[] = {{"f", {0, 4}, 0, le_f32({1.0f})}, {"q", {0, 32}, 8, {}}};
    for (const auto& c : cases) {
        SCOPED_TRACE(c.name);
        TempDir dir("zero_dim");
        GgufBuilder b;
        b.tensor(c.name, c.dims, c.kind, c.data);
        const auto bytes = b.build();
        const auto p = dir.write("z.gguf", bytes);
        for (const Route r : {Route::Heap, Route::Mmap, Route::Bytes}) {
            const auto m = load(r, p, bytes);
            ASSERT_FALSE(m.has_value());
            EXPECT_EQ(m.error().to_string(),
                      "GGUF parse error: Graph validation failed: Shape has zero dimension at axis 0")
                << route_name(r);
        }
    }
}

TEST(GgufTensors, wrapped_range_panics_like_rust_slice) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    GgufBuilder b;
    b.tensor("w", {1}, 0, {});
    const auto probe = b.build();
    const size_t ds = detail::parse_header(probe)->data_start;
    // start = ds + offset wraps to 2^64 - 2; end = start + 4 wraps to 2 <= file size → Rust's
    // bounds check passes and `&bytes[start..end]` panics.
    b.tensors[0].offset_override = std::numeric_limits<uint64_t>::max() - 1 - ds;
    const auto bytes = b.build();
    EXPECT_DEATH((void)GgufLoader::tensors_from_bytes(bytes), "slice index starts at");
}

// ── File-level errors per entry point ────────────────────────────────────────────────────────
TEST(GgufLoader, missing_file_is_model_not_found_everywhere) {
    TempDir dir("missing");
    const auto p = dir.path() / "missing.gguf";
    const std::string want = "Model not found at path '" + sapient::io::display_path(p) + ": " +
                             sapient::io::rust_std::os_error_message(2) + "'";
    EXPECT_EQ(GgufLoader::load_tensors(p).error().to_string(), want);
    EXPECT_EQ(GgufLoader::load_tensors_with_metadata(p).error().to_string(), want);
    EXPECT_EQ(GgufLoader::load_tensors_mmap(p).error().to_string(), want);
    EXPECT_EQ(GgufLoader::parse_metadata_only(p).error().to_string(), want);
}

TEST(GgufLoader, empty_file_is_eof_on_every_entry_point) {
    TempDir dir("empty");
    const auto p = dir.write("e.gguf", {});
    const std::string eof = "GGUF parse error: failed to fill whole buffer";
    EXPECT_EQ(GgufLoader::load_tensors(p).error().to_string(), eof);
    EXPECT_EQ(GgufLoader::load_tensors_mmap(p).error().to_string(), eof);
    EXPECT_EQ(GgufLoader::parse_metadata_only(p).error().to_string(), eof);
    EXPECT_EQ(GgufLoader::tensors_from_bytes({}).error().to_string(), eof);
}

#if !defined(_WIN32)
TEST(GgufLoader, directory_wrapping_per_entry_point) {
    TempDir dir("dir");
    const auto p = dir.path();
    // Heap path: std::fs::read → open succeeds, read fails EISDIR → ModelNotFound.
    EXPECT_EQ(GgufLoader::load_tensors(p).error().to_string(),
              "Model not found at path '" + sapient::io::display_path(p) +
                  ": Is a directory (os error 21)'");
    // Mmap paths: open succeeds, mmap fails (macOS EINVAL / Linux ENODEV) → two different prefixes.
    const auto mm = GgufLoader::load_tensors_mmap(p).error().to_string();
    EXPECT_TRUE(mm.starts_with("GGUF parse error: mmap failed: ")) << mm;
    EXPECT_TRUE(mm.ends_with(")") && mm.find("(os error ") != std::string::npos) << mm;
    const auto md = GgufLoader::parse_metadata_only(p).error().to_string();
    EXPECT_TRUE(md.starts_with("GGUF parse error: mmap failed for header read: ")) << md;
    EXPECT_EQ(mm.substr(std::string("GGUF parse error: mmap failed: ").size()),
              md.substr(std::string("GGUF parse error: mmap failed for header read: ").size()));
}
#endif
```

Create `cpp/libs/sapient-io/tests/real_file_test.cpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
// Env-gated real-file checks (spec §4 plan B gate). Unset → SKIP with the exact text the CI step
// greps for; set but unreadable → FAIL (a stale path must not silently downgrade the gate).
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>

#include "sapient/core/dtype.hpp"
#include "sapient/io/gguf.hpp"

using sapient::io::gguf::GgufLoader;

TEST(RealFile, gguf_heap_and_mmap_agree) {
    const char* env = std::getenv("SAPIENT_TEST_GGUF");
    if (env == nullptr || *env == '\0') GTEST_SKIP() << "SAPIENT_TEST_GGUF unset";
    const std::filesystem::path p(env);
    std::error_code ec;
    ASSERT_TRUE(std::filesystem::is_regular_file(p, ec)) << "SAPIENT_TEST_GGUF=" << env
                                                         << " is not a readable file";
    auto heap = GgufLoader::load_tensors_with_metadata(p);
    ASSERT_TRUE(heap.has_value()) << heap.error().to_string();
    auto mm = GgufLoader::load_tensors_mmap(p);
    ASSERT_TRUE(mm.has_value()) << mm.error().to_string();
    auto md = GgufLoader::parse_metadata_only(p);
    ASSERT_TRUE(md.has_value()) << md.error().to_string();

    EXPECT_EQ(heap->first, mm->first);
    EXPECT_EQ(heap->first, *md);
    ASSERT_FALSE(heap->second.empty());
    ASSERT_EQ(heap->second.size(), mm->second.size());
    size_t n_mmap = 0;
    size_t n_requant = 0;
    for (const auto& [name, h] : heap->second) {
        SCOPED_TRACE(name);
        const auto it = mm->second.find(name);
        ASSERT_NE(it, mm->second.end());
        const auto& m = it->second;
        EXPECT_EQ(h.dtype(), m.dtype());
        EXPECT_EQ(h.shape(), m.shape());
        EXPECT_FALSE(h.is_mmap());
        const auto hb = h.bytes();
        const auto mb = m.bytes();
        ASSERT_EQ(hb.size(), mb.size());
        EXPECT_TRUE(std::equal(hb.begin(), hb.end(), mb.begin()));
        if (m.is_mmap()) {
            ++n_mmap;
            EXPECT_TRUE(sapient::core::is_quantized(m.dtype()));
        } else if (sapient::core::is_quantized(m.dtype())) {
            ++n_requant; // only a Q5_0 source re-quantised to Q8_0 is quantized AND not mmap'd
            EXPECT_EQ(m.dtype(), sapient::core::DType::Q8_0);
        } else {
            EXPECT_EQ(m.dtype(), sapient::core::DType::F32);
        }
    }
    EXPECT_GT(n_mmap, 0u);
    std::printf("[real-file] %s: %zu tensors, %zu zero-copy, %zu requantised Q5_0->Q8_0, %zu KVs\n",
                env, heap->second.size(), n_mmap, n_requant, heap->first.size());
}
```

In `cpp/libs/sapient-io/CMakeLists.txt`: add `tests/real_file_test.cpp` to `sapient_io_tests`, and change its link line to `target_link_libraries(sapient_io_tests PRIVATE sapient::io sapient::backends_cpu GTest::gtest_main)` with a comment: `# backends_cpu is TEST-ONLY: plan D's golden-gated quantize_q8_0_block is the differential oracle for quantize_to_q8_0. The io library itself must never link it.`

- [ ] **Step 2: Run the build to verify it fails**

Run: `cd cpp && cmake --build --preset dev`
Expected: FAIL — `no member named 'quantize_to_q8_0' in namespace 'sapient::io::gguf::detail'` (and `GgufLoader` undeclared).

- [ ] **Step 3: Append the tensor/loader half to `gguf.hpp`**

Add `#include <filesystem>`, `#include <memory>`, `#include "sapient/core/buffer.hpp"`, `#include "sapient/io/mmap.hpp"` to the include block, then append inside `namespace sapient::io::gguf` (before its closing brace; the `detail` additions go in a second `namespace detail { … }` block):

```cpp
/// Rust `pub struct GgufLoader` (gguf.rs:680-778) minus `load -> Graph` (sub-project 8).
class GgufLoader {
public:
    /// Header KV only, via a mapping that is dropped before returning. Zero tensor allocation.
    static core::Result<GgufMetadata> parse_metadata_only(const std::filesystem::path& path);
    /// Q4_0/Q8_0/Q4_K/Q5_K/Q6_K are zero-copy views into the mapping (is_mmap()); F32/F16/BF16
    /// become F32 and Q5_0 becomes Q8_0 on the heap. (Rust's doc comment claims K-quants are
    /// dequantised here — stale; trust the code.)
    static core::Result<std::pair<GgufMetadata, TensorMap>>
    load_tensors_mmap(const std::filesystem::path& path);
    static core::Result<TensorMap> load_tensors(const std::filesystem::path& path);
    /// Reads the WHOLE file into the heap, then copies every tensor out (the ~2× peak-RSS path).
    static core::Result<std::pair<GgufMetadata, TensorMap>>
    load_tensors_with_metadata(const std::filesystem::path& path);
    static core::Result<TensorMap> tensors_from_bytes(std::span<const uint8_t> bytes);
};

namespace detail {

// io's own dequantisers (gguf.rs:259-476) — core's PER-BLOCK arithmetic with io's block
// counts: Q4_0/Q8_0 run bytes/block_bytes blocks and skip writes past numel; Q5_0 and the
// K-quants run numel/block_numel blocks. Q5_K is core's per-element form (io's own copy is stale
// and unreachable — spec §2.1). Returned vectors always have exactly `numel` elements.
std::vector<float> dequantize_q4_0(std::span<const uint8_t> data, size_t numel);
std::vector<float> dequantize_q8_0(std::span<const uint8_t> data, size_t numel);
std::vector<float> dequantize_q5_0(std::span<const uint8_t> data, size_t numel);
std::vector<float> dequantize_q4_k(std::span<const uint8_t> data, size_t numel);
std::vector<float> dequantize_q5_k(std::span<const uint8_t> data, size_t numel);
std::vector<float> dequantize_q6_k(std::span<const uint8_t> data, size_t numel);

/// F32 → ggml Q8_0 blocks (gguf.rs:289-301). A trailing partial chunk is dropped.
std::vector<uint8_t> quantize_to_q8_0(std::span<const float> data);

/// gguf.rs:478-505. Errors: "unsupported GGUF quantization type {Debug}".
core::Result<std::vector<float>> dequantize_to_f32(GgmlType kind, std::span<const uint8_t> bytes,
                                                   size_t numel);

/// A read-only window into a MappedFile (gguf.rs:38-78). Holds the mapping alive.
class MmapBuffer final : public core::Buffer {
public:
    MmapBuffer(std::shared_ptr<const MappedFile> mmap, size_t offset, size_t len)
        : mmap_(std::move(mmap)), offset_(offset), len_(len) {}
    std::span<const uint8_t> bytes() const override { return mmap_->bytes().subspan(offset_, len_); }
    std::span<uint8_t> bytes_mut() override;
    size_t len() const override { return len_; }
    bool is_mmap() const override { return true; }
    size_t alignment() const override { return 32; } // hard-coded GGUF default, as in Rust
    std::string_view device() const override { return "cpu-mmap"; }
    size_t offset() const { return offset_; }

private:
    std::shared_ptr<const MappedFile> mmap_;
    size_t offset_;
    size_t len_;
};

core::Result<core::Tensor> make_tensor(const GgufTensorInfo& info, std::span<const uint8_t> bytes,
                                       size_t data_start);
core::Result<core::Tensor> make_tensor_mmap(const GgufTensorInfo& info,
                                            const std::shared_ptr<const MappedFile>& mmap,
                                            size_t data_start);

} // namespace detail
```

- [ ] **Step 4: Append the implementation to `gguf.cpp`**

Add `#include <algorithm>`, `#include <array>`, `#include <cmath>`, `#include "sapient/core/dequant.hpp"`, `#include "sapient/core/f16.hpp"` to the include block. Inside `namespace sapient::io::gguf::detail`, after `parse_header`:

```cpp
// ── Dequantisation (gguf.rs:259-476) ─────────────────────────────────────────────────────────
namespace {

void require_bytes(std::span<const uint8_t> data, size_t need) {
    // Rust indexes `data[base..]` unchecked-by-us and panics past the end; never read OOB in C++.
    if (data.size() < need) core::panic("index out of bounds");
}

/// io's Q4_0/Q8_0 semantics: bytes/block_bytes blocks, writes past numel skipped.
template <size_t BlockBytes, void (*Block)(const uint8_t*, float*)>
std::vector<float> guarded_blocks(std::span<const uint8_t> data, size_t numel) {
    std::vector<float> out(numel, 0.0f);
    std::array<float, 32> tmp{};
    const size_t nblocks = data.size() / BlockBytes;
    for (size_t b = 0; b < nblocks; ++b) {
        Block(data.data() + b * BlockBytes, tmp.data());
        for (size_t j = 0; j < 32; ++j)
            if (b * 32 + j < numel) out[b * 32 + j] = tmp[j];
    }
    return out;
}

/// io's K-quant semantics: numel/256 blocks written straight into the output.
template <size_t BlockBytes, void (*Block)(const uint8_t*, float*)>
std::vector<float> k_blocks(std::span<const uint8_t> data, size_t numel) {
    const size_t nblocks = numel / QK_K;
    require_bytes(data, nblocks * BlockBytes);
    std::vector<float> out(numel, 0.0f);
    for (size_t b = 0; b < nblocks; ++b)
        Block(data.data() + b * BlockBytes, out.data() + b * QK_K);
    return out;
}

} // namespace

std::vector<float> dequantize_q4_0(std::span<const uint8_t> data, size_t numel) {
    return guarded_blocks<18, core::dequant::q4_0_block>(data, numel);
}
std::vector<float> dequantize_q8_0(std::span<const uint8_t> data, size_t numel) {
    return guarded_blocks<34, core::dequant::q8_0_block>(data, numel);
}

std::vector<float> dequantize_q5_0(std::span<const uint8_t> data, size_t numel) {
    const size_t nblocks = numel / 32;
    require_bytes(data, nblocks * 22);
    std::vector<float> out(numel, 0.0f);
    for (size_t b = 0; b < nblocks; ++b) {
        const uint8_t* base = data.data() + b * 22;
        const float scale = core::f16_le_to_f32(base);
        uint32_t qh = 0;
        std::memcpy(&qh, base + 2, 4);
        for (uint32_t j = 0; j < 16; ++j) {
            const uint8_t byte = base[6 + j];
            const uint32_t xh_0 = ((qh >> j) << 4) & 0x10u;
            const uint32_t xh_1 = (qh >> (j + 12)) & 0x10u;
            const int32_t x0 = static_cast<int32_t>(static_cast<uint32_t>(byte & 0x0Fu) | xh_0) - 16;
            const int32_t x1 = static_cast<int32_t>(static_cast<uint32_t>(byte >> 4) | xh_1) - 16;
            out[b * 32 + j] = static_cast<float>(x0) * scale;
            out[b * 32 + j + 16] = static_cast<float>(x1) * scale;
        }
    }
    return out;
}

std::vector<float> dequantize_q4_k(std::span<const uint8_t> data, size_t numel) {
    return k_blocks<144, core::dequant::q4_k_block>(data, numel);
}
std::vector<float> dequantize_q5_k(std::span<const uint8_t> data, size_t numel) {
    return k_blocks<176, core::dequant::q5_k_block>(data, numel);
}
std::vector<float> dequantize_q6_k(std::span<const uint8_t> data, size_t numel) {
    return k_blocks<210, core::dequant::q6_k_block>(data, numel);
}

std::vector<uint8_t> quantize_to_q8_0(std::span<const float> data) {
    std::vector<uint8_t> out;
    out.reserve(data.size() / 32 * 34); // sized from real data, not a header field
    for (size_t b = 0; b + 32 <= data.size(); b += 32) { // chunks_exact(32)
        float amax = 0.0f;
        for (size_t i = 0; i < 32; ++i)
            amax = std::fmax(amax, std::fabs(data[b + i])); // f32::max drops NaN
        const float d = amax / 127.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        uint8_t h[2];
        core::f16_to_le(core::f32_to_f16_bits(d), h); // RNE
        out.push_back(h[0]);
        out.push_back(h[1]);
        for (size_t i = 0; i < 32; ++i) {
            // `(v * id).round().clamp(-127.0, 127.0) as i8 as u8`: roundf, clamp, NaN → 0.
            const float q = std::clamp(std::roundf(data[b + i] * id), -127.0f, 127.0f);
            const int8_t v = std::isnan(q) ? int8_t{0} : static_cast<int8_t>(q);
            out.push_back(static_cast<uint8_t>(v));
        }
    }
    return out;
}

core::Result<std::vector<float>> dequantize_to_f32(GgmlType kind, std::span<const uint8_t> bytes,
                                                   size_t numel) {
    switch (kind) {
    case GgmlType::F32: {
        if (numel > bytes.size() / 4) core::panic("range end index out of range for slice");
        std::vector<float> out(numel);
        std::memcpy(out.data(), bytes.data(), numel * 4);
        return out;
    }
    case GgmlType::F16:
    case GgmlType::BF16: {
        if (numel > bytes.size() / 2) core::panic("range end index out of range for slice");
        std::vector<float> out(numel);
        for (size_t i = 0; i < numel; ++i)
            out[i] = kind == GgmlType::F16 ? core::f16_le_to_f32(bytes.data() + 2 * i)
                                           : core::bf16_le_to_f32(bytes.data() + 2 * i);
        return out;
    }
    case GgmlType::Q4_0: return dequantize_q4_0(bytes, numel);
    case GgmlType::Q5_0: return dequantize_q5_0(bytes, numel);
    case GgmlType::Q8_0: return dequantize_q8_0(bytes, numel);
    case GgmlType::Q4_K: return dequantize_q4_k(bytes, numel);
    case GgmlType::Q5_K: return dequantize_q5_k(bytes, numel);
    case GgmlType::Q6_K: return dequantize_q6_k(bytes, numel);
    default:
        return tl::unexpected(core::Error::gguf_parse("unsupported GGUF quantization type " +
                                                      std::string(debug_name(kind))));
    }
}

// ── MmapBuffer / make_tensor(_mmap) (gguf.rs:38-78, 580-676) ─────────────────────────────────
std::span<uint8_t> MmapBuffer::bytes_mut() {
    core::panic("MmapBuffer is read-only \xE2\x80\x94 model weights cannot be mutated in-place");
}

namespace {

size_t numel_of(const std::vector<size_t>& dims) {
    size_t n = 1;
    for (const size_t d : dims)
        n *= d; // wraps like Rust release
    return std::max<size_t>(n, 1);
}

core::Shape shape_of(const std::vector<size_t>& dims) {
    return dims.empty() ? core::Shape{1} : core::Shape(dims);
}

/// `e.to_string()` wrapped as GgufParseError — Rust's `.map_err(|e| GgufParseError(e.to_string()))`.
core::Result<core::Tensor> wrap(core::Result<core::Tensor> r) {
    if (!r) return tl::unexpected(core::Error::gguf_parse(r.error().to_string()));
    return r;
}

struct Range {
    size_t start;
    size_t end;
};

core::Result<Range> data_range(const GgufTensorInfo& info, size_t data_start, size_t byte_len,
                               size_t file_len) {
    const size_t start = data_start + static_cast<size_t>(info.offset); // wraps like Rust release
    const size_t end = start + byte_len;
    if (end > file_len)
        return tl::unexpected(core::Error::gguf_parse(
            "tensor '" + info.name + "': data range [" + std::to_string(start) + ".." +
            std::to_string(end) + "] exceeds file size " + std::to_string(file_len)));
    // A wrapped `end` passes Rust's check and then the slice panics (spec §3 rule 8).
    if (end < start)
        core::panic("slice index starts at " + std::to_string(start) + " but ends at " +
                    std::to_string(end));
    return Range{start, end};
}

/// The identical else-branch of make_tensor / make_tensor_mmap: a type SAPIENT does not keep.
core::Result<core::Tensor> convert_unkept(GgmlType kind, std::span<const uint8_t> raw, size_t numel,
                                          core::Shape shape) {
    SAPIENT_TRY_ASSIGN(const std::vector<float> f32_data, dequantize_to_f32(kind, raw, numel));
    if (block_size(kind) > 1 && numel % 32 == 0) { // in practice: Q5_0 only
        const std::vector<uint8_t> q8 = quantize_to_q8_0(f32_data);
        return wrap(core::Tensor::from_quant_bytes(q8, std::move(shape), core::DType::Q8_0));
    }
    return wrap(core::Tensor::from_f32(f32_data, std::move(shape))); // align 64 — not from_f32_vec
}

} // namespace

core::Result<core::Tensor> make_tensor(const GgufTensorInfo& info, std::span<const uint8_t> bytes,
                                       size_t data_start) {
    const size_t numel = numel_of(info.dims);
    const size_t byte_len = tensor_byte_len(info.kind, numel);
    SAPIENT_TRY_ASSIGN(const Range r, data_range(info, data_start, byte_len, bytes.size()));
    const auto raw = bytes.subspan(r.start, byte_len);
    core::Shape shape = shape_of(info.dims);
    if (const auto dtype = to_sapient_dtype(info.kind))
        return wrap(core::Tensor::from_quant_bytes(raw, std::move(shape), *dtype)); // copy, align 16
    return convert_unkept(info.kind, raw, numel, std::move(shape));
}

core::Result<core::Tensor> make_tensor_mmap(const GgufTensorInfo& info,
                                            const std::shared_ptr<const MappedFile>& mmap,
                                            size_t data_start) {
    const size_t numel = numel_of(info.dims);
    const size_t byte_len = tensor_byte_len(info.kind, numel);
    SAPIENT_TRY_ASSIGN(const Range r, data_range(info, data_start, byte_len, mmap->size()));
    core::Shape shape = shape_of(info.dims);
    if (const auto dtype = to_sapient_dtype(info.kind)) {
        // Zero-copy: MmapBuffer.offset = data_start + info.offset, Tensor.offset = 0 (Rust's
        // two-level offset).
        auto buf = std::make_shared<MmapBuffer>(mmap, r.start, byte_len);
        return wrap(core::Tensor::from_buffer(std::move(shape), *dtype, std::move(buf), 0));
    }
    return convert_unkept(info.kind, mmap->bytes().subspan(r.start, byte_len), numel,
                          std::move(shape));
}

} // namespace detail
```

Then, outside `detail` but inside `namespace sapient::io::gguf`:

```cpp
// ── GgufLoader (gguf.rs:680-778) ─────────────────────────────────────────────────────────────
namespace {

core::Error open_or_map_error(const std::filesystem::path& path, const MapError& e,
                              std::string_view map_prefix) {
    if (e.stage == MapStage::Open)
        return core::Error::model_not_found(display_path(path) + ": " + e.os.message);
    return core::Error::gguf_parse(std::string(map_prefix) + e.os.message);
}

core::Result<TensorMap> materialise(const detail::ParsedHeader& h, std::span<const uint8_t> bytes) {
    TensorMap tensors; // no reserve(tensor_count)
    for (const auto& info : h.tensor_infos) {
        SAPIENT_TRY_ASSIGN(core::Tensor t, detail::make_tensor(info, bytes, h.data_start));
        tensors.insert_or_assign(info.name, std::move(t)); // HashMap::insert: last wins
    }
    return tensors;
}

} // namespace

core::Result<GgufMetadata> GgufLoader::parse_metadata_only(const std::filesystem::path& path) {
    auto m = MappedFile::open(path);
    if (!m) return tl::unexpected(open_or_map_error(path, m.error(), "mmap failed for header read: "));
    SAPIENT_TRY_ASSIGN(detail::ParsedHeader h, detail::parse_header((*m)->bytes()));
    return std::move(h.metadata);
}

core::Result<std::pair<GgufMetadata, TensorMap>>
GgufLoader::load_tensors_mmap(const std::filesystem::path& path) {
    auto m = MappedFile::open(path);
    if (!m) return tl::unexpected(open_or_map_error(path, m.error(), "mmap failed: "));
    const std::shared_ptr<const MappedFile>& mmap = *m;
    SAPIENT_TRY_ASSIGN(detail::ParsedHeader h, detail::parse_header(mmap->bytes()));
    TensorMap tensors;
    for (const auto& info : h.tensor_infos) {
        SAPIENT_TRY_ASSIGN(core::Tensor t, detail::make_tensor_mmap(info, mmap, h.data_start));
        tensors.insert_or_assign(info.name, std::move(t));
    }
    return std::make_pair(std::move(h.metadata), std::move(tensors));
}

core::Result<TensorMap> GgufLoader::load_tensors(const std::filesystem::path& path) {
    SAPIENT_TRY_ASSIGN(auto both, load_tensors_with_metadata(path));
    return std::move(both.second);
}

core::Result<std::pair<GgufMetadata, TensorMap>>
GgufLoader::load_tensors_with_metadata(const std::filesystem::path& path) {
    auto bytes = read_file(path);
    if (!bytes)
        return tl::unexpected(
            core::Error::model_not_found(display_path(path) + ": " + bytes.error().message));
    SAPIENT_TRY_ASSIGN(detail::ParsedHeader h, detail::parse_header(*bytes));
    SAPIENT_TRY_ASSIGN(TensorMap tensors, materialise(h, *bytes));
    return std::make_pair(std::move(h.metadata), std::move(tensors));
}

core::Result<TensorMap> GgufLoader::tensors_from_bytes(std::span<const uint8_t> bytes) {
    SAPIENT_TRY_ASSIGN(const detail::ParsedHeader h, detail::parse_header(bytes));
    return materialise(h, bytes);
}
```

**Cautions for this step:**
- `core::Shape{1}` must use the `initializer_list` constructor (dims `{1}`), not `Shape(size_t)` — there is no such constructor, but double-check `shape_of` returns dims `{1}` (the `scalar` fixture tensor pins it).
- In `load_tensors_mmap`, `mmap` is passed by `const&` into every `make_tensor_mmap`, which copies the `shared_ptr` into each `MmapBuffer` — the loader's own handle dies at return; the tensors keep the mapping alive (`mmap_tensor_outlives_the_loader` pins it).
- `MmapBuffer::bytes_mut` is declared returning `std::span<uint8_t>` and its body only calls the `[[noreturn]]` `core::panic` — no `return` is needed, and adding one would be dead code.

- [ ] **Step 5: Add the CI real-file step**

In `.github/workflows/ci.yml`, job `cpp-parity`, append after the `Greedy token parity (Rust self-check until the C++ tool exists)` step (the model it downloads is what this step reads; the earlier `C++ tests against the dumps` step runs before the download, so the test SKIPs there):

```yaml
      - name: Real-file io check (GGUF heap vs mmap, plan B)
        shell: bash
        working-directory: cpp
        run: |
          GGUF=$(find ~/.cache/huggingface -iname '*smollm2-135m*.gguf' | head -1)
          test -n "$GGUF" || { echo "parity model GGUF not found in the HF cache" >&2; exit 1; }
          SAPIENT_TEST_GGUF="$GGUF" ctest --preset ${{ matrix.preset }} \
            -R '^RealFile\.gguf_heap_and_mmap_agree$' -V | tee "$RUNNER_TEMP/realfile.log"
          grep -q "\[real-file\]" "$RUNNER_TEMP/realfile.log"
          ! grep -q "SAPIENT_TEST_GGUF unset" "$RUNNER_TEMP/realfile.log"
```

(`shell: bash` in GitHub Actions runs with `-eo pipefail`, so a failing ctest fails the step even through `tee`; the two greps make a vacuous skip fail it too.) Validate the YAML parses: `python3 -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml'))"` (install PyYAML into `.superpowers/tools-venv` if missing).

- [ ] **Step 6: Build and run the full suite, then the real-file check locally**

Run: `cd cpp && cmake --build --preset dev && ctest --preset dev`
Expected: PASS — 100% of the run suite; `RealFile.gguf_heap_and_mmap_agree` reports SKIPPED ("SAPIENT_TEST_GGUF unset").

Then run the gate on the two GGUFs cached on this Mac and keep the `[real-file]` lines for Task 5's `PARITY.md` rows:

```bash
cd cpp
for f in $(find ~/.cache/huggingface -iname '*smollm2-135m*q4_k_m*.gguf' | head -1) \
         $(find ~/.cache/huggingface -iname 'qwen2.5-1.5b-instruct-q4_k_m.gguf' | head -1); do
  SAPIENT_TEST_GGUF="$f" ctest --preset dev -R '^RealFile\.gguf_heap_and_mmap_agree$' -V | grep -E '\[real-file\]|Passed|Failed'
done
```

Expected: two `[real-file] …` lines and two `Passed`. If no GGUF is cached, download one with the Rust binary (`cargo run --release -p sapient-cli -- pull smollm2-135m-q4`) — do not skip the local run.

- [ ] **Step 7: Cross-compile** `src/gguf.cpp`, `tests/gguf_test.cpp` and `tests/real_file_test.cpp` for x86_64 (Global Constraints commands). Expected: exit 0.

- [ ] **Step 8: Confirm the Rust tree is untouched and its io tests still pass**

Run: `git status --short crates/ && cargo test -p sapient-io 2>&1 | grep "test result"`
Expected: no `crates/` changes; `test result: ok. 2 passed; 0 failed`.

- [ ] **Step 9: Format, SPDX-lint, commit**

```bash
git add cpp/libs/sapient-io .github/workflows/ci.yml
.superpowers/tools-venv/bin/clang-format -i $(git diff --cached --name-only -- 'cpp/*.hpp' 'cpp/*.cpp')
git add cpp/libs/sapient-io
python3 cpp/scripts/check_spdx.py cpp crates
git commit -m "$(cat <<'EOF'
cpp(io): GGUF tensors + the five GgufLoader entry points, real-file gate

make_tensor(_mmap) keep Q4_0/Q8_0/Q4_K/Q5_K/Q6_K as blocks (zero-copy
MmapBuffer views on the mmap path), F32/F16/BF16 -> F32 (from_f32, align 64),
Q5_0 -> Q8_0 when numel % 32 == 0. quantize_to_q8_0 is differential-tested
against plan D's golden-gated quantize_q8_0_block (test-only link). Every
error text is Rust's, per entry point; the env-gated RealFile test compares the
heap and mmap paths byte-for-byte and runs in cpp-parity after the model
download.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: safetensors, `io.hpp` (`lib.rs`), the nlohmann/json pin

**Files:**
- Modify: `cpp/cmake/deps.cmake` (nlohmann/json), `cpp/third_party/LICENSES.md`, `cpp/libs/sapient-io/CMakeLists.txt` (sources, tests, the PRIVATE nlohmann link), `cpp/libs/sapient-io/tests/real_file_test.cpp` (append)
- Create: `cpp/libs/sapient-io/include/sapient/io/safetensors.hpp`, `cpp/libs/sapient-io/src/safetensors.cpp`, `cpp/libs/sapient-io/include/sapient/io/io.hpp`, `cpp/libs/sapient-io/src/io.cpp`, `cpp/libs/sapient-io/tests/safetensors_test.cpp`

**Interfaces:**
- Consumes: Task 1 (`MappedFile`, `MapStage`, `display_path`, `rust_std::utf8_error`); Task 3 (`gguf::GgufLoader`, `gguf::TensorMap`, `gguf::GgufValue`); plan A (`Tensor::from_bf16_bytes`, `Tensor::from_f16_bytes`, `Tensor::from_buffer`, `CpuBuffer::with_capacity`, `Shape::validate`, `Error::shape_mismatch`, `Error::safetensors_parse`, `sapient::core::to_string(DType)`).
- Produces (namespace `sapient::io::safetensors`): `using TensorMap = gguf::TensorMap`; `class SafetensorsLoader` with `static core::Result<TensorMap> load(const std::filesystem::path&)`, `static core::Result<TensorMap> from_bytes(std::span<const uint8_t>)`, `static core::Result<TensorMap> load_tensors(const std::filesystem::path&)`.
- Produces (namespace `sapient::io`, `io.hpp`): `using gguf::GgufLoader; using gguf::GgufValue; using safetensors::SafetensorsLoader; using TensorMap = gguf::TensorMap;` and `core::Result<TensorMap> load_gguf(const std::filesystem::path&)`, `core::Result<TensorMap> load_safetensors(const std::filesystem::path&)`.

Rust reference: `safetensors.rs` (all), `lib.rs:35-43`. Porting map §B5, §B6.

- [ ] **Step 1: Pin nlohmann/json**

Append to `cpp/cmake/deps.cmake`:

```cmake
# nlohmann/json — JSON for the safetensors header now (sub-project 1a plan B), config.json /
# tokenizer.json later (1b). MIT. The release tarball (no tests, ~200 KB) instead of a git clone.
FetchContent_Declare(nlohmann_json
  URL      https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(nlohmann_json)
```

If nlohmann's headers trip our `-Werror` warnings, mark them SYSTEM in `cpp/libs/sapient-io/CMakeLists.txt` with `set_target_properties(nlohmann_json PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES $<TARGET_PROPERTY:nlohmann_json,INTERFACE_INCLUDE_DIRECTORIES>)` (the project floor is CMake 3.24, so `FetchContent_Declare(… SYSTEM)` is unavailable) — never relax the warnings. The digest was computed on 2026-09-23 with `curl -sL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz | shasum -a 256`; re-run that to confirm it before committing. (If `FetchContent_Declare` warns about `DOWNLOAD_EXTRACT_TIMESTAMP`, add `DOWNLOAD_EXTRACT_TIMESTAMP TRUE`.)

In `cpp/third_party/LICENSES.md`, move nlohmann/json from the "Planned" table to "Pinned now" as:
`| nlohmann/json | v3.11.3 | MIT | safetensors header (1a plan B); config.json / tokenizer.json / HTTP bodies later | `cmake/deps.cmake` |`
and delete its "Planned" row.

- [ ] **Step 2: Write the failing tests** — create `cpp/libs/sapient-io/tests/safetensors_test.cpp` (every full-text literal is verbatim Rust output):

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
// SafetensorsLoader (safetensors.rs) + io.hpp (lib.rs). Rust has no tests here; these cover every
// branch. Texts after "tensor '{name}': " (serde_json's in Rust) are C++-authored and only the
// prefix is asserted.
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "io_test_util.hpp"
#include "sapient/core/dtype.hpp"
#include "sapient/io/gguf.hpp"
#include "sapient/io/io.hpp"
#include "sapient/io/mmap.hpp"
#include "sapient/io/rust_std.hpp"
#include "sapient/io/safetensors.hpp"

using sapient::core::DType;
using sapient::io::safetensors::SafetensorsLoader;
using sapient::io::test::le_f32;
using sapient::io::test::le_u16;
using sapient::io::test::put_le;
using sapient::io::test::TempDir;

namespace {
std::vector<uint8_t> st(std::string_view header, const std::vector<uint8_t>& data) {
    std::vector<uint8_t> out;
    put_le<uint64_t>(out, header.size());
    out.insert(out.end(), header.begin(), header.end());
    out.insert(out.end(), data.begin(), data.end());
    return out;
}
std::string err(const std::vector<uint8_t>& file) {
    const auto r = SafetensorsLoader::from_bytes(file);
    return r.has_value() ? std::string("<ok>") : r.error().to_string();
}
std::vector<float> f32_of(const sapient::core::Tensor& t) {
    const auto s = t.f32_slice();
    return {s.begin(), s.end()};
}
} // namespace

TEST(Safetensors, loads_f32_and_keeps_half_types_raw) {
    std::vector<uint8_t> data = le_f32({1.5f, -2.0f});
    const auto h = le_u16({0x3C00, 0xC000});   // F16 1, -2
    const auto bh = le_u16({0x3F80, 0xBFC0}); // BF16 1, -1.5
    data.insert(data.end(), h.begin(), h.end());
    data.insert(data.end(), bh.begin(), bh.end());
    const auto file = st(R"({"__metadata__":{"format":"pt"},)"
                         R"("a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},)"
                         R"("h":{"dtype":"F16","shape":[2],"data_offsets":[8,12]},)"
                         R"("b":{"dtype":"BF16","shape":[1,2],"data_offsets":[12,16],"extra":1}})",
                         data);
    const auto m = SafetensorsLoader::from_bytes(file);
    ASSERT_TRUE(m.has_value()) << m.error().to_string();
    ASSERT_EQ(m->size(), 3u);
    const auto& a = m->at("a");
    EXPECT_EQ(a.dtype(), DType::F32);
    EXPECT_EQ(f32_of(a), (std::vector<float>{1.5f, -2.0f}));
    EXPECT_EQ(a.buffer().alignment(), 64u); // Rust Tensor::from_f32
    EXPECT_FALSE(a.is_mmap());
}

TEST(Safetensors, half_types_stay_raw) {
    const auto file = st(R"({"h":{"dtype":"F16","shape":[2],"data_offsets":[0,4]},)"
                         R"("b":{"dtype":"BF16","shape":[2],"data_offsets":[4,8]}})",
                         [] {
                             auto v = le_u16({0x3C00, 0xC000});
                             const auto w = le_u16({0x3F80, 0xBFC0});
                             v.insert(v.end(), w.begin(), w.end());
                             return v;
                         }());
    const auto m = SafetensorsLoader::from_bytes(file);
    ASSERT_TRUE(m.has_value()) << m.error().to_string();
    const auto& h = m->at("h");
    EXPECT_EQ(h.dtype(), DType::F16); // NOT converted at load (unlike GGUF F16 → F32)
    EXPECT_EQ(h.buffer().alignment(), 16u);
    EXPECT_EQ(h.to_f32_vec(), (std::vector<float>{1.0f, -2.0f}));
    EXPECT_EQ(m->at("b").dtype(), DType::BF16);
    EXPECT_EQ(m->at("b").to_f32_vec(), (std::vector<float>{1.0f, -1.5f}));
}

TEST(Safetensors, f32_trailing_partial_chunk_is_dropped) {
    // Rust `raw.chunks_exact(4)`: 10 bytes → 2 floats, the 2 trailing bytes are ignored.
    auto data = le_f32({3.0f, 4.0f});
    data.push_back(0xAA);
    data.push_back(0xBB);
    const auto m = SafetensorsLoader::from_bytes(
        st(R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,10]}})", data));
    ASSERT_TRUE(m.has_value()) << m.error().to_string();
    EXPECT_EQ(f32_of(m->at("a")), (std::vector<float>{3.0f, 4.0f}));
}

TEST(Safetensors, scalar_shape_and_metadata_of_any_type) {
    const auto m = SafetensorsLoader::from_bytes(st(
        R"({"__metadata__":"not even an object","s":{"dtype":"F32","shape":[],"data_offsets":[0,4]}})",
        le_f32({9.0f})));
    ASSERT_TRUE(m.has_value()) << m.error().to_string();
    EXPECT_TRUE(m->at("s").shape().dims.empty());
    EXPECT_EQ(f32_of(m->at("s")), (std::vector<float>{9.0f}));
}

TEST(Safetensors, duplicate_keys_last_wins) {
    const auto m = SafetensorsLoader::from_bytes(
        st(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},)"
           R"("a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})",
           le_f32({1.0f, 2.0f})));
    ASSERT_TRUE(m.has_value()) << m.error().to_string();
    EXPECT_EQ(f32_of(m->at("a")), (std::vector<float>{2.0f}));
}

TEST(Safetensors, header_errors_match_rust) {
    const std::vector<uint8_t> four(4, 0);
    EXPECT_EQ(err({'a', 'b', 'c'}), "Safetensors parse error: file too short");
    {
        std::vector<uint8_t> f;
        put_le<uint64_t>(f, 100);
        f.push_back('{');
        f.push_back('}');
        EXPECT_EQ(err(f), "Safetensors parse error: header overflows file");
    }
    EXPECT_EQ(err(st(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,8]}})", four)),
              "Safetensors parse error: tensor 'a' data out of bounds");
    for (const auto& [dt, disp] : {std::pair{"I32", "i32"}, std::pair{"I64", "i64"},
                                   std::pair{"U8", "u8"}, std::pair{"BOOL", "bool"}})
        EXPECT_EQ(err(st(std::string(R"({"a":{"dtype":")") + dt +
                             R"(","shape":[1],"data_offsets":[0,4]}})",
                         four)),
                  std::string("Safetensors parse error: unsupported safetensors dtype '") + disp +
                      "' for tensor 'a'");
    EXPECT_EQ(err(st(R"({"a":{"dtype":"Q4","shape":[1],"data_offsets":[0,4]}})", four)),
              "Safetensors parse error: unknown dtype 'Q4'");
    EXPECT_EQ(err(st(R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}})", four)),
              "Safetensors parse error: Shape mismatch: expected [2], got [1]");
    EXPECT_EQ(err(st(R"({"a":{"dtype":"F16","shape":[2],"data_offsets":[0,2]}})", four)),
              "Safetensors parse error: Shape mismatch: expected [2], got [1]");
    EXPECT_EQ(err(st(R"({"a":{"dtype":"F32","shape":[0],"data_offsets":[0,4]}})", four)),
              "Safetensors parse error: Graph validation failed: Shape has zero dimension at axis 0");
    {
        std::vector<uint8_t> f;
        put_le<uint64_t>(f, 1);
        f.push_back(0xFF);
        EXPECT_EQ(err(f), "Safetensors parse error: invalid utf-8 sequence of 1 bytes from index 0");
    }
}

TEST(Safetensors, strict_stmeta_decoding_prefix_only) {
    // serde rejects all of these; the text after the prefix is C++-authored (not parity-bound).
    const std::vector<uint8_t> four(4, 0);
    const char* bad_entries[] = {
        R"({"a":{"dtype":"F32","shape":[1.0],"data_offsets":[0,4]}})",
        R"({"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}})",
        R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4,4]}})",
        R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0]}})",
        R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4.5]}})",
        R"({"a":{"shape":[1],"data_offsets":[0,4]}})",
        R"({"a":{"dtype":5,"shape":[1],"data_offsets":[0,4]}})",
        R"({"a":{"dtype":"F32","shape":"1","data_offsets":[0,4]}})",
        R"({"a":[1,2]})",
    };
    for (const char* h : bad_entries)
        EXPECT_TRUE(err(st(h, four)).starts_with("Safetensors parse error: tensor 'a': ")) << h;
    for (const char* h : {"[1]", "{", "nope", "{\"a\":}"}) {
        const std::string e = err(st(h, four));
        EXPECT_TRUE(e.starts_with("Safetensors parse error: ")) << h;
        EXPECT_NE(e, "<ok>") << h;
    }
}

TEST(Safetensors, malformed_ranges_panic_like_rust_slices) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    const std::vector<uint8_t> eight(8, 0);
    // start > end with end in range: Rust `&data_section[start..end]` panics.
    const auto f = st(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}})", eight);
    EXPECT_DEATH((void)SafetensorsLoader::from_bytes(f), "slice index starts at");
    // 8 + header_len wraps to 4 (<= file size): Rust's check passes, then `bytes[8..4]` panics.
    std::vector<uint8_t> g;
    put_le<uint64_t>(g, std::numeric_limits<uint64_t>::max() - 3);
    g.insert(g.end(), 8, 0);
    EXPECT_DEATH((void)SafetensorsLoader::from_bytes(g), "slice index starts at");
}

TEST(Safetensors, empty_file_is_too_short) {
    TempDir dir("st_empty");
    const auto p = dir.write("e.safetensors", {});
    EXPECT_EQ(SafetensorsLoader::load(p).error().to_string(), "Safetensors parse error: file too short");
}

TEST(Safetensors, missing_file_and_directory) {
    TempDir dir("st_missing");
    const auto p = dir.path() / "missing.safetensors";
    EXPECT_EQ(SafetensorsLoader::load(p).error().to_string(),
              "Model not found at path '" + sapient::io::display_path(p) + ": " +
                  sapient::io::rust_std::os_error_message(2) + "'");
#if !defined(_WIN32)
    // A directory opens and fails at mmap: SafetensorsParseError(e.to_string()) — NO prefix.
    const auto e = SafetensorsLoader::load(dir.path()).error().to_string();
    EXPECT_TRUE(e.starts_with("Safetensors parse error: ")) << e;
    EXPECT_NE(e.find("(os error "), std::string::npos) << e;
    EXPECT_EQ(e.find("mmap failed"), std::string::npos) << e;
#endif
}

TEST(Safetensors, load_copies_everything_out_of_the_mapping) {
    TempDir dir("st_load");
    const auto p = dir.write("w.safetensors",
                             st(R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})",
                                le_f32({5.0f, 6.0f})));
    auto m = SafetensorsLoader::load(p);
    ASSERT_TRUE(m.has_value()) << m.error().to_string();
    EXPECT_FALSE(m->at("a").is_mmap()); // Rust drops the Mmap at the end of load()
    EXPECT_EQ(f32_of(m->at("a")), (std::vector<float>{5.0f, 6.0f}));
    auto again = SafetensorsLoader::load_tensors(p);
    ASSERT_TRUE(again.has_value());
    EXPECT_EQ(again->size(), 1u);
}

TEST(Io, load_gguf_and_load_safetensors_delegate) {
    TempDir dir("io_lib");
    const auto sp = dir.write("w.safetensors",
                              st(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})",
                                 le_f32({1.0f})));
    auto s = sapient::io::load_safetensors(sp);
    ASSERT_TRUE(s.has_value()) << s.error().to_string();
    EXPECT_EQ(s->size(), 1u);
    sapient::io::test::GgufBuilder b;
    b.tensor("w", {2}, 0, le_f32({1.0f, 2.0f}));
    const auto gp = dir.write("w.gguf", b.build());
    auto g = sapient::io::load_gguf(gp);
    ASSERT_TRUE(g.has_value()) << g.error().to_string();
    EXPECT_EQ(g->size(), 1u);
    // The lib.rs re-exports exist at crate root.
    static_assert(std::is_same_v<sapient::io::GgufLoader, sapient::io::gguf::GgufLoader>);
    static_assert(std::is_same_v<sapient::io::SafetensorsLoader, SafetensorsLoader>);
}
```

Append to `cpp/libs/sapient-io/tests/real_file_test.cpp` (add `#include "sapient/io/safetensors.hpp"`):

```cpp
TEST(RealFile, safetensors_loads) {
    const char* env = std::getenv("SAPIENT_TEST_SAFETENSORS");
    if (env == nullptr || *env == '\0') GTEST_SKIP() << "SAPIENT_TEST_SAFETENSORS unset";
    const std::filesystem::path p(env);
    std::error_code ec;
    ASSERT_TRUE(std::filesystem::is_regular_file(p, ec)) << "SAPIENT_TEST_SAFETENSORS=" << env
                                                         << " is not a readable file";
    auto m = sapient::io::safetensors::SafetensorsLoader::load(p);
    ASSERT_TRUE(m.has_value()) << m.error().to_string();
    ASSERT_FALSE(m->empty());
    for (const auto& [name, t] : *m) {
        SCOPED_TRACE(name);
        EXPECT_FALSE(t.is_mmap());
        const auto dt = t.dtype();
        EXPECT_TRUE(dt == sapient::core::DType::F32 || dt == sapient::core::DType::F16 ||
                    dt == sapient::core::DType::BF16);
        EXPECT_EQ(t.bytes().size(), sapient::core::byte_count(dt, t.numel()));
    }
    std::printf("[real-file] %s: %zu tensors\n", env, m->size());
}
```

In `cpp/libs/sapient-io/CMakeLists.txt`: add `src/safetensors.cpp src/io.cpp` to `sapient_io`; add `target_link_libraries(sapient_io PRIVATE nlohmann_json::nlohmann_json)` (PRIVATE: no public header includes `<nlohmann/json.hpp>`); add `tests/safetensors_test.cpp` to `sapient_io_tests`.

- [ ] **Step 3: Run the build to verify it fails**

Run: `cd cpp && cmake --preset dev && cmake --build --preset dev`
Expected: FAIL — `sapient/io/safetensors.hpp: file not found`.

- [ ] **Step 4: Write `safetensors.hpp/.cpp` and `io.hpp/.cpp`**

`cpp/libs/sapient-io/include/sapient/io/safetensors.hpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-io/src/safetensors.rs minus `load_as_graph` (sub-project 8). Despite the
// Rust module doc, nothing is zero-copy: load() maps the file, copies every tensor out, and drops
// the mapping. The reachable dtype set is {F32, F16, BF16}; F16/BF16 keep their raw bytes.

#include <cstdint>
#include <filesystem>
#include <span>

#include "sapient/core/error.hpp"
#include "sapient/io/gguf.hpp"

namespace sapient::io::safetensors {

using TensorMap = gguf::TensorMap;

class SafetensorsLoader {
public:
    static sapient::core::Result<TensorMap> load(const std::filesystem::path& path);
    static sapient::core::Result<TensorMap> from_bytes(std::span<const uint8_t> bytes);
    /// Alias for `load`.
    static sapient::core::Result<TensorMap> load_tensors(const std::filesystem::path& path);
};

} // namespace sapient::io::safetensors
```

`cpp/libs/sapient-io/src/safetensors.cpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/io/safetensors.hpp"

#include <array>
#include <bit>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "sapient/core/buffer.hpp"
#include "sapient/core/dtype.hpp"
#include "sapient/core/panic.hpp"
#include "sapient/core/shape.hpp"
#include "sapient/core/tensor.hpp"
#include "sapient/io/mmap.hpp"
#include "sapient/io/rust_std.hpp"

static_assert(std::endian::native == std::endian::little,
              "sapient::io assumes a little-endian host (as the Rust crate does)");

namespace sapient::io::safetensors {

namespace core = sapient::core;

namespace {

core::Error st_err(std::string msg) {
    return core::Error::safetensors_parse(std::move(msg));
}

// Rust `#[derive(Deserialize)] struct StMeta` (safetensors.rs:21-26).
struct StMeta {
    std::string dtype;
    std::vector<size_t> shape;
    std::array<size_t, 2> data_offsets{};
};

/// serde's strictness: a JSON unsigned integer only (no float coercion, no negatives).
std::optional<std::string> as_usize(const nlohmann::json& j, size_t& out) {
    if (!j.is_number_unsigned())
        return std::string("invalid type: ") + j.type_name() + ", expected usize";
    out = j.get<size_t>();
    return std::nullopt;
}

/// serde_json::from_value::<StMeta>. Unknown fields are ignored (no deny_unknown_fields). The
/// messages are C++-authored: only the caller's "tensor '{name}': " prefix is parity-bound.
std::optional<std::string> decode_meta(const nlohmann::json& v, StMeta& m) {
    if (!v.is_object())
        return std::string("invalid type: ") + v.type_name() + ", expected struct StMeta";
    const auto dt = v.find("dtype");
    if (dt == v.end()) return std::string("missing field `dtype`");
    if (!dt->is_string()) return std::string("invalid type: ") + dt->type_name() + ", expected a string";
    m.dtype = dt->get<std::string>();
    const auto sh = v.find("shape");
    if (sh == v.end()) return std::string("missing field `shape`");
    if (!sh->is_array()) return std::string("invalid type: ") + sh->type_name() + ", expected a sequence";
    for (const auto& e : *sh) {
        size_t d = 0;
        if (auto err = as_usize(e, d)) return err;
        m.shape.push_back(d);
    }
    const auto off = v.find("data_offsets");
    if (off == v.end()) return std::string("missing field `data_offsets`");
    if (!off->is_array()) return std::string("invalid type: ") + off->type_name() + ", expected an array";
    if (off->size() != 2)
        return "invalid length " + std::to_string(off->size()) + ", expected an array of length 2";
    for (size_t i = 0; i < 2; ++i)
        if (auto err = as_usize((*off)[i], m.data_offsets[i])) return err;
    return std::nullopt;
}

/// Rust: `raw.chunks_exact(4)` → Vec<f32> → `Tensor::from_f32` (validate, then count check, then an
/// align-64 copy). One copy here with the same observables — incl. the dropped partial chunk.
core::Result<core::Tensor> f32_tensor(std::span<const uint8_t> raw, core::Shape shape) {
    SAPIENT_TRY(shape.validate());
    const size_t count = raw.size() / 4;
    if (count != shape.numel())
        return tl::unexpected(core::Error::shape_mismatch(shape.dims, {count}));
    SAPIENT_TRY_ASSIGN(auto buf, core::CpuBuffer::with_capacity(count * 4, 64));
    std::memcpy(buf->data(), raw.data(), count * 4);
    return core::Tensor::from_buffer(std::move(shape), core::DType::F32, std::move(buf), 0);
}

core::Result<core::Tensor> wrap(core::Result<core::Tensor> r) {
    if (!r) return tl::unexpected(st_err(r.error().to_string()));
    return r;
}

} // namespace

core::Result<TensorMap> SafetensorsLoader::load(const std::filesystem::path& path) {
    auto m = MappedFile::open(path);
    if (!m) {
        if (m.error().stage == MapStage::Open)
            return tl::unexpected(
                core::Error::model_not_found(display_path(path) + ": " + m.error().os.message));
        return tl::unexpected(st_err(m.error().os.message)); // no "mmap failed" prefix here
    }
    return from_bytes((*m)->bytes()); // every tensor is copied; the mapping dies with `m`
}

core::Result<TensorMap> SafetensorsLoader::from_bytes(std::span<const uint8_t> bytes) {
    if (bytes.size() < 8) return tl::unexpected(st_err("file too short"));
    uint64_t header_len64 = 0;
    std::memcpy(&header_len64, bytes.data(), 8);
    const auto header_len = static_cast<size_t>(header_len64);
    const size_t header_end = 8 + header_len; // wraps like Rust release
    if (header_end > bytes.size()) return tl::unexpected(st_err("header overflows file"));
    if (header_end < 8) // wrapped: Rust's check passed and `bytes[8..header_end]` panics
        core::panic("slice index starts at 8 but ends at " + std::to_string(header_end));
    const auto header = bytes.subspan(8, header_len);
    if (auto e = rust_std::utf8_error(header)) return tl::unexpected(st_err(*e));

    nlohmann::json root;
    try { // the only exception in sapient::io — caught here, never crosses the library boundary
        const char* first = reinterpret_cast<const char*>(header.data());
        root = nlohmann::json::parse(first, first + header.size());
    } catch (const nlohmann::json::exception& e) {
        return tl::unexpected(st_err(e.what()));
    }
    if (!root.is_object())
        return tl::unexpected(st_err(std::string("invalid type: ") + root.type_name() + ", expected a map"));

    const auto data = bytes.subspan(header_end);
    TensorMap tensors;
    for (const auto& [name, value] : root.items()) {
        if (name == "__metadata__") continue;
        StMeta meta;
        if (auto e = decode_meta(value, meta))
            return tl::unexpected(st_err("tensor '" + name + "': " + *e));
        core::DType dtype = core::DType::F32;
        if (meta.dtype == "F32") dtype = core::DType::F32;
        else if (meta.dtype == "F16") dtype = core::DType::F16;
        else if (meta.dtype == "BF16") dtype = core::DType::BF16;
        else if (meta.dtype == "I32") dtype = core::DType::I32;
        else if (meta.dtype == "I64") dtype = core::DType::I64;
        else if (meta.dtype == "U8") dtype = core::DType::U8;
        else if (meta.dtype == "BOOL") dtype = core::DType::Bool;
        else return tl::unexpected(st_err("unknown dtype '" + meta.dtype + "'"));

        const auto [start, end] = meta.data_offsets;
        if (end > data.size())
            return tl::unexpected(st_err("tensor '" + name + "' data out of bounds"));
        if (start > end) // Rust `&data_section[start..end]` panics
            core::panic("slice index starts at " + std::to_string(start) + " but ends at " +
                        std::to_string(end));
        const auto raw = data.subspan(start, end - start);
        core::Shape shape(meta.shape);

        core::Result<core::Tensor> t = tl::unexpected(core::Error::internal("unreachable"));
        switch (dtype) {
        case core::DType::F32: t = wrap(f32_tensor(raw, std::move(shape))); break;
        case core::DType::BF16: t = wrap(core::Tensor::from_bf16_bytes(raw, std::move(shape))); break;
        case core::DType::F16: t = wrap(core::Tensor::from_f16_bytes(raw, std::move(shape))); break;
        default:
            return tl::unexpected(st_err("unsupported safetensors dtype '" +
                                         core::to_string(dtype) + "' for tensor '" + name + "'"));
        }
        if (!t) return tl::unexpected(t.error());
        tensors.insert_or_assign(name, std::move(*t)); // last wins
    }
    return tensors;
}

core::Result<TensorMap> SafetensorsLoader::load_tensors(const std::filesystem::path& path) {
    return load(path);
}

} // namespace sapient::io::safetensors
```

**Check before moving on:** confirm nlohmann's `parse` keeps the **last** value for a duplicated object key — `Safetensors.duplicate_keys_last_wins` pins it. If it fails, the fix is a `parser_callback_t` that erases an existing key on `parse_event_t::key` at depth 1, not a change to the test.

`cpp/libs/sapient-io/include/sapient/io/io.hpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-io/src/lib.rs: the crate-root re-exports and the two convenience
// loaders. `load_graph` and `OnnxLoader` are sub-project 8.

#include <filesystem>

#include "sapient/core/error.hpp"
#include "sapient/io/gguf.hpp"
#include "sapient/io/safetensors.hpp"

namespace sapient::io {

using gguf::GgufLoader;
using gguf::GgufValue;
using safetensors::SafetensorsLoader;
using TensorMap = gguf::TensorMap;

/// `GgufLoader::load_tensors(path)` (the heap path).
sapient::core::Result<TensorMap> load_gguf(const std::filesystem::path& path);
/// `SafetensorsLoader::load_tensors(path)`.
sapient::core::Result<TensorMap> load_safetensors(const std::filesystem::path& path);

} // namespace sapient::io
```

`cpp/libs/sapient-io/src/io.cpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/io/io.hpp"

namespace sapient::io {

sapient::core::Result<TensorMap> load_gguf(const std::filesystem::path& path) {
    return GgufLoader::load_tensors(path);
}

sapient::core::Result<TensorMap> load_safetensors(const std::filesystem::path& path) {
    return SafetensorsLoader::load_tensors(path);
}

} // namespace sapient::io
```

- [ ] **Step 5: Build and run the full suite, then the safetensors real-file check locally**

Run: `cd cpp && cmake --preset dev && cmake --build --preset dev && ctest --preset dev`
Expected: PASS — 100% of the run suite; both `RealFile.*` SKIPPED.

Then: `SAPIENT_TEST_SAFETENSORS="$(find ~/.cache/huggingface -path '*kokoro-82m-safetensors*' -name model.safetensors | head -1)" ctest --preset dev -R '^RealFile\.safetensors_loads$' -V | grep -E '\[real-file\]|Passed|Failed'` (from `cpp/`). Expected: one `[real-file] …` line and `Passed`. Also run it on `models--HuggingFaceTB--SmolVLM-256M-Instruct/…/model.safetensors` if cached. Keep the lines for Task 5.

- [ ] **Step 6: Cross-compile** `src/safetensors.cpp`, `src/io.cpp` and `tests/safetensors_test.cpp` for x86_64 (the library command includes the nlohmann `-isystem`). Expected: exit 0.

- [ ] **Step 7: Format, SPDX-lint, commit**

```bash
git add cpp/cmake/deps.cmake cpp/third_party/LICENSES.md cpp/libs/sapient-io
.superpowers/tools-venv/bin/clang-format -i $(git diff --cached --name-only -- 'cpp/*.hpp' 'cpp/*.cpp')
git add cpp/libs/sapient-io
python3 cpp/scripts/check_spdx.py cpp crates
git commit -m "$(cat <<'EOF'
cpp(io): port the safetensors loader + lib.rs helpers; pin nlohmann/json 3.11.3

The header is decoded as strictly as serde did (unsigned integers only, exactly
two data_offsets, unknown fields ignored, __metadata__ skipped by name);
F16/BF16 keep their raw bytes, F32 is copied once into an align-64 buffer with
from_f32's check order. Every SAPIENT-authored error text is Rust's; only the
tail of an embedded serde_json message is C++-authored.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: docs, `PARITY.md` rows, spec as-built notes — sub-project 1a complete

**Files:**
- Modify: `CLAUDE.md`, `docs/ROADMAP.md`, `docs/PARITY.md`, `docs/PROJECT_GUIDE.md`, `CHANGELOG.md`, `docs/superpowers/specs/2026-09-21-cpp-sp1a-core-io-cpu-design.md`

**Interfaces:**
- Consumes: the `[real-file]` lines and suite counts recorded in Tasks 3 and 4, the commit hashes of Tasks 1–4 (`git log --oneline -5`).
- Produces: documentation only.

- [ ] **Step 1: `docs/PARITY.md`**

Under `## Sub-project 1a — core + IO + CPU kernels`, add a `### Deviations and faithful quirks recorded by plan B (io)` subsection listing, one bullet each: (1) hardening — no allocation from untrusted counts/lengths (Rust aborts; C++ returns Rust's own EOF text; every Rust-`Err` input gives the identical text); (2) zero-width array item types skip the no-op `count` loop (behaviour-identical, avoids a hang); (3) io's Q5_K dequantiser is core's per-element form, not the stale `qh[is/8]` copy (unreachable in both trees — `Q5_K` is kept as blocks); (4) io's whole-tensor dequant semantics kept (Q4_0/Q8_0 guarded by `numel`, K-quants `numel/256` blocks) over core's per-block functions; (5) safetensors F32 copied once instead of twice (same alignment, same check order, same values); (6) `MAP_SHARED` as memmap2 does (spec said `MAP_PRIVATE`; indistinguishable read-only); (7) faithful quirks reproduced on purpose: value types ≥ 13 and nested-array skips consume nothing, non-u32/f32/str arrays are `Other`, v1 accepted, `I32` alignment ignored, release-wrapping arithmetic, last-wins duplicates, and the panics (zero alignment, wrapped/negative slice ranges); (8) exempt texts — serde_json message tails and Windows OS descriptions.

Add Results rows (fill the C++ commit from Task 3/4 and the counts gtest printed):

| Date | Gate | Host / ISA path | Rust commit | C++ commit | Result |
|---|---|---|---|---|---|
| 2026-09-2x | sp1a plan B — 2 sapient-io Rust tests ported by name (`Gguf.q8_0_quantize_roundtrips_and_sizes`, `Gguf.q8_0_quantize_handles_all_zeros`) + the synthetic GGUF/safetensors/mmap/rust_std suites (N tests) | macOS arm64 (Apple M5) | 52bc163 | `<task 4 hash>` | pass — `cargo test -p sapient-io`: 2 passed; C++ suite `<registered>` registered / `<run>` run, 100% |
| 2026-09-2x | `quantize_to_q8_0` vs plan D's golden-gated `quantize_q8_0_block` (random blocks + inf/NaN inputs) | macOS arm64 | — | `<task 3 hash>` | bit-identical |
| 2026-09-2x | Error-text parity: every Result-path literal in the io tests generated from the real Rust crate (probe crate, 2026-09-23) | macOS arm64 | 52bc163 | `<task 4 hash>` | byte-identical |
| 2026-09-2x | `RealFile.gguf_heap_and_mmap_agree` on SmolLM2-135M-Instruct Q4_K_M and Qwen2.5-1.5B-Instruct Q4_K_M (heap vs mmap vs metadata-only) | macOS arm64 | — | `<task 3 hash>` | pass — paste both `[real-file]` lines |
| 2026-09-2x | `RealFile.safetensors_loads` on kokoro-82m `model.safetensors` (+ SmolVLM-256M if run) | macOS arm64 | — | `<task 4 hash>` | pass — paste the `[real-file]` line(s) |
| — | `RealFile.gguf_heap_and_mmap_agree` in CI `cpp-parity` (macOS + Linux, after the model download) | CI | — | — | not yet run — first push after plan B |

- [ ] **Step 2: `CLAUDE.md`** — after the "Sub-project 1a, plan E landed" bullet, add a "Sub-project 1a, plan B landed (`sapient::io`) — sub-project 1a complete" bullet: what shipped (the four modules, the five `GgufLoader` entry points, `SafetensorsLoader`, `load_gguf`/`load_safetensors`, nlohmann/json 3.11.3), the error-text parity rule and its exemptions, the open-vs-map wrapping table in one sentence, the memmap2 empty-file/directory semantics, the no-allocation-from-untrusted-fields carve-out, `from_f32` (align 64) vs `from_f32_vec`, the test-only `backends_cpu` link, the real-file gate + CI step, and lessons (at minimum: Windows runs ctest — plan E's "compile-only" was wrong; generate expected error literals from a Rust probe, never recall them). In the C++ rewrite section, replace the bullet `- Next: plan B (io) closes sub-project 1a; then sub-project 1b (`chat --prompt` vertical slice).` with `- Next: sub-project 1b (`chat --prompt` vertical slice).`, and in the plan-E bullet correct "Windows is compile-only in CI" to say the `cpp-build-windows` job builds AND runs ctest (the naming ruling itself stands). Update the existing `### GGUF loading (Phase 4: mmap)` paragraph's stale claim ("K-quants dequantized from mmap bytes") to say Q4_0/Q8_0/Q4_K/Q5_K/Q6_K are all zero-copy on the mmap path, citing `to_sapient_dtype`.

- [ ] **Step 3: `docs/ROADMAP.md`** — row `| 1a | Core + IO + CPU kernels | … |`: status → "complete — plans A, C, D, E, B implemented on feat/cpp-sp1a; next: sub-project 1b (CPU chat vertical slice)".

- [ ] **Step 4: `docs/PROJECT_GUIDE.md`** — in §6 "The C++ tree (in progress)", extend the paragraph: plan B added `sapient::io` (`cpp/libs/sapient-io/`: reading GGUF and safetensors files, either copied into memory or memory-mapped so the OS pages weights in on demand), closing out sub-project 1a; mention `SAPIENT_TEST_GGUF`/`SAPIENT_TEST_SAFETENSORS` for the real-file check. In §4's `sapient-io` entry, add one sentence that the C++ port lives in `cpp/libs/sapient-io/` (ONNX deferred to sub-project 8).

- [ ] **Step 5: `CHANGELOG.md`** — under `## [Unreleased]`, after the plan E entry, add `### 🧱 C++ rewrite — sub-project 1a, plan B (sapient::io GGUF + safetensors loaders)` with a 2–3 line summary ending "Sub-project 1a is complete."

- [ ] **Step 6: The spec's as-built notes** — in `docs/superpowers/specs/2026-09-21-cpp-sp1a-core-io-cpu-design.md`: §2.2 `mmap` row, append "**As built (plan B):** `MAP_SHARED` as memmap2 0.9.11 does (read-only: indistinguishable from `MAP_PRIVATE`); `open` reports the failing stage because the four Rust call sites wrap open and map failures differently; an empty file maps to an empty span." §2.2 `gguf` row, append "**As built (plan B):** Q5_K routes through core's per-element dequant; no `reserve` from header fields (Rust aborts, C++ returns Rust's EOF text)." §5 row B "Doc updates" cell → "**satisfied 2026-09-2x** — see `docs/PARITY.md` sub-project 1a Results; same doc set as A/C/D/E, plus PROJECT_GUIDE §6".

- [ ] **Step 7: Verify and commit**

Run: `cd cpp && ctest --preset dev` (docs-only change; must still be 100%) and `python3 cpp/scripts/check_spdx.py cpp crates`.

```bash
git add CLAUDE.md docs/ROADMAP.md docs/PARITY.md docs/PROJECT_GUIDE.md CHANGELOG.md docs/superpowers/specs/2026-09-21-cpp-sp1a-core-io-cpu-design.md
git commit -m "$(cat <<'EOF'
docs(sp1a plan B): parity rows, CLAUDE.md/ROADMAP/PROJECT_GUIDE/CHANGELOG, spec as-built

Sub-project 1a is complete. CONTRIBUTING.md and README.md need no change for a
library-internal plan.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
```
