# SAPIENT → C++ rewrite — design spec

> Status: **approved 2026-09-20** (brainstorming session, six scoping decisions below).
> Each sub-project in D5 gets its own implementation plan under `docs/superpowers/plans/`.
> Parity results are recorded in `docs/PARITY.md` as sub-projects land.

## Context

The user asked to "analyze and understand this project, then rewrite everything
as it is from scratch in C++". The repo (`Yash-Raw/sapient-cpp`) is a fresh copy
of the Rust SAPIENT edge-LLM inference engine (v0.6.0, 342 commits, last commit
2026-07-16). The repo name signals this repo is the intended home of the C++
version. Nothing C++ exists yet.

"As it is" = behavioural parity with the Rust build, not a redesign. The Rust
build is the correctness oracle for every C++ component.

## Inventory of what exists (Rust, measured 2026-09-20)

Total **54,201 LOC** across 152 `.rs` + 20 `.wgsl` files, 16 workspace members.

| Tier | What | Rust crates / files | ≈LOC |
|---|---|---|---|
| Core engine | Tensor/DType/Shape/Buffer (mmap + heap), GGUF + safetensors + ONNX loaders, CPU kernels (quant 3.8k, matmul 1.5k, attention, spinpool, thermal), `LlamaForward` (+MoE/GLM), `PhiForward`, `Gemma3Forward`, weights/gguf name mapping, sampler, `Pipeline`, KV cache | sapient-core, sapient-io, sapient-backends/cpu, sapient-models (forward/{llama,phi,gemma3,common,backend,mod}, weights, gguf_weights), sapient-generate (pipeline, sampler, kv_cache, device) | ~20k |
| GPU | `MlxForwardEngine` (macOS, mlx-rs), `WgpuForwardEngine` + wgpu backend crate + 20 WGSL shaders | sapient-backends/metal, sapient-backends/wgpu, forward/{mlx_engine,wgpu_engine} | ~4k + shaders |
| Text plumbing | HF tokenizer wrapper + Jinja chat templates (minijinja) + tool calling, HF Hub client + curated registry + ModelInfo, speculative decoding | sapient-tokenizers, sapient-hub, generate/speculative | ~5k |
| Audio | Whisper STT (CPU + wgpu), Kokoro-82M TTS (ALBERT/LSTM/iSTFT), Orpheus+SNAC TTS, mel front-end, VAD, mic/speaker (cpal), converse loop, LiveStt | sapient-audio, forward/{whisper,whisper_wgpu,kokoro/*,snac,conv}, generate/{transcribe,speak,kokoro_tts,converse,sentence} | ~7k |
| Vision | SigLIP tower + `VlmPipeline` (SmolVLM, Gemma3 multimodal) | forward/siglip, generate/vlm | ~1.5k |
| Product | CLI (20 subcommands), live Markdown TUI (termimad+syntect), OpenAI-compatible HTTP server (axum), `stats` monitor (sysinfo), self-updater, progress/ui | sapient-cli | ~7.5k |
| Embedding / distribution | UniFFI surface (Swift/Kotlin), packaging scripts, TS + RN SDKs, sample apps, CI/release workflows, Homebrew formula, install scripts | sapient-ffi, sdks/, examples/, scripts/, .github/ | ~1k Rust + non-Rust |
| Dead for live inference | IR graph path: sapient-ir / runtime / scheduler / telemetry, `architectures/*.rs` IR builders | 4 crates + architectures/ | ~3k |

### External dependency surface (each needs a C++ answer)

- **Compute:** rayon, matrixmultiply, half, memmap2, bytemuck
- **GPU:** wgpu 22 (+pollster), mlx-rs
- **Text:** HF `tokenizers` 0.23, `minijinja` (chat templates incl. `tojson`), `misaki-rs` (G2P for Kokoro)
- **Hub/HTTP:** hf-hub, reqwest (rustls), ureq, sha2, flate2/tar/zip (self-update)
- **Server:** tokio, axum, tower-http
- **Audio:** symphonia (WAV/FLAC/OGG/MP3/AAC/ALAC), rubato (sinc resample), realfft, cpal (mic/speaker), objc/block (macOS mic permission)
- **Image:** image (png/jpeg/webp)
- **CLI/TUI:** clap, rustyline, termimad, syntect, indicatif, console, sysinfo
- **FFI:** uniffi 0.29.3 (+ uniffi-bindgen-react-native lockstep)
- **Serde:** serde/serde_json, prost (ONNX)
- **Test:** criterion, proptest, approx

### Cross-cutting invariants the C++ tree must inherit

- Two-line SPDX header on every source file (AGPL-3.0-only OR commercial; OpenHorizon Labs).
- "Must follow" doc rules: PROJECT_GUIDE.md, CLAUDE.md, CONTRIBUTING.md, README.md, ROADMAP.md all need C++ rewrites — a real cost line.
- The hard-won traps in CLAUDE.md (Q6_K scale indexing, Q5_K per-element qh, llama q/k unpermute, RoPE axis, flash-attn masked-NaN, embedding row-gather, per-block activation scales, MoE routing order, Whisper explicit zero mask, Gemma3 arch-match order, …) each need a regression test ported.

## Decisions (six scoping questions, answered 2026-09-20)

1. Dependency policy — **answered 2026-09-20**: "convert the current codebase from
   Rust to C++, everything should work and stay the same, it's just a conversion".
   Interpretation (stated as an assumption in the plan): SAPIENT-authored code is
   hand-converted; third-party crates map to equivalent C++ libraries, not
   reimplemented. Scope = all 16 workspace members.
2. Scope boundary — implied by Q1: everything. Ordering of sub-projects is
   proposed by me (see Design), not asked.
3. Repo layout — **answered**: side-by-side. C++ tree at repo root, Rust `crates/`
   kept building as the oracle; Rust removed only after full parity in a final
   sub-project.
4. Toolchain — **answered**: C++20 + CMake ≥3.24 (presets, FetchContent, toolchain
   files for iOS/Android). `tl::expected` polyfill for `Result<T,E>` (std::expected is
   C++23). **Compiler: Clang 16+ everywhere, including clang-cl on Windows** — a
   constraint of the bit-identity goal: rustc is LLVM, so Clang reproduces its
   codegen/libm behaviour; GCC defaults to `-ffp-contract=fast` and MPFR-folds libm
   calls, MSVC lacks `__attribute__((target))`, has no ARM64 inline asm and uneven
   dotprod/i8mm intrinsics. GCC 12+ stays a *secondary, non-parity* build (Pi users).
5. GPU strategy — **answered**: MLX native C++ API for the Metal engine (drops the
   mlx-rs wrapper layer); **wgpu-native** (official C API of the same wgpu crate, same
   naga shader compiler) for the cross-platform engine — the 20 WGSL shaders ship
   unchanged. wgpu-native is consumed as a prebuilt opaque C library.
6. Dead IR path — **answered**: convert LAST as its own final sub-project (honours
   "everything", gates nothing user-visible, trivially droppable later).

## Survey findings (from the three exploration passes, 2026-09-20)

### Verification inventory (what the port must pass)

- **279 unit tests** in `crates/*/src` + **60 integration test fns** in 20 files. 19 are
  `#[ignore]` (real-model golden gates, need downloads / env vars). Densest bug-memory
  file: `crates/sapient-backends/cpu/src/kernels/quant.rs` (24 named regression tests).
- Tiers: T1 real CI gate = 272 unit + 34 integration; T2 vacuous-in-CI = 23 wgpu
  `resident.rs` kernel tests (need a GPU) + 1 network test; T3 ignored = 19; T4
  feature-gated (`wgpu_coherence.rs` 5, `whisper_wgpu_coherence.rs` 1 — CI never
  passes `--features wgpu` to `cargo test`); T5 orphaned = `tests/integration/mlp_test.rs`
  (never compiled). Don't port T5.
- **Oracle build flags matter for bit-identity:** Rust oracle is stock-ISA (no
  `target-cpu=native`; NEON via `#[target_feature]`), tests at `opt-level=1`,
  release `lto=thin, codegen-units=1, panic=abort`. A C++ build with `-march=native`
  or `-ffast-math` will not be bit-identical on Q*_K paths → forbid both.
- `-- --test-threads=1` is load-bearing: `tests/cpu_repack.rs` mutates process-global
  env `SAPIENT_NO_REPACK` mid-test. C++ harness must serialize that test or refactor
  the switch into a parameter.
- **Fixtures (all committed golden data = 719 KB):** `tests/fixtures/snac_decode.json`
  (267 KB, `snac_coherence.rs`) and `tests/fixtures/kokoro_hello.safetensors` (452 KB,
  `kokoro/stage_tests.rs`). Everything else golden is downloaded or produced by
  `scripts/convert_{snac,kokoro}_to_safetensors.py`.
- Benchmarks: `crates/sapient-backends/cpu/benches/cpu_ops.rs` (criterion, f32 only) +
  9 scripts (`scripts/bench*.py`, `benchmark*.sh`) — scripts drive the CLI/HTTP and are
  language-neutral.
- **CI (`ci.yml`, 9 jobs):** auto-fmt (commits back), clippy `-D warnings`, test-macos
  (macos-14 × {aarch64, x86_64-under-Rosetta}), test-linux (needs `libasound2-dev`),
  test-windows (build only), cross-rpi (`cargo check` on `ubuntu-24.04-arm`), docs,
  package-swift (macos-14, `--smoke` runs a binary), package-android, sdk-ts.
- **Release (`release.yml`, 10-build matrix):** 6 triples × {default, `mlx`→`-metal`,
  `wgpu`→`-gpu`} → `sapient-{triple}{suffix}.{tar.gz|zip}` + `.sha256`; `-metal` archive
  also ships `mlx.metallib`; mlx leg needs cmake + `MACOSX_DEPLOYMENT_TARGET=14.0` +
  `clang_rt.osx` link. Release runs zero tests.
- **SPDX header is unenforced** (152/152 files comply by convention). Add a real gate
  in the C++ CI.
- Untested Rust surface the port inherits as risk: `sapient-tokenizers` `tokenizer.rs`
  + `whisper.rs`, `sapient-runtime`, `sapient-telemetry`, `backends/metal`,
  `sapient-audio` capture/playback/permissions.

### Public API shape & concurrency model (what the C++ headers must mirror)

- **Core types:** `DType` (14 variants incl. `Q4_K_R4`/`Q6_K_R4` internal layouts; block bytes
  Q4_0=18, Q8_0=34, Q4_K=144, Q5_K=176, Q6_K=210; `#[non_exhaustive]`), `Shape(Vec<usize>)`,
  `Tensor{shape,dtype,strides,buffer: Arc<dyn Buffer>,offset}` with ~30 pub fns
  (`as_bytes` bounded by `byte_count` for quant dtypes, `to_f32_cow` the only `Cow`,
  `as_bytes_mut` only valid on heap buffers), `Buffer` trait (2 impls: `CpuBuffer` with
  manual aligned alloc, private `MmapBuffer` in sapient-io), `SapientError` (25 variants).
- **Two-layer backend abstraction, not one hierarchy:** `LlmBackendDispatch{Cpu,Metal}`
  is a per-op backend (12-method `LlmBackend` trait) used by `LlamaForward`/`PhiForward`/
  `Gemma3Forward`; wgpu is a *whole engine* variant `ForwardEngine::Wgpu` (no per-op
  variant; `LlmBackendKind::Wgpu` → `from_kind` silently yields Cpu). `ForwardEngine` is
  `enum {Llama, Phi, Gemma3, MlxLlama(cfg), Wgpu(cfg)}` with per-variant capability gaps
  (Gemma3 lacks `forward_all_logits*`/`embed`; MLX `truncate_cache` resets → 0).
- **MoE is not an enum:** `ModelInfo.moe: Option<MoeConfig>` + `is_moe_layer(idx)` predicate;
  experts live in the flat `HashMap<String,Tensor>` under formatted keys. (CLAUDE.md's
  `Ffn::{Dense,Moe}` wording is descriptive, not a type.)
- **10 traits;** only 6 used via `dyn` (`Buffer`, `Tts`, `TokenListener`, `Pass`,
  `Telemetry`, `BatchScheduler`) → C++ abstract classes; `LlmBackend`/`ExecutionBackend`
  are enum-dispatched in practice → `std::variant` or concrete classes.
- **Data-carrying enums → `std::variant`:** `ForwardEngine`, `AudioEngine{Whisper,WhisperWgpu}`,
  `LlmBackendDispatch`, `SamplingStrategy{Greedy,Temperature,TopK,TopP,Combined}`,
  `GgufValue` (17 variants), `ArchType` (14 unit + `Unknown(String)`), `BackendPlan`,
  `SapientError` ×2 (core 25-variant; ffi 4-variant with field `reason`), `WgpuError`.
- **Async is almost entirely a calling-convention artifact.** tokio runtime exists in exactly
  two places (`sapient-cli` `#[tokio::main]`, `sapient-ffi` private 2-worker runtime).
  Genuine async I/O = `HubClient` downloads only (+ axum `serve`). `Pipeline::generate/chat`
  are `block_in_place` wrappers; `*_stream` = `spawn_blocking` worker + `mpsc::channel(64)`
  of `String`; cancellation = receiver drop → `blocking_send` errs → loop breaks. Only real
  overlap: `ConversePipeline::respond_streaming` (LLM ‖ TTS ‖ playback via two unbounded
  channels). wgpu uses `pollster`, not tokio. **→ C++ core needs no async runtime: a thread
  pool + bounded MPSC queue per stream reproduces all of sapient-generate.** HTTP server
  and downloader pick their own threading.
- **Hard threading constraint:** MLX/Metal engine has thread-local command-stream state;
  `eval()` must run on the engine's creating thread (why `block_in_place` not
  `spawn_blocking`). C++: pin the Metal engine to one thread.
- **Streams carry errors in-band as text `"Error: {e}"`** (serve SSE clients + FFI see this).
  Keep for parity; note as a later improvement (`std::expected` item type).
- **rayon + custom spinpool coexist**, switched at runtime by thermal state (spinning
  workers don't shed heat). Spinpool used in exactly one place (`matmul.rs` GEMV decode
  path). C++ needs a parking pool AND a spinning pool with identical chunk geometry.
- **Error handling split:** typed `Result<T,SapientError>` in core/io/ir/backends/
  tokenizers; `anyhow::Result` in models/generate/hub/ffi/cli; ffi re-types by
  string-formatting. Port decision: one `std::expected<T, Error>` (with a `Context`/
  message-chain variant standing in for anyhow) all the way up; no exceptions across
  the C ABI.
- **Lock-poisoning recovery** (`unwrap_or_else(|e| e.into_inner())`) is load-bearing: a
  panic mid-generation must not kill `serve`. C++: exception-safe engine lock + catch at
  the request boundary.
- **Other Rust-isms:** `self: &Arc<Self>` receivers (→ `enable_shared_from_this`),
  builder-by-value chains (→ `&&`-qualified or mutating builders), `Box::leak` singleton
  spinpool, `unsafe impl Send/Sync` on `CpuBuffer`/`MmapBuffer`/`SpinPool`, shelling out
  to `sysctl` (→ `sysctlbyname`/`/proc/meminfo`), env knobs (`SAPIENT_*`) must be preserved,
  string-keyed weight maps (`format!("layers.{i}....")`) everywhere.
- **Chat templates are runtime Jinja2 programs** (minijinja with `tojson`) → C++ needs an
  HF-compatible Jinja engine. **Tokenizer** is HF `tokenizer.json` (BPE/ByteLevel +
  normalizers/pre-tokenizers/post-processors) loaded for BOTH safetensors and GGUF models
  (via `tokenizer_fallback_model`) → C++ needs a tokenizer.json-compatible tokenizer.
  These two are the highest-risk third-party gaps (see Design → library map).

### Distribution & docs coupling

- Artifact scheme is language-neutral: `install.sh`/`install.ps1`/`Formula/sapient.rb`/
  `update-homebrew-formula.sh` work unchanged for a C++ binary named `sapient` whose
  `--version` output contains "sapient". **The Formula is already broken independent of
  language**: its `post_install` calls `sapient completions …`, a subcommand that does not
  exist (verified: no `clap_complete`/`Completions` in `sapient-cli`), it pins `0.1.5`, and
  carries placeholder SHA256s. Fixing it is sub-project 9 scope; `completions` itself is
  new functionality, not parity.
- **Repo constant is split**: `SkidGod4444/sapient` (install scripts, Formula, SDK
  package.json, CHANGELOG) vs `openhorizon-labs/sapient` (`update.rs`). Pick one.
- **License-ID inconsistency** (pre-existing, not language-related): Formula, both SDK
  `package.json`, `docs/MOBILE.md` §4 say `GPL-3.0-only`; everything else `AGPL-3.0-only`.
  Fix while touching them.
- `NOTICE:63` references `Cargo.toml / Cargo.lock` as the dependency manifest → point at
  the C++ manifest.
- SPDX header text is `//`-comment, transplants verbatim to `.cpp/.h/.hpp/.metal/.wgsl`;
  need a `#` variant for CMake/shell/Python.
- `.cargo/config.toml` `-framework Metal/Foundation/…` link args → CMake
  `target_link_libraries`.
- **FFI/mobile breakage** when UniFFI goes away: `uniffi-bindgen` generates the Swift
  file, the Kotlin file (package `uniffi.sapient_ffi`, baked into `examples/android-chat`
  imports), the C header + modulemap; `package-android.sh` gates on
  `uniffi_sapient_ffi_fn` symbol count and JNA dep. RN SDK (`sdks/react-native`) is
  generated by uniffi-bindgen-react-native (JSI glue, TS, podspec). **TypeScript SDK
  (`sdks/typescript`) is pure HTTP → unchanged.** `examples/react-native-chat` is fully
  insulated by the Transport abstraction.
- Canonical FFI contract (language-neutral wording) is `docs/MOBILE.md` §3:
  `version`, `list_models`, `resolve_alias`, `LlmSession.load/chat/chat_stream/reset/
  transcript/model/backend_label/is_mmap`, `GenerationOptions`, `set_thermal_level`/
  `thermal_level`, `TokenListener.on_token -> bool`. Swift example symbols: `LlmSession`,
  `TokenListener`, `ThermalLevel`, `GenerationOptions`, `chatStream`, `transcript`,
  `backendLabel`, `setThermalLevel`, `reset`.
- Docs needing rewrite (🔴 sections): PROJECT_GUIDE §4 (crates), §5 (deps), §6 (build),
  §7 (dep graph); CONTRIBUTING prerequisites/build/checks/testing/lint/CI/releases;
  README "Rust API", "Build from Source", "Cross-platform GPU"; ROADMAP Phase 5;
  MOBILE §2, §4 (building), §5.1, §8. `docs/PI.md` is fully neutral.
- Scripts: `justfile`, `.githooks/pre-push`, `package-*.sh`, `bench_gpu_7_6.sh` reference
  cargo; `scripts/yank-all.sh` is crates.io-only → delete, don't port.
- CHANGELOG: `release.yml` awk-extracts the tagged version section → keep the
  `## [X.Y.Z] - date` heading contract.

## Design

### D0. Approach chosen (of three considered)

- **A. Bottom-up, layer by layer** — port crate by crate from `sapient-core` upward, gating
  each layer against Rust. Fully verifiable, but no runnable product until ~7 layers in.
- **B. Vertical slice first, then widen (chosen, hybridised with A)** — first ship
  `sapient chat --prompt` on one Q4_K_M GGUF, CPU-only, token-identical to Rust, using the
  *minimum* of every layer. This de-risks the two scariest third-party gaps (tokenizer,
  Jinja) in week one, proves the oracle harness, and gives a demo. Then widen each layer to
  full parity in the order below.
- **C. Mechanical file-by-file transliteration, test at the end** — rejected: nothing is
  verified until everything exists; the CLAUDE.md trap list shows these bugs are silent
  (coherent-but-wrong output) and only a running oracle catches them.

### D1. Verification backbone (designed first — everything else hangs off it)

The Rust build is the oracle. Every C++ artefact is gated at one of four tiers:

| Tier | Mechanism | Gate |
|---|---|---|
| **Kernel** | Port every named unit test 1:1 (same test name, same inputs). PLUS a cross-language **golden-dump harness**: a test-only Rust binary (`crates/sapient-backends/cpu/examples/dump_kernels.rs`, new, non-behavioural) writes seeded-random inputs + outputs for each quant/matmul/attention/rope kernel (22 cases through the public entry points; `.sapd` v1 format documented in the example header); C++ GoogleTests load and compare. **Dumps are generated in the same CI job on the same host, never committed** — NEON 4-lane accumulation and the scalar x86 path legitimately differ, so a Mac-made dump would fail on the Linux runner | **bit-identical** for integer/quant paths (SDOT/SMMLA/Q8_K/R4), `max_err` bound for f32 GEMM/attention (summation order) |
| **Engine** | `sapient run/chat --prompt … --raw` with greedy decoding from both binaries; compare token ids. Models reach the job via the **Rust binary's `pull` into the shared HF cache**, which the C++ binary then loads — doubling as the first cache-layout-compatibility test (sub-project 3 adds the C++ downloader) | **token-identical** for N≥64 tokens on: `qwen2.5-0.5b-q4`, `llama-3.2-1b` (tied Q6_K embed), `smollm2-135m-q4`, `phi-4-mini`, `gemma-3-1b`; MoE when hardware allows |
| **Product** | A **live TypeScript SDK smoke** (`SapientClient` against the C++ `serve` on smollm2 — the SDK's own `npm test` is a mocked local server, verified in `sdks/typescript/test/client.test.mjs:26`, so it is not a product gate) + `scripts/bench_serve.py` + `scripts/sts_test.py`; `install.sh` smoke; Swift `--smoke` | pass |
| **Fixture** | `tests/fixtures/snac_decode.json`, `kokoro_hello.safetensors` re-used byte-for-byte | existing bounds |

**Build-flag discipline (bit-identity depends on it):** no `-march=native`, no `-ffast-math`,
**`-ffp-contract=off`** (Clang defaults to `on`, GCC to `fast`; Rust never contracts
implicitly — explicit `vfmaq_f32`/`mul_add` intrinsics match on both sides). SIMD via
Clang `__attribute__((target("dotprod")))`/`("i8mm")`/`("avx2,fma")` functions + runtime
detection (`getauxval` Linux / `sysctlbyname` macOS / `IsProcessorFeaturePresent` Windows /
`__builtin_cpu_supports` x86), mirroring `is_*_feature_detected!`. **The per-ISA dispatch
table is mirrored exactly, not extended:** aarch64 has the NEON/SDOT/SMMLA K-quant kernels;
x86_64 has only `dot_f32_avx2` (matmul.rs:196) and `dot_q8_0_row_avx2` (quant.rs:543) —
every K-quant path on x86 is scalar in Rust and stays scalar in C++ until parity is
recorded. New x86 kernels are post-parity work.
Tests build at `-O1` like `[profile.test]`. `-fno-exceptions` is NOT used (exception-safety
replaces lock-poisoning recovery at request boundaries); no exception crosses the C ABI.

**Determinism notes:** sampler RNG is a hand-rolled xorshift (`sampler.rs:128`) → port
bit-exact, so seeded non-greedy runs are also comparable. libm (`expf`, `tanhf`) is the
platform's in both languages → identical on the same OS. `select_nth_unstable` tie order ≠
`std::nth_element` → only affects top-p on exact ties; document, don't chase.

**Harness location:** `cpp/tests/parity/` — `greedy_parity.sh` (builds both, runs the model
list, diffs), `golden/` dumps; `cpp/scripts/shader_sync.py` (WGSL copies identical until Rust
removal) and `cpp/scripts/check_spdx.py` (SPDX header gate).

### D2. Repository layout (side-by-side)

(`cpp/` is the working name for the root; it is a one-line rename if you prefer `src/`
or `engine/`.)

```
cpp/                                   ← the C++ SAPIENT (this is what ships)
  CMakeLists.txt  CMakePresets.json
  cmake/          toolchains/{ios,ios-sim,android-arm64,aarch64-linux}.cmake, deps.cmake
                  (FetchContent pins), spdx_check.cmake, warnings.cmake
  third_party/    FetchContent manifests only (no vendored source; pinned tags/hashes)
  libs/
    sapient-core/         include/sapient/core/*.hpp  src/*.cpp  tests/
    sapient-io/           …
    sapient-backends-cpu/ src/kernels/{quant,matmul,attention,rope,…}.cpp, spinpool, thermal
    sapient-backends-metal/   [SAPIENT_MLX]   (MLX C++)
    sapient-backends-wgpu/    [SAPIENT_WGPU]  shaders/*.wgsl (copy; embedded via CMake → .hpp)
    sapient-tokenizers/   tokenizer.json engine + minja chat templates + whisper tokenizer
    sapient-hub/          HF client (libcurl), registry, ModelInfo, cache layout == hf-hub's
    sapient-models/       forward/{llama,phi,gemma3,common,backend,whisper,kokoro/,snac,siglip,
                          mlx_engine,wgpu_engine,whisper_wgpu}, weights, gguf_weights
    sapient-audio/        io (dr_libs), mel (pocketfft), vad, capture/playback (miniaudio)
    sapient-generate/     pipeline, sampler, speculative, transcribe, speak, kokoro_tts,
                          converse, vlm, device, sentence, kv_cache
    sapient-ir/ sapient-runtime/ sapient-scheduler/ sapient-telemetry/   (sub-project 8)
  apps/sapient-cli/       main, server, markdown, stats, update, ui, progress, hub
  ffi/sapient-ffi/        include/sapient_ffi.h (C ABI), src/, swift/ (Package + wrapper),
                          kotlin/ (JNI + wrapper, package `uniffi.sapient_ffi` kept for
                          source-compat), rn/ (JSI module + TS mirror of the ubrn output)
  tests/parity/           see D1
crates/                                ← Rust oracle, untouched except two test-only examples (dump_kernels, greedy_ids) and two behaviour-identical clippy-1.98 lint fixes
sdks/typescript                        ← unchanged
sdks/react-native                      ← regenerated by hand from ffi/sapient-ffi/rn
scripts/, .github/, install.*          ← updated per sub-project 9
```

One CMake target per Rust crate, same name (`sapient::core`, `sapient::backends_cpu`…),
namespace `sapient::core` etc. One `.hpp/.cpp` pair per `.rs` module, **same file stem**
(`tensor.rs` → `tensor.hpp` + `tensor.cpp`) so a reviewer can diff Rust↔C++ side by side.
Feature flags → CMake options `SAPIENT_MLX`, `SAPIENT_WGPU`, `SAPIENT_AUDIO_IO` (default ON
for the CLI, mirroring `default = ["audio-io"]`), `SAPIENT_ACCELERATE`.

### D3. Conversion conventions (the "as it is" rules)

| Rust | C++ |
|---|---|
| `Result<T, SapientError>` / `anyhow::Result<T>` | `sapient::Result<T> = tl::expected<T, sapient::Error>`; `Error` = enum code (the 25 core variants + `Internal`) + message + optional cause chain (stands in for anyhow context). `SAPIENT_TRY(expr)` macro ≈ `?` |
| `Option<T>` / `&[T]` / `Vec<T>` / `String` | `std::optional` / `std::span<const T>` / `std::vector` / `std::string` (UTF-8) |
| `Arc<T>` / `Arc<Mutex<T>>` / `Box<dyn Trait>` | `std::shared_ptr<T>` / `std::shared_ptr<Locked<T>>` (mutex+value helper) / `std::unique_ptr<Interface>` |
| enum with data (`SamplingStrategy`, `GgufValue`, `ArchType`…) | `std::variant` + `std::visit`; `ForwardEngine`/`AudioEngine` = class wrapping a `std::variant` whose alternatives are `#if`-gated by the CMake options |
| trait used via `dyn` | abstract class with virtual methods (`Buffer`, `Tts`, `TokenListener`, `Pass`, `Telemetry`, `BatchScheduler`) |
| trait enum-dispatched (`LlmBackend`) | abstract class too (simpler than duplicating the enum); `LlmBackendDispatch` holds `std::unique_ptr<LlmBackend>` |
| closures `FnMut(&str)` etc. | `std::function` on cold paths; template param on hot paths (`generate_token_ids_streaming`) — mirrors generic-vs-dyn usage in Rust |
| `rayon::par_iter`/`join` | `sapient::parallel_for(n, chunk, f)` / `parallel_join(a, b)` on a parking pool; **chunk geometry identical to `gemv_chunk`** |
| `spinpool` | direct port (atomics, `Pad<T>` 128 B, seqlock, QoS pinning on macOS) |
| tokio `mpsc(64)` + `ReceiverStream<String>` | `sapient::Channel<std::string>` bounded MPSC; `TokenStream` with `next()`; drop = cancel (same semantics) |
| `#[cfg(feature="x")]` | `#if SAPIENT_HAS_X` from CMake-generated `config.hpp` |
| `#[test]` | GoogleTest in `libs/<lib>/tests/<module>_test.cpp`, same test names |
| `unsafe` NEON/`asm!("sdot")` | intrinsics (`vdotq_s32`, `vmmlaq_s32`) in Clang `target`-attributed functions (MSVC unsupported — clang-cl on Windows) |
| `include_str!(shader)` | CMake `file(READ)` → generated `shaders.hpp` |
| env knobs `SAPIENT_*` | same names, same defaults (perf parity depends on them) |
| SPDX header | same two lines on every `.hpp/.cpp/.mm/.wgsl`; `#` form on CMake/sh/py; **CI gate** (new) |

### D4. Third-party library map (Rust crate → C++ dependency)

| Need | Rust | C++ (pinned via FetchContent unless system) | Parity risk |
|---|---|---|---|
| JSON | serde_json | nlohmann/json | none |
| CLI parsing | clap | CLI11 | none (help text re-authored) |
| HTTP server + SSE + multipart | axum/tokio/tower-http | cpp-httplib (thread-per-connection; SSE via chunked content provider) | low |
| HTTP client / downloads | reqwest, ureq, hf-hub | libcurl (system or FetchContent) + hand-written HF Hub client reproducing hf-hub's on-disk cache layout (`models--org--name/snapshots/<rev>/`, `blobs/`, `refs/`) so existing downloads are reused | low–medium (cache-layout compat test) |
| Tokenizer | HF `tokenizers` | **own `tokenizer.json` engine** (ByteLevel BPE, SentencePiece-BPE w/ byte fallback, WordPiece; normalizers; pre-tokenizers; post-processors; decoders) + **PCRE2** (UTF+UCP) for pre-tokenizer regexes | **high** — gated by token-id parity over a multilingual corpus for every catalog tokenizer (Rust dumps ids) |
| Chat templates | minijinja (+`tojson`) | **minja** (ggml-org, header-only, built for HF templates) | **medium** — gated by render parity for every catalog template × message/tool fixtures |
| G2P (Kokoro) | misaki-rs | **own port of misaki (en)** — dictionary + fallback rules | **medium** — phoneme-string parity vs Rust over a sentence corpus |
| Parallelism | rayon | own parking pool (see D3) | none if chunking mirrored |
| mmap | memmap2 | `mmap`/`MapViewOfFile` wrapper | none |
| f16/bf16 | half | own conversions + F16C/NEON | none |
| f32 GEMM | matrixmultiply | own packed SGEMM (NEON/AVX2 microkernel), optional Accelerate cblas | `max_err` only |
| GPU (portable) | wgpu 22 | **wgpu-native** pinned to the v22 line (`webgpu.h` + `wgpu.h`) | low (same naga) |
| GPU (Apple) | mlx-rs | **MLX C++** (FetchContent `ml-explore/mlx`, pinned) | low–medium (API drift; `mlx.metallib` colocation unchanged) |
| Audio decode | symphonia | dr_wav, dr_flac, dr_mp3, stb_vorbis; AAC/ALAC via AudioToolbox (macOS) / Media Foundation (Windows); **Linux AAC/ALAC = decision in sub-project 5a** (FFmpeg dep vs unsupported) | medium (declare) |
| Resampling | rubato | own windowed-sinc resampler mirroring rubato `SincFixedIn` params (libsamplerate as fallback) | `max_err` |
| FFT | realfft | pocketfft (header-only) | `max_err` |
| Mic/speaker | cpal | miniaudio (single header; callback model like cpal) | none |
| macOS mic permission | objc/block | Objective-C++ `.mm` → AVCaptureDevice | none |
| Images | image | stb_image (png/jpeg) + libwebp | none |
| Line editor | rustyline | replxx or isocline — **requirement: bracketed-paste mode** (the v0.3.x paste bug) | low |
| Markdown TUI | termimad + syntect | md4c + own ANSI renderer (commit-and-preview logic ported); syntax highlighting = small keyword highlighter | **declared degradation**: colours won't match Sublime grammars |
| Progress/ANSI | indicatif/console | indicators + own helpers (`measure_text_width` port) | none |
| System stats | sysinfo | own per-OS (mach/sysctl, /proc, PDH/PSAPI) | none |
| Hashing/archives | sha2, flate2, tar, zip | own SHA-256; miniz-ng + own tar reader | none |
| Logging | tracing, log, env_logger | spdlog (or own minimal) honouring `RUST_LOG`-equivalent `SAPIENT_LOG` | none |
| Telemetry (dead path) | metrics/otel | no-op + console impls only | n/a |
| FFI codegen | uniffi (+ubrn) | hand-written C ABI + Swift/Kotlin/JSI wrappers with the **same public names** (`docs/MOBILE.md` §3 contract) | medium (examples must compile unchanged) |
| Tests / bench | cargo test, criterion, proptest | GoogleTest + ctest; Google Benchmark; hand-written randomized loops | none |
| Lint / fmt | clippy, rustfmt | clang-tidy (curated check set, warnings-as-errors) + clang-format (enforced in CI and pre-push) | none |

### D5. Sub-project decomposition and order

Each sub-project gets its own spec → plan → implementation cycle (brainstorming → writing-plans),
its own parity gate, and updates the five must-follow docs. Sizes are Rust LOC being ported.

| # | Sub-project | Ports | Gate | ≈LOC |
|---|---|---|---|---|
| 0 | **Scaffold + oracle harness** | `cpp/` CMake skeleton, presets, dep pins, SPDX gate, clang-format/tidy config, GoogleTest, `tests/parity/` scripts, Rust `dump_kernels` example, CI `cpp-*` jobs (build+test on macOS arm64 / Linux x86_64 / Windows clang-cl, `-ffp-contract=off` enforced), **third-party licence inventory** (every dep is MIT/BSD/public-domain → listed in `NOTICE`, replacing the `Cargo.lock` reference) | CI green on an empty lib | — |
| 1a | **Core + IO + CPU kernels** — spec: `2026-09-21-cpp-sp1a-core-io-cpu-design.md` | sapient-core (all), sapient-io (GGUF heap+mmap+metadata, safetensors, Q5_0→Q8_0), backends-cpu (all kernels, **per-ISA dispatch mirrored exactly**: aarch64 NEON/SDOT/SMMLA + Q8_K + R4 repack; x86_64 = scalar K-quants + the two existing AVX2 dots; spinpool, thermal, parking pool). **Moved to #8 (2026-09-21):** ONNX reader, the `Graph`-producing loader entry points, `backend.rs` + `pool.rs` (IR-path only), serde derives. Deviations approved: one shared dequant implementation; sgemm-backed f32 paths max-error gated | all 72+22+2 unit tests ported; golden-dump bit-identity on both arm64 and x86_64 hosts; `q6_k_scale_indexing_matches_ggml`, `q5_k_dequant_high_bits_per_element`, `leading_masked_positions_do_not_nan` etc. present by name | ~8k |
| 1b | **Vertical slice: CPU chat parity** | tokenizers (tokenizer.json engine, ChatML/builtins via minja, EOS candidates), hub (registry, ModelInfo from GGUF metadata + config.json, cache resolution — **no network yet**, loads from existing HF cache), models (weights, gguf_weights incl. unpermute + name map + `resolve_lm_head`, `LlamaForward` dense + KV cache Q8_0 + ctx cap, `embed_tokens` row-gather, `common.rs`), generate (`Pipeline` sync core, `Sampler` bit-exact xorshift, stop sequences, prefix cache, truncation flag), minimal CLI (`chat --prompt --raw`, `run --prompt`) | **greedy token-identical** vs Rust on qwen2.5-0.5b-q4, smollm2-135m-q4, llama-3.2-1b; tokenizer id-parity corpus; template render parity; `cpu_repack` bit-identical | ~6k |
| 2 | **Text engine completeness** | `PhiForward`, `Gemma3Forward` (norm folding, QK-norm, sliding/global, `from_hf_arch_name` order), MoE branch (Mixtral + GLM: `route_topk`, zero-copy `split_moe_gguf_experts`, sigmoid gate, shared expert, partial RoPE, `validate_moe_support`, mmap default), safetensors path + online Q8_0 + sharded index + `should_quantize_online`, hybrid split decision logic (CPU result for now), `SpeculativePipeline` (cache-aware verify, vocab guard, auto-draft), tool-calling templates, `builtin_template_for`, `tokenizer_fallback_model`, split-GGUF merge | all sapient-models unit + `moe_coherence`/`forward_test`/`inference_test` ported; greedy parity on phi-4-mini, gemma-3-1b; `moe_e2e` when a ≥32 GB box is available | ~7k |
| 3 | **Hub + CLI + serve** | `HubClient` (libcurl, parallel chunked download, gguf selection/override, split shards, token/login, `download_files`), all non-audio/vision CLI commands (chat REPL w/ line editor + Markdown renderer, pull, list, models (3 sections), rm, reset, info, backend-info, devices, stats, login, bench, bench-llm, inspect, update (variants, asset names, `REPO` unified), completions), `serve` (all `/v1/*` text endpoints, tool calling, `ModelCache` LRU + byte budget, `inference_sem`, prefix caching, SSE, `ServedModel` variant) | 39 CLI + 31 hub unit tests; **live TS-SDK smoke + `bench_serve.py` pass against C++ serve**; `install.sh` smoke; `sapient --version` contains "sapient" (the Formula's `completions` call targets a subcommand the Rust binary never had — Formula is pre-broken at `0.1.5` with placeholder hashes; adding `completions` is new scope, not parity) | ~10k |
| 4 | **GPU engines** | wgpu-native context (adapter probe, limits, pipeline cache, batching), `GpuBuffer`/`GpuQ8`/`Q4K`/`Q6K` uploads, all kernel launchers, `WgpuForwardEngine` (f16 packed KV, chunked prefill, transpose dance), `WhisperWgpuEngine` deferred to 5a; MLX C++ `MetalLlmBackend` + `MlxForwardEngine` (RoPE axis −2, fused SDPA, one `eval`/step, thread pinning, `SAPIENT_MLX_*` debug hooks, head_dim gate), `use_mlx_engine`/`use_wgpu_engine` Auto resolution, `device.rs` detection | 23 `resident.rs` tests (run on a real GPU, not vacuous), `wgpu_coherence` ×5, greedy parity vs Rust `-metal`/`-gpu` builds | ~5k + shaders |
| 5a | **Speech-to-text** | sapient-audio (decode, resample, mel, wav, VAD), `WhisperTokenizer`, `WhisperForward` + `AudioEngine`, `WhisperWgpuEngine`, `TranscribePipeline` (streaming, timestamps, beam, suppress tokens, `WhisperGenConfig`), `LiveStt`, CLI `transcribe`, serve `/v1/audio/transcriptions` | 13 audio + `mel_tone` + `whisper_coherence` ×4 + `whisper_wgpu_coherence`; `transcribe_e2e` (JFK → "country"); `live_stt_e2e` | ~4k |
| 5b | **Text-to-speech** | `SnacDecoder` + `normalize_snac_weights` + `SpeakPipeline` (prompt format w/ BOS, de-framing, streaming stable-prefix), Kokoro (albert, predictor, text_encoder, decoder, ops, loader, id bounds), **misaki G2P port**, `KokoroTts`, CLI `speak` (+play), serve `/v1/audio/speech` (+stream) | `snac_coherence` (fixture), 5 Kokoro stage tests (fixture), 9 `kokoro/ops` tests, phoneme parity corpus, **speak→transcribe round-trip** ("Hello there.") | ~4k |
| 5c | **Converse (speech-to-speech)** | `MicCapture`/`SpeakerPlayback` (miniaudio), permissions (.mm), `EnergyVad` live taps, `SentenceChunker`, `ConversePipeline` (streaming overlap, barge-in, pre-roll), CLI `converse` UX (meter, banner, breakdown) | unit tests; `sts_test.py`; manual live loop on macOS | ~1.5k |
| 6 | **Vision** | `SiglipVision`, pixel-shuffle, `VlmPipeline` (SmolVLM + Gemma3 connector, embedding splice), image preprocessing, CLI `see` | `vlm_e2e` (red), `vlm_geometry_probe`, `gemma3_e2e` multimodal | ~1.5k |
| 7 | **FFI + mobile + RN** | C ABI (`sapient_ffi.h`), Swift package (same API names, XCFramework via CMake toolchains + `xcodebuild -create-xcframework`), Kotlin JNI module (package kept), **RN JSI module hand-written to replicate ubrn's generated surface** (TS types, Promise marshalling for `loadSession`/`chat*`, the `onToken → bool` cancel callback crossing JSI on the JS thread, podspec, Android CMake), `package-swift.sh`/`package-android.sh` rewritten (symbol gates re-targeted, libc++ static + `-lc++abi` kept), `set_thermal_level`, `set_cache_dir`, async variants via thread pool | `--smoke` runs; `examples/swift-chat`, `android-chat`, `react-native-chat` **compile unchanged**; e2e smollm2 stream smoke; simulator/emulator inference turn on wgpu-native | ~1k Rust, but **≈4–5k C++/Swift/Kotlin/TS of binding code** — UniFFI+ubrn generated all of it; this is the most under-represented row by Rust LOC |
| 8 | **Dead IR path** | sapient-ir (Graph, passes, shape inference), runtime, scheduler, telemetry, `architectures/*` builders, ONNX → Graph | their 8 unit tests + revived `mlp_test` | ~3k |
| 9 | **Docs, release, Rust removal** | Rewrite 🔴 sections of PROJECT_GUIDE/CONTRIBUTING/README/ROADMAP/MOBILE/CLAUDE.md for C++; `NOTICE:63`; license-ID fixes (GPL→AGPL in Formula/SDK/MOBILE); unify `REPO`; `release.yml` CMake matrix producing identical artefact names (+`mlx.metallib`), `ci.yml` C++ jobs replace Rust jobs; `.githooks/pre-push` → clang-format/tidy; `justfile`; delete `scripts/yank-all.sh`, `.cargo/`, `Cargo.*`, `crates/` **only after** every gate in 1–8 has passed and the parity records are committed under `docs/PARITY.md` | full CI + release dry-run; `install.sh` end-to-end | docs |

Order rationale: 0→1a→1b is the vertical slice; 2→3 completes the text product (the
TypeScript SDK becomes a free regression suite at #3); 4 before audio because
`WhisperWgpuEngine` needs the wgpu layer; 5a→5b→5c is the STT→TTS→loop dependency chain;
6 and 7 are independent of each other and can run in parallel; 8 and 9 close out.

### D6. Error handling, threading, and lifetime decisions

- **One error type** `sapient::Error{code, message, cause}` end to end; models/generate/
  hub/cli use the same type (anyhow context → `cause` chain via `.context("…")`). The FFI
  maps to the 4-variant `SapientError{reason}` exactly as today.
- **No async runtime in the core.** Streaming = worker thread + bounded channel; cancel =
  drop the receiver. `serve` = cpp-httplib's thread pool; downloads = libcurl multi.
  `ConversePipeline::respond_streaming` keeps its two-channel overlap.
- **Engine lock** `std::shared_ptr<Locked<ForwardEngine>>`; a throw inside generation is
  caught at the `Pipeline` boundary and returned as `Error` (replaces poisoning recovery).
- **MLX engine is thread-pinned**: `MlxForwardEngine` owns a dedicated worker thread and a
  request queue; all `eval()` happen there.
- **Buffers:** `CpuBuffer` = aligned `operator new` with explicit alignment + size, RAII;
  `MmapBuffer` RAII over mmap/MapViewOfFile; `Buffer` interface identical; `Tensor::as_bytes`
  bounded by `byte_count` for quant dtypes exactly as Rust (the zero-copy MoE views rely on it).
- **Weight maps** stay `std::unordered_map<std::string, Tensor>` keyed by the same formatted
  names (string-keyed lookups are part of "as it is"; a later optimisation can index them).

## Verification

See D1 for the four-tier backbone and D5 for per-sub-project gates. In addition:

- **CI (from sub-project 0):** `cpp-lint` (clang-format --dry-run + clang-tidy + the SPDX header
  gate over `cpp/**` + the shader-sync diff of WGSL copies vs `crates/`), `cpp-test-macos`
  (macos-14, arm64), `cpp-test-linux` (ubuntu, `libasound2-dev`), `cpp-build-windows`,
  `cpp-parity` **on both macos-14 (arm64, NEON path) and ubuntu (x86_64, scalar/AVX2 path)**:
  builds both languages in the same job, pulls `smollm2-135m-q4` with the Rust binary,
  regenerates the kernel golden dumps on that host, then greedy-diffs the C++ binary against
  the Rust one. GPU tiers stay local/manual, as today.
- **Local dev loop:** `cmake --preset dev && ctest --preset dev` (+ `--test-threads`-equivalent:
  the `cpu_repack` env-var test is serialised via a ctest `RESOURCE_LOCK`).
- **Definition of done for the whole programme:** every table row in D5 has its gate recorded
  in `docs/PARITY.md` with model, commit hashes of both trees, token count, and result; the
  release matrix produces all 10 artefacts from `cpp/`; `crates/` deleted in the final PR.

## Risks and declared deviations

1. **Tokenizer engine** is the largest hand-written third-party replacement (~3–5k LOC).
   Mitigation: id-parity corpus per catalog tokenizer; start in 1b with ByteLevel BPE only
   (Qwen/SmolLM/Llama-3), add SentencePiece-BPE (Mistral/Gemma) in #2.
2. **minja coverage** of exotic templates (Phi-4, GLM, tool-calling Qwen). Mitigation: render
   parity fixtures in #1b/#2; fall back to `builtin_template_for` arms as Rust already does.
3. **Syntax highlighting** in the chat TUI will not match syntect colours — visual only.
4. **AAC/ALAC on Linux** — decide in 5a (FFmpeg dependency vs "unsupported on Linux").
5. **misaki G2P** fidelity — phoneme parity; any divergence changes Kokoro output audibly.
6. **wgpu-native / MLX version drift** — pin both; upgrade only with the coherence gates.
7. **Kotlin package name** `uniffi.sapient_ffi` is kept for source compatibility even though
   UniFFI is gone; revisit after the examples pass.
8. **Effort:** ~54k Rust LOC → an estimated 65–80k C++ LOC plus the hand-written replacements
   (tokenizer, HF client, G2P, sysinfo, archive readers). This is a multi-month programme;
   sub-project 1b is the first point where value is demonstrable.
9. **Doc rules cost:** every sub-project must update PROJECT_GUIDE, CLAUDE.md, CONTRIBUTING,
   README, ROADMAP — budget it per sub-project, not at the end.

## Next steps

1. Sub-project 0 (scaffold + oracle harness) — implementation plan via `superpowers:writing-plans`.
2. Sub-project 1a, then 1b (the vertical slice). Each later sub-project re-enters
   brainstorming for its own spec, referencing this document for the programme-level
   decisions.
