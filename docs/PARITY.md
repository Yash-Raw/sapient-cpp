# Parity ledger — Rust oracle vs C++ port

Every sub-project of the C++ rewrite (spec: `docs/superpowers/specs/2026-09-20-cpp-rewrite-design.md`,
§D5) records its gate results here. A row is only added after the gate ran on real hardware; a gate
that could not run is recorded as such, never as a pass. Commit hashes refer to this repository.

Conventions: **bit-identical** = byte-equal output; **token-identical** = equal `prompt_ids` and
`output_ids` from `cpp/tests/parity/greedy_parity.sh`; **max_err** = largest absolute difference.

## Harness self-validation (sub-project 0)

| Date | Gate | Host / ISA path | Rust commit | C++ commit | Result |
|---|---|---|---|---|---|
| 2026-09-20 | `.sapd` format round-trip: Rust `dump_kernels --format-sample` → C++ `sapient::testing::read_golden` (`Golden.ReadsFormatSampleWrittenByRust`) | macOS arm64 (Apple M5; NEON/SDOT) | 34d8449 | 2110b99 | pass — 7 arrays decoded exactly (fixture 306 B, byte-identical to a fresh tool run) |
| 2026-09-20 | Kernel dump determinism: two `dump_kernels --out` runs, same seed | macOS arm64 (Apple M5; NEON/SDOT) | 34d8449 | — | bit-identical across 3 runs, 22 cases |
| 2026-09-20 | `Golden.InventoryFromEnv` over the 22 host dumps | macOS arm64 (Apple M5; NEON/SDOT) | 34d8449 | 2110b99 | pass |
| 2026-09-20 | `greedy_parity.sh --self-check` smollm2-135m-q4, 32 tokens | macOS arm64 (Apple M5; NEON/SDOT) | 43c61da | — (Rust vs Rust) | token-identical |

## Sub-project 1a — core + IO + CPU kernels

### Open gaps carried from sub-project 0

- (a) The golden dumps cover the 22 public entry points plus, as of plan A, the R4 **dequant** path
  (`dequant_q4_k_r4`/`dequant_q6_k_r4`), as of plan C `apply_rope_partial(_scaled)`, the
  masked-attention branch, the two float GEMV paths, `layer_norm`, `reduce`, `gelu`, `softmax`
  axis 0, `log_softmax` and `conv2d`, and as of plan D the activation quantisers
  (`quantize_row_to_i8_blocks`, `i8_block_sums`, `quantize_row_to_q8k`), both repacks, `matmul_nt`
  over Q4_0/Q5_K/Q8_0 (m=8, the blocked GEMM) and the R4 layouts at m ∈ {1,2,3,8}, the twelve
  `_q8k_off` twins, and the odd-length dense cases (75 dumps: 63 default + 12 knob-off) — **closed**.
- (b) The FMA-contraction detector's x86 twin (`BuildFlags.FpContractIsOffUnderFmaTarget`) is
  exercised only on the Linux/Windows CI hosts — this repo's dev/test host is arm64, where the
  `#if defined(__x86_64__) || defined(_M_X64)` block compiles out entirely.
- (c) The `sapient::testing` golden reader has no `expect_bit_identical`/`max_abs_err`/f16-tag
  helpers yet. — **closed by plan A (`compare.hpp`)**: `sapient::testing::{bit_identical,
  max_abs_err, within_abs, SAPIENT_GOLDEN_CASE}`.

### Gaps closed and opened by plan E (spinpool + thermal)

- Plan C landed `thermal`/`spinpool` as **inert stubs** matching plan E's signatures — the
  types existed and `gemv_chunk` already consulted `enabled()`/`parallelism()`, but nothing
  was real: no sysfs governor, no worker threads, and `for_each_out_chunk` dispatched
  unconditionally through `parallel::par_chunks_mut`, never through a pool — **closed by plan
  E**: `ThermalGovernor` and `SpinPool` are full ports of `thermal.rs`/`spinpool.rs`, and
  `for_each_out_chunk` now branches on `spinpool::enabled()` — both routes proven bit-identical
  (see Results below).

New gaps opened by plan E, recorded honestly:

- **The bit-identity claim for the two golden ctest entries (`spinpool_on`/`spinpool_off`) is
  measured only over a degenerate partition.** `SpinPool::run` has a serial fast path: when
  `n_chunks == 1` it runs the callback inline on the caller and never touches the publish mutex,
  the seqlock handoff, `execute_blocks`, the workers, or the completion barrier
  (`spinpool.cpp:191-195`). Every golden `matmul_nt` case has `n ∈ {8, 16}`, and `gemv_chunk`
  floors its chunk size at 16 on all three branches (`matmul.cpp:699-709`, including the R4
  sites, which pass `gchunk*4 == 16`) — so every gated golden dispatch has `n_chunks == 1`. This
  was measured, not inferred: running the `spinpool_on` filter with
  `SAPIENT_SPINPOOL_DEBUG=1 --gtest_repeat=400` emits only
  `[spinpool-debug] spin=2000 rayon=0 chunk=16 len=8 n_chunks=1` (and its 4000/6000/8000 twins) —
  `rayon=0` proves the pool route was taken, `n_chunks=1` proves the partition was degenerate. So
  the two golden entries prove route *selection* and result identity over the one-chunk
  partition; they do not exercise the pool's parallel machinery (multi-chunk claiming, the
  seqlock handoff, worker wake/park). That machinery genuinely IS gated elsewhere, on every
  platform regardless of `enabled()`: the `Spinpool.*` unit tests drive `pool.run` directly with
  up to 61 chunks, 4 concurrent publishers, and constant park/wake
  (`Spinpool.rapid_ops_with_constant_parking` on its own `create(4, 50)` pool); and
  `Matmul.matmul_nt_q8_0_gguf_dimflip_matches_float` (`matmul_test.cpp:329-347`,
  `out_features = 64`) dispatches `chunk=16 len=64 n_chunks=4` through the pool under the ambient
  macOS default — a gate-strength and record gap, not a coverage hole.
- **The spin pool's *default-on* configuration and macOS-only mechanisms are macOS arm64 only;
  the pool route itself is exercised on every CI target.** The env override beats the platform
  default everywhere: a non-null `SAPIENT_SPINPOOL` short-circuits the `#if` platform block
  entirely (`spinpool.cpp:275-276`), and the `spinpool_on` ctest entry sets
  `SAPIENT_SPINPOOL=1` (`CMakeLists.txt:47`) — so `cpp-test-linux`, `cpp-build-windows`, and
  both `cpp-parity` legs run that entry through the pool too, spawning real workers, and on
  Linux executing the `__linux__` `pthread_setname_np` arm (`spinpool.cpp:83-85`). The
  `Spinpool.*` unit tests reach the non-macOS block-size arm the same way: they call
  `pool.run(n, …)` with `n` up to 61 on every platform, so on a Linux runner they execute
  `default_block = max(n_chunks / (3 · participants), 1)` (`spinpool.cpp:230`). What genuinely
  IS macOS-arm64-only is the *default-on* configuration (`enabled()` returning true with no env
  var set) and `pin_qos_user_interactive` (`spinpool.cpp:66-70`). `SAPIENT_SPINPOOL_BLOCK`'s
  non-macOS default (`n_chunks / (3 · participants)`, the homogeneous-server-ARM block size)
  still never executes *on this host* — this Mac cannot compile the `__linux__` arm at all, and
  without an explicit `SAPIENT_SPINPOOL_BLOCK` override the macOS branch always takes `block = 1`.
- **No performance claim is made or measured.** `Spinpool.DISABLED_pool_speedup_probe` exists
  (ported from Rust's `#[ignore] pool_speedup_probe`) but gates nothing — it is not part of any
  ctest entry's filter. Spec §2.3/§7 and the programme spec explicitly defer performance past
  sub-project 1a.

### Known oracle defects the port reproduces on purpose

| Found | Where (Rust) | Defect | Port | Status |
|---|---|---|---|---|
| 2026-09-21 (plan C) | `crates/sapient-backends/cpu/src/kernels/matmul.rs` `dot_f32_x_f16_neon` (aarch64; reached by `matmul_nt` at m=1, k≥64 with F16 weights) | The NEON f16→f32 bit-surgery shifts the unmasked u16 right by 10, so the f16 sign bit lands in bit 5 of the exponent field: negative weights decode ×2^32 (standalone probe: `[1,1,1,1]·[-1,1,-2,0.5]` = −1.29e10 instead of −1.5); subnormal/inf/NaN also decode differently from the scalar tail. Dormant in production because F16 linears are online-quantised to Q8_0 at load and Rust's own unit test uses k=2 (< 64). | Reproduced verbatim (spec §3.5); pinned bit-exactly by `matmul_nt_f16_m1` (random signed weights) | Open — Rust is frozen during the port; fix both trees together after parity, or document as a limitation. The user decides. |
| 2026-09-21 (plan D) | `crates/sapient-backends/cpu/src/kernels/quant.rs` `dot_q8_0_row_avx2` (x86_64 with AVX2+FMA; reached by every Q8_0 `matmul_nt` on x86 and by `dot_q8_0_row_f32`) | For every 8-quant group `g` the second activation load is `_mm256_loadu_ps(xp + g*8 + 4)` — 8 floats, of which only the first 4 pair with quants (the other 4 lanes of `_mm256_cvtepi8_epi32(_mm_loadu_si32(…))` are zero). At `g == 3` the load runs 4 floats past the 32-element block, i.e. **4 floats past the end of the activation slice on the row's last block** (an out-of-bounds read whose values only ever multiply zero). | Ported with the second half loaded in-bounds and zero-extended (`_mm256_insertf128_ps(_mm256_setzero_ps(), _mm_loadu_ps(…), 0)`): bit-identical for finite activations (`0·x = ±0` never changes a `+0.0` accumulator lane); a non-finite activation in those positions can yield `NaN` in Rust (`0·inf`) and `±inf` in C++. Gated by the x86 `cpp-parity` job's `matmul_nt_q8_0_m{1,3,8}` and `dot_q8_0_row_f32` cases (finite inputs). | Open — Rust is frozen during the port; the Rust fix is a 4-float load (`_mm_loadu_ps` zero-extended) or an `x` slice of `k + 4`. The user decides. |

### Deviations and faithful quirks recorded by plan B (io)

Deviations from a literal line-by-line port of `sapient-io`:

1. **Hardening — no allocation from untrusted counts/lengths.** Rust's `HashMap::with_capacity(kv_count)`, `Vec::with_capacity(tensor_count)`, `Vec::with_capacity(n_dims)`, and `vec![0u8; len]` for a string all *abort the process* on an absurd header value. C++ never `reserve`s from a header field and checks a string's `len` against the remaining bytes before allocating: for every input where Rust returns an `Err`, C++ returns the byte-identical error text; for inputs where Rust aborts on allocation, C++ returns Rust's own `read_exact` EOF text instead.
2. **Zero-width array item types skip the no-op `count` loop.** An array whose item type consumes no bytes (unknown type, or nested array — type 9) must not loop `count` times doing nothing; skipping the loop is behaviour-identical (zero bytes consumed either way) and avoids a hang on a `count` of 2⁶⁰.
3. **io's Q5_K dequantiser is core's per-element form, not the stale `qh[is/8]` copy.** `sapient-io`'s own `dequantize_q5_k` still carries the historical per-32-subblock `qh` bug that `sapient-core`'s `Tensor::to_f32_vec` fixed; the port uses `sapient::core::dequant`'s corrected per-element form for both loaders. Unreachable in both trees today — `GgmlType::Q5_K` is a *kept* type (`to_sapient_dtype` returns `Some`), so neither loader's `dequantize_to_f32` path is ever asked to dequantize a Q5_K tensor; the stale Rust copy and the fixed C++ port only diverge if that changes.
4. **io keeps its own whole-tensor dequant semantics over core's per-block functions.** `sapient::core::dequant::q*` panics when `bytes` holds more blocks than `numel`. io's own semantics are looser: Q4_0/Q8_0 run `bytes.size() / block_bytes` blocks and silently skip writes past `numel`; K-quants run `numel / 256` blocks. So `gguf::detail::dequantize_q*` calls core's **per-block** functions in a loop, never the whole-tensor ones — reproducing io's looser bounds behaviour while reusing core's corrected arithmetic.
5. **Safetensors F32 is copied once, not twice.** Rust's safetensors F32 path copies out of the raw JSON-adjacent buffer and then copies again into the tensor; C++ copies once into an aligned `from_f32` allocation with the same alignment (64) and the same validate→count→compare check order — same values, one fewer copy.
6. **`MAP_SHARED`, as memmap2 0.9.11 actually uses, not `MAP_PRIVATE` as originally specified.** For a read-only mapping the two are indistinguishable, so this changes nothing observable; recorded because the design spec's §2.2 `mmap` row said `MAP_PRIVATE`.
7. **Safetensors iteration order.** Rust iterates the header `HashMap` (nondeterministic order); C++ iterates nlohmann's sorted object map (deterministic order by key). When a header has several independently-bad entries, *which* `Err`/panic surfaces first can differ between the two trees; the set of accepted files and every successful result are identical either way. (GGUF iterates tensor infos in file order in both trees — no difference there.)
8. **`std::bad_alloc` from nlohmann on a huge (but file-size-bounded) header terminates the process** — the C++ twin of Rust's allocation-failure abort on an equivalently absurd header.

Faithful quirks reproduced on purpose (not deviations — pinned by tests so a future change doesn't accidentally "fix" them):

- A metadata value type ≥ 13 decodes to `Other` and consumes zero bytes; `skip_value` of an unknown or nested-array (type 9) item type also consumes zero bytes; arrays of any item type other than 4 (u32), 6 (f32), 8 (string) decode to `Other` (e.g. an `i32` array such as `tokenizer.ggml.token_type` is `Other`); GGUF v1 is accepted and parsed with v2/v3 widths (u64 counts); `general.alignment` stored as anything but `U32`/`U64` (e.g. `I32`) is ignored and the default 32 is used; release-profile integer arithmetic wraps (`dims` product, `numel * type_size`, `data_start + offset`, `start + byte_len`, `8 + header_len` are all wrapping `size_t` arithmetic, matching Rust's `[profile.release]` having no overflow checks); duplicate metadata keys, duplicate tensor names, and duplicate safetensors JSON keys all resolve **last-one-wins** (`insert_or_assign`, never `emplace`/`insert`); and the malformed-input panics (`general.alignment == 0` → "attempt to divide by zero", a wrapped-but-in-file-size tensor/safetensors range → the Rust slice-index panic text) are reproduced verbatim.
- Exempt texts (not parity-bound): the message tail *after* the SAPIENT-authored prefix when Rust embeds a `serde_json` error (C++ embeds nlohmann's own text there instead — only the prefix is byte-identical), and OS error descriptions on Windows (`FormatMessage` wording vs Rust's).
- **Wrapped `numel * type_size` never aborts.** GGUF F32/F16/BF16 decode the wrapped byte count, so an absurd dim produces Rust's `GGUF parse error: Shape mismatch: expected [...], got [...]` `Err` on every route — never an abort (the plan's first draft aborted here instead; fixed in e1ca4e4).
- **mmap route, kept types (Q4_0/Q8_0/Q4_K/Q5_K/Q6_K):** Rust never slices at load, so a tensor whose byte range wraps loads `Ok` and panics only on first byte access; `MmapBuffer::bytes()` bounds-checks exactly like Rust's slice, reproducing the deferred panic. The heap route and non-kept types on the mmap route panic at load time, as Rust does.
- **Safetensors JSON accept/reject matches serde_json 1.0.150:** a 3-element array is accepted as `StMeta` (serde's sequence form); a UTF-8-BOM-prefixed header is rejected; nesting deeper than 127 levels (including the root object) is rejected, matching serde's recursion limit. Only the message tail after the SAPIENT-authored prefix differs (exempt, per above).

### Results

| Date | Gate | Host / ISA path | Rust commit | C++ commit | Result |
|---|---|---|---|---|---|
| 2026-09-21 | 22 sapient-core unit tests ported by name (+ f16 exhaustive round trip, dequant unit tests) | macOS arm64 (Apple M5) | — | adbf167 | pass |
| 2026-09-21 | `dequant_{q4_0,q8_0,q4_k,q5_k,q6_k,q4_k_r4,q6_k_r4}` (4×512) + `dequant_{f16,bf16}` (64) vs Rust `Tensor::to_f32_vec` / `half` narrowing | macOS arm64 | 8fdf25f | adbf167 | bit-identical, 9/9 |
| 2026-09-21 | 26 dense + 5 float-matmul backends-cpu unit tests ported by name (+ C++-only `parallel`/`sgemm`/`cpu_features`/error-text tests) | macOS arm64 (Apple M5; NEON) | — | `0d46ef9` | pass |
| 2026-09-21 | Dense golden cases `rms_norm`, `layer_norm`, `softmax`, `softmax_axis0`, `log_softmax`, `silu`, `gelu_erf`, `gelu`, `reduce` (4 outputs), `apply_rope`, `apply_rope_partial`, `apply_rope_partial_scaled`, `attention_prefill`, `attention_decode`, `attention_masked`, `matmul_nt_f32_m1_k512`, `matmul_nt_f16_m1` vs the Rust kernels | macOS arm64 (Apple M5; NEON) | `faddb9f` | `0d46ef9` | bit-identical, 17/17 |
| 2026-09-21 | sgemm-backed cases vs `matrixmultiply`: `matmul_nt_f32_m1`, `matmul_nt_f32_m4`, `conv2d_s1`, `conv2d_s2` | macOS arm64 | `faddb9f` | `0d46ef9` | within 1e-5·max(1, max\|ref\|); measured max_err 0 for all four (identical results to matrixmultiply at these sizes) |
| 2026-09-21 | 24 quant + 5 quant-matmul backends-cpu unit tests ported by name (+ C++-only pins of the `as`-cast/`roundf`/`fmaxf` rules, the seven guard texts, repack death tests) | macOS arm64 (Apple M5; NEON/dotprod/i8mm) | — | `b93f5fb` | pass |
| 2026-09-21 | Quantized golden cases: `quantize_q{4,8}_0_block`, `dot_q{4,8}_0_row_f32`, `dot_q{4,5,6}_k_row_f32`, `quantize_row_to_i8_blocks`, `i8_block_sums`, `quantize_row_to_q8k`, `repack_q{4,6}_k_rows4`, `matmul_nt_{q8_0_m{1,3,8},q4_0_m{1,3},q5_k_m1,q4_k_m{1,3},q6_k_m{1,3},q4_k_r4_m{1,2,3,8},q6_k_r4_m{1,2,3,8}}` (default `SAPIENT_Q8K_ACT`) and the twelve `_q8k_off` twins (`SAPIENT_Q8K_ACT=0`) vs the Rust kernels | macOS arm64 (Apple M5; NEON/dotprod/i8mm) | `b93f5fb` | `b93f5fb` | bit-identical, 42/42 (30 default-env + 12 knob-off) |
| 2026-09-21 | Odd-length dense cases `matmul_nt_f32_m1_k519`, `matmul_nt_f16_m1_k67`, `attention_decode_hd10` (SIMD body + scalar tail) | macOS arm64 | `b93f5fb` | `b93f5fb` | bit-identical, 3/3 |
| 2026-09-22 | sp1a plan E — spinpool/thermal unit tests: 6 `Thermal.*` + 5 `Spinpool.*` Rust tests ported by name (incl. `rapid_ops_with_constant_parking`, the SIGSEGV drain-order reproducer), 1 Rust `#[ignore]` probe ported as `Spinpool.DISABLED_pool_speedup_probe`; 2 new C++-only route probes (`Spinpool.route_is_{on,off}_under_env`) exist solely so the pool-on/pool-off ctest entries are non-vacuous | macOS arm64 (Apple M5) | 39742bf | c01f5a5 | pass — `cargo test -p sapient-backends-cpu`: 71 passed; 0 failed; 1 ignored; C++ registers the same 11 ported names + the 2 route probes, all pass (probe DISABLED, not run) |
| 2026-09-22 | sp1a plan E — golden suite, pool ON: `GoldenKernels.*` + `GoldenQuant.*` (54 cases — 24 plan-C dense + 30 plan-D quant, excluding the twelve `_q8k_off` cases which need their own `SAPIENT_Q8K_ACT=0` process; plan A's 9 core dequant dumps run under a separate ctest entry, not this one) plus `Spinpool.route_is_on_under_env`, under `SAPIENT_SPINPOOL=1;SAPIENT_THERMAL=off`, ctest entry `sapient_backends_cpu_tests.spinpool_on` | macOS arm64 (Apple M5; NEON/dotprod/i8mm) | 39742bf | c01f5a5 | bit-identical to the Rust dumps — pass, 55/55 `[ OK ]` (`ctest --preset dev -R 'spinpool_on$' -V \| grep -c '\[ *OK *\]'` with `SAPIENT_GOLDEN_DIR` set: 54 golden comparisons + the route probe). No new dump cases and no Rust changes: the gate re-runs plan D's existing dumps with the CPU GEMVs routed through the spin pool instead of `parallel::par_chunks_mut`. Anti-vacuous guard independently verified: removing `SAPIENT_SPINPOOL=1` from the entry's `ENVIRONMENT` makes it fail on the `FAIL_REGULAR_EXPRESSION "spinpool route probe"` text, which appears in the binary's output exactly twice (only in the two route probes' skip messages) |
| 2026-09-22 | sp1a plan E — golden suite, pool OFF: same 54 cases + `Spinpool.route_is_off_under_env`, under `SAPIENT_SPINPOOL=0`, ctest entry `sapient_backends_cpu_tests.spinpool_off` | macOS arm64 (Apple M5; NEON/dotprod/i8mm) | 39742bf | c01f5a5 | bit-identical to the Rust dumps — pass, 55/55 `[ OK ]` (same count and method as pool ON). Pool-on == pool-off == Rust: the spin pool and `parallel::par_chunks_mut` produce the identical `(chunk index → [start, end))` partition, so switching the route changes nothing observable |
| 2026-09-22 | sp1a plan E — stress: `Spinpool.rapid_ops_with_constant_parking` (the seqlock drain-order SIGSEGV reproducer) × 20 consecutive runs | macOS arm64 (Apple M5) | 39742bf | c01f5a5 | clean — 20/20 consecutive runs, no crash |
| 2026-09-22 | sp1a plan E — independent review: atomic-ordering audit (all 18 atomic operations in `spinpool.cpp` tabulated against `spinpool.rs`) + ThreadSanitizer probe build (4 runs: 400 rounds of the park/wake stress shape, plus 4 concurrent publishers × 40 ops × 37 chunks) | macOS arm64 (Apple M5) | 39742bf | c01f5a5 | 18/18 orderings match; TSan: 0 reports across 4 runs |
| 2026-09-21 | x86_64 (scalar K-quants, AVX2 Q8_0) and Windows | CI `cpp-parity` (ubuntu) / `cpp-build-windows` | — | — | not yet run — first push of the stacked branches; plan E's pool route is additionally untested there (see gaps above) |
| 2026-09-23 | sp1a plan B — 2 sapient-io Rust tests ported by name (`Gguf.q8_0_quantize_roundtrips_and_sizes`, `Gguf.q8_0_quantize_handles_all_zeros`) + the synthetic GGUF/safetensors/mmap/rust_std suites (70 tests total in `sapient_io_tests`) | macOS arm64 (Apple M5) | 52bc163 | 4e90a1e | pass — `cargo test -p sapient-io`: 2 passed; C++ suite `sapient_io_tests`: 70 registered / 68 run, 100% (2 `RealFile.*` tests skipped without `SAPIENT_TEST_GGUF`/`SAPIENT_TEST_SAFETENSORS` set — see the two `RealFile` rows below); workspace-wide `ctest --preset dev`: 327 registered / 326 run (1 `Spinpool.DISABLED_pool_speedup_probe` not run), 100% passing |
| 2026-09-23 | `quantize_to_q8_0` vs plan D's golden-gated `quantize_q8_0_block` (random blocks + inf/NaN inputs) | macOS arm64 | — | e1ca4e4 | bit-identical — `GgufQ8.matches_backends_cpu_quantize_q8_0_block` + `GgufQ8.non_finite_inputs_match_rust_casts` |
| 2026-09-23 | Error-text parity: every Result-path literal in the io tests generated from the real Rust crate (probe crate, 2026-09-23) | macOS arm64 | 52bc163 | 4e90a1e | byte-identical |
| 2026-09-23 | `RealFile.gguf_heap_and_mmap_agree` on SmolLM2-135M-Instruct Q4_K_M and Qwen2.5-1.5B-Instruct Q4_K_M (heap vs mmap vs metadata-only) | macOS arm64 | — | e1ca4e4 | pass — `[real-file] …/models--unsloth--SmolLM2-135M-Instruct-GGUF/…/SmolLM2-135M-Instruct-Q4_K_M.gguf: 272 tensors, 45 zero-copy, 166 requantised Q5_0->Q8_0, 33 KVs` and `[real-file] …/models--Qwen--Qwen2.5-1.5B-Instruct-GGUF/…/qwen2.5-1.5b-instruct-q4_k_m.gguf: 339 tensors, 198 zero-copy, 0 requantised Q5_0->Q8_0, 26 KVs` (the SmolLM2 file's 166 Q5_0→Q8_0 requantisations are most likely llama.cpp's Q4_K→Q5_0 fallback for tensors whose row width, 576, isn't a multiple of 256 — likely, not verified) |
| 2026-09-23 | `RealFile.safetensors_loads` on kokoro-82m `model.safetensors` (+ SmolVLM-256M `model.safetensors`, run separately with `SAPIENT_TEST_SAFETENSORS` re-pointed) | macOS arm64 | — | 4e90a1e | pass — `[real-file] …/models--sai1974dev--kokoro-82m-safetensors/snapshots/3048ee36953f8fadf68a2016ef4fb3f8fc945650/model.safetensors: 548 tensors` and `[real-file] …/models--HuggingFaceTB--SmolVLM-256M-Instruct/…/model.safetensors: 471 tensors` |
| — | `RealFile.gguf_heap_and_mmap_agree` in CI `cpp-parity` (macOS + Linux, after the model download) | CI | — | — | not yet run — first push after plan B |

## Sub-project 1b — CPU chat vertical slice

_(no rows yet)_
