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

## Sub-project 1b — CPU chat vertical slice

_(no rows yet)_
