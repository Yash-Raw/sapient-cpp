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
| 2026-09-21 | x86_64 (scalar K-quants, AVX2 Q8_0) and Windows | CI `cpp-parity` (ubuntu) / `cpp-build-windows` | — | — | not yet run — first push of the stacked branches |

## Sub-project 1b — CPU chat vertical slice

_(no rows yet)_
