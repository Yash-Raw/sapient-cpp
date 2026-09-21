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
  (`dequant_q4_k_r4`/`dequant_q6_k_r4`) and, as of plan C, `apply_rope_partial(_scaled)`, the
  masked-attention branch, the two float GEMV paths (`matmul_nt_f16_m1`, `matmul_nt_f32_m1_k512`),
  `layer_norm`, `reduce`, `gelu`, `softmax` axis 0, `log_softmax` and `conv2d` (43 cases) — the R4
  **matmul** cases (SDOT/SMMLA/Q8_K activation formats) and the `quantize_row_to_i8_blocks` /
  `quantize_row_to_q8k` activation formats remain open for plan D.
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

### Results

| Date | Gate | Host / ISA path | Rust commit | C++ commit | Result |
|---|---|---|---|---|---|
| 2026-09-21 | 22 sapient-core unit tests ported by name (+ f16 exhaustive round trip, dequant unit tests) | macOS arm64 (Apple M5) | — | adbf167 | pass |
| 2026-09-21 | `dequant_{q4_0,q8_0,q4_k,q5_k,q6_k,q4_k_r4,q6_k_r4}` (4×512) + `dequant_{f16,bf16}` (64) vs Rust `Tensor::to_f32_vec` / `half` narrowing | macOS arm64 | 8fdf25f | adbf167 | bit-identical, 9/9 |
| 2026-09-21 | 26 dense + 5 float-matmul backends-cpu unit tests ported by name (+ C++-only `parallel`/`sgemm`/`cpu_features`/error-text tests) | macOS arm64 (Apple M5; NEON) | — | `0d46ef9` | pass |
| 2026-09-21 | Dense golden cases `rms_norm`, `layer_norm`, `softmax`, `softmax_axis0`, `log_softmax`, `silu`, `gelu_erf`, `gelu`, `reduce` (4 outputs), `apply_rope`, `apply_rope_partial`, `apply_rope_partial_scaled`, `attention_prefill`, `attention_decode`, `attention_masked`, `matmul_nt_f32_m1_k512`, `matmul_nt_f16_m1` vs the Rust kernels | macOS arm64 (Apple M5; NEON) | `faddb9f` | `0d46ef9` | bit-identical, 17/17 |
| 2026-09-21 | sgemm-backed cases vs `matrixmultiply`: `matmul_nt_f32_m1`, `matmul_nt_f32_m4`, `conv2d_s1`, `conv2d_s2` | macOS arm64 | `faddb9f` | `0d46ef9` | within 1e-5·max(1, max\|ref\|); measured max_err 0 for all four (identical results to matrixmultiply at these sizes) |

## Sub-project 1b — CPU chat vertical slice

_(no rows yet)_
