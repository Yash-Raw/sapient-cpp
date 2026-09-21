# Sub-project 1a, Plan D: `sapient::backends_cpu` quantized kernels — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port `crates/sapient-backends/cpu/src/kernels/quant.rs` (3787 lines: Q4_0/Q8_0/Q4_K/Q5_K/Q6_K dot products, the int8 activation quantisers incl. the Q8_K format, the NEON/SDOT/SMMLA aarch64 kernels, the one x86 AVX2 dot, the R4 repacks) to `kernels/quant.hpp/.cpp`, replace the seven quantized arms of `matmul_nt` that plan C stubbed, reproduce all 24 + 5 Rust unit tests by name (incl. the two `to_bits` gates), and prove every quantized path bit-identical to the Rust oracle through new golden cases (activation quantisers, repacks, `matmul_nt` over every quantized dtype incl. R4 at m ∈ {1,2,3,8}, Q8_0 at m=8, both `SAPIENT_Q8K_ACT` settings) plus the plan-C carry-over odd-length dense cases.

**Architecture:** One `.hpp/.cpp` pair (`kernels/quant`) mirroring the Rust module: standalone (no `Tensor`), spans in / floats out, ISA variants mirrored never upgraded — scalar everywhere; on aarch64 plain NEON (compile-time baseline), NEON+dotprod (`vdotq_s32` replaces the `sdot` inline asm) and NEON+i8mm (`vmmlaq_s32` replaces `smmla`), each dotprod/i8mm kernel carrying a Clang `target` attribute so the TU builds on an ARMv8.0 baseline; on x86_64 exactly one AVX2+FMA kernel. `matmul.cpp` gains the seven quantized arms with Rust's runtime dispatch (`has_dotprod()`/`has_i8mm()`, the `q8k_activations()` knob, `m ≥ 8` blocked Q8_0 GEMM, `m ≥ 2` SMMLA prefill, portable fallbacks). Every function keeps Rust's accumulation order, reduction intrinsics and rounding.

**Tech Stack:** C++20, CMake presets from sub-project 0, GoogleTest, `sapient::core` (plan A: `f16.hpp`, `dequant.hpp`'s `get_scale_min_k4`/`q4_0_block`, `panic.hpp`, `Tensor`), `sapient::backends_cpu` (plan C: `cpu_features`, `parallel`, `matmul` dispatcher, `env.hpp`), `sapient::testing` (plan A/C: `bit_identical`, `exact_equal<T>`, `SAPIENT_GOLDEN_CASE`), the Rust `dump_kernels` example (test-only Rust, allowed), NEON intrinsics (aarch64), AVX2+FMA intrinsics (x86_64, runtime-gated).

**Spec:** `docs/superpowers/specs/2026-09-21-cpp-sp1a-core-io-cpu-design.md` (§2.3 `kernels/quant` + `kernels/matmul` rows, §3, §4, §5 row D) under `docs/superpowers/specs/2026-09-20-cpp-rewrite-design.md`. **Porting map (line-cited; read §0, §1, §2, §4, §5 before touching a kernel):** `docs/superpowers/notes/2026-09-21-sp1a-porting-map-cpu-kernels.md`. **Plan C** (the conventions, the `parallel`/`cpu_features`/`matmul` API this plan consumes, the stubs it replaces): `docs/superpowers/plans/2026-09-21-cpp-sp1a-plan-c-dense-kernels.md`.

## Global Constraints

- **Branch:** `feat/cpp-sp1a` (stacked on `feat/cpp-sp0-scaffold`; plan C is complete at a4f9b1e). Worktree `.claude/worktrees/feat-cpp-sp0-scaffold`. Commit after every task, with a blank line before the `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` trailer (copy it byte-for-byte). **Never push.**
- **Compiler/flags (programme spec D1 + spec §3.1):** Clang only; `-ffp-contract=off -fno-math-errno` are applied by `cpp/libs/CMakeLists.txt`; never add `-march=native`/`-ffast-math`. Build and test with `cd cpp && cmake --preset dev && cmake --build --preset dev && ctest --preset dev`.
- **Warnings are errors** (`-Wall -Wextra -Wpedantic -Wshadow -Werror`): no shadowing (`d`, `is`, `acc`, lambda parameters), no unused variables in `#if` branches, explicit `static_cast` for every narrowing.
- **CI clang-tidy gate** (`.clang-tidy`, `WarningsAsErrors: '*'`; CI-only on macOS; `just cpp-tidy`): cast to `size_t` **before** multiplying (`bugprone-implicit-widening-of-multiplication-result`), take `std::function` by `const&`, explicit `static_cast<float>(int)`; every function-like macro inside `NOLINTBEGIN/NOLINTEND(bugprone-macro-parentheses)`; no dead `using` declarations (`misc-unused-using-decls` — plan C's one Critical); sign-extend bytes through the ONE helper `i8v()` (Task 1) which carries `// NOLINT(bugprone-signed-char-misuse)`.
- **Portability of the test binaries:** include every std header you use (`<algorithm>`, `<array>`, `<bit>`, `<cmath>`, `<cstdint>`, `<cstdio>`, `<cstring>`, `<optional>`, `<span>`, `<stdexcept>`, `<string>`, `<utility>`, `<vector>`); **no gtest assertion inside a lambda handed to `par_for`/`par_chunks_mut`/`for_each_out_chunk`** (Windows runs ctest); `std::bit_cast<uint32_t>(f)` is the `to_bits()` twin.
- **Formatting:** run the CI-pinned clang-format 18 (`.superpowers/tools-venv/bin/clang-format -i`, or `just cpp-fmt`) on every new/changed `.hpp/.cpp` **after `git add`**, then re-add.
- **Naming (spec D3):** `include/sapient/backends_cpu/kernels/quant.hpp`, `src/kernels/quant.cpp`, `tests/quant_test.cpp`, `tests/golden_quant_test.cpp`; namespace `sapient::backends_cpu::kernels::quant`; Rust's private-but-tested helpers live in `quant::detail`; gtest suite `Quant` for the 24 quant tests, `Matmul` for the 5 matmul tests, `GoldenQuant` for the plan-D golden gate; test names are the Rust test names verbatim.
- **SPDX header verbatim** on every new `.hpp/.cpp` (`//` form) and on `CMakeLists.txt`/`.sh` edits (`#` form):
  `// SPDX-License-Identifier: AGPL-3.0-only`
  `// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)`
- **Bit-identity rules (spec §3, applied to this plan):**
  1. Every float literal `f`-suffixed; libm by its float C name (`::fabsf`, `::roundf`, `::fmaxf`). `f16` ↔ `f32` via `sapient::core::f16_le_to_f32(p)` (Rust `f16::from_le_bytes([p[0],p[1]]).to_f32()`) and `sapient::core::f32_to_f16_bits`/`f16_to_le` (Rust `f16::from_f32(v).to_le_bytes()`). Never `vcvt_f32_f16`.
  2. **Same accumulation order, same reduction intrinsics** (porting map §4): `vaddvq_f32`/`vaddvq_s32`, never a lane loop; the AVX2 hsum is the exact `_mm_movehdup_ps`/`_mm_movehl_ps` sequence; explicit FMA (`vfmaq_f32`, `_mm256_fmadd_ps`) only where Rust used the intrinsic, plain `a * b + c` elsewhere; the f32 combine statements keep Rust's operand order (`acc += xs_lo * (d1 * dot_lo - m1v * sum_lo)`, `acc += x_scales[b] * (d * isum - dmin * imin)`, `acc += ((d * sc) * xs) * dot`). Where Rust's `for` runs an unrolled macro, an equivalent counted loop is fine — the sequence of operations is what matters.
  3. **Rounding and casts (spec §3.4):** `(v * inv).round().clamp(-127.0, 127.0) as i8` → `detail::round_clamp_i8` (`::roundf`, `std::clamp`, then Rust's `as i8`: NaN → 0); `(scaled + 8.5) as i32` → `detail::f32_to_i32_sat` (Rust `as i32`: saturating, NaN → 0) then clamp 0..15; `fold(0.0, f32::max)` → `::fmaxf` (NaN dropped). C++ `static_cast` from an out-of-range or NaN float is UB — use the two helpers at every site.
  4. **ISA mirror (spec §3.2):** aarch64 NEON is compile-time (`#if defined(__aarch64__) || defined(_M_ARM64)`, no attribute); every dotprod kernel is `SAPIENT_TARGET_DOTPROD` (`__attribute__((target("dotprod")))`), every i8mm kernel `SAPIENT_TARGET_I8MM` (`__attribute__((target("i8mm")))`); **i8mm does not imply dotprod in Clang's feature model** — the SMMLA kernels never call the `sdot_s32` helper (they don't in Rust either); shared plain-NEON helpers (`vtrn1q_s64_s8`, `vtrn2q_s64_s8`, the Q6_K unpack) carry no attribute. **No lambda may touch a target-guarded intrinsic** (a lambda's `operator()` does not inherit the enclosing function's attribute) — write attributed `static inline` helpers instead. x86_64: `detail::dot_q8_0_row_avx2` is the only SIMD function, `__attribute__((target("avx2,fma")))`, reached only through `dot_q8_0_row_f32`'s `has_avx2_fma()` gate. The dotprod/i8mm public kernels keep Rust's `unsafe` precondition ("caller verified the feature") as a documented precondition; they are reached only through `cpu_features`-gated dispatch and feature-skipped tests.
  5. **Attribute presence is verified by codegen, not syntax** (probe 2026-09-21: `-fsyntax-only` accepts a missing attribute; only `-c` reports `always_inline function 'vdotq_s32' requires target feature 'dotprod'`, and this Mac's default baseline already has both features so the normal build is silent). Every aarch64 task runs, from `cpp/`:
     `clang++ -std=c++20 -mcpu=cortex-a53 -Wall -Wextra -Wpedantic -Wshadow -Werror -ffp-contract=off -fno-math-errno -Ilibs/sapient-core/include -Ilibs/sapient-backends-cpu/include -c libs/sapient-backends-cpu/src/kernels/quant.cpp -o /dev/null`
     (quant.cpp depends only on `f16.hpp`/`dequant.hpp`/`panic.hpp` — no `tl::expected`, so two `-I`s suffice), and the x86 twin
     `clang++ -std=c++20 --target=x86_64-apple-macos -Wall -Wextra -Wpedantic -Wshadow -Werror -ffp-contract=off -fno-math-errno -Ilibs/sapient-core/include -Ilibs/sapient-backends-cpu/include -c libs/sapient-backends-cpu/src/kernels/quant.cpp -o /dev/null`.
  6. **Rust index/arithmetic panics become explicit checks + `sapient::core::panic()`** (texts not parity-bound), never UB: each kernel checks at ENTRY that the activation-side spans cover the row it is about to read (`x.size() ≥ nb·256`, `x_scales.size() ≥ …`, `x_sums.size() ≥ …`) and that a block span has the block's exact size. `chunks_exact` semantics (a trailing partial block is silently ignored) are kept: `nb = row.size() / BLOCK_BYTES`. **`.take(nb)` semantics** (the Q8_K 4-row/R4/SMMLA kernels iterate `x_scales.iter().enumerate().take(nb)`): a short `x_scales` silently *truncates* the block loop there — port as `nb_eff = min(nb, x_scales.size())` (and `min` over both rows' scales for the x2 kernels); the per-32 kernels index `x_scales[b]` directly and panic. Rust `debug_assert!`s (block length, `k % 32`) are real checks here.
  7. **Result-path messages are byte-identical** (all `Error::internal`, printed as `Internal error: …`): `Q4_0 matmul_nt: k must be a multiple of the block size (32)`, `Q8_0 matmul_nt: k must be a multiple of the block size (32)`, `Q4_K_R4: k must be a multiple of 256 and rows a multiple of 4`, `Q4_K: k must be a multiple of 256`, `Q5_K: k must be a multiple of 256`, `Q6_K_R4: k must be a multiple of 256 and rows a multiple of 4`, `Q6_K: k must be a multiple of 256`.
  8. **Single dequantiser (spec §2.1, approved deviation):** `get_scale_min_k4` is `sapient::core::dequant::get_scale_min_k4` (integer, identical); `quant::dequantize_q4_0_block` is a thin wrapper over `core::dequant::q4_0_block` (verified 2026-09-21 to evaluate `static_cast<float>(lo) * d` in Rust's order).
- **Ruling — second known oracle defect, ported with a stated deviation:** Rust's `dot_q8_0_row_avx2` (`quant.rs:543-580`) loads `xv_b = _mm256_loadu_ps(xp + g*8 + 4)` — 8 floats — at every `g`; at `g == 3` that is `x[28..36)`, four floats past the 32-element block and, on the row's LAST block, four floats past the end of the activation slice (an out-of-bounds read). Those four lanes only ever multiply the zero upper lanes of `_mm256_cvtepi8_epi32(_mm_loadu_si32(…))`, so for finite activations they contribute exactly `+0.0` and the result is unaffected. The port loads the second half **in-bounds and zero-extended** (`_mm256_insertf128_ps(_mm256_setzero_ps(), _mm_loadu_ps(xp + g*8 + 4), 0)`), which is bit-identical for all finite inputs (`0 · x = ±0` never changes a `+0.0` accumulator lane); the one divergence is non-finite `x` (Rust can turn an `inf` activation into `NaN` through a `0 · inf` zero lane; the port yields `±inf`). Recorded in `docs/PARITY.md`'s known-defects table (Task 8). Costs if wrong: none observable on finite inputs; reproducing the read would be UB.
- **Ruling — `SAPIENT_Q8K_ACT` both settings (spec §4):** the knob is read once per process on both sides (Rust `OnceLock`, C++ `static const`), so the OFF setting needs a second process. Rust: `dump_kernels --q8k-off` computes every case as usual (RNG draws stay aligned), renames the twelve knob-sensitive cases (`matmul_nt_q4_k_m{1,3}`, `matmul_nt_q6_k_m{1,3}`, `matmul_nt_q4_k_r4_m{1,2,3,8}`, `matmul_nt_q6_k_r4_m{1,2,3,8}`) with a `_q8k_off` suffix and `retain`s only those; `golden_dump.sh` runs the tool twice, the second under `SAPIENT_Q8K_ACT=0 … --q8k-off`. C++: the twelve `GoldenQuant.*_q8k_off` gtests `GTEST_SKIP()` while `matmul::detail::q8k_activations()` is true (aarch64 only — Rust's knob is `#[cfg(aarch64)]`; on x86 the results are knob-independent and the tests simply run), and one extra ctest entry `sapient_backends_cpu_tests.q8k_off` runs the binary with `--gtest_filter=GoldenQuant.*_q8k_off` under `ENVIRONMENT "SAPIENT_Q8K_ACT=0"` (precedent: `cpp/libs/sapient-core/CMakeLists.txt:25-28`). Costs if wrong: one CMake entry and one shell line.
- **Ruling — plan-C carry-overs:** M1-a (odd-length dense cases) lands in Task 7 (`matmul_nt_f32_m1_k519`, `matmul_nt_f16_m1_k67`, `attention_decode_hd10`, appended at the end of `build_cases`); the `Matmul.quantized_weight_dtypes_are_stubbed_until_plan_d` test is deleted in Task 6; `x_sums` is a parameter of every W4A8 kernel (never re-reduced in a loop); the grouped `switch` in `matmul_nt` becomes seven `case` arms. **M2 (`-0.0f` seed pin): closed by inspection, no observable pin exists** — the only float `Iterator::sum` sites in the C++ tree are the x86 no-AVX2 scalar `dot_f32_fast` and the non-aarch64 `dot_f32_x_f16`, unreachable on every CI host (macos-14 arm64, ubuntu/windows x86_64 with AVX2); this plan's own float sums are all explicit `let mut acc = 0.0f32` (seed `0.0f`) and its `Iterator::sum`s are integer. Do not manufacture a test that cannot run.
- **Ruling — R4 golden weights:** the R4 `matmul_nt` cases feed C++ the Rust-repacked bytes (`in:w_blocks` is the packed stream) so the matmul gate is independent of the repack gate; the repacks themselves are gated by `repack_q4_k_rows4`/`repack_q6_k_rows4` (u8 exact).
- **Ruling — output construction:** as plan C, `Tensor::from_f32_vec(std::move(out), Shape{m, n})` where Rust builds a `Vec<f32>`.
- **`parallel` is a partition, not a scheduler:** the Rust `par_chunks_mut(k).zip(…)` over activation rows (Q8_0 m ≥ 8) is a `parallel::par_for(m, …)` writing disjoint row ranges; `out_t.par_chunks_mut(wchunk * m)` / `par_chunks_mut(4 * m)` / `out.par_chunks_mut(n)` are `parallel::par_chunks_mut(out_t, …, f)` with the identical chunk → range map; every GEMV goes through `detail::for_each_out_chunk` (plan E's insertion point untouched).
- **Rust tree frozen** except `crates/sapient-backends/cpu/examples/dump_kernels.rs` (Task 7: new cases appended **at the end of `build_cases`** so earlier cases' RNG draws are unchanged; the `--q8k-off` mode only renames/filters). `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets -- -D warnings` must stay clean; `cargo test -p sapient-backends-cpu` stays `71 passed; 1 ignored`.
- **Test skips (spec §4):** Rust tests that early-`return` without `dotprod`/`i8mm` become `GTEST_SKIP() << "dotprod not available"` / `"i8mm not available"`; tests under `#[cfg(target_arch = "aarch64")]` are inside `#if defined(__aarch64__) || defined(_M_ARM64)`. Three Rust tests call dotprod kernels with **no** runtime check (`q6_k_r4_kernel_matches_single_row` uses plain NEON — fine; `q4_k_r4_q8k_kernels_match_single_row` and `q4_k_plain_4rows_q8k_matches_single_row` would SIGILL on a non-dotprod aarch64): the port adds the `has_dotprod()` skip there and says so in a comment.
- **No exceptions across library boundaries**; recoverable failures are `Result`, unrecoverable ones `panic()`. Death tests pin the two repack asserts and the `k`-multiple guard of a block quantiser.
- **Docs rule:** Task 8 updates CLAUDE.md, docs/ROADMAP.md, docs/PARITY.md, docs/PROJECT_GUIDE.md, CHANGELOG.md and the spec's §4 "as built" note. CONTRIBUTING/README need no change for a library-internal plan; say so in the commit. `cpp/third_party/LICENSES.md` unchanged (no third-party code added).

## File structure

```
cpp/libs/sapient-backends-cpu/include/sapient/backends_cpu/kernels/quant.hpp   (Tasks 1–5 append their declarations)
cpp/libs/sapient-backends-cpu/src/kernels/quant.cpp                              (Tasks 1–5 append their definitions; ONE TU)
cpp/libs/sapient-backends-cpu/tests/quant_test.cpp                               (Tasks 1–5 append their tests)
cpp/libs/sapient-backends-cpu/include/sapient/backends_cpu/kernels/matmul.hpp   + detail::q8k_activations (Task 6)
cpp/libs/sapient-backends-cpu/src/kernels/matmul.cpp                             + the seven quantized arms (Task 6)
cpp/libs/sapient-backends-cpu/tests/matmul_test.cpp                              + 5 Rust tests, − the plan-D stub test (Task 6)
cpp/libs/sapient-backends-cpu/include/sapient/backends_cpu/kernels.hpp           + quant.hpp include (Task 1)
cpp/libs/sapient-backends-cpu/CMakeLists.txt                                     + quant.cpp, quant_test.cpp (Task 1), golden_quant_test.cpp + q8k_off entry (Task 7)
cpp/libs/sapient-backends-cpu/tests/golden_quant_test.cpp                        (Task 7)
cpp/libs/sapient-backends-cpu/tests/golden_kernels_test.cpp                      + 3 odd-length dense cases (Task 7)
cpp/tests/parity/golden_dump.sh                                                  + the second, SAPIENT_Q8K_ACT=0 pass (Task 7)
crates/sapient-backends/cpu/examples/dump_kernels.rs                             + plan-D cases, --q8k-off (Task 7)
docs: CLAUDE.md, docs/ROADMAP.md, docs/PARITY.md, docs/PROJECT_GUIDE.md, CHANGELOG.md, spec §4 note (Task 8)
```

Layout of `quant.cpp` (keep this order so reviewers can line it up with `quant.rs`): includes and the attribute macros → the two cast helpers and `i8v()` → Q4_0 → Q8_0 → activation quantisers → SDOT Q8_0 → AVX2 Q8_0 → K-quant constants → Q4_K (scalar, NEON f32, W4A8 scalar, `sdot_s32`, `smmla_s32`, NEON W4A8, 4-row, repack, R4, Q8_K variants, SMMLA) → Q5_K → Q6_K (W6A8 scalar/NEON, Q8_K scalar/NEON, repack, R4 W6A8/Q8_K, R4 f32, SMMLA) → Q6_K f32 scalar/NEON. Each task appends its section in that order.

---

### Task 1: `kernels/quant.hpp/.cpp` skeleton — Q4_0, Q8_0, the activation quantisers, SDOT and AVX2 Q8_0

**Files:**
- Create: `cpp/libs/sapient-backends-cpu/include/sapient/backends_cpu/kernels/quant.hpp`, `cpp/libs/sapient-backends-cpu/src/kernels/quant.cpp`, `cpp/libs/sapient-backends-cpu/tests/quant_test.cpp`
- Modify: `cpp/libs/sapient-backends-cpu/CMakeLists.txt` (add `src/kernels/quant.cpp` to the library and `tests/quant_test.cpp` to the test executable), `cpp/libs/sapient-backends-cpu/include/sapient/backends_cpu/kernels.hpp` (add `#include "sapient/backends_cpu/kernels/quant.hpp"` in alphabetical position, before `reduce.hpp`)

**Interfaces:**
- Consumes: `sapient::core::{f16_le_to_f32, f32_to_f16_bits, f16_to_le, panic}`, `sapient::core::dequant::q4_0_block`, `sapient::backends_cpu::cpu_features::has_avx2_fma`.
- Produces (namespace `sapient::backends_cpu::kernels::quant`): constants `QK=32, Q4_0_BLOCK_BYTES=18, Q8_0_BLOCK_BYTES=34, QK_K=256, Q4_K_BLOCK_BYTES=144, Q5_K_BLOCK_BYTES=176, Q6_K_BLOCK_BYTES=210`; `struct I8Blocks { std::vector<int8_t> q; std::vector<float> scales; }` (Rust `(Vec<i8>, Vec<f32>)`), `struct Q8kRow { std::vector<int8_t> q; std::vector<float> scales; std::vector<int32_t> sums; }` (Rust `(Vec<i8>, Vec<f32>, Vec<i32>)`); `quantize_q4_0_block(std::span<const float>) -> std::array<uint8_t,18>`, `dequantize_q4_0_block(std::span<const uint8_t>, std::span<float>)`, `dot_q4_0_block_f32`, `dot_q4_0_row_f32`, `quantize_q4_0_row(std::span<const float>) -> std::vector<uint8_t>`, `quantize_q8_0_block -> std::array<uint8_t,34>`, `dot_q8_0_block_f32`, `quantize_row_to_i8_blocks -> I8Blocks`, `i8_block_sums(std::span<const int8_t>) -> std::vector<int32_t>`, `quantize_row_to_q8k -> Q8kRow`, `dot_q8_0_row_i8_scalar(row_blocks, x_i8, x_scales)`, `dot_q8_0_row_f32(row_blocks, x)`; aarch64: `dot_q8_0_row_sdot(row_blocks, x_i8, x_scales)`; `detail::{f32_to_i32_sat, round_clamp_i8, nibble, i8v, dot_q4_0_block_scalar, dot_q8_0_block_scalar}`, aarch64 `detail::dot_q8_0_block_sdot -> int32_t`, x86_64 `detail::dot_q8_0_row_avx2`. The macros `SAPIENT_TARGET_DOTPROD`/`SAPIENT_TARGET_I8MM` and the aarch64 guard `SAPIENT_AARCH64` are private to `quant.cpp`. Tasks 2–7 consume all of these.

- [ ] **Step 1: Write the failing tests** — `tests/quant_test.cpp` (the four Rust tests of this family by name + four C++-only tests of the rounding/NaN semantics; later tasks append to this file, so keep the helper namespace open-ended):

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
// Port of the `#[cfg(test)] mod tests` of crates/sapient-backends/cpu/src/kernels/quant.rs — the 24
// Rust tests by name (Tasks 1-5 of plan D) plus C++-only pins of the rounding/NaN rules the spec
// (§3.4) makes explicit. aarch64-only Rust tests are compiled only on aarch64; runtime `return`s
// without dotprod/i8mm become GTEST_SKIPs.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "sapient/backends_cpu/cpu_features.hpp"
#include "sapient/backends_cpu/kernels/quant.hpp"
#include "sapient/core/f16.hpp"

using namespace sapient::backends_cpu::kernels::quant;
using sapient::backends_cpu::cpu_features::has_dotprod;
using sapient::backends_cpu::cpu_features::has_i8mm;

namespace {

// Deterministic pseudo-random f32 in roughly [-1, 1] (quant.rs `seq`, xorshift64).
std::vector<float> seq(size_t n) {
    uint64_t s = 0x9E3779B97F4A7C15ULL;
    std::vector<float> v(n);
    for (float& x : v) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        x = (static_cast<float>(s >> 40) / static_cast<float>(1u << 24)) * 2.0f - 1.0f;
    }
    return v;
}

// quant.rs `lcg_bytes` / `rand_x`: the 6364136223846793005 · s + 1442695040888963407 LCG.
uint64_t lcg_step(uint64_t& s) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return s;
}
std::vector<uint8_t> lcg_bytes(uint64_t seed, size_t n) {
    std::vector<uint8_t> v(n);
    for (uint8_t& b : v)
        b = static_cast<uint8_t>(lcg_step(seed) >> 33);
    return v;
}
std::vector<float> rand_x(uint64_t seed, size_t n) {
    std::vector<float> v(n);
    for (float& x : v)
        x = (static_cast<float>(lcg_step(seed) >> 33) /
             static_cast<float>(std::numeric_limits<uint32_t>::max())) *
                3.0f -
            1.5f;
    return v;
}

// quant.rs `q8_0_weight_row`: quantize an f32 row into packed Q8_0 weight blocks.
std::vector<uint8_t> q8_0_weight_row(std::span<const float> w) {
    std::vector<uint8_t> out;
    out.reserve(w.size() / QK * Q8_0_BLOCK_BYTES);
    for (size_t b = 0; b + QK <= w.size(); b += QK) {
        const auto blk = quantize_q8_0_block(w.subspan(b, QK));
        out.insert(out.end(), blk.begin(), blk.end());
    }
    return out;
}

uint32_t bits(float f) {
    return std::bit_cast<uint32_t>(f);
}

} // namespace

// ── Q4_0 / Q8_0 (Task 1) ─────────────────────────────────────────────────────

// The Q8_0 SDOT path quantizes activations to int8. With a per-block scale it must stay close to
// the exact f32 path even when the activation row contains an outlier channel — a per-row scale
// (the old behaviour) diverges wildly here and produced garbage LLM output.
#if defined(__aarch64__) || defined(_M_ARM64)
TEST(Quant, sdot_q8_0_row_blockwise_survives_activation_outlier) {
    if (!has_dotprod()) GTEST_SKIP() << "dotprod not available";
    const size_t k = 256;
    const auto wf = seq(k);
    auto xf = seq(k);
    xf[100] = 60.0f; // outlier channel, ~60× the rest

    const auto w_blocks = q8_0_weight_row(wf);
    const float reference = dot_q8_0_row_f32(w_blocks, xf);

    const auto xq = quantize_row_to_i8_blocks(xf);
    const float blockwise = dot_q8_0_row_sdot(w_blocks, xq.q, xq.scales);
    const float rel = ::fabsf(blockwise - reference) / ::fmaxf(::fabsf(reference), 1e-3f);
    EXPECT_LT(rel, 0.05f) << "blockwise SDOT rel err " << rel << " too high (got " << blockwise
                          << ", ref " << reference << ")";

    // Old path: a single per-row scale set by the outlier collapses the rest.
    float max_abs = 0.0f;
    for (float v : xf)
        max_abs = ::fmaxf(max_abs, ::fabsf(v));
    const float row_scale = max_abs / 127.0f;
    const float inv = 1.0f / row_scale;
    std::vector<int8_t> x_row_i8(k);
    for (size_t i = 0; i < k; ++i)
        x_row_i8[i] = detail::round_clamp_i8(xf[i] * inv);
    const std::vector<float> per_row_scales(k / QK, row_scale);
    const float perrow = dot_q8_0_row_sdot(w_blocks, x_row_i8, per_row_scales);
    const float perrow_rel = ::fabsf(perrow - reference) / ::fmaxf(::fabsf(reference), 1e-3f);
    EXPECT_GT(perrow_rel, rel * 2.0f)
        << "per-row scale should be clearly worse than blockwise (per-row rel " << perrow_rel
        << ", blockwise rel " << rel << ")";
}

TEST(Quant, sdot_q8_0_block_matches_scalar_integer_dot) {
    if (!has_dotprod()) GTEST_SKIP() << "dotprod not available";
    const auto wf = seq(QK);
    const auto xf = seq(QK);
    std::vector<int8_t> w_i8(QK), x_i8(QK);
    for (size_t i = 0; i < QK; ++i) {
        w_i8[i] = detail::round_clamp_i8(wf[i] * 100.0f);
        x_i8[i] = detail::round_clamp_i8(xf[i] * 100.0f);
    }
    std::vector<uint8_t> block(Q8_0_BLOCK_BYTES, 0);
    sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(1.0f), block.data());
    for (size_t i = 0; i < QK; ++i)
        block[2 + i] = static_cast<uint8_t>(w_i8[i]);

    int32_t reference = 0;
    for (size_t i = 0; i < QK; ++i)
        reference += static_cast<int32_t>(w_i8[i]) * static_cast<int32_t>(x_i8[i]);
    EXPECT_EQ(detail::dot_q8_0_block_sdot(block, x_i8), reference)
        << "SDOT integer dot must match scalar reference";
}
#endif

TEST(Quant, q4_0_on_the_fly_dot_matches_dequantized_reference) {
    const size_t k = 256;
    const auto w = seq(k);
    auto x = seq(k);
    for (float& v : x)
        v *= 0.5f;

    const auto blocks = quantize_q4_0_row(w);
    // Storage: 18 bytes per 32 weights = 0.5625 B/weight vs 4 B for F32.
    ASSERT_EQ(blocks.size(), k / QK * Q4_0_BLOCK_BYTES);

    // Reference: dequantize fully, then dot.
    std::vector<float> w_hat(k, 0.0f);
    for (size_t b = 0; b < k / QK; ++b)
        dequantize_q4_0_block(std::span<const uint8_t>(blocks).subspan(b * Q4_0_BLOCK_BYTES,
                                                                       Q4_0_BLOCK_BYTES),
                              std::span<float>(w_hat).subspan(b * QK, QK));
    float reference = -0.0f; // iter().zip().map().sum() seeds at -0.0
    for (size_t i = 0; i < k; ++i)
        reference += w_hat[i] * x[i];

    const float on_the_fly = dot_q4_0_row_f32(blocks, x);
    EXPECT_LT(::fabsf(on_the_fly - reference), 1e-3f)
        << "on-the-fly " << on_the_fly << " vs reference " << reference;
}

TEST(Quant, q4_0_quantization_error_is_bounded) {
    // Dequantized weights should track the originals within Q4 granularity.
    const auto w = seq(QK * 4);
    const auto blocks = quantize_q4_0_row(w);
    std::vector<float> w_hat(w.size(), 0.0f);
    for (size_t b = 0; b < w.size() / QK; ++b)
        dequantize_q4_0_block(std::span<const uint8_t>(blocks).subspan(b * Q4_0_BLOCK_BYTES,
                                                                       Q4_0_BLOCK_BYTES),
                              std::span<float>(w_hat).subspan(b * QK, QK));
    float max_err = 0.0f;
    for (size_t i = 0; i < w.size(); ++i)
        max_err = ::fmaxf(max_err, ::fabsf(w[i] - w_hat[i]));
    EXPECT_LT(max_err, 0.2f) << "max quant error " << max_err << " too large";
}

// ── C++-only pins of spec §3.4 (Rust `as` casts, `round`, `f32::max`) ────────

TEST(Quant, nibble_truncates_toward_zero_and_clamps) {
    // ggml: MIN(15, (int)(x*id + 8.5)) — `as i32` truncates toward zero, then clamp 0..15.
    EXPECT_EQ(detail::nibble(0.6f), 9);       // 9.1 → 9
    EXPECT_EQ(detail::nibble(-0.6f), 7);      // 7.9 → 7 (not 8)
    EXPECT_EQ(detail::nibble(-9.0f), 0);      // -0.5 → 0
    EXPECT_EQ(detail::nibble(-9.4f), 0);      // -0.9 → 0 (truncation toward zero)
    EXPECT_EQ(detail::nibble(8.0f), 15);      // 16.5 → 16 → clamp 15
    EXPECT_EQ(detail::nibble(1e30f), 15);     // saturating `as i32` → clamp 15
    EXPECT_EQ(detail::nibble(-1e30f), 0);
    EXPECT_EQ(detail::nibble(std::numeric_limits<float>::quiet_NaN()), 0); // NaN as i32 == 0
    EXPECT_EQ(detail::f32_to_i32_sat(3e9f), std::numeric_limits<int32_t>::max());
    EXPECT_EQ(detail::f32_to_i32_sat(-3e9f), std::numeric_limits<int32_t>::min());
    EXPECT_EQ(detail::f32_to_i32_sat(-2.9f), -2);
}

TEST(Quant, round_clamp_i8_rounds_half_away_from_zero_and_maps_nan_to_zero) {
    EXPECT_EQ(detail::round_clamp_i8(0.5f), 1);
    EXPECT_EQ(detail::round_clamp_i8(-0.5f), -1);
    EXPECT_EQ(detail::round_clamp_i8(2.5f), 3);   // NOT banker's rounding
    EXPECT_EQ(detail::round_clamp_i8(-2.5f), -3);
    EXPECT_EQ(detail::round_clamp_i8(127.6f), 127);
    EXPECT_EQ(detail::round_clamp_i8(-200.0f), -127); // clamp, not -128
    EXPECT_EQ(detail::round_clamp_i8(std::numeric_limits<float>::infinity()), 127);
    EXPECT_EQ(detail::round_clamp_i8(std::numeric_limits<float>::quiet_NaN()), 0);
    EXPECT_EQ(detail::i8v(0xFF), -1);
    EXPECT_EQ(detail::i8v(0x80), -128);
    EXPECT_EQ(detail::i8v(0x7F), 127);
}

TEST(Quant, activation_quantizers_zero_row_and_nan_semantics) {
    // Zero row: the activation quantisers use scale 1.0 (not 0.0); the weight quantiser uses
    // scale 0 → f16(0) and inv 0 (quant.rs:223-240 vs :317-341, :366-392).
    const std::vector<float> zeros(QK_K, 0.0f);
    const auto i8 = quantize_row_to_i8_blocks(zeros);
    ASSERT_EQ(i8.scales.size(), QK_K / QK);
    for (float s : i8.scales)
        EXPECT_EQ(bits(s), bits(1.0f));
    const auto q8k = quantize_row_to_q8k(zeros);
    ASSERT_EQ(q8k.scales.size(), 1u);
    EXPECT_EQ(bits(q8k.scales[0]), bits(1.0f));
    ASSERT_EQ(q8k.sums.size(), QK_K / QK);
    for (int32_t s : q8k.sums)
        EXPECT_EQ(s, 0);
    const auto blk = quantize_q8_0_block(std::span<const float>(zeros).subspan(0, QK));
    EXPECT_EQ(sapient::core::f16_le_to_f32(blk.data()), 0.0f);
    for (size_t i = 2; i < Q8_0_BLOCK_BYTES; ++i)
        EXPECT_EQ(blk[i], 0);

    // A NaN element: fmaxf drops it from max_abs, and it quantizes to 0 (Rust `NaN as i8`).
    std::vector<float> row(QK, 0.5f);
    row[3] = std::numeric_limits<float>::quiet_NaN();
    const auto r = quantize_row_to_i8_blocks(row);
    EXPECT_EQ(bits(r.scales[0]), bits(0.5f / 127.0f));
    EXPECT_EQ(r.q[3], 0);
    EXPECT_EQ(r.q[0], 127);
    // Q8_K sums are the per-32 sums of the Q8_K quants, i.e. i8_block_sums(q).
    auto x = seq(QK_K);
    x[17] = 20.0f;
    const auto k8 = quantize_row_to_q8k(x);
    const auto sums = i8_block_sums(k8.q);
    ASSERT_EQ(sums.size(), k8.sums.size());
    for (size_t i = 0; i < sums.size(); ++i)
        EXPECT_EQ(sums[i], k8.sums[i]);
}

TEST(Quant, dot_q8_0_row_paths_agree) {
    // f32 path vs the two int8-activation paths: the scalar i8 dot and (under dotprod) SDOT
    // compute the same integer dots and combine in the same order → bit-identical to each other,
    // and both within activation-quantisation error of the f32 path.
    const size_t k = 128;
    const auto wf = seq(k);
    const auto xf = rand_x(0x1234, k);
    const auto w_blocks = q8_0_weight_row(wf);
    const float f32_dot = dot_q8_0_row_f32(w_blocks, xf);
    const auto xq = quantize_row_to_i8_blocks(xf);
    const float i8_dot = dot_q8_0_row_i8_scalar(w_blocks, xq.q, xq.scales);
    EXPECT_LT(::fabsf(f32_dot - i8_dot) / ::fmaxf(::fabsf(f32_dot), 1e-3f), 0.03f)
        << f32_dot << " vs " << i8_dot;
#if defined(__aarch64__) || defined(_M_ARM64)
    if (has_dotprod()) {
        const float sdot = dot_q8_0_row_sdot(w_blocks, xq.q, xq.scales);
        EXPECT_EQ(bits(sdot), bits(i8_dot)) << sdot << " vs " << i8_dot;
    }
#endif
}

TEST(QuantDeath, block_quantizers_panic_on_wrong_length) {
    const std::vector<float> x(31, 0.0f);
    EXPECT_DEATH((void)quantize_q8_0_block(x), "");
    EXPECT_DEATH((void)quantize_q4_0_block(x), "");
    const std::vector<float> x33(33, 0.0f);
    EXPECT_DEATH((void)quantize_row_to_i8_blocks(x33), "");
}
```

- [ ] **Step 2: Build to see it fail**

Add to `CMakeLists.txt`: `src/kernels/quant.cpp` after `src/kernels/matmul.cpp` in `add_library`, and `tests/quant_test.cpp` after `tests/matmul_test.cpp` in `add_executable`. Run `cd cpp && cmake --preset dev && cmake --build --preset dev`. Expected: FAIL — `sapient/backends_cpu/kernels/quant.hpp` not found.

- [ ] **Step 3: Write `quant.hpp` (this task's declarations; later tasks append theirs under the marked sections)**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-backends/cpu/src/kernels/quant.rs: quantized weight blocks (ggml layouts)
// and the on-the-fly dequantizing dot products that keep weights quantized in memory. Standalone
// like the Rust module — no Tensor dependency; byte/float/int8 spans in, f32 out.
//
// ISA variants are mirrored, never upgraded (spec §3.2): scalar everywhere; on aarch64 plain NEON
// (compile-time baseline), NEON+dotprod (`vdotq_s32` = Rust's `sdot` inline asm) and NEON+i8mm
// (`vmmlaq_s32` = `smmla`), each dotprod/i8mm kernel carrying `__attribute__((target(...)))`;
// on x86_64 exactly one AVX2+FMA kernel (`detail::dot_q8_0_row_avx2`), reached only through
// `dot_q8_0_row_f32`'s runtime gate. Rust's `unsafe fn` preconditions ("caller verified dotprod /
// i8mm") are preconditions here too: matmul.cpp gates on cpu_features before calling them.
//
// Every accumulation order, reduction intrinsic and rounding rule follows the Rust source (porting
// map §4); Rust index panics are entry checks that `sapient::core::panic()`.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace sapient::backends_cpu::kernels::quant {

/// Weights per Q4_0/Q8_0 block.
inline constexpr size_t QK = 32;
/// Bytes per Q4_0 block: 2 (f16 scale) + 16 (packed nibbles).
inline constexpr size_t Q4_0_BLOCK_BYTES = 18;
/// Bytes per Q8_0 block: 2 (f16 scale) + 32 (i8 quants).
inline constexpr size_t Q8_0_BLOCK_BYTES = 34;
/// Weights per K-quant super-block.
inline constexpr size_t QK_K = 256;
inline constexpr size_t Q4_K_BLOCK_BYTES = 144;
inline constexpr size_t Q5_K_BLOCK_BYTES = 176;
inline constexpr size_t Q6_K_BLOCK_BYTES = 210;

/// Rust `(Vec<i8>, Vec<f32>)` from `quantize_row_to_i8_blocks`: int8 quants + ONE scale per 32.
struct I8Blocks {
    std::vector<int8_t> q;
    std::vector<float> scales;
};
/// Rust `(Vec<i8>, Vec<f32>, Vec<i32>)` from `quantize_row_to_q8k`: int8 quants, ONE scale per
/// 256-element super-block, per-32 sums (llama.cpp `block_q8_K`).
struct Q8kRow {
    std::vector<int8_t> q;
    std::vector<float> scales;
    std::vector<int32_t> sums;
};

// ── Q4_0 ────────────────────────────────────────────────────────────────────────
/// Quantize 32 f32 into one Q4_0 block (ggml: `d = vmax / -8`, sign preserved; nibbles truncated).
std::array<uint8_t, Q4_0_BLOCK_BYTES> quantize_q4_0_block(std::span<const float> x);
/// One Q4_0 block → 32 values (delegates to the shared core::dequant::q4_0_block).
void dequantize_q4_0_block(std::span<const uint8_t> block, std::span<float> out);
/// One block · 32 activations (NEON on aarch64, scalar elsewhere).
float dot_q4_0_block_f32(std::span<const uint8_t> block, std::span<const float> x);
/// Whole row: f32 `acc += block_dot` sequentially over the 18-byte blocks.
float dot_q4_0_row_f32(std::span<const uint8_t> row_blocks, std::span<const float> x);
/// Quantize a full f32 row (`w.size() % 32 == 0`) into packed Q4_0 blocks.
std::vector<uint8_t> quantize_q4_0_row(std::span<const float> w);

// ── Q8_0 ────────────────────────────────────────────────────────────────────────
/// Quantize 32 f32 into one Q8_0 block (`scale = max_abs/127`, f16 scale, roundf then clamp).
std::array<uint8_t, Q8_0_BLOCK_BYTES> quantize_q8_0_block(std::span<const float> x);
/// One block · 32 activations (NEON on aarch64, scalar elsewhere).
float dot_q8_0_block_f32(std::span<const uint8_t> block, std::span<const float> x);
/// Whole row · f32 activations: AVX2+FMA on x86_64 with `has_avx2_fma()`, else per-block.
float dot_q8_0_row_f32(std::span<const uint8_t> row_blocks, std::span<const float> x);
/// Whole row · int8 activations with per-32 scales — the portable twin of `dot_q8_0_row_sdot`
/// (same integer dots, same f32 combine order).
float dot_q8_0_row_i8_scalar(std::span<const uint8_t> row_blocks,
                             std::span<const int8_t> x_i8,
                             std::span<const float> x_scales);

// ── activation quantisers ───────────────────────────────────────────────────────
/// Per-32-block int8 activations (`x.size() % 32 == 0`); a zero block gets scale 1.0.
I8Blocks quantize_row_to_i8_blocks(std::span<const float> x);
/// Per-32 Σq (i32) of an int8 row — the precomputed `x_sums` every W4A8 kernel takes.
std::vector<int32_t> i8_block_sums(std::span<const int8_t> q);
/// Q8_K-style activations (`x.size() % 256 == 0`): one scale per 256, sums per 32.
Q8kRow quantize_row_to_q8k(std::span<const float> x);

#if defined(__aarch64__) || defined(_M_ARM64)
/// Q8_0 row · int8 activations via `sdot`. Precondition: `cpu_features::has_dotprod()`.
float dot_q8_0_row_sdot(std::span<const uint8_t> row_blocks,
                        std::span<const int8_t> x_i8,
                        std::span<const float> x_scales);
#endif

namespace detail {
/// Rust `f as i32`: truncation toward zero, saturating, NaN → 0.
int32_t f32_to_i32_sat(float v);
/// Rust `(v).round().clamp(-127.0, 127.0) as i8`: half away from zero, NaN → 0.
int8_t round_clamp_i8(float v);
/// ggml nibble: `clamp((scaled + 8.5) as i32, 0, 15)`.
uint8_t nibble(float scaled);
/// Rust `byte as i8 as i32` — the one place bytes are sign-extended.
int32_t i8v(uint8_t b);
float dot_q4_0_block_scalar(std::span<const uint8_t> block, std::span<const float> x);
float dot_q8_0_block_scalar(std::span<const uint8_t> block, std::span<const float> x);
#if defined(__aarch64__) || defined(_M_ARM64)
/// Integer dot of one Q8_0 block with 32 int8 activations (two `sdot`, `vaddvq_s32`).
int32_t dot_q8_0_block_sdot(std::span<const uint8_t> block, std::span<const int8_t> x_i8);
#endif
#if defined(__x86_64__) || defined(_M_X64)
/// The one x86 SIMD kernel. Precondition: `cpu_features::has_avx2_fma()`.
float dot_q8_0_row_avx2(std::span<const uint8_t> row_blocks, std::span<const float> x);
#endif
} // namespace detail

// ── Q4_K (Task 2) ───────────────────────────────────────────────────────────────
// ── Q4_K × Q8_K, SMMLA (Task 3) ─────────────────────────────────────────────────
// ── Q5_K, Q6_K f32, Q6_K repack/R4 f32 (Task 4) ─────────────────────────────────
// ── Q6_K W6A8 / Q8_K / SMMLA (Task 5) ───────────────────────────────────────────

} // namespace sapient::backends_cpu::kernels::quant
```

- [ ] **Step 4: Write `quant.cpp` (this task's section; the file grows in the layout order given under "File structure")**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/backends_cpu/kernels/quant.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#if defined(__aarch64__) || defined(_M_ARM64)
#define SAPIENT_AARCH64 1
#include <arm_neon.h>
// Clang's target-guarded NEON intrinsics: a dotprod/i8mm intrinsic may only be used inside a
// function compiled with that feature. Every such kernel carries one of these (Rust:
// `#[target_feature(enable = "neon,dotprod")]` / `"neon,i8mm"`). i8mm does NOT imply dotprod.
#define SAPIENT_TARGET_DOTPROD __attribute__((target("dotprod")))
#define SAPIENT_TARGET_I8MM __attribute__((target("i8mm")))
#elif defined(__x86_64__) || defined(_M_X64)
#define SAPIENT_X86_64 1
#include <immintrin.h>
#endif

#include "sapient/backends_cpu/cpu_features.hpp"
#include "sapient/core/dequant.hpp"
#include "sapient/core/f16.hpp"
#include "sapient/core/panic.hpp"

namespace sapient::backends_cpu::kernels::quant {

using sapient::core::f16_le_to_f32;
using sapient::core::panic;

namespace detail {

int32_t f32_to_i32_sat(float v) {
    // Rust `as i32`: NaN → 0, ±inf and out-of-range saturate, otherwise truncate toward zero.
    if (std::isnan(v)) return 0;
    if (v >= 2147483648.0f) return std::numeric_limits<int32_t>::max();
    if (v <= -2147483648.0f) return std::numeric_limits<int32_t>::min();
    return static_cast<int32_t>(v);
}

int8_t round_clamp_i8(float v) {
    const float r = std::clamp(::roundf(v), -127.0f, 127.0f); // clamp lets NaN through, like Rust
    if (std::isnan(r)) return 0;                                // Rust `NaN as i8` == 0
    return static_cast<int8_t>(r);
}

uint8_t nibble(float scaled) {
    // ggml: MIN(15, (int)(x*id + 8.5)). Clamp into [0, 15].
    return static_cast<uint8_t>(std::clamp(f32_to_i32_sat(scaled + 8.5f), 0, 15));
}

int32_t i8v(uint8_t b) {
    return static_cast<int32_t>(static_cast<int8_t>(b)); // NOLINT(bugprone-signed-char-misuse)
}

} // namespace detail

namespace {

void check_block(std::span<const uint8_t> block, size_t bytes, const char* who) {
    if (block.size() != bytes) panic(who);
}
void check_len(size_t have, size_t need, const char* who) {
    if (have < need) panic(who);
}

// ── Q4_0 ─────────────────────────────────────────────────────────────────────

#if SAPIENT_AARCH64
// NEON Q4_0 block dot product: 16 packed bytes → lo nibbles (elements 0..15) and hi nibbles
// (16..31), minus 8, widened i8 → i16 → i32 → f32, FMA with the activations (quant.rs:123-175).
float dot_q4_0_block_neon(const uint8_t* block, const float* x) {
    const float scale = f16_le_to_f32(block);
    const uint8x16_t packed = vld1q_u8(block + 2);
    const uint8x16_t lo_u8 = vandq_u8(packed, vdupq_n_u8(0x0F));
    const uint8x16_t hi_u8 = vshrq_n_u8(packed, 4);
    const uint8x16_t eight = vdupq_n_u8(8);
    const int8x16_t lo_i8 = vreinterpretq_s8_u8(vsubq_u8(lo_u8, eight));
    const int8x16_t hi_i8 = vreinterpretq_s8_u8(vsubq_u8(hi_u8, eight));

    // to_f32x4!(v, vmovl_s8_low / vmovl_s8_high)
    const int16x8_t lo16a = vmovl_s8(vget_low_s8(lo_i8));
    const float32x4_t lo_f32_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16a)));
    const float32x4_t lo_f32_1 = vcvtq_f32_s32(vmovl_high_s16(lo16a));
    const int16x8_t lo16b = vmovl_high_s8(lo_i8);
    const float32x4_t lo_f32_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16b)));
    const float32x4_t lo_f32_3 = vcvtq_f32_s32(vmovl_high_s16(lo16b));
    const int16x8_t hi16a = vmovl_s8(vget_low_s8(hi_i8));
    const float32x4_t hi_f32_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16a)));
    const float32x4_t hi_f32_1 = vcvtq_f32_s32(vmovl_high_s16(hi16a));
    const int16x8_t hi16b = vmovl_high_s8(hi_i8);
    const float32x4_t hi_f32_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16b)));
    const float32x4_t hi_f32_3 = vcvtq_f32_s32(vmovl_high_s16(hi16b));

    const float32x4_t x0 = vld1q_f32(x);
    const float32x4_t x1 = vld1q_f32(x + 4);
    const float32x4_t x2 = vld1q_f32(x + 8);
    const float32x4_t x3 = vld1q_f32(x + 12);
    const float32x4_t x4 = vld1q_f32(x + 16);
    const float32x4_t x5 = vld1q_f32(x + 20);
    const float32x4_t x6 = vld1q_f32(x + 24);
    const float32x4_t x7 = vld1q_f32(x + 28);

    float32x4_t acc = vmulq_f32(lo_f32_0, x0);
    acc = vfmaq_f32(acc, lo_f32_1, x1);
    acc = vfmaq_f32(acc, lo_f32_2, x2);
    acc = vfmaq_f32(acc, lo_f32_3, x3);
    acc = vfmaq_f32(acc, hi_f32_0, x4);
    acc = vfmaq_f32(acc, hi_f32_1, x5);
    acc = vfmaq_f32(acc, hi_f32_2, x6);
    acc = vfmaq_f32(acc, hi_f32_3, x7);
    return vaddvq_f32(acc) * scale;
}

// NEON Q8_0 block dot: four groups of 8 i8 widened to f32, two vfmaq per group (quant.rs:266-290).
float dot_q8_0_block_neon(const uint8_t* block, const float* x) {
    const float scale = f16_le_to_f32(block);
    const int8_t* q_ptr = reinterpret_cast<const int8_t*>(block + 2);
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (size_t off = 0; off < QK; off += 8) { // fma_group!(0,0) (8,8) (16,16) (24,24)
        const int8x8_t q8 = vld1_s8(q_ptr + off);
        const int16x8_t q16 = vmovl_s8(q8);
        const float32x4_t qlo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16)));
        const float32x4_t qhi = vcvtq_f32_s32(vmovl_high_s16(q16));
        acc = vfmaq_f32(acc, qlo, vld1q_f32(x + off));
        acc = vfmaq_f32(acc, qhi, vld1q_f32(x + off + 4));
    }
    return vaddvq_f32(acc) * scale;
}
#endif

} // namespace

std::array<uint8_t, Q4_0_BLOCK_BYTES> quantize_q4_0_block(std::span<const float> x) {
    if (x.size() != QK) panic("quantize_q4_0_block: x must have 32 elements");
    // Scale from the value with the largest magnitude, preserving its sign (ggml derives `d`
    // this way, which is why d can be negative).
    float amax = 0.0f;
    float vmax = 0.0f;
    for (const float v : x) {
        if (::fabsf(v) > amax) {
            amax = ::fabsf(v);
            vmax = v;
        }
    }
    const float d = vmax / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;

    std::array<uint8_t, Q4_0_BLOCK_BYTES> out{};
    sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(d), out.data());
    for (size_t j = 0; j < QK / 2; ++j) {
        const uint8_t q0 = detail::nibble(x[j] * id);
        const uint8_t q1 = detail::nibble(x[j + QK / 2] * id);
        out[2 + j] = static_cast<uint8_t>(q0 | (q1 << 4));
    }
    return out;
}

void dequantize_q4_0_block(std::span<const uint8_t> block, std::span<float> out) {
    check_block(block, Q4_0_BLOCK_BYTES, "dequantize_q4_0_block: block must be 18 bytes");
    if (out.size() != QK) panic("dequantize_q4_0_block: out must have 32 elements");
    sapient::core::dequant::q4_0_block(block.data(), out.data()); // same arithmetic, same order
}

float detail::dot_q4_0_block_scalar(std::span<const uint8_t> block, std::span<const float> x) {
    const float d = f16_le_to_f32(block.data());
    float acc = 0.0f;
    for (size_t j = 0; j < QK / 2; ++j) {
        const uint8_t byte = block[2 + j];
        const int32_t lo = static_cast<int32_t>(byte & 0x0F) - 8;
        const int32_t hi = static_cast<int32_t>(byte >> 4) - 8;
        acc += static_cast<float>(lo) * x[j] + static_cast<float>(hi) * x[j + QK / 2];
    }
    return acc * d;
}

float dot_q4_0_block_f32(std::span<const uint8_t> block, std::span<const float> x) {
    check_block(block, Q4_0_BLOCK_BYTES, "dot_q4_0_block_f32: block must be 18 bytes");
    if (x.size() != QK) panic("dot_q4_0_block_f32: x must have 32 elements");
#if SAPIENT_AARCH64
    return dot_q4_0_block_neon(block.data(), x.data());
#else
    return detail::dot_q4_0_block_scalar(block, x);
#endif
}

float dot_q4_0_row_f32(std::span<const uint8_t> row_blocks, std::span<const float> x) {
    const size_t k = x.size();
    if (k % QK != 0) panic("dot_q4_0_row_f32: k must be a multiple of 32");
    const size_t nb = row_blocks.size() / Q4_0_BLOCK_BYTES; // chunks_exact
    check_len(k, nb * QK, "dot_q4_0_row_f32: x shorter than the row");
    float acc = 0.0f;
    for (size_t b = 0; b < nb; ++b)
        acc += dot_q4_0_block_f32(row_blocks.subspan(b * Q4_0_BLOCK_BYTES, Q4_0_BLOCK_BYTES),
                                  x.subspan(b * QK, QK));
    return acc;
}

std::vector<uint8_t> quantize_q4_0_row(std::span<const float> w) {
    if (w.size() % QK != 0) panic("quantize_q4_0_row: length must be a multiple of 32");
    std::vector<uint8_t> out;
    out.reserve(w.size() / QK * Q4_0_BLOCK_BYTES);
    for (size_t b = 0; b + QK <= w.size(); b += QK) {
        const auto blk = quantize_q4_0_block(w.subspan(b, QK));
        out.insert(out.end(), blk.begin(), blk.end());
    }
    return out;
}

// ── Q8_0 ─────────────────────────────────────────────────────────────────────

std::array<uint8_t, Q8_0_BLOCK_BYTES> quantize_q8_0_block(std::span<const float> x) {
    if (x.size() != QK) panic("quantize_q8_0_block: x must have 32 elements");
    float max_abs = 0.0f; // .map(|v| v.abs()).fold(0.0, f32::max) — NaN-dropping
    for (const float v : x)
        max_abs = ::fmaxf(max_abs, ::fabsf(v));
    const float scale = max_abs / 127.0f;
    const uint16_t d = sapient::core::f32_to_f16_bits(scale);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    std::array<uint8_t, Q8_0_BLOCK_BYTES> out{};
    sapient::core::f16_to_le(d, out.data());
    for (size_t i = 0; i < QK; ++i)
        out[2 + i] = static_cast<uint8_t>(detail::round_clamp_i8(x[i] * inv_scale));
    return out;
}

float detail::dot_q8_0_block_scalar(std::span<const uint8_t> block, std::span<const float> x) {
    const float d = f16_le_to_f32(block.data());
    float acc = 0.0f;
    for (size_t j = 0; j < QK; ++j)
        acc += static_cast<float>(detail::i8v(block[2 + j])) * x[j];
    return acc * d;
}

float dot_q8_0_block_f32(std::span<const uint8_t> block, std::span<const float> x) {
    check_block(block, Q8_0_BLOCK_BYTES, "dot_q8_0_block_f32: block must be 34 bytes");
    if (x.size() != QK) panic("dot_q8_0_block_f32: x must have 32 elements");
#if SAPIENT_AARCH64
    return dot_q8_0_block_neon(block.data(), x.data());
#else
    return detail::dot_q8_0_block_scalar(block, x);
#endif
}

// ── activation quantisers (quant.rs:317-392) ─────────────────────────────────

I8Blocks quantize_row_to_i8_blocks(std::span<const float> x) {
    if (x.size() % QK != 0) panic("quantize_row_to_i8_blocks: length must be a multiple of 32");
    const size_t nblocks = x.size() / QK;
    I8Blocks r{std::vector<int8_t>(x.size(), 0), std::vector<float>(nblocks, 0.0f)};
    for (size_t b = 0; b < nblocks; ++b) {
        const auto blk = x.subspan(b * QK, QK);
        float max_abs = 0.0f;
        for (const float v : blk)
            max_abs = ::fmaxf(max_abs, ::fabsf(v));
        const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        for (size_t i = 0; i < QK; ++i)
            r.q[b * QK + i] = detail::round_clamp_i8(blk[i] * inv);
        r.scales[b] = scale;
    }
    return r;
}

std::vector<int32_t> i8_block_sums(std::span<const int8_t> q) {
    if (q.size() % QK != 0) panic("i8_block_sums: length must be a multiple of 32");
    std::vector<int32_t> sums(q.size() / QK, 0);
    for (size_t b = 0; b < sums.size(); ++b) {
        int32_t s = 0;
        for (size_t i = 0; i < QK; ++i)
            s += static_cast<int32_t>(q[b * QK + i]);
        sums[b] = s;
    }
    return sums;
}

Q8kRow quantize_row_to_q8k(std::span<const float> x) {
    if (x.size() % QK_K != 0) panic("quantize_row_to_q8k: length must be a multiple of 256");
    const size_t nsuper = x.size() / QK_K;
    Q8kRow r{std::vector<int8_t>(x.size(), 0),
             std::vector<float>(nsuper, 0.0f),
             std::vector<int32_t>(x.size() / QK, 0)};
    for (size_t b = 0; b < nsuper; ++b) {
        const auto blk = x.subspan(b * QK_K, QK_K);
        float max_abs = 0.0f;
        for (const float v : blk)
            max_abs = ::fmaxf(max_abs, ::fabsf(v));
        const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        for (size_t i = 0; i < QK_K; ++i)
            r.q[b * QK_K + i] = detail::round_clamp_i8(blk[i] * inv);
        r.scales[b] = scale;
        for (size_t j = 0; j < QK_K / QK; ++j) {
            const size_t base = b * QK_K + j * QK;
            int32_t s = 0;
            for (size_t i = 0; i < QK; ++i)
                s += static_cast<int32_t>(r.q[base + i]);
            r.sums[b * (QK_K / QK) + j] = s;
        }
    }
    return r;
}

// ── SDOT (ARMv8.4-A dotprod) Q8_0 (quant.rs:442-500) ─────────────────────────

#if SAPIENT_AARCH64
SAPIENT_TARGET_DOTPROD int32_t detail::dot_q8_0_block_sdot(std::span<const uint8_t> block,
                                                           std::span<const int8_t> x_i8) {
    check_block(block, Q8_0_BLOCK_BYTES, "dot_q8_0_block_sdot: block must be 34 bytes");
    if (x_i8.size() != QK) panic("dot_q8_0_block_sdot: x_i8 must have 32 elements");
    const int8_t* w_ptr = reinterpret_cast<const int8_t*>(block.data() + 2);
    const int8_t* x_ptr = x_i8.data();
    const int8x16_t w0 = vld1q_s8(w_ptr);
    const int8x16_t x0 = vld1q_s8(x_ptr);
    const int8x16_t w1 = vld1q_s8(w_ptr + 16);
    const int8x16_t x1 = vld1q_s8(x_ptr + 16);
    int32x4_t acc = vdupq_n_s32(0);
    acc = vdotq_s32(acc, w0, x0); // sdot v_acc.4s, v_w.16b, v_x.16b
    acc = vdotq_s32(acc, w1, x1);
    return vaddvq_s32(acc);
}

SAPIENT_TARGET_DOTPROD float dot_q8_0_row_sdot(std::span<const uint8_t> row_blocks,
                                               std::span<const int8_t> x_i8,
                                               std::span<const float> x_scales) {
    const size_t nb = row_blocks.size() / Q8_0_BLOCK_BYTES;
    check_len(x_i8.size(), nb * QK, "dot_q8_0_row_sdot: x_i8 shorter than the row");
    check_len(x_scales.size(), nb, "dot_q8_0_row_sdot: x_scales shorter than the row");
    float acc = 0.0f;
    size_t x_off = 0;
    for (size_t bi = 0; bi < nb; ++bi) {
        const auto block = row_blocks.subspan(bi * Q8_0_BLOCK_BYTES, Q8_0_BLOCK_BYTES);
        const float w_scale = f16_le_to_f32(block.data());
        const int32_t dot = detail::dot_q8_0_block_sdot(block, x_i8.subspan(x_off, QK));
        acc += w_scale * x_scales[bi] * static_cast<float>(dot);
        x_off += QK;
    }
    return acc;
}
#endif

float dot_q8_0_row_i8_scalar(std::span<const uint8_t> row_blocks,
                             std::span<const int8_t> x_i8,
                             std::span<const float> x_scales) {
    const size_t nb = row_blocks.size() / Q8_0_BLOCK_BYTES;
    check_len(x_i8.size(), nb * QK, "dot_q8_0_row_i8_scalar: x_i8 shorter than the row");
    check_len(x_scales.size(), nb, "dot_q8_0_row_i8_scalar: x_scales shorter than the row");
    float acc = 0.0f;
    size_t x_off = 0;
    for (size_t bi = 0; bi < nb; ++bi) {
        const uint8_t* block = row_blocks.data() + bi * Q8_0_BLOCK_BYTES;
        const float w_scale = f16_le_to_f32(block);
        int32_t dot = 0; // zip().map().sum::<i32>()
        for (size_t j = 0; j < QK; ++j)
            dot += detail::i8v(block[2 + j]) * static_cast<int32_t>(x_i8[x_off + j]);
        acc += w_scale * x_scales[bi] * static_cast<float>(dot);
        x_off += QK;
    }
    return acc;
}

// ── AVX2+FMA Q8_0 row dot — the ONE x86 SIMD kernel (quant.rs:543-580) ───────

#if SAPIENT_X86_64
__attribute__((target("avx2,fma"))) float
detail::dot_q8_0_row_avx2(std::span<const uint8_t> row_blocks, std::span<const float> x) {
    const size_t k = x.size();
    if (k % QK != 0) panic("dot_q8_0_row_avx2: k must be a multiple of 32");
    const size_t nb = row_blocks.size() / Q8_0_BLOCK_BYTES;
    check_len(k, nb * QK, "dot_q8_0_row_avx2: x shorter than the row");
    __m256 row_acc = _mm256_setzero_ps();
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* block = row_blocks.data() + b * Q8_0_BLOCK_BYTES;
        const float scale = f16_le_to_f32(block);
        const uint8_t* q_ptr = block + 2; // Rust: `*const i32`, stepped 4 bytes per unit
        const float* xp = x.data() + b * QK;
        __m256 block_acc = _mm256_setzero_ps();
        for (size_t g = 0; g < 4; ++g) {
            // 4 bytes each (zero upper lanes) → 8 i32 lanes of which 4 carry quants.
            const __m128i q_i32_4 = _mm_loadu_si32(q_ptr + 8 * g);
            const __m128i q_i32_4b = _mm_loadu_si32(q_ptr + 8 * g + 4);
            const __m256i q_a = _mm256_cvtepi8_epi32(q_i32_4);
            const __m256i q_b = _mm256_cvtepi8_epi32(q_i32_4b);
            const __m256 xv_a = _mm256_loadu_ps(xp + g * 8); // x[8g, 8g+8): in-bounds
            // Rust loads 8 floats from xp + g*8 + 4 here — at g == 3 that runs 4 floats past
            // the block (past the end of x on the row's last block). Those lanes only multiply
            // q_b's zero lanes, so an in-bounds, zero-extended load is bit-identical for finite x
            // (plan D ruling; recorded in docs/PARITY.md as a known oracle defect).
            const __m256 xv_b =
                _mm256_insertf128_ps(_mm256_setzero_ps(), _mm_loadu_ps(xp + g * 8 + 4), 0);
            const __m256 qf_a = _mm256_cvtepi32_ps(q_a);
            const __m256 qf_b = _mm256_cvtepi32_ps(q_b);
            block_acc = _mm256_fmadd_ps(qf_a, xv_a, block_acc);
            block_acc = _mm256_fmadd_ps(qf_b, xv_b, block_acc);
        }
        const __m256 scale_v = _mm256_set1_ps(scale);
        row_acc = _mm256_fmadd_ps(block_acc, scale_v, row_acc);
    }
    // Horizontal sum of the 8-lane accumulator — the exact sequence of quant.rs:572-579.
    const __m128 lo = _mm256_castps256_ps128(row_acc);
    const __m128 hi = _mm256_extractf128_ps(row_acc, 1);
    const __m128 sum4 = _mm_add_ps(lo, hi);
    const __m128 shuf = _mm_movehdup_ps(sum4);
    const __m128 sum2 = _mm_add_ps(sum4, shuf);
    const __m128 sum1 = _mm_add_ss(sum2, _mm_movehl_ps(shuf, sum2));
    return _mm_cvtss_f32(sum1);
}
#endif

float dot_q8_0_row_f32(std::span<const uint8_t> row_blocks, std::span<const float> x) {
#if SAPIENT_X86_64
    if (cpu_features::has_avx2_fma()) return detail::dot_q8_0_row_avx2(row_blocks, x);
#endif
    const size_t k = x.size();
    if (k % QK != 0) panic("dot_q8_0_row_f32: k must be a multiple of 32");
    const size_t nb = row_blocks.size() / Q8_0_BLOCK_BYTES;
    check_len(k, nb * QK, "dot_q8_0_row_f32: x shorter than the row");
    float acc = 0.0f;
    for (size_t b = 0; b < nb; ++b)
        acc += dot_q8_0_block_f32(row_blocks.subspan(b * Q8_0_BLOCK_BYTES, Q8_0_BLOCK_BYTES),
                                  x.subspan(b * QK, QK));
    return acc;
}

// ── K-quants: Q4_K (Task 2) ──────────────────────────────────────────────────

} // namespace sapient::backends_cpu::kernels::quant
```

The anonymous-namespace NEON helpers take raw pointers because the public callers already validated the spans.

- [ ] **Step 5: Build and run the family**

`cd cpp && cmake --build --preset dev && ./build/dev/libs/sapient-backends-cpu/sapient_backends_cpu_tests --gtest_filter='Quant*'` (find the binary path with `ctest --preset dev -N | head` if it differs). Expected: all `Quant.*` and `QuantDeath.*` pass (the two SDOT tests run, this Mac has dotprod).

- [ ] **Step 6: Attribute-presence and x86 codegen checks (Global Constraints rule 5)** — both commands must succeed. Expected error text if an attribute is missing: `always_inline function 'vdotq_s32' requires target feature 'dotprod'`.

- [ ] **Step 7: Format, run the whole suite, commit**

```bash
git add cpp/libs/sapient-backends-cpu/CMakeLists.txt cpp/libs/sapient-backends-cpu/include/sapient/backends_cpu/kernels.hpp cpp/libs/sapient-backends-cpu/include/sapient/backends_cpu/kernels/quant.hpp cpp/libs/sapient-backends-cpu/src/kernels/quant.cpp cpp/libs/sapient-backends-cpu/tests/quant_test.cpp
.superpowers/tools-venv/bin/clang-format -i $(git diff --cached --name-only -- 'cpp/*.hpp' 'cpp/*.cpp')
git add -u cpp
cd cpp && cmake --build --preset dev && ctest --preset dev && cd ..
git commit -m "cpp(backends-cpu): kernels/quant — Q4_0/Q8_0 blocks, activation quantisers, SDOT and AVX2 Q8_0 rows

Port of quant.rs Q4_0/Q8_0 + quantize_row_to_{i8_blocks,q8k} + i8_block_sums with the Rust
rounding rules (roundf half-away, saturating as-casts, fmaxf); the AVX2 row dot loads its second
half in-bounds (Rust reads 4 floats past the block; zero lanes — bit-identical for finite x).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

Expected ctest: 160 + 9 = **169** entries, all pass (no dumps: the 31 golden skips as before).

---

### Task 2: Q4_K — f32 dot (scalar + NEON), W4A8 per-32 (scalar + SDOT), 4-row, repack, R4

**Files:**
- Modify: `quant.hpp` (fill the `Q4_K (Task 2)` section), `quant.cpp` (append after the Task 1 section), `tests/quant_test.cpp` (append)

**Interfaces:**
- Consumes: Task 1 (`I8Blocks`, `quantize_row_to_i8_blocks`, `i8_block_sums`, `detail::i8v`, the `check_len` helper, `SAPIENT_TARGET_DOTPROD`), `sapient::core::dequant::get_scale_min_k4(size_t, const uint8_t*) -> ScaleMin{sc, m}`, `sapient::core::Tensor::{from_quant_bytes, to_f32_vec}` (test only).
- Produces: `dot_q4_k_row_f32(row_data, x)`, `dot_q4_k_row_q8_scalar(row_data, x_i8, x_scales, x_sums)`, `repack_q4_k_rows4(blocks, n, k) -> std::vector<uint8_t>`; aarch64 (dotprod): `dot_q4_k_row_q8_neon(row_data, x_i8, x_scales, x_sums)`, `dot_q4_k_4rows_q8_neon(std::array<std::span<const uint8_t>, 4> rows, x_i8, x_scales, x_sums) -> std::array<float, 4>`, `dot_q4_k_4rows_r4_neon(packed, x_i8, x_scales, x_sums) -> std::array<float, 4>`; `detail::dot_q4_k_row_f32_scalar`, aarch64 `detail::dot_q4_k_row_f32_neon`; private in `quant.cpp`: `sdot_s32(int32x4_t, int8x16_t, int8x16_t)` (`SAPIENT_TARGET_DOTPROD static inline`), `q4k_header(const uint8_t* block) -> std::pair<float,float>` (d, dmin). Task 3 and Task 6 consume these.

- [ ] **Step 1: Append the failing tests**

```cpp
// ── Q4_K (Task 2) ────────────────────────────────────────────────────────────

namespace {
// quant.rs: 8 pseudo-random Q4_K rows with SMALL positive f16 d/dmin (bytes 0x11,0x2c) so the
// magnitudes stay sane; the LCG only advances on non-header bytes (the `_ => nb()` arm).
std::vector<uint8_t> q4_k_test_rows(size_t n, size_t k, uint64_t seed) {
    const size_t row_bytes = k / 256 * Q4_K_BLOCK_BYTES;
    std::vector<uint8_t> rows(n * row_bytes);
    for (size_t i = 0; i < rows.size(); ++i) {
        switch (i % Q4_K_BLOCK_BYTES) {
        case 0:
        case 2:
            rows[i] = 0x11;
            break;
        case 1:
        case 3:
            rows[i] = 0x2c;
            break;
        default:
            rows[i] = static_cast<uint8_t>(lcg_step(seed) >> 33);
        }
    }
    return rows;
}
std::vector<float> ramp(size_t k, size_t mul, size_t mod, float sub, float step) {
    std::vector<float> x(k);
    for (size_t i = 0; i < k; ++i)
        x[i] = (static_cast<float>(i * mul % mod) - sub) * step;
    return x;
}
std::span<const uint8_t> row_of(const std::vector<uint8_t>& rows, size_t r, size_t row_bytes) {
    return std::span<const uint8_t>(rows).subspan(r * row_bytes, row_bytes);
}
} // namespace

TEST(Quant, q4_k_w4a8_matches_f32_path) {
    // The W4A8 (int8-activation) Q4_K dot must agree with the proven f32 path within
    // activation-quantization error. A layout/scale bug (the kind that produced Q6_K salad) shows
    // up as a wildly-wrong result, not a few %.
    uint64_t seed = 0x12345678ULL;
    auto next = [&seed]() { return static_cast<uint32_t>(lcg_step(seed) >> 33); };
    const size_t nblocks = 2; // 512 weights → 16 activation blocks of 32
    std::vector<uint8_t> row(nblocks * Q4_K_BLOCK_BYTES, 0);
    for (size_t b = 0; b < nblocks; ++b) {
        uint8_t* blk = row.data() + b * Q4_K_BLOCK_BYTES;
        sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(0.05f), blk);      // d
        sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(0.018f), blk + 2); // dmin
        for (size_t i = 4; i < Q4_K_BLOCK_BYTES; ++i)
            blk[i] = static_cast<uint8_t>(next() & 0xFF); // scales + packed nibbles
    }
    std::vector<float> x(nblocks * QK_K);
    for (float& v : x)
        v = (static_cast<float>(next()) / static_cast<float>(std::numeric_limits<uint32_t>::max())) *
                4.0f -
            2.0f;

    const float f32_dot = dot_q4_k_row_f32(row, x);
    const auto xq = quantize_row_to_i8_blocks(x);
    const auto xsums = i8_block_sums(xq.q);
    const float q8_dot = dot_q4_k_row_q8_scalar(row, xq.q, xq.scales, xsums);
    const float rel = ::fabsf(f32_dot - q8_dot) / ::fmaxf(::fabsf(f32_dot), 1e-3f);
    EXPECT_LT(rel, 0.03f) << "W4A8 mismatch: f32=" << f32_dot << " q8=" << q8_dot;

    // The NEON SDOT kernel must match the scalar W4A8 reference exactly (same integer dot; only
    // f32 reduction order differs → tiny tolerance).
#if defined(__aarch64__) || defined(_M_ARM64)
    if (has_dotprod()) {
        const float neon = dot_q4_k_row_q8_neon(row, xq.q, xq.scales, xsums);
        const float rel_n = ::fabsf(neon - q8_dot) / ::fmaxf(::fabsf(q8_dot), 1e-3f);
        EXPECT_LT(rel_n, 1e-4f) << "NEON≠scalar W4A8: neon=" << neon << " scalar=" << q8_dot;
    }
#endif
}

#if defined(__aarch64__) || defined(_M_ARM64)
TEST(Quant, q4_k_4rows_matches_single_row) {
    if (!has_dotprod()) GTEST_SKIP() << "dotprod not available";
    const size_t k = 512;
    const size_t row_bytes = k / 256 * Q4_K_BLOCK_BYTES;
    const auto rows = q4_k_test_rows(8, k, 0x5EEDULL);
    const auto x = ramp(k, 37, 97, 48.0f, 0.02f);
    const auto xq = quantize_row_to_i8_blocks(x);
    const auto x_sums = i8_block_sums(xq.q);
    for (size_t group = 0; group < 2; ++group) {
        const size_t j = group * 4;
        const std::array<std::span<const uint8_t>, 4> r4 = {row_of(rows, j, row_bytes),
                                                            row_of(rows, j + 1, row_bytes),
                                                            row_of(rows, j + 2, row_bytes),
                                                            row_of(rows, j + 3, row_bytes)};
        const auto got = dot_q4_k_4rows_q8_neon(r4, xq.q, xq.scales, x_sums);
        for (size_t o = 0; o < 4; ++o) {
            const float want = dot_q4_k_row_q8_neon(r4[o], xq.q, xq.scales, x_sums);
            EXPECT_EQ(bits(got[o]), bits(want))
                << "row " << j + o << " differs: " << got[o] << " vs " << want;
        }
    }
}
#endif

TEST(Quant, q4_k_r4_repack_roundtrips_through_dequant) {
    // to_f32_vec on a repacked tensor must equal to_f32_vec on the original (the de-interleave
    // map is the inverse of the repack permutation).
    using sapient::core::DType;
    using sapient::core::Shape;
    using sapient::core::Tensor;
    const size_t n = 8, k = 512;
    const auto blocks = q4_k_test_rows(n, k, 0x00D5ULL);
    auto orig = Tensor::from_quant_bytes(blocks, Shape{n, k}, DType::Q4_K);
    ASSERT_TRUE(orig.has_value()) << orig.error().to_string();
    const auto packed = repack_q4_k_rows4(blocks, n, k);
    auto r4 = Tensor::from_quant_bytes(packed, Shape{n, k}, DType::Q4_K_R4);
    ASSERT_TRUE(r4.has_value()) << r4.error().to_string();
    EXPECT_EQ(orig->to_f32_vec(), r4->to_f32_vec());
}

#if defined(__aarch64__) || defined(_M_ARM64)
TEST(Quant, q4_k_r4_kernel_matches_single_row) {
    if (!has_dotprod()) GTEST_SKIP() << "dotprod not available";
    const size_t n = 4, k = 512;
    const size_t row_bytes = k / 256 * Q4_K_BLOCK_BYTES;
    const auto blocks = q4_k_test_rows(n, k, 0x0B0BULL);
    const auto x = ramp(k, 53, 89, 44.0f, 0.02f);
    const auto xq = quantize_row_to_i8_blocks(x);
    const auto x_sums = i8_block_sums(xq.q);
    const auto packed = repack_q4_k_rows4(blocks, n, k);
    const auto got = dot_q4_k_4rows_r4_neon(packed, xq.q, xq.scales, x_sums);
    for (size_t r = 0; r < 4; ++r) {
        const float want = dot_q4_k_row_q8_neon(row_of(blocks, r, row_bytes), xq.q, xq.scales, x_sums);
        EXPECT_EQ(bits(got[r]), bits(want)) << "row " << r << ": " << got[r] << " vs " << want;
    }
}
#endif

TEST(QuantDeath, repack_q4_k_rows4_asserts_like_rust) {
    const std::vector<uint8_t> two_rows(2 * Q4_K_BLOCK_BYTES, 0);
    EXPECT_DEATH((void)repack_q4_k_rows4(two_rows, 2, 256), "multiple of 4");
    const std::vector<uint8_t> four_rows(4 * Q4_K_BLOCK_BYTES, 0);
    EXPECT_DEATH((void)repack_q4_k_rows4(four_rows, 4, 200), "");
    EXPECT_DEATH((void)repack_q4_k_rows4(two_rows, 4, 256), ""); // blocks.len() != n * row_bytes
}
```

`tests/quant_test.cpp` needs `#include "sapient/core/dtype.hpp"`, `"sapient/core/shape.hpp"` and `"sapient/core/tensor.hpp"` for the round-trip test — add them to the include block.

- [ ] **Step 2: Build → FAIL** (undeclared `dot_q4_k_row_f32` …).

- [ ] **Step 3: Header declarations** (replace the `// ── Q4_K (Task 2)` marker line):

```cpp
// ── Q4_K ────────────────────────────────────────────────────────────────────────
// Block: [0..2) d f16 | [2..4) dmin f16 | [4..16) 12 packed 6-bit (scale,min) pairs | [16..144) 128
// nibble bytes. Per 64-weight group g: lo nibbles ↔ x[64g..64g+32) with (sc,m) pair 2g, hi nibbles
// ↔ x[64g+32..64g+64) with pair 2g+1.
/// Row · f32 activations (NEON on aarch64, scalar elsewhere).
float dot_q4_k_row_f32(std::span<const uint8_t> row_data, std::span<const float> x);
/// W4A8: row · per-32 int8 activations; `x_sums` = `i8_block_sums(x_i8)` (precomputed, never
/// re-reduced). Scalar reference for the SDOT kernels (same integer dot, same f32 combine order).
float dot_q4_k_row_q8_scalar(std::span<const uint8_t> row_data,
                             std::span<const int8_t> x_i8,
                             std::span<const float> x_scales,
                             std::span<const int32_t> x_sums);
/// Repack `n` Q4_K rows into the Q4_K_R4 layout: groups of 4 rows, super-blocks block-major within
/// the group (`[r0.b0, r1.b0, r2.b0, r3.b0, r0.b1, …]`). Panics unless `n % 4 == 0`, `k % 256 == 0`
/// and `blocks.size() == n · k/256 · 144`.
std::vector<uint8_t> repack_q4_k_rows4(std::span<const uint8_t> blocks, size_t n, size_t k);
#if defined(__aarch64__) || defined(_M_ARM64)
/// NEON W4A8 row dot via `sdot`; bit-identical to `dot_q4_k_row_q8_scalar`. Precondition: dotprod.
float dot_q4_k_row_q8_neon(std::span<const uint8_t> row_data,
                           std::span<const int8_t> x_i8,
                           std::span<const float> x_scales,
                           std::span<const int32_t> x_sums);
/// Four row-major Q4_K rows against ONE int8 activation row (activations loaded once per 64-weight
/// group); each lane bit-identical to `dot_q4_k_row_q8_neon`. Precondition: dotprod.
std::array<float, 4> dot_q4_k_4rows_q8_neon(std::array<std::span<const uint8_t>, 4> rows,
                                            std::span<const int8_t> x_i8,
                                            std::span<const float> x_scales,
                                            std::span<const int32_t> x_sums);
/// Four Q4_K rows in the R4 layout (`packed` = one whole row-group) against one int8 activation
/// row; each lane bit-identical to `dot_q4_k_row_q8_neon`. Precondition: dotprod.
std::array<float, 4> dot_q4_k_4rows_r4_neon(std::span<const uint8_t> packed,
                                            std::span<const int8_t> x_i8,
                                            std::span<const float> x_scales,
                                            std::span<const int32_t> x_sums);
#endif
namespace detail {
float dot_q4_k_row_f32_scalar(std::span<const uint8_t> row_data, std::span<const float> x);
#if defined(__aarch64__) || defined(_M_ARM64)
float dot_q4_k_row_f32_neon(std::span<const uint8_t> row_data, std::span<const float> x);
#endif
} // namespace detail
```

- [ ] **Step 4: Definitions** (append to `quant.cpp` before the closing namespace, replacing the `// ── K-quants: Q4_K (Task 2)` marker; add `#include <utility>` to the include block):

```cpp
// ── K-quants: Q4_K (quant.rs:588-1330) ───────────────────────────────────────

namespace {

using sapient::core::dequant::get_scale_min_k4;

// (d, dmin) of a Q4_K/Q5_K super-block header.
std::pair<float, float> q4k_header(const uint8_t* block) {
    return {f16_le_to_f32(block), f16_le_to_f32(block + 2)};
}

// Entry checks for the activation-side spans of a K-quant row of `nb` super-blocks.
void check_q8_row(size_t nb,
                  std::span<const int8_t> x_i8,
                  std::span<const float> x_scales,
                  size_t scales_per_block,
                  const char* who) {
    check_len(x_i8.size(), nb * QK_K, who);
    check_len(x_scales.size(), nb * scales_per_block, who);
}

#if SAPIENT_AARCH64
// Accumulate four 4-element i8 dot products into an i32x4 — Rust's `sdot_s32` inline asm.
SAPIENT_TARGET_DOTPROD inline int32x4_t sdot_s32(int32x4_t acc, int8x16_t w, int8x16_t x) {
    return vdotq_s32(acc, w, x);
}
#endif

} // namespace

float detail::dot_q4_k_row_f32_scalar(std::span<const uint8_t> row_data, std::span<const float> x) {
    const size_t nb = row_data.size() / Q4_K_BLOCK_BYTES;
    check_len(x.size(), nb * QK_K, "dot_q4_k_row_f32: x shorter than the row");
    float acc = 0.0f;
    size_t x_off = 0;
    for (size_t bi = 0; bi < nb; ++bi) {
        const uint8_t* block = row_data.data() + bi * Q4_K_BLOCK_BYTES;
        const auto [d, dmin] = q4k_header(block);
        const uint8_t* scales = block + 4;
        const uint8_t* qs = block + 16;
        size_t q_off = 0;
        size_t is = 0;
        for (size_t g = 0; g < QK_K / 64; ++g) {
            const auto [sc1, m1] = get_scale_min_k4(is, scales);
            const float d1 = d * static_cast<float>(sc1);
            const float m1v = dmin * static_cast<float>(m1);
            const auto [sc2, m2] = get_scale_min_k4(is + 1, scales);
            const float d2 = d * static_cast<float>(sc2);
            const float m2v = dmin * static_cast<float>(m2);
            for (size_t l = 0; l < 32; ++l) {
                acc += (d1 * static_cast<float>(qs[q_off + l] & 0x0F) - m1v) * x[x_off + l];
                acc += (d2 * static_cast<float>(qs[q_off + l] >> 4) - m2v) * x[x_off + l + 32];
            }
            x_off += 64;
            q_off += 32;
            is += 2;
        }
    }
    return acc;
}

#if SAPIENT_AARCH64
// NEON Q4_K row dot: 8 packed bytes (16 nibbles) per iteration, FMA for the lo- and hi-nibble
// sub-blocks, plus the Σx vectors for the min correction (quant.rs:661-748).
float detail::dot_q4_k_row_f32_neon(std::span<const uint8_t> row_data, std::span<const float> x) {
    const size_t nb = row_data.size() / Q4_K_BLOCK_BYTES;
    check_len(x.size(), nb * QK_K, "dot_q4_k_row_f32: x shorter than the row");
    float acc = 0.0f;
    size_t x_off = 0;
    const uint8x8_t mask4 = vdup_n_u8(0x0F);
    for (size_t bi = 0; bi < nb; ++bi) {
        const uint8_t* block = row_data.data() + bi * Q4_K_BLOCK_BYTES;
        const auto [d, dmin] = q4k_header(block);
        const uint8_t* scales = block + 4;
        const uint8_t* qs = block + 16;
        size_t q_off = 0;
        size_t is = 0;
        for (size_t g = 0; g < QK_K / 64; ++g) {
            const auto [sc1, m1] = get_scale_min_k4(is, scales);
            const auto [sc2, m2] = get_scale_min_k4(is + 1, scales);
            const float d1 = d * static_cast<float>(sc1);
            const float m1v = dmin * static_cast<float>(m1);
            const float d2 = d * static_cast<float>(sc2);
            const float m2v = dmin * static_cast<float>(m2);
            const float* x_lo = x.data() + x_off;
            const float* x_hi = x.data() + x_off + 32;

            float32x4_t vsum_lo = vdupq_n_f32(0.0f); // dot(lo_nibbles, x_lo)
            float32x4_t vsum_hi = vdupq_n_f32(0.0f); // dot(hi_nibbles, x_hi)
            float32x4_t vsum_xl = vdupq_n_f32(0.0f); // sum(x_lo) for the min correction
            float32x4_t vsum_xh = vdupq_n_f32(0.0f); // sum(x_hi)
            for (size_t chunk = 0; chunk < 4; ++chunk) {
                const uint8x8_t q8 = vld1_u8(qs + q_off + chunk * 8);
                const uint8x8_t lo8 = vand_u8(q8, mask4);
                const uint8x8_t hi8 = vshr_n_u8(q8, 4);
                const uint16x8_t lo16 = vmovl_u8(lo8);
                const float32x4_t lof0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16)));
                const float32x4_t lof1 = vcvtq_f32_u32(vmovl_high_u16(lo16));
                const uint16x8_t hi16 = vmovl_u8(hi8);
                const float32x4_t hif0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16)));
                const float32x4_t hif1 = vcvtq_f32_u32(vmovl_high_u16(hi16));
                const float32x4_t xl0 = vld1q_f32(x_lo + chunk * 8);
                const float32x4_t xl1 = vld1q_f32(x_lo + chunk * 8 + 4);
                const float32x4_t xh0 = vld1q_f32(x_hi + chunk * 8);
                const float32x4_t xh1 = vld1q_f32(x_hi + chunk * 8 + 4);
                vsum_lo = vfmaq_f32(vsum_lo, lof0, xl0);
                vsum_lo = vfmaq_f32(vsum_lo, lof1, xl1);
                vsum_hi = vfmaq_f32(vsum_hi, hif0, xh0);
                vsum_hi = vfmaq_f32(vsum_hi, hif1, xh1);
                vsum_xl = vaddq_f32(vsum_xl, vaddq_f32(xl0, xl1));
                vsum_xh = vaddq_f32(vsum_xh, vaddq_f32(xh0, xh1));
            }
            acc += d1 * vaddvq_f32(vsum_lo) - m1v * vaddvq_f32(vsum_xl);
            acc += d2 * vaddvq_f32(vsum_hi) - m2v * vaddvq_f32(vsum_xh);
            x_off += 64;
            q_off += 32;
            is += 2;
        }
    }
    return acc;
}
#endif

float dot_q4_k_row_f32(std::span<const uint8_t> row_data, std::span<const float> x) {
#if SAPIENT_AARCH64
    return detail::dot_q4_k_row_f32_neon(row_data, x);
#else
    return detail::dot_q4_k_row_f32_scalar(row_data, x);
#endif
}

float dot_q4_k_row_q8_scalar(std::span<const uint8_t> row_data,
                             std::span<const int8_t> x_i8,
                             std::span<const float> x_scales,
                             std::span<const int32_t> x_sums) {
    const size_t nb = row_data.size() / Q4_K_BLOCK_BYTES;
    check_q8_row(nb, x_i8, x_scales, QK_K / QK, "dot_q4_k_row_q8_scalar: activations shorter than the row");
    check_len(x_sums.size(), nb * (QK_K / QK), "dot_q4_k_row_q8_scalar: x_sums shorter than the row");
    float acc = 0.0f;
    size_t x_off = 0;
    for (size_t bi = 0; bi < nb; ++bi) {
        const uint8_t* block = row_data.data() + bi * Q4_K_BLOCK_BYTES;
        const auto [d, dmin] = q4k_header(block);
        const uint8_t* scales = block + 4;
        const uint8_t* qs = block + 16;
        size_t q_off = 0;
        size_t is = 0;
        for (size_t g = 0; g < QK_K / 64; ++g) {
            const auto [sc1, m1] = get_scale_min_k4(is, scales);
            const auto [sc2, m2] = get_scale_min_k4(is + 1, scales);
            const float d1 = d * static_cast<float>(sc1);
            const float m1v = dmin * static_cast<float>(m1);
            const float d2 = d * static_cast<float>(sc2);
            const float m2v = dmin * static_cast<float>(m2);
            // lo nibbles → activation block at x_off; hi nibbles → block at x_off + 32.
            const size_t blk_lo = x_off / QK;
            const size_t blk_hi = (x_off + 32) / QK;
            const int8_t* xlo = x_i8.data() + x_off;
            const int8_t* xhi = x_i8.data() + x_off + 32;
            int32_t dot_lo = 0;
            int32_t dot_hi = 0;
            for (size_t l = 0; l < 32; ++l) {
                const int32_t nlo = static_cast<int32_t>(qs[q_off + l] & 0x0F);
                const int32_t nhi = static_cast<int32_t>(qs[q_off + l] >> 4);
                dot_lo += nlo * static_cast<int32_t>(xlo[l]);
                dot_hi += nhi * static_cast<int32_t>(xhi[l]);
            }
            // Σx per sub-block comes precomputed (once per activation row).
            acc += x_scales[blk_lo] *
                   (d1 * static_cast<float>(dot_lo) - m1v * static_cast<float>(x_sums[blk_lo]));
            acc += x_scales[blk_hi] *
                   (d2 * static_cast<float>(dot_hi) - m2v * static_cast<float>(x_sums[blk_hi]));
            x_off += 64;
            q_off += 32;
            is += 2;
        }
    }
    return acc;
}

#if SAPIENT_AARCH64
// NEON W4A8 Q4_K row dot — the fast decode kernel: `sdot` on the nibbles vs int8 activations,
// Σx from the precomputed block sums (quant.rs:1084-1141). Bit-identical to the scalar W4A8.
SAPIENT_TARGET_DOTPROD float dot_q4_k_row_q8_neon(std::span<const uint8_t> row_data,
                                                  std::span<const int8_t> x_i8,
                                                  std::span<const float> x_scales,
                                                  std::span<const int32_t> x_sums) {
    const size_t nb = row_data.size() / Q4_K_BLOCK_BYTES;
    check_q8_row(nb, x_i8, x_scales, QK_K / QK, "dot_q4_k_row_q8_neon: activations shorter than the row");
    check_len(x_sums.size(), nb * (QK_K / QK), "dot_q4_k_row_q8_neon: x_sums shorter than the row");
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    float acc = 0.0f;
    size_t x_off = 0;
    for (size_t bi = 0; bi < nb; ++bi) {
        const uint8_t* block = row_data.data() + bi * Q4_K_BLOCK_BYTES;
        const auto [d, dmin] = q4k_header(block);
        const uint8_t* scales = block + 4;
        const uint8_t* qs = block + 16;
        size_t q_off = 0;
        size_t is = 0;
        for (size_t g = 0; g < QK_K / 64; ++g) {
            const auto [sc1, m1] = get_scale_min_k4(is, scales);
            const auto [sc2, m2] = get_scale_min_k4(is + 1, scales);
            const float d1 = d * static_cast<float>(sc1);
            const float m1v = dmin * static_cast<float>(m1);
            const float d2 = d * static_cast<float>(sc2);
            const float m2v = dmin * static_cast<float>(m2);

            // 32 packed bytes → 32 lo nibbles (sub-block 2g) + 32 hi nibbles (2g+1).
            const uint8x16_t q0 = vld1q_u8(qs + q_off);
            const uint8x16_t q1 = vld1q_u8(qs + q_off + 16);
            const int8x16_t lo0 = vreinterpretq_s8_u8(vandq_u8(q0, mask));
            const int8x16_t lo1 = vreinterpretq_s8_u8(vandq_u8(q1, mask));
            const int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q0, 4));
            const int8x16_t hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q1, 4));

            const int8x16_t xlo0 = vld1q_s8(x_i8.data() + x_off);
            const int8x16_t xlo1 = vld1q_s8(x_i8.data() + x_off + 16);
            const int8x16_t xhi0 = vld1q_s8(x_i8.data() + x_off + 32);
            const int8x16_t xhi1 = vld1q_s8(x_i8.data() + x_off + 48);

            const int32x4_t zero = vdupq_n_s32(0);
            const int32_t dot_lo = vaddvq_s32(sdot_s32(sdot_s32(zero, lo0, xlo0), lo1, xlo1));
            const int32_t dot_hi = vaddvq_s32(sdot_s32(sdot_s32(zero, hi0, xhi0), hi1, xhi1));

            const size_t blk_lo = x_off / QK;
            const size_t blk_hi = (x_off + 32) / QK;
            acc += x_scales[blk_lo] *
                   (d1 * static_cast<float>(dot_lo) - m1v * static_cast<float>(x_sums[blk_lo]));
            acc += x_scales[blk_hi] *
                   (d2 * static_cast<float>(dot_hi) - m2v * static_cast<float>(x_sums[blk_hi]));
            x_off += 64;
            q_off += 32;
            is += 2;
        }
    }
    return acc;
}

// Four Q4_K rows against ONE int8 activation vector — the multi-row GEMV core: the activation
// registers and sums are loaded once per 64-weight group and reused across the four rows; per-row
// arithmetic (values and order) is identical to dot_q4_k_row_q8_neon (quant.rs:1154-1225).
SAPIENT_TARGET_DOTPROD std::array<float, 4>
dot_q4_k_4rows_q8_neon(std::array<std::span<const uint8_t>, 4> rows,
                       std::span<const int8_t> x_i8,
                       std::span<const float> x_scales,
                       std::span<const int32_t> x_sums) {
    const size_t n_blocks = rows[0].size() / Q4_K_BLOCK_BYTES;
    for (const auto& r : rows)
        check_len(r.size(), n_blocks * Q4_K_BLOCK_BYTES, "dot_q4_k_4rows_q8_neon: row shorter than row 0");
    check_q8_row(n_blocks, x_i8, x_scales, QK_K / QK, "dot_q4_k_4rows_q8_neon: activations shorter than the row");
    check_len(x_sums.size(), n_blocks * (QK_K / QK), "dot_q4_k_4rows_q8_neon: x_sums shorter than the row");
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    std::array<float, 4> acc{};
    size_t x_off = 0;
    for (size_t bi = 0; bi < n_blocks; ++bi) {
        const size_t base = bi * Q4_K_BLOCK_BYTES;
        // Per-row super-block headers, hoisted once per block.
        float dv[4];
        float dminv[4];
        for (size_t r = 0; r < 4; ++r) {
            const auto [d, dmin] = q4k_header(rows[r].data() + base);
            dv[r] = d;
            dminv[r] = dmin;
        }
        size_t q_off = 0;
        size_t is = 0;
        for (size_t g = 0; g < QK_K / 64; ++g) {
            // Shared activation work: loaded ONCE for all 4 rows; sums precomputed.
            const int8x16_t xlo0 = vld1q_s8(x_i8.data() + x_off);
            const int8x16_t xlo1 = vld1q_s8(x_i8.data() + x_off + 16);
            const int8x16_t xhi0 = vld1q_s8(x_i8.data() + x_off + 32);
            const int8x16_t xhi1 = vld1q_s8(x_i8.data() + x_off + 48);
            const int32_t sum_lo = x_sums[x_off / QK];
            const int32_t sum_hi = x_sums[(x_off + 32) / QK];
            const float xs_lo = x_scales[x_off / QK];
            const float xs_hi = x_scales[(x_off + 32) / QK];

            for (size_t r = 0; r < 4; ++r) {
                const uint8_t* b = rows[r].data() + base;
                const uint8_t* scales = b + 4;
                const uint8_t* qs = b + 16;
                const auto [sc1, m1] = get_scale_min_k4(is, scales);
                const auto [sc2, m2] = get_scale_min_k4(is + 1, scales);
                const float d1 = dv[r] * static_cast<float>(sc1);
                const float m1v = dminv[r] * static_cast<float>(m1);
                const float d2 = dv[r] * static_cast<float>(sc2);
                const float m2v = dminv[r] * static_cast<float>(m2);

                const uint8x16_t q0 = vld1q_u8(qs + q_off);
                const uint8x16_t q1 = vld1q_u8(qs + q_off + 16);
                const int8x16_t lo0 = vreinterpretq_s8_u8(vandq_u8(q0, mask));
                const int8x16_t lo1 = vreinterpretq_s8_u8(vandq_u8(q1, mask));
                const int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q0, 4));
                const int8x16_t hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q1, 4));

                const int32x4_t zero = vdupq_n_s32(0);
                const int32_t dot_lo = vaddvq_s32(sdot_s32(sdot_s32(zero, lo0, xlo0), lo1, xlo1));
                const int32_t dot_hi = vaddvq_s32(sdot_s32(sdot_s32(zero, hi0, xhi0), hi1, xhi1));

                acc[r] += xs_lo * (d1 * static_cast<float>(dot_lo) - m1v * static_cast<float>(sum_lo));
                acc[r] += xs_hi * (d2 * static_cast<float>(dot_hi) - m2v * static_cast<float>(sum_hi));
            }
            x_off += 64;
            q_off += 32;
            is += 2;
        }
    }
    return acc;
}
#endif

std::vector<uint8_t> repack_q4_k_rows4(std::span<const uint8_t> blocks, size_t n, size_t k) {
    if (n % 4 != 0) panic("Q4_K_R4 repack: rows must be a multiple of 4");
    if (k % QK_K != 0) panic("Q4_K_R4 repack: k must be a multiple of 256");
    const size_t nb = k / QK_K;
    const size_t row_bytes = nb * Q4_K_BLOCK_BYTES;
    if (blocks.size() != n * row_bytes) panic("Q4_K_R4 repack: blocks.len() != n * row_bytes");
    std::vector<uint8_t> out(blocks.size(), 0);
    for (size_t g = 0; g < n / 4; ++g)
        for (size_t b = 0; b < nb; ++b)
            for (size_t r = 0; r < 4; ++r) {
                const size_t src = ((g * 4 + r) * nb + b) * Q4_K_BLOCK_BYTES;
                const size_t dst = (g * 4 * nb + b * 4 + r) * Q4_K_BLOCK_BYTES;
                std::copy_n(blocks.data() + src, Q4_K_BLOCK_BYTES, out.data() + dst);
            }
    return out;
}

#if SAPIENT_AARCH64
// Four Q4_K rows in the R4 layout: one contiguous stream `[r0.b, r1.b, r2.b, r3.b]` per block;
// per-row arithmetic identical to dot_q4_k_row_q8_neon (quant.rs:1259-1330).
SAPIENT_TARGET_DOTPROD std::array<float, 4>
dot_q4_k_4rows_r4_neon(std::span<const uint8_t> packed,
                       std::span<const int8_t> x_i8,
                       std::span<const float> x_scales,
                       std::span<const int32_t> x_sums) {
    const size_t nb = packed.size() / (4 * Q4_K_BLOCK_BYTES);
    check_q8_row(nb, x_i8, x_scales, QK_K / QK, "dot_q4_k_4rows_r4_neon: activations shorter than the row");
    check_len(x_sums.size(), nb * (QK_K / QK), "dot_q4_k_4rows_r4_neon: x_sums shorter than the row");
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    std::array<float, 4> acc{};
    size_t x_off = 0;
    for (size_t b = 0; b < nb; ++b) {
        const size_t gbase = b * 4 * Q4_K_BLOCK_BYTES;
        float dv[4];
        float dminv[4];
        for (size_t r = 0; r < 4; ++r) {
            const auto [d, dmin] = q4k_header(packed.data() + gbase + r * Q4_K_BLOCK_BYTES);
            dv[r] = d;
            dminv[r] = dmin;
        }
        size_t q_off = 0;
        size_t is = 0;
        for (size_t g = 0; g < QK_K / 64; ++g) {
            const int8x16_t xlo0 = vld1q_s8(x_i8.data() + x_off);
            const int8x16_t xlo1 = vld1q_s8(x_i8.data() + x_off + 16);
            const int8x16_t xhi0 = vld1q_s8(x_i8.data() + x_off + 32);
            const int8x16_t xhi1 = vld1q_s8(x_i8.data() + x_off + 48);
            const int32_t sum_lo = x_sums[x_off / QK];
            const int32_t sum_hi = x_sums[(x_off + 32) / QK];
            const float xs_lo = x_scales[x_off / QK];
            const float xs_hi = x_scales[(x_off + 32) / QK];

            for (size_t r = 0; r < 4; ++r) {
                const uint8_t* blk = packed.data() + gbase + r * Q4_K_BLOCK_BYTES;
                const uint8_t* scales = blk + 4;
                const uint8_t* qs = blk + 16;
                const auto [sc1, m1] = get_scale_min_k4(is, scales);
                const auto [sc2, m2] = get_scale_min_k4(is + 1, scales);
                const float d1 = dv[r] * static_cast<float>(sc1);
                const float m1v = dminv[r] * static_cast<float>(m1);
                const float d2 = dv[r] * static_cast<float>(sc2);
                const float m2v = dminv[r] * static_cast<float>(m2);

                const uint8x16_t q0 = vld1q_u8(qs + q_off);
                const uint8x16_t q1 = vld1q_u8(qs + q_off + 16);
                const int8x16_t lo0 = vreinterpretq_s8_u8(vandq_u8(q0, mask));
                const int8x16_t lo1 = vreinterpretq_s8_u8(vandq_u8(q1, mask));
                const int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q0, 4));
                const int8x16_t hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q1, 4));

                const int32x4_t zero = vdupq_n_s32(0);
                const int32_t dot_lo = vaddvq_s32(sdot_s32(sdot_s32(zero, lo0, xlo0), lo1, xlo1));
                const int32_t dot_hi = vaddvq_s32(sdot_s32(sdot_s32(zero, hi0, xhi0), hi1, xhi1));

                acc[r] += xs_lo * (d1 * static_cast<float>(dot_lo) - m1v * static_cast<float>(sum_lo));
                acc[r] += xs_hi * (d2 * static_cast<float>(dot_hi) - m2v * static_cast<float>(sum_hi));
            }
            x_off += 64;
            q_off += 32;
            is += 2;
        }
    }
    return acc;
}
#endif

// ── Q4_K × Q8_K, SMMLA (Task 3) ──────────────────────────────────────────────
```

- [ ] **Step 5: Build, run `--gtest_filter='Quant*'`** — expected: all pass (4 new Rust tests + the death test).
- [ ] **Step 6: The two codegen checks** (rule 5). Expected: both compile.
- [ ] **Step 7: Format after `git add`, `ctest --preset dev`, commit**

```bash
git commit -m "cpp(backends-cpu): kernels/quant — Q4_K f32/NEON, W4A8 scalar+SDOT, 4-row, R4 repack and kernel

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
Expected ctest: 169 + 5 = **174**.

---

### Task 3: Q4_K × Q8_K (integer-domain combine) and the SMMLA prefill kernels

**Files:**
- Modify: `quant.hpp` (fill the `Q4_K × Q8_K, SMMLA (Task 3)` section), `quant.cpp` (append), `tests/quant_test.cpp` (append)

**Interfaces:**
- Consumes: Task 1 (`Q8kRow`, `quantize_row_to_q8k`), Task 2 (`q4k_header`, `check_q8_row`, `sdot_s32`, `repack_q4_k_rows4`, `dot_q4_k_row_q8_neon`, `dot_q4_k_row_f32`, `q4_k_test_rows`, `ramp`, `row_of`).
- Produces: `dot_q4_k_row_q8k_scalar(row_data, x_i8, x_scales, x_sums)`; aarch64: `dot_q4_k_row_q8k_neon(...)` (dotprod), `dot_q4_k_4rows_q8k_neon(rows, ...) -> std::array<float,4>` (dotprod), `dot_q4_k_4rows_r4_q8k_neon(packed, ...) -> std::array<float,4>` (dotprod), `dot_q4_k_4rows_r4_x2_smmla(packed, x0_i8, x0_scales, x0_sums, x1_i8, x1_scales, x1_sums) -> std::array<std::array<float,2>,4>` (i8mm), `dot_q4_k_4rows_r4_x2_q8k_smmla(...)` (i8mm); private: `smmla_s32` (`SAPIENT_TARGET_I8MM inline`), `vtrn1q_s64_s8`/`vtrn2q_s64_s8` (plain NEON `inline`). Task 6 consumes the public ones.

- [ ] **Step 1: Append the failing tests**

```cpp
// ── Q4_K × Q8_K, SMMLA (Task 3) ───────────────────────────────────────────────

namespace {
// quant.rs q4_k_q8k_scalar_matches_f32_path / *_q8k_kernels_match_single_row: rows filled with
// `(i*A + B) % M` bytes, then every block's d/dmin overwritten with fixed f16 values.
std::vector<uint8_t> q4_k_mod_rows(size_t n, size_t k, size_t a, size_t b, size_t m, float d, float dmin) {
    const size_t row_bytes = k / QK_K * Q4_K_BLOCK_BYTES;
    std::vector<uint8_t> rows(n * row_bytes);
    for (size_t i = 0; i < rows.size(); ++i)
        rows[i] = static_cast<uint8_t>((i * a + b) % m);
    for (size_t r = 0; r < n; ++r)
        for (size_t blk = 0; blk < k / QK_K; ++blk) {
            uint8_t* base = rows.data() + r * row_bytes + blk * Q4_K_BLOCK_BYTES;
            sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(d), base);
            sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(dmin), base + 2);
        }
    return rows;
}
} // namespace

#if defined(__aarch64__) || defined(_M_ARM64)
TEST(Quant, q4_k_smmla_x2_matches_single_row) {
    if (!has_i8mm()) GTEST_SKIP() << "i8mm not available";
    const size_t n = 4, k = 512;
    const size_t row_bytes = k / 256 * Q4_K_BLOCK_BYTES;
    const auto rows = q4_k_test_rows(n, k, 0x18AAULL);
    const auto x0 = ramp(k, 37, 97, 48.0f, 0.02f);
    const auto x1 = ramp(k, 59, 101, 50.0f, 0.015f);
    const auto q0 = quantize_row_to_i8_blocks(x0);
    const auto q1 = quantize_row_to_i8_blocks(x1);
    const auto b0 = i8_block_sums(q0.q);
    const auto b1 = i8_block_sums(q1.q);
    const auto packed = repack_q4_k_rows4(rows, n, k);
    const auto got = dot_q4_k_4rows_r4_x2_smmla(packed, q0.q, q0.scales, b0, q1.q, q1.scales, b1);
    for (size_t r = 0; r < 4; ++r) {
        const auto row = row_of(rows, r, row_bytes);
        const float w0 = dot_q4_k_row_q8_neon(row, q0.q, q0.scales, b0);
        const float w1 = dot_q4_k_row_q8_neon(row, q1.q, q1.scales, b1);
        EXPECT_EQ(bits(got[r][0]), bits(w0)) << "row " << r << " x0: " << got[r][0] << " vs " << w0;
        EXPECT_EQ(bits(got[r][1]), bits(w1)) << "row " << r << " x1: " << got[r][1] << " vs " << w1;
    }
}
#endif

TEST(Quant, q4_k_q8k_scalar_matches_f32_path) {
    const size_t nblocks = 3;
    const auto row = q4_k_mod_rows(1, nblocks * QK_K, 197, 13, 251, 0.05f, 0.03f);
    uint64_t state = 0x2545F4914F6CDD1DULL;
    auto next = [&state]() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    };
    std::vector<float> x(nblocks * QK_K);
    for (float& v : x)
        v = (static_cast<float>(next()) / static_cast<float>(std::numeric_limits<uint64_t>::max())) *
                4.0f -
            2.0f;

    const float f32_dot = dot_q4_k_row_f32(row, x);
    const auto k8 = quantize_row_to_q8k(x);
    const float q8k_dot = dot_q4_k_row_q8k_scalar(row, k8.q, k8.scales, k8.sums);
    const float rel = ::fabsf(f32_dot - q8k_dot) / ::fmaxf(::fabsf(f32_dot), 1e-3f);
    EXPECT_LT(rel, 0.03f) << "Q8_K mismatch: f32=" << f32_dot << " q8k=" << q8k_dot;

    // The per-256 format must stay in the accuracy class of the accepted per-32 W4A8 path.
    const auto p = quantize_row_to_i8_blocks(x);
    const auto psum = i8_block_sums(p.q);
    const float w4a8 = dot_q4_k_row_q8_scalar(row, p.q, p.scales, psum);
    const float rel_vs = ::fabsf(w4a8 - q8k_dot) / ::fmaxf(::fabsf(w4a8), 1e-3f);
    EXPECT_LT(rel_vs, 0.03f) << "Q8_K vs W4A8 divergence: w4a8=" << w4a8 << " q8k=" << q8k_dot;

#if defined(__aarch64__) || defined(_M_ARM64)
    if (has_dotprod()) {
        const float neon = dot_q4_k_row_q8k_neon(row, k8.q, k8.scales, k8.sums);
        EXPECT_EQ(bits(neon), bits(q8k_dot)) << "NEON≠scalar Q8_K: " << neon << " vs " << q8k_dot;
    }
#endif
}

#if defined(__aarch64__) || defined(_M_ARM64)
TEST(Quant, q4_k_r4_q8k_kernels_match_single_row) {
    // Rust runs the dotprod kernels unconditionally here; the port skips on a non-dotprod host.
    if (!has_dotprod()) GTEST_SKIP() << "dotprod not available";
    const size_t n = 4, k = 512;
    const size_t row_bytes = k / QK_K * Q4_K_BLOCK_BYTES;
    const auto rows = q4_k_mod_rows(n, k, 149, 29, 249, 0.04f, 0.02f);
    const auto x0 = ramp(k, 37, 97, 48.0f, 0.02f);
    const auto x1 = ramp(k, 59, 101, 50.0f, 0.015f);
    const auto r0 = quantize_row_to_q8k(x0);
    const auto r1 = quantize_row_to_q8k(x1);
    const auto packed = repack_q4_k_rows4(rows, n, k);

    // 4-row R4 kernel vs single-row Q8_K kernel, exact bits.
    const auto got4 = dot_q4_k_4rows_r4_q8k_neon(packed, r0.q, r0.scales, r0.sums);
    for (size_t r = 0; r < 4; ++r) {
        const float want = dot_q4_k_row_q8k_neon(row_of(rows, r, row_bytes), r0.q, r0.scales, r0.sums);
        EXPECT_EQ(bits(got4[r]), bits(want)) << "r4 row " << r << ": " << got4[r] << " vs " << want;
    }
    // SMMLA x2 kernel vs single-row, exact bits over both x rows.
    if (has_i8mm()) {
        const auto got = dot_q4_k_4rows_r4_x2_q8k_smmla(packed, r0.q, r0.scales, r0.sums, r1.q,
                                                        r1.scales, r1.sums);
        for (size_t r = 0; r < 4; ++r) {
            const auto row = row_of(rows, r, row_bytes);
            const float w0 = dot_q4_k_row_q8k_neon(row, r0.q, r0.scales, r0.sums);
            const float w1 = dot_q4_k_row_q8k_neon(row, r1.q, r1.scales, r1.sums);
            EXPECT_EQ(bits(got[r][0]), bits(w0)) << "smmla row " << r << " x0";
            EXPECT_EQ(bits(got[r][1]), bits(w1)) << "smmla row " << r << " x1";
        }
    }
}

TEST(Quant, q4_k_plain_4rows_q8k_matches_single_row) {
    if (!has_dotprod()) GTEST_SKIP() << "dotprod not available"; // port-added skip (see Task 3)
    const size_t n = 4, k = 512;
    const size_t row_bytes = k / QK_K * Q4_K_BLOCK_BYTES;
    const auto rows = q4_k_mod_rows(n, k, 167, 43, 247, 0.04f, 0.02f);
    const auto x = ramp(k, 41, 103, 51.0f, 0.02f);
    const auto r = quantize_row_to_q8k(x);
    const std::array<std::span<const uint8_t>, 4> r4 = {row_of(rows, 0, row_bytes), row_of(rows, 1, row_bytes),
                                                        row_of(rows, 2, row_bytes), row_of(rows, 3, row_bytes)};
    const auto got = dot_q4_k_4rows_q8k_neon(r4, r.q, r.scales, r.sums);
    for (size_t o = 0; o < 4; ++o) {
        const float want = dot_q4_k_row_q8k_neon(r4[o], r.q, r.scales, r.sums);
        EXPECT_EQ(bits(got[o]), bits(want)) << "row " << o << ": " << got[o] << " vs " << want;
    }
}
#endif
```

- [ ] **Step 2: Build → FAIL.**

- [ ] **Step 3: Header declarations** (replace the `Q4_K × Q8_K, SMMLA (Task 3)` marker):

```cpp
// ── Q4_K × Q8_K activations (integer-domain sub-scale combine), SMMLA prefill ───
/// Scalar oracle: per super-block `isum = Σ sc·dot`, `imin = Σ mn·bsum` in i32, then ONE
/// `acc += x_scales[b] · (d·isum − dmin·imin)`. `x_scales` has one f32 per 256, `x_sums` one i32 per 32.
float dot_q4_k_row_q8k_scalar(std::span<const uint8_t> row_data,
                              std::span<const int8_t> x_i8,
                              std::span<const float> x_scales,
                              std::span<const int32_t> x_sums);
#if defined(__aarch64__) || defined(_M_ARM64)
/// `sdot` core + integer-domain combine; bit-identical to `dot_q4_k_row_q8k_scalar`. Precondition: dotprod.
float dot_q4_k_row_q8k_neon(std::span<const uint8_t> row_data,
                            std::span<const int8_t> x_i8,
                            std::span<const float> x_scales,
                            std::span<const int32_t> x_sums);
/// Four row-major rows × one Q8_K row; lanes bit-identical to `dot_q4_k_row_q8k_neon`. Iterates
/// `min(n_blocks, x_scales.size())` blocks (Rust `.take(n_blocks)`). Precondition: dotprod.
std::array<float, 4> dot_q4_k_4rows_q8k_neon(std::array<std::span<const uint8_t>, 4> rows,
                                             std::span<const int8_t> x_i8,
                                             std::span<const float> x_scales,
                                             std::span<const int32_t> x_sums);
/// Four R4 rows × one Q8_K row; same lane identity and `.take` rule. Precondition: dotprod.
std::array<float, 4> dot_q4_k_4rows_r4_q8k_neon(std::span<const uint8_t> packed,
                                                std::span<const int8_t> x_i8,
                                                std::span<const float> x_scales,
                                                std::span<const int32_t> x_sums);
/// Four R4 rows × TWO per-32 int8 activation rows via `smmla` — the prefill kernel. Returns
/// `[[row0·x0, row0·x1], …, [row3·x0, row3·x1]]`, every lane bit-identical to
/// `dot_q4_k_row_q8_neon`. Precondition: i8mm.
std::array<std::array<float, 2>, 4> dot_q4_k_4rows_r4_x2_smmla(std::span<const uint8_t> packed,
                                                                std::span<const int8_t> x0_i8,
                                                                std::span<const float> x0_scales,
                                                                std::span<const int32_t> x0_sums,
                                                                std::span<const int8_t> x1_i8,
                                                                std::span<const float> x1_scales,
                                                                std::span<const int32_t> x1_sums);
/// Four R4 rows × TWO Q8_K rows via `smmla`; lanes bit-identical to `dot_q4_k_row_q8k_neon`;
/// iterates `min(nb, x0_scales.size(), x1_scales.size())` blocks. Precondition: i8mm.
std::array<std::array<float, 2>, 4>
dot_q4_k_4rows_r4_x2_q8k_smmla(std::span<const uint8_t> packed,
                               std::span<const int8_t> x0_i8,
                               std::span<const float> x0_scales,
                               std::span<const int32_t> x0_sums,
                               std::span<const int8_t> x1_i8,
                               std::span<const float> x1_scales,
                               std::span<const int32_t> x1_sums);
#endif
```

- [ ] **Step 4: Definitions** (append to `quant.cpp`, replacing the Task 3 marker):

```cpp
// ── Q4_K × Q8_K, SMMLA (quant.rs:394-429, 829-1000, 1024-1082, 1335-1615) ───

float dot_q4_k_row_q8k_scalar(std::span<const uint8_t> row_data,
                              std::span<const int8_t> x_i8,
                              std::span<const float> x_scales,
                              std::span<const int32_t> x_sums) {
    const size_t nb = row_data.size() / Q4_K_BLOCK_BYTES;
    check_q8_row(nb, x_i8, x_scales, 1, "dot_q4_k_row_q8k_scalar: activations shorter than the row");
    check_len(x_sums.size(), nb * (QK_K / QK), "dot_q4_k_row_q8k_scalar: x_sums shorter than the row");
    float acc = 0.0f;
    size_t x_off = 0;
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* block = row_data.data() + b * Q4_K_BLOCK_BYTES;
        const auto [d, dmin] = q4k_header(block);
        const uint8_t* scales = block + 4;
        const uint8_t* qs = block + 16;
        size_t q_off = 0;
        size_t is = 0;
        int32_t isum = 0;
        int32_t imin = 0;
        for (size_t g = 0; g < QK_K / 64; ++g) {
            const auto [sc1, m1] = get_scale_min_k4(is, scales);
            const auto [sc2, m2] = get_scale_min_k4(is + 1, scales);
            const int8_t* xlo = x_i8.data() + x_off;
            const int8_t* xhi = x_i8.data() + x_off + 32;
            int32_t dot_lo = 0;
            int32_t dot_hi = 0;
            for (size_t l = 0; l < 32; ++l) {
                dot_lo += static_cast<int32_t>(qs[q_off + l] & 0x0F) * static_cast<int32_t>(xlo[l]);
                dot_hi += static_cast<int32_t>(qs[q_off + l] >> 4) * static_cast<int32_t>(xhi[l]);
            }
            isum += static_cast<int32_t>(sc1) * dot_lo + static_cast<int32_t>(sc2) * dot_hi;
            imin += static_cast<int32_t>(m1) * x_sums[x_off / QK] +
                    static_cast<int32_t>(m2) * x_sums[(x_off + 32) / QK];
            x_off += 64;
            q_off += 32;
            is += 2;
        }
        acc += x_scales[b] * (d * static_cast<float>(isum) - dmin * static_cast<float>(imin));
    }
    return acc;
}

#if SAPIENT_AARCH64
namespace {
// 2×2 int8 matrix-multiply-accumulate — Rust's `smmla_s32` inline asm: treats `a` and `b` as
// row-major 2×8 i8 matrices and accumulates a·bᵀ into the four lanes `[a0·b0, a0·b1, a1·b0, a1·b1]`.
SAPIENT_TARGET_I8MM inline int32x4_t smmla_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    return vmmlaq_s32(acc, a, b);
}
// TRN1 / TRN2 on the 64-bit halves of two i8 vectors: `[a.lo, b.lo]` / `[a.hi, b.hi]`.
inline int8x16_t vtrn1q_s64_s8(int8x16_t a, int8x16_t b) {
    return vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s8(a), vreinterpretq_s64_s8(b)));
}
inline int8x16_t vtrn2q_s64_s8(int8x16_t a, int8x16_t b) {
    return vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s8(a), vreinterpretq_s64_s8(b)));
}
} // namespace

SAPIENT_TARGET_DOTPROD float dot_q4_k_row_q8k_neon(std::span<const uint8_t> row_data,
                                                   std::span<const int8_t> x_i8,
                                                   std::span<const float> x_scales,
                                                   std::span<const int32_t> x_sums) {
    const size_t nb = row_data.size() / Q4_K_BLOCK_BYTES;
    check_q8_row(nb, x_i8, x_scales, 1, "dot_q4_k_row_q8k_neon: activations shorter than the row");
    check_len(x_sums.size(), nb * (QK_K / QK), "dot_q4_k_row_q8k_neon: x_sums shorter than the row");
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    float acc = 0.0f;
    size_t x_off = 0;
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* block = row_data.data() + b * Q4_K_BLOCK_BYTES;
        const auto [d, dmin] = q4k_header(block);
        const uint8_t* scales = block + 4;
        const uint8_t* qs = block + 16;
        size_t q_off = 0;
        size_t is = 0;
        int32_t isum = 0;
        int32_t imin = 0;
        for (size_t g = 0; g < QK_K / 64; ++g) {
            const auto [sc1, m1] = get_scale_min_k4(is, scales);
            const auto [sc2, m2] = get_scale_min_k4(is + 1, scales);
            const uint8x16_t q0 = vld1q_u8(qs + q_off);
            const uint8x16_t q1 = vld1q_u8(qs + q_off + 16);
            const int8x16_t lo0 = vreinterpretq_s8_u8(vandq_u8(q0, mask));
            const int8x16_t lo1 = vreinterpretq_s8_u8(vandq_u8(q1, mask));
            const int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q0, 4));
            const int8x16_t hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q1, 4));
            const int8x16_t xlo0 = vld1q_s8(x_i8.data() + x_off);
            const int8x16_t xlo1 = vld1q_s8(x_i8.data() + x_off + 16);
            const int8x16_t xhi0 = vld1q_s8(x_i8.data() + x_off + 32);
            const int8x16_t xhi1 = vld1q_s8(x_i8.data() + x_off + 48);
            const int32x4_t zero = vdupq_n_s32(0);
            const int32_t dot_lo = vaddvq_s32(sdot_s32(sdot_s32(zero, lo0, xlo0), lo1, xlo1));
            const int32_t dot_hi = vaddvq_s32(sdot_s32(sdot_s32(zero, hi0, xhi0), hi1, xhi1));
            isum += static_cast<int32_t>(sc1) * dot_lo + static_cast<int32_t>(sc2) * dot_hi;
            imin += static_cast<int32_t>(m1) * x_sums[x_off / QK] +
                    static_cast<int32_t>(m2) * x_sums[(x_off + 32) / QK];
            x_off += 64;
            q_off += 32;
            is += 2;
        }
        acc += x_scales[b] * (d * static_cast<float>(isum) - dmin * static_cast<float>(imin));
    }
    return acc;
}

SAPIENT_TARGET_DOTPROD std::array<float, 4>
dot_q4_k_4rows_q8k_neon(std::array<std::span<const uint8_t>, 4> rows,
                        std::span<const int8_t> x_i8,
                        std::span<const float> x_scales,
                        std::span<const int32_t> x_sums) {
    const size_t n_blocks = rows[0].size() / Q4_K_BLOCK_BYTES;
    for (const auto& r : rows)
        check_len(r.size(), n_blocks * Q4_K_BLOCK_BYTES, "dot_q4_k_4rows_q8k_neon: row shorter than row 0");
    const size_t nb_eff = std::min(n_blocks, x_scales.size()); // .take(n_blocks) over x_scales
    check_len(x_i8.size(), nb_eff * QK_K, "dot_q4_k_4rows_q8k_neon: x_i8 shorter than the row");
    check_len(x_sums.size(), nb_eff * (QK_K / QK), "dot_q4_k_4rows_q8k_neon: x_sums shorter than the row");
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    std::array<float, 4> acc{};
    size_t x_off = 0;
    for (size_t bi = 0; bi < nb_eff; ++bi) {
        const float db = x_scales[bi];
        const size_t base = bi * Q4_K_BLOCK_BYTES;
        float dv[4];
        float dminv[4];
        for (size_t r = 0; r < 4; ++r) {
            const auto [d, dmin] = q4k_header(rows[r].data() + base);
            dv[r] = d;
            dminv[r] = dmin;
        }
        size_t q_off = 0;
        size_t is = 0;
        int32_t isum[4] = {0, 0, 0, 0};
        int32_t imin[4] = {0, 0, 0, 0};
        for (size_t g = 0; g < QK_K / 64; ++g) {
            const int8x16_t xlo0 = vld1q_s8(x_i8.data() + x_off);
            const int8x16_t xlo1 = vld1q_s8(x_i8.data() + x_off + 16);
            const int8x16_t xhi0 = vld1q_s8(x_i8.data() + x_off + 32);
            const int8x16_t xhi1 = vld1q_s8(x_i8.data() + x_off + 48);
            const int32_t sum_lo = x_sums[x_off / QK];
            const int32_t sum_hi = x_sums[(x_off + 32) / QK];
            for (size_t r = 0; r < 4; ++r) {
                const uint8_t* b = rows[r].data() + base;
                const uint8_t* scales = b + 4;
                const uint8_t* qs = b + 16;
                const auto [sc1, m1] = get_scale_min_k4(is, scales);
                const auto [sc2, m2] = get_scale_min_k4(is + 1, scales);
                const uint8x16_t q0 = vld1q_u8(qs + q_off);
                const uint8x16_t q1 = vld1q_u8(qs + q_off + 16);
                const int8x16_t lo0 = vreinterpretq_s8_u8(vandq_u8(q0, mask));
                const int8x16_t lo1 = vreinterpretq_s8_u8(vandq_u8(q1, mask));
                const int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q0, 4));
                const int8x16_t hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q1, 4));
                const int32x4_t zero = vdupq_n_s32(0);
                const int32_t dot_lo = vaddvq_s32(sdot_s32(sdot_s32(zero, lo0, xlo0), lo1, xlo1));
                const int32_t dot_hi = vaddvq_s32(sdot_s32(sdot_s32(zero, hi0, xhi0), hi1, xhi1));
                isum[r] += static_cast<int32_t>(sc1) * dot_lo + static_cast<int32_t>(sc2) * dot_hi;
                imin[r] += static_cast<int32_t>(m1) * sum_lo + static_cast<int32_t>(m2) * sum_hi;
            }
            x_off += 64;
            q_off += 32;
            is += 2;
        }
        for (size_t r = 0; r < 4; ++r)
            acc[r] += db * (dv[r] * static_cast<float>(isum[r]) - dminv[r] * static_cast<float>(imin[r]));
    }
    return acc;
}

SAPIENT_TARGET_DOTPROD std::array<float, 4>
dot_q4_k_4rows_r4_q8k_neon(std::span<const uint8_t> packed,
                           std::span<const int8_t> x_i8,
                           std::span<const float> x_scales,
                           std::span<const int32_t> x_sums) {
    const size_t nb = packed.size() / (4 * Q4_K_BLOCK_BYTES);
    const size_t nb_eff = std::min(nb, x_scales.size()); // .take(nb)
    check_len(x_i8.size(), nb_eff * QK_K, "dot_q4_k_4rows_r4_q8k_neon: x_i8 shorter than the row");
    check_len(x_sums.size(), nb_eff * (QK_K / QK), "dot_q4_k_4rows_r4_q8k_neon: x_sums shorter than the row");
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    std::array<float, 4> acc{};
    size_t x_off = 0;
    for (size_t b = 0; b < nb_eff; ++b) {
        const float db = x_scales[b];
        const size_t gbase = b * 4 * Q4_K_BLOCK_BYTES;
        float dv[4];
        float dminv[4];
        for (size_t r = 0; r < 4; ++r) {
            const auto [d, dmin] = q4k_header(packed.data() + gbase + r * Q4_K_BLOCK_BYTES);
            dv[r] = d;
            dminv[r] = dmin;
        }
        size_t q_off = 0;
        size_t is = 0;
        int32_t isum[4] = {0, 0, 0, 0};
        int32_t imin[4] = {0, 0, 0, 0};
        for (size_t g = 0; g < QK_K / 64; ++g) {
            const int8x16_t xlo0 = vld1q_s8(x_i8.data() + x_off);
            const int8x16_t xlo1 = vld1q_s8(x_i8.data() + x_off + 16);
            const int8x16_t xhi0 = vld1q_s8(x_i8.data() + x_off + 32);
            const int8x16_t xhi1 = vld1q_s8(x_i8.data() + x_off + 48);
            const int32_t sum_lo = x_sums[x_off / QK];
            const int32_t sum_hi = x_sums[(x_off + 32) / QK];
            for (size_t r = 0; r < 4; ++r) {
                const uint8_t* blk = packed.data() + gbase + r * Q4_K_BLOCK_BYTES;
                const uint8_t* scales = blk + 4;
                const uint8_t* qs = blk + 16;
                const auto [sc1, m1] = get_scale_min_k4(is, scales);
                const auto [sc2, m2] = get_scale_min_k4(is + 1, scales);
                const uint8x16_t q0 = vld1q_u8(qs + q_off);
                const uint8x16_t q1 = vld1q_u8(qs + q_off + 16);
                const int8x16_t lo0 = vreinterpretq_s8_u8(vandq_u8(q0, mask));
                const int8x16_t lo1 = vreinterpretq_s8_u8(vandq_u8(q1, mask));
                const int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q0, 4));
                const int8x16_t hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q1, 4));
                const int32x4_t zero = vdupq_n_s32(0);
                const int32_t dot_lo = vaddvq_s32(sdot_s32(sdot_s32(zero, lo0, xlo0), lo1, xlo1));
                const int32_t dot_hi = vaddvq_s32(sdot_s32(sdot_s32(zero, hi0, xhi0), hi1, xhi1));
                isum[r] += static_cast<int32_t>(sc1) * dot_lo + static_cast<int32_t>(sc2) * dot_hi;
                imin[r] += static_cast<int32_t>(m1) * sum_lo + static_cast<int32_t>(m2) * sum_hi;
            }
            x_off += 64;
            q_off += 32;
            is += 2;
        }
        for (size_t r = 0; r < 4; ++r)
            acc[r] += db * (dv[r] * static_cast<float>(isum[r]) - dminv[r] * static_cast<float>(imin[r]));
    }
    return acc;
}

// Four Q4_K rows (R4) × TWO per-32 int8 activation rows via `smmla` (quant.rs:861-990). Each
// 16-weight segment-pair costs two `trn` shuffles + one `smmla` per weight-row pair; the dots come
// out in lane order [r0·x0, r0·x1, r1·x0, r1·x1] and the f32 combine is dot_q4_k_row_q8_neon's.
SAPIENT_TARGET_I8MM std::array<std::array<float, 2>, 4>
dot_q4_k_4rows_r4_x2_smmla(std::span<const uint8_t> packed,
                           std::span<const int8_t> x0_i8,
                           std::span<const float> x0_scales,
                           std::span<const int32_t> x0_sums,
                           std::span<const int8_t> x1_i8,
                           std::span<const float> x1_scales,
                           std::span<const int32_t> x1_sums) {
    const size_t nb = packed.size() / (4 * Q4_K_BLOCK_BYTES);
    check_q8_row(nb, x0_i8, x0_scales, QK_K / QK, "dot_q4_k_4rows_r4_x2_smmla: x0 shorter than the row");
    check_q8_row(nb, x1_i8, x1_scales, QK_K / QK, "dot_q4_k_4rows_r4_x2_smmla: x1 shorter than the row");
    check_len(x0_sums.size(), nb * (QK_K / QK), "dot_q4_k_4rows_r4_x2_smmla: x0_sums shorter than the row");
    check_len(x1_sums.size(), nb * (QK_K / QK), "dot_q4_k_4rows_r4_x2_smmla: x1_sums shorter than the row");
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    std::array<std::array<float, 2>, 4> acc{};
    size_t x_off = 0;
    for (size_t b = 0; b < nb; ++b) {
        const size_t gbase = b * 4 * Q4_K_BLOCK_BYTES;
        float dv[4];
        float dminv[4];
        for (size_t r = 0; r < 4; ++r) {
            const auto [d, dmin] = q4k_header(packed.data() + gbase + r * Q4_K_BLOCK_BYTES);
            dv[r] = d;
            dminv[r] = dmin;
        }
        size_t q_off = 0;
        size_t is = 0;
        for (size_t g = 0; g < QK_K / 64; ++g) {
            // Activation vectors for BOTH rows, once; per-sub-block sums precomputed.
            const int8x16_t x0lo0 = vld1q_s8(x0_i8.data() + x_off);
            const int8x16_t x0lo1 = vld1q_s8(x0_i8.data() + x_off + 16);
            const int8x16_t x0hi0 = vld1q_s8(x0_i8.data() + x_off + 32);
            const int8x16_t x0hi1 = vld1q_s8(x0_i8.data() + x_off + 48);
            const int8x16_t x1lo0 = vld1q_s8(x1_i8.data() + x_off);
            const int8x16_t x1lo1 = vld1q_s8(x1_i8.data() + x_off + 16);
            const int8x16_t x1hi0 = vld1q_s8(x1_i8.data() + x_off + 32);
            const int8x16_t x1hi1 = vld1q_s8(x1_i8.data() + x_off + 48);
            const int32_t sum_lo[2] = {x0_sums[x_off / QK], x1_sums[x_off / QK]};
            const int32_t sum_hi[2] = {x0_sums[(x_off + 32) / QK], x1_sums[(x_off + 32) / QK]};
            const float xs_lo[2] = {x0_scales[x_off / QK], x1_scales[x_off / QK]};
            const float xs_hi[2] = {x0_scales[(x_off + 32) / QK], x1_scales[(x_off + 32) / QK]};
            // Pair the two activation rows per 8-byte k-segment: [x0_seg, x1_seg].
            const int8x16_t xlo_a = vtrn1q_s64_s8(x0lo0, x1lo0);
            const int8x16_t xlo_b = vtrn2q_s64_s8(x0lo0, x1lo0);
            const int8x16_t xlo_c = vtrn1q_s64_s8(x0lo1, x1lo1);
            const int8x16_t xlo_d = vtrn2q_s64_s8(x0lo1, x1lo1);
            const int8x16_t xhi_a = vtrn1q_s64_s8(x0hi0, x1hi0);
            const int8x16_t xhi_b = vtrn2q_s64_s8(x0hi0, x1hi0);
            const int8x16_t xhi_c = vtrn1q_s64_s8(x0hi1, x1hi1);
            const int8x16_t xhi_d = vtrn2q_s64_s8(x0hi1, x1hi1);

            for (size_t pair = 0; pair < 2; ++pair) {
                const size_t r0 = pair * 2;
                const size_t r1 = pair * 2 + 1;
                const uint8_t* qs0 = packed.data() + gbase + r0 * Q4_K_BLOCK_BYTES + 16;
                const uint8_t* qs1 = packed.data() + gbase + r1 * Q4_K_BLOCK_BYTES + 16;
                const uint8x16_t q0a = vld1q_u8(qs0 + q_off);
                const uint8x16_t q0b = vld1q_u8(qs0 + q_off + 16);
                const uint8x16_t q1a = vld1q_u8(qs1 + q_off);
                const uint8x16_t q1b = vld1q_u8(qs1 + q_off + 16);
                const int8x16_t lo0a = vreinterpretq_s8_u8(vandq_u8(q0a, mask));
                const int8x16_t lo0b = vreinterpretq_s8_u8(vandq_u8(q0b, mask));
                const int8x16_t lo1a = vreinterpretq_s8_u8(vandq_u8(q1a, mask));
                const int8x16_t lo1b = vreinterpretq_s8_u8(vandq_u8(q1b, mask));
                const int8x16_t hi0a = vreinterpretq_s8_u8(vshrq_n_u8(q0a, 4));
                const int8x16_t hi0b = vreinterpretq_s8_u8(vshrq_n_u8(q0b, 4));
                const int8x16_t hi1a = vreinterpretq_s8_u8(vshrq_n_u8(q1a, 4));
                const int8x16_t hi1b = vreinterpretq_s8_u8(vshrq_n_u8(q1b, 4));
                // Weight-row pairs per 8-byte k-segment: [w_r0_seg, w_r1_seg].
                const int8x16_t wlo_a = vtrn1q_s64_s8(lo0a, lo1a);
                const int8x16_t wlo_b = vtrn2q_s64_s8(lo0a, lo1a);
                const int8x16_t wlo_c = vtrn1q_s64_s8(lo0b, lo1b);
                const int8x16_t wlo_d = vtrn2q_s64_s8(lo0b, lo1b);
                const int8x16_t whi_a = vtrn1q_s64_s8(hi0a, hi1a);
                const int8x16_t whi_b = vtrn2q_s64_s8(hi0a, hi1a);
                const int8x16_t whi_c = vtrn1q_s64_s8(hi0b, hi1b);
                const int8x16_t whi_d = vtrn2q_s64_s8(hi0b, hi1b);

                const int32x4_t zero = vdupq_n_s32(0);
                int32x4_t dlo = smmla_s32(zero, wlo_a, xlo_a);
                dlo = smmla_s32(dlo, wlo_b, xlo_b);
                dlo = smmla_s32(dlo, wlo_c, xlo_c);
                dlo = smmla_s32(dlo, wlo_d, xlo_d);
                int32x4_t dhi = smmla_s32(zero, whi_a, xhi_a);
                dhi = smmla_s32(dhi, whi_b, xhi_b);
                dhi = smmla_s32(dhi, whi_c, xhi_c);
                dhi = smmla_s32(dhi, whi_d, xhi_d);
                const int32_t dlo_arr[4] = {vgetq_lane_s32(dlo, 0), vgetq_lane_s32(dlo, 1),
                                            vgetq_lane_s32(dlo, 2), vgetq_lane_s32(dlo, 3)};
                const int32_t dhi_arr[4] = {vgetq_lane_s32(dhi, 0), vgetq_lane_s32(dhi, 1),
                                            vgetq_lane_s32(dhi, 2), vgetq_lane_s32(dhi, 3)};

                const size_t rows2[2] = {r0, r1};
                for (size_t ri = 0; ri < 2; ++ri) {
                    const size_t row = rows2[ri];
                    const uint8_t* scales = packed.data() + gbase + row * Q4_K_BLOCK_BYTES + 4;
                    const auto [sc1, m1] = get_scale_min_k4(is, scales);
                    const auto [sc2, m2] = get_scale_min_k4(is + 1, scales);
                    const float d1 = dv[row] * static_cast<float>(sc1);
                    const float m1v = dminv[row] * static_cast<float>(m1);
                    const float d2 = dv[row] * static_cast<float>(sc2);
                    const float m2v = dminv[row] * static_cast<float>(m2);
                    for (size_t xr = 0; xr < 2; ++xr) {
                        const int32_t dot_lo = dlo_arr[ri * 2 + xr];
                        const int32_t dot_hi = dhi_arr[ri * 2 + xr];
                        acc[row][xr] += xs_lo[xr] * (d1 * static_cast<float>(dot_lo) -
                                                     m1v * static_cast<float>(sum_lo[xr]));
                        acc[row][xr] += xs_hi[xr] * (d2 * static_cast<float>(dot_hi) -
                                                     m2v * static_cast<float>(sum_hi[xr]));
                    }
                }
            }
            x_off += 64;
            q_off += 32;
            is += 2;
        }
    }
    return acc;
}

// Same trn/smmla core with the integer-domain combine of dot_q4_k_row_q8k_neon (quant.rs:1487-1615).
SAPIENT_TARGET_I8MM std::array<std::array<float, 2>, 4>
dot_q4_k_4rows_r4_x2_q8k_smmla(std::span<const uint8_t> packed,
                               std::span<const int8_t> x0_i8,
                               std::span<const float> x0_scales,
                               std::span<const int32_t> x0_sums,
                               std::span<const int8_t> x1_i8,
                               std::span<const float> x1_scales,
                               std::span<const int32_t> x1_sums) {
    const size_t nb = packed.size() / (4 * Q4_K_BLOCK_BYTES);
    const size_t nb_eff = std::min({nb, x0_scales.size(), x1_scales.size()}); // zip().take(nb)
    check_len(x0_i8.size(), nb_eff * QK_K, "dot_q4_k_4rows_r4_x2_q8k_smmla: x0 shorter than the row");
    check_len(x1_i8.size(), nb_eff * QK_K, "dot_q4_k_4rows_r4_x2_q8k_smmla: x1 shorter than the row");
    check_len(x0_sums.size(), nb_eff * (QK_K / QK), "dot_q4_k_4rows_r4_x2_q8k_smmla: x0_sums shorter than the row");
    check_len(x1_sums.size(), nb_eff * (QK_K / QK), "dot_q4_k_4rows_r4_x2_q8k_smmla: x1_sums shorter than the row");
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    std::array<std::array<float, 2>, 4> acc{};
    size_t x_off = 0;
    for (size_t b = 0; b < nb_eff; ++b) {
        const float db[2] = {x0_scales[b], x1_scales[b]};
        const size_t gbase = b * 4 * Q4_K_BLOCK_BYTES;
        float dv[4];
        float dminv[4];
        for (size_t r = 0; r < 4; ++r) {
            const auto [d, dmin] = q4k_header(packed.data() + gbase + r * Q4_K_BLOCK_BYTES);
            dv[r] = d;
            dminv[r] = dmin;
        }
        size_t q_off = 0;
        size_t is = 0;
        int32_t isum[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
        int32_t imin[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
        for (size_t g = 0; g < QK_K / 64; ++g) {
            const int8x16_t x0lo0 = vld1q_s8(x0_i8.data() + x_off);
            const int8x16_t x0lo1 = vld1q_s8(x0_i8.data() + x_off + 16);
            const int8x16_t x0hi0 = vld1q_s8(x0_i8.data() + x_off + 32);
            const int8x16_t x0hi1 = vld1q_s8(x0_i8.data() + x_off + 48);
            const int8x16_t x1lo0 = vld1q_s8(x1_i8.data() + x_off);
            const int8x16_t x1lo1 = vld1q_s8(x1_i8.data() + x_off + 16);
            const int8x16_t x1hi0 = vld1q_s8(x1_i8.data() + x_off + 32);
            const int8x16_t x1hi1 = vld1q_s8(x1_i8.data() + x_off + 48);
            const int32_t sum_lo[2] = {x0_sums[x_off / QK], x1_sums[x_off / QK]};
            const int32_t sum_hi[2] = {x0_sums[(x_off + 32) / QK], x1_sums[(x_off + 32) / QK]};
            const int8x16_t xlo_a = vtrn1q_s64_s8(x0lo0, x1lo0);
            const int8x16_t xlo_b = vtrn2q_s64_s8(x0lo0, x1lo0);
            const int8x16_t xlo_c = vtrn1q_s64_s8(x0lo1, x1lo1);
            const int8x16_t xlo_d = vtrn2q_s64_s8(x0lo1, x1lo1);
            const int8x16_t xhi_a = vtrn1q_s64_s8(x0hi0, x1hi0);
            const int8x16_t xhi_b = vtrn2q_s64_s8(x0hi0, x1hi0);
            const int8x16_t xhi_c = vtrn1q_s64_s8(x0hi1, x1hi1);
            const int8x16_t xhi_d = vtrn2q_s64_s8(x0hi1, x1hi1);

            for (size_t pair = 0; pair < 2; ++pair) {
                const size_t r0 = pair * 2;
                const size_t r1 = pair * 2 + 1;
                const uint8_t* qs0 = packed.data() + gbase + r0 * Q4_K_BLOCK_BYTES + 16;
                const uint8_t* qs1 = packed.data() + gbase + r1 * Q4_K_BLOCK_BYTES + 16;
                const uint8x16_t q0a = vld1q_u8(qs0 + q_off);
                const uint8x16_t q0b = vld1q_u8(qs0 + q_off + 16);
                const uint8x16_t q1a = vld1q_u8(qs1 + q_off);
                const uint8x16_t q1b = vld1q_u8(qs1 + q_off + 16);
                const int8x16_t lo0a = vreinterpretq_s8_u8(vandq_u8(q0a, mask));
                const int8x16_t lo0b = vreinterpretq_s8_u8(vandq_u8(q0b, mask));
                const int8x16_t lo1a = vreinterpretq_s8_u8(vandq_u8(q1a, mask));
                const int8x16_t lo1b = vreinterpretq_s8_u8(vandq_u8(q1b, mask));
                const int8x16_t hi0a = vreinterpretq_s8_u8(vshrq_n_u8(q0a, 4));
                const int8x16_t hi0b = vreinterpretq_s8_u8(vshrq_n_u8(q0b, 4));
                const int8x16_t hi1a = vreinterpretq_s8_u8(vshrq_n_u8(q1a, 4));
                const int8x16_t hi1b = vreinterpretq_s8_u8(vshrq_n_u8(q1b, 4));
                const int8x16_t wlo_a = vtrn1q_s64_s8(lo0a, lo1a);
                const int8x16_t wlo_b = vtrn2q_s64_s8(lo0a, lo1a);
                const int8x16_t wlo_c = vtrn1q_s64_s8(lo0b, lo1b);
                const int8x16_t wlo_d = vtrn2q_s64_s8(lo0b, lo1b);
                const int8x16_t whi_a = vtrn1q_s64_s8(hi0a, hi1a);
                const int8x16_t whi_b = vtrn2q_s64_s8(hi0a, hi1a);
                const int8x16_t whi_c = vtrn1q_s64_s8(hi0b, hi1b);
                const int8x16_t whi_d = vtrn2q_s64_s8(hi0b, hi1b);

                const int32x4_t zero = vdupq_n_s32(0);
                int32x4_t dlo = smmla_s32(zero, wlo_a, xlo_a);
                dlo = smmla_s32(dlo, wlo_b, xlo_b);
                dlo = smmla_s32(dlo, wlo_c, xlo_c);
                dlo = smmla_s32(dlo, wlo_d, xlo_d);
                int32x4_t dhi = smmla_s32(zero, whi_a, xhi_a);
                dhi = smmla_s32(dhi, whi_b, xhi_b);
                dhi = smmla_s32(dhi, whi_c, xhi_c);
                dhi = smmla_s32(dhi, whi_d, xhi_d);
                const int32_t dlo_arr[4] = {vgetq_lane_s32(dlo, 0), vgetq_lane_s32(dlo, 1),
                                            vgetq_lane_s32(dlo, 2), vgetq_lane_s32(dlo, 3)};
                const int32_t dhi_arr[4] = {vgetq_lane_s32(dhi, 0), vgetq_lane_s32(dhi, 1),
                                            vgetq_lane_s32(dhi, 2), vgetq_lane_s32(dhi, 3)};

                const size_t rows2[2] = {r0, r1};
                for (size_t ri = 0; ri < 2; ++ri) {
                    const size_t row = rows2[ri];
                    const uint8_t* scales = packed.data() + gbase + row * Q4_K_BLOCK_BYTES + 4;
                    const auto [sc1, m1] = get_scale_min_k4(is, scales);
                    const auto [sc2, m2] = get_scale_min_k4(is + 1, scales);
                    for (size_t xr = 0; xr < 2; ++xr) {
                        const int32_t dot_lo = dlo_arr[ri * 2 + xr];
                        const int32_t dot_hi = dhi_arr[ri * 2 + xr];
                        isum[row][xr] += static_cast<int32_t>(sc1) * dot_lo + static_cast<int32_t>(sc2) * dot_hi;
                        imin[row][xr] += static_cast<int32_t>(m1) * sum_lo[xr] + static_cast<int32_t>(m2) * sum_hi[xr];
                    }
                }
            }
            x_off += 64;
            q_off += 32;
            is += 2;
        }
        for (size_t r = 0; r < 4; ++r)
            for (size_t xr = 0; xr < 2; ++xr)
                acc[r][xr] += db[xr] * (dv[r] * static_cast<float>(isum[r][xr]) -
                                        dminv[r] * static_cast<float>(imin[r][xr]));
    }
    return acc;
}
#endif

// ── Q5_K, Q6_K f32, Q6_K repack/R4 f32 (Task 4) ──────────────────────────────
```

- [ ] **Step 5: Build, `--gtest_filter='Quant*'`** — all pass.
- [ ] **Step 6: The two codegen checks (rule 5).** The `-mcpu=cortex-a53` build is where a missing `SAPIENT_TARGET_I8MM` or a lambda around `vmmlaq_s32` shows up.
- [ ] **Step 7: Format after `git add`, `ctest --preset dev`, commit**

```bash
git commit -m "cpp(backends-cpu): kernels/quant — Q4_K Q8_K integer-domain kernels and SMMLA x2 prefill kernels

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
Expected ctest: 174 + 4 = **178**.

---

### Task 4: Q5_K and Q6_K f32 dots (scalar + NEON), `repack_q6_k_rows4`, the R4 f32 kernel, the scale-indexing and corruption-magnitude tests

**Files:**
- Modify: `quant.hpp` (fill the `Q5_K, Q6_K f32, Q6_K repack/R4 f32 (Task 4)` section), `quant.cpp` (append), `tests/quant_test.cpp` (append)

**Interfaces:**
- Consumes: Task 1 (`detail::i8v`, `check_len`, `dot_q8_0_row_f32`, `dot_q8_0_row_sdot`, `quantize_row_to_i8_blocks`, `quantize_q8_0_block`, test helpers `lcg_bytes`, `rand_x`, `q8_0_weight_row`), Task 2 (`q4k_header`, `get_scale_min_k4`, `row_of`), `sapient::core::Tensor` (test only).
- Produces: `dot_q5_k_row_f32(row_data, x)`, `dot_q6_k_row_f32(row_data, x)`, `repack_q6_k_rows4(blocks, n, k)`; aarch64 (plain NEON): `dot_q6_k_4rows_r4_neon(packed, std::span<const float> x) -> std::array<float,4>`; `detail::{dot_q5_k_row_f32_scalar, dot_q6_k_row_f32_scalar}`, aarch64 `detail::{dot_q5_k_row_f32_neon, dot_q6_k_row_f32_neon}`; private: `q6_unpack(const uint8_t* ql, const uint8_t* qh, size_t ql_off, size_t qh_off, size_t l0, uint8x16_t q[4])` (plain NEON, the four 6-bit sub-positions of a 16-lane group) and `q6_scale(const uint8_t* sc, size_t i) -> float` (`sc[i] as i8 as f32`). Task 5 and Task 6 consume these.

- [ ] **Step 1: Append the failing tests**

```cpp
// ── Q5_K, Q6_K f32 (Task 4) ───────────────────────────────────────────────────

namespace {
// Q6_K must map weight i to scale i/16 (16 scales per 256-weight super-block), matching ggml
// dequantize_row_q6_K. Every 6-bit quant decodes to +1 (raw 33 = low nibble 1 | hi bits 2 << 4)
// and scales = 0..16, so with x = 1 and d = 1 the dot is Σ_i scale[i/16] = 16·(0+…+15) = 1920.
std::vector<uint8_t> canonical_q6_k_block() {
    std::vector<uint8_t> block(Q6_K_BLOCK_BYTES, 0);
    for (size_t i = 0; i < 128; ++i)
        block[i] = 0x11; // every low nibble = 1
    for (size_t i = 128; i < 192; ++i)
        block[i] = 0xAA; // every 2-bit hi field = 0b10 = 2
    for (size_t j = 0; j < 16; ++j)
        block[192 + j] = static_cast<uint8_t>(j); // scales 0..15
    sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(1.0f), block.data() + 208);
    return block;
}
std::vector<uint8_t> rand_q6_k_block(uint64_t seed) {
    auto blk = lcg_bytes(seed, Q6_K_BLOCK_BYTES);
    sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(0.04f), blk.data() + 208);
    return blk;
}
std::vector<uint8_t> rand_q5_k_block(uint64_t seed) {
    auto blk = lcg_bytes(seed, Q5_K_BLOCK_BYTES);
    sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(0.05f), blk.data());
    sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(0.02f), blk.data() + 2);
    return blk;
}
float q6_scale_of(const uint8_t* sc, size_t i) {
    return static_cast<float>(detail::i8v(sc[i]));
}
// Buggy Q6_K dot: one scale per 32-element sub-group (the shipped bug — sc[ib..ib+4], ib += 4 per
// 128-block), which only ever touches scales 0..7.
float dot_q6_k_buggy(std::span<const uint8_t> row_data, std::span<const float> x) {
    float acc = 0.0f;
    size_t x_off = 0;
    const size_t nb = row_data.size() / Q6_K_BLOCK_BYTES;
    for (size_t bi = 0; bi < nb; ++bi) {
        const uint8_t* block = row_data.data() + bi * Q6_K_BLOCK_BYTES;
        const uint8_t* ql = block;
        const uint8_t* qh = block + 128;
        const uint8_t* sc = block + 192;
        const float d = sapient::core::f16_le_to_f32(block + 208);
        size_t ql_off = 0, qh_off = 0, ib = 0;
        for (size_t half = 0; half < QK_K / 128; ++half) {
            for (size_t l = 0; l < 32; ++l) {
                const float q1 = static_cast<float>(static_cast<int32_t>((ql[ql_off + l] & 0x0F) | ((qh[qh_off + l] & 3) << 4)) - 32);
                const float q2 = static_cast<float>(static_cast<int32_t>((ql[ql_off + l + 32] & 0x0F) | (((qh[qh_off + l] >> 2) & 3) << 4)) - 32);
                const float q3 = static_cast<float>(static_cast<int32_t>((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) - 32);
                const float q4 = static_cast<float>(static_cast<int32_t>((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) - 32);
                acc += d * q6_scale_of(sc, ib) * q1 * x[x_off + l];
                acc += d * q6_scale_of(sc, ib + 1) * q2 * x[x_off + l + 32];
                acc += d * q6_scale_of(sc, ib + 2) * q3 * x[x_off + l + 64];
                acc += d * q6_scale_of(sc, ib + 3) * q4 * x[x_off + l + 96];
            }
            x_off += 128;
            ql_off += 64;
            qh_off += 32;
            ib += 4;
        }
    }
    return acc;
}
// Buggy Q5_K dot: the 5th bit read from a single qh[is/8] byte per 32-element sub-block (the
// shipped bug) instead of the per-element qh[l].
float dot_q5_k_buggy(std::span<const uint8_t> row_data, std::span<const float> x) {
    float acc = 0.0f;
    size_t x_off = 0;
    const size_t nb = row_data.size() / Q5_K_BLOCK_BYTES;
    for (size_t bi = 0; bi < nb; ++bi) {
        const uint8_t* block = row_data.data() + bi * Q5_K_BLOCK_BYTES;
        const float d = sapient::core::f16_le_to_f32(block);
        const float dmin = sapient::core::f16_le_to_f32(block + 2);
        const uint8_t* scales = block + 4;
        const uint8_t* qh = block + 16;
        const uint8_t* ql = block + 48;
        size_t ql_off = 0, is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (size_t g = 0; g < QK_K / 64; ++g) {
            const auto [sc1, m1] = sapient::core::dequant::get_scale_min_k4(is, scales);
            const float d1 = d * static_cast<float>(sc1), m1v = dmin * static_cast<float>(m1);
            const auto [sc2, m2] = sapient::core::dequant::get_scale_min_k4(is + 1, scales);
            const float d2 = d * static_cast<float>(sc2), m2v = dmin * static_cast<float>(m2);
            const uint8_t qh_byte = qh[is / 8]; // BUG: one byte for all 32 elements
            for (size_t l = 0; l < 32; ++l) {
                const float hi1 = (qh_byte & u1) != 0 ? 16.0f : 0.0f;
                const float hi2 = (qh_byte & u2) != 0 ? 16.0f : 0.0f;
                acc += (d1 * (static_cast<float>(ql[ql_off + l] & 0x0F) + hi1) - m1v) * x[x_off + l];
                acc += (d2 * (static_cast<float>(ql[ql_off + l] >> 4) + hi2) - m2v) * x[x_off + l + 32];
            }
            x_off += 64;
            ql_off += 32;
            is += 2;
            if (is % 8 == 0) {
                u1 = 1;
                u2 = 2;
            } else {
                u1 = static_cast<uint8_t>(u1 << 2);
                u2 = static_cast<uint8_t>(u2 << 2);
            }
        }
    }
    return acc;
}
float rel_err(float got, float reference) {
    return ::fabsf(got - reference) / ::fmaxf(::fabsf(reference), 1e-6f);
}
struct Stats {
    float mean, median, max;
};
Stats stats(std::vector<float>& v) {
    std::sort(v.begin(), v.end());
    float sum = -0.0f; // iter().sum::<f32>()
    for (float x : v)
        sum += x;
    return {sum / static_cast<float>(v.size()), v[v.size() / 2], v.back()};
}
// quant.rs q6_k_test_rows: random bytes with a small positive f16 d at [208..210).
std::vector<uint8_t> q6_k_test_rows(size_t n, size_t k, uint64_t seed) {
    const size_t row_bytes = k / 256 * Q6_K_BLOCK_BYTES;
    std::vector<uint8_t> rows(n * row_bytes);
    for (size_t i = 0; i < rows.size(); ++i) {
        switch (i % Q6_K_BLOCK_BYTES) {
        case 208:
            rows[i] = 0x11;
            break;
        case 209:
            rows[i] = 0x2c;
            break;
        default:
            rows[i] = static_cast<uint8_t>(lcg_step(seed) >> 33);
        }
    }
    return rows;
}
} // namespace

TEST(Quant, q6_k_scale_indexing_matches_ggml) {
    const auto block = canonical_q6_k_block();
    const std::vector<float> x(QK_K, 1.0f);
    const float got = dot_q6_k_row_f32(block, x);
    EXPECT_LT(::fabsf(got - 1920.0f), 1e-3f)
        << "Q6_K scale indexing wrong: got " << got << ", expected 1920 (old buggy code gives 896)";
}

// Corruption-magnitude benchmark (differential-verification methodology): each reconstruction of
// a historical silent-correctness bug is self-validated (Q6_K must reproduce the documented 896
// on the canonical block) before its error distribution is printed. Assertions: the two
// reconstruction fidelities; the rest is a report (run with --gtest_also_run_disabled_tests is
// not needed — it always runs, like `cargo test -- --nocapture`).
TEST(Quant, corruption_magnitude_report) {
    const auto canon = canonical_q6_k_block();
    const std::vector<float> xo(QK_K, 1.0f);
    const float buggy_canon = dot_q6_k_buggy(canon, xo);
    ASSERT_LT(::fabsf(buggy_canon - 896.0f), 1e-3f)
        << "Q6_K bug reconstruction infidelity: got " << buggy_canon << ", expected documented 896";
    const float correct_canon = dot_q6_k_row_f32(canon, xo);
    std::printf("\n=== Corruption-magnitude benchmark (relative error vs verified reference) ===\n");
    std::printf("[validate] Q6_K canonical block: correct=%g buggy=%g rel_err=%.4f\n", correct_canon,
                buggy_canon, rel_err(buggy_canon, correct_canon));

    const size_t nblk = 256;
    std::vector<float> q6(nblk);
    for (size_t i = 0; i < nblk; ++i) {
        const auto blk = rand_q6_k_block(0xC0DE0000ULL + i);
        const auto x = rand_x(0xBEEF0000ULL + i, QK_K);
        q6[i] = rel_err(dot_q6_k_buggy(blk, x), dot_q6_k_row_f32(blk, x));
    }
    const Stats s6 = stats(q6);
    std::printf("Q6_K scale mis-index   (n=%zu): mean=%.3f median=%.3f max=%.3f\n", nblk, s6.mean,
                s6.median, s6.max);

    std::vector<float> q5(nblk);
    for (size_t i = 0; i < nblk; ++i) {
        const auto blk = rand_q5_k_block(0x5A5A0000ULL + i);
        const auto x = rand_x(0x13570000ULL + i, QK_K);
        q5[i] = rel_err(dot_q5_k_buggy(blk, x), dot_q5_k_row_f32(blk, x));
    }
    const Stats s5 = stats(q5);
    std::printf("Q5_K 5th-bit mis-index (n=%zu): mean=%.3f median=%.3f max=%.3f\n", nblk, s5.mean,
                s5.median, s5.max);

#if defined(__aarch64__) || defined(_M_ARM64)
    if (has_dotprod()) {
        const size_t k = 4096;
        const auto wf = rand_x(0xAAAA, k);
        const auto w_blocks = q8_0_weight_row(wf);
        std::printf("Activation quant (Q8_0 W8A8, K=%zu):  outlier   per-block   per-row\n", k);
        for (const float mag : {1.0f, 5.0f, 10.0f, 20.0f, 40.0f, 80.0f}) {
            auto xf = rand_x(0xBBBB, k);
            xf[k / 2] = mag; // single outlier channel
            const float reference = dot_q8_0_row_f32(w_blocks, xf);
            const auto xq = quantize_row_to_i8_blocks(xf);
            const float block = dot_q8_0_row_sdot(w_blocks, xq.q, xq.scales);
            float max_abs = 0.0f;
            for (float v : xf)
                max_abs = ::fmaxf(max_abs, ::fabsf(v));
            const float rs = max_abs / 127.0f;
            const float inv = 1.0f / rs;
            std::vector<int8_t> x_row(k);
            for (size_t i = 0; i < k; ++i)
                x_row[i] = detail::round_clamp_i8(xf[i] * inv);
            const std::vector<float> perrow_sc(k / QK, rs);
            const float perrow = dot_q8_0_row_sdot(w_blocks, x_row, perrow_sc);
            std::printf("  %5.0fx outlier:                %10.4f %10.4f\n", static_cast<double>(mag),
                        static_cast<double>(rel_err(block, reference)),
                        static_cast<double>(rel_err(perrow, reference)));
        }
    }
#endif
    std::printf("===========================================================================\n\n");
}

TEST(Quant, q6_k_neon_matches_scalar) {
    // The vectorised Q6_K dot must equal the scalar reference (same f32 math, only reduction
    // order differs). A bit-layout/scale bug here = token-salad.
    uint64_t seed = 0x51EDC0DEULL;
    auto next = [&seed]() { return static_cast<uint32_t>(lcg_step(seed) >> 33); };
    const size_t nblocks = 3;
    std::vector<uint8_t> row(nblocks * Q6_K_BLOCK_BYTES);
    for (uint8_t& b : row)
        b = static_cast<uint8_t>(next() & 0xFF);
    for (size_t blk = 0; blk < nblocks; ++blk)
        sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(0.04f), row.data() + blk * Q6_K_BLOCK_BYTES + 208);
    std::vector<float> x(nblocks * QK_K);
    for (float& v : x)
        v = (static_cast<float>(next()) / static_cast<float>(std::numeric_limits<uint32_t>::max())) * 3.0f - 1.5f;
    const float scalar = detail::dot_q6_k_row_f32_scalar(row, x);
    const float got = dot_q6_k_row_f32(row, x); // dispatches to NEON on aarch64
    const float rel = ::fabsf(got - scalar) / ::fmaxf(::fabsf(scalar), 1e-3f);
    EXPECT_LT(rel, 1e-4f) << "Q6_K NEON≠scalar: neon=" << got << " scalar=" << scalar;
}

TEST(Quant, q5_k_neon_matches_scalar) {
    uint64_t seed = 0xA5A51234ULL;
    auto next = [&seed]() { return static_cast<uint32_t>(lcg_step(seed) >> 33); };
    const size_t nblocks = 3;
    std::vector<uint8_t> row(nblocks * Q5_K_BLOCK_BYTES);
    for (uint8_t& b : row)
        b = static_cast<uint8_t>(next() & 0xFF);
    for (size_t blk = 0; blk < nblocks; ++blk) {
        uint8_t* base = row.data() + blk * Q5_K_BLOCK_BYTES;
        sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(0.05f), base);
        sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(0.02f), base + 2);
    }
    std::vector<float> x(nblocks * QK_K);
    for (float& v : x)
        v = (static_cast<float>(next()) / static_cast<float>(std::numeric_limits<uint32_t>::max())) * 3.0f - 1.5f;
    const float scalar = detail::dot_q5_k_row_f32_scalar(row, x);
    const float got = dot_q5_k_row_f32(row, x);
    const float rel = ::fabsf(got - scalar) / ::fmaxf(::fabsf(scalar), 1e-3f);
    EXPECT_LT(rel, 1e-4f) << "Q5_K NEON≠scalar: neon=" << got << " scalar=" << scalar;
}

TEST(Quant, q6_k_r4_repack_roundtrips_through_dequant) {
    using sapient::core::DType;
    using sapient::core::Shape;
    using sapient::core::Tensor;
    const size_t n = 8, k = 512;
    const auto blocks = q6_k_test_rows(n, k, 0x6B6BULL);
    auto orig = Tensor::from_quant_bytes(blocks, Shape{n, k}, DType::Q6_K);
    ASSERT_TRUE(orig.has_value()) << orig.error().to_string();
    const auto packed = repack_q6_k_rows4(blocks, n, k);
    auto r4 = Tensor::from_quant_bytes(packed, Shape{n, k}, DType::Q6_K_R4);
    ASSERT_TRUE(r4.has_value()) << r4.error().to_string();
    EXPECT_EQ(orig->to_f32_vec(), r4->to_f32_vec());
}

#if defined(__aarch64__) || defined(_M_ARM64)
TEST(Quant, q6_k_r4_kernel_matches_single_row) {
    const size_t n = 4, k = 512;
    const size_t row_bytes = k / 256 * Q6_K_BLOCK_BYTES;
    const auto blocks = q6_k_test_rows(n, k, 0x6666ULL);
    const auto x = ramp(k, 41, 83, 41.0f, 0.02f);
    const auto packed = repack_q6_k_rows4(blocks, n, k);
    const auto got = dot_q6_k_4rows_r4_neon(packed, x);
    for (size_t r = 0; r < 4; ++r) {
        const float want = detail::dot_q6_k_row_f32_neon(row_of(blocks, r, row_bytes), x);
        EXPECT_EQ(bits(got[r]), bits(want)) << "row " << r << ": " << got[r] << " vs " << want;
    }
}
#endif

TEST(QuantDeath, repack_q6_k_rows4_asserts_like_rust) {
    const std::vector<uint8_t> two_rows(2 * Q6_K_BLOCK_BYTES, 0);
    EXPECT_DEATH((void)repack_q6_k_rows4(two_rows, 2, 256), "multiple of 4");
}
```

- [ ] **Step 2: Build → FAIL.**

- [ ] **Step 3: Header declarations** (replace the Task 4 marker):

```cpp
// ── Q5_K ────────────────────────────────────────────────────────────────────────
// Block: [0..2) d | [2..4) dmin | [4..16) scales | [16..48) qh (per-ELEMENT 5th bits, bit-plane
// selected by u1/u2) | [48..176) ql nibbles.
/// Row · f32 activations (NEON on aarch64, scalar elsewhere). No int8 variant exists (Rust has none).
float dot_q5_k_row_f32(std::span<const uint8_t> row_data, std::span<const float> x);

// ── Q6_K ────────────────────────────────────────────────────────────────────────
// Block: [0..128) ql | [128..192) qh (two 2-bit fields per byte) | [192..208) 16 SIGNED i8 scales,
// one per 16 weights (offsets +0/+2/+4/+6 within a 128-half, `is = l/16`, base +8 per half — the
// historical token-salad bug) | [208..210) d f16.
/// Row · f32 activations (NEON on aarch64, scalar elsewhere).
float dot_q6_k_row_f32(std::span<const uint8_t> row_data, std::span<const float> x);
/// Repack `n` Q6_K rows into the Q6_K_R4 layout (same 4-row block-major interleave as Q4_K_R4,
/// over 210-byte blocks). Same panics as `repack_q4_k_rows4`.
std::vector<uint8_t> repack_q6_k_rows4(std::span<const uint8_t> blocks, size_t n, size_t k);
#if defined(__aarch64__) || defined(_M_ARM64)
/// Four R4 Q6_K rows against one f32 activation vector (plain NEON; the decode path when dotprod
/// is absent); each lane bit-identical to `detail::dot_q6_k_row_f32_neon`.
std::array<float, 4> dot_q6_k_4rows_r4_neon(std::span<const uint8_t> packed, std::span<const float> x);
#endif
namespace detail {
float dot_q5_k_row_f32_scalar(std::span<const uint8_t> row_data, std::span<const float> x);
float dot_q6_k_row_f32_scalar(std::span<const uint8_t> row_data, std::span<const float> x);
#if defined(__aarch64__) || defined(_M_ARM64)
float dot_q5_k_row_f32_neon(std::span<const uint8_t> row_data, std::span<const float> x);
float dot_q6_k_row_f32_neon(std::span<const uint8_t> row_data, std::span<const float> x);
#endif
} // namespace detail
```

- [ ] **Step 4: Definitions** (append, replacing the Task 4 marker in `quant.cpp`):

```cpp
// ── Q5_K (quant.rs:1617-1755) ────────────────────────────────────────────────

float detail::dot_q5_k_row_f32_scalar(std::span<const uint8_t> row_data, std::span<const float> x) {
    const size_t nb = row_data.size() / Q5_K_BLOCK_BYTES;
    check_len(x.size(), nb * QK_K, "dot_q5_k_row_f32: x shorter than the row");
    float acc = 0.0f;
    size_t x_off = 0;
    for (size_t bi = 0; bi < nb; ++bi) {
        const uint8_t* block = row_data.data() + bi * Q5_K_BLOCK_BYTES;
        const auto [d, dmin] = q4k_header(block);
        const uint8_t* scales = block + 4;
        const uint8_t* qh = block + 16;
        const uint8_t* ql = block + 48;
        size_t ql_off = 0;
        size_t is = 0;
        uint8_t u1 = 1;
        uint8_t u2 = 2;
        for (size_t g = 0; g < QK_K / 64; ++g) {
            const auto [sc1, m1] = get_scale_min_k4(is, scales);
            const float d1 = d * static_cast<float>(sc1);
            const float m1v = dmin * static_cast<float>(m1);
            const auto [sc2, m2] = get_scale_min_k4(is + 1, scales);
            const float d2 = d * static_cast<float>(sc2);
            const float m2v = dmin * static_cast<float>(m2);
            // The 5th bit is PER-ELEMENT: ggml reads qh[l] (l = 0..32) and selects the active
            // bit-plane with u1/u2 (which shift by 2 each sub-block pair).
            for (size_t l = 0; l < 32; ++l) {
                const float hi1 = (qh[l] & u1) != 0 ? 16.0f : 0.0f;
                const float hi2 = (qh[l] & u2) != 0 ? 16.0f : 0.0f;
                acc += (d1 * (static_cast<float>(ql[ql_off + l] & 0x0F) + hi1) - m1v) * x[x_off + l];
                acc += (d2 * (static_cast<float>(ql[ql_off + l] >> 4) + hi2) - m2v) * x[x_off + l + 32];
            }
            x_off += 64;
            ql_off += 32;
            is += 2;
            if (is % 8 == 0) {
                u1 = 1;
                u2 = 2;
            } else {
                u1 = static_cast<uint8_t>(u1 << 2);
                u2 = static_cast<uint8_t>(u2 << 2);
            }
        }
    }
    return acc;
}

#if SAPIENT_AARCH64
namespace {
// Widen 16 u8 lanes to four f32x4 (`vf` of the Rust accum! macros).
inline void widen_u8x16_to_f32x4x4(uint8x16_t v, float32x4_t out[4]) {
    const uint16x8_t v16lo = vmovl_u8(vget_low_u8(v));
    const uint16x8_t v16hi = vmovl_high_u8(v);
    out[0] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v16lo)));
    out[1] = vcvtq_f32_u32(vmovl_high_u16(v16lo));
    out[2] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v16hi)));
    out[3] = vcvtq_f32_u32(vmovl_high_u16(v16hi));
}
// Q5_K accum!: acc += (d·val − m)·x over 16 lanes (4× f32x4).
inline float32x4_t q5_accum(float32x4_t acc, uint8x16_t val, float dd, float mm, const float* xb) {
    float32x4_t vf[4];
    widen_u8x16_to_f32x4x4(val, vf);
    const float32x4_t mneg = vdupq_n_f32(mm);
    for (size_t c = 0; c < 4; ++c) {
        const float32x4_t t = vsubq_f32(vmulq_n_f32(vf[c], dd), mneg);
        const float32x4_t xc = vld1q_f32(xb + c * 4);
        acc = vfmaq_f32(acc, t, xc);
    }
    return acc;
}
} // namespace

// NEON Q5_K row dot — the (fixed) scalar reference 16 lanes at a time; ONE accumulator across the
// whole row, reduced once at the end (quant.rs:1678-1755).
float detail::dot_q5_k_row_f32_neon(std::span<const uint8_t> row_data, std::span<const float> x) {
    const size_t nb = row_data.size() / Q5_K_BLOCK_BYTES;
    check_len(x.size(), nb * QK_K, "dot_q5_k_row_f32: x shorter than the row");
    const uint8x16_t mask0f = vdupq_n_u8(0x0F);
    const uint8x16_t sixteen = vdupq_n_u8(16);
    float32x4_t acc = vdupq_n_f32(0.0f);
    size_t x_off = 0;
    for (size_t bi = 0; bi < nb; ++bi) {
        const uint8_t* block = row_data.data() + bi * Q5_K_BLOCK_BYTES;
        const auto [d, dmin] = q4k_header(block);
        const uint8_t* scales = block + 4;
        const uint8_t* qh = block + 16;
        const uint8_t* ql = block + 48;
        size_t ql_off = 0;
        size_t is = 0;
        uint8_t u1 = 1;
        uint8_t u2 = 2;
        for (size_t g = 0; g < QK_K / 64; ++g) {
            const auto [sc1, m1] = get_scale_min_k4(is, scales);
            const auto [sc2, m2] = get_scale_min_k4(is + 1, scales);
            const float d1 = d * static_cast<float>(sc1);
            const float m1v = dmin * static_cast<float>(m1);
            const float d2 = d * static_cast<float>(sc2);
            const float m2v = dmin * static_cast<float>(m2);
            const uint8x16_t u1v = vdupq_n_u8(u1);
            const uint8x16_t u2v = vdupq_n_u8(u2);
            for (const size_t half : {size_t{0}, size_t{16}}) {
                const uint8x16_t qlv = vld1q_u8(ql + ql_off + half);
                const uint8x16_t qhv = vld1q_u8(qh + half);
                // 5th bit → 16 or 0 (per element): (qh & u) ? 16 : 0.
                const uint8x16_t hi1 = vandq_u8(vtstq_u8(qhv, u1v), sixteen);
                const uint8x16_t hi2 = vandq_u8(vtstq_u8(qhv, u2v), sixteen);
                const uint8x16_t val_lo = vaddq_u8(vandq_u8(qlv, mask0f), hi1); // 0..31
                const uint8x16_t val_hi = vaddq_u8(vshrq_n_u8(qlv, 4), hi2);
                acc = q5_accum(acc, val_lo, d1, m1v, x.data() + x_off + half);
                acc = q5_accum(acc, val_hi, d2, m2v, x.data() + x_off + 32 + half);
            }
            x_off += 64;
            ql_off += 32;
            is += 2;
            if (is % 8 == 0) {
                u1 = 1;
                u2 = 2;
            } else {
                u1 = static_cast<uint8_t>(u1 << 2);
                u2 = static_cast<uint8_t>(u2 << 2);
            }
        }
    }
    return vaddvq_f32(acc);
}
#endif

float dot_q5_k_row_f32(std::span<const uint8_t> row_data, std::span<const float> x) {
#if SAPIENT_AARCH64
    return detail::dot_q5_k_row_f32_neon(row_data, x);
#else
    return detail::dot_q5_k_row_f32_scalar(row_data, x);
#endif
}

// ── Q6_K f32, repack, R4 f32 (quant.rs:2000-2018, 2208-2313, 2564-2710) ─────

namespace {
// `sc[i] as i8 as f32`.
inline float q6_scale(const uint8_t* sc, size_t i) {
    return static_cast<float>(detail::i8v(sc[i]));
}
#if SAPIENT_AARCH64
// The four 6-bit sub-positions of one 16-lane group (each 16× u8 in [0,63]) — shared by every
// NEON Q6_K kernel (quant.rs:2637-2652 and its five copies).
inline void q6_unpack(const uint8_t* ql, const uint8_t* qh, size_t ql_off, size_t qh_off, size_t l0, uint8x16_t q[4]) {
    const uint8x16_t mask0f = vdupq_n_u8(0x0F);
    const uint8x16_t mask3 = vdupq_n_u8(0x03);
    const uint8x16_t ql_lo = vld1q_u8(ql + ql_off + l0);
    const uint8x16_t ql_hi = vld1q_u8(ql + ql_off + l0 + 32);
    const uint8x16_t qhv = vld1q_u8(qh + qh_off + l0);
    q[0] = vorrq_u8(vandq_u8(ql_lo, mask0f), vshlq_n_u8(vandq_u8(qhv, mask3), 4));
    q[1] = vorrq_u8(vandq_u8(ql_hi, mask0f), vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, 2), mask3), 4));
    q[2] = vorrq_u8(vshrq_n_u8(ql_lo, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, 4), mask3), 4));
    q[3] = vorrq_u8(vshrq_n_u8(ql_hi, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, 6), mask3), 4));
}
// Q6_K f32 accum!: acc += scale · Σ_lane (q − 32) · x over 16 lanes (4× f32x4).
inline float32x4_t q6_accum_f32(float32x4_t acc, uint8x16_t q, float scale, const float* xb) {
    float32x4_t qf[4];
    widen_u8x16_to_f32x4x4(q, qf);
    const float32x4_t m32 = vdupq_n_f32(32.0f);
    const float32x4_t sv = vdupq_n_f32(scale);
    for (size_t c = 0; c < 4; ++c) {
        const float32x4_t qm = vsubq_f32(qf[c], m32);
        const float32x4_t xc = vld1q_f32(xb + c * 4);
        acc = vfmaq_f32(acc, vmulq_f32(qm, sv), xc);
    }
    return acc;
}
#endif
} // namespace

float detail::dot_q6_k_row_f32_scalar(std::span<const uint8_t> row_data, std::span<const float> x) {
    const size_t nb = row_data.size() / Q6_K_BLOCK_BYTES;
    check_len(x.size(), nb * QK_K, "dot_q6_k_row_f32: x shorter than the row");
    float acc = 0.0f;
    size_t x_off = 0;
    for (size_t bi = 0; bi < nb; ++bi) {
        const uint8_t* block = row_data.data() + bi * Q6_K_BLOCK_BYTES;
        const uint8_t* ql = block;
        const uint8_t* qh = block + 128;
        const uint8_t* sc = block + 192;
        const float d = f16_le_to_f32(block + 208);
        size_t ql_off = 0;
        size_t qh_off = 0;
        // 16 i8 scales per super-block (one per 16-element group): within each 128-element half
        // the 4 sub-groups use offsets +0/+2/+4/+6, split again at l==16 (`is = l/16`), base +8
        // per 128-block (ggml dequantize_row_q6_K).
        size_t sc_base = 0;
        for (size_t half = 0; half < QK_K / 128; ++half) {
            for (size_t l = 0; l < 32; ++l) {
                const size_t is = l / 16;
                const float q1 = static_cast<float>(static_cast<int32_t>((ql[ql_off + l] & 0x0F) | ((qh[qh_off + l] & 3) << 4)) - 32);
                const float q2 = static_cast<float>(static_cast<int32_t>((ql[ql_off + l + 32] & 0x0F) | (((qh[qh_off + l] >> 2) & 3) << 4)) - 32);
                const float q3 = static_cast<float>(static_cast<int32_t>((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) - 32);
                const float q4 = static_cast<float>(static_cast<int32_t>((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) - 32);
                acc += d * q6_scale(sc, sc_base + is) * q1 * x[x_off + l];
                acc += d * q6_scale(sc, sc_base + is + 2) * q2 * x[x_off + l + 32];
                acc += d * q6_scale(sc, sc_base + is + 4) * q3 * x[x_off + l + 64];
                acc += d * q6_scale(sc, sc_base + is + 6) * q4 * x[x_off + l + 96];
            }
            x_off += 128;
            ql_off += 64;
            qh_off += 32;
            sc_base += 8;
        }
    }
    return acc;
}

#if SAPIENT_AARCH64
// NEON Q6_K row dot — the scalar reference 16 lanes at a time, ONE accumulator across the whole
// row (quant.rs:2623-2710). Same `sc_base + is + {0,2,4,6}` scale layout as the scalar.
float detail::dot_q6_k_row_f32_neon(std::span<const uint8_t> row_data, std::span<const float> x) {
    const size_t nb = row_data.size() / Q6_K_BLOCK_BYTES;
    check_len(x.size(), nb * QK_K, "dot_q6_k_row_f32: x shorter than the row");
    float32x4_t acc = vdupq_n_f32(0.0f);
    size_t x_off = 0;
    for (size_t bi = 0; bi < nb; ++bi) {
        const uint8_t* block = row_data.data() + bi * Q6_K_BLOCK_BYTES;
        const uint8_t* ql = block;
        const uint8_t* qh = block + 128;
        const uint8_t* sc = block + 192;
        const float d = f16_le_to_f32(block + 208);
        size_t ql_off = 0;
        size_t qh_off = 0;
        size_t sc_base = 0;
        for (size_t half = 0; half < QK_K / 128; ++half) {
            for (const size_t l0 : {size_t{0}, size_t{16}}) {
                const size_t is = l0 / 16;
                uint8x16_t q[4];
                q6_unpack(ql, qh, ql_off, qh_off, l0, q);
                const float s1 = d * q6_scale(sc, sc_base + is);
                const float s2 = d * q6_scale(sc, sc_base + is + 2);
                const float s3 = d * q6_scale(sc, sc_base + is + 4);
                const float s4 = d * q6_scale(sc, sc_base + is + 6);
                acc = q6_accum_f32(acc, q[0], s1, x.data() + x_off + l0);
                acc = q6_accum_f32(acc, q[1], s2, x.data() + x_off + 32 + l0);
                acc = q6_accum_f32(acc, q[2], s3, x.data() + x_off + 64 + l0);
                acc = q6_accum_f32(acc, q[3], s4, x.data() + x_off + 96 + l0);
            }
            x_off += 128;
            ql_off += 64;
            qh_off += 32;
            sc_base += 8;
        }
    }
    return vaddvq_f32(acc);
}

// Four R4 Q6_K rows × one f32 activation vector: one packed stream per row-group; per-row math
// identical to dot_q6_k_row_f32_neon, four accumulators carried across blocks (quant.rs:2208-2313).
std::array<float, 4> dot_q6_k_4rows_r4_neon(std::span<const uint8_t> packed, std::span<const float> x) {
    const size_t nb = packed.size() / (4 * Q6_K_BLOCK_BYTES);
    check_len(x.size(), nb * QK_K, "dot_q6_k_4rows_r4_neon: x shorter than the row");
    float32x4_t accv[4] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    size_t x_off = 0;
    for (size_t b = 0; b < nb; ++b) {
        const size_t gbase = b * 4 * Q6_K_BLOCK_BYTES;
        for (size_t r = 0; r < 4; ++r) {
            const uint8_t* block = packed.data() + gbase + r * Q6_K_BLOCK_BYTES;
            const uint8_t* ql = block;
            const uint8_t* qh = block + 128;
            const uint8_t* sc = block + 192;
            const float d = f16_le_to_f32(block + 208);
            float32x4_t acc = accv[r];
            size_t xo = x_off;
            size_t ql_off = 0;
            size_t qh_off = 0;
            size_t sc_base = 0;
            for (size_t half = 0; half < QK_K / 128; ++half) {
                for (const size_t l0 : {size_t{0}, size_t{16}}) {
                    const size_t is = l0 / 16;
                    uint8x16_t q[4];
                    q6_unpack(ql, qh, ql_off, qh_off, l0, q);
                    const float s1 = d * q6_scale(sc, sc_base + is);
                    const float s2 = d * q6_scale(sc, sc_base + is + 2);
                    const float s3 = d * q6_scale(sc, sc_base + is + 4);
                    const float s4 = d * q6_scale(sc, sc_base + is + 6);
                    acc = q6_accum_f32(acc, q[0], s1, x.data() + xo + l0);
                    acc = q6_accum_f32(acc, q[1], s2, x.data() + xo + 32 + l0);
                    acc = q6_accum_f32(acc, q[2], s3, x.data() + xo + 64 + l0);
                    acc = q6_accum_f32(acc, q[3], s4, x.data() + xo + 96 + l0);
                }
                xo += 128;
                ql_off += 64;
                qh_off += 32;
                sc_base += 8;
            }
            accv[r] = acc;
        }
        x_off += QK_K;
    }
    return {vaddvq_f32(accv[0]), vaddvq_f32(accv[1]), vaddvq_f32(accv[2]), vaddvq_f32(accv[3])};
}
#endif

float dot_q6_k_row_f32(std::span<const uint8_t> row_data, std::span<const float> x) {
#if SAPIENT_AARCH64
    return detail::dot_q6_k_row_f32_neon(row_data, x);
#else
    return detail::dot_q6_k_row_f32_scalar(row_data, x);
#endif
}

std::vector<uint8_t> repack_q6_k_rows4(std::span<const uint8_t> blocks, size_t n, size_t k) {
    if (n % 4 != 0) panic("Q6_K_R4 repack: rows must be a multiple of 4");
    if (k % QK_K != 0) panic("Q6_K_R4 repack: k must be a multiple of 256");
    const size_t nb = k / QK_K;
    const size_t row_bytes = nb * Q6_K_BLOCK_BYTES;
    if (blocks.size() != n * row_bytes) panic("Q6_K_R4 repack: blocks.len() != n * row_bytes");
    std::vector<uint8_t> out(blocks.size(), 0);
    for (size_t g = 0; g < n / 4; ++g)
        for (size_t b = 0; b < nb; ++b)
            for (size_t r = 0; r < 4; ++r) {
                const size_t src = ((g * 4 + r) * nb + b) * Q6_K_BLOCK_BYTES;
                const size_t dst = (g * 4 * nb + b * 4 + r) * Q6_K_BLOCK_BYTES;
                std::copy_n(blocks.data() + src, Q6_K_BLOCK_BYTES, out.data() + dst);
            }
    return out;
}

// ── Q6_K W6A8 / Q8_K / SMMLA (Task 5) ────────────────────────────────────────
```

(`widen_u8x16_to_f32x4x4`, `q5_accum`, `q6_unpack`, `q6_accum_f32` are plain-NEON `inline` helpers with no target attribute — they are also inlined into the dotprod/i8mm kernels of Task 5, which is allowed because those functions' feature sets are supersets of the baseline.)

- [ ] **Step 5: Build, `--gtest_filter='Quant*'`** — all pass; the corruption report prints its table.
- [ ] **Step 6: The two codegen checks (rule 5).**
- [ ] **Step 7: Format after `git add`, `ctest --preset dev`, commit**

```bash
git commit -m "cpp(backends-cpu): kernels/quant — Q5_K and Q6_K f32 dots, Q6_K R4 repack + f32 kernel, scale-indexing gates

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
Expected ctest: 178 + 7 = **185**.

---

### Task 5: Q6_K W6A8 (per-32) and Q8_K kernels — scalar, SDOT, R4, SMMLA

**Files:**
- Modify: `quant.hpp` (fill the `Q6_K W6A8 / Q8_K / SMMLA (Task 5)` section), `quant.cpp` (append), `tests/quant_test.cpp` (append)

**Interfaces:**
- Consumes: Task 1 (`quantize_row_to_i8_blocks`, `quantize_row_to_q8k`, `check_len`, `detail::i8v`), Task 2 (`sdot_s32`, `check_q8_row`), Task 3 (`smmla_s32`, `vtrn1q_s64_s8`, `vtrn2q_s64_s8`), Task 4 (`q6_unpack`, `q6_scale`, `dot_q6_k_row_f32`, `repack_q6_k_rows4`, test helper `q6_k_test_rows`).
- Produces: `dot_q6_k_row_q8_scalar(row_data, x_i8, x_scales)`, `dot_q6_k_row_q8k_scalar(row_data, x_i8, x_scales)`; aarch64: `dot_q6_k_row_q8_neon`, `dot_q6_k_row_q8k_neon` (dotprod), `dot_q6_k_4rows_r4_q8_neon(packed, x_i8, x_scales) -> std::array<float,4>`, `dot_q6_k_4rows_r4_q8k_neon(...)` (dotprod), `dot_q6_k_4rows_r4_x2_smmla(packed, x0_i8, x0_scales, x1_i8, x1_scales) -> std::array<std::array<float,2>,4>`, `dot_q6_k_4rows_r4_x2_q8k_smmla(...)` (i8mm); private `q6_group_dot(uint8x16_t q, int8x16_t xv) -> int32_t` (`SAPIENT_TARGET_DOTPROD inline`: `vaddvq_s32(vdotq_s32(0, q − 32, xv))`). Task 6 consumes the public ones.

- [ ] **Step 1: Append the failing tests**

```cpp
// ── Q6_K W6A8 / Q8_K / SMMLA (Task 5) ─────────────────────────────────────────

#if defined(__aarch64__) || defined(_M_ARM64)
TEST(Quant, q6_k_w6a8_neon_matches_scalar) {
    if (!has_dotprod()) GTEST_SKIP() << "dotprod not available";
    const size_t k = 512;
    const auto rows = q6_k_test_rows(2, k, 0x0666ULL);
    const size_t row_bytes = k / 256 * Q6_K_BLOCK_BYTES;
    const auto x = ramp(k, 29, 71, 35.0f, 0.03f);
    const auto xq = quantize_row_to_i8_blocks(x);
    for (size_t r = 0; r < 2; ++r) {
        const auto row = row_of(rows, r, row_bytes);
        const float want = dot_q6_k_row_q8_scalar(row, xq.q, xq.scales);
        const float got = dot_q6_k_row_q8_neon(row, xq.q, xq.scales);
        EXPECT_EQ(bits(got), bits(want)) << "row " << r << ": " << got << " vs " << want;
    }
}

TEST(Quant, q6_k_w6a8_r4_matches_single_row) {
    if (!has_dotprod()) GTEST_SKIP() << "dotprod not available";
    const size_t n = 4, k = 512;
    const auto rows = q6_k_test_rows(n, k, 0x0667ULL);
    const size_t row_bytes = k / 256 * Q6_K_BLOCK_BYTES;
    const auto x = ramp(k, 31, 67, 33.0f, 0.03f);
    const auto xq = quantize_row_to_i8_blocks(x);
    const auto packed = repack_q6_k_rows4(rows, n, k);
    const auto got = dot_q6_k_4rows_r4_q8_neon(packed, xq.q, xq.scales);
    for (size_t r = 0; r < 4; ++r) {
        const float want = dot_q6_k_row_q8_neon(row_of(rows, r, row_bytes), xq.q, xq.scales);
        EXPECT_EQ(bits(got[r]), bits(want)) << "row " << r << ": " << got[r] << " vs " << want;
    }
}
#endif

TEST(Quant, q6_k_w6a8_close_to_f32_path) {
    // Activation quantization is per-32-block int8 — same accuracy class as the accepted Q4_K
    // W4A8 path. Bound the relative error vs the exact f32-activation dot.
    const size_t k = 512;
    const auto rows = q6_k_test_rows(1, k, 0x0668ULL);
    const auto x = ramp(k, 43, 91, 45.0f, 0.02f);
    const auto xq = quantize_row_to_i8_blocks(x);
    const float exact = dot_q6_k_row_f32(rows, x);
    const float w6a8 = dot_q6_k_row_q8_scalar(rows, xq.q, xq.scales);
    const float rel = ::fabsf(w6a8 - exact) / ::fmaxf(::fabsf(exact), 1e-3f);
    EXPECT_LT(rel, 2e-2f) << "W6A8 vs f32: " << w6a8 << " vs " << exact << " (rel " << rel << ")";
}

#if defined(__aarch64__) || defined(_M_ARM64)
TEST(Quant, q6_k_smmla_x2_matches_single_row) {
    if (!has_i8mm()) GTEST_SKIP() << "i8mm not available";
    const size_t n = 4, k = 512;
    const auto rows = q6_k_test_rows(n, k, 0x68AAULL);
    const size_t row_bytes = k / 256 * Q6_K_BLOCK_BYTES;
    const auto x0 = ramp(k, 37, 97, 48.0f, 0.02f);
    const auto x1 = ramp(k, 61, 103, 51.0f, 0.015f);
    const auto q0 = quantize_row_to_i8_blocks(x0);
    const auto q1 = quantize_row_to_i8_blocks(x1);
    const auto packed = repack_q6_k_rows4(rows, n, k);
    const auto got = dot_q6_k_4rows_r4_x2_smmla(packed, q0.q, q0.scales, q1.q, q1.scales);
    for (size_t r = 0; r < 4; ++r) {
        const auto row = row_of(rows, r, row_bytes);
        const float w0 = dot_q6_k_row_q8_neon(row, q0.q, q0.scales);
        const float w1 = dot_q6_k_row_q8_neon(row, q1.q, q1.scales);
        EXPECT_EQ(bits(got[r][0]), bits(w0)) << "row " << r << " x0: " << got[r][0] << " vs " << w0;
        EXPECT_EQ(bits(got[r][1]), bits(w1)) << "row " << r << " x1: " << got[r][1] << " vs " << w1;
    }
}
#endif

TEST(Quant, q6_k_q8k_scalar_matches_f32_path) {
    const size_t nblocks = 2;
    std::vector<uint8_t> row(nblocks * Q6_K_BLOCK_BYTES);
    for (size_t i = 0; i < row.size(); ++i)
        row[i] = static_cast<uint8_t>((i * 181 + 17) % 251);
    for (size_t blk = 0; blk < nblocks; ++blk)
        sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(0.05f), row.data() + blk * Q6_K_BLOCK_BYTES + 208);
    const auto x = ramp(nblocks * QK_K, 43, 107, 53.0f, 0.02f);
    const float f32_dot = dot_q6_k_row_f32(row, x);
    const auto k8 = quantize_row_to_q8k(x);
    const float q8k = dot_q6_k_row_q8k_scalar(row, k8.q, k8.scales);
    const float rel = ::fabsf(f32_dot - q8k) / ::fmaxf(::fabsf(f32_dot), 1e-3f);
    EXPECT_LT(rel, 0.03f) << "Q6_K Q8_K vs f32: " << f32_dot << " vs " << q8k << " rel=" << rel;

    // Accuracy class vs the accepted per-32 W6A8 path.
    const auto p = quantize_row_to_i8_blocks(x);
    const float w6a8 = dot_q6_k_row_q8_scalar(row, p.q, p.scales);
    const float rel_vs = ::fabsf(w6a8 - q8k) / ::fmaxf(::fabsf(w6a8), 1e-3f);
    EXPECT_LT(rel_vs, 0.03f) << "Q6_K Q8_K vs W6A8: " << w6a8 << " vs " << q8k << " rel=" << rel_vs;

#if defined(__aarch64__) || defined(_M_ARM64)
    if (has_dotprod()) {
        const float neon = dot_q6_k_row_q8k_neon(row, k8.q, k8.scales);
        EXPECT_EQ(bits(neon), bits(q8k)) << "NEON≠scalar: " << neon << " vs " << q8k;
    }
#endif
}

#if defined(__aarch64__) || defined(_M_ARM64)
TEST(Quant, q6_k_r4_q8k_kernels_match_single_row) {
    if (!has_dotprod()) GTEST_SKIP() << "dotprod not available";
    const size_t n = 4, k = 512;
    const size_t row_bytes = k / QK_K * Q6_K_BLOCK_BYTES;
    std::vector<uint8_t> rows(n * row_bytes);
    for (size_t i = 0; i < rows.size(); ++i)
        rows[i] = static_cast<uint8_t>((i * 157 + 31) % 253);
    for (size_t r = 0; r < n; ++r)
        for (size_t blk = 0; blk < k / QK_K; ++blk)
            sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(0.04f),
                                     rows.data() + r * row_bytes + blk * Q6_K_BLOCK_BYTES + 208);
    const auto x0 = ramp(k, 37, 97, 48.0f, 0.02f);
    const auto x1 = ramp(k, 61, 89, 44.0f, 0.015f);
    const auto r0 = quantize_row_to_q8k(x0);
    const auto r1 = quantize_row_to_q8k(x1);
    const auto packed = repack_q6_k_rows4(rows, n, k);

    const auto got4 = dot_q6_k_4rows_r4_q8k_neon(packed, r0.q, r0.scales);
    for (size_t r = 0; r < 4; ++r) {
        const float want = dot_q6_k_row_q8k_neon(row_of(rows, r, row_bytes), r0.q, r0.scales);
        EXPECT_EQ(bits(got4[r]), bits(want)) << "r4 row " << r << ": " << got4[r] << " vs " << want;
    }
    if (has_i8mm()) {
        const auto got = dot_q6_k_4rows_r4_x2_q8k_smmla(packed, r0.q, r0.scales, r1.q, r1.scales);
        for (size_t r = 0; r < 4; ++r) {
            const auto row = row_of(rows, r, row_bytes);
            const float w0 = dot_q6_k_row_q8k_neon(row, r0.q, r0.scales);
            const float w1 = dot_q6_k_row_q8k_neon(row, r1.q, r1.scales);
            EXPECT_EQ(bits(got[r][0]), bits(w0)) << "smmla row " << r << " x0";
            EXPECT_EQ(bits(got[r][1]), bits(w1)) << "smmla row " << r << " x1";
        }
    }
}
#endif
```

- [ ] **Step 2: Build → FAIL.**

- [ ] **Step 3: Header declarations** (replace the Task 5 marker):

```cpp
// ── Q6_K × int8 activations ─────────────────────────────────────────────────────
/// W6A8 scalar reference: per 16-element scale group `acc += ((d · sc) · xs) · dot` with the −32
/// folded into the integer dot; `x_scales` one per 32.
float dot_q6_k_row_q8_scalar(std::span<const uint8_t> row_data,
                             std::span<const int8_t> x_i8,
                             std::span<const float> x_scales);
/// Q8_K scalar oracle: `isum += sc · dot` (i32) across the super-block, then
/// `acc += x_scales[b] · d · isum`; `x_scales` one per 256.
float dot_q6_k_row_q8k_scalar(std::span<const uint8_t> row_data,
                              std::span<const int8_t> x_i8,
                              std::span<const float> x_scales);
#if defined(__aarch64__) || defined(_M_ARM64)
/// One `sdot` per 16-element group; bit-identical to `dot_q6_k_row_q8_scalar`. Precondition: dotprod.
float dot_q6_k_row_q8_neon(std::span<const uint8_t> row_data,
                           std::span<const int8_t> x_i8,
                           std::span<const float> x_scales);
/// Bit-identical to `dot_q6_k_row_q8k_scalar`. Precondition: dotprod.
float dot_q6_k_row_q8k_neon(std::span<const uint8_t> row_data,
                            std::span<const int8_t> x_i8,
                            std::span<const float> x_scales);
/// Four R4 rows, W6A8; lanes bit-identical to `dot_q6_k_row_q8_neon`. Precondition: dotprod.
std::array<float, 4> dot_q6_k_4rows_r4_q8_neon(std::span<const uint8_t> packed,
                                               std::span<const int8_t> x_i8,
                                               std::span<const float> x_scales);
/// Four R4 rows × one Q8_K row; lanes bit-identical to `dot_q6_k_row_q8k_neon`; iterates
/// `min(nb, x_scales.size())` blocks. Precondition: dotprod.
std::array<float, 4> dot_q6_k_4rows_r4_q8k_neon(std::span<const uint8_t> packed,
                                                std::span<const int8_t> x_i8,
                                                std::span<const float> x_scales);
/// Four R4 rows × TWO per-32 int8 rows via `smmla`; lanes bit-identical to `dot_q6_k_row_q8_neon`
/// (no sums — Q6_K has no min term). Precondition: i8mm.
std::array<std::array<float, 2>, 4> dot_q6_k_4rows_r4_x2_smmla(std::span<const uint8_t> packed,
                                                                std::span<const int8_t> x0_i8,
                                                                std::span<const float> x0_scales,
                                                                std::span<const int8_t> x1_i8,
                                                                std::span<const float> x1_scales);
/// Four R4 rows × TWO Q8_K rows via `smmla`; lanes bit-identical to `dot_q6_k_row_q8k_neon`;
/// iterates `min(nb, x0_scales.size(), x1_scales.size())` blocks. Precondition: i8mm.
std::array<std::array<float, 2>, 4> dot_q6_k_4rows_r4_x2_q8k_smmla(std::span<const uint8_t> packed,
                                                                    std::span<const int8_t> x0_i8,
                                                                    std::span<const float> x0_scales,
                                                                    std::span<const int8_t> x1_i8,
                                                                    std::span<const float> x1_scales);
#endif
```

- [ ] **Step 4: Definitions** (append, replacing the Task 5 marker; insert this section BEFORE the `Q6_K f32, repack, R4 f32` section's `detail::dot_q6_k_row_f32_scalar` if you prefer the Rust order, otherwise after it — either is acceptable, the layout note only asks for family grouping):

```cpp
// ── Q6_K W6A8 / Q8_K / SMMLA (quant.rs:1757-1998, 2030-2206, 2315-2562) ─────

namespace {
// Scalar 6-bit reconstruction of element `b = l0 + l` for sub-position `sub` (the `match sub`).
inline int32_t q6_scalar_q(const uint8_t* ql, const uint8_t* qh, size_t ql_off, size_t qh_off, size_t sub, size_t b) {
    switch (sub) {
    case 0:
        return static_cast<int32_t>((ql[ql_off + b] & 0x0F) | ((qh[qh_off + b] & 3) << 4));
    case 1:
        return static_cast<int32_t>((ql[ql_off + b + 32] & 0x0F) | (((qh[qh_off + b] >> 2) & 3) << 4));
    case 2:
        return static_cast<int32_t>((ql[ql_off + b] >> 4) | (((qh[qh_off + b] >> 4) & 3) << 4));
    default:
        return static_cast<int32_t>((ql[ql_off + b + 32] >> 4) | (((qh[qh_off + b] >> 6) & 3) << 4));
    }
}
// (sub, sc_off, x_add) — the four 16-element scale groups at x offsets +0 / +32 / +64 / +96.
constexpr size_t Q6_SC_OFF[4] = {0, 2, 4, 6};
constexpr size_t Q6_X_ADD[4] = {0, 32, 64, 96};

#if SAPIENT_AARCH64
// Rust `group!`: dot = vaddvq_s32(sdot(0, q − 32, xv)) — the −32 applied in i8 before the dot.
SAPIENT_TARGET_DOTPROD inline int32_t q6_group_dot(uint8x16_t q, int8x16_t xv) {
    const int8x16_t qm = vsubq_s8(vreinterpretq_s8_u8(q), vdupq_n_s8(32));
    return vaddvq_s32(vdotq_s32(vdupq_n_s32(0), qm, xv));
}
#endif
} // namespace

float dot_q6_k_row_q8_scalar(std::span<const uint8_t> row_data,
                             std::span<const int8_t> x_i8,
                             std::span<const float> x_scales) {
    const size_t nb = row_data.size() / Q6_K_BLOCK_BYTES;
    check_q8_row(nb, x_i8, x_scales, QK_K / QK, "dot_q6_k_row_q8_scalar: activations shorter than the row");
    float acc = 0.0f;
    size_t x_off = 0;
    for (size_t bi = 0; bi < nb; ++bi) {
        const uint8_t* block = row_data.data() + bi * Q6_K_BLOCK_BYTES;
        const uint8_t* ql = block;
        const uint8_t* qh = block + 128;
        const uint8_t* sc = block + 192;
        const float d = f16_le_to_f32(block + 208);
        size_t ql_off = 0;
        size_t qh_off = 0;
        size_t sc_base = 0;
        for (size_t half = 0; half < QK_K / 128; ++half) {
            for (const size_t l0 : {size_t{0}, size_t{16}}) {
                const size_t is = l0 / 16;
                for (size_t sub = 0; sub < 4; ++sub) {
                    int32_t dot = 0;
                    for (size_t l = 0; l < 16; ++l) {
                        const int32_t q = q6_scalar_q(ql, qh, ql_off, qh_off, sub, l0 + l);
                        const size_t xi = x_off + Q6_X_ADD[sub] + l0 + l;
                        dot += (q - 32) * static_cast<int32_t>(x_i8[xi]);
                    }
                    const float xs = x_scales[(x_off + Q6_X_ADD[sub] + l0) / QK];
                    acc += d * q6_scale(sc, sc_base + is + Q6_SC_OFF[sub]) * xs * static_cast<float>(dot);
                }
            }
            x_off += 128;
            ql_off += 64;
            qh_off += 32;
            sc_base += 8;
        }
    }
    return acc;
}

float dot_q6_k_row_q8k_scalar(std::span<const uint8_t> row_data,
                              std::span<const int8_t> x_i8,
                              std::span<const float> x_scales) {
    const size_t nb = row_data.size() / Q6_K_BLOCK_BYTES;
    check_q8_row(nb, x_i8, x_scales, 1, "dot_q6_k_row_q8k_scalar: activations shorter than the row");
    float acc = 0.0f;
    size_t x_off = 0;
    for (size_t bi = 0; bi < nb; ++bi) {
        const uint8_t* block = row_data.data() + bi * Q6_K_BLOCK_BYTES;
        const uint8_t* ql = block;
        const uint8_t* qh = block + 128;
        const uint8_t* sc = block + 192;
        const float d = f16_le_to_f32(block + 208);
        size_t ql_off = 0;
        size_t qh_off = 0;
        size_t sc_base = 0;
        size_t xo = x_off;
        int32_t isum = 0;
        for (size_t half = 0; half < QK_K / 128; ++half) {
            for (const size_t l0 : {size_t{0}, size_t{16}}) {
                const size_t is = l0 / 16;
                for (size_t sub = 0; sub < 4; ++sub) {
                    int32_t dot = 0;
                    for (size_t l = 0; l < 16; ++l) {
                        const int32_t q = q6_scalar_q(ql, qh, ql_off, qh_off, sub, l0 + l);
                        const size_t xi = xo + Q6_X_ADD[sub] + l0 + l;
                        dot += (q - 32) * static_cast<int32_t>(x_i8[xi]);
                    }
                    isum += detail::i8v(sc[sc_base + is + Q6_SC_OFF[sub]]) * dot;
                }
            }
            xo += 128;
            ql_off += 64;
            qh_off += 32;
            sc_base += 8;
        }
        acc += x_scales[bi] * d * static_cast<float>(isum);
        x_off += QK_K;
    }
    return acc;
}

#if SAPIENT_AARCH64
SAPIENT_TARGET_DOTPROD float dot_q6_k_row_q8_neon(std::span<const uint8_t> row_data,
                                                  std::span<const int8_t> x_i8,
                                                  std::span<const float> x_scales) {
    const size_t nb = row_data.size() / Q6_K_BLOCK_BYTES;
    check_q8_row(nb, x_i8, x_scales, QK_K / QK, "dot_q6_k_row_q8_neon: activations shorter than the row");
    float acc = 0.0f;
    size_t x_off = 0;
    for (size_t bi = 0; bi < nb; ++bi) {
        const uint8_t* block = row_data.data() + bi * Q6_K_BLOCK_BYTES;
        const uint8_t* ql = block;
        const uint8_t* qh = block + 128;
        const uint8_t* sc = block + 192;
        const float d = f16_le_to_f32(block + 208);
        size_t ql_off = 0;
        size_t qh_off = 0;
        size_t sc_base = 0;
        for (size_t half = 0; half < QK_K / 128; ++half) {
            for (const size_t l0 : {size_t{0}, size_t{16}}) {
                const size_t is = l0 / 16;
                uint8x16_t q[4];
                q6_unpack(ql, qh, ql_off, qh_off, l0, q);
                for (size_t sub = 0; sub < 4; ++sub) { // group!(q1,0,0) (q2,2,32) (q3,4,64) (q4,6,96)
                    const size_t xi = x_off + Q6_X_ADD[sub] + l0;
                    const int8x16_t xv = vld1q_s8(x_i8.data() + xi);
                    const int32_t dot = q6_group_dot(q[sub], xv);
                    const float xs = x_scales[xi / QK];
                    acc += d * q6_scale(sc, sc_base + is + Q6_SC_OFF[sub]) * xs * static_cast<float>(dot);
                }
            }
            x_off += 128;
            ql_off += 64;
            qh_off += 32;
            sc_base += 8;
        }
    }
    return acc;
}

SAPIENT_TARGET_DOTPROD float dot_q6_k_row_q8k_neon(std::span<const uint8_t> row_data,
                                                   std::span<const int8_t> x_i8,
                                                   std::span<const float> x_scales) {
    const size_t nb = row_data.size() / Q6_K_BLOCK_BYTES;
    check_q8_row(nb, x_i8, x_scales, 1, "dot_q6_k_row_q8k_neon: activations shorter than the row");
    float acc = 0.0f;
    size_t x_off = 0;
    for (size_t bi = 0; bi < nb; ++bi) {
        const uint8_t* block = row_data.data() + bi * Q6_K_BLOCK_BYTES;
        const uint8_t* ql = block;
        const uint8_t* qh = block + 128;
        const uint8_t* sc = block + 192;
        const float d = f16_le_to_f32(block + 208);
        size_t ql_off = 0;
        size_t qh_off = 0;
        size_t sc_base = 0;
        size_t xo = x_off;
        int32_t isum = 0;
        for (size_t half = 0; half < QK_K / 128; ++half) {
            for (const size_t l0 : {size_t{0}, size_t{16}}) {
                const size_t is = l0 / 16;
                uint8x16_t q[4];
                q6_unpack(ql, qh, ql_off, qh_off, l0, q);
                for (size_t sub = 0; sub < 4; ++sub) {
                    const size_t xi = xo + Q6_X_ADD[sub] + l0;
                    const int8x16_t xv = vld1q_s8(x_i8.data() + xi);
                    const int32_t dot = q6_group_dot(q[sub], xv);
                    isum += detail::i8v(sc[sc_base + is + Q6_SC_OFF[sub]]) * dot;
                }
            }
            xo += 128;
            ql_off += 64;
            qh_off += 32;
            sc_base += 8;
        }
        acc += x_scales[bi] * d * static_cast<float>(isum);
        x_off += QK_K;
    }
    return acc;
}

SAPIENT_TARGET_DOTPROD std::array<float, 4>
dot_q6_k_4rows_r4_q8_neon(std::span<const uint8_t> packed,
                          std::span<const int8_t> x_i8,
                          std::span<const float> x_scales) {
    const size_t nb = packed.size() / (4 * Q6_K_BLOCK_BYTES);
    check_q8_row(nb, x_i8, x_scales, QK_K / QK, "dot_q6_k_4rows_r4_q8_neon: activations shorter than the row");
    std::array<float, 4> acc{};
    size_t x_off = 0;
    for (size_t b = 0; b < nb; ++b) {
        const size_t gbase = b * 4 * Q6_K_BLOCK_BYTES;
        size_t ql_off = 0;
        size_t qh_off = 0;
        size_t sc_base = 0;
        size_t xo = x_off;
        for (size_t half = 0; half < QK_K / 128; ++half) {
            for (const size_t l0 : {size_t{0}, size_t{16}}) {
                const size_t is = l0 / 16;
                // Shared activation vectors + scales for the four 16-groups.
                int8x16_t xv[4];
                float xs[4];
                for (size_t sub = 0; sub < 4; ++sub) {
                    xv[sub] = vld1q_s8(x_i8.data() + xo + Q6_X_ADD[sub] + l0);
                    xs[sub] = x_scales[(xo + Q6_X_ADD[sub] + l0) / QK];
                }
                for (size_t r = 0; r < 4; ++r) {
                    const uint8_t* block = packed.data() + gbase + r * Q6_K_BLOCK_BYTES;
                    const uint8_t* sc = block + 192;
                    const float d = f16_le_to_f32(block + 208);
                    uint8x16_t q[4];
                    q6_unpack(block, block + 128, ql_off, qh_off, l0, q);
                    for (size_t sub = 0; sub < 4; ++sub) {
                        const int32_t dot = q6_group_dot(q[sub], xv[sub]);
                        acc[r] += d * q6_scale(sc, sc_base + is + Q6_SC_OFF[sub]) * xs[sub] * static_cast<float>(dot);
                    }
                }
            }
            xo += 128;
            ql_off += 64;
            qh_off += 32;
            sc_base += 8;
        }
        x_off += QK_K;
    }
    return acc;
}

SAPIENT_TARGET_DOTPROD std::array<float, 4>
dot_q6_k_4rows_r4_q8k_neon(std::span<const uint8_t> packed,
                           std::span<const int8_t> x_i8,
                           std::span<const float> x_scales) {
    const size_t nb = packed.size() / (4 * Q6_K_BLOCK_BYTES);
    const size_t nb_eff = std::min(nb, x_scales.size()); // .take(nb)
    check_len(x_i8.size(), nb_eff * QK_K, "dot_q6_k_4rows_r4_q8k_neon: x_i8 shorter than the row");
    std::array<float, 4> acc{};
    size_t x_off = 0;
    for (size_t b = 0; b < nb_eff; ++b) {
        const float db = x_scales[b];
        const size_t gbase = b * 4 * Q6_K_BLOCK_BYTES;
        size_t ql_off = 0;
        size_t qh_off = 0;
        size_t sc_base = 0;
        size_t xo = x_off;
        int32_t isum[4] = {0, 0, 0, 0};
        for (size_t half = 0; half < QK_K / 128; ++half) {
            for (const size_t l0 : {size_t{0}, size_t{16}}) {
                const size_t is = l0 / 16;
                int8x16_t xv[4];
                for (size_t sub = 0; sub < 4; ++sub)
                    xv[sub] = vld1q_s8(x_i8.data() + xo + Q6_X_ADD[sub] + l0);
                for (size_t r = 0; r < 4; ++r) {
                    const uint8_t* block = packed.data() + gbase + r * Q6_K_BLOCK_BYTES;
                    const uint8_t* sc = block + 192;
                    uint8x16_t q[4];
                    q6_unpack(block, block + 128, ql_off, qh_off, l0, q);
                    for (size_t sub = 0; sub < 4; ++sub) {
                        const int32_t dot = q6_group_dot(q[sub], xv[sub]);
                        isum[r] += detail::i8v(sc[sc_base + is + Q6_SC_OFF[sub]]) * dot;
                    }
                }
            }
            xo += 128;
            ql_off += 64;
            qh_off += 32;
            sc_base += 8;
        }
        for (size_t r = 0; r < 4; ++r) {
            const float d = f16_le_to_f32(packed.data() + gbase + r * Q6_K_BLOCK_BYTES + 208);
            acc[r] += db * d * static_cast<float>(isum[r]);
        }
        x_off += QK_K;
    }
    return acc;
}

namespace {
// The i8 (q − 32) groups of one row for one 16-lane position — shared by both SMMLA kernels.
SAPIENT_TARGET_I8MM inline void q6_unpack_m32(const uint8_t* block, size_t ql_off, size_t qh_off, size_t l0, int8x16_t qm[4]) {
    uint8x16_t q[4];
    q6_unpack(block, block + 128, ql_off, qh_off, l0, q);
    const int8x16_t m32 = vdupq_n_s8(32);
    for (size_t g = 0; g < 4; ++g)
        qm[g] = vsubq_s8(vreinterpretq_s8_u8(q[g]), m32);
}
// [w_r0_seg, w_r1_seg] × [x0_seg, x1_seg] per 8-byte half → lanes [r0·x0, r0·x1, r1·x0, r1·x1].
SAPIENT_TARGET_I8MM inline void q6_smmla_pair(int8x16_t w0, int8x16_t w1, int8x16_t x0, int8x16_t x1, int32_t dots[4]) {
    const int8x16_t wa = vtrn1q_s64_s8(w0, w1);
    const int8x16_t wb = vtrn2q_s64_s8(w0, w1);
    const int8x16_t xa = vtrn1q_s64_s8(x0, x1);
    const int8x16_t xb = vtrn2q_s64_s8(x0, x1);
    const int32x4_t d2 = smmla_s32(smmla_s32(vdupq_n_s32(0), wa, xa), wb, xb);
    dots[0] = vgetq_lane_s32(d2, 0);
    dots[1] = vgetq_lane_s32(d2, 1);
    dots[2] = vgetq_lane_s32(d2, 2);
    dots[3] = vgetq_lane_s32(d2, 3);
}
} // namespace

// Four R4 Q6_K rows × TWO per-32 int8 activation rows via `smmla` (quant.rs:2315-2444): per
// 16-element scale group, two trn shuffles + two smmla per weight-row pair; combine order is
// dot_q6_k_row_q8_neon's, so all 8 lanes are bit-identical to it.
SAPIENT_TARGET_I8MM std::array<std::array<float, 2>, 4>
dot_q6_k_4rows_r4_x2_smmla(std::span<const uint8_t> packed,
                           std::span<const int8_t> x0_i8,
                           std::span<const float> x0_scales,
                           std::span<const int8_t> x1_i8,
                           std::span<const float> x1_scales) {
    const size_t nb = packed.size() / (4 * Q6_K_BLOCK_BYTES);
    check_q8_row(nb, x0_i8, x0_scales, QK_K / QK, "dot_q6_k_4rows_r4_x2_smmla: x0 shorter than the row");
    check_q8_row(nb, x1_i8, x1_scales, QK_K / QK, "dot_q6_k_4rows_r4_x2_smmla: x1 shorter than the row");
    std::array<std::array<float, 2>, 4> acc{};
    size_t x_off = 0;
    for (size_t b = 0; b < nb; ++b) {
        const size_t gbase = b * 4 * Q6_K_BLOCK_BYTES;
        size_t ql_off = 0;
        size_t qh_off = 0;
        size_t sc_base = 0;
        size_t xo = x_off;
        for (size_t half = 0; half < QK_K / 128; ++half) {
            for (const size_t l0 : {size_t{0}, size_t{16}}) {
                const size_t is = l0 / 16;
                // Both activation rows' vectors for the four 16-groups, once.
                int8x16_t x0v[4];
                int8x16_t x1v[4];
                float xs0[4];
                float xs1[4];
                for (size_t g = 0; g < 4; ++g) {
                    x0v[g] = vld1q_s8(x0_i8.data() + xo + Q6_X_ADD[g] + l0);
                    x1v[g] = vld1q_s8(x1_i8.data() + xo + Q6_X_ADD[g] + l0);
                    xs0[g] = x0_scales[(xo + Q6_X_ADD[g] + l0) / QK];
                    xs1[g] = x1_scales[(xo + Q6_X_ADD[g] + l0) / QK];
                }
                for (size_t pair = 0; pair < 2; ++pair) {
                    const size_t r0 = pair * 2;
                    const size_t r1 = pair * 2 + 1;
                    const uint8_t* blk0 = packed.data() + gbase + r0 * Q6_K_BLOCK_BYTES;
                    const uint8_t* blk1 = packed.data() + gbase + r1 * Q6_K_BLOCK_BYTES;
                    int8x16_t qm0[4];
                    int8x16_t qm1[4];
                    q6_unpack_m32(blk0, ql_off, qh_off, l0, qm0);
                    q6_unpack_m32(blk1, ql_off, qh_off, l0, qm1);
                    for (size_t g = 0; g < 4; ++g) {
                        int32_t dots[4];
                        q6_smmla_pair(qm0[g], qm1[g], x0v[g], x1v[g], dots);
                        const size_t sc_off = is + 2 * g;
                        const size_t rows2[2] = {r0, r1};
                        for (size_t ri = 0; ri < 2; ++ri) {
                            const uint8_t* blk = packed.data() + gbase + rows2[ri] * Q6_K_BLOCK_BYTES;
                            const float d = f16_le_to_f32(blk + 208);
                            const float scv = q6_scale(blk + 192, sc_base + sc_off);
                            acc[rows2[ri]][0] += d * scv * xs0[g] * static_cast<float>(dots[ri * 2]);
                            acc[rows2[ri]][1] += d * scv * xs1[g] * static_cast<float>(dots[ri * 2 + 1]);
                        }
                    }
                }
            }
            xo += 128;
            ql_off += 64;
            qh_off += 32;
            sc_base += 8;
        }
        x_off += QK_K;
    }
    return acc;
}

// Same core, integer-domain combine of dot_q6_k_row_q8k_neon (quant.rs:2446-2562).
SAPIENT_TARGET_I8MM std::array<std::array<float, 2>, 4>
dot_q6_k_4rows_r4_x2_q8k_smmla(std::span<const uint8_t> packed,
                               std::span<const int8_t> x0_i8,
                               std::span<const float> x0_scales,
                               std::span<const int8_t> x1_i8,
                               std::span<const float> x1_scales) {
    const size_t nb = packed.size() / (4 * Q6_K_BLOCK_BYTES);
    const size_t nb_eff = std::min({nb, x0_scales.size(), x1_scales.size()}); // zip().take(nb)
    check_len(x0_i8.size(), nb_eff * QK_K, "dot_q6_k_4rows_r4_x2_q8k_smmla: x0 shorter than the row");
    check_len(x1_i8.size(), nb_eff * QK_K, "dot_q6_k_4rows_r4_x2_q8k_smmla: x1 shorter than the row");
    std::array<std::array<float, 2>, 4> acc{};
    size_t x_off = 0;
    for (size_t b = 0; b < nb_eff; ++b) {
        const float db0 = x0_scales[b];
        const float db1 = x1_scales[b];
        const size_t gbase = b * 4 * Q6_K_BLOCK_BYTES;
        size_t ql_off = 0;
        size_t qh_off = 0;
        size_t sc_base = 0;
        size_t xo = x_off;
        int32_t isum[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
        for (size_t half = 0; half < QK_K / 128; ++half) {
            for (const size_t l0 : {size_t{0}, size_t{16}}) {
                const size_t is = l0 / 16;
                int8x16_t x0v[4];
                int8x16_t x1v[4];
                for (size_t g = 0; g < 4; ++g) {
                    x0v[g] = vld1q_s8(x0_i8.data() + xo + Q6_X_ADD[g] + l0);
                    x1v[g] = vld1q_s8(x1_i8.data() + xo + Q6_X_ADD[g] + l0);
                }
                for (size_t pair = 0; pair < 2; ++pair) {
                    const size_t r0 = pair * 2;
                    const size_t r1 = pair * 2 + 1;
                    const uint8_t* blk0 = packed.data() + gbase + r0 * Q6_K_BLOCK_BYTES;
                    const uint8_t* blk1 = packed.data() + gbase + r1 * Q6_K_BLOCK_BYTES;
                    int8x16_t qm0[4];
                    int8x16_t qm1[4];
                    q6_unpack_m32(blk0, ql_off, qh_off, l0, qm0);
                    q6_unpack_m32(blk1, ql_off, qh_off, l0, qm1);
                    for (size_t g = 0; g < 4; ++g) {
                        int32_t dots[4];
                        q6_smmla_pair(qm0[g], qm1[g], x0v[g], x1v[g], dots);
                        const size_t sc_off = is + 2 * g;
                        const size_t rows2[2] = {r0, r1};
                        for (size_t ri = 0; ri < 2; ++ri) {
                            const uint8_t* blk = packed.data() + gbase + rows2[ri] * Q6_K_BLOCK_BYTES;
                            const int32_t scv = detail::i8v(blk[192 + sc_base + sc_off]);
                            isum[rows2[ri]][0] += scv * dots[ri * 2];
                            isum[rows2[ri]][1] += scv * dots[ri * 2 + 1];
                        }
                    }
                }
            }
            xo += 128;
            ql_off += 64;
            qh_off += 32;
            sc_base += 8;
        }
        for (size_t row = 0; row < 4; ++row) {
            const float d = f16_le_to_f32(packed.data() + gbase + row * Q6_K_BLOCK_BYTES + 208);
            acc[row][0] += db0 * d * static_cast<float>(isum[row][0]);
            acc[row][1] += db1 * d * static_cast<float>(isum[row][1]);
        }
        x_off += QK_K;
    }
    return acc;
}
#endif
```

Two things to check while transcribing: (1) `q6_unpack_m32` and `q6_smmla_pair` carry `SAPIENT_TARGET_I8MM` because they are inlined into i8mm functions and call `smmla_s32`/plain NEON — a plain-NEON helper without the attribute would also be legal for `q6_unpack_m32`, but the attributed form is what keeps `-mcpu=cortex-a53` honest; (2) the `d * scv * xs0[g] * dots` combine is left-associative exactly like Rust's `d * scv * xs0[g] * dots[ri*2] as f32`.

- [ ] **Step 5: Build, `--gtest_filter='Quant*'`** — all 24 Rust-named tests + the C++-only ones pass on this Mac (dotprod + i8mm present).
- [ ] **Step 6: The two codegen checks (rule 5).**
- [ ] **Step 7: Format after `git add`, `ctest --preset dev`, commit**

```bash
git commit -m "cpp(backends-cpu): kernels/quant — Q6_K W6A8/Q8_K scalar, SDOT, R4 and SMMLA x2 kernels (quant.rs port complete)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
Expected ctest: 185 + 6 = **191**.

---

### Task 6: The seven quantized arms of `matmul_nt`, `q8k_activations()`, the five matmul tests

**Files:**
- Modify: `cpp/libs/sapient-backends-cpu/include/sapient/backends_cpu/kernels/matmul.hpp` (header comment; `detail::q8k_activations`), `cpp/libs/sapient-backends-cpu/src/kernels/matmul.cpp` (the arms; the dispatcher's seven `case`s), `cpp/libs/sapient-backends-cpu/tests/matmul_test.cpp` (+5 Rust tests, +1 C++-only, −1 stub test)

**Interfaces:**
- Consumes: everything `quant.hpp` exports (Tasks 1–5), `cpu_features::{has_dotprod, has_i8mm}`, `parallel::{par_for, par_chunks_mut}`, `detail::{gemv_chunk, for_each_out_chunk}`, `Tensor::{to_f32_cow, quant_blocks, from_f32_vec, from_quant_bytes, reshape, to_f32_vec}`.
- Produces: `Result<Tensor> matmul_nt(x, w)` now dispatching Q4_0/Q8_0/Q4_K/Q4_K_R4/Q5_K/Q6_K/Q6_K_R4 to real kernels; aarch64 `matmul::detail::q8k_activations() -> bool` (Task 7's golden tests consume it).

- [ ] **Step 1: Tests** — in `tests/matmul_test.cpp` DELETE `TEST(Matmul, quantized_weight_dtypes_are_stubbed_until_plan_d)` and append (add `#include "sapient/backends_cpu/cpu_features.hpp"`, `#include "sapient/backends_cpu/kernels/quant.hpp"`, `<array>`, `<bit>` to the include block):

```cpp
namespace {
namespace quant = sapient::backends_cpu::kernels::quant;
using sapient::backends_cpu::cpu_features::has_dotprod;
using sapient::backends_cpu::cpu_features::has_i8mm;

// Rust: `.chunks_exact(32).flat_map(quantize_q8_0_block)` over a flat f32 array.
std::vector<uint8_t> q8_0_blocks(std::span<const float> w) {
    std::vector<uint8_t> out;
    for (size_t b = 0; b + 32 <= w.size(); b += 32) {
        const auto blk = quant::quantize_q8_0_block(w.subspan(b, 32));
        out.insert(out.end(), blk.begin(), blk.end());
    }
    return out;
}
Tensor quant_tensor(std::vector<uint8_t> bytes, Shape shape, DType dtype) {
    auto r = Tensor::from_quant_bytes(bytes, std::move(shape), dtype);
    if (!r) throw std::runtime_error(r.error().to_string());
    return std::move(*r);
}
uint32_t bits(float f) {
    return std::bit_cast<uint32_t>(f);
}
} // namespace

// ── quantized arms (plan D) ──────────────────────────────────────────────────

TEST(Matmul, matmul_nt_q4_0_matches_float) {
    // 4 output rows, 64 input features (64 is a multiple of 32 = two blocks/row).
    const size_t n_out = 4, k = 64;
    std::vector<float> w_f32(n_out * k);
    for (size_t i = 0; i < w_f32.size(); ++i)
        w_f32[i] = (::fmodf(static_cast<float>(i), 16.0f) - 8.0f) * 0.05f;
    std::vector<float> x_f32(k);
    for (size_t i = 0; i < k; ++i)
        x_f32[i] = static_cast<float>(i) * 0.01f - 0.3f;

    const auto w_t = f32(w_f32, Shape{n_out, k});
    const auto x_t = f32(x_f32, Shape{1, k});
    auto ref_out = matmul_nt(x_t, w_t);
    ASSERT_TRUE(ref_out.has_value()) << ref_out.error().to_string();
    const auto ref_data = ref_out->to_f32_vec();

    // Quantize each row to Q4_0.
    std::vector<uint8_t> w_blocks;
    for (size_t r = 0; r < n_out; ++r) {
        const auto row = quant::quantize_q4_0_row(std::span<const float>(w_f32).subspan(r * k, k));
        w_blocks.insert(w_blocks.end(), row.begin(), row.end());
    }
    const auto w_q = quant_tensor(w_blocks, Shape{n_out, k}, DType::Q4_0);
    auto quant_out = matmul_nt(x_t, w_q);
    ASSERT_TRUE(quant_out.has_value()) << quant_out.error().to_string();
    const auto quant_data = quant_out->to_f32_vec();
    ASSERT_EQ(ref_data.size(), quant_data.size());
    for (size_t i = 0; i < ref_data.size(); ++i)
        EXPECT_LT(::fabsf(ref_data[i] - quant_data[i]), 5e-3f)
            << "row " << i << ": ref=" << ref_data[i] << " quant=" << quant_data[i];
}

#if defined(__aarch64__) || defined(_M_ARM64)
TEST(Matmul, q8_0_gemm_path_matches_per_row_path) {
    if (!has_dotprod()) GTEST_SKIP() << "dotprod not available";
    // m = 16 triggers the blocked GEMM; compare each row against the m = 1 (per-row GEMV) path —
    // same kernel, same scales → bit-identical.
    const size_t m = 16, k = 96, n = 24;
    uint64_t seed = 0x8A8AULL;
    auto nf = [&seed]() {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        return (static_cast<float>(seed >> 40) / static_cast<float>(1ULL << 24)) * 2.0f - 1.0f;
    };
    std::vector<float> xv(m * k);
    for (float& v : xv)
        v = nf() * 0.5f;
    std::vector<float> wv(n * k);
    for (float& v : wv)
        v = nf() * 0.2f;
    const auto w_q8 = quant_tensor(q8_0_blocks(wv), Shape{n, k}, DType::Q8_0);

    auto full = matmul_nt(f32(xv, Shape{m, k}), w_q8);
    ASSERT_TRUE(full.has_value()) << full.error().to_string();
    const auto fd = full->to_f32_vec();
    for (size_t i = 0; i < m; ++i) {
        const std::vector<float> x_row(xv.begin() + static_cast<std::ptrdiff_t>(i * k),
                                       xv.begin() + static_cast<std::ptrdiff_t>((i + 1) * k));
        auto want = matmul_nt(f32(x_row, Shape{1, k}), w_q8);
        ASSERT_TRUE(want.has_value());
        const auto wd = want->to_f32_vec();
        for (size_t j = 0; j < n; ++j)
            EXPECT_EQ(bits(fd[i * n + j]), bits(wd[j]))
                << "row " << i << " col " << j << ": " << fd[i * n + j] << " vs " << wd[j];
    }
}
#endif

TEST(Matmul, matmul_nt_q8_0_matches_float) {
    // Larger k so several 32-blocks per row exercise the SDOT/NEON path, and an activation
    // outlier to stress per-block activation quantization.
    const size_t n_out = 8, k = 256;
    std::vector<float> w_f32(n_out * k);
    for (size_t i = 0; i < w_f32.size(); ++i)
        w_f32[i] = (static_cast<float>(i * 7 % 31) - 15.0f) * 0.03f;
    std::vector<float> x_f32(k);
    for (size_t i = 0; i < k; ++i)
        x_f32[i] = ::sinf(static_cast<float>(i) * 0.013f) * 0.4f;
    x_f32[100] = 25.0f; // outlier channel

    const auto w_q = quant_tensor(q8_0_blocks(w_f32), Shape{n_out, k}, DType::Q8_0);
    const auto x_t = f32(x_f32, Shape{1, k});
    // Reference: dequantize the SAME Q8_0 weights to f32, then exact f32 matmul.
    const auto w_ref = f32(w_q.to_f32_vec(), Shape{n_out, k});
    auto ref_out = matmul_nt(x_t, w_ref);
    ASSERT_TRUE(ref_out.has_value());
    const auto ref_data = ref_out->to_f32_vec();
    auto quant_out = matmul_nt(x_t, w_q);
    ASSERT_TRUE(quant_out.has_value()) << quant_out.error().to_string();
    const auto quant_data = quant_out->to_f32_vec();
    ASSERT_EQ(ref_data.size(), quant_data.size());
    for (size_t i = 0; i < ref_data.size(); ++i) {
        const float tol = 0.02f * ::fmaxf(::fabsf(ref_data[i]), 1.0f);
        EXPECT_LT(::fabsf(ref_data[i] - quant_data[i]), tol)
            << "row " << i << ": ref=" << ref_data[i] << " quant=" << quant_data[i];
    }
}

// Replicates the GGUF load path for Q8_0 weights: the tensor is built with the ggml dim order
// [in, out] and then reshaped to HF [out, in] (exactly what map_gguf_tensors_to_hf does).
TEST(Matmul, matmul_nt_q8_0_gguf_dimflip_matches_float) {
    const size_t out_features = 64, in_features = 128;
    std::vector<float> w_f32(out_features * in_features);
    for (size_t i = 0; i < w_f32.size(); ++i)
        w_f32[i] = (static_cast<float>(i * 13 % 29) - 14.0f) * 0.02f;
    std::vector<float> x_f32(in_features);
    for (size_t i = 0; i < in_features; ++i)
        x_f32[i] = ::cosf(static_cast<float>(i) * 0.02f) * 0.5f;
    const auto x_t = f32(x_f32, Shape{1, in_features});

    // ggml stores ne[0]=in contiguous, so its flat byte order == row-major [out, in] == w_f32.
    auto w_gguf = quant_tensor(q8_0_blocks(w_f32), Shape{in_features, out_features}, DType::Q8_0)
                      .reshape(Shape{out_features, in_features});
    ASSERT_TRUE(w_gguf.has_value()) << w_gguf.error().to_string();
    const auto w_ref = f32(w_gguf->to_f32_vec(), Shape{out_features, in_features});
    auto ref_out = matmul_nt(x_t, w_ref);
    ASSERT_TRUE(ref_out.has_value());
    const auto ref_data = ref_out->to_f32_vec();
    auto got = matmul_nt(x_t, *w_gguf);
    ASSERT_TRUE(got.has_value()) << got.error().to_string();
    const auto gd = got->to_f32_vec();
    for (size_t i = 0; i < ref_data.size(); ++i)
        EXPECT_LT(::fabsf(ref_data[i] - gd[i]), 0.02f * ::fmaxf(::fabsf(ref_data[i]), 1.0f))
            << "out " << i << ": ref=" << ref_data[i] << " quant=" << gd[i];
}

#if defined(__aarch64__) || defined(_M_ARM64)
// Panel-blocked SMMLA prefill must be bit-identical to per-row GEMV: m = 259 spans two full
// 128-row panels plus an odd 3-row remainder (pair loop + odd final row on every group).
TEST(Matmul, q4_k_r4_prefill_matches_per_row) {
    const size_t n = 8, k = 256, m = 259;
    std::vector<uint8_t> blocks(n * quant::Q4_K_BLOCK_BYTES);
    for (size_t i = 0; i < blocks.size(); ++i)
        blocks[i] = static_cast<uint8_t>((i * 131 + 7) % 251);
    for (size_t r = 0; r < n; ++r) {
        uint8_t* base = blocks.data() + r * quant::Q4_K_BLOCK_BYTES;
        sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(0.05f), base);     // d
        sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(0.03f), base + 2); // dmin
    }
    const auto packed = quant::repack_q4_k_rows4(blocks, n, k);
    const auto w_r4 = quant_tensor(packed, Shape{n, k}, DType::Q4_K_R4);

    std::vector<float> x_f32(m * k);
    for (size_t i = 0; i < x_f32.size(); ++i)
        x_f32[i] = (static_cast<float>(i * 37 % 97) - 48.0f) * 0.02f;
    auto panelled = matmul_nt(f32(x_f32, Shape{m, k}), w_r4);
    ASSERT_TRUE(panelled.has_value()) << panelled.error().to_string();
    const auto pd = panelled->to_f32_vec();
    for (size_t i = 0; i < m; ++i) {
        const std::vector<float> x_row(x_f32.begin() + static_cast<std::ptrdiff_t>(i * k),
                                       x_f32.begin() + static_cast<std::ptrdiff_t>((i + 1) * k));
        auto row_out = matmul_nt(f32(x_row, Shape{1, k}), w_r4);
        ASSERT_TRUE(row_out.has_value());
        const auto rd = row_out->to_f32_vec();
        for (size_t j = 0; j < n; ++j)
            EXPECT_EQ(bits(pd[i * n + j]), bits(rd[j]))
                << "row " << i << " col " << j << ": " << pd[i * n + j] << " vs " << rd[j];
    }
}
#endif

// C++-only: the seven parity-bound guard texts of the quantized arms (rule 7). byte_count
// truncates, so a [1, 48] Q4_0 tensor builds from one 18-byte block and still fails k % 32.
TEST(Matmul, quantized_guard_messages_match_rust) {
    auto q = [](size_t n, size_t k, DType dt, size_t nbytes) {
        return quant_tensor(std::vector<uint8_t>(nbytes, 0), Shape{n, k}, dt);
    };
    const auto x48 = f32(std::vector<float>(48, 0.0f), Shape{1, 48});
    EXPECT_EQ(matmul_nt(x48, q(1, 48, DType::Q4_0, 18)).error().to_string(),
              "Internal error: Q4_0 matmul_nt: k must be a multiple of the block size (32)");
    EXPECT_EQ(matmul_nt(x48, q(1, 48, DType::Q8_0, 34)).error().to_string(),
              "Internal error: Q8_0 matmul_nt: k must be a multiple of the block size (32)");
    const auto x128 = f32(std::vector<float>(128, 0.0f), Shape{1, 128});
    EXPECT_EQ(matmul_nt(x128, q(2, 128, DType::Q4_K, 144)).error().to_string(),
              "Internal error: Q4_K: k must be a multiple of 256");
    EXPECT_EQ(matmul_nt(x128, q(2, 128, DType::Q5_K, 176)).error().to_string(),
              "Internal error: Q5_K: k must be a multiple of 256");
    EXPECT_EQ(matmul_nt(x128, q(2, 128, DType::Q6_K, 210)).error().to_string(),
              "Internal error: Q6_K: k must be a multiple of 256");
    EXPECT_EQ(matmul_nt(x128, q(2, 128, DType::Q4_K_R4, 144)).error().to_string(),
              "Internal error: Q4_K_R4: k must be a multiple of 256 and rows a multiple of 4");
    EXPECT_EQ(matmul_nt(x128, q(2, 128, DType::Q6_K_R4, 210)).error().to_string(),
              "Internal error: Q6_K_R4: k must be a multiple of 256 and rows a multiple of 4");
    const auto x256 = f32(std::vector<float>(256, 0.0f), Shape{1, 256});
    EXPECT_EQ(matmul_nt(x256, q(2, 256, DType::Q4_K_R4, 2 * 144)).error().to_string(),
              "Internal error: Q4_K_R4: k must be a multiple of 256 and rows a multiple of 4");
}
```

- [ ] **Step 2: Build → the new tests fail** (`matmul_nt` still returns the plan-D error).

- [ ] **Step 3: `matmul.hpp`** — replace the header comment's `Plan D adds the seven quantized arms (today they return an explicit Error); plan E adds the spin-pool branch of `for_each_out_chunk`.` with `Plan D added the seven quantized arms (Q4_0/Q8_0/Q4_K/Q4_K_R4/Q5_K/Q6_K/Q6_K_R4 with Rust's runtime dispatch); plan E adds the spin-pool branch of `for_each_out_chunk`.`; in the `matmul_nt` doc comment replace `without expanding quantized weights (plan D)` with `without expanding quantized weights`; and add inside `namespace detail`:

```cpp
#if defined(__aarch64__) || defined(_M_ARM64)
/// Rust `q8k_activations()`: `SAPIENT_Q8K_ACT != "0"`, read ONCE per process (OnceLock twin),
/// default true — the Q8_K (per-256) activation format for the Q4_K/Q6_K SDOT/SMMLA paths;
/// `SAPIENT_Q8K_ACT=0` reverts to the per-32 W4A8/W6A8 format. aarch64-only, as in Rust.
bool q8k_activations();
#endif
```

- [ ] **Step 4: `matmul.cpp`** — add `#include "sapient/backends_cpu/kernels/quant.hpp"`, `<array>`, `<cstdlib>`, `<string_view>`; replace the grouped stub arm in `matmul_nt` with:

```cpp
    case DType::Q4_0:
        return matmul_nt_q4_0(x, w, m, k, n);
    case DType::Q8_0:
        return matmul_nt_q8_0(x, w, m, k, n);
    case DType::Q4_K:
        return matmul_nt_q4_k(x, w, m, k, n);
    case DType::Q4_K_R4:
        return matmul_nt_q4_k_r4(x, w, m, k, n);
    case DType::Q5_K:
        return matmul_nt_q5_k(x, w, m, k, n);
    case DType::Q6_K:
        return matmul_nt_q6_k(x, w, m, k, n);
    case DType::Q6_K_R4:
        return matmul_nt_q6_k_r4(x, w, m, k, n);
```

and add, inside the existing anonymous namespace AFTER `matmul_nt_float` (the arms use `detail::gemv_chunk`/`for_each_out_chunk`, which are declared in the header, so order within the TU is free), the block below; `detail::q8k_activations` goes into the existing `namespace detail` section:

```cpp
// ── quantized arms (matmul.rs:538-1170) ──────────────────────────────────────

namespace quant = sapient::backends_cpu::kernels::quant;

// gemv_parallel!: n rows of a quantized GEMV, `dot(w_row_bytes, x_row)` per row, rows batched per
// task by gemv_chunk (matmul.rs:538-548).
template <class Dot>
void gemv_parallel(std::span<float> out_row,
                   size_t n,
                   size_t row_bytes,
                   std::span<const uint8_t> w_blocks,
                   std::span<const float> x_row,
                   Dot dot) {
    const size_t chunk = detail::gemv_chunk(n);
    detail::for_each_out_chunk(out_row, chunk, [&](size_t chunk_idx, std::span<float> cs) {
        for (size_t local = 0; local < cs.size(); ++local) {
            const size_t j = chunk_idx * chunk + local;
            cs[local] = dot(w_blocks.subspan(j * row_bytes, row_bytes), x_row);
        }
    });
}

// Rust's slice panics on `x_data[i*k..(i+1)*k]` / `w_blocks[j*row_bytes..]`, checked once.
void check_quant_operands(std::span<const float> x_data,
                          std::span<const uint8_t> w_blocks,
                          size_t m,
                          size_t k,
                          size_t n,
                          size_t row_bytes) {
    if (x_data.size() < m * k) sapient::core::panic("matmul_nt: x shorter than [m, k]");
    if (w_blocks.size() < n * row_bytes)
        sapient::core::panic("matmul_nt: quantized weight buffer shorter than [n, k]");
}

#if defined(__aarch64__) || defined(_M_ARM64)
// Per-32 int8 activations + their sums, or the Q8_K format — the `(x_i8, x_scales, x_sums)`
// triple every Q4_K SDOT path builds per activation row (matmul.rs:733-744, 811-815, 884-890).
quant::Q8kRow quantize_q4k_activations(std::span<const float> row, bool q8k) {
    if (q8k) return quant::quantize_row_to_q8k(row);
    auto q = quant::quantize_row_to_i8_blocks(row);
    auto sums = quant::i8_block_sums(q.q);
    return quant::Q8kRow{std::move(q.q), std::move(q.scales), std::move(sums)};
}
// The `(x_i8, x_scales)` pair of the Q6_K SDOT paths (Q8_K sums dropped) (matmul.rs:995-1003).
quant::I8Blocks quantize_q6k_activations(std::span<const float> row, bool q8k) {
    if (!q8k) return quant::quantize_row_to_i8_blocks(row);
    auto r = quant::quantize_row_to_q8k(row);
    return quant::I8Blocks{std::move(r.q), std::move(r.scales)};
}
#endif

Result<Tensor> matmul_nt_q4_0(const Tensor& x, const Tensor& w, size_t m, size_t k, size_t n) {
    if (k % quant::QK != 0)
        return tl::unexpected(
            Error::internal("Q4_0 matmul_nt: k must be a multiple of the block size (32)"));
    const auto x_cow = x.to_f32_cow();
    const auto x_data = x_cow.get();
    const auto w_blocks = w.quant_blocks();
    const size_t row_bytes = k / quant::QK * quant::Q4_0_BLOCK_BYTES;
    check_quant_operands(x_data, w_blocks, m, k, n, row_bytes);
    std::vector<float> out(m * n, 0.0f);
    for (size_t i = 0; i < m; ++i)
        gemv_parallel(std::span<float>(out).subspan(i * n, n), n, row_bytes, w_blocks,
                      x_data.subspan(i * k, k), quant::dot_q4_0_row_f32);
    return Tensor::from_f32_vec(std::move(out), Shape{m, n});
}

Result<Tensor> matmul_nt_q8_0(const Tensor& x, const Tensor& w, size_t m, size_t k, size_t n) {
    if (k % quant::QK != 0)
        return tl::unexpected(
            Error::internal("Q8_0 matmul_nt: k must be a multiple of the block size (32)"));
    const auto x_cow = x.to_f32_cow();
    const auto x_data = x_cow.get();
    const auto w_blocks = w.quant_blocks();
    const size_t row_bytes = k / quant::QK * quant::Q8_0_BLOCK_BYTES;
    check_quant_operands(x_data, w_blocks, m, k, n, row_bytes);
    std::vector<float> out(m * n, 0.0f);

#if defined(__aarch64__) || defined(_M_ARM64)
    // ── SDOT path (aarch64 dotprod): activations quantized to per-32 int8 ONCE per row ──
    if (cpu_features::has_dotprod()) {
        // Blocked W8A8 GEMM (m ≥ 8: prefill / vision towers): all rows quantize once, ONE parallel
        // region over weight-row chunks, output built [n, m] then flipped. Same kernel, same
        // scales as the per-row path → bit-identical (matmul.rs:608-650).
        if (m >= 8) {
            const size_t bpr = k / quant::QK; // activation blocks per row
            std::vector<int8_t> x_i8(m * k, 0);
            std::vector<float> x_scales(m * bpr, 0.0f);
            parallel::par_for(m, [&](size_t i) {
                const auto r = quant::quantize_row_to_i8_blocks(x_data.subspan(i * k, k));
                std::copy(r.q.begin(), r.q.end(), x_i8.begin() + static_cast<std::ptrdiff_t>(i * k));
                std::copy(r.scales.begin(), r.scales.end(),
                          x_scales.begin() + static_cast<std::ptrdiff_t>(i * bpr));
            });
            std::vector<float> out_t(n * m, 0.0f); // [n, m]
            const size_t wchunk = detail::gemv_chunk(n);
            parallel::par_chunks_mut(out_t, wchunk * m, [&](size_t ci, std::span<float> oc) {
                const size_t j0 = ci * wchunk;
                for (size_t jl = 0; jl * m < oc.size(); ++jl) { // oc.chunks_mut(m)
                    const size_t j = j0 + jl;
                    const auto wrow = w_blocks.subspan(j * row_bytes, row_bytes);
                    for (size_t i = 0; i < m; ++i)
                        oc[jl * m + i] = quant::dot_q8_0_row_sdot(
                            wrow, std::span<const int8_t>(x_i8).subspan(i * k, k),
                            std::span<const float>(x_scales).subspan(i * bpr, bpr));
                }
            });
            // Transpose [n, m] → [m, n] (parallel over output rows).
            parallel::par_chunks_mut(out, n, [&](size_t i, std::span<float> orow) {
                for (size_t j = 0; j < orow.size(); ++j)
                    orow[j] = out_t[j * m + i];
            });
            return Tensor::from_f32_vec(std::move(out), Shape{m, n});
        }
        for (size_t i = 0; i < m; ++i) {
            // Per-block activation scales — a single per-row scale is destroyed by outlier
            // activation channels and yields incoherent output.
            const auto xq = quant::quantize_row_to_i8_blocks(x_data.subspan(i * k, k));
            const size_t chunk = detail::gemv_chunk(n);
            detail::for_each_out_chunk(std::span<float>(out).subspan(i * n, n), chunk,
                                       [&](size_t ci, std::span<float> cs) {
                                           for (size_t local = 0; local < cs.size(); ++local) {
                                               const size_t j = ci * chunk + local;
                                               cs[local] = quant::dot_q8_0_row_sdot(
                                                   w_blocks.subspan(j * row_bytes, row_bytes),
                                                   xq.q, xq.scales);
                                           }
                                       });
        }
        return Tensor::from_f32_vec(std::move(out), Shape{m, n});
    }
#endif
    // ── Fallback: NEON widening or AVX2 ──
    for (size_t i = 0; i < m; ++i)
        gemv_parallel(std::span<float>(out).subspan(i * n, n), n, row_bytes, w_blocks,
                      x_data.subspan(i * k, k), quant::dot_q8_0_row_f32);
    return Tensor::from_f32_vec(std::move(out), Shape{m, n});
}

// Q4_K_R4 (row-interleaved) GEMV: weight rows come in groups of 4 whose super-blocks are
// block-interleaved into one contiguous stream (matmul.rs:711-860).
Result<Tensor> matmul_nt_q4_k_r4(const Tensor& x, const Tensor& w, size_t m, size_t k, size_t n) {
    if (k % 256 != 0 || n % 4 != 0)
        return tl::unexpected(
            Error::internal("Q4_K_R4: k must be a multiple of 256 and rows a multiple of 4"));
    const auto x_cow = x.to_f32_cow();
    const auto x_data = x_cow.get();
    const auto w_blocks = w.quant_blocks();
    const size_t row_bytes = k / 256 * quant::Q4_K_BLOCK_BYTES;
    check_quant_operands(x_data, w_blocks, m, k, n, row_bytes);
    std::vector<float> out(m * n, 0.0f);

#if defined(__aarch64__) || defined(_M_ARM64)
    const size_t group_bytes = 4 * row_bytes;
    // ── i8mm SMMLA prefill path (m ≥ 2, ARMv8.6): two activation rows per pass through each
    // weight group; output built group-major (transposed) so tasks own contiguous chunks ──
    if (m >= 2 && cpu_features::has_i8mm()) {
        const bool q8k = detail::q8k_activations();
        std::vector<quant::Q8kRow> quantized;
        quantized.reserve(m);
        for (size_t i = 0; i < m; ++i)
            quantized.push_back(quantize_q4k_activations(x_data.subspan(i * k, k), q8k));
        const size_t groups = n / 4;
        std::vector<float> out_t(n * m, 0.0f); // [group-rows][m]
        parallel::par_chunks_mut(out_t, 4 * m, [&](size_t g, std::span<float> chunk) {
            const auto group = w_blocks.subspan(g * group_bytes, group_bytes);
            size_t xi = 0;
            while (xi + 2 <= m) {
                const auto& a = quantized[xi];
                const auto& b = quantized[xi + 1];
                const auto v = q8k ? quant::dot_q4_k_4rows_r4_x2_q8k_smmla(group, a.q, a.scales, a.sums, b.q, b.scales, b.sums)
                                   : quant::dot_q4_k_4rows_r4_x2_smmla(group, a.q, a.scales, a.sums, b.q, b.scales, b.sums);
                for (size_t r = 0; r < 4; ++r) {
                    chunk[r * m + xi] = v[r][0];
                    chunk[r * m + xi + 1] = v[r][1];
                }
                xi += 2;
            }
            if (xi < m) {
                const auto& a = quantized[xi];
                const auto v = q8k ? quant::dot_q4_k_4rows_r4_q8k_neon(group, a.q, a.scales, a.sums)
                                   : quant::dot_q4_k_4rows_r4_neon(group, a.q, a.scales, a.sums);
                for (size_t r = 0; r < 4; ++r)
                    chunk[r * m + xi] = v[r];
            }
        });
        for (size_t g = 0; g < groups; ++g)
            for (size_t r = 0; r < 4; ++r)
                for (size_t i = 0; i < m; ++i)
                    out[i * n + g * 4 + r] = out_t[(g * 4 + r) * m + i];
        return Tensor::from_f32_vec(std::move(out), Shape{m, n});
    }

    // ── SDOT decode path: one contiguous stream per 4-row group ──
    if (cpu_features::has_dotprod()) {
        const bool q8k = detail::q8k_activations();
        for (size_t i = 0; i < m; ++i) {
            const auto xq = quantize_q4k_activations(x_data.subspan(i * k, k), q8k);
            const size_t gchunk = std::max<size_t>(detail::gemv_chunk(n) / 4, 1);
            detail::for_each_out_chunk(
                std::span<float>(out).subspan(i * n, n), gchunk * 4,
                [&](size_t ci, std::span<float> cs) {
                    const size_t g0 = ci * gchunk;
                    for (size_t gl = 0; gl * 4 < cs.size(); ++gl) { // cs.chunks_mut(4)
                        const size_t g = g0 + gl;
                        const auto group = w_blocks.subspan(g * group_bytes, group_bytes);
                        const auto v = q8k ? quant::dot_q4_k_4rows_r4_q8k_neon(group, xq.q, xq.scales, xq.sums)
                                           : quant::dot_q4_k_4rows_r4_neon(group, xq.q, xq.scales, xq.sums);
                        const auto slots = cs.subspan(gl * 4, std::min<size_t>(4, cs.size() - gl * 4));
                        std::copy_n(v.begin(), slots.size(), slots.begin());
                    }
                });
        }
        return Tensor::from_f32_vec(std::move(out), Shape{m, n});
    }
#endif
    // Portable fallback (tests / x86): de-interleave each group's rows and use the scalar W4A8
    // dot — per-32 activations, never Q8_K (matmul.rs:841-859).
    for (size_t i = 0; i < m; ++i) {
        const auto xq = quant::quantize_row_to_i8_blocks(x_data.subspan(i * k, k));
        const auto x_sums = quant::i8_block_sums(xq.q);
        const size_t nb = k / 256;
        std::vector<uint8_t> row_buf(row_bytes, 0);
        for (size_t g = 0; g < n / 4; ++g)
            for (size_t r = 0; r < 4; ++r) {
                for (size_t b = 0; b < nb; ++b) {
                    const size_t src = (g * 4 * nb + b * 4 + r) * quant::Q4_K_BLOCK_BYTES;
                    std::copy_n(w_blocks.data() + src, quant::Q4_K_BLOCK_BYTES,
                                row_buf.data() + b * quant::Q4_K_BLOCK_BYTES);
                }
                out[i * n + g * 4 + r] = quant::dot_q4_k_row_q8_scalar(row_buf, xq.q, xq.scales, x_sums);
            }
    }
    return Tensor::from_f32_vec(std::move(out), Shape{m, n});
}

Result<Tensor> matmul_nt_q4_k(const Tensor& x, const Tensor& w, size_t m, size_t k, size_t n) {
    if (k % 256 != 0) return tl::unexpected(Error::internal("Q4_K: k must be a multiple of 256"));
    const auto x_cow = x.to_f32_cow();
    const auto x_data = x_cow.get();
    const auto w_blocks = w.quant_blocks();
    const size_t row_bytes = k / 256 * quant::Q4_K_BLOCK_BYTES;
    check_quant_operands(x_data, w_blocks, m, k, n, row_bytes);
    std::vector<float> out(m * n, 0.0f);

#if defined(__aarch64__) || defined(_M_ARM64)
    // ── W4A8 SDOT path: 4 weight rows share one pass over the activations; remainder rows use
    // the single-row kernel (per-row results bit-identical either way) (matmul.rs:879-929) ──
    if (cpu_features::has_dotprod()) {
        const bool q8k = detail::q8k_activations();
        for (size_t i = 0; i < m; ++i) {
            const auto xq = quantize_q4k_activations(x_data.subspan(i * k, k), q8k);
            const size_t chunk = detail::gemv_chunk(n);
            detail::for_each_out_chunk(
                std::span<float>(out).subspan(i * n, n), chunk, [&](size_t ci, std::span<float> cs) {
                    const size_t start = ci * chunk;
                    size_t local = 0;
                    while (local + 4 <= cs.size()) {
                        const size_t j = start + local;
                        const std::array<std::span<const uint8_t>, 4> rows = {
                            w_blocks.subspan(j * row_bytes, row_bytes),
                            w_blocks.subspan((j + 1) * row_bytes, row_bytes),
                            w_blocks.subspan((j + 2) * row_bytes, row_bytes),
                            w_blocks.subspan((j + 3) * row_bytes, row_bytes)};
                        const auto v = q8k ? quant::dot_q4_k_4rows_q8k_neon(rows, xq.q, xq.scales, xq.sums)
                                           : quant::dot_q4_k_4rows_q8_neon(rows, xq.q, xq.scales, xq.sums);
                        std::copy_n(v.begin(), 4, cs.begin() + static_cast<std::ptrdiff_t>(local));
                        local += 4;
                    }
                    for (; local < cs.size(); ++local) {
                        const size_t j = start + local;
                        const auto row = w_blocks.subspan(j * row_bytes, row_bytes);
                        cs[local] = q8k ? quant::dot_q4_k_row_q8k_neon(row, xq.q, xq.scales, xq.sums)
                                        : quant::dot_q4_k_row_q8_neon(row, xq.q, xq.scales, xq.sums);
                    }
                });
        }
        return Tensor::from_f32_vec(std::move(out), Shape{m, n});
    }
#endif
    for (size_t i = 0; i < m; ++i)
        gemv_parallel(std::span<float>(out).subspan(i * n, n), n, row_bytes, w_blocks,
                      x_data.subspan(i * k, k), quant::dot_q4_k_row_f32);
    return Tensor::from_f32_vec(std::move(out), Shape{m, n});
}

Result<Tensor> matmul_nt_q5_k(const Tensor& x, const Tensor& w, size_t m, size_t k, size_t n) {
    if (k % 256 != 0) return tl::unexpected(Error::internal("Q5_K: k must be a multiple of 256"));
    const auto x_cow = x.to_f32_cow();
    const auto x_data = x_cow.get();
    const auto w_blocks = w.quant_blocks();
    const size_t row_bytes = k / 256 * quant::Q5_K_BLOCK_BYTES;
    check_quant_operands(x_data, w_blocks, m, k, n, row_bytes);
    std::vector<float> out(m * n, 0.0f);
    // No SIMD/dotprod branch at all — Rust has none for Q5_K.
    for (size_t i = 0; i < m; ++i)
        gemv_parallel(std::span<float>(out).subspan(i * n, n), n, row_bytes, w_blocks,
                      x_data.subspan(i * k, k), quant::dot_q5_k_row_f32);
    return Tensor::from_f32_vec(std::move(out), Shape{m, n});
}

// Q6_K_R4 (row-interleaved) GEMV — same scheme as matmul_nt_q4_k_r4 (matmul.rs:971-1109).
Result<Tensor> matmul_nt_q6_k_r4(const Tensor& x, const Tensor& w, size_t m, size_t k, size_t n) {
    if (k % 256 != 0 || n % 4 != 0)
        return tl::unexpected(
            Error::internal("Q6_K_R4: k must be a multiple of 256 and rows a multiple of 4"));
    const auto x_cow = x.to_f32_cow();
    const auto x_data = x_cow.get();
    const auto w_blocks = w.quant_blocks();
    const size_t row_bytes = k / 256 * quant::Q6_K_BLOCK_BYTES;
    check_quant_operands(x_data, w_blocks, m, k, n, row_bytes);
    std::vector<float> out(m * n, 0.0f);

#if defined(__aarch64__) || defined(_M_ARM64)
    const size_t group_bytes = 4 * row_bytes;
    // ── i8mm SMMLA prefill path (m ≥ 2) ──
    if (m >= 2 && cpu_features::has_i8mm()) {
        const bool q8k = detail::q8k_activations();
        std::vector<quant::I8Blocks> quantized;
        quantized.reserve(m);
        for (size_t i = 0; i < m; ++i)
            quantized.push_back(quantize_q6k_activations(x_data.subspan(i * k, k), q8k));
        const size_t groups = n / 4;
        std::vector<float> out_t(n * m, 0.0f); // [group-rows][m]
        parallel::par_chunks_mut(out_t, 4 * m, [&](size_t g, std::span<float> chunk) {
            const auto group = w_blocks.subspan(g * group_bytes, group_bytes);
            size_t xi = 0;
            while (xi + 2 <= m) {
                const auto& a = quantized[xi];
                const auto& b = quantized[xi + 1];
                const auto v = q8k ? quant::dot_q6_k_4rows_r4_x2_q8k_smmla(group, a.q, a.scales, b.q, b.scales)
                                   : quant::dot_q6_k_4rows_r4_x2_smmla(group, a.q, a.scales, b.q, b.scales);
                for (size_t r = 0; r < 4; ++r) {
                    chunk[r * m + xi] = v[r][0];
                    chunk[r * m + xi + 1] = v[r][1];
                }
                xi += 2;
            }
            if (xi < m) {
                const auto& a = quantized[xi];
                const auto v = q8k ? quant::dot_q6_k_4rows_r4_q8k_neon(group, a.q, a.scales)
                                   : quant::dot_q6_k_4rows_r4_q8_neon(group, a.q, a.scales);
                for (size_t r = 0; r < 4; ++r)
                    chunk[r * m + xi] = v[r];
            }
        });
        for (size_t g = 0; g < groups; ++g)
            for (size_t r = 0; r < 4; ++r)
                for (size_t i = 0; i < m; ++i)
                    out[i * n + g * 4 + r] = out_t[(g * 4 + r) * m + i];
        return Tensor::from_f32_vec(std::move(out), Shape{m, n});
    }

    // ── NEON decode path: W6A8/Q8_K when dotprod is present, f32 activations otherwise ──
    {
        const bool dotprod = cpu_features::has_dotprod();
        for (size_t i = 0; i < m; ++i) {
            const auto x_row = x_data.subspan(i * k, k);
            const bool q8k = detail::q8k_activations();
            std::optional<quant::I8Blocks> quantized;
            if (dotprod) quantized = quantize_q6k_activations(x_row, q8k);
            const size_t gchunk = std::max<size_t>(detail::gemv_chunk(n) / 4, 1);
            detail::for_each_out_chunk(
                std::span<float>(out).subspan(i * n, n), gchunk * 4,
                [&](size_t ci, std::span<float> cs) {
                    const size_t g0 = ci * gchunk;
                    for (size_t gl = 0; gl * 4 < cs.size(); ++gl) {
                        const size_t g = g0 + gl;
                        const auto group = w_blocks.subspan(g * group_bytes, group_bytes);
                        std::array<float, 4> v{};
                        if (quantized.has_value())
                            v = q8k ? quant::dot_q6_k_4rows_r4_q8k_neon(group, quantized->q, quantized->scales)
                                    : quant::dot_q6_k_4rows_r4_q8_neon(group, quantized->q, quantized->scales);
                        else
                            v = quant::dot_q6_k_4rows_r4_neon(group, x_row); // f32 activations
                        const auto slots = cs.subspan(gl * 4, std::min<size_t>(4, cs.size() - gl * 4));
                        std::copy_n(v.begin(), slots.size(), slots.begin());
                    }
                });
        }
        return Tensor::from_f32_vec(std::move(out), Shape{m, n});
    }
#else
    // Portable fallback (Rust's #[allow(unreachable_code)] block): de-interleave each group and
    // use the f32 dot (matmul.rs:1090-1108).
    for (size_t i = 0; i < m; ++i) {
        const auto x_row = x_data.subspan(i * k, k);
        const size_t nb = k / 256;
        std::vector<uint8_t> row_buf(row_bytes, 0);
        for (size_t g = 0; g < n / 4; ++g)
            for (size_t r = 0; r < 4; ++r) {
                for (size_t b = 0; b < nb; ++b) {
                    const size_t src = (g * 4 * nb + b * 4 + r) * quant::Q6_K_BLOCK_BYTES;
                    std::copy_n(w_blocks.data() + src, quant::Q6_K_BLOCK_BYTES,
                                row_buf.data() + b * quant::Q6_K_BLOCK_BYTES);
                }
                out[i * n + g * 4 + r] = quant::dot_q6_k_row_f32(row_buf, x_row);
            }
    }
    return Tensor::from_f32_vec(std::move(out), Shape{m, n});
#endif
}

Result<Tensor> matmul_nt_q6_k(const Tensor& x, const Tensor& w, size_t m, size_t k, size_t n) {
    if (k % 256 != 0) return tl::unexpected(Error::internal("Q6_K: k must be a multiple of 256"));
    const auto x_cow = x.to_f32_cow();
    const auto x_data = x_cow.get();
    const auto w_blocks = w.quant_blocks();
    const size_t row_bytes = k / 256 * quant::Q6_K_BLOCK_BYTES;
    check_quant_operands(x_data, w_blocks, m, k, n, row_bytes);
    std::vector<float> out(m * n, 0.0f);

#if defined(__aarch64__) || defined(_M_ARM64)
    // ── W6A8 SDOT path: one `sdot` per 16-element scale group (matmul.rs:1126-1153) ──
    if (cpu_features::has_dotprod()) {
        const bool q8k = detail::q8k_activations();
        for (size_t i = 0; i < m; ++i) {
            const auto xq = quantize_q6k_activations(x_data.subspan(i * k, k), q8k);
            const size_t chunk = detail::gemv_chunk(n);
            detail::for_each_out_chunk(
                std::span<float>(out).subspan(i * n, n), chunk, [&](size_t ci, std::span<float> cs) {
                    for (size_t local = 0; local < cs.size(); ++local) {
                        const size_t j = ci * chunk + local;
                        const auto row = w_blocks.subspan(j * row_bytes, row_bytes);
                        cs[local] = q8k ? quant::dot_q6_k_row_q8k_neon(row, xq.q, xq.scales)
                                        : quant::dot_q6_k_row_q8_neon(row, xq.q, xq.scales);
                    }
                });
        }
        return Tensor::from_f32_vec(std::move(out), Shape{m, n});
    }
#endif
    for (size_t i = 0; i < m; ++i)
        gemv_parallel(std::span<float>(out).subspan(i * n, n), n, row_bytes, w_blocks,
                      x_data.subspan(i * k, k), quant::dot_q6_k_row_f32);
    return Tensor::from_f32_vec(std::move(out), Shape{m, n});
}
```

and in `namespace detail`:

```cpp
#if defined(__aarch64__) || defined(_M_ARM64)
bool q8k_activations() {
    // OnceLock twin: SAPIENT_Q8K_ACT read once; `v != "0"`, unset → true (matmul.rs:458-470).
    static const bool on = [] {
        const char* v = std::getenv("SAPIENT_Q8K_ACT");
        return v == nullptr || std::string_view(v) != "0";
    }();
    return on;
}
#endif
```

Notes for the transcription: `w.quant_blocks()` is the C++ `as_quant_blocks()` (bounded by `byte_count`); `x_data.subspan(i * k, k)` after `check_quant_operands` is in bounds; every lambda handed to `for_each_out_chunk`/`par_chunks_mut` writes only its own `cs`/`chunk`/`orow` (disjoint by the partition contract) and reads shared inputs — no gtest assertions inside them; `quantized` in the Q6_K_R4 decode path is `std::optional` because Rust's `dotprod.then(...)` is one; the Q4_K_R4 fallback is reachable on a non-dotprod aarch64 (Rust too), the Q6_K_R4 one is not (Rust's `#[allow(unreachable_code)]`), hence `#else`.

- [ ] **Step 5: Build, run `--gtest_filter='Matmul*'`** — expected: all pass (this Mac takes the dotprod/i8mm paths).

- [ ] **Step 6: x86 codegen check of matmul.cpp** (it needs `tl::expected`'s include dir):
`cd cpp && clang++ -std=c++20 --target=x86_64-apple-macos -Wall -Wextra -Wpedantic -Wshadow -Werror -ffp-contract=off -fno-math-errno -Ilibs/sapient-core/include -Ilibs/sapient-backends-cpu/include -I$(find build/dev/_deps -type d -path '*expected*/include' | head -1) -c libs/sapient-backends-cpu/src/kernels/matmul.cpp -o /dev/null`. Expected: compiles (the x86 build sees only the fallback arms and no `q8k_activations`).

- [ ] **Step 7: Format after `git add`, `ctest --preset dev`, commit**

```bash
git commit -m "cpp(backends-cpu): matmul_nt quantized arms — Q4_0/Q8_0/Q4_K/Q4_K_R4/Q5_K/Q6_K/Q6_K_R4 with Rust's runtime dispatch

Replaces plan C's plan-D stub arm (and its test). SDOT/SMMLA paths behind cpu_features, the
SAPIENT_Q8K_ACT knob read once (OnceLock twin), m>=8 blocked Q8_0 GEMM, m>=2 SMMLA prefill,
portable fallbacks; five Rust matmul tests by name incl. the two to_bits gates.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
Expected ctest: 191 − 1 + 6 = **196**.

---

### Task 7: Plan-D golden cases (`dump_kernels.rs` + `--q8k-off`), `golden_dump.sh` second pass, `golden_quant_test.cpp`, the odd-length dense cases

**Files:**
- Modify: `crates/sapient-backends/cpu/examples/dump_kernels.rs` (doc comment; `build_cases(seed, q8k_off)`; 20 new cases appended at the end; `--q8k-off` flag), `cpp/tests/parity/golden_dump.sh` (second pass), `cpp/libs/sapient-backends-cpu/CMakeLists.txt` (`tests/golden_quant_test.cpp`; the `q8k_off` ctest entry), `cpp/libs/sapient-backends-cpu/tests/golden_kernels_test.cpp` (+3 tests; header comment)
- Create: `cpp/libs/sapient-backends-cpu/tests/golden_quant_test.cpp`

**Interfaces:**
- Consumes: `sapient::testing::{bit_identical, exact_equal<T>, SAPIENT_GOLDEN_CASE, GoldenCase::get(name).as<T>()}`, all `quant::` entry points, `matmul::matmul_nt`, `matmul::detail::q8k_activations` (aarch64), `Tensor::{from_quant_bytes, from_f16_bytes, from_f32_vec}`.
- Produces: the 63-case default dump + the 12-case `_q8k_off` dump; 42 `GoldenQuant.*` tests; 3 more `GoldenKernels.*` tests; the `sapient_backends_cpu_tests.q8k_off` ctest entry.

- [ ] **Step 1: `dump_kernels.rs`** — (a) doc comment: after the "Plan C (dense kernels)" paragraph add

```rust
//! Plan D (quantized kernels): `quantize_row_to_i8_blocks`, `i8_block_sums`, `quantize_row_to_q8k`,
//! `repack_q{4,6}_k_rows4`, `matmul_nt_{q4_0_m{1,3},q5_k_m1,q8_0_m8,q4_k_r4_m{1,2,3,8},q6_k_r4_m{1,2,3,8}}`,
//! and the plan-C carry-overs `matmul_nt_f32_m1_k519`, `matmul_nt_f16_m1_k67`, `attention_decode_hd10`
//! (SIMD body + scalar tail). `--q8k-off` (run under `SAPIENT_Q8K_ACT=0`) writes ONLY the twelve
//! knob-sensitive `matmul_nt_{q4_k,q6_k,q4_k_r4,q6_k_r4}_m*` cases, renamed `*_q8k_off`; every RNG
//! draw happens exactly as in the default run so the inputs are byte-identical across the two.
```

and change the usage lines to `//!   cargo run --release -p sapient-backends-cpu --example dump_kernels -- --out <dir> [--seed N] [--q8k-off]` and `const USAGE: &str = "usage: dump_kernels --out <dir> [--seed N] [--q8k-off] | --format-sample <file>";`.

(b) `fn build_cases(seed: u64, q8k_off: bool) -> Result<Vec<Case>, Box<dyn Error>>`; right after `let mut cases: Vec<Case> = Vec::new();` add
```rust
    // `--q8k-off`: the twelve SAPIENT_Q8K_ACT-sensitive matmul_nt cases get this suffix and are
    // the only ones kept (see the end of this function). Every other case is still computed so
    // the RNG sequence — and therefore every input — is identical to the default run.
    let q8k_suffix = if q8k_off { "_q8k_off" } else { "" };
```
and in the sub-project-0 quantized `matmul_nt` loop change `format!("matmul_nt_{tag}_m{m}")` to
```rust
                format!(
                    "matmul_nt_{tag}_m{m}{}",
                    if tag == "q8_0" { "" } else { q8k_suffix }
                ),
```

(c) Append before `Ok(cases)`:

```rust
    // ── plan D: quantized kernels (sub-project 1a) ──────────────────────────────────────────────
    // Activation quantisers: per-32 int8 blocks (+ the precomputed sums) and the Q8_K per-256
    // format, with one outlier channel so the per-block scale isolation is exercised. Integer
    // outputs are compared exactly on the C++ side; scales bit-identically.
    {
        let mut xa = rng.f32s(512, -2.0, 2.0);
        xa[100] = 25.0;
        let (q, s) = quant::quantize_row_to_i8_blocks(&xa);
        cases.push(case(
            "quantize_row_to_i8_blocks",
            vec![
                Array::f32("in:x", &[512], &xa),
                Array::i8("out:q", &[512], &q),
                Array::f32("out:scales", &[16], &s),
            ],
        ));
        let sums = quant::i8_block_sums(&q);
        cases.push(case(
            "i8_block_sums",
            vec![
                Array::i8("in:q", &[512], &q),
                Array::i32("out:sums", &[16], &sums),
            ],
        ));
        let (q, s, sums) = quant::quantize_row_to_q8k(&xa);
        cases.push(case(
            "quantize_row_to_q8k",
            vec![
                Array::f32("in:x", &[512], &xa),
                Array::i8("out:q", &[512], &q),
                Array::f32("out:scales", &[2], &s),
                Array::i32("out:sums", &[16], &sums),
            ],
        ));
    }
    // Row-interleaved repacks (u8 exact) — the only producers of R4 bytes — and the matmul_nt
    // paths plan C stubbed: Q4_0 (m=1 GEMV, m=3), Q5_K (m=1, no SIMD/dotprod path exists), Q8_0
    // at m=8 (the blocked W8A8 GEMM), and the R4 layouts at m ∈ {1 (decode), 2 (SMMLA pairs),
    // 3 (pair + odd tail), 8}. The R4 cases feed the Rust-repacked bytes so the matmul gate is
    // independent of the repack gate.
    {
        let (r4_rows, r4_k) = (8usize, 512usize);
        let q4k_plain = kquant_rows(&mut rng, r4_rows, r4_k, q4_k_block);
        let q4k_packed = quant::repack_q4_k_rows4(&q4k_plain, r4_rows, r4_k);
        cases.push(case(
            "repack_q4_k_rows4",
            vec![
                Array::u8("in:blocks", &[q4k_plain.len()], &q4k_plain),
                Array::u32("param:shape", &[2], &[r4_rows as u32, r4_k as u32]),
                Array::u8("out:packed", &[q4k_packed.len()], &q4k_packed),
            ],
        ));
        let q6k_plain = kquant_rows(&mut rng, r4_rows, r4_k, q6_k_block);
        let q6k_packed = quant::repack_q6_k_rows4(&q6k_plain, r4_rows, r4_k);
        cases.push(case(
            "repack_q6_k_rows4",
            vec![
                Array::u8("in:blocks", &[q6k_plain.len()], &q6k_plain),
                Array::u32("param:shape", &[2], &[r4_rows as u32, r4_k as u32]),
                Array::u8("out:packed", &[q6k_packed.len()], &q6k_packed),
            ],
        ));

        let mut quant_matmul = |name: String,
                                wbytes: &[u8],
                                dtype: DType,
                                m: usize,
                                rng: &mut Rng|
         -> Result<(), Box<dyn Error>> {
            let wt = Tensor::from_quant_bytes(wbytes, vec![rows, kq_len], dtype)?;
            let xt = Tensor::from_f32(&rng.f32s(m * kq_len, -1.0, 1.0), vec![m, kq_len])?;
            let y = matmul::matmul_nt(&xt, &wt)?;
            cases.push(case(
                name,
                vec![
                    Array::tensor("in:x", &xt),
                    Array::u8("in:w_blocks", &[wbytes.len()], wbytes),
                    Array::u32("param:w_shape", &[2], &[rows as u32, kq_len as u32]),
                    Array::tensor("out:y", &y),
                ],
            ));
            Ok(())
        };
        let wq4 = q4_0_row(&rng.f32s(rows * kq_len, -1.0, 1.0));
        for m in [1usize, 3] {
            quant_matmul(format!("matmul_nt_q4_0_m{m}"), &wq4, DType::Q4_0, m, &mut rng)?;
        }
        let wq5 = kquant_rows(&mut rng, rows, kq_len, q5_k_block);
        quant_matmul("matmul_nt_q5_k_m1".to_string(), &wq5, DType::Q5_K, 1, &mut rng)?;
        let wq8 = q8_0_row(&rng.f32s(rows * kq_len, -1.0, 1.0));
        quant_matmul("matmul_nt_q8_0_m8".to_string(), &wq8, DType::Q8_0, 8, &mut rng)?;
        for m in [1usize, 2, 3, 8] {
            quant_matmul(
                format!("matmul_nt_q4_k_r4_m{m}{q8k_suffix}"),
                &q4k_packed,
                DType::Q4_K_R4,
                m,
                &mut rng,
            )?;
        }
        for m in [1usize, 2, 3, 8] {
            quant_matmul(
                format!("matmul_nt_q6_k_r4_m{m}{q8k_suffix}"),
                &q6k_packed,
                DType::Q6_K_R4,
                m,
                &mut rng,
            )?;
        }
    }
    // Plan-C carry-over: the SIMD-body + scalar-tail mixes of the dense dots. k=519 → dot_f32_fast
    // runs its 16-wide body, one 4-wide step and a 3-element scalar tail; k=67 → dot_f32_x_f16
    // mixes the NEON bit-surgery body with the software-f16 tail in ONE dot; head_dim=10 →
    // attention's dot_f32_neon/saxpby_neon take their 4-wide body + 2-lane tails.
    {
        let (k, n) = (519usize, 8usize);
        let wt = Tensor::from_f32(&rng.f32s(n * k, -1.0, 1.0), vec![n, k])?;
        let xt = Tensor::from_f32(&rng.f32s(k, -1.0, 1.0), vec![1, k])?;
        let y = matmul::matmul_nt(&xt, &wt)?;
        cases.push(case(
            "matmul_nt_f32_m1_k519",
            vec![
                Array::tensor("in:x", &xt),
                Array::tensor("in:w", &wt),
                Array::tensor("out:y", &y),
            ],
        ));
        let (k, n) = (67usize, 8usize);
        let w_src = rng.f32s(n * k, -1.0, 1.0);
        let w_f16: Vec<u8> = w_src
            .iter()
            .flat_map(|v| f16::from_f32(*v).to_le_bytes())
            .collect();
        let wt = Tensor::from_f16_bytes(&w_f16, vec![n, k])?;
        let xt = Tensor::from_f32(&rng.f32s(k, -1.0, 1.0), vec![1, k])?;
        let y = matmul::matmul_nt(&xt, &wt)?;
        cases.push(case(
            "matmul_nt_f16_m1_k67",
            vec![
                Array::tensor("in:x", &xt),
                Array::u8("in:w_f16", &[w_f16.len()], &w_f16),
                Array::u32("param:w_shape", &[2], &[n as u32, k as u32]),
                Array::tensor("out:y", &y),
            ],
        ));
        let hd10 = 10usize;
        let q = Tensor::from_f32(&rng.f32s(4 * hd10, -1.0, 1.0), vec![1, 4, 1, hd10])?;
        let k = Tensor::from_f32(&rng.f32s(2 * 5 * hd10, -1.0, 1.0), vec![1, 2, 5, hd10])?;
        let v = Tensor::from_f32(&rng.f32s(2 * 5 * hd10, -1.0, 1.0), vec![1, 2, 5, hd10])?;
        let y = attention::scaled_dot_product_attention(&q, &k, &v, None, None, 2)?;
        cases.push(case(
            "attention_decode_hd10",
            vec![
                Array::tensor("in:q", &q),
                Array::tensor("in:k", &k),
                Array::tensor("in:v", &v),
                Array::u32("param:n_kv_heads", &[1], &[2]),
                Array::tensor("out:y", &y),
            ],
        ));
    }
    if q8k_off {
        cases.retain(|(name, _)| name.ends_with("_q8k_off"));
    }
```

(`rows`/`kq_len` are the `(8, 512)` bound earlier in `build_cases`; the closure captures `cases` mutably and takes `rng` as a parameter to satisfy the borrow checker — if rustc still complains about `cases` being borrowed by the closure while `cases.push` is used after, end the closure's scope before the carry-over block, which the braces above already do.)

(d) `main`: add `let mut q8k_off = false;` next to `seed`, the arm `"--q8k-off" => { q8k_off = true; i += 1; }`, and call `build_cases(seed, q8k_off)?`.

Verify: `cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings` clean; `cargo run --release -q -p sapient-backends-cpu --example dump_kernels -- --out /tmp/sapient-golden | tail -1` prints `wrote 63 cases`; `SAPIENT_Q8K_ACT=0 cargo run --release -q -p sapient-backends-cpu --example dump_kernels -- --out /tmp/sapient-golden --q8k-off | tail -1` prints `wrote 12 cases`; the two runs' `matmul_nt_q4_k_r4_m2.sapd` and `matmul_nt_q4_k_r4_m2_q8k_off.sapd` differ only in `out:y` (`cmp` shows a difference; the name and inputs are the same — spot-check with `cmp -l | head`); run the default twice → identical files.

- [ ] **Step 2: `golden_dump.sh`** — after the first `cargo run` line add:

```bash
# Second pass (spec §4 "both SAPIENT_Q8K_ACT settings"): the knob is read once per process on both
# sides, so the twelve knob-sensitive matmul_nt cases are regenerated under SAPIENT_Q8K_ACT=0 with
# a `_q8k_off` suffix (`--q8k-off` keeps every RNG draw and writes only those cases).
(cd "$repo" && SAPIENT_Q8K_ACT=0 cargo run --release -q -p sapient-backends-cpu --example dump_kernels -- --out "$out" --q8k-off "$@")
```
`cpp/tests/parity/golden_dump.sh /tmp/sapient-golden` → `golden_dump: 75 cases`.

- [ ] **Step 3: `golden_quant_test.cpp`**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
// Plan D gate: the quantized kernels vs the Rust oracle's dumps (SAPIENT_GOLDEN_DIR; unset → SKIP,
// set-but-missing → FAIL). Everything here is BIT-IDENTICAL (integer arrays exactly, f32 arrays on
// their bit patterns) — no quantized path touches sgemm. The `_q8k_off` cases need
// SAPIENT_Q8K_ACT=0 in the process environment (the `sapient_backends_cpu_tests.q8k_off` ctest
// entry sets it); under the default environment they skip on aarch64 and simply run elsewhere
// (Rust's knob is aarch64-only, so the results are knob-independent on x86).
#include <gtest/gtest.h>

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sapient/backends_cpu/kernels/matmul.hpp"
#include "sapient/backends_cpu/kernels/quant.hpp"
#include "sapient/core/dtype.hpp"
#include "sapient/core/shape.hpp"
#include "sapient/core/tensor.hpp"
#include "sapient/testing/compare.hpp"

namespace quant = sapient::backends_cpu::kernels::quant;
namespace matmul = sapient::backends_cpu::kernels::matmul;
using sapient::core::DType;
using sapient::core::Shape;
using sapient::core::Tensor;
using sapient::testing::bit_identical;
using sapient::testing::exact_equal;
using sapient::testing::GoldenCase;

namespace {

Tensor tensor_of(const GoldenCase& c, std::string_view name) {
    const auto& a = c.get(name);
    std::vector<size_t> dims(a.dims.begin(), a.dims.end());
    auto t = Tensor::from_f32_vec(a.as<float>(), Shape(dims));
    if (!t) throw std::runtime_error(std::string(name) + ": " + t.error().to_string());
    return std::move(*t);
}
Tensor quant_tensor_of(const GoldenCase& c, DType dtype) {
    const auto bytes = c.get("in:w_blocks").as<uint8_t>();
    const auto shape = c.get("param:w_shape").as<uint32_t>();
    auto t = Tensor::from_quant_bytes(bytes, Shape{shape.at(0), shape.at(1)}, dtype);
    if (!t) throw std::runtime_error("in:w_blocks: " + t.error().to_string());
    return std::move(*t);
}
template <class T> ::testing::AssertionResult same(const std::vector<T>& got, const std::vector<T>& ref) {
    return exact_equal<T>(std::span<const T>(got), std::span<const T>(ref));
}
std::vector<float> f1(float v) {
    return {v};
}

bool q8k_on() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return matmul::detail::q8k_activations();
#else
    return false;
#endif
}

// One matmul_nt golden case: x from `in:x`, W from the raw quantized bytes, y bit-identical.
void check_matmul_nt(const char* name, DType dtype) {
    SAPIENT_GOLDEN_CASE(c, name);
    auto y = matmul::matmul_nt(tensor_of(c, "in:x"), quant_tensor_of(c, dtype));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), c.get("out:y").as<float>()));
}
void check_matmul_nt_q8k_off(const char* name, DType dtype) {
    if (q8k_on())
        GTEST_SKIP() << "needs SAPIENT_Q8K_ACT=0 (run by the sapient_backends_cpu_tests.q8k_off ctest entry)";
    check_matmul_nt(name, dtype);
}

} // namespace

// ── block quantizers and row dots (sub-project 0 cases, consumed here for the first time) ──

TEST(GoldenQuant, quantize_q8_0_block) {
    SAPIENT_GOLDEN_CASE(c, "quantize_q8_0_block");
    const auto blk = quant::quantize_q8_0_block(c.get("in:x").as<float>());
    EXPECT_TRUE(same(std::vector<uint8_t>(blk.begin(), blk.end()), c.get("out:block").as<uint8_t>()));
}
TEST(GoldenQuant, quantize_q4_0_block) {
    SAPIENT_GOLDEN_CASE(c, "quantize_q4_0_block");
    const auto blk = quant::quantize_q4_0_block(c.get("in:x").as<float>());
    EXPECT_TRUE(same(std::vector<uint8_t>(blk.begin(), blk.end()), c.get("out:block").as<uint8_t>()));
}
TEST(GoldenQuant, dot_q8_0_row_f32) {
    SAPIENT_GOLDEN_CASE(c, "dot_q8_0_row_f32");
    const float y = quant::dot_q8_0_row_f32(c.get("in:row_blocks").as<uint8_t>(), c.get("in:x").as<float>());
    EXPECT_TRUE(bit_identical(f1(y), c.get("out:y").as<float>()));
}
TEST(GoldenQuant, dot_q4_0_row_f32) {
    SAPIENT_GOLDEN_CASE(c, "dot_q4_0_row_f32");
    const float y = quant::dot_q4_0_row_f32(c.get("in:row_blocks").as<uint8_t>(), c.get("in:x").as<float>());
    EXPECT_TRUE(bit_identical(f1(y), c.get("out:y").as<float>()));
}
TEST(GoldenQuant, dot_q4_k_row_f32) {
    SAPIENT_GOLDEN_CASE(c, "dot_q4_k_row_f32");
    const float y = quant::dot_q4_k_row_f32(c.get("in:row_blocks").as<uint8_t>(), c.get("in:x").as<float>());
    EXPECT_TRUE(bit_identical(f1(y), c.get("out:y").as<float>()));
}
TEST(GoldenQuant, dot_q5_k_row_f32) {
    SAPIENT_GOLDEN_CASE(c, "dot_q5_k_row_f32");
    const float y = quant::dot_q5_k_row_f32(c.get("in:row_blocks").as<uint8_t>(), c.get("in:x").as<float>());
    EXPECT_TRUE(bit_identical(f1(y), c.get("out:y").as<float>()));
}
TEST(GoldenQuant, dot_q6_k_row_f32) {
    SAPIENT_GOLDEN_CASE(c, "dot_q6_k_row_f32");
    const float y = quant::dot_q6_k_row_f32(c.get("in:row_blocks").as<uint8_t>(), c.get("in:x").as<float>());
    EXPECT_TRUE(bit_identical(f1(y), c.get("out:y").as<float>()));
}

// ── activation quantisers and repacks (plan D cases) ──

TEST(GoldenQuant, quantize_row_to_i8_blocks) {
    SAPIENT_GOLDEN_CASE(c, "quantize_row_to_i8_blocks");
    const auto r = quant::quantize_row_to_i8_blocks(c.get("in:x").as<float>());
    EXPECT_TRUE(same(r.q, c.get("out:q").as<int8_t>()));
    EXPECT_TRUE(bit_identical(r.scales, c.get("out:scales").as<float>()));
}
TEST(GoldenQuant, i8_block_sums) {
    SAPIENT_GOLDEN_CASE(c, "i8_block_sums");
    EXPECT_TRUE(same(quant::i8_block_sums(c.get("in:q").as<int8_t>()), c.get("out:sums").as<int32_t>()));
}
TEST(GoldenQuant, quantize_row_to_q8k) {
    SAPIENT_GOLDEN_CASE(c, "quantize_row_to_q8k");
    const auto r = quant::quantize_row_to_q8k(c.get("in:x").as<float>());
    EXPECT_TRUE(same(r.q, c.get("out:q").as<int8_t>()));
    EXPECT_TRUE(bit_identical(r.scales, c.get("out:scales").as<float>()));
    EXPECT_TRUE(same(r.sums, c.get("out:sums").as<int32_t>()));
}
TEST(GoldenQuant, repack_q4_k_rows4) {
    SAPIENT_GOLDEN_CASE(c, "repack_q4_k_rows4");
    const auto shape = c.get("param:shape").as<uint32_t>();
    EXPECT_TRUE(same(quant::repack_q4_k_rows4(c.get("in:blocks").as<uint8_t>(), shape.at(0), shape.at(1)),
                     c.get("out:packed").as<uint8_t>()));
}
TEST(GoldenQuant, repack_q6_k_rows4) {
    SAPIENT_GOLDEN_CASE(c, "repack_q6_k_rows4");
    const auto shape = c.get("param:shape").as<uint32_t>();
    EXPECT_TRUE(same(quant::repack_q6_k_rows4(c.get("in:blocks").as<uint8_t>(), shape.at(0), shape.at(1)),
                     c.get("out:packed").as<uint8_t>()));
}

// ── matmul_nt over every quantized dtype (default SAPIENT_Q8K_ACT) ──

TEST(GoldenQuant, matmul_nt_q8_0_m1) { check_matmul_nt("matmul_nt_q8_0_m1", DType::Q8_0); }
TEST(GoldenQuant, matmul_nt_q8_0_m3) { check_matmul_nt("matmul_nt_q8_0_m3", DType::Q8_0); }
TEST(GoldenQuant, matmul_nt_q8_0_m8) { check_matmul_nt("matmul_nt_q8_0_m8", DType::Q8_0); } // blocked GEMM
TEST(GoldenQuant, matmul_nt_q4_0_m1) { check_matmul_nt("matmul_nt_q4_0_m1", DType::Q4_0); }
TEST(GoldenQuant, matmul_nt_q4_0_m3) { check_matmul_nt("matmul_nt_q4_0_m3", DType::Q4_0); }
TEST(GoldenQuant, matmul_nt_q5_k_m1) { check_matmul_nt("matmul_nt_q5_k_m1", DType::Q5_K); }
TEST(GoldenQuant, matmul_nt_q4_k_m1) { check_matmul_nt("matmul_nt_q4_k_m1", DType::Q4_K); }
TEST(GoldenQuant, matmul_nt_q4_k_m3) { check_matmul_nt("matmul_nt_q4_k_m3", DType::Q4_K); }
TEST(GoldenQuant, matmul_nt_q6_k_m1) { check_matmul_nt("matmul_nt_q6_k_m1", DType::Q6_K); }
TEST(GoldenQuant, matmul_nt_q6_k_m3) { check_matmul_nt("matmul_nt_q6_k_m3", DType::Q6_K); }
TEST(GoldenQuant, matmul_nt_q4_k_r4_m1) { check_matmul_nt("matmul_nt_q4_k_r4_m1", DType::Q4_K_R4); }
TEST(GoldenQuant, matmul_nt_q4_k_r4_m2) { check_matmul_nt("matmul_nt_q4_k_r4_m2", DType::Q4_K_R4); }
TEST(GoldenQuant, matmul_nt_q4_k_r4_m3) { check_matmul_nt("matmul_nt_q4_k_r4_m3", DType::Q4_K_R4); }
TEST(GoldenQuant, matmul_nt_q4_k_r4_m8) { check_matmul_nt("matmul_nt_q4_k_r4_m8", DType::Q4_K_R4); }
TEST(GoldenQuant, matmul_nt_q6_k_r4_m1) { check_matmul_nt("matmul_nt_q6_k_r4_m1", DType::Q6_K_R4); }
TEST(GoldenQuant, matmul_nt_q6_k_r4_m2) { check_matmul_nt("matmul_nt_q6_k_r4_m2", DType::Q6_K_R4); }
TEST(GoldenQuant, matmul_nt_q6_k_r4_m3) { check_matmul_nt("matmul_nt_q6_k_r4_m3", DType::Q6_K_R4); }
TEST(GoldenQuant, matmul_nt_q6_k_r4_m8) { check_matmul_nt("matmul_nt_q6_k_r4_m8", DType::Q6_K_R4); }

// ── the same twelve knob-sensitive cases under SAPIENT_Q8K_ACT=0 (per-32 W4A8/W6A8 kernels) ──

TEST(GoldenQuant, matmul_nt_q4_k_m1_q8k_off) { check_matmul_nt_q8k_off("matmul_nt_q4_k_m1_q8k_off", DType::Q4_K); }
TEST(GoldenQuant, matmul_nt_q4_k_m3_q8k_off) { check_matmul_nt_q8k_off("matmul_nt_q4_k_m3_q8k_off", DType::Q4_K); }
TEST(GoldenQuant, matmul_nt_q6_k_m1_q8k_off) { check_matmul_nt_q8k_off("matmul_nt_q6_k_m1_q8k_off", DType::Q6_K); }
TEST(GoldenQuant, matmul_nt_q6_k_m3_q8k_off) { check_matmul_nt_q8k_off("matmul_nt_q6_k_m3_q8k_off", DType::Q6_K); }
TEST(GoldenQuant, matmul_nt_q4_k_r4_m1_q8k_off) { check_matmul_nt_q8k_off("matmul_nt_q4_k_r4_m1_q8k_off", DType::Q4_K_R4); }
TEST(GoldenQuant, matmul_nt_q4_k_r4_m2_q8k_off) { check_matmul_nt_q8k_off("matmul_nt_q4_k_r4_m2_q8k_off", DType::Q4_K_R4); }
TEST(GoldenQuant, matmul_nt_q4_k_r4_m3_q8k_off) { check_matmul_nt_q8k_off("matmul_nt_q4_k_r4_m3_q8k_off", DType::Q4_K_R4); }
TEST(GoldenQuant, matmul_nt_q4_k_r4_m8_q8k_off) { check_matmul_nt_q8k_off("matmul_nt_q4_k_r4_m8_q8k_off", DType::Q4_K_R4); }
TEST(GoldenQuant, matmul_nt_q6_k_r4_m1_q8k_off) { check_matmul_nt_q8k_off("matmul_nt_q6_k_r4_m1_q8k_off", DType::Q6_K_R4); }
TEST(GoldenQuant, matmul_nt_q6_k_r4_m2_q8k_off) { check_matmul_nt_q8k_off("matmul_nt_q6_k_r4_m2_q8k_off", DType::Q6_K_R4); }
TEST(GoldenQuant, matmul_nt_q6_k_r4_m3_q8k_off) { check_matmul_nt_q8k_off("matmul_nt_q6_k_r4_m3_q8k_off", DType::Q6_K_R4); }
TEST(GoldenQuant, matmul_nt_q6_k_r4_m8_q8k_off) { check_matmul_nt_q8k_off("matmul_nt_q6_k_r4_m8_q8k_off", DType::Q6_K_R4); }
```

(`SAPIENT_GOLDEN_CASE` inside a `void` helper is fine: `FAIL()`/`GTEST_SKIP()` are `return`s of a void expression, which is exactly why gtest requires them in void functions; the skip/failure is recorded on the running test.)

- [ ] **Step 4: `golden_kernels_test.cpp`** — in the header comment replace `The six matmul_nt_q*_m{1,3} dumps are plan D's.` with `The quantized matmul_nt dumps are consumed by golden_quant_test.cpp (plan D).`, and append:

```cpp
// ── plan-C carry-over: SIMD body + scalar tail (plan D, M1-a) ────────────────

TEST(GoldenKernels, matmul_nt_f32_m1_k519) { // 16-wide body, one 4-wide step, 3-element tail
    SAPIENT_GOLDEN_CASE(c, "matmul_nt_f32_m1_k519");
    auto y = matmul::matmul_nt(tensor_of(c, "in:x"), tensor_of(c, "in:w"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

TEST(GoldenKernels, matmul_nt_f16_m1_k67) { // NEON bit-surgery body + software-f16 tail in one dot
    SAPIENT_GOLDEN_CASE(c, "matmul_nt_f16_m1_k67");
    const auto shape = c.get("param:w_shape").as<uint32_t>();
    auto w = Tensor::from_f16_bytes(c.get("in:w_f16").as<uint8_t>(), Shape{shape.at(0), shape.at(1)});
    ASSERT_TRUE(w.has_value()) << w.error().to_string();
    auto y = matmul::matmul_nt(tensor_of(c, "in:x"), *w);
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

TEST(GoldenKernels, attention_decode_hd10) { // dot_f32_neon / saxpby_neon 4-wide body + 2-lane tail
    SAPIENT_GOLDEN_CASE(c, "attention_decode_hd10");
    auto y = attention::scaled_dot_product_attention(tensor_of(c, "in:q"), tensor_of(c, "in:k"),
                                                     tensor_of(c, "in:v"), nullptr, std::nullopt,
                                                     param<uint32_t>(c, "param:n_kv_heads"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}
```
(Match the argument spelling of the existing `attention_decode` test in that file if it differs from the above — the existing test is the authority for how `mask`/`scale`/`n_kv_heads` are passed.)

- [ ] **Step 5: CMake** — add `tests/golden_quant_test.cpp` after `tests/golden_kernels_test.cpp` in `add_executable`, and after `gtest_discover_tests(sapient_backends_cpu_tests)`:

```cmake
  # The SAPIENT_Q8K_ACT knob is read once per process (Rust OnceLock twin), so the OFF setting
  # needs its own ctest entry; the `GoldenQuant.*_q8k_off` gtests skip under the default env.
  add_test(NAME sapient_backends_cpu_tests.q8k_off
           COMMAND sapient_backends_cpu_tests --gtest_filter=GoldenQuant.*_q8k_off)
  set_tests_properties(sapient_backends_cpu_tests.q8k_off PROPERTIES ENVIRONMENT "SAPIENT_Q8K_ACT=0")
```

- [ ] **Step 6: Run the gate both ways**

```bash
cd cpp && cmake --preset dev && cmake --build --preset dev \
  && ctest --preset dev -N | tail -1 \
  && ctest --preset dev \
  && SAPIENT_GOLDEN_DIR=/tmp/sapient-golden ctest --preset dev \
  && SAPIENT_GOLDEN_DIR=/tmp/sapient-golden ./build/dev/libs/sapient-backends-cpu/sapient_backends_cpu_tests --gtest_filter='Golden*' 2>&1 | tail -3 \
  && SAPIENT_Q8K_ACT=0 SAPIENT_GOLDEN_DIR=/tmp/sapient-golden ./build/dev/libs/sapient-backends-cpu/sapient_backends_cpu_tests --gtest_filter='GoldenQuant.*_q8k_off' 2>&1 | tail -3
```
Expected, derived (report the actual numbers; a mismatch is reported, never "fixed" in a test): `ctest -N` = 196 + 42 + 3 + 1 = **242**; without dumps: 76 skipped (31 + 45 new golden tests), 166 passed; with dumps: 13 skipped on this Mac (the 12 `_q8k_off` under the default env + `Compare.golden_case_macro_skips_without_env`), 229 passed; the last two commands show every `GoldenQuant`/`GoldenKernels` test PASSED with none failing (the 12 `_q8k_off` pass under `SAPIENT_Q8K_ACT=0`). Also record `ls /tmp/sapient-golden | wc -l` = 75.

- [ ] **Step 7: Format after `git add` (the `.cpp` files; the `.rs` via `cargo fmt --all`), `cargo clippy --workspace --all-targets -- -D warnings`, commit**

```bash
git add crates/sapient-backends/cpu/examples/dump_kernels.rs cpp/tests/parity/golden_dump.sh cpp/libs/sapient-backends-cpu/CMakeLists.txt cpp/libs/sapient-backends-cpu/tests/golden_quant_test.cpp cpp/libs/sapient-backends-cpu/tests/golden_kernels_test.cpp
git commit -m "cpp(backends-cpu): golden_quant_test — quant kernels, repacks, quantized matmul_nt (both SAPIENT_Q8K_ACT settings) bit-identical to Rust; odd-length dense cases

dump_kernels: 20 plan-D cases appended at the end of build_cases (63 total) plus the
--q8k-off pass (12 renamed cases) that golden_dump.sh now runs under SAPIENT_Q8K_ACT=0.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Docs, parity ledger, final verification

**Files:**
- Modify: `CLAUDE.md`, `docs/ROADMAP.md`, `docs/PARITY.md`, `docs/PROJECT_GUIDE.md`, `CHANGELOG.md`, `docs/superpowers/specs/2026-09-21-cpp-sp1a-core-io-cpu-design.md` (one "as built" sentence in §4)

- [ ] **Step 1: CLAUDE.md** — in the `## C++ rewrite programme` section: (a) in the plan-C bullet replace the trailing `Next: plan D (quant kernels).` with `Plan D followed (next bullet).`; (b) append this bullet after the plan-C bullet:

```markdown
- **Sub-project 1a, plan D landed (`sapient::backends_cpu` quantized kernels):** `cpp/libs/sapient-backends-cpu/src/kernels/quant.cpp` ports `kernels/quant.rs` 1:1 (all 24 Rust tests by name) and `matmul_nt` gained its seven quantized arms with Rust's runtime dispatch (the five quant matmul tests incl. the two `to_bits` gates): Q4_0/Q8_0 blocks, the per-32 int8 and Q8_K activation quantisers with precomputed block sums (`x_sums` is always a parameter, never re-reduced), Q4_K/Q5_K/Q6_K f32 dots, the W4A8/W6A8 and Q8_K integer-domain kernels, the 4-row and R4 kernels, the SMMLA x2 prefill kernels, the R4 repacks. ISA mirror: scalar everywhere; aarch64 NEON compile-time, dotprod (`vdotq_s32` = Rust's `sdot` asm) and i8mm (`vmmlaq_s32` = `smmla`) kernels each carry `__attribute__((target("dotprod"/"i8mm")))` — **i8mm does not imply dotprod in Clang, and a lambda does not inherit its enclosing function's target attribute** (write attributed `static inline` helpers); the attribute's presence is only diagnosed at codegen (`-c -mcpu=cortex-a53`), never by `-fsyntax-only`, and this Mac's default baseline already has both features, so the plan's per-task `-mcpu=cortex-a53` compile check is the gate; x86_64 has exactly one SIMD kernel (`dot_q8_0_row_avx2`, runtime-gated). Rust `as` casts are explicit helpers (`f32_to_i32_sat`, `round_clamp_i8`: saturating, NaN → 0 — a C++ `static_cast` would be UB); `roundf` is half-away-from-zero; `fmaxf` drops NaN. `get_scale_min_k4` and `dequantize_q4_0_block` reuse `sapient::core::dequant` (the one dequantiser). `SAPIENT_Q8K_ACT` is read once per process on both sides, so the golden gate runs a second `dump_kernels --q8k-off` pass under `SAPIENT_Q8K_ACT=0` and a second ctest entry (`sapient_backends_cpu_tests.q8k_off`) for the twelve `_q8k_off` cases. **Second known oracle defect reproduced with a stated deviation:** Rust's `dot_q8_0_row_avx2` loads 8 floats at `x + 8g + 4` for every group, reading 4 floats past the block (past the end of the activation slice on the last block); those lanes only multiply zero quant lanes, so the port loads them in-bounds and zero-extended — bit-identical for finite activations (non-finite ones may give `inf` where Rust gives `NaN`); recorded in `docs/PARITY.md`. Gate: 75 golden dumps (63 default + 12 knob-off) — every quantized case bit-identical on this Mac (NEON/dotprod/i8mm), the three odd-length dense carry-over cases too. Next: plan E (spinpool + thermal).
```

- [ ] **Step 1b: spec §4** — in `docs/superpowers/specs/2026-09-21-cpp-sp1a-core-io-cpu-design.md`, append to the "Dump cases added by plan" bullet (after `B — none (…)`): `**As built (plan D):** "both \`SAPIENT_Q8K_ACT\` settings" is a second \`dump_kernels --q8k-off\` pass under \`SAPIENT_Q8K_ACT=0\` (the knob is read once per process on both sides) whose twelve knob-sensitive cases carry a \`_q8k_off\` suffix, consumed by a second ctest entry with that environment; the D suite also carries plan C's odd-length carry-over cases (\`matmul_nt_f32_m1_k519\`, \`matmul_nt_f16_m1_k67\`, \`attention_decode_hd10\`).`

- [ ] **Step 2: `docs/ROADMAP.md`** — Phase 7 table row 1a status → `in progress — plans A (core), C (dense kernels) and D (quant kernels) implemented on feat/cpp-sp1a; plans E, B pending`.

- [ ] **Step 3: `docs/PARITY.md`** — three edits under "Sub-project 1a":

(a) Replace gap (a) with:
```markdown
- (a) The golden dumps cover the 22 public entry points plus, as of plan A, the R4 **dequant** path
  (`dequant_q4_k_r4`/`dequant_q6_k_r4`), as of plan C `apply_rope_partial(_scaled)`, the
  masked-attention branch, the two float GEMV paths, `layer_norm`, `reduce`, `gelu`, `softmax`
  axis 0, `log_softmax` and `conv2d`, and as of plan D the activation quantisers
  (`quantize_row_to_i8_blocks`, `i8_block_sums`, `quantize_row_to_q8k`), both repacks, `matmul_nt`
  over Q4_0/Q5_K/Q8_0 (m=8, the blocked GEMM) and the R4 layouts at m ∈ {1,2,3,8}, the twelve
  `_q8k_off` twins, and the odd-length dense cases (75 dumps: 63 default + 12 knob-off) — **closed**.
```

(b) Append a row to the "Known oracle defects the port reproduces on purpose" table:
```markdown
| 2026-09-21 (plan D) | `crates/sapient-backends/cpu/src/kernels/quant.rs` `dot_q8_0_row_avx2` (x86_64 with AVX2+FMA; reached by every Q8_0 `matmul_nt` on x86 and by `dot_q8_0_row_f32`) | For every 8-quant group `g` the second activation load is `_mm256_loadu_ps(xp + g*8 + 4)` — 8 floats, of which only the first 4 pair with quants (the other 4 lanes of `_mm256_cvtepi8_epi32(_mm_loadu_si32(…))` are zero). At `g == 3` the load runs 4 floats past the 32-element block, i.e. **4 floats past the end of the activation slice on the row's last block** (an out-of-bounds read whose values only ever multiply zero). | Ported with the second half loaded in-bounds and zero-extended (`_mm256_insertf128_ps(_mm256_setzero_ps(), _mm_loadu_ps(…), 0)`): bit-identical for finite activations (`0·x = ±0` never changes a `+0.0` accumulator lane); a non-finite activation in those positions can yield `NaN` in Rust (`0·inf`) and `±inf` in C++. Gated by the x86 `cpp-parity` job's `matmul_nt_q8_0_m{1,3,8}` and `dot_q8_0_row_f32` cases (finite inputs). | Open — Rust is frozen during the port; the Rust fix is a 4-float load (`_mm_loadu_ps` zero-extended) or an `x` slice of `k + 4`. The user decides. |
```

(c) Append to the `### Results` table (fill the hashes with `git rev-parse --short` of the `dump_kernels` commit — Task 7's — and of HEAD):
```markdown
| 2026-09-21 | 24 quant + 5 quant-matmul backends-cpu unit tests ported by name (+ C++-only pins of the `as`-cast/`roundf`/`fmaxf` rules, the seven guard texts, repack death tests) | macOS arm64 (Apple M5; NEON/dotprod/i8mm) | — | `<cpp sha>` | pass |
| 2026-09-21 | Quantized golden cases: `quantize_q{4,8}_0_block`, `dot_q{4,8}_0_row_f32`, `dot_q{4,5,6}_k_row_f32`, `quantize_row_to_i8_blocks`, `i8_block_sums`, `quantize_row_to_q8k`, `repack_q{4,6}_k_rows4`, `matmul_nt_{q8_0_m{1,3,8},q4_0_m{1,3},q5_k_m1,q4_k_m{1,3},q6_k_m{1,3},q4_k_r4_m{1,2,3,8},q6_k_r4_m{1,2,3,8}}` (default `SAPIENT_Q8K_ACT`) and the twelve `_q8k_off` twins (`SAPIENT_Q8K_ACT=0`) vs the Rust kernels | macOS arm64 (Apple M5; NEON/dotprod/i8mm) | `<rust sha>` | `<cpp sha>` | bit-identical, 42/42 (30 default-env + 12 knob-off) |
| 2026-09-21 | Odd-length dense cases `matmul_nt_f32_m1_k519`, `matmul_nt_f16_m1_k67`, `attention_decode_hd10` (SIMD body + scalar tail) | macOS arm64 | `<rust sha>` | `<cpp sha>` | bit-identical, 3/3 |
| 2026-09-21 | x86_64 (scalar K-quants, AVX2 Q8_0) and Windows | CI `cpp-parity` (ubuntu) / `cpp-build-windows` | — | — | not yet run — first push of the stacked branches |
```

- [ ] **Step 4: `docs/PROJECT_GUIDE.md`** — in the "### The C++ tree (in progress)" paragraph, replace `and plan C added \`sapient::backends_cpu\`'s dense kernels (\`cpp/libs/sapient-backends-cpu/\`: attention, RoPE, norms, softmax, reductions, conv2d and the float matmul paths, with a rayon stand-in and an own SGEMM) — the golden-dump gates on both are bit-identical to the Rust oracle, except the SGEMM-backed paths, which are held within a tolerance.` with `plan C added \`sapient::backends_cpu\`'s dense kernels (\`cpp/libs/sapient-backends-cpu/\`: attention, RoPE, norms, softmax, reductions, conv2d and the float matmul paths, with a rayon stand-in and an own SGEMM), and plan D its quantized kernels (\`kernels/quant\`: every Q4_0/Q8_0/K-quant dot product, the int8 activation quantisers, the NEON/SDOT/SMMLA and R4 kernels, and the quantized arms of \`matmul_nt\`) — the golden-dump gates on all of them are bit-identical to the Rust oracle, except the SGEMM-backed paths, which are held within a tolerance.`

- [ ] **Step 5: `CHANGELOG.md`** — under `## [Unreleased]`, after the plan-C section, add:
```markdown
### 🧱 C++ rewrite — sub-project 1a, plan D (sapient::backends_cpu quantized kernels)
- `cpp/libs/sapient-backends-cpu`: `kernels/quant` ported 1:1 (Q4_0/Q8_0/Q4_K/Q5_K/Q6_K dots, int8 + Q8_K activation quantisers, NEON/SDOT/SMMLA and R4 kernels, repacks) with all 24 Rust tests; `matmul_nt` gains its seven quantized arms with Rust's runtime dispatch (5 Rust tests incl. the two bit-identity gates). 20 new golden cases plus a second `SAPIENT_Q8K_ACT=0` dump pass (75 dumps total) — every quantized case bit-identical to the Rust oracle on arm64.
- Test-only Rust: `dump_kernels` gains the plan-D cases and a `--q8k-off` mode. No product behaviour change. Recorded in `docs/PARITY.md`: a dormant out-of-bounds read in the Rust AVX2 Q8_0 row dot (zero lanes; no effect on finite inputs), which the port replaces with an in-bounds load.
```

- [ ] **Step 6: Final verification (record every summary line in the report)**

```bash
just cpp-lint && cd cpp && cmake --preset dev && cmake --build --preset dev \
  && ./tests/parity/golden_dump.sh /tmp/sapient-golden \
  && ctest --preset dev -N | tail -1 \
  && ctest --preset dev && SAPIENT_GOLDEN_DIR=/tmp/sapient-golden ctest --preset dev && cd .. \
  && cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings \
  && cargo test -p sapient-backends-cpu -- --test-threads=1 2>&1 | grep "^test result"
```
Expected, **derived** (report the actual numbers; a mismatch is reported in the hand-back, never "fixed" by editing tests): lint OK; `golden_dump: 75 cases`; `ctest -N` total = **242**; without dumps **166 passed / 76 skipped**; with dumps **229 passed / 13 skipped** (the twelve `_q8k_off` gtests under the default environment + `Compare.golden_case_macro_skips_without_env`, all by design); `cargo fmt`/`clippy` clean; Rust backends-cpu tests unchanged (`71 passed; 0 failed; 1 ignored`). Also run the two codegen checks of rule 5 one last time on the final `quant.cpp`.

- [ ] **Step 7: Commit**

```bash
git add CLAUDE.md docs/ROADMAP.md docs/PARITY.md docs/PROJECT_GUIDE.md CHANGELOG.md docs/superpowers/specs/2026-09-21-cpp-sp1a-core-io-cpu-design.md
git commit -m "docs(sp1a-D): quant kernels landed — CLAUDE.md, roadmap, parity ledger (+ AVX2 OOB-read oracle defect), spec as-built note

CONTRIBUTING and README need no change for a library-internal plan.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

**Do NOT push.** The user pushes `feat/cpp-sp1a` (stacked on `feat/cpp-sp0-scaffold`).

---

## Self-review against the spec

- **§2.3 `kernels/quant` row:** standalone (no Tensor) ✓ (T1 header: byte/float/int8 spans); every function of the porting map's inventory — §1a scalar (T1: Q4_0/Q8_0 quantisers, block/row dots, `quantize_row_to_i8_blocks`, `i8_block_sums`, `quantize_row_to_q8k`, `dot_q8_0_row_i8_scalar`, `dot_q8_0_row_f32`; T2: `dot_q4_k_row_f32(_scalar)`, `dot_q4_k_row_q8_scalar`, `repack_q4_k_rows4`; T3: `dot_q4_k_row_q8k_scalar`; T4: `dot_q5_k_row_f32(_scalar)`, `dot_q6_k_row_f32(_scalar)`, `repack_q6_k_rows4`; T5: `dot_q6_k_row_q8_scalar`, `dot_q6_k_row_q8k_scalar`; `get_scale_min_k4` via core), §1b NEON (T1: `dot_q4_0_block_neon`, `dot_q8_0_block_neon`; T2: `dot_q4_k_row_f32_neon`; T3: `vtrn1q/2q_s64_s8`; T4: `dot_q5_k_row_f32_neon`, `dot_q6_k_row_f32_neon`, `dot_q6_k_4rows_r4_neon`), §1c dotprod (T1: `dot_q8_0_block_sdot`, `dot_q8_0_row_sdot`; T2: `sdot_s32`, `dot_q4_k_row_q8_neon`, `dot_q4_k_4rows_q8_neon`, `dot_q4_k_4rows_r4_neon`; T3: `dot_q4_k_row_q8k_neon`, `dot_q4_k_4rows_q8k_neon`, `dot_q4_k_4rows_r4_q8k_neon`; T5: `dot_q6_k_row_q8_neon`, `dot_q6_k_row_q8k_neon`, `dot_q6_k_4rows_r4_q8_neon`, `dot_q6_k_4rows_r4_q8k_neon`), §1d i8mm (T3: `smmla_s32`, `dot_q4_k_4rows_r4_x2_smmla`, `dot_q4_k_4rows_r4_x2_q8k_smmla`; T5: `dot_q6_k_4rows_r4_x2_smmla`, `dot_q6_k_4rows_r4_x2_q8k_smmla`), §1e AVX2 (T1: `dot_q8_0_row_avx2`) ✓; grouped by ISA with the attribute macros ✓; same accumulation orders and horizontal reductions (`vaddvq_f32`/`vaddvq_s32`, the exact AVX2 hsum) ✓; quantiser rounding (`roundf` then clamp, nibble truncation, `fmaxf`) ✓; the two repacks ✓.
- **§2.3 `kernels/matmul` row (quant arms):** `matmul_nt` dispatch table verbatim (T6); `m ≥ 8` Q8_0 blocked GEMM, `q8k_activations()`, R4 prefill via SMMLA at `m ≥ 2 && i8mm`, R4 decode via dotprod, portable fallbacks (Q4_K_R4 per-32 scalar, Q6_K_R4 f32 in `#else`) ✓; Q5_K has no SIMD branch ✓; `gemv_chunk`/`for_each_out_chunk` reused unchanged (plan E's insertion point intact) ✓.
- **§3 rules:** flags unchanged (T1 inherits plan C's) ✓; ISA mirror, no upgrades — x86 K-quants scalar, AVX2 only for `dot_q8_0_row_avx2` (rule 4 + T1) ✓; accumulation order (rule 2) ✓; rounding (rule 3) ✓; software f16 everywhere (rule 1) ✓; libm per platform verified by dumps (T7) ✓; chunk geometry identical (the arms only call `for_each_out_chunk`/`par_chunks_mut`) ✓; no exceptions, `Result` + `panic()` with the seven byte-identical guard texts (rule 7, T6 test) ✓.
- **§4 verification:** 24 quant tests by name (T1 4, T2 4, T3 4, T4 6, T5 6) ✓ incl. the aarch64 `#cfg`s and runtime skips as `GTEST_SKIP`; 5 quant matmul tests by name with the two `to_bits` gates (T6) ✓; dump cases for D — `quantize_row_to_i8_blocks` ✓, `quantize_row_to_q8k` ✓, `i8_block_sums` ✓, `repack_q4_k_rows4`/`q6_k` ✓, `matmul_nt` over `Q4_K_R4`/`Q6_K_R4` at m ∈ {1,2,3,8} ✓, `Q8_0` at m=8 ✓, `Q5_K` ✓, both `SAPIENT_Q8K_ACT` settings (the `--q8k-off` pass + ctest entry ruling) ✓; `golden_quant_test.cpp` with `exact_equal<T>` for integer arrays and `bit_identical` for floats ✓; the seven sub-project-0 kernel-level dumps consumed for the first time ✓; hosts note unchanged (x86 via CI, recorded as "not yet run") ✓.
- **§5 row D doc updates:** T8 (CLAUDE.md, ROADMAP, PARITY incl. the second known-defect row, PROJECT_GUIDE, CHANGELOG, spec as-built note) ✓.
- **Plan-C carry-overs:** M1-a odd-length cases (T7) ✓; stub test deleted (T6) ✓; `x_sums` a parameter everywhere (T2/T3) ✓; `switch` arms (T6) ✓; M2 ruled closed by inspection (Global Constraints) ✓; `exact_equal<T>` used (T7) ✓.
- **Rulings recorded in the Global Constraints:** AVX2 OOB read ported in-bounds; Q8K-off second pass + ctest entry; M2 closed; R4 golden weights are Rust-repacked; target attributes + codegen check; `.take(nb)` truncation semantics; saturating casts via helpers; port-added dotprod skips on two tests.
- **Placeholder scan:** the `<cpp sha>`/`<rust sha>` tokens in T8 step 3 are deliberate fill-ins with the command to compute them; no TBD/TODO elsewhere; every code step carries its code.
- **Type consistency:** `I8Blocks{q, scales}` / `Q8kRow{q, scales, sums}` (T1) consumed in T2–T7 with those field names; `std::array<std::span<const uint8_t>, 4>` rows (T2/T3) built the same way in T6's Q4_K arm; `std::array<float, 4>` / `std::array<std::array<float, 2>, 4>` returns (T2–T5) indexed `v[r]` / `v[r][0..1]` in T6; `detail::{f32_to_i32_sat, round_clamp_i8, nibble, i8v}` (T1) used in T1's tests and T4/T5; `detail::q8k_activations()` (T6, aarch64) used in T7's `q8k_on()`; `check_len`/`check_q8_row`/`q4k_header`/`sdot_s32`/`smmla_s32`/`q6_unpack`/`q6_scale`/`widen_u8x16_to_f32x4x4` are `quant.cpp`-private helpers defined before their first use in the layout order; `Tensor::quant_blocks()` is the C++ `as_quant_blocks` (verified in tensor.cpp:109). Test-count arithmetic: 160 → 169 (T1 9) → 174 (T2 5) → 178 (T3 4) → 185 (T4 7) → 191 (T5 6) → 196 (T6 −1 +6) → 242 (T7 42 + 3 + 1); T8 adds none.
