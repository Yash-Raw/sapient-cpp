# Sub-project 1a, Plan C: `sapient::backends_cpu` dense kernels — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the C++ `sapient-backends-cpu` library (`cpp/libs/sapient-backends-cpu`, namespace `sapient::backends_cpu`) with the rayon stand-in (`parallel`), the runtime ISA probes (`cpu_features`), the `matrixmultiply` stand-in (`sgemm`), the eight dense kernel modules (`elementwise`, `softmax`, `reduce`, `layernorm`, `rope`, `attention`, `conv2d`, and the float paths + dispatcher skeleton of `matmul`), every one of their 26 + 5 Rust unit tests reproduced by name, and prove the dense kernels bit-identical to the Rust oracle (sgemm-backed paths max-error gated) through 12 new `dump_kernels` cases and a `golden_kernels_test`.

**Architecture:** One `.hpp/.cpp` pair per Rust module under `include/sapient/backends_cpu/kernels/` and `src/kernels/`, each in the namespace that mirrors its Rust path (`sapient::backends_cpu::kernels::attention::scaled_dot_product_attention`, …). Three modules Rust gets from crates are hand-written: `parallel` (a persistent pool reproducing rayon's `par_chunks_mut` chunk→range partition), `cpu_features` (cached `is_*_feature_detected!` twins) and `sgemm` (a packed, cache-blocked kernel with a 4-wide FMA strip; max-error gated, but its per-element result is independent of how the caller blocks rows — required because the callers block by thread count). `thermal` and `spinpool` ship as **inert stubs with plan E's final signatures** so `gemv_chunk`/`matmul_nt` compile now; `matmul_nt`'s quantized arms return an explicit "plan D" error. Every ported function keeps Rust's accumulation order, its per-ISA variant, and its libm calls.

**Tech Stack:** C++20, CMake presets from sub-project 0, GoogleTest, `sapient::core` (plan A), `sapient::testing` (plan A), the Rust `dump_kernels` example (test-only Rust, allowed), NEON intrinsics (aarch64), SSE/AVX2+FMA intrinsics (x86_64, runtime-gated).

**Spec:** `docs/superpowers/specs/2026-09-21-cpp-sp1a-core-io-cpu-design.md` (§2.3, §3, §4, §5 row C) under `docs/superpowers/specs/2026-09-20-cpp-rewrite-design.md`. **Porting map (line-cited; read §0, §2.2–2.3, §3, §4 and §8 before touching a module):** `docs/superpowers/notes/2026-09-21-sp1a-porting-map-cpu-kernels.md`. **Plan A** (the conventions and the `sapient::core` API this plan consumes): `docs/superpowers/plans/2026-09-21-cpp-sp1a-plan-a-core.md`.

## Global Constraints

- **Branch:** `feat/cpp-sp1a` (stacked on `feat/cpp-sp0-scaffold`; plan A is complete at a06bba9). Commit after every task, with a blank line before the `Co-Authored-By` trailer. **Never push.**
- **Compiler/flags (programme spec D1 + spec §3.1):** Clang only; `-ffp-contract=off` is applied by `cpp/libs/CMakeLists.txt` to everything under `libs/`; never add `-march=native`/`-ffast-math`. **Ruling (this plan, Task 1): `-fno-math-errno` joins the parity flags** — rustc lowers `sin`/`cos`/`exp`/`pow`/`sqrt` to LLVM intrinsics with no errno; Clang does the same only under `-fno-math-errno`, which is the Darwin default but **not** the Linux default. Without it the `cpp-parity` Linux job may reach different libm entry points (e.g. `sincosf` vs `sinf`+`cosf`) than the Rust build. Build and test with `cd cpp && cmake --preset dev && cmake --build --preset dev && ctest --preset dev`.
- **Warnings are errors** (`sapient_apply_warnings`: `-Wall -Wextra -Wpedantic -Wshadow -Werror`). No narrowing in braces, no unused parameters (`conv2d`'s unused `kernel_shape` is `[[maybe_unused]]`), no shadowing (lambdas inside loops: pick distinct names).
- **CI clang-tidy gate** (`.clang-tidy`, `WarningsAsErrors: '*'`, `bugprone-*`/`performance-*`; LLVM-18 tidy cannot parse this Mac's libc++, so it runs only in CI, which has never run on this branch): avoid the hits plan A took — wrap function-like macros in `NOLINTBEGIN/NOLINTEND(bugprone-macro-parentheses)`, group identical switch arms under shared `case` labels (`bugprone-branch-clone`) — and the ones this plan invites: `bugprone-implicit-widening-of-multiplication-result` (cast operands to `size_t` **before** multiplying: `static_cast<size_t>(a) * b`, never `size_t x = int_a * int_b`), `performance-unnecessary-value-param` (take `std::function` by `const&`), `bugprone-narrowing-conversions` (explicit `static_cast<float>(size_t)`).
- **Formatting:** run the CI-pinned clang-format 18 (`.superpowers/tools-venv/bin/clang-format -i`, or `just cpp-fmt`, which resolves it) on every new/changed `.hpp/.cpp` **after `git add`** (untracked files are invisible to `git ls-files`), then re-add. Homebrew's clang-format 23 disagrees with CI.
- **Naming (spec D3):** target `sapient_backends_cpu` / alias `sapient::backends_cpu`; headers under `cpp/libs/sapient-backends-cpu/include/sapient/backends_cpu/` (kernels under `kernels/`), sources under `src/` (`src/kernels/`), tests under `tests/` named `<module>_test.cpp`; namespaces mirror the Rust module path exactly (`sapient::backends_cpu::kernels::<module>`, `sapient::backends_cpu::{parallel, cpu_features, thermal, spinpool}`; `sgemm` is a free function in `sapient::backends_cpu`); gtest names are the Rust test names (`TEST(Attention, flash_matches_naive)` …).
- **SPDX header verbatim** on every new `.hpp/.cpp` (`//` form) and on `CMakeLists.txt`/`.cmake` edits (`#` form); `lint.spdx_headers` fails otherwise:
  `// SPDX-License-Identifier: AGPL-3.0-only`
  `// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)`
- **Bit-identity rules (spec §3, applied to this plan):**
  1. Every float literal carries an `f` suffix (`0.5f`, `1e-5f`, `0.797'884'56f`); every libm call is the **float** function by its C name (`::expf`, `::powf`, `::sinf`, `::cosf`, `::logf`, `::sqrtf`, `::tanhf`, `::fabsf`, `::roundf`, `::floorf`, `::ceilf`, `::fmaxf`, `::fminf`, `::copysignf`). `std::pow(base, 2.0 * i / d)` or a bare `2.0 * x` promotes to double and breaks bit-identity invisibly. `f32::EPSILON` → `FLT_EPSILON`; `f32::NEG_INFINITY` → `-std::numeric_limits<float>::infinity()` (or `-INFINITY`).
  2. **Rust's `Iterator::sum::<f32>()` seeds its accumulator at `-0.0`** (verified on this toolchain, rustc 1.98.1: `[-0.0f32].iter().sum() == -0.0` and an empty sum is `-0.0`). Every port of an `.iter().sum()` / `.map(..).sum()` starts at `-0.0f`. Explicit `fold(0.0, …)` sites (`reduce_sum`'s `init`) stay `0.0f`.
  3. Same accumulation order and the same per-ISA variant as Rust (porting map §2.1/§4): NEON on aarch64 is compile-time and unconditional; x86_64 has exactly one runtime-gated AVX2+FMA function in this plan (`dot_f32_avx2`, gated on `has_avx2_fma()`); everything else is scalar. Explicit FMA (`vfmaq_f32`, `_mm256_fmadd_ps`) only where Rust used the intrinsic; plain `a * b + c` elsewhere (contraction is off). Horizontal sums via `vaddvq_f32` / the exact AVX2 `_mm_movehdup_ps`/`_mm_movehl_ps` sequence — never a serial lane loop.
  4. `f32::max`/`f32::min` → `::fmaxf`/`::fminf` (NaN-dropping); `f32::round` → `::roundf` (half away from zero); `f32::clamp` → `std::clamp` (NaN passes through in both); `f32::signum` → NaN stays NaN, otherwise `::copysignf(1.0f, x)` (`+0.0 → 1`, `-0.0 → -1`, verified); f16→f32 via `sapient::core::f16_bits_to_f32` (software) everywhere except the one NEON bit-surgery in `dot_f32_x_f16_neon`, reproduced verbatim (normals only; the scalar tail uses the software conversion — divergence included).
  5. **Rust index/arithmetic panics become explicit checks + `sapient::core::panic()`**, never C++ UB: a `std::span`/`std::vector` index Rust would bounds-check is preceded by a check that panics (message text is not parity-bound). Each task lists its sites.
  6. **Result-path messages are byte-identical** to Rust: `matmul_nt expects 2-D tensors`; `RoPE requires even head_dim`, `positions length must match seq_len`, `rotary_dim must be in 1..=head_dim`, `RoPE requires even rotary_dim`; `softmax axis {axis} out of range for rank {ndim}`; conv2d's `conv2d: groups={g}, c_in={c_in}, c_in/group={c_in_g}: {c_in_g}*{g}!=c_in` (`InvalidGraph`); the `RankMismatch`/`ShapeMismatch`/`TypeMismatch` field values listed per task. `TypeMismatch.got` is `sapient::core::to_string(dtype)` — **qualify it** (plan A's member-name-hiding trap; inside `namespace elementwise` the module's own `exp/log/abs/sqrt/…(const Tensor&)` also hide the unqualified libm names, hence rule 1's `::expf` spelling).
  7. `Tensor::to_f32_cow()` on an F32 tensor is **unbounded** (the whole buffer from `offset`), exactly like Rust's `as_f32_slice`. Kernels that build their output from `data.len()` therefore return `ShapeMismatch` for a sliced F32 view, in C++ as in Rust — mirror it, don't "fix" it.
- **Ruling — output construction:** where Rust builds a local `Vec<f32>` and copies it via `Tensor::from_f32(&out, shape)`, the C++ port moves it via `Tensor::from_f32_vec(std::move(out), shape)`. Values are identical; only the buffer alignment differs (4 vs 64) and no consumer depends on it (plan A precedent: safetensors F32 copied once). Costs if wrong: a copy back.
- **Ruling — plan D/E hand-offs:** `matmul_nt`'s seven quantized-dtype arms return `Error::internal("matmul_nt: quantized weights (<dtype>) land in plan D")` (never a silent wrong result); `thermal::{tick, effective_threads}` and `spinpool::{enabled, parallelism}` are inert stubs whose bodies plan E replaces (`tick` no-op, `effective_threads() == parallel::num_threads()`, `enabled() == false`, `parallelism() == parallel::num_threads()`); `for_each_out_chunk` has only the `par_chunks_mut` branch (plan E adds the pool branch and the `SAPIENT_SPINPOOL_DEBUG` census in front of it). Each header says so.
- **`parallel` is a partition, not a scheduler:** the only thing parity depends on is that `par_chunks_mut(out, chunk, f)` calls `f(ci, out[ci*chunk, min((ci+1)*chunk, len)))` for every `ci` exactly once. Which thread runs which chunk is irrelevant; work stealing is not ported. `num_threads()` follows rayon-core 1.13 (`RAYON_NUM_THREADS` ≥1 → it; `0` → default; unparsable → `RAYON_RS_NUM_CPUS` with the same rule; else the logical CPU count, ≥1).
- **Rust tree frozen** except `crates/sapient-backends/cpu/examples/dump_kernels.rs` (Task 8 appends cases **at the end of `build_cases`** so earlier cases' RNG draws are unchanged). `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets -- -D warnings` must stay clean.
- **No exceptions across library boundaries**; recoverable failures are `Result`, unrecoverable ones are `panic()`. gtest death tests pin the Rust-parity panics this plan introduces (`par_chunks_mut(chunk = 0)`).
- **Docs rule:** Task 9 updates CLAUDE.md, docs/ROADMAP.md, docs/PARITY.md, docs/PROJECT_GUIDE.md (its C++-tree paragraph names what has landed — keep it true), CHANGELOG.md, and the `justfile` (`cpp-tidy`). CONTRIBUTING/README need no change for a library-internal plan; say so in the commit. `cpp/third_party/LICENSES.md` is unchanged (this plan adds no third-party code; `Threads::Threads` is the system library).

## File structure

```
cpp/cmake/flags.cmake                                   + -fno-math-errno (Task 1)
cpp/libs/CMakeLists.txt                                 + add_subdirectory(sapient-backends-cpu) (Task 1)
cpp/libs/sapient-backends-cpu/CMakeLists.txt            sources + tests per task
cpp/libs/sapient-backends-cpu/include/sapient/backends_cpu/
  cpu_features.hpp   has_dotprod / has_i8mm / has_avx2_fma (cached)                      (Task 1)
  parallel.hpp       num_threads / par_for / par_chunks_mut — the rayon twin              (Task 1)
  env.hpp            env_usize(name): Rust usize::from_str on getenv (RAYON_*, SAPIENT_GEMV_TPC, plan E knobs) (Task 1)
  thermal.hpp        inert stub: tick(), effective_threads()   → plan E                   (Task 1)
  spinpool.hpp       inert stub: enabled(), parallelism()      → plan E                   (Task 1)
  sgemm.hpp          sgemm(m,k,n,alpha,a,rsa,csa,b,rsb,csb,beta,c,rsc,csc)                (Task 2)
  kernels/elementwise.hpp   add…clip, erf_approx                                          (Task 3)
  kernels/softmax.hpp       softmax, log_softmax                                          (Task 4)
  kernels/reduce.hpp        reduce_sum/mean/max/min                                       (Task 4)
  kernels/layernorm.hpp     layer_norm, rms_norm                                          (Task 4)
  kernels/rope.hpp          apply_rope, apply_rope_partial(_scaled), rope_cos_sin_cache   (Task 4)
  kernels/attention.hpp     scaled_dot_product_attention, causal_mask                     (Task 5)
  kernels/conv2d.hpp        conv2d, IM2COL_NS, GEMM_NS                                    (Task 6)
  kernels/matmul.hpp        matmul, matmul_nt, gemm, detail::{gemv_chunk, for_each_out_chunk} (Task 7)
  kernels.hpp               umbrella (= kernels/mod.rs)                                   (Task 7)
cpp/libs/sapient-backends-cpu/src/{cpu_features,parallel,thermal,spinpool,sgemm}.cpp
cpp/libs/sapient-backends-cpu/src/kernels/{elementwise,softmax,reduce,layernorm,rope,attention,conv2d,matmul}.cpp
cpp/libs/sapient-backends-cpu/tests/{cpu_features,parallel,sgemm,elementwise,softmax,reduce,layernorm,rope,attention,conv2d,matmul,golden_kernels}_test.cpp
cpp/libs/sapient-testing/include/sapient/testing/compare.hpp + src/compare.cpp   + within_rel_of_max, exact_equal<T> (Task 8)
cpp/libs/sapient-testing/tests/golden_test.cpp                                    + 2 tests (Task 8)
crates/sapient-backends/cpu/examples/dump_kernels.rs                              + 12 dense cases (Task 8)
docs: CLAUDE.md, docs/ROADMAP.md, docs/PARITY.md, docs/PROJECT_GUIDE.md, CHANGELOG.md, justfile (Task 9)
```

---

### Task 1: Library skeleton, `-fno-math-errno`, `cpu_features`, `parallel`, and the plan-E stubs

**Files:**
- Modify: `cpp/cmake/flags.cmake` (both branches of `sapient_apply_parity_flags_here`)
- Modify: `cpp/libs/CMakeLists.txt` (one `add_subdirectory`)
- Create: `cpp/libs/sapient-backends-cpu/CMakeLists.txt`
- Create: `include/sapient/backends_cpu/{cpu_features,env,parallel,thermal,spinpool}.hpp`, `src/{cpu_features,parallel,thermal,spinpool}.cpp` (all paths below are relative to `cpp/libs/sapient-backends-cpu/`)
- Test: `tests/cpu_features_test.cpp`, `tests/parallel_test.cpp`

**Interfaces:**
- Produces: `sapient::backends_cpu::cpu_features::{has_dotprod, has_i8mm, has_avx2_fma}() -> bool`; `sapient::backends_cpu::env_usize(const char* name) -> std::optional<size_t>` (header-only); `sapient::backends_cpu::parallel::{num_threads() -> size_t, par_for(size_t n, const std::function<void(size_t)>&), par_chunks_mut(std::span<float> out, size_t chunk, const std::function<void(size_t, std::span<float>)>&)}`; `sapient::backends_cpu::thermal::{tick() -> void, effective_threads() -> size_t}`; `sapient::backends_cpu::spinpool::{enabled() -> bool, parallelism() -> size_t}`. Tasks 2, 5, 6, 7 consume `parallel`; Task 7 consumes `cpu_features`, `thermal`, `spinpool`.

- [ ] **Step 1: Write the failing tests**

`tests/cpu_features_test.cpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include "sapient/backends_cpu/cpu_features.hpp"

using namespace sapient::backends_cpu::cpu_features;

// C++-only: the probes are cached, consistent across calls, and platform-plausible. The values
// themselves are host facts (this Mac: dotprod=1, i8mm=1 via sysctl hw.optional.arm.FEAT_*).
TEST(CpuFeatures, probes_are_cached_and_platform_consistent) {
    EXPECT_EQ(has_dotprod(), has_dotprod());
    EXPECT_EQ(has_i8mm(), has_i8mm());
    EXPECT_EQ(has_avx2_fma(), has_avx2_fma());
#if defined(__aarch64__) || defined(_M_ARM64)
    EXPECT_FALSE(has_avx2_fma());
    if (has_i8mm()) EXPECT_TRUE(has_dotprod()) << "ARMv8.6 i8mm implies ARMv8.2 dotprod";
#else
    EXPECT_FALSE(has_dotprod());
    EXPECT_FALSE(has_i8mm());
#endif
    RecordProperty("dotprod", has_dotprod() ? 1 : 0);
    RecordProperty("i8mm", has_i8mm() ? 1 : 0);
    RecordProperty("avx2_fma", has_avx2_fma() ? 1 : 0);
}
```

`tests/parallel_test.cpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <span>
#include <vector>

#include "sapient/backends_cpu/parallel.hpp"

using namespace sapient::backends_cpu::parallel;

// C++-only tests: `parallel` replaces rayon; what parity depends on is the chunk→range partition.

TEST(Parallel, num_threads_is_at_least_one_and_stable) {
    EXPECT_GE(num_threads(), 1u);
    EXPECT_EQ(num_threads(), num_threads());
}

TEST(Parallel, par_for_visits_each_index_exactly_once) {
    const size_t n = 1000;
    std::vector<std::atomic<int>> hits(n);
    for (auto& h : hits) h.store(0);
    par_for(n, [&](size_t i) { hits[i].fetch_add(1); });
    for (size_t i = 0; i < n; ++i) EXPECT_EQ(hits[i].load(), 1) << "index " << i;
}

TEST(Parallel, par_for_zero_makes_no_calls) {
    std::atomic<int> calls{0};
    par_for(0, [&](size_t) { calls.fetch_add(1); });
    EXPECT_EQ(calls.load(), 0);
}

// rayon: out.par_chunks_mut(64).enumerate() over 1000 elements → chunk ci covers
// [ci*64, min((ci+1)*64, 1000)); 16 chunks, the last one 40 long.
TEST(Parallel, par_chunks_mut_partition_matches_rayon) {
    std::vector<float> out(1000, -1.0f);
    std::vector<std::atomic<int>> seen(16);
    for (auto& s : seen) s.store(0);
    std::vector<size_t> lens(16, 0);
    par_chunks_mut(out, 64, [&](size_t ci, std::span<float> cs) {
        ASSERT_LT(ci, 16u);
        seen[ci].fetch_add(1);
        lens[ci] = cs.size();
        for (float& v : cs) v = static_cast<float>(ci);
    });
    for (size_t ci = 0; ci < 16; ++ci) {
        EXPECT_EQ(seen[ci].load(), 1) << "chunk " << ci;
        EXPECT_EQ(lens[ci], ci == 15 ? 40u : 64u) << "chunk " << ci;
    }
    for (size_t i = 0; i < out.size(); ++i) EXPECT_EQ(out[i], static_cast<float>(i / 64)) << i;

    // Exact multiple: one chunk that is the whole slice.
    std::atomic<int> calls{0};
    par_chunks_mut(out, 1000, [&](size_t ci, std::span<float> cs) {
        EXPECT_EQ(ci, 0u);
        EXPECT_EQ(cs.size(), 1000u);
        calls.fetch_add(1);
    });
    EXPECT_EQ(calls.load(), 1);

    // Empty slice: no calls (rayon yields no chunks).
    std::vector<float> empty;
    par_chunks_mut(empty, 8, [&](size_t, std::span<float>) { FAIL() << "called on an empty slice"; });
}

TEST(Parallel, nested_par_for_completes) {
    std::atomic<int> count{0};
    par_for(8, [&](size_t) { par_for(8, [&](size_t) { count.fetch_add(1); }); });
    EXPECT_EQ(count.load(), 64);
}

// rayon panics on `par_chunks_mut(0)` ("chunk size must not be zero"); the twin panics too.
TEST(ParallelDeath, par_chunks_mut_zero_chunk_panics) {
    GTEST_FLAG_SET(death_test_style, "threadsafe"); // pool threads may already exist
    std::vector<float> out(4, 0.0f);
    EXPECT_DEATH(par_chunks_mut(out, 0, [](size_t, std::span<float>) {}), "chunk");
}
```

- [ ] **Step 2: Build to verify the tests fail**

Run: `cd cpp && cmake --preset dev && cmake --build --preset dev`
Expected: the new test files are not yet in any target — nothing builds them. After Step 3's `CMakeLists.txt` exists but before the headers do, the build FAILS with `'sapient/backends_cpu/parallel.hpp' file not found`.

- [ ] **Step 3: CMake wiring**

Append to `cpp/libs/CMakeLists.txt` (after the `sapient-testing` block):

```cmake
add_subdirectory(sapient-backends-cpu)
```

In `cpp/cmake/flags.cmake`, inside `sapient_apply_parity_flags_here()`, change the two `add_compile_options` lines:

```cmake
    add_compile_options(/clang:-ffp-contract=off /clang:-fno-math-errno)
```
and
```cmake
    add_compile_options(-ffp-contract=off -fno-fast-math -fno-math-errno)
```
and add above the function a comment line:
```cmake
# -fno-math-errno: rustc lowers sin/cos/exp/pow/sqrt to LLVM intrinsics (no errno); Clang matches
# that lowering only with this flag (Darwin default, NOT the Linux default) — plan C ruling.
```

Create `cpp/libs/sapient-backends-cpu/CMakeLists.txt`:

```cmake
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
# Port of crates/sapient-backends/cpu (sub-project 1a plans C/D/E). `parallel` stands in for rayon,
# `sgemm` for matrixmultiply, `cpu_features` for is_*_feature_detected!; thermal/spinpool are
# plan-E stubs until plan E lands.
find_package(Threads REQUIRED)
add_library(sapient_backends_cpu STATIC
  src/cpu_features.cpp src/parallel.cpp src/thermal.cpp src/spinpool.cpp)
add_library(sapient::backends_cpu ALIAS sapient_backends_cpu)
target_include_directories(sapient_backends_cpu PUBLIC include)
target_link_libraries(sapient_backends_cpu PUBLIC sapient::core Threads::Threads)
sapient_apply_warnings(sapient_backends_cpu)

if(SAPIENT_BUILD_TESTS)
  add_executable(sapient_backends_cpu_tests
    tests/cpu_features_test.cpp tests/parallel_test.cpp)
  target_link_libraries(sapient_backends_cpu_tests PRIVATE sapient::backends_cpu sapient::testing GTest::gtest_main)
  sapient_apply_warnings(sapient_backends_cpu_tests)
  gtest_discover_tests(sapient_backends_cpu_tests)
endif()
```

- [ ] **Step 4: `cpu_features.hpp` / `.cpp`**

`include/sapient/backends_cpu/cpu_features.hpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Cached runtime ISA probes — the twins of Rust's `is_aarch64_feature_detected!("dotprod"/"i8mm")`
// and `is_x86_feature_detected!("avx2") && ("fma")` in sapient-backends-cpu (porting map §2.1).
// NEON is compile-time on aarch64 and never probed, exactly as in Rust.

namespace sapient::backends_cpu::cpu_features {

/// aarch64 `dotprod` (ARMv8.2 SDOT); always false on other architectures.
bool has_dotprod();
/// aarch64 `i8mm` (ARMv8.6 SMMLA); always false elsewhere — including Windows/arm64, where
/// std_detect exposes no i8mm probe (not a release target).
bool has_i8mm();
/// x86_64 `avx2 && fma`, gated like std_detect on OS support for the YMM state (OSXSAVE and
/// XCR0 bits 1|2); always false elsewhere.
bool has_avx2_fma();

} // namespace sapient::backends_cpu::cpu_features
```

`src/cpu_features.cpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/backends_cpu/cpu_features.hpp"

#include <cstddef>
#include <cstdint>

#if defined(__aarch64__) || defined(_M_ARM64)
#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1UL << 20)
#endif
#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM (1UL << 13)
#endif
#elif defined(_WIN32)
#include <windows.h>
#ifndef PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE
#define PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE 43
#endif
#endif
#elif defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace sapient::backends_cpu::cpu_features {
namespace {

#if defined(__aarch64__) || defined(_M_ARM64)
#if defined(__APPLE__)
bool sysctl_flag(const char* name) {
    int v = 0;
    size_t len = sizeof(v);
    return ::sysctlbyname(name, &v, &len, nullptr, 0) == 0 && v != 0;
}
#endif
bool probe_dotprod() {
#if defined(__APPLE__)
    return sysctl_flag("hw.optional.arm.FEAT_DotProd");
#elif defined(__linux__)
    return (::getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#elif defined(_WIN32)
    return ::IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE) != 0;
#else
    return false;
#endif
}
bool probe_i8mm() {
#if defined(__APPLE__)
    return sysctl_flag("hw.optional.arm.FEAT_I8MM");
#elif defined(__linux__)
    return (::getauxval(AT_HWCAP2) & HWCAP2_I8MM) != 0;
#else
    return false;
#endif
}
bool probe_avx2_fma() { return false; }

#elif defined(__x86_64__) || defined(_M_X64)
bool probe_dotprod() { return false; }
bool probe_i8mm() { return false; }

struct CpuidRegs {
    uint32_t eax{0}, ebx{0}, ecx{0}, edx{0};
};
CpuidRegs cpuid(uint32_t leaf, uint32_t sub) {
    CpuidRegs r;
#if defined(_MSC_VER)
    int out[4] = {0, 0, 0, 0};
    __cpuidex(out, static_cast<int>(leaf), static_cast<int>(sub));
    r.eax = static_cast<uint32_t>(out[0]);
    r.ebx = static_cast<uint32_t>(out[1]);
    r.ecx = static_cast<uint32_t>(out[2]);
    r.edx = static_cast<uint32_t>(out[3]);
#else
    __cpuid_count(leaf, sub, r.eax, r.ebx, r.ecx, r.edx);
#endif
    return r;
}
uint64_t xgetbv0() {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    uint32_t eax = 0;
    uint32_t edx = 0;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<uint64_t>(edx) << 32) | eax;
#endif
}
// std_detect (os/x86.rs): FMA (leaf 1 ECX bit 12) and AVX2 (leaf 7 EBX bit 5) are reported only
// when the OS saves the YMM state — OSXSAVE (leaf 1 ECX bit 27) and XCR0 bits 1 and 2 set.
bool probe_avx2_fma() {
    if (cpuid(0, 0).eax < 7) return false;
    const CpuidRegs l1 = cpuid(1, 0);
    if ((l1.ecx & (1u << 27)) == 0) return false;
    if ((xgetbv0() & 0x6u) != 0x6u) return false;
    const bool fma = (l1.ecx & (1u << 12)) != 0;
    const bool avx2 = (cpuid(7, 0).ebx & (1u << 5)) != 0;
    return fma && avx2;
}

#else
bool probe_dotprod() { return false; }
bool probe_i8mm() { return false; }
bool probe_avx2_fma() { return false; }
#endif

} // namespace

bool has_dotprod() {
    static const bool v = probe_dotprod();
    return v;
}
bool has_i8mm() {
    static const bool v = probe_i8mm();
    return v;
}
bool has_avx2_fma() {
    static const bool v = probe_avx2_fma();
    return v;
}

} // namespace sapient::backends_cpu::cpu_features
```

- [ ] **Step 4b: `env.hpp`** — `include/sapient/backends_cpu/env.hpp` (header-only; also used by Task 7's `SAPIENT_GEMV_TPC` and plan E's spinpool knobs)

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Rust-compatible env-knob parsing, shared by the rayon twin (RAYON_NUM_THREADS), matmul
// (SAPIENT_GEMV_TPC) and plan E's spinpool knobs — the C++ spelling of
// `std::env::var(name).ok().and_then(|v| v.parse::<usize>().ok())`.

#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <optional>
#include <string_view>
#include <system_error>

namespace sapient::backends_cpu {

/// `usize::from_str` applied to `getenv(name)`: an optional leading '+', then decimal digits only
/// (no whitespace, no '-'). nullopt when the variable is unset, empty or unparsable.
inline std::optional<size_t> env_usize(const char* name) {
    const char* s = std::getenv(name);
    if (s == nullptr) return std::nullopt;
    std::string_view v(s);
    if (!v.empty() && v.front() == '+') v.remove_prefix(1);
    if (v.empty()) return std::nullopt;
    size_t out = 0;
    const auto r = std::from_chars(v.data(), v.data() + v.size(), out);
    if (r.ec != std::errc{} || r.ptr != v.data() + v.size()) return std::nullopt;
    return out;
}

} // namespace sapient::backends_cpu
```

- [ ] **Step 5: `parallel.hpp` / `.cpp`**

`include/sapient/backends_cpu/parallel.hpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// The rayon stand-in (spec §2.3 `parallel`). rayon is not ported: what parity depends on is the
// chunk→range PARTITION of `par_chunks_mut`, which is deterministic and thread-independent, so a
// persistent pool that hands out chunk indices reproduces every kernel's per-slot result exactly.
// Work stealing, `join`, and rayon's sleep protocol are deliberately absent.

#include <cstddef>
#include <functional>
#include <span>

namespace sapient::backends_cpu::parallel {

/// `rayon::current_num_threads()` of the global pool (rayon-core 1.13 rules): `RAYON_NUM_THREADS`
/// parsed as usize — ≥1 → that; 0 → the default; unset/unparsable → `RAYON_RS_NUM_CPUS` with the
/// same rule; else the logical CPU count (≥1). Computed once, at first call.
size_t num_threads();

/// Calls `f(i)` for every `i` in `[0, n)` exactly once, on the calling thread and the pool's
/// workers, and returns when all have completed. Re-entrant: `f` may itself call `par_for`.
void par_for(size_t n, const std::function<void(size_t)>& f);

/// rayon `out.par_chunks_mut(chunk).enumerate().for_each(|(ci, cs)| f(ci, cs))`: chunk `ci` is
/// `out[ci*chunk, min((ci+1)*chunk, out.size()))`. An empty `out` makes no calls. `chunk == 0`
/// panics (rayon: "chunk size must not be zero").
void par_chunks_mut(std::span<float> out,
                    size_t chunk,
                    const std::function<void(size_t, std::span<float>)>& f);

} // namespace sapient::backends_cpu::parallel
```

`src/parallel.cpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/backends_cpu/parallel.hpp"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <thread>

#include "sapient/backends_cpu/env.hpp"
#include "sapient/core/panic.hpp"

namespace sapient::backends_cpu::parallel {
namespace {

size_t default_threads() {
    const unsigned hc = std::thread::hardware_concurrency();
    return hc == 0 ? 1 : static_cast<size_t>(hc);
}

// rayon-core 1.13 ThreadPoolBuilder::get_num_threads: Some(x ≥ 1) → x, Some(0) → default,
// None → fall through to the deprecated RAYON_RS_NUM_CPUS, then the default.
size_t compute_num_threads() {
    for (const char* name : {"RAYON_NUM_THREADS", "RAYON_RS_NUM_CPUS"}) {
        const auto v = env_usize(name);
        if (!v.has_value()) continue;
        return *v >= 1 ? *v : default_threads();
    }
    return default_threads();
}

struct Job {
    const std::function<void(size_t)>* f;
    size_t n;
    size_t next{0}; // guarded by Pool::mu_
    size_t done{0}; // guarded by Pool::mu_
};

// A job queue with the caller always participating in its own job: progress never depends on a
// worker being free, so nested par_for (a chunk that itself calls par_for) cannot deadlock —
// every waiting thread only waits on chunks that some running thread has already claimed.
class Pool {
public:
    // Leaked on purpose, like rayon's global registry: worker threads are detached and must never
    // outlive their mutex/condvars, which a static destructor at exit would destroy.
    static Pool& instance() {
        static Pool* p = new Pool();
        return *p;
    }

    void run(Job& job) {
        if (workers_ == 0) {
            for (size_t i = 0; i < job.n; ++i) (*job.f)(i);
            return;
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            queue_.push_back(&job);
        }
        cv_.notify_all();
        while (true) {
            size_t i = 0;
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (job.next >= job.n) {
                    erase_locked(&job);
                    break;
                }
                i = job.next++;
            }
            (*job.f)(i);
            finish(job);
        }
        std::unique_lock<std::mutex> lk(mu_);
        done_cv_.wait(lk, [&] { return job.done == job.n; });
    }

private:
    Pool() : workers_(num_threads() - 1) {
        for (size_t w = 0; w < workers_; ++w) std::thread([this] { worker(); }).detach();
    }

    void worker() {
        while (true) {
            Job* job = nullptr;
            size_t i = 0;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return !queue_.empty(); });
                job = queue_.front();
                if (job->next >= job->n) {
                    queue_.pop_front(); // fully claimed; its owner is waiting on `done`
                    continue;
                }
                i = job->next++;
            }
            (*job->f)(i);
            finish(*job);
        }
    }

    void finish(Job& job) {
        std::lock_guard<std::mutex> lk(mu_);
        if (++job.done == job.n) done_cv_.notify_all();
    }

    void erase_locked(Job* job) {
        const auto it = std::find(queue_.begin(), queue_.end(), job);
        if (it != queue_.end()) queue_.erase(it);
    }

    std::mutex mu_;
    std::condition_variable cv_;
    std::condition_variable done_cv_;
    std::deque<Job*> queue_;
    size_t workers_;
};

} // namespace

size_t num_threads() {
    static const size_t n = compute_num_threads();
    return n;
}

void par_for(size_t n, const std::function<void(size_t)>& f) {
    if (n == 0) return;
    Job job{&f, n};
    Pool::instance().run(job);
}

void par_chunks_mut(std::span<float> out,
                    size_t chunk,
                    const std::function<void(size_t, std::span<float>)>& f) {
    if (chunk == 0) sapient::core::panic("par_chunks_mut: chunk size must not be zero");
    if (out.empty()) return;
    const size_t len = out.size();
    const size_t n_chunks = (len + chunk - 1) / chunk;
    par_for(n_chunks, [&](size_t ci) {
        const size_t start = ci * chunk;
        const size_t end = std::min(start + chunk, len);
        f(ci, out.subspan(start, end - start));
    });
}

} // namespace sapient::backends_cpu::parallel
```

Notes for the implementer: the pool serialises chunk *claiming* under one mutex (chunks are coarse — ≥16 GEMV rows, one attention head, one im2col row — so this costs microseconds per chunk and buys a simple, provably deadlock-free design). A `Job` lives on the caller's stack; it is only reachable through `queue_` (under `mu_`) or by a worker already executing one of its chunks, and `run` cannot return before `done == n`, so no worker ever touches a dead job. `std::function` is taken by `const&` (tidy `performance-unnecessary-value-param`). If `num_threads()` is 1 there are no workers and everything runs on the caller.

- [ ] **Step 6: The plan-E stubs**

`include/sapient/backends_cpu/thermal.hpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-backends/cpu/src/thermal.rs — PLAN E. Plan C ships only the two entry
// points `matmul` needs, with inert bodies that equal Rust's behaviour on a host without thermal
// zones and no external level: plan E replaces the bodies (governor, external cap, tick sampling)
// and adds ThermalGovernor, set_external_thermal_level, external_thermal_level.

#include <cstddef>

namespace sapient::backends_cpu::thermal {

/// Rust `thermal::effective_threads()`: the stricter of the sysfs governor and the external level.
/// Plan C: always `parallel::num_threads()` (no governor, level 0).
size_t effective_threads();

/// Rust `thermal::tick()`: rate-limited governor sample at the top of `matmul_nt`.
/// Plan C: no-op (no governor).
void tick();

} // namespace sapient::backends_cpu::thermal
```

`src/thermal.cpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/backends_cpu/thermal.hpp"

#include "sapient/backends_cpu/parallel.hpp"

namespace sapient::backends_cpu::thermal {

size_t effective_threads() { return parallel::num_threads(); } // plan E: governor/external min

void tick() {} // plan E: 500 ms rate-limited sysfs sample

} // namespace sapient::backends_cpu::thermal
```

`include/sapient/backends_cpu/spinpool.hpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-backends/cpu/src/spinpool.rs — PLAN E. Plan C ships only the two entry
// points `matmul::gemv_chunk` needs, with the values Rust yields under `SAPIENT_SPINPOOL=0`:
// plan E replaces the bodies and adds SpinPool, pool(), and the env knobs.

#include <cstddef>

namespace sapient::backends_cpu::spinpool {

/// Rust `spinpool::enabled()`: env/platform default AND thermal::effective_threads() >=
/// parallel::num_threads(). Plan C: always false (the pool does not exist yet).
bool enabled();

/// Rust `spinpool::parallelism()` = workers + 1. Plan C: `parallel::num_threads()`.
size_t parallelism();

} // namespace sapient::backends_cpu::spinpool
```

`src/spinpool.cpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/backends_cpu/spinpool.hpp"

#include "sapient/backends_cpu/parallel.hpp"

namespace sapient::backends_cpu::spinpool {

bool enabled() { return false; } // plan E: SAPIENT_SPINPOOL / platform default && thermal check

size_t parallelism() { return parallel::num_threads(); } // plan E: workers + 1

} // namespace sapient::backends_cpu::spinpool
```

- [ ] **Step 7: Build and run the tests**

Run: `cd cpp && cmake --preset dev && cmake --build --preset dev && ctest --preset dev -R 'CpuFeatures|Parallel'`
Expected: 7 tests pass (`CpuFeatures.*` 1, `Parallel.*` 5, `ParallelDeath.*` 1). Also run the full suite once (`ctest --preset dev`): the plan A tests still pass with `-fno-math-errno` (79 + 7 = 86 entries).

- [ ] **Step 8: Format, then commit**

```bash
git add cpp/cmake/flags.cmake cpp/libs/CMakeLists.txt cpp/libs/sapient-backends-cpu
git ls-files -- 'cpp/*.hpp' 'cpp/*.cpp' | xargs .superpowers/tools-venv/bin/clang-format -i
git add cpp/libs/sapient-backends-cpu
git commit -m "cpp(backends-cpu): library skeleton, cpu_features probes, rayon-twin parallel pool, plan-E stubs

-fno-math-errno joins the parity flags so Clang lowers libm calls to the same LLVM
intrinsics rustc emits on every OS (Darwin already defaulted to it; Linux did not).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `sgemm.hpp/.cpp` — the `matrixmultiply::sgemm` stand-in

**Files:**
- Modify: `cpp/libs/sapient-backends-cpu/CMakeLists.txt` (add `src/sgemm.cpp`, `tests/sgemm_test.cpp`)
- Create: `include/sapient/backends_cpu/sgemm.hpp`, `src/sgemm.cpp`
- Test: `tests/sgemm_test.cpp`

**Interfaces:**
- Consumes: `cpu_features::has_avx2_fma()` (Task 1, x86_64 only).
- Produces: `void sapient::backends_cpu::sgemm(size_t m, size_t k, size_t n, float alpha, const float* a, std::ptrdiff_t rsa, std::ptrdiff_t csa, const float* b, std::ptrdiff_t rsb, std::ptrdiff_t csb, float beta, float* c, std::ptrdiff_t rsc, std::ptrdiff_t csc)` — the exact `matrixmultiply::sgemm` signature (`isize` strides → `std::ptrdiff_t`), `C = alpha·A·B + beta·C`, `beta == 0` ⇒ `C` is never read. Tasks 6 and 7 call it with the four Rust call-site stride patterns (porting map §8b).

**Design (spec §2.3, approved deviation: max-error gated, not bit-identical):** pack a `kc × 4` strip of `B` columns (zero-padded past `n`) and an `mc × kc` block of `A`, then for each row and strip run a 4-wide FMA loop over `p` (`vfmaq_f32` on aarch64, `_mm_fmadd_ps` on x86_64 with FMA, plain `acc + a*b` otherwise). Blocking constants `KC=256, MC=64, NC=1024`. **Requirement beyond the tolerance:** the callers (`matmul_nt_float` prefill, `conv2d`) split `m` into `mblock` row blocks sized by `num_threads()`, so `C[i][j]` must not depend on `m`, `n`, `i0`, or `j0` — otherwise the C++ result would vary with `RAYON_NUM_THREADS`. The design gives that for free: the `p`-order is fixed by `KC`, zero-padded strips make every column's arithmetic identical, and the per-panel `alpha·acc` accumulation into `C` is per element. Test 3 pins it.

- [ ] **Step 1: Write the failing tests** — `tests/sgemm_test.cpp`

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "sapient/backends_cpu/sgemm.hpp"

using sapient::backends_cpu::sgemm;

namespace {

struct Lcg {
    uint64_t s;
    float next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return (static_cast<float>(s >> 40) / static_cast<float>(1ULL << 24)) * 2.0f - 1.0f;
    }
};

std::vector<float> rand_vec(size_t n, uint64_t seed) {
    Lcg g{seed};
    std::vector<float> v(n);
    for (float& x : v) x = g.next();
    return v;
}

// Reference in double: C = alpha·A·B + beta·C with explicit strides.
std::vector<float> naive(size_t m, size_t k, size_t n, float alpha, const float* a, std::ptrdiff_t rsa,
                         std::ptrdiff_t csa, const float* b, std::ptrdiff_t rsb, std::ptrdiff_t csb,
                         float beta, const std::vector<float>& c_in) {
    std::vector<float> out(m * n, 0.0f);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j) {
            double acc = 0.0;
            for (size_t p = 0; p < k; ++p)
                acc += static_cast<double>(a[static_cast<std::ptrdiff_t>(i) * rsa +
                                              static_cast<std::ptrdiff_t>(p) * csa]) *
                       static_cast<double>(b[static_cast<std::ptrdiff_t>(p) * rsb +
                                              static_cast<std::ptrdiff_t>(j) * csb]);
            const double prev = beta == 0.0f ? 0.0 : static_cast<double>(beta) * c_in[i * n + j];
            out[i * n + j] = static_cast<float>(static_cast<double>(alpha) * acc + prev);
        }
    return out;
}

float max_abs(const std::vector<float>& v) {
    float m = 0.0f;
    for (float x : v) m = std::max(m, std::fabs(x));
    return m;
}

void expect_close(const std::vector<float>& got, const std::vector<float>& ref, float rel) {
    ASSERT_EQ(got.size(), ref.size());
    const float tol = rel * std::max(1.0f, max_abs(ref));
    for (size_t i = 0; i < got.size(); ++i)
        ASSERT_NEAR(got[i], ref[i], tol) << "index " << i;
}

} // namespace

// C++-only tests (matrixmultiply has none we could port). Tolerance 1e-4·max(1, max|ref|) with k ≤ 300
// and |values| ≤ 1: a float re-ordering error is ~1e-6, an indexing/stride bug is O(1).
TEST(Sgemm, matches_naive_reference_over_shapes) {
    const size_t shapes[][3] = {{1, 1, 1}, {3, 5, 7}, {17, 300, 9}, {65, 257, 1030}, {2, 64, 16}, {4, 3, 1}};
    uint64_t seed = 1;
    for (const auto& s : shapes) {
        const size_t m = s[0], k = s[1], n = s[2];
        const auto a = rand_vec(m * k, seed++);
        const auto b = rand_vec(k * n, seed++);
        std::vector<float> c(m * n, 0.0f);
        sgemm(m, k, n, 1.0f, a.data(), static_cast<std::ptrdiff_t>(k), 1, b.data(),
              static_cast<std::ptrdiff_t>(n), 1, 0.0f, c.data(), static_cast<std::ptrdiff_t>(n), 1);
        const auto ref = naive(m, k, n, 1.0f, a.data(), static_cast<std::ptrdiff_t>(k), 1, b.data(),
                               static_cast<std::ptrdiff_t>(n), 1, 0.0f, c);
        expect_close(c, ref, 1e-4f);
    }
}

// The four Rust call-site patterns: matmul (k,1)/(n,1); matmul_nt prefill B transposed via (1,k);
// gemm with arbitrary strides (here A transposed via (1,m)); conv2d (k,1)/(n2,1).
TEST(Sgemm, transposed_operands_via_strides) {
    const size_t m = 6, k = 70, n = 11;
    const auto at = rand_vec(k * m, 7); // A(i,p) = at[p*m + i]  → rsa = 1, csa = m
    const auto bt = rand_vec(n * k, 8); // B(p,j) = bt[j*k + p]  → rsb = 1, csb = k
    std::vector<float> c(m * n, 0.0f);
    sgemm(m, k, n, 1.0f, at.data(), 1, static_cast<std::ptrdiff_t>(m), bt.data(), 1,
          static_cast<std::ptrdiff_t>(k), 0.0f, c.data(), static_cast<std::ptrdiff_t>(n), 1);
    const auto ref = naive(m, k, n, 1.0f, at.data(), 1, static_cast<std::ptrdiff_t>(m), bt.data(), 1,
                           static_cast<std::ptrdiff_t>(k), 0.0f, c);
    expect_close(c, ref, 1e-4f);

    // Output with a column stride (csc = 2, interleaved into a 2× buffer).
    std::vector<float> c2(m * n * 2, 0.0f);
    sgemm(m, k, n, 1.0f, at.data(), 1, static_cast<std::ptrdiff_t>(m), bt.data(), 1,
          static_cast<std::ptrdiff_t>(k), 0.0f, c2.data(), static_cast<std::ptrdiff_t>(2 * n), 2);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            EXPECT_EQ(std::bit_cast<uint32_t>(c2[i * 2 * n + 2 * j]), std::bit_cast<uint32_t>(c[i * n + j]));
}

// matmul_nt_float and conv2d split rows into thread-count-sized blocks; the result must not
// depend on the blocking (or RAYON_NUM_THREADS would change the model's numbers).
TEST(Sgemm, row_and_column_blocks_are_bit_identical_to_the_full_call) {
    const size_t m = 37, k = 300, n = 53;
    const auto a = rand_vec(m * k, 21);
    const auto b = rand_vec(k * n, 22);
    std::vector<float> full(m * n, 0.0f);
    sgemm(m, k, n, 1.0f, a.data(), static_cast<std::ptrdiff_t>(k), 1, b.data(),
          static_cast<std::ptrdiff_t>(n), 1, 0.0f, full.data(), static_cast<std::ptrdiff_t>(n), 1);

    std::vector<float> rows(m * n, 0.0f);
    const size_t mblock = 8;
    for (size_t m0 = 0; m0 < m; m0 += mblock) {
        const size_t mc = std::min(mblock, m - m0);
        sgemm(mc, k, n, 1.0f, a.data() + m0 * k, static_cast<std::ptrdiff_t>(k), 1, b.data(),
              static_cast<std::ptrdiff_t>(n), 1, 0.0f, rows.data() + m0 * n, static_cast<std::ptrdiff_t>(n), 1);
    }
    std::vector<float> cols(m * n, 0.0f);
    const size_t split = 20;
    sgemm(m, k, split, 1.0f, a.data(), static_cast<std::ptrdiff_t>(k), 1, b.data(),
          static_cast<std::ptrdiff_t>(n), 1, 0.0f, cols.data(), static_cast<std::ptrdiff_t>(n), 1);
    sgemm(m, k, n - split, 1.0f, a.data(), static_cast<std::ptrdiff_t>(k), 1, b.data() + split,
          static_cast<std::ptrdiff_t>(n), 1, 0.0f, cols.data() + split, static_cast<std::ptrdiff_t>(n), 1);
    for (size_t i = 0; i < m * n; ++i) {
        EXPECT_EQ(std::bit_cast<uint32_t>(rows[i]), std::bit_cast<uint32_t>(full[i])) << "row-blocked " << i;
        EXPECT_EQ(std::bit_cast<uint32_t>(cols[i]), std::bit_cast<uint32_t>(full[i])) << "col-blocked " << i;
    }
}

// matrixmultiply: beta == 0 means C is write-only (a NaN-filled C must not leak in).
TEST(Sgemm, beta_zero_never_reads_c) {
    const size_t m = 3, k = 9, n = 4;
    const auto a = rand_vec(m * k, 31);
    const auto b = rand_vec(k * n, 32);
    std::vector<float> c(m * n, std::numeric_limits<float>::quiet_NaN());
    sgemm(m, k, n, 1.0f, a.data(), static_cast<std::ptrdiff_t>(k), 1, b.data(),
          static_cast<std::ptrdiff_t>(n), 1, 0.0f, c.data(), static_cast<std::ptrdiff_t>(n), 1);
    const auto ref = naive(m, k, n, 1.0f, a.data(), static_cast<std::ptrdiff_t>(k), 1, b.data(),
                           static_cast<std::ptrdiff_t>(n), 1, 0.0f, c);
    for (float v : c) EXPECT_TRUE(std::isfinite(v));
    expect_close(c, ref, 1e-4f);
}

TEST(Sgemm, alpha_and_beta_scale) {
    const size_t m = 5, k = 33, n = 6;
    const auto a = rand_vec(m * k, 41);
    const auto b = rand_vec(k * n, 42);
    const auto c0 = rand_vec(m * n, 43);
    auto c = c0;
    sgemm(m, k, n, 2.0f, a.data(), static_cast<std::ptrdiff_t>(k), 1, b.data(),
          static_cast<std::ptrdiff_t>(n), 1, 0.5f, c.data(), static_cast<std::ptrdiff_t>(n), 1);
    const auto ref = naive(m, k, n, 2.0f, a.data(), static_cast<std::ptrdiff_t>(k), 1, b.data(),
                           static_cast<std::ptrdiff_t>(n), 1, 0.5f, c0);
    expect_close(c, ref, 1e-4f);
}

TEST(Sgemm, zero_k_or_zero_alpha_only_applies_beta) {
    const size_t m = 2, n = 3;
    const auto c0 = rand_vec(m * n, 51);
    const auto a = rand_vec(m * 4, 52);
    const auto b = rand_vec(4 * n, 53);
    auto c = c0;
    sgemm(m, 0, n, 1.0f, a.data(), 4, 1, b.data(), static_cast<std::ptrdiff_t>(n), 1, 0.5f, c.data(),
          static_cast<std::ptrdiff_t>(n), 1);
    for (size_t i = 0; i < m * n; ++i) EXPECT_EQ(c[i], 0.5f * c0[i]);
    c = c0;
    sgemm(m, 4, n, 0.0f, a.data(), 4, 1, b.data(), static_cast<std::ptrdiff_t>(n), 1, 0.0f, c.data(),
          static_cast<std::ptrdiff_t>(n), 1);
    for (float v : c) EXPECT_EQ(v, 0.0f);
}
```

- [ ] **Step 2: Add the files to CMake and build to verify failure**

In `CMakeLists.txt` add `src/sgemm.cpp` to the library sources and `tests/sgemm_test.cpp` to the test sources.
Run: `cd cpp && cmake --preset dev && cmake --build --preset dev`
Expected: FAILS — `'sapient/backends_cpu/sgemm.hpp' file not found`.

- [ ] **Step 3: `sgemm.hpp`**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// The `matrixmultiply::sgemm` stand-in (spec §2.3; approved deviation: MAX-ERROR gated against the
// Rust build, never bit-identical — matrixmultiply's blocking is not reproducible without porting
// it, and performance is not a 1a goal). What IS required: C[i][j] must not depend on m, n, or
// which rows/columns a call covers, because matmul_nt_float and conv2d split rows into blocks
// sized by the thread count and the model's numbers must not vary with RAYON_NUM_THREADS.

#include <cstddef>

namespace sapient::backends_cpu {

/// C = alpha·A·B + beta·C for A (m×k), B (k×n), C (m×n) addressed as a[i*rsa + p*csa],
/// b[p*rsb + j*csb], c[i*rsc + j*csc]. `beta == 0` ⇒ C is write-only (may hold NaN on entry).
/// `alpha == 0` or `k == 0` ⇒ C = beta·C.
void sgemm(size_t m,
           size_t k,
           size_t n,
           float alpha,
           const float* a,
           std::ptrdiff_t rsa,
           std::ptrdiff_t csa,
           const float* b,
           std::ptrdiff_t rsb,
           std::ptrdiff_t csb,
           float beta,
           float* c,
           std::ptrdiff_t rsc,
           std::ptrdiff_t csc);

} // namespace sapient::backends_cpu
```

- [ ] **Step 4: `sgemm.cpp`**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/backends_cpu/sgemm.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

#include "sapient/backends_cpu/cpu_features.hpp"
#endif

namespace sapient::backends_cpu {
namespace {

constexpr size_t KC = 256;  // k panel: the fixed p-order that makes results blocking-independent
constexpr size_t MC = 64;   // A rows packed per block
constexpr size_t NC = 1024; // B columns packed per panel

inline std::ptrdiff_t idx(size_t i, size_t j, std::ptrdiff_t rs, std::ptrdiff_t cs) {
    return static_cast<std::ptrdiff_t>(i) * rs + static_cast<std::ptrdiff_t>(j) * cs;
}

// acc[q] += Σ_p a[p] · bp[4p + q] for q in 0..4, where `bp` is one packed kc×4 strip of B.
// FMA where the ISA has it (single rounding per step), plain mul+add otherwise; the scalar
// fallback is only reached on hosts with neither NEON nor AVX2+FMA.
#if defined(__aarch64__) || defined(_M_ARM64)
void strip4(size_t kc, const float* a, const float* bp, float* acc) {
    float32x4_t v = vld1q_f32(acc);
    for (size_t p = 0; p < kc; ++p) v = vfmaq_f32(v, vld1q_f32(bp + 4 * p), vdupq_n_f32(a[p]));
    vst1q_f32(acc, v);
}
#else
void strip4_scalar(size_t kc, const float* a, const float* bp, float* acc) {
    for (size_t p = 0; p < kc; ++p)
        for (size_t q = 0; q < 4; ++q) acc[q] = acc[q] + a[p] * bp[4 * p + q];
}
#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx2,fma"))) void
strip4_fma(size_t kc, const float* a, const float* bp, float* acc) {
    __m128 v = _mm_loadu_ps(acc);
    for (size_t p = 0; p < kc; ++p) v = _mm_fmadd_ps(_mm_loadu_ps(bp + 4 * p), _mm_set1_ps(a[p]), v);
    _mm_storeu_ps(acc, v);
}
void strip4(size_t kc, const float* a, const float* bp, float* acc) {
    if (cpu_features::has_avx2_fma())
        strip4_fma(kc, a, bp, acc);
    else
        strip4_scalar(kc, a, bp, acc);
}
#else
void strip4(size_t kc, const float* a, const float* bp, float* acc) { strip4_scalar(kc, a, bp, acc); }
#endif
#endif

} // namespace

void sgemm(size_t m,
           size_t k,
           size_t n,
           float alpha,
           const float* a,
           std::ptrdiff_t rsa,
           std::ptrdiff_t csa,
           const float* b,
           std::ptrdiff_t rsb,
           std::ptrdiff_t csb,
           float beta,
           float* c,
           std::ptrdiff_t rsc,
           std::ptrdiff_t csc) {
    if (m == 0 || n == 0) return;
    // beta pass: matrixmultiply never reads C when beta == 0.
    if (beta == 0.0f) {
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j) c[idx(i, j, rsc, csc)] = 0.0f;
    } else if (beta != 1.0f) {
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j) c[idx(i, j, rsc, csc)] *= beta;
    }
    if (k == 0 || alpha == 0.0f) return;

    std::vector<float> bp;
    std::vector<float> ap;
    for (size_t j0 = 0; j0 < n; j0 += NC) {
        const size_t nc = std::min(NC, n - j0);
        const size_t strips = (nc + 3) / 4;
        for (size_t p0 = 0; p0 < k; p0 += KC) {
            const size_t kc = std::min(KC, k - p0);
            // Pack B[p0..p0+kc, j0..j0+nc) as `strips` contiguous kc×4 strips, zero-padded past nc so
            // every column — including the ragged last strip — sees identical arithmetic.
            bp.assign(strips * kc * 4, 0.0f);
            for (size_t s = 0; s < strips; ++s)
                for (size_t p = 0; p < kc; ++p)
                    for (size_t q = 0; q < 4; ++q) {
                        const size_t j = 4 * s + q;
                        if (j < nc) bp[(s * kc + p) * 4 + q] = b[idx(p0 + p, j0 + j, rsb, csb)];
                    }
            for (size_t i0 = 0; i0 < m; i0 += MC) {
                const size_t mc = std::min(MC, m - i0);
                ap.resize(mc * kc);
                for (size_t i = 0; i < mc; ++i)
                    for (size_t p = 0; p < kc; ++p) ap[i * kc + p] = a[idx(i0 + i, p0 + p, rsa, csa)];
                for (size_t i = 0; i < mc; ++i)
                    for (size_t s = 0; s < strips; ++s) {
                        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        strip4(kc, ap.data() + i * kc, bp.data() + s * kc * 4, acc);
                        for (size_t q = 0; q < 4; ++q) {
                            const size_t j = 4 * s + q;
                            if (j < nc) c[idx(i0 + i, j0 + j, rsc, csc)] += alpha * acc[q];
                        }
                    }
            }
        }
    }
}

} // namespace sapient::backends_cpu
```

- [ ] **Step 5: Build and run**

Run: `cd cpp && cmake --build --preset dev && ctest --preset dev -R Sgemm`
Expected: 6 tests pass.

- [ ] **Step 6: Format, then commit**

```bash
git add cpp/libs/sapient-backends-cpu
git ls-files -- 'cpp/*.hpp' 'cpp/*.cpp' | xargs .superpowers/tools-venv/bin/clang-format -i
git add cpp/libs/sapient-backends-cpu
git commit -m "cpp(backends-cpu): sgemm — packed, cache-blocked matrixmultiply stand-in (max-error gated)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `kernels/elementwise.hpp/.cpp`

**Files:**
- Modify: `cpp/libs/sapient-backends-cpu/CMakeLists.txt` (add `src/kernels/elementwise.cpp`, `tests/elementwise_test.cpp`)
- Create: `include/sapient/backends_cpu/kernels/elementwise.hpp`, `src/kernels/elementwise.cpp`
- Test: `tests/elementwise_test.cpp`

**Interfaces:**
- Consumes: `sapient::core::{Tensor, Shape, DType, Result, Error, to_string(DType)}`.
- Produces (namespace `sapient::backends_cpu::kernels::elementwise`): `Result<Tensor> add/sub/mul/div/pow(const Tensor&, const Tensor&)`; `Result<Tensor> neg/abs/sqrt/exp/log/erf/floor/ceil/round/relu/sigmoid/tanh_act/gelu/gelu_erf/silu/hard_swish(const Tensor&)`; `Result<Tensor> leaky_relu(const Tensor&, float alpha)`; `Result<Tensor> clip(const Tensor&, std::optional<float> min, std::optional<float> max)`; `float erf_approx(float)` (Rust-private, exposed because `test_erf` calls it). Task 8's dumps gate `silu`, `gelu_erf`, `gelu`.

**Rust behaviour to mirror (elementwise.rs, porting map §8):** `unary_f32` returns `TypeMismatch{expected: "f32", got: dtype.to_string()}` for non-F32 input; `binary_f32` has no dtype check, zips equal-length data, else broadcasts whichever side has length 1, else `ShapeMismatch{expected: a.dims, got: b.dims}`; the output shape is `a`'s (or `b`'s when `a` is the scalar). Both read the **unbounded** `to_f32_cow()` (rule 7). No panics are reachable.

- [ ] **Step 1: Write the failing tests** — `tests/elementwise_test.cpp`

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "sapient/backends_cpu/kernels/elementwise.hpp"
#include "sapient/core/tensor.hpp"

using namespace sapient::backends_cpu::kernels::elementwise;
using sapient::core::Shape;
using sapient::core::Tensor;

namespace {
// Rust: fn t(data: &[f32]) -> Tensor { Tensor::from_f32(data, vec![data.len()]).unwrap() }
Tensor t(std::vector<float> data) {
    const size_t n = data.size();
    auto r = Tensor::from_f32_vec(std::move(data), Shape{n});
    if (!r) throw std::runtime_error(r.error().to_string());
    return std::move(*r);
}
std::vector<float> vec(const Tensor& x) {
    const auto s = x.f32_slice();
    return {s.begin(), s.end()};
}
} // namespace

TEST(Elementwise, test_add) {
    auto r = add(t({1.0f, 2.0f}), t({3.0f, 4.0f}));
    ASSERT_TRUE(r.has_value());
    EXPECT_LT(std::fabs(r->f32_slice()[0] - 4.0f), 1e-6f);
}

TEST(Elementwise, test_relu) {
    auto r = relu(t({-1.0f, 0.0f, 1.0f}));
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(vec(*r), (std::vector<float>{0.0f, 0.0f, 1.0f}));
}

TEST(Elementwise, test_sigmoid) {
    auto r = sigmoid(t({0.0f}));
    ASSERT_TRUE(r.has_value());
    EXPECT_LT(std::fabs(r->f32_slice()[0] - 0.5f), 1e-6f);
}

TEST(Elementwise, test_gelu) {
    auto r = gelu(t({0.0f}));
    ASSERT_TRUE(r.has_value());
    EXPECT_LT(std::fabs(r->f32_slice()[0]), 1e-5f);
}

TEST(Elementwise, test_erf) {
    const float v = erf_approx(0.0f);
    EXPECT_LT(std::fabs(v), 1e-6f) << "erf(0) should be ~0, got " << v;
}

TEST(Elementwise, test_gelu_erf) {
    // Exact GELU: g(0)=0, g(1)=0.8413447, g(-1)=-0.1586553.
    auto out = gelu_erf(t({0.0f, 1.0f, -1.0f}));
    ASSERT_TRUE(out.has_value());
    const auto v = vec(*out);
    EXPECT_LT(std::fabs(v[0]), 1e-6f);
    EXPECT_LT(std::fabs(v[1] - 0.8413447f), 1e-4f) << "g(1)=" << v[1];
    EXPECT_LT(std::fabs(v[2] - (-0.1586553f)), 1e-4f) << "g(-1)=" << v[2];
}

TEST(Elementwise, test_scalar_broadcast) {
    auto r = mul(t({1.0f, 2.0f, 3.0f}), t({2.0f}));
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(vec(*r), (std::vector<float>{2.0f, 4.0f, 6.0f}));
}

// C++-only: the two Result-path messages are parity-bound.
TEST(Elementwise, error_messages_match_rust) {
    auto bytes = std::vector<uint8_t>(4, 0);
    auto h = Tensor::from_f16_bytes(bytes, Shape{2});
    ASSERT_TRUE(h.has_value());
    auto e = neg(*h);
    ASSERT_FALSE(e.has_value());
    EXPECT_EQ(e.error().to_string(), "Type mismatch: expected f32, got f16");
    auto s = add(t({1.0f, 2.0f}), t({1.0f, 2.0f, 3.0f}));
    ASSERT_FALSE(s.has_value());
    EXPECT_EQ(s.error().to_string(), "Shape mismatch: expected [2], got [3]");
}
```

- [ ] **Step 2: Add to CMake, build to verify failure**

Add `src/kernels/elementwise.cpp` and `tests/elementwise_test.cpp` to `CMakeLists.txt`.
Run: `cd cpp && cmake --preset dev && cmake --build --preset dev`
Expected: FAILS — `'sapient/backends_cpu/kernels/elementwise.hpp' file not found`.

- [ ] **Step 3: `elementwise.hpp`**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-backends/cpu/src/kernels/elementwise.rs. All kernels are scalar f32.
// Binary ops zip equal-length operands or broadcast a numel-1 side; unary ops require F32 input.

#include <optional>

#include "sapient/core/error.hpp"
#include "sapient/core/tensor.hpp"

namespace sapient::backends_cpu::kernels::elementwise {

using sapient::core::Result;
using sapient::core::Tensor;

Result<Tensor> add(const Tensor& a, const Tensor& b);
Result<Tensor> sub(const Tensor& a, const Tensor& b);
Result<Tensor> mul(const Tensor& a, const Tensor& b);
Result<Tensor> div(const Tensor& a, const Tensor& b);
Result<Tensor> pow(const Tensor& a, const Tensor& b);

Result<Tensor> neg(const Tensor& x);
Result<Tensor> abs(const Tensor& x);
Result<Tensor> sqrt(const Tensor& x);
Result<Tensor> exp(const Tensor& x);
Result<Tensor> log(const Tensor& x);
Result<Tensor> erf(const Tensor& x);
Result<Tensor> floor(const Tensor& x);
Result<Tensor> ceil(const Tensor& x);
Result<Tensor> round(const Tensor& x);

Result<Tensor> relu(const Tensor& x);
Result<Tensor> sigmoid(const Tensor& x);
Result<Tensor> tanh_act(const Tensor& x);
/// GELU tanh approximation: 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))).
Result<Tensor> gelu(const Tensor& x);
/// Exact (erf-based) GELU 0.5·x·(1 + erf(x/√2)) — the Whisper variant; uses erf_approx, not erff.
Result<Tensor> gelu_erf(const Tensor& x);
Result<Tensor> silu(const Tensor& x);
Result<Tensor> hard_swish(const Tensor& x);
Result<Tensor> leaky_relu(const Tensor& x, float alpha);
Result<Tensor> clip(const Tensor& x, std::optional<float> min, std::optional<float> max);

/// Abramowitz & Stegun 7.1.26 rational approximation (max error ~1.5e-7), verbatim from Rust —
/// NEVER replaced by `::erff` (bit-identity). Rust-private; exposed for the ported `test_erf`.
float erf_approx(float x);

} // namespace sapient::backends_cpu::kernels::elementwise
```

- [ ] **Step 4: `elementwise.cpp`**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/backends_cpu/kernels/elementwise.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "sapient/core/dtype.hpp"

// Inside this namespace the module's own `exp`, `log`, `abs`, `sqrt`, `floor`, `ceil`, `round`,
// `pow` (const Tensor&) HIDE the unqualified libm names — every libm call below is spelled with
// its global float name (`::expf`, …). That is also the bit-identity rule (float, never double).
namespace sapient::backends_cpu::kernels::elementwise {

using sapient::core::DType;
using sapient::core::Error;
using sapient::core::Shape;

namespace {

// Rust `unary_f32`: F32 only (TypeMismatch otherwise); reads the UNBOUNDED f32 view, so a sliced
// F32 view yields data.len() != numel and from_f32 returns ShapeMismatch — as in Rust.
template <class F> Result<Tensor> unary_f32(const Tensor& x, F f) {
    if (x.dtype() != DType::F32)
        return tl::unexpected(Error::type_mismatch("f32", sapient::core::to_string(x.dtype())));
    const auto cow = x.to_f32_cow();
    const auto src = cow.get();
    std::vector<float> data(src.size());
    for (size_t i = 0; i < src.size(); ++i) data[i] = f(src[i]);
    return Tensor::from_f32_vec(std::move(data), x.shape());
}

// Rust `binary_f32`: same length → zip; else numel-1 side broadcasts; else ShapeMismatch.
template <class F> Result<Tensor> binary_f32(const Tensor& a, const Tensor& b, F f) {
    const auto a_cow = a.to_f32_cow();
    const auto a_data = a_cow.get();
    const auto b_cow = b.to_f32_cow();
    const auto b_data = b_cow.get();
    std::vector<float> out;
    Shape shape;
    if (a_data.size() == b_data.size()) {
        out.resize(a_data.size());
        for (size_t i = 0; i < out.size(); ++i) out[i] = f(a_data[i], b_data[i]);
        shape = a.shape();
    } else if (b_data.size() == 1) {
        const float scalar = b_data[0];
        out.resize(a_data.size());
        for (size_t i = 0; i < out.size(); ++i) out[i] = f(a_data[i], scalar);
        shape = a.shape();
    } else if (a_data.size() == 1) {
        const float scalar = a_data[0];
        out.resize(b_data.size());
        for (size_t i = 0; i < out.size(); ++i) out[i] = f(scalar, b_data[i]);
        shape = b.shape();
    } else {
        return tl::unexpected(Error::shape_mismatch(a.shape().dims, b.shape().dims));
    }
    return Tensor::from_f32_vec(std::move(out), std::move(shape));
}

} // namespace

// ── Arithmetic ────────────────────────────────────────────────────────────────
Result<Tensor> add(const Tensor& a, const Tensor& b) {
    return binary_f32(a, b, [](float x, float y) { return x + y; });
}
Result<Tensor> sub(const Tensor& a, const Tensor& b) {
    return binary_f32(a, b, [](float x, float y) { return x - y; });
}
Result<Tensor> mul(const Tensor& a, const Tensor& b) {
    return binary_f32(a, b, [](float x, float y) { return x * y; });
}
Result<Tensor> div(const Tensor& a, const Tensor& b) {
    return binary_f32(a, b, [](float x, float y) { return x / y; });
}
Result<Tensor> pow(const Tensor& a, const Tensor& b) {
    return binary_f32(a, b, [](float x, float y) { return ::powf(x, y); });
}

Result<Tensor> neg(const Tensor& x) {
    return unary_f32(x, [](float v) { return -v; });
}
Result<Tensor> abs(const Tensor& x) {
    return unary_f32(x, [](float v) { return ::fabsf(v); });
}
Result<Tensor> sqrt(const Tensor& x) {
    return unary_f32(x, [](float v) { return ::sqrtf(v); });
}
Result<Tensor> exp(const Tensor& x) {
    return unary_f32(x, [](float v) { return ::expf(v); });
}
Result<Tensor> log(const Tensor& x) {
    return unary_f32(x, [](float v) { return ::logf(v); }); // Rust `ln`
}
Result<Tensor> erf(const Tensor& x) {
    return unary_f32(x, erf_approx);
}
Result<Tensor> floor(const Tensor& x) {
    return unary_f32(x, [](float v) { return ::floorf(v); });
}
Result<Tensor> ceil(const Tensor& x) {
    return unary_f32(x, [](float v) { return ::ceilf(v); });
}
Result<Tensor> round(const Tensor& x) {
    return unary_f32(x, [](float v) { return ::roundf(v); }); // half away from zero, like f32::round
}

// ── Activations ───────────────────────────────────────────────────────────────
Result<Tensor> relu(const Tensor& x) {
    return unary_f32(x, [](float v) { return ::fmaxf(v, 0.0f); }); // f32::max
}
Result<Tensor> sigmoid(const Tensor& x) {
    return unary_f32(x, [](float v) { return 1.0f / (1.0f + ::expf(-v)); });
}
Result<Tensor> tanh_act(const Tensor& x) {
    return unary_f32(x, [](float v) { return ::tanhf(v); });
}
Result<Tensor> gelu(const Tensor& x) {
    constexpr float SQRT_2_OVER_PI = 0.79788456f; // Rust 0.797_884_56
    constexpr float COEF = 0.044715f;             // Rust 0.044_715
    return unary_f32(x, [](float v) {
        const float inner = SQRT_2_OVER_PI * (v + COEF * v * v * v);
        return 0.5f * v * (1.0f + ::tanhf(inner));
    });
}
Result<Tensor> gelu_erf(const Tensor& x) {
    constexpr float INV_SQRT_2 = 0.70710678118654752440f; // std::f32::consts::FRAC_1_SQRT_2 (0x3F3504F3)
    return unary_f32(x, [](float v) { return 0.5f * v * (1.0f + erf_approx(v * INV_SQRT_2)); });
}
Result<Tensor> silu(const Tensor& x) {
    return unary_f32(x, [](float v) { return v / (1.0f + ::expf(-v)); });
}
Result<Tensor> hard_swish(const Tensor& x) {
    // Rust: v * (v + 3.0).clamp(0.0, 6.0) / 6.0 — std::clamp has the same NaN pass-through.
    return unary_f32(x, [](float v) { return v * std::clamp(v + 3.0f, 0.0f, 6.0f) / 6.0f; });
}
Result<Tensor> leaky_relu(const Tensor& x, float alpha) {
    return unary_f32(x, [alpha](float v) { return v >= 0.0f ? v : alpha * v; });
}
Result<Tensor> clip(const Tensor& x, std::optional<float> min, std::optional<float> max) {
    return unary_f32(x, [min, max](float v) {
        const float lo = min.has_value() ? ::fmaxf(v, *min) : v; // f32::max
        return max.has_value() ? ::fminf(lo, *max) : lo;         // f32::min
    });
}

// ── Erf approximation (Abramowitz & Stegun), coefficients verbatim from Rust ─────────────
float erf_approx(float x) {
    // f32::signum: NaN stays NaN; otherwise ±1 by sign bit (+0.0 → 1, -0.0 → -1).
    const float sign = std::isnan(x) ? x : ::copysignf(1.0f, x);
    x = ::fabsf(x);
    const float t = 1.0f / (1.0f + 0.3275911f * x);
    const float y = 1.0f - (0.25482959f + (-0.28449674f + (1.42141374f + (-1.45315203f + 1.06140543f * t) * t) * t) * t) *
                               t * ::expf(-x * x);
    return sign * y;
}

} // namespace sapient::backends_cpu::kernels::elementwise
```

- [ ] **Step 5: Build and run**

Run: `cd cpp && cmake --build --preset dev && ctest --preset dev -R Elementwise`
Expected: 8 tests pass (7 Rust names + `error_messages_match_rust`).

- [ ] **Step 6: Format, then commit**

```bash
git add cpp/libs/sapient-backends-cpu
git ls-files -- 'cpp/*.hpp' 'cpp/*.cpp' | xargs .superpowers/tools-venv/bin/clang-format -i
git add cpp/libs/sapient-backends-cpu
git commit -m "cpp(backends-cpu): kernels/elementwise — arithmetic, activations, A&S erf (7 Rust tests by name)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: `kernels/{softmax,reduce,layernorm,rope}.hpp/.cpp` — the four scalar modules

**Files:**
- Modify: `cpp/libs/sapient-backends-cpu/CMakeLists.txt` (add the four `src/kernels/*.cpp` and four `tests/*_test.cpp`)
- Create: `include/sapient/backends_cpu/kernels/{softmax,reduce,layernorm,rope}.hpp`, `src/kernels/{softmax,reduce,layernorm,rope}.cpp`
- Test: `tests/{softmax,reduce,layernorm,rope}_test.cpp`

**Interfaces:**
- Consumes: `sapient::core::{Tensor, Shape, Result, Error, panic}`.
- Produces:
  - `sapient::backends_cpu::kernels::softmax::{softmax, log_softmax}(const Tensor& x, int64_t axis) -> Result<Tensor>`
  - `sapient::backends_cpu::kernels::reduce::{reduce_sum, reduce_mean, reduce_max, reduce_min}(const Tensor& x, std::span<const int64_t> axes, bool keep_dims) -> Result<Tensor>`
  - `sapient::backends_cpu::kernels::layernorm::layer_norm(const Tensor& x, const Tensor* weight, const Tensor* bias, int64_t axis, float epsilon) -> Result<Tensor>`; `rms_norm(const Tensor& x, const Tensor* weight, float epsilon) -> Result<Tensor>`
  - `sapient::backends_cpu::kernels::rope::apply_rope(const Tensor& x, std::span<const size_t> positions, float base)`; `apply_rope_partial(x, positions, base, size_t rotary_dim)`; `apply_rope_partial_scaled(x, positions, base, rotary_dim, float pos_scale)` — all `-> Result<Tensor>`; `rope_cos_sin_cache(size_t max_seq_len, size_t head_dim, float base) -> std::pair<std::vector<float>, std::vector<float>>` (cos, sin).
  Task 8's dumps gate all four modules (`softmax`, `softmax_axis0`, `log_softmax`, `reduce`, `rms_norm`, `layer_norm`, `apply_rope`, `apply_rope_partial`, `apply_rope_partial_scaled`).

**Rust behaviour to mirror (porting map §8):** all four are single-threaded, sequential f32. Sums written as `.iter().sum()` seed at `-0.0f` (rule 2): `layer_norm` mean and variance, `rms_norm` `rms_sq`, `softmax` `sum_e`. `reduce_sum`'s fold seeds at `0.0f`. Axis normalisation is `(ndim as i64 + axis) as usize` (wrapping). Explicit panics replacing Rust index panics: `layer_norm` with `ax > ndim` (Rust slices `dims[..ax]`); `layer_norm`/`rms_norm` weight or bias shorter than the normalised size; `reduce_mean` with an out-of-range axis (`dims()[a]`); `reduce` with a zero dim (`% 0`) or an output index past the buffer; `apply_rope*`/`softmax` data shorter than the shape. Result-path texts: rope's four `internal` messages and softmax's `softmax axis {axis} out of range for rank {ndim}` (both integers in decimal), `RankMismatch{expected: 4, got: ndim}` from rope.

- [ ] **Step 1: Write the failing tests**

`tests/softmax_test.cpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "sapient/backends_cpu/kernels/softmax.hpp"
#include "sapient/core/tensor.hpp"

using namespace sapient::backends_cpu::kernels::softmax;
using sapient::core::Shape;
using sapient::core::Tensor;

namespace {
Tensor f32(std::vector<float> data, Shape shape) {
    auto r = Tensor::from_f32_vec(std::move(data), std::move(shape));
    if (!r) throw std::runtime_error(r.error().to_string());
    return std::move(*r);
}
} // namespace

TEST(Softmax, softmax_sums_to_one) {
    const auto x = f32({1.0f, 2.0f, 3.0f, 4.0f}, Shape{1, 4});
    auto y = softmax(x, 1);
    ASSERT_TRUE(y.has_value());
    float sum = 0.0f;
    for (float v : y->f32_slice()) sum += v;
    EXPECT_LT(std::fabs(sum - 1.0f), 1e-6f) << "sum = " << sum;
}

TEST(Softmax, softmax_stable_large) {
    const auto x = f32({1000.0f, 1001.0f, 1002.0f}, Shape{1, 3});
    auto y = softmax(x, 1);
    ASSERT_TRUE(y.has_value());
    float sum = 0.0f;
    for (float v : y->f32_slice()) {
        EXPECT_TRUE(std::isfinite(v)) << "non-finite: " << v;
        sum += v;
    }
    EXPECT_LT(std::fabs(sum - 1.0f), 1e-5f) << "sum = " << sum;
}

TEST(Softmax, log_softmax_finite) {
    const auto x = f32({1.0f, 2.0f, 3.0f}, Shape{1, 3});
    auto y = log_softmax(x, 1);
    ASSERT_TRUE(y.has_value());
    for (float v : y->f32_slice()) EXPECT_TRUE(std::isfinite(v));
}

// C++-only: the Result-path message is parity-bound; a very negative axis wraps like Rust's `as usize`.
TEST(Softmax, axis_error_message_matches_rust) {
    const auto x = f32({1.0f, 2.0f, 3.0f}, Shape{1, 3});
    auto e = softmax(x, 2);
    ASSERT_FALSE(e.has_value());
    EXPECT_EQ(e.error().to_string(), "Internal error: softmax axis 2 out of range for rank 2");
    auto w = softmax(x, -7);
    ASSERT_FALSE(w.has_value());
    EXPECT_EQ(w.error().to_string(), "Internal error: softmax axis -7 out of range for rank 2");
}
```

`tests/reduce_test.cpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "sapient/backends_cpu/kernels/reduce.hpp"
#include "sapient/core/tensor.hpp"

using namespace sapient::backends_cpu::kernels::reduce;
using sapient::core::Shape;
using sapient::core::Tensor;

namespace {
Tensor f32(std::vector<float> data, Shape shape) {
    auto r = Tensor::from_f32_vec(std::move(data), std::move(shape));
    if (!r) throw std::runtime_error(r.error().to_string());
    return std::move(*r);
}
} // namespace

TEST(Reduce, sum_all) {
    const auto x = f32({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2});
    auto y = reduce_sum(x, {}, false);
    ASSERT_TRUE(y.has_value());
    EXPECT_TRUE(y->shape().dims.empty()) << "all-axes reduction is a scalar";
    EXPECT_LT(std::fabs(y->f32_slice()[0] - 10.0f), 1e-5f);
}

TEST(Reduce, mean_axis0) {
    const auto x = f32({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2});
    const int64_t axes[] = {0};
    auto y = reduce_mean(x, axes, false);
    ASSERT_TRUE(y.has_value());
    const auto d = y->f32_slice();
    EXPECT_LT(std::fabs(d[0] - 2.0f), 1e-5f) << "d[0]=" << d[0];
    EXPECT_LT(std::fabs(d[1] - 3.0f), 1e-5f) << "d[1]=" << d[1];
}
```

`tests/layernorm_test.cpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "sapient/backends_cpu/kernels/layernorm.hpp"
#include "sapient/core/tensor.hpp"

using namespace sapient::backends_cpu::kernels::layernorm;
using sapient::core::Shape;
using sapient::core::Tensor;

namespace {
Tensor f32(std::vector<float> data, Shape shape) {
    auto r = Tensor::from_f32_vec(std::move(data), std::move(shape));
    if (!r) throw std::runtime_error(r.error().to_string());
    return std::move(*r);
}
} // namespace

TEST(LayerNorm, layernorm_zero_mean_unit_var) {
    const auto x = f32({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2});
    auto y = layer_norm(x, nullptr, nullptr, -1, 1e-5f);
    ASSERT_TRUE(y.has_value());
    const auto d = y->f32_slice();
    // Each pair: mean=1.5, var=0.25, std=0.5. (1-1.5)/0.5 = -1, (2-1.5)/0.5 = 1.
    EXPECT_LT(std::fabs(d[0] + 1.0f), 1e-4f) << "d[0]=" << d[0];
    EXPECT_LT(std::fabs(d[1] - 1.0f), 1e-4f) << "d[1]=" << d[1];
    EXPECT_LT(std::fabs(d[2] + 1.0f), 1e-4f) << "d[2]=" << d[2];
    EXPECT_LT(std::fabs(d[3] - 1.0f), 1e-4f) << "d[3]=" << d[3];
}

TEST(LayerNorm, rmsnorm_identity_weight) {
    // rms = sqrt((9 + 16) / 2) = sqrt(12.5); output = [3, 4] / sqrt(12.5).
    const auto x = f32({3.0f, 4.0f}, Shape{1, 2});
    const auto w = f32({1.0f, 1.0f}, Shape{2});
    auto y = rms_norm(x, &w, 0.0f);
    ASSERT_TRUE(y.has_value());
    const auto d = y->f32_slice();
    const float expected0 = 3.0f / std::sqrt(12.5f);
    const float expected1 = 4.0f / std::sqrt(12.5f);
    EXPECT_LT(std::fabs(d[0] - expected0), 1e-5f) << "d[0]=" << d[0] << " expected " << expected0;
    EXPECT_LT(std::fabs(d[1] - expected1), 1e-5f) << "d[1]=" << d[1] << " expected " << expected1;
}
```

`tests/rope_test.cpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "sapient/backends_cpu/kernels/rope.hpp"
#include "sapient/core/tensor.hpp"

using namespace sapient::backends_cpu::kernels::rope;
using sapient::core::Shape;
using sapient::core::Tensor;

namespace {
Tensor f32(std::vector<float> data, Shape shape) {
    auto r = Tensor::from_f32_vec(std::move(data), std::move(shape));
    if (!r) throw std::runtime_error(r.error().to_string());
    return std::move(*r);
}
} // namespace

TEST(Rope, rope_output_shape) {
    const auto x = f32(std::vector<float>(64, 0.1f), Shape{1, 2, 4, 8});
    const size_t positions[] = {0, 1, 2, 3};
    auto out = apply_rope(x, positions, 10000.0f);
    ASSERT_TRUE(out.has_value());
    EXPECT_EQ(out->shape().dims, (std::vector<size_t>{1, 2, 4, 8}));
}

TEST(Rope, rope_partial_leaves_tail_unchanged) {
    // head_dim=8, rotary_dim=4 → channels [4..8) pass through unchanged; at a non-zero position
    // the rotary channels [0..4) must change.
    std::vector<float> data;
    for (int v = 1; v <= 8; ++v) data.push_back(static_cast<float>(v));
    const auto x = f32(data, Shape{1, 1, 1, 8});
    const size_t positions[] = {3};
    auto out = apply_rope_partial(x, positions, 10000.0f, 4);
    ASSERT_TRUE(out.has_value());
    const auto o = out->f32_slice();
    for (size_t i = 4; i < 8; ++i) EXPECT_LT(std::fabs(o[i] - data[i]), 1e-6f) << "tail channel " << i << " changed";
    bool any_changed = false;
    for (size_t i = 0; i < 4; ++i) any_changed = any_changed || std::fabs(o[i] - data[i]) > 1e-6f;
    EXPECT_TRUE(any_changed);
}

TEST(Rope, rope_partial_full_matches_apply_rope) {
    std::vector<float> data;
    for (int v = 0; v < 16; ++v) data.push_back(static_cast<float>(v) * 0.1f);
    const auto x = f32(data, Shape{1, 1, 2, 8});
    const size_t positions[] = {2, 5};
    auto full = apply_rope(x, positions, 10000.0f);
    auto part = apply_rope_partial(x, positions, 10000.0f, 8);
    ASSERT_TRUE(full.has_value());
    ASSERT_TRUE(part.has_value());
    const auto a = full->f32_slice();
    const auto b = part->f32_slice();
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) EXPECT_LT(std::fabs(a[i] - b[i]), 1e-6f) << i;
}

TEST(Rope, rope_position_zero_is_identity) {
    const std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    const auto x = f32(data, Shape{1, 1, 1, 4});
    const size_t positions[] = {0};
    auto out = apply_rope(x, positions, 10000.0f);
    ASSERT_TRUE(out.has_value());
    const auto o = out->f32_slice();
    for (size_t i = 0; i < data.size(); ++i) EXPECT_LT(std::fabs(data[i] - o[i]), 1e-6f) << "position 0 should be identity";
}

// C++-only: the four Result-path messages and the RankMismatch fields are parity-bound.
TEST(Rope, error_messages_match_rust) {
    const auto x3 = f32(std::vector<float>(8, 0.0f), Shape{1, 2, 4});
    const auto x4 = f32(std::vector<float>(8, 0.0f), Shape{1, 1, 1, 8});
    const auto x_odd = f32(std::vector<float>(7, 0.0f), Shape{1, 1, 1, 7});
    const size_t one[] = {0};
    const size_t two[] = {0, 1};
    EXPECT_EQ(apply_rope(x3, one, 1.0f).error().to_string(), "Rank mismatch: expected 4, got 3");
    EXPECT_EQ(apply_rope(x_odd, one, 1.0f).error().to_string(), "Internal error: RoPE requires even head_dim");
    EXPECT_EQ(apply_rope(x4, two, 1.0f).error().to_string(), "Internal error: positions length must match seq_len");
    EXPECT_EQ(apply_rope_partial(x4, one, 1.0f, 0).error().to_string(), "Internal error: rotary_dim must be in 1..=head_dim");
    EXPECT_EQ(apply_rope_partial(x4, one, 1.0f, 3).error().to_string(), "Internal error: RoPE requires even rotary_dim");
    EXPECT_EQ(apply_rope_partial(x4, two, 1.0f, 8).error().to_string(), "Internal error: positions length must match seq_len");
}
```

- [ ] **Step 2: Add to CMake, build to verify failure**

Add `src/kernels/{softmax,reduce,layernorm,rope}.cpp` and `tests/{softmax,reduce,layernorm,rope}_test.cpp` to `CMakeLists.txt`.
Run: `cd cpp && cmake --preset dev && cmake --build --preset dev`
Expected: FAILS — the four headers are not found.

- [ ] **Step 3: `softmax.hpp` / `softmax.cpp`**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-backends/cpu/src/kernels/softmax.rs — max-subtracted softmax /
// log-softmax along one axis, single-threaded.

#include <cstdint>

#include "sapient/core/error.hpp"
#include "sapient/core/tensor.hpp"

namespace sapient::backends_cpu::kernels::softmax {

using sapient::core::Result;
using sapient::core::Tensor;

/// Numerically stable softmax along `axis` (negative counts from the end).
Result<Tensor> softmax(const Tensor& x, int64_t axis);
/// Numerically stable log-softmax along `axis`.
Result<Tensor> log_softmax(const Tensor& x, int64_t axis);

} // namespace sapient::backends_cpu::kernels::softmax
```

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/backends_cpu/kernels/softmax.hpp"

#include <cfloat>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "sapient/core/panic.hpp"

namespace sapient::backends_cpu::kernels::softmax {

using sapient::core::Error;
using sapient::core::Shape;

namespace {

// Rust `(ndim as i64 + axis) as usize`: a still-negative sum wraps to a huge index, which the
// range check in the caller then rejects. The unsigned conversion here wraps identically.
size_t normalise_axis(int64_t axis, size_t ndim) {
    return axis < 0 ? static_cast<size_t>(static_cast<int64_t>(ndim) + axis)
                    : static_cast<size_t>(axis);
}

Result<Tensor> apply_softmax_impl(const Tensor& x, int64_t axis, bool log_mode) {
    const Shape& shape = x.shape();
    const size_t ndim = shape.ndim();
    const size_t ax = normalise_axis(axis, ndim);
    if (ax >= ndim)
        return tl::unexpected(Error::internal("softmax axis " + std::to_string(axis) +
                                              " out of range for rank " + std::to_string(ndim)));

    const auto cow = x.to_f32_cow();
    const auto data = cow.get();
    if (data.size() < shape.numel()) sapient::core::panic("softmax: data shorter than shape");
    std::vector<float> out(data.size(), 0.0f); // Rust: vec![0.0; data.len()] (unbounded view, rule 7)

    size_t outer = 1;
    for (size_t i = 0; i < ax; ++i) outer *= shape.dims[i];
    const size_t dim_size = shape.dims[ax];
    size_t inner = 1;
    for (size_t i = ax + 1; i < ndim; ++i) inner *= shape.dims[i];

    std::vector<float> slice(dim_size);
    std::vector<float> exps(dim_size);
    for (size_t o = 0; o < outer; ++o) {
        for (size_t i = 0; i < inner; ++i) {
            for (size_t d = 0; d < dim_size; ++d) slice[d] = data[(o * dim_size + d) * inner + i];
            // fold(NEG_INFINITY, f32::max) — fmaxf drops a NaN operand exactly like f32::max.
            float max_v = -std::numeric_limits<float>::infinity();
            for (size_t d = 0; d < dim_size; ++d) max_v = ::fmaxf(max_v, slice[d]);
            if (max_v == -std::numeric_limits<float>::infinity()) max_v = 0.0f;
            for (size_t d = 0; d < dim_size; ++d) exps[d] = ::expf(slice[d] - max_v);
            float sum_e = -0.0f; // iter().sum::<f32>() seeds at -0.0
            for (size_t d = 0; d < dim_size; ++d) sum_e += exps[d];
            if (sum_e == 0.0f) sum_e = FLT_EPSILON;
            for (size_t d = 0; d < dim_size; ++d) {
                const size_t idx = (o * dim_size + d) * inner + i;
                out[idx] = log_mode ? (slice[d] - max_v) - ::logf(sum_e) : exps[d] / sum_e;
            }
        }
    }
    return Tensor::from_f32_vec(std::move(out), shape);
}

} // namespace

Result<Tensor> softmax(const Tensor& x, int64_t axis) { return apply_softmax_impl(x, axis, false); }
Result<Tensor> log_softmax(const Tensor& x, int64_t axis) { return apply_softmax_impl(x, axis, true); }

} // namespace sapient::backends_cpu::kernels::softmax
```

- [ ] **Step 4: `reduce.hpp` / `reduce.cpp`**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-backends/cpu/src/kernels/reduce.rs — sum/mean/max/min over one or more
// axes (empty `axes` = all), element-wise scatter-accumulate, single-threaded.

#include <cstdint>
#include <span>

#include "sapient/core/error.hpp"
#include "sapient/core/tensor.hpp"

namespace sapient::backends_cpu::kernels::reduce {

using sapient::core::Result;
using sapient::core::Tensor;

Result<Tensor> reduce_sum(const Tensor& x, std::span<const int64_t> axes, bool keep_dims);
Result<Tensor> reduce_mean(const Tensor& x, std::span<const int64_t> axes, bool keep_dims);
Result<Tensor> reduce_max(const Tensor& x, std::span<const int64_t> axes, bool keep_dims);
Result<Tensor> reduce_min(const Tensor& x, std::span<const int64_t> axes, bool keep_dims);

} // namespace sapient::backends_cpu::kernels::reduce
```

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/backends_cpu/kernels/reduce.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "sapient/core/panic.hpp"

namespace sapient::backends_cpu::kernels::reduce {

using sapient::core::Error;
using sapient::core::Shape;

namespace {

std::vector<size_t> normalise_axes(std::span<const int64_t> axes, size_t ndim) {
    std::vector<size_t> out;
    if (axes.empty()) {
        for (size_t i = 0; i < ndim; ++i) out.push_back(i);
        return out;
    }
    for (const int64_t a : axes)
        out.push_back(a < 0 ? static_cast<size_t>(static_cast<int64_t>(ndim) + a)
                            : static_cast<size_t>(a)); // Rust: `(ndim as i64 + a) as usize`
    return out;
}

/// Rust `fn reduce<F>(x, axes, keep_dims, init, f)`.
template <class F>
Result<Tensor>
reduce_impl(const Tensor& x, std::span<const int64_t> axes, bool keep_dims, float init, F f) {
    const Shape& shape = x.shape();
    const auto cow = x.to_f32_cow();
    const auto data = cow.get(); // unbounded for F32 (rule 7): Rust iterates data.len(), so do we
    const std::vector<size_t> norm_axes = normalise_axes(axes, shape.ndim());
    const auto reduced = [&](size_t i) {
        return std::find(norm_axes.begin(), norm_axes.end(), i) != norm_axes.end();
    };

    std::vector<size_t> out_dims;
    for (size_t i = 0; i < shape.ndim(); ++i) {
        if (reduced(i)) {
            if (keep_dims) out_dims.push_back(1);
        } else {
            out_dims.push_back(shape.dims[i]);
        }
    }
    size_t out_numel = 1;
    for (const size_t d : out_dims) out_numel *= d;
    out_numel = std::max<size_t>(out_numel, 1);
    std::vector<float> out_data(out_numel, init);

    // Rust rebuilds `Shape(out_dims.clone()).strides()` inside the per-element loop; it is a pure
    // function of out_dims, so it is hoisted here (behaviour-identical).
    const std::vector<size_t> out_strides = Shape(out_dims).strides();
    std::vector<size_t> multi(shape.ndim(), 0);
    for (size_t flat = 0; flat < data.size(); ++flat) {
        size_t r = flat;
        for (size_t i = shape.ndim(); i-- > 0;) {
            if (shape.dims[i] == 0) sapient::core::panic("reduce: zero dimension"); // Rust: `% 0` panic
            multi[i] = r % shape.dims[i];
            r /= shape.dims[i];
        }
        size_t out_flat = 0;
        size_t oi = 0;
        for (size_t i = 0; i < multi.size(); ++i) {
            if (!reduced(i)) {
                out_flat += multi[i] * (oi < out_strides.size() ? out_strides[oi] : 1);
                ++oi;
            } else if (keep_dims) {
                ++oi; // dim = 1, stride may still be 1.
            }
        }
        if (out_flat >= out_data.size()) sapient::core::panic("reduce: output index out of range");
        out_data[out_flat] = f(out_data[out_flat], data[flat]);
    }
    return Tensor::from_f32_vec(std::move(out_data), Shape(out_dims));
}

} // namespace

Result<Tensor> reduce_sum(const Tensor& x, std::span<const int64_t> axes, bool keep_dims) {
    return reduce_impl(x, axes, keep_dims, 0.0f, [](float acc, float v) { return acc + v; });
}

Result<Tensor> reduce_mean(const Tensor& x, std::span<const int64_t> axes, bool keep_dims) {
    auto sum = reduce_sum(x, axes, keep_dims);
    if (!sum.has_value()) return tl::unexpected(std::move(sum.error()));
    const std::vector<size_t> norm_axes = normalise_axes(axes, x.ndim());
    size_t count = 1;
    for (const size_t a : norm_axes) {
        if (a >= x.ndim()) sapient::core::panic("reduce_mean: axis out of range"); // Rust: dims()[a] panic
        count *= x.shape().dims[a];
    }
    const auto s = sum->f32_slice();
    std::vector<float> d(s.size());
    for (size_t i = 0; i < s.size(); ++i) d[i] = s[i] / static_cast<float>(count);
    return Tensor::from_f32_vec(std::move(d), sum->shape());
}

Result<Tensor> reduce_max(const Tensor& x, std::span<const int64_t> axes, bool keep_dims) {
    return reduce_impl(x, axes, keep_dims, -std::numeric_limits<float>::infinity(),
                       [](float acc, float v) { return ::fmaxf(acc, v); }); // f32::max
}

Result<Tensor> reduce_min(const Tensor& x, std::span<const int64_t> axes, bool keep_dims) {
    return reduce_impl(x, axes, keep_dims, std::numeric_limits<float>::infinity(),
                       [](float acc, float v) { return ::fminf(acc, v); }); // f32::min
}

} // namespace sapient::backends_cpu::kernels::reduce
```

- [ ] **Step 5: `layernorm.hpp` / `layernorm.cpp`**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-backends/cpu/src/kernels/layernorm.rs — LayerNorm over the axes from
// `axis` to the end, RMSNorm over the last axis. Sequential f32 sums, single-threaded.

#include <cstdint>

#include "sapient/core/error.hpp"
#include "sapient/core/tensor.hpp"

namespace sapient::backends_cpu::kernels::layernorm {

using sapient::core::Result;
using sapient::core::Tensor;

/// y = (x - mean) / sqrt(var + eps) * weight + bias; `axis` is the first normalised axis
/// (typically -1). `weight`/`bias` may be null.
Result<Tensor> layer_norm(
    const Tensor& x, const Tensor* weight, const Tensor* bias, int64_t axis, float epsilon);

/// y = x / sqrt(mean(x²) + eps) * weight over the last axis. `weight` may be null (= 1).
Result<Tensor> rms_norm(const Tensor& x, const Tensor* weight, float epsilon);

} // namespace sapient::backends_cpu::kernels::layernorm
```

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/backends_cpu/kernels/layernorm.hpp"

#include <cmath>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "sapient/core/panic.hpp"

namespace sapient::backends_cpu::kernels::layernorm {

using sapient::core::F32Cow;
using sapient::core::Shape;

Result<Tensor> layer_norm(
    const Tensor& x, const Tensor* weight, const Tensor* bias, int64_t axis, float epsilon) {
    const Shape& shape = x.shape();
    const size_t ndim = shape.ndim();
    const size_t ax = axis < 0 ? static_cast<size_t>(static_cast<int64_t>(ndim) + axis)
                               : static_cast<size_t>(axis);
    if (ax > ndim) sapient::core::panic("layer_norm: axis out of range"); // Rust: dims()[..ax] panic

    size_t outer = 1;
    for (size_t i = 0; i < ax; ++i) outer *= shape.dims[i];
    size_t norm_size = 1;
    for (size_t i = ax; i < ndim; ++i) norm_size *= shape.dims[i];

    const auto cow = x.to_f32_cow();
    const auto data = cow.get();
    if (outer * norm_size > data.size()) sapient::core::panic("layer_norm: data shorter than shape");
    std::vector<float> out(data.size(), 0.0f);

    std::optional<F32Cow> w_cow;
    std::span<const float> w;
    if (weight != nullptr) {
        w_cow = weight->to_f32_cow();
        w = w_cow->get();
        if (w.size() < norm_size) sapient::core::panic("layer_norm: weight shorter than the normalised size");
    }
    std::optional<F32Cow> b_cow;
    std::span<const float> b;
    if (bias != nullptr) {
        b_cow = bias->to_f32_cow();
        b = b_cow->get();
        if (b.size() < norm_size) sapient::core::panic("layer_norm: bias shorter than the normalised size");
    }

    for (size_t o = 0; o < outer; ++o) {
        const size_t base = o * norm_size;
        const float* slice = data.data() + base;

        float sum = -0.0f; // iter().sum::<f32>() seeds at -0.0
        for (size_t i = 0; i < norm_size; ++i) sum += slice[i];
        const float mean = sum / static_cast<float>(norm_size);

        float var_sum = -0.0f;
        for (size_t i = 0; i < norm_size; ++i) var_sum += (slice[i] - mean) * (slice[i] - mean);
        const float var = var_sum / static_cast<float>(norm_size);

        const float inv_std = 1.0f / ::sqrtf(var + epsilon);

        for (size_t i = 0; i < norm_size; ++i) {
            const float normed = (slice[i] - mean) * inv_std;
            float y = normed;
            if (weight != nullptr && bias != nullptr)
                y = normed * w[i] + b[i];
            else if (weight != nullptr)
                y = normed * w[i];
            else if (bias != nullptr)
                y = normed + b[i];
            out[base + i] = y;
        }
    }
    return Tensor::from_f32_vec(std::move(out), shape);
}

Result<Tensor> rms_norm(const Tensor& x, const Tensor* weight, float epsilon) {
    const Shape& shape = x.shape();
    const size_t ndim = shape.ndim();

    size_t outer = 1;
    for (size_t i = 0; i + 1 < ndim; ++i) outer *= shape.dims[i]; // dims[..ndim.saturating_sub(1)]
    const size_t dim = ndim > 0 ? shape.dims[ndim - 1] : 1;

    const auto cow = x.to_f32_cow();
    const auto data = cow.get();
    if (outer * dim > data.size()) sapient::core::panic("rms_norm: data shorter than shape");
    std::vector<float> out(data.size(), 0.0f);

    std::optional<F32Cow> w_cow;
    std::span<const float> w;
    if (weight != nullptr) {
        w_cow = weight->to_f32_cow();
        w = w_cow->get();
        if (w.size() < dim) sapient::core::panic("rms_norm: weight shorter than the last dim");
    }

    for (size_t o = 0; o < outer; ++o) {
        const size_t base = o * dim;
        const float* slice = data.data() + base;

        float sq = -0.0f; // iter().map(v*v).sum::<f32>() seeds at -0.0
        for (size_t i = 0; i < dim; ++i) sq += slice[i] * slice[i];
        const float rms_sq = sq / static_cast<float>(dim);
        const float inv_rms = 1.0f / ::sqrtf(rms_sq + epsilon);

        for (size_t i = 0; i < dim; ++i) {
            const float wv = weight != nullptr ? w[i] : 1.0f; // w.map_or(1.0, |ww| ww[i])
            out[base + i] = slice[i] * inv_rms * wv;
        }
    }
    return Tensor::from_f32_vec(std::move(out), shape);
}

} // namespace sapient::backends_cpu::kernels::layernorm
```

- [ ] **Step 6: `rope.hpp` / `rope.cpp`**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-backends/cpu/src/kernels/rope.rs — NEOX rotate-half RoPE on
// [batch, n_heads, seq, head_dim]. Scalar; `powf`/`sinf`/`cosf` from the platform libm, exactly
// the calls rustc's LLVM intrinsics lower to (the biggest libm-portability risk in the crate).

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include "sapient/core/error.hpp"
#include "sapient/core/tensor.hpp"

namespace sapient::backends_cpu::kernels::rope {

using sapient::core::Result;
using sapient::core::Tensor;

/// θ_i = pos / base^(2i/head_dim); [x0, x1] → [x0·cos − x1·sin, x1·cos + x0·sin] over the two halves.
Result<Tensor> apply_rope(const Tensor& x, std::span<const size_t> positions, float base);
/// Rotates only the first `rotary_dim` channels of each head (Phi partial RoPE).
Result<Tensor>
apply_rope_partial(const Tensor& x, std::span<const size_t> positions, float base, size_t rotary_dim);
/// `apply_rope_partial` with linear position scaling (effective position = pos / pos_scale).
Result<Tensor> apply_rope_partial_scaled(
    const Tensor& x, std::span<const size_t> positions, float base, size_t rotary_dim, float pos_scale);
/// (cos, sin) tables of shape [max_seq_len, head_dim/2].
std::pair<std::vector<float>, std::vector<float>>
rope_cos_sin_cache(size_t max_seq_len, size_t head_dim, float base);

} // namespace sapient::backends_cpu::kernels::rope
```

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/backends_cpu/kernels/rope.hpp"

#include <cmath>

#include "sapient/core/panic.hpp"

namespace sapient::backends_cpu::kernels::rope {

using sapient::core::Error;
using sapient::core::Shape;

Result<Tensor> apply_rope(const Tensor& x, std::span<const size_t> positions, float base) {
    const auto& dims = x.shape().dims;
    if (dims.size() != 4) return tl::unexpected(Error::rank_mismatch(4, dims.size()));
    const size_t batch = dims[0], n_heads = dims[1], seq_len = dims[2], head_dim = dims[3];
    if (head_dim % 2 != 0) return tl::unexpected(Error::internal("RoPE requires even head_dim"));
    if (positions.size() != seq_len)
        return tl::unexpected(Error::internal("positions length must match seq_len"));

    const size_t half = head_dim / 2;
    const auto cow = x.to_f32_cow();
    const auto x_data = cow.get();
    std::vector<float> out(x_data.begin(), x_data.end()); // x_data.to_vec() (unbounded, rule 7)
    if (x_data.size() < batch * n_heads * seq_len * head_dim)
        sapient::core::panic("apply_rope: data shorter than shape");

    for (size_t b = 0; b < batch; ++b)
        for (size_t h = 0; h < n_heads; ++h)
            for (size_t s = 0; s < seq_len; ++s) {
                const size_t pos = positions[s];
                const size_t base_idx = ((b * n_heads + h) * seq_len + s) * head_dim;
                for (size_t i = 0; i < half; ++i) {
                    // θ_i = pos / base^(2i / head_dim)
                    const float freq = static_cast<float>(pos) /
                                       ::powf(base, 2.0f * static_cast<float>(i) / static_cast<float>(head_dim));
                    const float sin_f = ::sinf(freq);
                    const float cos_f = ::cosf(freq);
                    const float x0 = x_data[base_idx + i];
                    const float x1 = x_data[base_idx + i + half];
                    out[base_idx + i] = x0 * cos_f - x1 * sin_f;
                    out[base_idx + i + half] = x1 * cos_f + x0 * sin_f;
                }
            }
    return Tensor::from_f32_vec(std::move(out), Shape{batch, n_heads, seq_len, head_dim});
}

Result<Tensor>
apply_rope_partial(const Tensor& x, std::span<const size_t> positions, float base, size_t rotary_dim) {
    return apply_rope_partial_scaled(x, positions, base, rotary_dim, 1.0f);
}

Result<Tensor> apply_rope_partial_scaled(
    const Tensor& x, std::span<const size_t> positions, float base, size_t rotary_dim, float pos_scale) {
    const auto& dims = x.shape().dims;
    if (dims.size() != 4) return tl::unexpected(Error::rank_mismatch(4, dims.size()));
    const size_t batch = dims[0], n_heads = dims[1], seq_len = dims[2], head_dim = dims[3];
    if (rotary_dim == 0 || rotary_dim > head_dim)
        return tl::unexpected(Error::internal("rotary_dim must be in 1..=head_dim"));
    if (rotary_dim % 2 != 0) return tl::unexpected(Error::internal("RoPE requires even rotary_dim"));
    if (positions.size() != seq_len)
        return tl::unexpected(Error::internal("positions length must match seq_len"));

    // The rotary half-split is over rotary_dim; channels [rotary_dim, head_dim) pass through.
    const size_t half = rotary_dim / 2;
    const auto cow = x.to_f32_cow();
    const auto x_data = cow.get();
    std::vector<float> out(x_data.begin(), x_data.end());
    if (x_data.size() < batch * n_heads * seq_len * head_dim)
        sapient::core::panic("apply_rope_partial_scaled: data shorter than shape");

    for (size_t b = 0; b < batch; ++b)
        for (size_t h = 0; h < n_heads; ++h)
            for (size_t s = 0; s < seq_len; ++s) {
                const size_t pos = positions[s];
                const size_t base_idx = ((b * n_heads + h) * seq_len + s) * head_dim;
                for (size_t i = 0; i < half; ++i) {
                    const float freq = (static_cast<float>(pos) / pos_scale) /
                                       ::powf(base, 2.0f * static_cast<float>(i) / static_cast<float>(rotary_dim));
                    const float sin_f = ::sinf(freq);
                    const float cos_f = ::cosf(freq);
                    const float x0 = x_data[base_idx + i];
                    const float x1 = x_data[base_idx + i + half];
                    out[base_idx + i] = x0 * cos_f - x1 * sin_f;
                    out[base_idx + i + half] = x1 * cos_f + x0 * sin_f;
                }
            }
    return Tensor::from_f32_vec(std::move(out), Shape{batch, n_heads, seq_len, head_dim});
}

std::pair<std::vector<float>, std::vector<float>>
rope_cos_sin_cache(size_t max_seq_len, size_t head_dim, float base) {
    const size_t half = head_dim / 2;
    std::vector<float> cos_table(max_seq_len * half, 0.0f);
    std::vector<float> sin_table(max_seq_len * half, 0.0f);
    for (size_t pos = 0; pos < max_seq_len; ++pos)
        for (size_t i = 0; i < half; ++i) {
            const float freq = static_cast<float>(pos) /
                               ::powf(base, 2.0f * static_cast<float>(i) / static_cast<float>(head_dim));
            cos_table[pos * half + i] = ::cosf(freq);
            sin_table[pos * half + i] = ::sinf(freq);
        }
    return {std::move(cos_table), std::move(sin_table)};
}

} // namespace sapient::backends_cpu::kernels::rope
```

- [ ] **Step 7: Build and run**

Run: `cd cpp && cmake --build --preset dev && ctest --preset dev -R 'Softmax|Reduce|LayerNorm|Rope'`
Expected: 13 tests pass (11 Rust names + `Softmax.axis_error_message_matches_rust` + `Rope.error_messages_match_rust`).

- [ ] **Step 8: Format, then commit**

```bash
git add cpp/libs/sapient-backends-cpu
git ls-files -- 'cpp/*.hpp' 'cpp/*.cpp' | xargs .superpowers/tools-venv/bin/clang-format -i
git add cpp/libs/sapient-backends-cpu
git commit -m "cpp(backends-cpu): kernels/{softmax,reduce,layernorm,rope} (11 Rust tests by name)

-0.0 seeds every ported iter().sum() (Rust ≥1.83 float Sum neutral element).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: `kernels/attention.hpp/.cpp` — Flash-Edge online-softmax attention

**Files:**
- Modify: `cpp/libs/sapient-backends-cpu/CMakeLists.txt` (add `src/kernels/attention.cpp`, `tests/attention_test.cpp`)
- Create: `include/sapient/backends_cpu/kernels/attention.hpp`, `src/kernels/attention.cpp`
- Test: `tests/attention_test.cpp`

**Interfaces:**
- Consumes: `parallel::par_chunks_mut` (Task 1); `sapient::core::{Tensor, Shape, Result, Error, F32Cow, panic}`.
- Produces (namespace `sapient::backends_cpu::kernels::attention`): `Result<Tensor> scaled_dot_product_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor* mask, std::optional<float> scale, size_t n_kv_heads)` (`mask == nullptr` ⇒ built-in causal masking with the KV-cache offset; `scale == nullopt` ⇒ `1/sqrt(head_dim)`); `Tensor causal_mask(size_t seq_q, size_t seq_k)`. Task 8's dumps gate `attention_prefill`, `attention_decode`, `attention_masked` bit-identically.

**Rust behaviour to mirror (attention.rs, porting map §4/§8):** `dot_f32_neon` (4-wide `vfmaq_f32` into one accumulator, `vaddvq_f32`, scalar tail) and `saxpby_neon` (`vfmaq_f32(vmulq_f32(vo, va), vv, vb)`) on aarch64, scalar twins elsewhere (the scalar dot is an `.iter().zip().map().sum()` — seed `-0.0f`); `flash_attn_row` keeps a running max `m` and denominator `l`, **`continue`s on `s == -inf`** (the Gemma3 NaN guard), corrects with `expf(m - m_new)`, normalises by `1/l` or `1/FLT_EPSILON` when `l == 0`; `scaled_dot_product_attention` converts K/V with `to_contiguous_f32_vec()`, reads Q through `to_f32_cow()` + `strides()` (zero-copy row iff `strides[3] == 1`, else gathered), `kv_offset = seq_k.saturating_sub(seq_q)`, `attend_len = mask ? seq_k : min(qi + kv_offset + 1, seq_k)`, one parallel task per `(batch, head)` via `par_chunks_mut(out, seq_q*head_dim)`. **Explicit panics** (Rust index/arithmetic panics): `k.ndim() < 3` (Rust reads `ks[2]` unchecked), `n_kv_heads == 0` and `n_heads < n_kv_heads` (divisions by zero), K/V data shorter than `[batch, n_kv_heads, seq_k, head_dim]`, Q row or gathered element past the Q data, mask shorter than `[seq_q, seq_k]`; `causal_mask` panics if the tensor cannot be built (Rust `unwrap`). Result-path: `RankMismatch{expected: 4, got: q.ndim()}` only.

- [ ] **Step 1: Write the failing tests** — `tests/attention_test.cpp`

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <optional>
#include <vector>

#include "sapient/backends_cpu/kernels/attention.hpp"
#include "sapient/core/tensor.hpp"

using namespace sapient::backends_cpu::kernels::attention;
using sapient::core::Shape;
using sapient::core::Tensor;

namespace {
Tensor f32(std::vector<float> data, Shape shape) {
    auto r = Tensor::from_f32_vec(std::move(data), std::move(shape));
    if (!r) throw std::runtime_error(r.error().to_string());
    return std::move(*r);
}
std::vector<float> vec(const Tensor& x) {
    const auto s = x.f32_slice();
    return {s.begin(), s.end()};
}
} // namespace

TEST(Attention, mha_output_shape) {
    // batch=1, heads=2, seq=3, dim=4
    const auto q = f32(std::vector<float>(24, 0.1f), Shape{1, 2, 3, 4});
    const auto k = f32(std::vector<float>(24, 0.1f), Shape{1, 2, 3, 4});
    const auto v = f32(std::vector<float>(24, 0.1f), Shape{1, 2, 3, 4});
    auto out = scaled_dot_product_attention(q, k, v, nullptr, std::nullopt, 2);
    ASSERT_TRUE(out.has_value()) << out.error().to_string();
    EXPECT_EQ(out->shape().dims, (std::vector<size_t>{1, 2, 3, 4}));
}

TEST(Attention, gqa_kv_repeat) {
    // batch=1, n_heads=4, n_kv_heads=2, seq=2, dim=4
    const auto q = f32(std::vector<float>(32, 0.1f), Shape{1, 4, 2, 4});
    const auto k = f32(std::vector<float>(16, 0.1f), Shape{1, 2, 2, 4});
    const auto v = f32(std::vector<float>(16, 0.1f), Shape{1, 2, 2, 4});
    auto out = scaled_dot_product_attention(q, k, v, nullptr, std::nullopt, 2);
    ASSERT_TRUE(out.has_value()) << out.error().to_string();
    EXPECT_EQ(out->shape().dims, (std::vector<size_t>{1, 4, 2, 4}));
}

TEST(Attention, causal_mask_shape) {
    const Tensor m = causal_mask(3, 3);
    const auto d = m.f32_slice();
    // Position (0,1) should be -inf (index 1)
    EXPECT_TRUE(std::isinf(d[1]) && d[1] < 0.0f);
    // Position (1,0) should be 0 (index 3)
    EXPECT_EQ(d[3], 0.0f);
}

// Leading -inf mask positions (a sliding window that has scrolled past them) must not NaN the
// online softmax — regression for the Gemma3 "coherent until position 512, then salad" bug.
TEST(Attention, leading_masked_positions_do_not_nan) {
    const size_t seq_k = 8, head_dim = 4;
    const auto q = f32(std::vector<float>(head_dim, 1.0f), Shape{1, 1, 1, head_dim});
    std::vector<float> kv(seq_k * head_dim);
    for (size_t i = 0; i < kv.size(); ++i) kv[i] = static_cast<float>(i % 7) * 0.1f;
    const auto k = f32(kv, Shape{1, 1, seq_k, head_dim});
    const auto v = f32(kv, Shape{1, 1, seq_k, head_dim});
    // Window of 3: only the last 3 positions visible.
    std::vector<float> mask(seq_k, -std::numeric_limits<float>::infinity());
    for (size_t i = seq_k - 3; i < seq_k; ++i) mask[i] = 0.0f;
    const auto mask_t = f32(mask, Shape{1, seq_k});
    auto out = scaled_dot_product_attention(q, k, v, &mask_t, std::nullopt, 1);
    ASSERT_TRUE(out.has_value()) << out.error().to_string();
    const auto ov = out->to_f32_vec();
    for (float x : ov) EXPECT_TRUE(std::isfinite(x)) << "NaN/inf in output";
    // Reference: naive softmax over the visible 3 positions.
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores;
    for (size_t ki = seq_k - 3; ki < seq_k; ++ki) {
        float s = 0.0f;
        for (size_t d = 0; d < head_dim; ++d) s += kv[ki * head_dim + d];
        scores.push_back(s * scale);
    }
    float mx = -std::numeric_limits<float>::infinity();
    for (float s : scores) mx = std::max(mx, s);
    float sum = 0.0f;
    for (float& s : scores) {
        s = std::exp(s - mx);
        sum += s;
    }
    for (size_t d = 0; d < head_dim; ++d) {
        float want = 0.0f;
        for (size_t j = 0; j < 3; ++j) want += scores[j] / sum * kv[(seq_k - 3 + j) * head_dim + d];
        EXPECT_LT(std::fabs(ov[d] - want), 1e-5f) << "d" << d << ": " << ov[d] << " vs " << want;
    }
}

TEST(Attention, uniform_attention_recovers_v) {
    // Equal scores → uniform weights → the output is the mean of the V rows: (1+2+3+4)/4 = 2.5.
    const size_t seq = 4, dim = 8;
    std::vector<float> v_data(seq * dim);
    for (size_t i = 0; i < seq; ++i)
        for (size_t d = 0; d < dim; ++d) v_data[i * dim + d] = static_cast<float>(i + 1);
    const auto q = f32(std::vector<float>(seq * dim, 1.0f), Shape{1, 1, seq, dim});
    const auto k = f32(std::vector<float>(seq * dim, 1.0f), Shape{1, 1, seq, dim});
    const auto v = f32(v_data, Shape{1, 1, seq, dim});
    // Explicit all-zero mask so every key is attended (no causal masking).
    const auto mask = f32(std::vector<float>(seq * seq, 0.0f), Shape{seq, seq});
    auto out = scaled_dot_product_attention(q, k, v, &mask, std::nullopt, 1);
    ASSERT_TRUE(out.has_value()) << out.error().to_string();
    const float expected = (1.0f + 2.0f + 3.0f + 4.0f) / static_cast<float>(seq);
    for (float val : out->f32_slice()) EXPECT_LT(std::fabs(val - expected), 1e-4f) << "Expected ~" << expected << ", got " << val;
}

// The online-softmax result matches a naive reference implementation.
TEST(Attention, flash_matches_naive) {
    const size_t batch = 1, n_heads = 2, seq_q = 4, seq_k = 4, head_dim = 8;
    const auto gen = [](size_t i) { return ::sinf(static_cast<float>(i) * 1.3f + 0.7f) * 0.5f + 0.5f; };
    std::vector<float> q_data(batch * n_heads * seq_q * head_dim), k_data(batch * n_heads * seq_k * head_dim),
        v_data(batch * n_heads * seq_k * head_dim);
    for (size_t i = 0; i < q_data.size(); ++i) q_data[i] = gen(i);
    for (size_t i = 0; i < k_data.size(); ++i) k_data[i] = gen(i + 100);
    for (size_t i = 0; i < v_data.size(); ++i) v_data[i] = gen(i + 200);
    const auto q = f32(q_data, Shape{batch, n_heads, seq_q, head_dim});
    const auto k = f32(k_data, Shape{batch, n_heads, seq_k, head_dim});
    const auto v = f32(v_data, Shape{batch, n_heads, seq_k, head_dim});
    const Tensor mask_t = causal_mask(seq_q, seq_k);
    auto flash_out = scaled_dot_product_attention(q, k, v, &mask_t, std::nullopt, n_heads);
    ASSERT_TRUE(flash_out.has_value()) << flash_out.error().to_string();

    // --- Naive reference ---
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const auto mask_data = mask_t.f32_slice();
    std::vector<float> ref_out(batch * n_heads * seq_q * head_dim, 0.0f);
    for (size_t b = 0; b < batch; ++b)
        for (size_t h = 0; h < n_heads; ++h) {
            const size_t q_off = (b * n_heads + h) * seq_q * head_dim;
            const size_t k_off = (b * n_heads + h) * seq_k * head_dim;
            const size_t v_off = k_off;
            const size_t o_off = q_off;
            for (size_t qi = 0; qi < seq_q; ++qi) {
                std::vector<float> scores(seq_k, 0.0f);
                for (size_t ki = 0; ki < seq_k; ++ki) {
                    float dot = 0.0f;
                    for (size_t d = 0; d < head_dim; ++d)
                        dot += q_data[q_off + qi * head_dim + d] * k_data[k_off + ki * head_dim + d];
                    scores[ki] = dot * scale + mask_data[qi * seq_k + ki];
                }
                float max_s = -std::numeric_limits<float>::infinity();
                for (float s : scores) max_s = std::max(max_s, s);
                if (std::isinf(max_s)) max_s = 0.0f;
                float sum = 0.0f;
                for (float& s : scores) {
                    s = std::exp(s - max_s);
                    sum += s;
                }
                if (sum < std::numeric_limits<float>::epsilon()) sum = std::numeric_limits<float>::epsilon();
                for (size_t d = 0; d < head_dim; ++d) {
                    float acc = 0.0f;
                    for (size_t ki = 0; ki < seq_k; ++ki) acc += scores[ki] / sum * v_data[v_off + ki * head_dim + d];
                    ref_out[o_off + qi * head_dim + d] = acc;
                }
            }
        }
    const auto flash_data = vec(*flash_out);
    ASSERT_EQ(flash_data.size(), ref_out.size());
    for (size_t i = 0; i < flash_data.size(); ++i) {
        const float diff = std::fabs(flash_data[i] - ref_out[i]);
        EXPECT_LT(diff, 1e-4f) << "Mismatch at index " << i << ": flash=" << flash_data[i] << " ref=" << ref_out[i];
    }
}

// Decode-mode (seq_q=1): output shape and no NaN/Inf.
TEST(Attention, decode_mode_no_nan) {
    const size_t batch = 1, n_heads = 4, seq_q = 1, seq_k = 16, head_dim = 8;
    const auto q = f32(std::vector<float>(batch * n_heads * seq_q * head_dim, 0.1f), Shape{batch, n_heads, seq_q, head_dim});
    const auto k = f32(std::vector<float>(batch * n_heads * seq_k * head_dim, 0.1f), Shape{batch, n_heads, seq_k, head_dim});
    const auto v = f32(std::vector<float>(batch * n_heads * seq_k * head_dim, 0.2f), Shape{batch, n_heads, seq_k, head_dim});
    auto out = scaled_dot_product_attention(q, k, v, nullptr, std::nullopt, n_heads);
    ASSERT_TRUE(out.has_value()) << out.error().to_string();
    EXPECT_EQ(out->shape().dims, (std::vector<size_t>{batch, n_heads, seq_q, head_dim}));
    for (float val : out->f32_slice()) EXPECT_TRUE(std::isfinite(val)) << "NaN/Inf in decode output: " << val;
}

// C++-only: the one Result-path error and one Rust index panic.
TEST(Attention, rank_error_and_kv_panic) {
    const auto q3 = f32(std::vector<float>(8, 0.1f), Shape{1, 2, 4});
    const auto q = f32(std::vector<float>(8, 0.1f), Shape{1, 1, 2, 4});
    const auto k = f32(std::vector<float>(8, 0.1f), Shape{1, 1, 2, 4});
    EXPECT_EQ(scaled_dot_product_attention(q3, k, k, nullptr, std::nullopt, 1).error().to_string(),
              "Rank mismatch: expected 4, got 3");
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    const auto k2 = f32(std::vector<float>(8, 0.1f), Shape{2, 4}); // Rust: ks[2] index panic
    EXPECT_DEATH((void)scaled_dot_product_attention(q, k2, k2, nullptr, std::nullopt, 1), "rank");
}
```

- [ ] **Step 2: Add to CMake, build to verify failure**

Add `src/kernels/attention.cpp` and `tests/attention_test.cpp` to `CMakeLists.txt`.
Run: `cd cpp && cmake --preset dev && cmake --build --preset dev`
Expected: FAILS — `'sapient/backends_cpu/kernels/attention.hpp' file not found`.

- [ ] **Step 3: `attention.hpp`**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-backends/cpu/src/kernels/attention.rs — Flash-Edge online-softmax
// attention (never materialises the seq_q × seq_k score matrix), causal masking via -inf,
// grouped-query attention by KV-head repeat, NEON dot/saxpby on aarch64.

#include <cstddef>
#include <optional>

#include "sapient/core/error.hpp"
#include "sapient/core/tensor.hpp"

namespace sapient::backends_cpu::kernels::attention {

using sapient::core::Result;
using sapient::core::Tensor;

/// q: [batch, n_heads, seq_q, head_dim]; k, v: [batch, n_kv_heads, seq_k, head_dim];
/// mask: optional additive [seq_q, seq_k] (−inf = masked). `mask == nullptr` means BUILT-IN CAUSAL
/// masking with the KV-cache offset (seq_k − seq_q) — pass an explicit all-zeros mask for
/// non-causal attention (the Whisper/SigLIP trap). `scale` defaults to 1/sqrt(head_dim).
/// Output: [batch, n_heads, seq_q, head_dim]. Parallel over (batch, head).
Result<Tensor> scaled_dot_product_attention(const Tensor& q,
                                            const Tensor& k,
                                            const Tensor& v,
                                            const Tensor* mask,
                                            std::optional<float> scale,
                                            size_t n_kv_heads);

/// Additive causal mask [seq_q, seq_k]: 0 where ki <= qi + (seq_k − seq_q), −inf beyond.
Tensor causal_mask(size_t seq_q, size_t seq_k);

} // namespace sapient::backends_cpu::kernels::attention
```

- [ ] **Step 4: `attention.cpp`**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/backends_cpu/kernels/attention.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

#include "sapient/backends_cpu/parallel.hpp"
#include "sapient/core/panic.hpp"

namespace sapient::backends_cpu::kernels::attention {

using sapient::core::Error;
using sapient::core::F32Cow;
using sapient::core::Shape;

namespace {

// ── SIMD helpers (attention.rs:29-85) ────────────────────────────────────────
#if defined(__aarch64__) || defined(_M_ARM64)
// 4-wide vfmaq_f32 into ONE accumulator, vaddvq_f32, then the scalar tail added after the
// horizontal reduction — this exact shape is what the golden dumps pin.
float dot_f32_neon(const float* a, const float* b, size_t n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) acc = vfmaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
    float s = vaddvq_f32(acc);
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

// o[i] = alpha * o[i] + beta * v[i]: vfmaq_f32(vmulq_f32(vo, va), vv, vb).
void saxpby_neon(float* o, const float* v, size_t n, float alpha, float beta) {
    const float32x4_t va = vdupq_n_f32(alpha);
    const float32x4_t vb = vdupq_n_f32(beta);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t vo = vld1q_f32(o + i);
        const float32x4_t vv = vld1q_f32(v + i);
        vst1q_f32(o + i, vfmaq_f32(vmulq_f32(vo, va), vv, vb));
    }
    for (; i < n; ++i) o[i] = alpha * o[i] + beta * v[i];
}
#else
// Scalar fallbacks (attention.rs:50, :78). The dot is `iter().zip().map().sum()` — seed -0.0.
float dot_f32_neon(const float* a, const float* b, size_t n) {
    float s = -0.0f;
    for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}
void saxpby_neon(float* o, const float* v, size_t n, float alpha, float beta) {
    for (size_t i = 0; i < n; ++i) o[i] = alpha * o[i] + beta * v[i];
}
#endif

// ── Flash-Edge online-softmax kernel for one query row (attention.rs:98) ──────
void flash_attn_row(const float* q_row,   // [head_dim]
                    const float* k_head,  // [seq_k * head_dim] contiguous
                    const float* v_head,  // [seq_k * head_dim] contiguous
                    float* o_row,         // [head_dim], written in place
                    float scale,
                    size_t head_dim,
                    size_t attend_len,    // k/v positions to visit (causal: qi + offset + 1)
                    const float* mask_row // optional additive mask row, length seq_k
) {
    float m = -std::numeric_limits<float>::infinity(); // running max
    float l = 0.0f;                                    // running sum of exp weights
    for (size_t d = 0; d < head_dim; ++d) o_row[d] = 0.0f;

    for (size_t ki = 0; ki < attend_len; ++ki) {
        const float* k_row = k_head + ki * head_dim;
        const float raw_s = dot_f32_neon(q_row, k_row, head_dim) * scale;
        // Rust: raw_s + mask_row.map(|m| m[ki]).unwrap_or(0.0) — the add happens either way.
        const float s = raw_s + (mask_row != nullptr ? mask_row[ki] : 0.0f);

        // Fully-masked position: skip — while m is still -inf the update below would compute
        // exp(-inf - -inf) = NaN and poison the row (Gemma3 sliding-window regression).
        if (s == -std::numeric_limits<float>::infinity()) continue;

        const float m_new = s > m ? s : m;
        const float p = ::expf(s - m_new);
        const float correction = ::expf(m - m_new);
        saxpby_neon(o_row, v_head + ki * head_dim, head_dim, correction, p); // O = c·O + p·v[ki]
        l = correction * l + p;
        m = m_new;
    }

    const float inv_l = l == 0.0f ? 1.0f / FLT_EPSILON : 1.0f / l;
    for (size_t d = 0; d < head_dim; ++d) o_row[d] *= inv_l;
}

} // namespace

Result<Tensor> scaled_dot_product_attention(const Tensor& q,
                                            const Tensor& k,
                                            const Tensor& v,
                                            const Tensor* mask,
                                            std::optional<float> scale,
                                            size_t n_kv_heads) {
    const std::vector<size_t> qs = q.shape().dims;
    const std::vector<size_t> ks = k.shape().dims;
    if (qs.size() != 4) return tl::unexpected(Error::rank_mismatch(4, qs.size()));
    if (ks.size() < 3) sapient::core::panic("scaled_dot_product_attention: k must have rank >= 3"); // Rust: ks[2]

    const size_t batch = qs[0], n_heads = qs[1], seq_q = qs[2], head_dim = qs[3];
    const size_t seq_k = ks[2];
    const float sc = scale.has_value() ? *scale : 1.0f / ::sqrtf(static_cast<float>(head_dim));
    if (n_kv_heads == 0) sapient::core::panic("scaled_dot_product_attention: n_kv_heads must be non-zero");
    const size_t kv_rep = n_heads / n_kv_heads; // 1 for MHA, >1 for GQA

    // K and V contiguous f32 once (they may be strided KV-cache views); Q zero-copy when possible.
    const std::vector<float> k_data = k.to_contiguous_f32_vec();
    const std::vector<float> v_data = v.to_contiguous_f32_vec();
    const F32Cow q_cow = q.to_f32_cow();
    const std::span<const float> q_data = q_cow.get();
    const std::span<const size_t> q_strides = q.strides();
    std::optional<F32Cow> mask_cow;
    std::span<const float> mask_data;
    if (mask != nullptr) {
        mask_cow = mask->to_f32_cow();
        mask_data = mask_cow->get();
    }

    const size_t kv_offset = seq_k >= seq_q ? seq_k - seq_q : 0; // saturating_sub: cached prefix
    const size_t head_out_size = seq_q * head_dim;
    const size_t kv_head_size = seq_k * head_dim;
    std::vector<float> out(batch * n_heads * head_out_size, 0.0f);

    parallel::par_chunks_mut(out, head_out_size, [&](size_t bh, std::span<float> out_chunk) {
        const size_t b = bh / n_heads;
        const size_t h = bh % n_heads;
        if (kv_rep == 0) sapient::core::panic("scaled_dot_product_attention: n_heads < n_kv_heads"); // Rust: h / 0
        const size_t kv_h = h / kv_rep;
        const size_t kv_base = (b * n_kv_heads + kv_h) * kv_head_size;
        if (kv_base + kv_head_size > k_data.size() || kv_base + kv_head_size > v_data.size())
            sapient::core::panic("scaled_dot_product_attention: k/v shorter than [batch, n_kv_heads, seq_k, head_dim]");
        const float* k_head = k_data.data() + kv_base;
        const float* v_head = v_data.data() + kv_base;

        std::vector<float> q_row_owned;
        for (size_t qi = 0; qi < seq_q; ++qi) {
            const size_t q_base_elem = b * q_strides[0] + h * q_strides[1] + qi * q_strides[2];
            const float* q_row = nullptr;
            if (q_strides[3] == 1) { // contiguous along head_dim: zero-copy
                if (q_base_elem + head_dim > q_data.size())
                    sapient::core::panic("scaled_dot_product_attention: q row out of range");
                q_row = q_data.data() + q_base_elem;
            } else {
                q_row_owned.resize(head_dim);
                for (size_t d = 0; d < head_dim; ++d) {
                    const size_t idx = q_base_elem + d * q_strides[3];
                    if (idx >= q_data.size()) sapient::core::panic("scaled_dot_product_attention: q element out of range");
                    q_row_owned[d] = q_data[idx];
                }
                q_row = q_row_owned.data();
            }
            // An explicit mask governs (it may already encode causality); otherwise built-in causal.
            const size_t attend_len = mask != nullptr ? seq_k : std::min(qi + kv_offset + 1, seq_k);
            const float* mask_row = nullptr;
            if (mask != nullptr) {
                if ((qi + 1) * seq_k > mask_data.size())
                    sapient::core::panic("scaled_dot_product_attention: mask shorter than [seq_q, seq_k]");
                mask_row = mask_data.data() + qi * seq_k;
            }
            flash_attn_row(q_row, k_head, v_head, out_chunk.data() + qi * head_dim, sc, head_dim, attend_len, mask_row);
        }
    });

    return Tensor::from_f32_vec(std::move(out), Shape{batch, n_heads, seq_q, head_dim});
}

Tensor causal_mask(size_t seq_q, size_t seq_k) {
    std::vector<float> data(seq_q * seq_k, 0.0f);
    const size_t offset = seq_k >= seq_q ? seq_k - seq_q : 0;
    for (size_t qi = 0; qi < seq_q; ++qi)
        for (size_t ki = 0; ki < seq_k; ++ki)
            if (ki > qi + offset) data[qi * seq_k + ki] = -std::numeric_limits<float>::infinity();
    auto t = Tensor::from_f32_vec(std::move(data), Shape{seq_q, seq_k});
    if (!t.has_value()) sapient::core::panic("causal_mask: " + t.error().to_string()); // Rust: unwrap
    return std::move(*t);
}

} // namespace sapient::backends_cpu::kernels::attention
```

- [ ] **Step 5: Build and run**

Run: `cd cpp && cmake --build --preset dev && ctest --preset dev -R Attention`
Expected: 8 tests pass (7 Rust names + `rank_error_and_kv_panic`).

- [ ] **Step 6: Format, then commit**

```bash
git add cpp/libs/sapient-backends-cpu
git ls-files -- 'cpp/*.hpp' 'cpp/*.cpp' | xargs .superpowers/tools-venv/bin/clang-format -i
git add cpp/libs/sapient-backends-cpu
git commit -m "cpp(backends-cpu): kernels/attention — Flash-Edge online softmax, GQA, NEON dot/saxpby (7 Rust tests by name)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: `kernels/conv2d.hpp/.cpp` — im2col + blocked GEMM convolution

**Files:**
- Modify: `cpp/libs/sapient-backends-cpu/CMakeLists.txt` (add `src/kernels/conv2d.cpp`, `tests/conv2d_test.cpp`)
- Create: `include/sapient/backends_cpu/kernels/conv2d.hpp`, `src/kernels/conv2d.cpp`
- Test: `tests/conv2d_test.cpp`

**Interfaces:**
- Consumes: `parallel::{num_threads, par_chunks_mut}` (Task 1), `sgemm` (Task 2), `sapient::core::{Tensor, Shape, Result, Error, F32Cow, panic}`.
- Produces (namespace `sapient::backends_cpu::kernels::conv2d`): `Result<Tensor> conv2d(const Tensor& x, const Tensor& weight, const Tensor* bias, std::array<size_t, 2> kernel_shape, std::array<size_t, 4> pads /*top,left,bottom,right*/, std::array<size_t, 2> strides, std::array<size_t, 2> dilations, size_t groups)`; `extern std::atomic<uint64_t> IM2COL_NS, GEMM_NS` (accumulated nanoseconds, Relaxed). Task 8's `conv2d_s1`/`conv2d_s2` dumps gate it with the sgemm tolerance.

**Rust behaviour to mirror (conv2d.rs, porting map §3/§8):** `h_out = (h_in + pad_t + pad_b − dil_h·(kh−1) − 1) / stride_h + 1` (same for w); the im2col matrix `col[c_in_g·kh·kw][h_out·w_out]` is filled one row per parallel task (`par_chunks_mut(col, col_cols)`) with row decode `kj = row % kw, ki = (row/kw) % kh, ci = row/(kh·kw)`, the stride-1 fast path and zero-filled out-of-range rows; the GEMM `W_group (c_out_g × col_rows) · col (col_rows × col_cols)` is split into out-channel row blocks `mblock = flops ≥ 2^20 ? max(ceil(m/threads), 8) : m` each calling `sgemm(mc, k, n2, 1, w + w_off + m0·k, k, 1, col, n2, 1, 0, block, n2, 1)`; the bias is added during copy-out; both timing counters are bumped. **Explicit panics:** `kh == 0`/`kw == 0` and a padded input smaller than the dilated kernel (Rust: `usize` subtraction overflow — debug assertions are on under `cargo test`), zero strides (`/ 0`), `groups == 0` past the `InvalidGraph` check (`c_out / g`; unreachable because zero dims are rejected by `Shape::validate`), x/weight/bias data shorter than their shapes. Result-path: `RankMismatch{4, ndim}` for x then weight; `InvalidGraph("conv2d: groups={g}, c_in={c_in}, c_in/group={c_in_g}: {c_in_g}*{g}!=c_in")`.

- [ ] **Step 1: Write the failing tests** — `tests/conv2d_test.cpp`

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "sapient/backends_cpu/kernels/conv2d.hpp"
#include "sapient/core/tensor.hpp"

using namespace sapient::backends_cpu::kernels::conv2d;
using sapient::core::Shape;
using sapient::core::Tensor;

namespace {
Tensor f32(std::vector<float> data, Shape shape) {
    auto r = Tensor::from_f32_vec(std::move(data), std::move(shape));
    if (!r) throw std::runtime_error(r.error().to_string());
    return std::move(*r);
}
std::vector<float> vec(const Tensor& x) {
    const auto s = x.f32_slice();
    return {s.begin(), s.end()};
}
std::vector<float> lcg(size_t n, uint64_t seed) {
    std::vector<float> v(n);
    for (float& x : v) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        x = (static_cast<float>(seed >> 40) / static_cast<float>(1ULL << 24)) * 2.0f - 1.0f;
    }
    return v;
}
} // namespace

TEST(Conv2d, conv2d_identity_kernel) {
    // 1×1 conv with identity weight.
    const auto x = f32({1.0f, 2.0f, 3.0f, 4.0f}, Shape{1, 1, 2, 2});
    const auto w = f32({1.0f}, Shape{1, 1, 1, 1});
    auto y = conv2d(x, w, nullptr, {1, 1}, {0, 0, 0, 0}, {1, 1}, {1, 1}, 1);
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_EQ(vec(*y), (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));
}

// C++-only: padding, stride, dilation, groups and bias against a naive double reference (the
// Rust unit test only covers the identity kernel; the golden dumps cover groups=1).
TEST(Conv2d, matches_naive_reference_with_padding_stride_dilation_groups) {
    const size_t n = 2, c_in = 4, h_in = 7, w_in = 6, c_out = 6, groups = 2, kh = 3, kw = 2;
    const size_t c_in_g = c_in / groups, c_out_g = c_out / groups;
    const std::array<size_t, 4> pads = {1, 0, 2, 1};
    const std::array<size_t, 2> strides = {2, 1};
    const std::array<size_t, 2> dilations = {1, 2};
    const auto xv = lcg(n * c_in * h_in * w_in, 11);
    const auto wv = lcg(c_out * c_in_g * kh * kw, 12);
    const auto bv = lcg(c_out, 13);
    const auto x = f32(xv, Shape{n, c_in, h_in, w_in});
    const auto w = f32(wv, Shape{c_out, c_in_g, kh, kw});
    const auto b = f32(bv, Shape{c_out});
    const uint64_t im2col_before = IM2COL_NS.load();
    const uint64_t gemm_before = GEMM_NS.load();
    auto y = conv2d(x, w, &b, {kh, kw}, pads, strides, dilations, groups);
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    const size_t h_out = (h_in + pads[0] + pads[2] - dilations[0] * (kh - 1) - 1) / strides[0] + 1;
    const size_t w_out = (w_in + pads[1] + pads[3] - dilations[1] * (kw - 1) - 1) / strides[1] + 1;
    EXPECT_EQ(y->shape().dims, (std::vector<size_t>{n, c_out, h_out, w_out}));
    EXPECT_GE(IM2COL_NS.load(), im2col_before);
    EXPECT_GE(GEMM_NS.load(), gemm_before);

    const auto got = vec(*y);
    float max_ref = 1.0f;
    std::vector<float> ref(got.size(), 0.0f);
    for (size_t bi = 0; bi < n; ++bi)
        for (size_t co = 0; co < c_out; ++co) {
            const size_t g = co / c_out_g;
            for (size_t oh = 0; oh < h_out; ++oh)
                for (size_t ow = 0; ow < w_out; ++ow) {
                    double acc = bv[co];
                    for (size_t cg = 0; cg < c_in_g; ++cg)
                        for (size_t ki = 0; ki < kh; ++ki)
                            for (size_t kj = 0; kj < kw; ++kj) {
                                const long ih = static_cast<long>(oh * strides[0] + ki * dilations[0]) - static_cast<long>(pads[0]);
                                const long iw = static_cast<long>(ow * strides[1] + kj * dilations[1]) - static_cast<long>(pads[1]);
                                if (ih < 0 || iw < 0 || ih >= static_cast<long>(h_in) || iw >= static_cast<long>(w_in)) continue;
                                const size_t ci = g * c_in_g + cg;
                                acc += static_cast<double>(xv[((bi * c_in + ci) * h_in + static_cast<size_t>(ih)) * w_in + static_cast<size_t>(iw)]) *
                                       static_cast<double>(wv[((co * c_in_g + cg) * kh + ki) * kw + kj]);
                            }
                    const size_t idx = ((bi * c_out + co) * h_out + oh) * w_out + ow;
                    ref[idx] = static_cast<float>(acc);
                    max_ref = std::max(max_ref, std::fabs(ref[idx]));
                }
        }
    for (size_t i = 0; i < got.size(); ++i) ASSERT_NEAR(got[i], ref[i], 1e-4f * max_ref) << "index " << i;
}

// C++-only: the parity-bound InvalidGraph text and the rank check order.
TEST(Conv2d, error_messages_match_rust) {
    const auto x = f32(std::vector<float>(1 * 4 * 2 * 2, 0.0f), Shape{1, 4, 2, 2});
    const auto w = f32(std::vector<float>(3 * 2 * 1 * 1, 0.0f), Shape{3, 2, 1, 1});
    auto e = conv2d(x, w, nullptr, {1, 1}, {0, 0, 0, 0}, {1, 1}, {1, 1}, 3);
    ASSERT_FALSE(e.has_value());
    EXPECT_EQ(e.error().to_string(), "Graph validation failed: conv2d: groups=3, c_in=4, c_in/group=2: 2*3!=c_in");
    const auto x3 = f32(std::vector<float>(4, 0.0f), Shape{1, 2, 2});
    EXPECT_EQ(conv2d(x3, w, nullptr, {1, 1}, {0, 0, 0, 0}, {1, 1}, {1, 1}, 1).error().to_string(),
              "Rank mismatch: expected 4, got 3");
}
```

- [ ] **Step 2: Add to CMake, build to verify failure**

Add `src/kernels/conv2d.cpp` and `tests/conv2d_test.cpp` to `CMakeLists.txt`.
Run: `cd cpp && cmake --preset dev && cmake --build --preset dev`
Expected: FAILS — `'sapient/backends_cpu/kernels/conv2d.hpp' file not found`.

- [ ] **Step 3: `conv2d.hpp`**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-backends/cpu/src/kernels/conv2d.rs — 2-D convolution as im2col (parallel
// per row) + an out-channel-blocked sgemm (max-error gated vs Rust, spec §2.3).

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

#include "sapient/core/error.hpp"
#include "sapient/core/tensor.hpp"

namespace sapient::backends_cpu::kernels::conv2d {

using sapient::core::Result;
using sapient::core::Tensor;

/// Per-op profiling counters (accumulated nanoseconds of the im2col build vs the GEMM); the
/// Kokoro decoder profile drains these. Relaxed atomics, like Rust's `AtomicU64`.
extern std::atomic<uint64_t> IM2COL_NS;
extern std::atomic<uint64_t> GEMM_NS;

/// (N, C_in, H, W) × weight (C_out, C_in/groups, kh, kw) → (N, C_out, H_out, W_out).
/// `kernel_shape` is accepted but unused (Rust `_kernel_shape`: the kernel dims come from `weight`).
/// `pads` = [top, left, bottom, right].
Result<Tensor> conv2d(const Tensor& x,
                      const Tensor& weight,
                      const Tensor* bias,
                      std::array<size_t, 2> kernel_shape,
                      std::array<size_t, 4> pads,
                      std::array<size_t, 2> strides,
                      std::array<size_t, 2> dilations,
                      size_t groups);

} // namespace sapient::backends_cpu::kernels::conv2d
```

- [ ] **Step 4: `conv2d.cpp`**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/backends_cpu/kernels/conv2d.hpp"

#include <algorithm>
#include <chrono>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "sapient/backends_cpu/parallel.hpp"
#include "sapient/backends_cpu/sgemm.hpp"
#include "sapient/core/panic.hpp"

namespace sapient::backends_cpu::kernels::conv2d {

using sapient::core::Error;
using sapient::core::F32Cow;
using sapient::core::Shape;

std::atomic<uint64_t> IM2COL_NS{0};
std::atomic<uint64_t> GEMM_NS{0};

namespace {
uint64_t elapsed_ns(std::chrono::steady_clock::time_point t0) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
}
} // namespace

Result<Tensor> conv2d(const Tensor& x,
                      const Tensor& weight,
                      const Tensor* bias,
                      [[maybe_unused]] std::array<size_t, 2> kernel_shape,
                      std::array<size_t, 4> pads,
                      std::array<size_t, 2> strides,
                      std::array<size_t, 2> dilations,
                      size_t groups) {
    const Shape& xs = x.shape();
    const Shape& ws = weight.shape();
    if (xs.ndim() != 4) return tl::unexpected(Error::rank_mismatch(4, xs.ndim()));
    if (ws.ndim() != 4) return tl::unexpected(Error::rank_mismatch(4, ws.ndim()));

    const size_t n = xs.dims[0], c_in = xs.dims[1], h_in = xs.dims[2], w_in = xs.dims[3];
    const size_t c_out = ws.dims[0], c_in_g = ws.dims[1], kh = ws.dims[2], kw = ws.dims[3];

    const size_t g = groups;
    if (c_in != c_in_g * g)
        return tl::unexpected(Error::invalid_graph(
            "conv2d: groups=" + std::to_string(g) + ", c_in=" + std::to_string(c_in) +
            ", c_in/group=" + std::to_string(c_in_g) + ": " + std::to_string(c_in_g) + "*" +
            std::to_string(g) + "!=c_in"));

    // Rust: `(h_in + pads[0] + pads[2] - dilations[0]*(kh-1) - 1) / strides[0] + 1` — usize
    // underflow and division by zero both panic there (debug assertions on under cargo test).
    if (kh == 0 || kw == 0) sapient::core::panic("conv2d: zero kernel size");
    if (strides[0] == 0 || strides[1] == 0) sapient::core::panic("conv2d: zero stride");
    const size_t h_span = h_in + pads[0] + pads[2];
    const size_t w_span = w_in + pads[1] + pads[3];
    const size_t h_need = dilations[0] * (kh - 1) + 1;
    const size_t w_need = dilations[1] * (kw - 1) + 1;
    if (h_span < h_need || w_span < w_need) sapient::core::panic("conv2d: kernel larger than the padded input");
    const size_t h_out = (h_span - h_need) / strides[0] + 1;
    const size_t w_out = (w_span - w_need) / strides[1] + 1;
    if (g == 0) sapient::core::panic("conv2d: groups must be non-zero"); // Rust: c_out / g

    const auto x_cow = x.to_f32_cow();
    const auto x_data = x_cow.get();
    const auto w_cow = weight.to_f32_cow();
    const auto w_data = w_cow.get();
    std::optional<F32Cow> b_cow;
    std::span<const float> b_data;
    if (bias != nullptr) {
        b_cow = bias->to_f32_cow();
        b_data = b_cow->get();
    }

    const size_t col_rows = c_in_g * kh * kw;
    const size_t col_cols = h_out * w_out;
    const size_t c_out_g = c_out / g;
    if (x_data.size() < n * c_in * h_in * w_in) sapient::core::panic("conv2d: x data shorter than shape");
    if (w_data.size() < c_out * col_rows) sapient::core::panic("conv2d: weight data shorter than shape");
    if (bias != nullptr && b_data.size() < c_out) sapient::core::panic("conv2d: bias shorter than c_out");

    std::vector<float> out_data(n * c_out * h_out * w_out, 0.0f);
    std::vector<float> col;
    std::vector<float> gemm_out;

    for (size_t batch = 0; batch < n; ++batch) {
        for (size_t group = 0; group < g; ++group) {
            // ── im2col for this (batch, group): one parallel task per row ──
            const auto t_col = std::chrono::steady_clock::now();
            col.assign(col_rows * col_cols, 0.0f);
            const size_t c_start = group * c_in_g;
            parallel::par_chunks_mut(col, col_cols, [&](size_t row, std::span<float> dst) {
                const size_t kj = row % kw;
                const size_t ki = (row / kw) % kh;
                const size_t ci = row / (kh * kw);
                const size_t c = c_start + ci;
                const size_t base = batch * (c_in * h_in * w_in) + c * (h_in * w_in);
                for (size_t oh = 0; oh < h_out; ++oh) {
                    const auto ih = static_cast<std::ptrdiff_t>(oh * strides[0]) +
                                    static_cast<std::ptrdiff_t>(ki * dilations[0]) -
                                    static_cast<std::ptrdiff_t>(pads[0]);
                    float* drow = dst.data() + oh * w_out;
                    if (ih < 0 || ih >= static_cast<std::ptrdiff_t>(h_in)) {
                        std::fill(drow, drow + w_out, 0.0f);
                        continue;
                    }
                    const float* xrow = x_data.data() + base + static_cast<size_t>(ih) * w_in;
                    const auto off = static_cast<std::ptrdiff_t>(kj * dilations[1]) -
                                     static_cast<std::ptrdiff_t>(pads[1]);
                    if (strides[1] == 1) { // contiguous middle, zero edges
                        for (size_t ow = 0; ow < w_out; ++ow) {
                            const std::ptrdiff_t iw = static_cast<std::ptrdiff_t>(ow) + off;
                            drow[ow] = (iw >= 0 && static_cast<size_t>(iw) < w_in) ? xrow[iw] : 0.0f;
                        }
                    } else {
                        for (size_t ow = 0; ow < w_out; ++ow) {
                            const std::ptrdiff_t iw = static_cast<std::ptrdiff_t>(ow * strides[1]) + off;
                            drow[ow] = (iw >= 0 && static_cast<size_t>(iw) < w_in) ? xrow[iw] : 0.0f;
                        }
                    }
                }
            });
            IM2COL_NS.fetch_add(elapsed_ns(t_col), std::memory_order_relaxed);

            // ── GEMM: W_group (c_out_g × col_rows) · col (col_rows × col_cols), split over
            //    out-channel row blocks (each an independent sgemm over the same K) ──
            const auto t_gemm = std::chrono::steady_clock::now();
            const size_t w_off = group * c_out_g * (c_in_g * kh * kw);
            const size_t m = c_out_g;
            const size_t k = col_rows;
            const size_t n2 = col_cols;
            gemm_out.assign(m * n2, 0.0f);
            const size_t flops = m * k * n2;
            const size_t threads = std::max<size_t>(parallel::num_threads(), 1);
            const size_t mblock = flops >= (size_t{1} << 20) ? std::max<size_t>((m + threads - 1) / threads, 8) : m;
            parallel::par_chunks_mut(gemm_out, mblock * n2, [&](size_t bi, std::span<float> out_block) {
                const size_t m0 = bi * mblock;
                const size_t mc = out_block.size() / n2;
                sgemm(mc, k, n2, 1.0f, w_data.data() + w_off + m0 * k, static_cast<std::ptrdiff_t>(k), 1,
                      col.data(), static_cast<std::ptrdiff_t>(n2), 1, 0.0f, out_block.data(),
                      static_cast<std::ptrdiff_t>(n2), 1);
            });
            GEMM_NS.fetch_add(elapsed_ns(t_gemm), std::memory_order_relaxed);

            // ── copy-out with bias ──
            const size_t c_out_start = group * c_out_g;
            for (size_t co = 0; co < c_out_g; ++co) {
                const float bias_v = bias != nullptr ? b_data[c_out_start + co] : 0.0f;
                for (size_t hw = 0; hw < col_cols; ++hw) {
                    const size_t out_idx = batch * (c_out * h_out * w_out) + (c_out_start + co) * (h_out * w_out) + hw;
                    out_data[out_idx] = gemm_out[co * n2 + hw] + bias_v;
                }
            }
        }
    }
    return Tensor::from_f32_vec(std::move(out_data), Shape{n, c_out, h_out, w_out});
}

} // namespace sapient::backends_cpu::kernels::conv2d
```

- [ ] **Step 5: Build and run**

Run: `cd cpp && cmake --build --preset dev && ctest --preset dev -R Conv2d`
Expected: 3 tests pass (1 Rust name + 2 C++-only).

- [ ] **Step 6: Format, then commit**

```bash
git add cpp/libs/sapient-backends-cpu
git ls-files -- 'cpp/*.hpp' 'cpp/*.cpp' | xargs .superpowers/tools-venv/bin/clang-format -i
git add cpp/libs/sapient-backends-cpu
git commit -m "cpp(backends-cpu): kernels/conv2d — parallel im2col + out-channel-blocked sgemm (1 Rust test by name)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: `kernels/matmul.hpp/.cpp` — float paths, dispatcher skeleton, `gemm`, `kernels.hpp`

**Files:**
- Modify: `cpp/libs/sapient-backends-cpu/CMakeLists.txt` (add `src/kernels/matmul.cpp`, `tests/matmul_test.cpp`)
- Create: `include/sapient/backends_cpu/kernels/matmul.hpp`, `src/kernels/matmul.cpp`, `include/sapient/backends_cpu/kernels.hpp`
- Test: `tests/matmul_test.cpp`

**Interfaces:**
- Consumes: `parallel::{num_threads, par_chunks_mut}`, `cpu_features::has_avx2_fma`, `thermal::{tick, effective_threads}`, `spinpool::{enabled, parallelism}`, `env_usize` (Task 1); `sgemm` (Task 2); `sapient::core::{Tensor, Shape, DType, Result, Error, f16_bits_to_f32, to_string(DType), panic}`.
- Produces (namespace `sapient::backends_cpu::kernels::matmul`): `Result<Tensor> matmul(const Tensor& a, const Tensor& b)` ((…,M,K)×(…,K,N)); `Result<Tensor> matmul_nt(const Tensor& x, const Tensor& w)` (`x [M,K] · Wᵀ`, `W [N,K]`; the dtype dispatcher — float arms live, quant arms stubbed for plan D); `Result<Tensor> gemm(const Tensor& a, const Tensor& b, const Tensor* bias, float alpha, float beta, bool trans_a, bool trans_b)`; `namespace detail { size_t gemv_chunk(size_t n); void for_each_out_chunk(std::span<float> out, size_t chunk, const std::function<void(size_t, std::span<float>)>& f); }` (Rust-private; exposed so plan D's quant arms, plan E's pool branch and the tests share one definition). `kernels.hpp` = `kernels/mod.rs` umbrella. Task 8's dumps gate `matmul_nt_f16_m1`, `matmul_nt_f32_m1_k512` (bit-identical) and `matmul_nt_f32_m1`, `matmul_nt_f32_m4` (sgemm tolerance).

**Rust behaviour to mirror (matmul.rs:1-450, 1160-1256; porting map §2.1–2.3, §3.1–3.2, §4):**
- `matmul_nt`: 2-D guard → `internal("matmul_nt expects 2-D tensors")`; `k != k2` → `ShapeMismatch{expected: [m, k], got: [n, k2]}`; then `thermal::tick()`; then dispatch on `w.dtype()`.
- `matmul_nt_float`: (1) `m == 1 && k >= 64 && F16` → `dot_f32_x_f16` GEMV over the raw little-endian u16 weight bytes, chunked by `gemv_chunk(n)` through `for_each_out_chunk`; (2) `m == 1 && k >= 512` → `dot_f32_fast` GEMV, same chunking; (3) else sgemm over row blocks `mblock = (m >= 2 && m·k·n >= 2^20) ? max(ceil(m / threads), 4) : m` via `par_chunks_mut(out, mblock·n)`, each block `sgemm(mc, k, n, 1, x + m0·k, k, 1, w, 1, k, 0, block, n, 1)` (W transposed via strides).
- `dot_f32_neon_fast` (aarch64): 16-wide unroll of four `vfmaq_f32` into ONE accumulator, 4-wide tail, `vaddvq_f32`, scalar tail; `dot_f32_avx2` (x86_64, `target("avx2,fma")`, runtime-gated): 8-wide `_mm256_fmadd_ps`, the exact `_mm_movehdup_ps`/`_mm_movehl_ps` horizontal sum, scalar tail; non-AVX2 x86 and other targets: scalar `.sum()` (seed `-0.0f`).
- `dot_f32_x_f16_neon` (aarch64): the f16→f32 bit surgery **verbatim** (`vmovl_u16`, sign `<<16`, `exp = (u32 >> 10) << 23 + (112 << 23)`, mantissa `<<13`, `vorrq`, `vreinterpretq_f32_u32`, `vfmaq_f32`), scalar tail via `f16_bits_to_f32`. It is valid for **positive normal** f16 values only: subnormal/inf/NaN decode differently from the tail, and the unmasked `>> 10` carries the f16 sign bit into bit 5 of `exp16`, so **negative weights gain 2^5 in the exponent field (× 2^32)**. Rust's own unit test (`k = 2`) never reaches this path (`k >= 64`), and F16 linears are online-quantised to Q8_0 at load, which is why the defect is dormant. **Reproduced, not fixed** (spec §3.5: divergence included; Rust is frozen) — Task 9 records it in `docs/PARITY.md` as a known oracle defect the port carries, and the golden case `matmul_nt_f16_m1` (random signed weights) proves the reproduction is bit-exact.
- `gemv_chunk(n)`: `rayon_n = max(num_threads, 1)`; `eff = thermal::effective_threads()`; governed (`eff < rayon_n`) → `max(n / max(eff,1), 16)`; else `ncpus = spinpool::enabled() ? parallelism() : rayon_n`; `SAPIENT_GEMV_TPC` (read **every call**, `usize >= 1`) → `max(n / (ncpus·tpc), 16)`; default `clamp(n / (ncpus·4), 16, 512)`.
- `for_each_out_chunk(out, chunk, f)`: empty → return; plan C: `par_chunks_mut(out, chunk, f)` only (plan E adds the debug census + pool branch in front; same partition).
- `matmul`: rank guard `RankMismatch{2, min(a_rank, b_rank)}`; `k != k2` → `ShapeMismatch{[m,k,n],[m,k2,n]}`; batch = product of `a`'s leading dims; per batch `sgemm(m,k,n,1,a+off,k,1,b+off,n,1,0,c+off,n,1)`; output dims = `a`'s leading dims + `[m, n]`. Explicit panic: `a`/`b` data shorter than the batched shape (Rust slice panic — `b`'s leading dims are never checked, only its length).
- `gemm`: `a2 = trans_a ? a.t()? : a` (same for `b`); `m,k` from `a2`, `k2,n` from `b2` (explicit panic if either has rank < 2 — Rust indexes `dims()[1]`); `k != k2` → `ShapeMismatch{[m,k],[k2,n]}`; `sgemm(m,k,n, alpha, a, a2.strides[0], a2.strides[1], b, b2.strides[0], b2.strides[1], 0.0, out, n, 1)` (the transposed view's data is the raw buffer — `to_f32_cow` on an F32 view is unbounded and the strides do the transpose); bias `f32_slice()` (panics unless F32, like `as_f32_slice`), length `n` or `1` else `ShapeMismatch{[n],[b_len]}`, `out[i·n+j] += beta·bias`.
- **Plan D stub:** the seven quantized arms return `Error::internal("matmul_nt: quantized weights (" + to_string(dtype) + ") land in plan D")`.

- [ ] **Step 1: Write the failing tests** — `tests/matmul_test.cpp`

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "sapient/backends_cpu/kernels/matmul.hpp"
#include "sapient/backends_cpu/parallel.hpp"
#include "sapient/core/dtype.hpp"
#include "sapient/core/f16.hpp"
#include "sapient/core/tensor.hpp"

using namespace sapient::backends_cpu::kernels::matmul;
using sapient::core::DType;
using sapient::core::Shape;
using sapient::core::Tensor;

namespace {
Tensor f32(std::vector<float> data, Shape shape) {
    auto r = Tensor::from_f32_vec(std::move(data), std::move(shape));
    if (!r) throw std::runtime_error(r.error().to_string());
    return std::move(*r);
}
std::vector<float> lcg(size_t n, uint64_t seed, float lo, float hi) {
    std::vector<float> v(n);
    for (float& x : v) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        x = lo + (hi - lo) * (static_cast<float>(seed >> 40) / static_cast<float>(1ULL << 24));
    }
    return v;
}
} // namespace

TEST(Matmul, matmul_2x2) {
    // [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]]
    const auto a = f32({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2});
    const auto b = f32({5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2});
    auto c = matmul(a, b);
    ASSERT_TRUE(c.has_value()) << c.error().to_string();
    const auto d = c->f32_slice();
    EXPECT_LT(std::fabs(d[0] - 19.0f), 1e-5f);
    EXPECT_LT(std::fabs(d[1] - 22.0f), 1e-5f);
    EXPECT_LT(std::fabs(d[2] - 43.0f), 1e-5f);
    EXPECT_LT(std::fabs(d[3] - 50.0f), 1e-5f);
}

TEST(Matmul, matmul_nt_linear) {
    // x = [1,2] (1x2); W = [[1,2],[3,4],[5,6]] shape [3,2]; y = x @ Wᵀ = [5, 11, 17].
    const auto x = f32({1.0f, 2.0f}, Shape{1, 2});
    const auto w = f32({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{3, 2});
    auto y = matmul_nt(x, w);
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    const auto d = y->f32_slice();
    EXPECT_EQ(y->shape().dims, (std::vector<size_t>{1, 3}));
    EXPECT_LT(std::fabs(d[0] - 5.0f), 1e-5f);
    EXPECT_LT(std::fabs(d[1] - 11.0f), 1e-5f);
    EXPECT_LT(std::fabs(d[2] - 17.0f), 1e-5f);
}

TEST(Matmul, matmul_nt_linear_f16_weight) {
    // Same as above but W is stored as F16 — must still be correct.
    const auto x = f32({1.0f, 2.0f}, Shape{1, 2});
    std::vector<uint8_t> bytes;
    for (float v : {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}) {
        uint8_t le[2];
        sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(v), le);
        bytes.push_back(le[0]);
        bytes.push_back(le[1]);
    }
    auto w = Tensor::from_f16_bytes(bytes, Shape{3, 2});
    ASSERT_TRUE(w.has_value());
    auto y = matmul_nt(x, *w);
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    const auto d = y->f32_slice();
    EXPECT_LT(std::fabs(d[0] - 5.0f), 1e-2f);
    EXPECT_LT(std::fabs(d[1] - 11.0f), 1e-2f);
    EXPECT_LT(std::fabs(d[2] - 17.0f), 1e-2f);
}

TEST(Matmul, matmul_rank_mismatch) {
    auto a = Tensor::zeros(Shape{4}, DType::F32);
    auto b = Tensor::zeros(Shape{4}, DType::F32);
    ASSERT_TRUE(a.has_value() && b.has_value());
    EXPECT_FALSE(matmul(*a, *b).has_value());
    EXPECT_EQ(matmul(*a, *b).error().to_string(), "Rank mismatch: expected 2, got 1");
}

TEST(Matmul, gemm_with_bias) {
    const auto a = f32({1.0f, 0.0f, 0.0f, 1.0f}, Shape{2, 2});
    const auto b = f32({2.0f, 3.0f, 4.0f, 5.0f}, Shape{2, 2});
    const auto bias = f32({1.0f, 1.0f}, Shape{2});
    auto c = gemm(a, b, &bias, 1.0f, 1.0f, false, false);
    ASSERT_TRUE(c.has_value()) << c.error().to_string();
    const auto d = c->f32_slice();
    // Identity × [[2,3],[4,5]] = [[2,3],[4,5]]; + bias [1,1] = [[3,4],[5,6]]
    EXPECT_LT(std::fabs(d[0] - 3.0f), 1e-5f) << "got " << d[0];
    EXPECT_LT(std::fabs(d[1] - 4.0f), 1e-5f) << "got " << d[1];
}

// ── C++-only ─────────────────────────────────────────────────────────────────

TEST(Matmul, gemv_chunk_follows_the_rust_formula) {
    if (std::getenv("SAPIENT_GEMV_TPC") != nullptr) GTEST_SKIP() << "SAPIENT_GEMV_TPC is set";
    // Plan C: spinpool disabled and thermal inert → ncpus = num_threads(); default clamp(n/(ncpus*4), 16, 512).
    const size_t ncpus = sapient::backends_cpu::parallel::num_threads();
    for (size_t n : {size_t{1}, size_t{16}, size_t{100}, size_t{4096}, size_t{151936}})
        EXPECT_EQ(detail::gemv_chunk(n), std::clamp<size_t>(n / (ncpus * 4), 16, 512)) << "n=" << n;
}

TEST(Matmul, for_each_out_chunk_partition_matches_rayon) {
    std::vector<float> out(100, -1.0f);
    detail::for_each_out_chunk(out, 16, [](size_t ci, std::span<float> cs) {
        for (float& v : cs) v = static_cast<float>(ci);
    });
    for (size_t i = 0; i < out.size(); ++i) EXPECT_EQ(out[i], static_cast<float>(i / 16)) << i;
    std::vector<float> empty;
    detail::for_each_out_chunk(empty, 16, [](size_t, std::span<float>) { FAIL() << "called on empty"; });
}

// m=1, k=512 takes the dot_f32_fast GEMV path (NEON / AVX2 / scalar): check it against a naive dot.
TEST(Matmul, f32_gemv_path_matches_naive_dot) {
    const size_t k = 512, n = 5;
    const auto xv = lcg(k, 3, -1.0f, 1.0f);
    const auto wv = lcg(n * k, 4, -1.0f, 1.0f);
    auto y = matmul_nt(f32(xv, Shape{1, k}), f32(wv, Shape{n, k}));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    for (size_t j = 0; j < n; ++j) {
        double ref = 0.0;
        for (size_t i = 0; i < k; ++i) ref += static_cast<double>(xv[i]) * static_cast<double>(wv[j * k + i]);
        EXPECT_NEAR(y->f32_slice()[j], static_cast<float>(ref), 1e-4f) << "col " << j;
    }
}

// m=1, k>=64, F16 weights takes the dot_f32_x_f16 GEMV path. POSITIVE normals only: the Rust NEON
// bit-surgery mis-decodes negative f16 values (sign bit leaks into the exponent), which this port
// reproduces on purpose — the golden case `matmul_nt_f16_m1` pins that reproduction bit-exactly.
TEST(Matmul, f16_gemv_path_matches_widened_reference_for_positive_normals) {
    const size_t k = 64, n = 3;
    const auto xv = lcg(k, 5, -1.0f, 1.0f);
    const auto wsrc = lcg(n * k, 6, 0.01f, 1.0f);
    std::vector<uint8_t> bytes;
    std::vector<float> widened;
    for (float v : wsrc) {
        const uint16_t h = sapient::core::f32_to_f16_bits(v);
        uint8_t le[2];
        sapient::core::f16_to_le(h, le);
        bytes.push_back(le[0]);
        bytes.push_back(le[1]);
        widened.push_back(sapient::core::f16_bits_to_f32(h));
    }
    auto w = Tensor::from_f16_bytes(bytes, Shape{n, k});
    ASSERT_TRUE(w.has_value());
    auto y = matmul_nt(f32(xv, Shape{1, k}), *w);
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    for (size_t j = 0; j < n; ++j) {
        double ref = 0.0;
        for (size_t i = 0; i < k; ++i) ref += static_cast<double>(xv[i]) * static_cast<double>(widened[j * k + i]);
        EXPECT_NEAR(y->f32_slice()[j], static_cast<float>(ref), 1e-4f) << "col " << j;
    }
}

// DELETE IN PLAN D (when the quant arms land). Pins the plan-C stub so quantized weights never
// yield a silent wrong result.
TEST(Matmul, quantized_weight_dtypes_are_stubbed_until_plan_d) {
    const auto x = f32(std::vector<float>(64, 0.5f), Shape{1, 64});
    auto w = Tensor::from_quant_bytes(std::vector<uint8_t>(2 * 34, 0), Shape{1, 64}, DType::Q8_0);
    ASSERT_TRUE(w.has_value()) << w.error().to_string();
    auto y = matmul_nt(x, *w);
    ASSERT_FALSE(y.has_value());
    EXPECT_EQ(y.error().to_string(),
              "Internal error: matmul_nt: quantized weights (" + sapient::core::to_string(DType::Q8_0) + ") land in plan D");
}

// C++-only: the parity-bound Result-path texts of matmul_nt / matmul / gemm.
TEST(Matmul, error_messages_match_rust) {
    const auto x = f32({1.0f, 2.0f}, Shape{1, 2});
    const auto w3 = f32({1.0f, 2.0f, 3.0f}, Shape{3, 1});
    const auto x1 = f32({1.0f, 2.0f}, Shape{2});
    EXPECT_EQ(matmul_nt(x1, w3).error().to_string(), "Internal error: matmul_nt expects 2-D tensors");
    EXPECT_EQ(matmul_nt(x, w3).error().to_string(), "Shape mismatch: expected [1, 2], got [3, 1]");
    EXPECT_EQ(matmul(x, w3).error().to_string(), "Shape mismatch: expected [1, 2, 1], got [1, 3, 1]");
    EXPECT_EQ(gemm(x, w3, nullptr, 1.0f, 0.0f, false, false).error().to_string(),
              "Shape mismatch: expected [1, 2], got [3, 1]");
    const auto b = f32({1.0f, 2.0f}, Shape{2, 1});
    const auto bad_bias = f32({1.0f, 2.0f, 3.0f}, Shape{3});
    EXPECT_EQ(gemm(x, b, &bad_bias, 1.0f, 1.0f, false, false).error().to_string(),
              "Shape mismatch: expected [1], got [3]");
}
```

- [ ] **Step 2: Add to CMake, build to verify failure**

Add `src/kernels/matmul.cpp` and `tests/matmul_test.cpp` to `CMakeLists.txt`.
Run: `cd cpp && cmake --preset dev && cmake --build --preset dev`
Expected: FAILS — `'sapient/backends_cpu/kernels/matmul.hpp' file not found`.

- [ ] **Step 3: `matmul.hpp` and `kernels.hpp`**

`include/sapient/backends_cpu/kernels/matmul.hpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-backends/cpu/src/kernels/matmul.rs. Plan C: `matmul`, the `matmul_nt`
// dispatcher with its FLOAT paths (F16 GEMV bit-surgery, f32 GEMV, sgemm prefill row blocks),
// `gemm`, `gemv_chunk`, `for_each_out_chunk`. Plan D adds the seven quantized arms (today they
// return an explicit Error); plan E adds the spin-pool branch of `for_each_out_chunk`.

#include <cstddef>
#include <functional>
#include <span>

#include "sapient/core/error.hpp"
#include "sapient/core/tensor.hpp"

namespace sapient::backends_cpu::kernels::matmul {

using sapient::core::Result;
using sapient::core::Tensor;

/// (…, M, K) × (…, K, N) → (…, M, N); batch = product of `a`'s leading dims. sgemm-backed.
Result<Tensor> matmul(const Tensor& a, const Tensor& b);

/// Linear projection x [M, K] · Wᵀ with W stored [N, K] (PyTorch nn.Linear layout) → [M, N].
/// Dispatches on W's dtype without expanding quantized weights (plan D); float weights take the
/// F16 GEMV (m == 1, k ≥ 64, F16), the f32 GEMV (m == 1, k ≥ 512) or the blocked sgemm path.
Result<Tensor> matmul_nt(const Tensor& x, const Tensor& w);

/// C = alpha · op(A) × op(B) + beta · bias (bias [n] or [1], broadcast over rows).
Result<Tensor> gemm(const Tensor& a,
                    const Tensor& b,
                    const Tensor* bias,
                    float alpha,
                    float beta,
                    bool trans_a,
                    bool trans_b);

namespace detail {
/// Rust `gemv_chunk(n)`: the rows-per-task size for a GEMV over n output rows (porting map §3.1).
/// Thermal-governed → exactly `effective_threads()` tasks, no cap; else `SAPIENT_GEMV_TPC` (read
/// every call) → max(n / (ncpus·tpc), 16); default clamp(n / (ncpus·4), 16, 512).
size_t gemv_chunk(size_t n);
/// Rust `for_each_out_chunk`: `f(ci, out[ci*chunk, min((ci+1)*chunk, len)))` for every chunk.
/// Plan C runs the rayon-twin branch only; plan E adds the spin-pool branch (identical partition).
void for_each_out_chunk(std::span<float> out,
                        size_t chunk,
                        const std::function<void(size_t, std::span<float>)>& f);
} // namespace detail

} // namespace sapient::backends_cpu::kernels::matmul
```

`include/sapient/backends_cpu/kernels.hpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// = crates/sapient-backends/cpu/src/kernels/mod.rs: one include per kernel family.
// kernels/quant.hpp joins in plan D.

#include "sapient/backends_cpu/kernels/attention.hpp"
#include "sapient/backends_cpu/kernels/conv2d.hpp"
#include "sapient/backends_cpu/kernels/elementwise.hpp"
#include "sapient/backends_cpu/kernels/layernorm.hpp"
#include "sapient/backends_cpu/kernels/matmul.hpp"
#include "sapient/backends_cpu/kernels/reduce.hpp"
#include "sapient/backends_cpu/kernels/rope.hpp"
#include "sapient/backends_cpu/kernels/softmax.hpp"
```

- [ ] **Step 4: `matmul.cpp`**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/backends_cpu/kernels/matmul.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "sapient/backends_cpu/cpu_features.hpp"
#include "sapient/backends_cpu/env.hpp"
#include "sapient/backends_cpu/parallel.hpp"
#include "sapient/backends_cpu/sgemm.hpp"
#include "sapient/backends_cpu/spinpool.hpp"
#include "sapient/backends_cpu/thermal.hpp"
#include "sapient/core/dtype.hpp"
#include "sapient/core/f16.hpp"
#include "sapient/core/panic.hpp"

namespace sapient::backends_cpu::kernels::matmul {

using sapient::core::DType;
using sapient::core::Error;
using sapient::core::Shape;

namespace {

// ── f32 dot products (matmul.rs:152-241) ─────────────────────────────────────
#if defined(__aarch64__) || defined(_M_ARM64)
// 16-element unroll: four vfmaq_f32 into ONE accumulator, then a 4-wide tail, vaddvq_f32, and
// the scalar tail added after the horizontal reduction.
float dot_f32_neon_fast(const float* a, const float* b, size_t n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        acc = vfmaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
        acc = vfmaq_f32(acc, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        acc = vfmaq_f32(acc, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        acc = vfmaq_f32(acc, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    for (; i + 4 <= n; i += 4) acc = vfmaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
    float s = vaddvq_f32(acc);
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}
float dot_f32_fast(const float* a, const float* b, size_t n) { return dot_f32_neon_fast(a, b, n); }

#elif defined(__x86_64__) || defined(_M_X64)
// AVX2+FMA (the ONE runtime-gated x86 dot in this plan); the horizontal sum is the exact
// sequence from matmul.rs:206-213.
__attribute__((target("avx2,fma"))) float dot_f32_avx2(const float* a, const float* b, size_t n) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
    const __m128 lo = _mm256_castps256_ps128(acc);
    const __m128 hi = _mm256_extractf128_ps(acc, 1);
    const __m128 sum4 = _mm_add_ps(lo, hi);
    const __m128 shuf = _mm_movehdup_ps(sum4);
    const __m128 sum2 = _mm_add_ps(sum4, shuf);
    const __m128 sum1 = _mm_add_ss(sum2, _mm_movehl_ps(shuf, sum2));
    float s = _mm_cvtss_f32(sum1);
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}
float dot_f32_fast(const float* a, const float* b, size_t n) {
    if (cpu_features::has_avx2_fma()) return dot_f32_avx2(a, b, n);
    float s = -0.0f; // iter().zip().map().sum() seeds at -0.0
    for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

#else
float dot_f32_fast(const float* a, const float* b, size_t n) {
    float s = -0.0f;
    for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}
#endif

// ── f32 × f16 dot (matmul.rs:252-315) ────────────────────────────────────────
#if defined(__aarch64__) || defined(_M_ARM64)
// F16→F32 by NEON integer bit surgery, VERBATIM from Rust (spec §3.5): valid for positive normal
// f16 values only. Subnormal/inf/NaN decode differently from the scalar tail below, and the
// unmasked `>> 10` carries the f16 sign bit into bit 5 of `exp16`, so negative weights end up with
// +32 in the exponent field (×2^32). Rust is frozen; the divergence is reproduced, recorded in
// docs/PARITY.md, and pinned bit-exactly by the `matmul_nt_f16_m1` golden case.
float dot_f32_x_f16_neon(const float* a_f32, const uint16_t* b_f16, size_t n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    size_t i = 0;
    const uint32x4_t mask_mant = vdupq_n_u32(0x000003FFu); // 10-bit mantissa mask
    const uint32x4_t mask_sign = vdupq_n_u32(0x00008000u); // sign bit in u16 position
    const uint32x4_t exp_bias = vdupq_n_u32(112u << 23);   // F32 bias 127 − F16 bias 15
    for (; i + 4 <= n; i += 4) {
        const float32x4_t av = vld1q_f32(a_f32 + i);
        const uint32x4_t u32x4 = vmovl_u16(vld1_u16(b_f16 + i)); // zero-extend u16 → u32
        const uint32x4_t sign = vshlq_n_u32(vandq_u32(u32x4, mask_sign), 16);
        const uint32x4_t exp16 = vshrq_n_u32(u32x4, 10);
        const uint32x4_t exp32 = vaddq_u32(vshlq_n_u32(exp16, 23), exp_bias);
        const uint32x4_t mant = vshlq_n_u32(vandq_u32(u32x4, mask_mant), 13);
        const float32x4_t bv = vreinterpretq_f32_u32(vorrq_u32(sign, vorrq_u32(exp32, mant)));
        acc = vfmaq_f32(acc, av, bv);
    }
    float s = vaddvq_f32(acc);
    for (; i < n; ++i) s += a_f32[i] * sapient::core::f16_bits_to_f32(b_f16[i]); // half::f16::from_bits().to_f32()
    return s;
}
float dot_f32_x_f16(const float* a, const uint16_t* b, size_t n) { return dot_f32_x_f16_neon(a, b, n); }
#else
float dot_f32_x_f16(const float* a, const uint16_t* b, size_t n) {
    float s = -0.0f; // iter().zip().map().sum() seeds at -0.0
    for (size_t i = 0; i < n; ++i) s += a[i] * sapient::core::f16_bits_to_f32(b[i]);
    return s;
}
#endif

// ── float path of matmul_nt (matmul.rs:317-397) ──────────────────────────────
Result<Tensor> matmul_nt_float(const Tensor& x, const Tensor& w, size_t m, size_t k, size_t n) {
    // F16 GEMV decode: F16 weights widened per row inside NEON registers — no f32 copy of W.
    if (m == 1 && k >= 64 && w.dtype() == DType::F16) {
        const auto x_cow = x.to_f32_cow();
        const auto x_data = x_cow.get();
        const auto w_bytes = w.bytes();
        if (x_data.size() < k) sapient::core::panic("matmul_nt: x shorter than k");
        if (w_bytes.size() < 2 * n * k) sapient::core::panic("matmul_nt: F16 weight buffer shorter than n*k");
        // Rust: slice::from_raw_parts(bytes as *const u16, len/2) — F16 storage is packed
        // little-endian u16 (CpuBuffer alignment ≥ 2, F16 view offsets are multiples of 2).
        const auto* w_f16 = reinterpret_cast<const uint16_t*>(w_bytes.data());
        std::vector<float> out(n, 0.0f);
        const size_t chunk = detail::gemv_chunk(n);
        detail::for_each_out_chunk(out, chunk, [&](size_t chunk_idx, std::span<float> cs) {
            for (size_t local = 0; local < cs.size(); ++local) {
                const size_t j = chunk_idx * chunk + local;
                cs[local] = dot_f32_x_f16(x_data.data(), w_f16 + j * k, k);
            }
        });
        return Tensor::from_f32_vec(std::move(out), Shape{m, n});
    }

    const auto x_cow = x.to_f32_cow();
    const auto w_cow = w.to_f32_cow();
    const auto x_data = x_cow.get();
    const auto w_data = w_cow.get();
    if (x_data.size() < m * k || w_data.size() < n * k) sapient::core::panic("matmul_nt: operand shorter than its shape");
    std::vector<float> out(m * n, 0.0f);

    if (m == 1 && k >= 512) {
        // F32 GEMV decode — NEON/AVX2-vectorised dot products.
        const size_t chunk = detail::gemv_chunk(n);
        detail::for_each_out_chunk(out, chunk, [&](size_t chunk_idx, std::span<float> cs) {
            for (size_t local = 0; local < cs.size(); ++local) {
                const size_t j = chunk_idx * chunk + local;
                cs[local] = dot_f32_fast(x_data.data(), w_data.data() + j * k, k);
            }
        });
    } else {
        // Batched sgemm for prefill, split across X row blocks (each block an independent sgemm
        // over the same K reduction writing a disjoint output slice).
        const size_t flops = m * k * n;
        const size_t threads = std::max<size_t>(parallel::num_threads(), 1);
        const size_t mblock = (m >= 2 && flops >= (size_t{1} << 20)) ? std::max<size_t>((m + threads - 1) / threads, 4) : m;
        parallel::par_chunks_mut(out, mblock * n, [&](size_t bi, std::span<float> out_block) {
            const size_t m0 = bi * mblock;
            const size_t mc = out_block.size() / n;
            sgemm(mc, k, n, 1.0f, x_data.data() + m0 * k, static_cast<std::ptrdiff_t>(k), 1, w_data.data(), 1,
                  static_cast<std::ptrdiff_t>(k), 0.0f, out_block.data(), static_cast<std::ptrdiff_t>(n), 1);
        });
    }
    return Tensor::from_f32_vec(std::move(out), Shape{m, n});
}

} // namespace

namespace detail {

size_t gemv_chunk(size_t n) {
    // The governed comparison is within RAYON's domain (the governor sheds rayon cores; the spin
    // pool is disabled entirely while governed) — matmul.rs:416-426.
    const size_t rayon_n = std::max<size_t>(parallel::num_threads(), 1);
    const size_t eff = thermal::effective_threads();
    if (eff < rayon_n) return std::max<size_t>(n / std::max<size_t>(eff, 1), 16); // governed: no ×4, no 512 cap
    const size_t ncpus = spinpool::enabled() ? spinpool::parallelism() : rayon_n;
    const std::optional<size_t> tpc = env_usize("SAPIENT_GEMV_TPC"); // read every call, like Rust
    if (tpc.has_value() && *tpc >= 1) return std::max<size_t>(n / (ncpus * *tpc), 16);
    return std::clamp<size_t>(n / (ncpus * 4), 16, 512);
}

void for_each_out_chunk(std::span<float> out,
                        size_t chunk,
                        const std::function<void(size_t, std::span<float>)>& f) {
    if (out.empty()) return;
    // PLAN E inserts here: the SAPIENT_SPINPOOL_DEBUG census and
    // `if (spinpool::enabled()) { spinpool::pool().run(n_chunks, …); return; }` — same partition.
    parallel::par_chunks_mut(out, chunk, f);
}

} // namespace detail

// ── matmul (matmul.rs:22-100) ────────────────────────────────────────────────
Result<Tensor> matmul(const Tensor& a, const Tensor& b) {
    const Shape& as = a.shape();
    const Shape& bs = b.shape();
    if (as.ndim() < 2 || bs.ndim() < 2)
        return tl::unexpected(Error::rank_mismatch(2, std::min(as.ndim(), bs.ndim())));
    const size_t a_rank = as.ndim();
    const size_t b_rank = bs.ndim();
    const size_t m = as.dims[a_rank - 2];
    const size_t k = as.dims[a_rank - 1];
    const size_t k2 = bs.dims[b_rank - 2];
    const size_t n = bs.dims[b_rank - 1];
    if (k != k2) return tl::unexpected(Error::shape_mismatch({m, k, n}, {m, k2, n}));

    size_t batch = 1;
    for (size_t i = 0; i + 2 < a_rank; ++i) batch *= as.dims[i];

    const auto a_cow = a.to_f32_cow();
    const auto a_data = a_cow.get();
    const auto b_cow = b.to_f32_cow();
    const auto b_data = b_cow.get();
    const size_t a_stride = m * k;
    const size_t b_stride = k * n;
    const size_t c_stride = m * n;
    if (a_data.size() < batch * a_stride || b_data.size() < batch * b_stride)
        sapient::core::panic("matmul: operand data shorter than the batched shape"); // Rust: slice panic
    std::vector<float> out_data(batch * c_stride, 0.0f);
    for (size_t bi = 0; bi < batch; ++bi)
        sgemm(m, k, n, 1.0f, a_data.data() + bi * a_stride, static_cast<std::ptrdiff_t>(k), 1,
              b_data.data() + bi * b_stride, static_cast<std::ptrdiff_t>(n), 1, 0.0f,
              out_data.data() + bi * c_stride, static_cast<std::ptrdiff_t>(n), 1);

    std::vector<size_t> out_dims(as.dims.begin(), as.dims.begin() + static_cast<std::ptrdiff_t>(a_rank - 2));
    out_dims.push_back(m);
    out_dims.push_back(n);
    return Tensor::from_f32_vec(std::move(out_data), Shape(out_dims));
}

// ── matmul_nt dispatcher (matmul.rs:114-145) ─────────────────────────────────
Result<Tensor> matmul_nt(const Tensor& x, const Tensor& w) {
    const auto& xd = x.shape().dims;
    const auto& wd = w.shape().dims;
    if (xd.size() != 2 || wd.size() != 2) return tl::unexpected(Error::internal("matmul_nt expects 2-D tensors"));
    const size_t m = xd[0], k = xd[1];
    const size_t n = wd[0], k2 = wd[1];
    if (k != k2) return tl::unexpected(Error::shape_mismatch({m, k}, {n, k2}));

    // Thermal governor sample point (rate-limited inside tick; plan E gives it a body).
    thermal::tick();

    switch (w.dtype()) {
    case DType::Q4_0:
    case DType::Q8_0:
    case DType::Q4_K:
    case DType::Q4_K_R4:
    case DType::Q5_K:
    case DType::Q6_K:
    case DType::Q6_K_R4:
        // PLAN D: matmul_nt_q4_0 / q8_0 / q4_k / q4_k_r4 / q5_k / q6_k / q6_k_r4 replace this arm.
        return tl::unexpected(Error::internal("matmul_nt: quantized weights (" +
                                              sapient::core::to_string(w.dtype()) + ") land in plan D"));
    default:
        return matmul_nt_float(x, w, m, k, n);
    }
}

// ── gemm (matmul.rs:1173-1256) ───────────────────────────────────────────────
Result<Tensor> gemm(const Tensor& a,
                    const Tensor& b,
                    const Tensor* bias,
                    float alpha,
                    float beta,
                    bool trans_a,
                    bool trans_b) {
    Tensor a2 = a;
    if (trans_a) { SAPIENT_TRY_ASSIGN(a2, a.t()); }
    Tensor b2 = b;
    if (trans_b) { SAPIENT_TRY_ASSIGN(b2, b.t()); }

    if (a2.ndim() < 2 || b2.ndim() < 2) sapient::core::panic("gemm: operands must be 2-D"); // Rust: dims()[1] panic
    const size_t m = a2.shape().dims[0];
    const size_t k = a2.shape().dims[1];
    const size_t k2 = b2.shape().dims[0];
    const size_t n = b2.shape().dims[1];
    if (k != k2) return tl::unexpected(Error::shape_mismatch({m, k}, {k2, n}));

    // A transposed view's data is the raw buffer; its strides do the transpose (as in Rust).
    const auto a_cow = a2.to_f32_cow();
    const auto a_data = a_cow.get();
    const auto b_cow = b2.to_f32_cow();
    const auto b_data = b_cow.get();
    const auto a_strides = a2.strides();
    const auto b_strides = b2.strides();
    std::vector<float> out(m * n, 0.0f);
    sgemm(m, k, n, alpha, a_data.data(), static_cast<std::ptrdiff_t>(a_strides[0]),
          static_cast<std::ptrdiff_t>(a_strides[1]), b_data.data(), static_cast<std::ptrdiff_t>(b_strides[0]),
          static_cast<std::ptrdiff_t>(b_strides[1]), 0.0f, out.data(), static_cast<std::ptrdiff_t>(n), 1);

    if (bias != nullptr) {
        const auto bias_data = bias->f32_slice(); // as_f32_slice: panics unless F32
        const size_t b_len = bias_data.size();
        if (b_len != n && b_len != 1) return tl::unexpected(Error::shape_mismatch({n}, {b_len}));
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j) {
                const float bv = b_len == 1 ? bias_data[0] : bias_data[j];
                out[i * n + j] += beta * bv;
            }
    }
    return Tensor::from_f32_vec(std::move(out), Shape{m, n});
}

} // namespace sapient::backends_cpu::kernels::matmul
```

- [ ] **Step 5: Build and run**

Run: `cd cpp && cmake --build --preset dev && ctest --preset dev -R Matmul`
Expected: 11 tests pass (5 Rust names + 6 C++-only).

- [ ] **Step 6: Format, then commit**

```bash
git add cpp/libs/sapient-backends-cpu
git ls-files -- 'cpp/*.hpp' 'cpp/*.cpp' | xargs .superpowers/tools-venv/bin/clang-format -i
git add cpp/libs/sapient-backends-cpu
git commit -m "cpp(backends-cpu): kernels/matmul — float GEMV/sgemm paths, dispatcher skeleton, gemm (5 Rust tests by name)

The NEON f16 bit-surgery is reproduced verbatim, including its mis-decode of negative and
non-normal f16 values (Rust is frozen; recorded in docs/PARITY.md by the docs task).
Quantized dtypes return an explicit plan-D error; thermal/spinpool are plan-E stubs.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Dense golden cases in `dump_kernels.rs`, `sapient::testing` carry-overs, `golden_kernels_test.cpp`

**Files:**
- Modify: `crates/sapient-backends/cpu/examples/dump_kernels.rs` (import line, module doc, 12 cases appended at the END of `build_cases`, before `Ok(cases)`)
- Modify: `cpp/libs/sapient-testing/include/sapient/testing/compare.hpp`, `cpp/libs/sapient-testing/src/compare.cpp`
- Modify: `cpp/libs/sapient-testing/tests/golden_test.cpp` (+2 tests)
- Create: `cpp/libs/sapient-backends-cpu/tests/golden_kernels_test.cpp`
- Modify: `cpp/libs/sapient-backends-cpu/CMakeLists.txt` (add the test file)

**Interfaces:**
- Consumes: every kernel of Tasks 3–7 via `sapient/backends_cpu/kernels.hpp`; `sapient::testing::{bit_identical, within_abs, max_abs_err, SAPIENT_GOLDEN_CASE, GoldenCase::get, GoldenArray::as<T>}` (plan A).
- Produces: `sapient::testing::within_rel_of_max(std::span<const float> got, std::span<const float> ref, float rel) -> ::testing::AssertionResult` (spec §4: `max_abs_err ≤ rel · max(1, max|ref|)`, non-finite reference values ignored in the max); `template <class T> sapient::testing::exact_equal(std::span<const T> got, std::span<const T> ref) -> ::testing::AssertionResult` (plan D's i8/i32 activation-quantiser gates); 12 new `.sapd` cases (31 → 43): `matmul_nt_f16_m1`, `matmul_nt_f32_m1_k512`, `layer_norm`, `reduce`, `apply_rope_partial`, `apply_rope_partial_scaled`, `attention_masked`, `gelu`, `softmax_axis0`, `log_softmax`, `conv2d_s1`, `conv2d_s2`.

**Gating table** (the C++ test names are the case names; `sgemm` cases carry a `_sgemm_tolerance` suffix):

| Case | Path in Rust | Gate |
|---|---|---|
| `rms_norm`, `layer_norm`, `softmax`, `softmax_axis0`, `log_softmax`, `silu`, `gelu_erf`, `gelu`, `reduce` (4 outputs), `apply_rope`, `apply_rope_partial`, `apply_rope_partial_scaled`, `attention_prefill`, `attention_decode`, `attention_masked` | scalar / NEON, libm | **bit-identical** |
| `matmul_nt_f32_m1_k512` | `dot_f32_fast` GEMV | **bit-identical** |
| `matmul_nt_f16_m1` | `dot_f32_x_f16` GEMV (raw f16 bytes; signed values exercise the reproduced mis-decode) | **bit-identical** |
| `matmul_nt_f32_m1` (k=64 < 512 → sgemm in Rust too), `matmul_nt_f32_m4`, `conv2d_s1`, `conv2d_s2` | `matrixmultiply::sgemm` | `within_rel_of_max(…, 1e-5f)` |
| `matmul_nt_{q8_0,q4_k,q6_k}_m{1,3}` | quant arms | **plan D** (not consumed here) |

A bit-identical case that fails is **reported, never loosened**: the implementer returns `DONE_WITH_CONCERNS` with the first mismatching index/bits from `bit_identical`'s message and the case name; the controller rules (a libm difference on this host would be a spec §3.6 finding, a kernel difference a port bug).

- [ ] **Step 1: Extend `dump_kernels.rs`**

Change the import to:
```rust
use sapient_backends_cpu::kernels::{
    attention, conv2d, elementwise, layernorm, matmul, quant, reduce, rope, softmax,
};
```
In the module doc comment, after the `dequant_*` sentence, add:
```rust
//! Plan C (dense kernels): `matmul_nt_{f16_m1,f32_m1_k512}` (the two GEMV paths; F16 weights as raw
//! bytes), `layer_norm`, `reduce` (four outputs), `apply_rope_partial(_scaled)`, `attention_masked`,
//! `gelu`, `softmax_axis0`, `log_softmax`, `conv2d_s{1,2}`.
```
Insert the following immediately before `    Ok(cases)` at the end of `build_cases` (appending keeps every earlier case's RNG draws unchanged):

```rust
    // ── plan C: dense kernels (sub-project 1a) ──────────────────────────────────────────────────
    // matmul_nt float GEMV paths. F16 weights are dumped as RAW bytes (`in:w_f16`): the aarch64
    // path widens f16 by NEON bit-surgery that differs from `half` for non-normal AND negative
    // values, and the C++ port must reproduce that, so the dump feeds real (signed) bits.
    // m=1, k>=64 (F16) → dot_f32_x_f16; m=1, k>=512 (F32) → dot_f32_fast.
    {
        let (k, n) = (64usize, 8usize);
        let w_src = rng.f32s(n * k, -1.0, 1.0);
        let w_f16: Vec<u8> = w_src
            .iter()
            .flat_map(|v| f16::from_f32(*v).to_le_bytes())
            .collect();
        let wt = Tensor::from_f16_bytes(&w_f16, vec![n, k])?;
        let xt = Tensor::from_f32(&rng.f32s(k, -1.0, 1.0), vec![1, k])?;
        let y = matmul::matmul_nt(&xt, &wt)?;
        cases.push(case(
            "matmul_nt_f16_m1",
            vec![
                Array::tensor("in:x", &xt),
                Array::u8("in:w_f16", &[w_f16.len()], &w_f16),
                Array::u32("param:w_shape", &[2], &[n as u32, k as u32]),
                Array::tensor("out:y", &y),
            ],
        ));
        let (k, n) = (512usize, 8usize);
        let wt = Tensor::from_f32(&rng.f32s(n * k, -1.0, 1.0), vec![n, k])?;
        let xt = Tensor::from_f32(&rng.f32s(k, -1.0, 1.0), vec![1, k])?;
        let y = matmul::matmul_nt(&xt, &wt)?;
        cases.push(case(
            "matmul_nt_f32_m1_k512",
            vec![
                Array::tensor("in:x", &xt),
                Array::tensor("in:w", &wt),
                Array::tensor("out:y", &y),
            ],
        ));
    }
    // layer_norm over the last axis with weight and bias.
    {
        let xt = Tensor::from_f32(&rng.f32s(3 * 32, -2.0, 2.0), vec![3, 32])?;
        let wt = Tensor::from_f32(&rng.f32s(32, 0.5, 1.5), vec![32])?;
        let bt = Tensor::from_f32(&rng.f32s(32, -0.5, 0.5), vec![32])?;
        let y = layernorm::layer_norm(&xt, Some(&wt), Some(&bt), -1, 1e-5)?;
        cases.push(case(
            "layer_norm",
            vec![
                Array::tensor("in:x", &xt),
                Array::tensor("in:weight", &wt),
                Array::tensor("in:bias", &bt),
                Array::f32("param:eps", &[1], &[1e-5]),
                Array::tensor("out:y", &y),
            ],
        ));
    }
    // reduce_{sum,mean,max,min} over a [2, 3, 4] tensor — one case, four outputs (mean over all
    // axes is a scalar: dims = []).
    {
        let xt = Tensor::from_f32(&rng.f32s(2 * 3 * 4, -3.0, 3.0), vec![2, 3, 4])?;
        cases.push(case(
            "reduce",
            vec![
                Array::tensor("in:x", &xt),
                Array::tensor("out:sum_axis1", &reduce::reduce_sum(&xt, &[1], false)?),
                Array::tensor("out:mean_all", &reduce::reduce_mean(&xt, &[], false)?),
                Array::tensor("out:max_axis0_keep", &reduce::reduce_max(&xt, &[0], true)?),
                Array::tensor("out:min_axis_neg1", &reduce::reduce_min(&xt, &[-1], false)?),
            ],
        ));
    }
    // Partial RoPE (Phi: rotary_dim < head_dim) and scaled partial RoPE (Gemma3: pos_scale 8).
    {
        let xt = Tensor::from_f32(&rng.f32s(2 * 3 * 16, -1.0, 1.0), vec![1, 2, 3, 16])?;
        let positions: Vec<usize> = vec![7, 8, 9];
        let pos_u64: Vec<u64> = positions.iter().map(|&p| p as u64).collect();
        let y = rope::apply_rope_partial(&xt, &positions, 10_000.0, 8)?;
        cases.push(case(
            "apply_rope_partial",
            vec![
                Array::tensor("in:x", &xt),
                Array::u64("param:positions", &[3], &pos_u64),
                Array::f32("param:base", &[1], &[10_000.0]),
                Array::u32("param:rotary_dim", &[1], &[8]),
                Array::tensor("out:y", &y),
            ],
        ));
        let y = rope::apply_rope_partial_scaled(&xt, &positions, 1_000_000.0, 16, 8.0)?;
        cases.push(case(
            "apply_rope_partial_scaled",
            vec![
                Array::tensor("in:x", &xt),
                Array::u64("param:positions", &[3], &pos_u64),
                Array::f32("param:base", &[1], &[1_000_000.0]),
                Array::u32("param:rotary_dim", &[1], &[16]),
                Array::f32("param:pos_scale", &[1], &[8.0]),
                Array::tensor("out:y", &y),
            ],
        ));
    }
    // Attention with an explicit additive mask (sliding window of 3, seq_q=2 over seq_k=6, GQA
    // 4/2, explicit scale) — exercises the mask branch, the -inf skip and the explicit scale.
    {
        let (seq_q, seq_k) = (2usize, 6usize);
        let q = Tensor::from_f32(&rng.f32s(4 * seq_q * hd, -1.0, 1.0), vec![1, 4, seq_q, hd])?;
        let k = Tensor::from_f32(&rng.f32s(2 * seq_k * hd, -1.0, 1.0), vec![1, 2, seq_k, hd])?;
        let v = Tensor::from_f32(&rng.f32s(2 * seq_k * hd, -1.0, 1.0), vec![1, 2, seq_k, hd])?;
        let mut m = vec![0.0f32; seq_q * seq_k];
        for qi in 0..seq_q {
            let pos = qi + (seq_k - seq_q);
            for ki in 0..seq_k {
                if ki > pos || ki + 3 <= pos {
                    m[qi * seq_k + ki] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::from_f32(&m, vec![seq_q, seq_k])?;
        let y = attention::scaled_dot_product_attention(&q, &k, &v, Some(&mask), Some(0.125), 2)?;
        cases.push(case(
            "attention_masked",
            vec![
                Array::tensor("in:q", &q),
                Array::tensor("in:k", &k),
                Array::tensor("in:v", &v),
                Array::tensor("in:mask", &mask),
                Array::f32("param:scale", &[1], &[0.125]),
                Array::u32("param:n_kv_heads", &[1], &[2]),
                Array::tensor("out:y", &y),
            ],
        ));
    }
    // gelu (tanh approximation), softmax over axis 0, log_softmax over the last axis.
    {
        let xt = Tensor::from_f32(&rng.f32s(64, -6.0, 6.0), vec![64])?;
        cases.push(case(
            "gelu",
            vec![
                Array::tensor("in:x", &xt),
                Array::tensor("out:y", &elementwise::gelu(&xt)?),
            ],
        ));
        let xt = Tensor::from_f32(&rng.f32s(3 * 5, -4.0, 4.0), vec![3, 5])?;
        cases.push(case(
            "softmax_axis0",
            vec![
                Array::tensor("in:x", &xt),
                Array::i32("param:axis", &[1], &[0]),
                Array::tensor("out:y", &softmax::softmax(&xt, 0)?),
            ],
        ));
        cases.push(case(
            "log_softmax",
            vec![
                Array::tensor("in:x", &xt),
                Array::i32("param:axis", &[1], &[-1]),
                Array::tensor("out:y", &softmax::log_softmax(&xt, -1)?),
            ],
        ));
    }
    // conv2d (im2col + sgemm → max-error gated): stride 1 with padding, stride 2 without;
    // groups 1, bias.
    {
        let xt = Tensor::from_f32(&rng.f32s(3 * 7 * 7, -1.0, 1.0), vec![1, 3, 7, 7])?;
        let wt = Tensor::from_f32(&rng.f32s(4 * 3 * 3 * 3, -1.0, 1.0), vec![4, 3, 3, 3])?;
        let bt = Tensor::from_f32(&rng.f32s(4, -0.5, 0.5), vec![4])?;
        let conv = |pads: [usize; 4],
                    strides: [usize; 2]|
         -> Result<Vec<Array>, Box<dyn Error>> {
            let y = conv2d::conv2d(&xt, &wt, Some(&bt), [3, 3], pads, strides, [1, 1], 1)?;
            Ok(vec![
                Array::tensor("in:x", &xt),
                Array::tensor("in:w", &wt),
                Array::tensor("in:bias", &bt),
                Array::u32("param:pads", &[4], &pads.map(|p| p as u32)),
                Array::u32("param:strides", &[2], &strides.map(|s| s as u32)),
                Array::tensor("out:y", &y),
            ])
        };
        cases.push(case("conv2d_s1", conv([1, 1, 1, 1], [1, 1])?));
        cases.push(case("conv2d_s2", conv([0, 0, 0, 0], [2, 2])?));
    }
```

Then:

```bash
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings \
  && cpp/tests/parity/golden_dump.sh /tmp/sapient-golden \
  && cpp/tests/parity/golden_dump.sh /tmp/sapient-golden-2 && diff -r /tmp/sapient-golden /tmp/sapient-golden-2 && echo DETERMINISTIC
```
Expected: fmt/clippy clean; `golden_dump: 43 cases`; `DETERMINISTIC` (same seed → byte-identical dumps, the PARITY.md determinism row stays true).

- [ ] **Step 2: `sapient::testing` carry-overs** — write the two failing tests first (append to `cpp/libs/sapient-testing/tests/golden_test.cpp`):

```cpp
TEST(Compare, within_rel_of_max_scales_tolerance_by_reference_magnitude) {
    using sapient::testing::within_rel_of_max;
    const float ref[] = {100.0f, -200.0f, 0.5f};
    const float got[] = {100.001f, -200.001f, 0.5005f}; // abs err 1e-3; max|ref| = 200 → tol 2e-3 at 1e-5
    EXPECT_TRUE(within_rel_of_max(got, ref, 1e-5f));
    EXPECT_FALSE(within_rel_of_max(got, ref, 1e-7f));
    const float small_ref[] = {0.001f};
    const float small_got[] = {0.001f + 5e-6f}; // the max(1, ·) floor: tol = rel · 1
    EXPECT_TRUE(within_rel_of_max(small_got, small_ref, 1e-5f));
    EXPECT_FALSE(within_rel_of_max(small_got, small_ref, 1e-6f));
}

TEST(Compare, exact_equal_reports_first_mismatch) {
    using sapient::testing::exact_equal;
    const int8_t a[] = {-128, 0, 127};
    const int8_t b[] = {-128, 1, 127};
    EXPECT_TRUE(exact_equal<int8_t>(a, a));
    const auto r = exact_equal<int8_t>(a, b);
    EXPECT_FALSE(r);
    EXPECT_NE(std::string(r.message()).find("index 1: got 0, ref 1"), std::string::npos) << r.message();
    const uint32_t c[] = {1, 2};
    const uint32_t d[] = {1};
    EXPECT_FALSE(exact_equal<uint32_t>(c, d));
}
```
(add `#include <cstdint>` and `#include <string>` to that file if missing.) Build: FAILS — `within_rel_of_max`/`exact_equal` undeclared.

Then add to `compare.hpp`, after `within_abs`'s declaration:

```cpp
/// Spec §4 tolerance for the sgemm-backed cases: `max_abs_err(got, ref) <= rel * max(1, max|ref|)`
/// (non-finite reference values are ignored when taking the max). Same length rule and NaN
/// reporting as `within_abs`.
::testing::AssertionResult
within_rel_of_max(std::span<const float> got, std::span<const float> ref, float rel);

/// Exact element equality for integer golden arrays (i8/i32/u8/u32/u64 — plan D's activation
/// quantisers); reports the first mismatching index and both values.
template <class T>
::testing::AssertionResult exact_equal(std::span<const T> got, std::span<const T> ref) {
    if (got.size() != ref.size())
        return ::testing::AssertionFailure() << "length " << got.size() << " != " << ref.size();
    for (size_t i = 0; i < got.size(); ++i)
        if (got[i] != ref[i])
            return ::testing::AssertionFailure()
                   << "index " << i << ": got " << +got[i] << ", ref " << +ref[i];
    return ::testing::AssertionSuccess();
}
```
(`+got[i]` promotes `int8_t`/`uint8_t` to `int` so they print as numbers.) Also close plan A's parked residual: above `#define SAPIENT_GOLDEN_CASE(var, name)` add the line
```cpp
// `__COUNTER__` is expanded twice per invocation — once per generated identifier — and each
// expansion yields a fresh value, which is exactly what keeps the two names distinct.
```
And in `compare.cpp`, after `within_abs`:

```cpp
::testing::AssertionResult
within_rel_of_max(std::span<const float> got, std::span<const float> ref, float rel) {
    float max_ref = 1.0f;
    for (const float r : ref)
        if (std::isfinite(r) && std::fabs(r) > max_ref) max_ref = std::fabs(r);
    return within_abs(got, ref, rel * max_ref);
}
```
Build and run: `cd cpp && cmake --build --preset dev && ctest --preset dev -R Compare` → 6 pass (4 existing + 2).

- [ ] **Step 3: `golden_kernels_test.cpp`** — add `tests/golden_kernels_test.cpp` to the backends-cpu test target, then write:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
// Plan C gate: the dense kernels vs the Rust oracle's dumps (SAPIENT_GOLDEN_DIR; unset → SKIP,
// set-but-missing → FAIL). Bit-identical everywhere except the sgemm-backed cases — matmul_nt_f32_m1
// (k=64 < 512 takes sgemm in Rust too), matmul_nt_f32_m4, conv2d_s1/s2 — which use the spec §4
// tolerance 1e-5·max(1, max|ref|). The six matmul_nt_q*_m{1,3} dumps are plan D's.
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "sapient/backends_cpu/kernels.hpp"
#include "sapient/core/tensor.hpp"
#include "sapient/testing/compare.hpp"

using namespace sapient::backends_cpu::kernels;
using sapient::core::Shape;
using sapient::core::Tensor;
using sapient::testing::bit_identical;
using sapient::testing::GoldenCase;
using sapient::testing::within_rel_of_max;

namespace {

constexpr float SGEMM_REL = 1e-5f; // spec §4

Tensor tensor_of(const GoldenCase& c, std::string_view name) {
    const auto& a = c.get(name);
    std::vector<size_t> dims(a.dims.begin(), a.dims.end());
    auto t = Tensor::from_f32_vec(a.as<float>(), Shape(dims));
    if (!t) throw std::runtime_error(std::string(name) + ": " + t.error().to_string());
    return std::move(*t);
}
std::vector<float> ref(const GoldenCase& c, std::string_view name = "out:y") {
    return c.get(name).as<float>();
}
template <class T> T param(const GoldenCase& c, std::string_view name) {
    return c.get(name).as<T>().at(0);
}
std::vector<size_t> positions_of(const GoldenCase& c) {
    const auto p = c.get("param:positions").as<uint64_t>();
    return {p.begin(), p.end()};
}

} // namespace

// ── norms / activations / softmax ────────────────────────────────────────────

TEST(GoldenKernels, rms_norm) {
    SAPIENT_GOLDEN_CASE(c, "rms_norm");
    const auto x = tensor_of(c, "in:x");
    const auto w = tensor_of(c, "in:weight");
    auto y = layernorm::rms_norm(x, &w, param<float>(c, "param:eps"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

TEST(GoldenKernels, layer_norm) {
    SAPIENT_GOLDEN_CASE(c, "layer_norm");
    const auto x = tensor_of(c, "in:x");
    const auto w = tensor_of(c, "in:weight");
    const auto b = tensor_of(c, "in:bias");
    auto y = layernorm::layer_norm(x, &w, &b, -1, param<float>(c, "param:eps"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

TEST(GoldenKernels, softmax) {
    SAPIENT_GOLDEN_CASE(c, "softmax");
    auto y = softmax::softmax(tensor_of(c, "in:x"), param<int32_t>(c, "param:axis"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

TEST(GoldenKernels, softmax_axis0) {
    SAPIENT_GOLDEN_CASE(c, "softmax_axis0");
    auto y = softmax::softmax(tensor_of(c, "in:x"), param<int32_t>(c, "param:axis"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

TEST(GoldenKernels, log_softmax) {
    SAPIENT_GOLDEN_CASE(c, "log_softmax");
    auto y = softmax::log_softmax(tensor_of(c, "in:x"), param<int32_t>(c, "param:axis"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

TEST(GoldenKernels, silu) {
    SAPIENT_GOLDEN_CASE(c, "silu");
    auto y = elementwise::silu(tensor_of(c, "in:x"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

TEST(GoldenKernels, gelu_erf) {
    SAPIENT_GOLDEN_CASE(c, "gelu_erf");
    auto y = elementwise::gelu_erf(tensor_of(c, "in:x"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

TEST(GoldenKernels, gelu) {
    SAPIENT_GOLDEN_CASE(c, "gelu");
    auto y = elementwise::gelu(tensor_of(c, "in:x"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

TEST(GoldenKernels, reduce) {
    SAPIENT_GOLDEN_CASE(c, "reduce");
    const auto x = tensor_of(c, "in:x");
    const int64_t ax1[] = {1};
    const int64_t ax0[] = {0};
    const int64_t axn1[] = {-1};
    auto s = reduce::reduce_sum(x, ax1, false);
    ASSERT_TRUE(s.has_value()) << s.error().to_string();
    EXPECT_TRUE(bit_identical(s->to_f32_vec(), ref(c, "out:sum_axis1")));
    auto m = reduce::reduce_mean(x, {}, false);
    ASSERT_TRUE(m.has_value()) << m.error().to_string();
    EXPECT_TRUE(m->shape().dims.empty()) << "all-axes mean is a scalar";
    EXPECT_TRUE(bit_identical(m->to_f32_vec(), ref(c, "out:mean_all")));
    auto mx = reduce::reduce_max(x, ax0, true);
    ASSERT_TRUE(mx.has_value()) << mx.error().to_string();
    EXPECT_EQ(mx->shape().dims, (std::vector<size_t>{1, 3, 4}));
    EXPECT_TRUE(bit_identical(mx->to_f32_vec(), ref(c, "out:max_axis0_keep")));
    auto mn = reduce::reduce_min(x, axn1, false);
    ASSERT_TRUE(mn.has_value()) << mn.error().to_string();
    EXPECT_TRUE(bit_identical(mn->to_f32_vec(), ref(c, "out:min_axis_neg1")));
}

// ── RoPE (powf/sinf/cosf — the libm-portability risk, spec §7) ───────────────

TEST(GoldenKernels, apply_rope) {
    SAPIENT_GOLDEN_CASE(c, "apply_rope");
    auto y = rope::apply_rope(tensor_of(c, "in:x"), positions_of(c), param<float>(c, "param:base"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

TEST(GoldenKernels, apply_rope_partial) {
    SAPIENT_GOLDEN_CASE(c, "apply_rope_partial");
    auto y = rope::apply_rope_partial(tensor_of(c, "in:x"), positions_of(c), param<float>(c, "param:base"),
                                      param<uint32_t>(c, "param:rotary_dim"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

TEST(GoldenKernels, apply_rope_partial_scaled) {
    SAPIENT_GOLDEN_CASE(c, "apply_rope_partial_scaled");
    auto y = rope::apply_rope_partial_scaled(tensor_of(c, "in:x"), positions_of(c), param<float>(c, "param:base"),
                                             param<uint32_t>(c, "param:rotary_dim"),
                                             param<float>(c, "param:pos_scale"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

// ── attention ────────────────────────────────────────────────────────────────

TEST(GoldenKernels, attention_prefill) {
    SAPIENT_GOLDEN_CASE(c, "attention_prefill");
    auto y = attention::scaled_dot_product_attention(tensor_of(c, "in:q"), tensor_of(c, "in:k"), tensor_of(c, "in:v"),
                                                     nullptr, std::nullopt, param<uint32_t>(c, "param:n_kv_heads"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

TEST(GoldenKernels, attention_decode) {
    SAPIENT_GOLDEN_CASE(c, "attention_decode");
    auto y = attention::scaled_dot_product_attention(tensor_of(c, "in:q"), tensor_of(c, "in:k"), tensor_of(c, "in:v"),
                                                     nullptr, std::nullopt, param<uint32_t>(c, "param:n_kv_heads"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

TEST(GoldenKernels, attention_masked) {
    SAPIENT_GOLDEN_CASE(c, "attention_masked");
    const auto mask = tensor_of(c, "in:mask");
    auto y = attention::scaled_dot_product_attention(tensor_of(c, "in:q"), tensor_of(c, "in:k"), tensor_of(c, "in:v"),
                                                     &mask, param<float>(c, "param:scale"),
                                                     param<uint32_t>(c, "param:n_kv_heads"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

// ── matmul_nt float paths ────────────────────────────────────────────────────

TEST(GoldenKernels, matmul_nt_f32_m1_k512) { // dot_f32_fast GEMV: bit-identical
    SAPIENT_GOLDEN_CASE(c, "matmul_nt_f32_m1_k512");
    auto y = matmul::matmul_nt(tensor_of(c, "in:x"), tensor_of(c, "in:w"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

TEST(GoldenKernels, matmul_nt_f16_m1) { // dot_f32_x_f16 GEMV incl. the reproduced mis-decode: bit-identical
    SAPIENT_GOLDEN_CASE(c, "matmul_nt_f16_m1");
    const auto shape = c.get("param:w_shape").as<uint32_t>();
    auto w = Tensor::from_f16_bytes(c.get("in:w_f16").as<uint8_t>(), Shape({shape[0], shape[1]}));
    ASSERT_TRUE(w.has_value()) << w.error().to_string();
    auto y = matmul::matmul_nt(tensor_of(c, "in:x"), *w);
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

TEST(GoldenKernels, matmul_nt_f32_m1_sgemm_tolerance) { // k=64 < 512: sgemm in Rust too
    SAPIENT_GOLDEN_CASE(c, "matmul_nt_f32_m1");
    auto y = matmul::matmul_nt(tensor_of(c, "in:x"), tensor_of(c, "in:w"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(within_rel_of_max(y->to_f32_vec(), ref(c), SGEMM_REL));
}

TEST(GoldenKernels, matmul_nt_f32_m4_sgemm_tolerance) {
    SAPIENT_GOLDEN_CASE(c, "matmul_nt_f32_m4");
    auto y = matmul::matmul_nt(tensor_of(c, "in:x"), tensor_of(c, "in:w"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(within_rel_of_max(y->to_f32_vec(), ref(c), SGEMM_REL));
}

// ── conv2d (sgemm-backed) ────────────────────────────────────────────────────

namespace {
void check_conv(const char* name) {
    SAPIENT_GOLDEN_CASE(c, name);
    const auto b = tensor_of(c, "in:bias");
    const auto p = c.get("param:pads").as<uint32_t>();
    const auto s = c.get("param:strides").as<uint32_t>();
    auto y = conv2d::conv2d(tensor_of(c, "in:x"), tensor_of(c, "in:w"), &b, {3, 3}, {p[0], p[1], p[2], p[3]},
                            {s[0], s[1]}, {1, 1}, 1);
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(within_rel_of_max(y->to_f32_vec(), ref(c), SGEMM_REL));
}
} // namespace

TEST(GoldenKernels, conv2d_s1_sgemm_tolerance) { check_conv("conv2d_s1"); }
TEST(GoldenKernels, conv2d_s2_sgemm_tolerance) { check_conv("conv2d_s2"); }
```

(`SAPIENT_GOLDEN_CASE` inside `check_conv` works because the helper returns `void`; a SKIP/FAIL inside it ends that test.)

- [ ] **Step 4: Build and run — both ways**

```bash
cd cpp && cmake --build --preset dev \
  && ctest --preset dev -R 'GoldenKernels' \
  && SAPIENT_GOLDEN_DIR=/tmp/sapient-golden ctest --preset dev -R 'GoldenKernels'
```
Expected: first run — 21 tests, all **skipped** ("SAPIENT_GOLDEN_DIR not set"); second run — 21 **passed**. If any bit-identical case fails, stop and report per the gating table (do not change tolerances). Record the max-error each `_sgemm_tolerance` case actually measured (temporarily print `max_abs_err` or read it from a deliberately failing tolerance) in the report — Task 9 puts the numbers in `docs/PARITY.md`.

Also run the full suite both ways and record the counts: `ctest --preset dev` and `SAPIENT_GOLDEN_DIR=/tmp/sapient-golden ctest --preset dev`.

- [ ] **Step 5: Format, then commit (two commits)**

```bash
cargo fmt --all
git add crates/sapient-backends/cpu/examples/dump_kernels.rs cpp/libs/sapient-testing
git ls-files -- 'cpp/*.hpp' 'cpp/*.cpp' | xargs .superpowers/tools-venv/bin/clang-format -i
git add cpp/libs/sapient-testing
git commit -m "oracle+testing: 12 dense dump cases (43 total), within_rel_of_max, exact_equal<T>

Test-only Rust (dump_kernels.rs); the Rust tree's behaviour is unchanged.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"

git add cpp/libs/sapient-backends-cpu
git ls-files -- 'cpp/*.hpp' 'cpp/*.cpp' | xargs .superpowers/tools-venv/bin/clang-format -i
git add cpp/libs/sapient-backends-cpu
git commit -m "cpp(backends-cpu): golden_kernels_test — 17 dense cases bit-identical, 4 sgemm cases within 1e-5·max(1,|ref|)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Docs, parity ledger, `just cpp-tidy`, final verification

**Files:**
- Modify: `CLAUDE.md`, `docs/ROADMAP.md`, `docs/PARITY.md`, `docs/PROJECT_GUIDE.md`, `CHANGELOG.md`, `justfile`

- [ ] **Step 1: CLAUDE.md** — in the `## C++ rewrite programme` section: (a) in the plan-A bullet replace the trailing `Next: plan C (dense kernels).` with `Plan C followed (next bullet).`; (b) append this bullet after it:

```markdown
- **Sub-project 1a, plan C landed (`sapient::backends_cpu` dense kernels):** `cpp/libs/sapient-backends-cpu/` ports `kernels/{elementwise,softmax,reduce,layernorm,rope,attention,conv2d}` and the float paths + dtype dispatcher of `kernels/matmul` 1:1 (26 + 5 Rust tests by name; namespaces mirror the Rust module paths, e.g. `kernels::attention::scaled_dot_product_attention`). Three stand-ins are hand-written: **`parallel`** (a persistent pool reproducing rayon's `par_chunks_mut` chunk→range partition — parity depends only on the partition, never on which thread runs a chunk; `num_threads()` follows rayon's `RAYON_NUM_THREADS` rules), **`cpu_features`** (`has_dotprod/has_i8mm/has_avx2_fma`, cached `is_*_feature_detected!` twins) and **`sgemm`** (packed, cache-blocked, 4-wide FMA; max-error gated vs `matrixmultiply`, but bit-independent of the callers' thread-count row blocking so results never vary with `RAYON_NUM_THREADS`). `thermal`/`spinpool` are inert stubs with plan E's signatures; `matmul_nt`'s seven quantized arms return an explicit plan-D error. Bit-identity rules that bit: Rust's float `Sum` seeds at `-0.0` (every ported `.iter().sum()` starts at `-0.0f`); `-fno-math-errno` is a parity flag (rustc lowers libm to LLVM intrinsics; Darwin defaulted to it, Linux did not); libm calls are the `f`-suffixed C names, never double; inside a kernel namespace the module's own `exp/log/abs/…(const Tensor&)` hide the unqualified libm names. **Known oracle defect reproduced on purpose:** Rust's NEON `dot_f32_x_f16_neon` (`matmul_nt` at m=1, k≥64 with F16 weights) mis-decodes negative f16 values (the sign bit leaks into the exponent field, ×2^32) — dormant because F16 linears are online-quantised to Q8_0 at load and Rust's own test uses k=2; ported verbatim, pinned by the `matmul_nt_f16_m1` golden case, recorded in `docs/PARITY.md`. Gate: 43 golden cases — 17 dense ones bit-identical, 4 sgemm-backed within `1e-5·max(1,|ref|)` (`sapient::testing::within_rel_of_max`; `exact_equal<T>` added for plan D). `just cpp-tidy` runs the CI-pinned clang-tidy (CI-only on this Mac: LLVM-18 tidy cannot parse current macOS libc++). Next: plan D (quant kernels).
```

- [ ] **Step 2: `docs/ROADMAP.md`** — Phase 7 table row 1a status → `in progress — plans A (core) and C (dense kernels) implemented on feat/cpp-sp1a; plans D, E, B pending`.

- [ ] **Step 3: `docs/PARITY.md`** — three edits under "Sub-project 1a":

(a) Replace gap (a) with:
```markdown
- (a) The golden dumps cover the 22 public entry points plus, as of plan A, the R4 **dequant** path
  (`dequant_q4_k_r4`/`dequant_q6_k_r4`) and, as of plan C, `apply_rope_partial(_scaled)`, the
  masked-attention branch, the two float GEMV paths (`matmul_nt_f16_m1`, `matmul_nt_f32_m1_k512`),
  `layer_norm`, `reduce`, `gelu`, `softmax` axis 0, `log_softmax` and `conv2d` (43 cases) — the R4
  **matmul** cases (SDOT/SMMLA/Q8_K activation formats) and the `quantize_row_to_i8_blocks` /
  `quantize_row_to_q8k` activation formats remain open for plan D.
```

(b) Insert a new subsection between "Open gaps carried from sub-project 0" and "### Results":
```markdown
### Known oracle defects the port reproduces on purpose

| Found | Where (Rust) | Defect | Port | Status |
|---|---|---|---|---|
| 2026-09-21 (plan C) | `crates/sapient-backends/cpu/src/kernels/matmul.rs` `dot_f32_x_f16_neon` (aarch64; reached by `matmul_nt` at m=1, k≥64 with F16 weights) | The NEON f16→f32 bit-surgery shifts the unmasked u16 right by 10, so the f16 sign bit lands in bit 5 of the exponent field: negative weights decode ×2^32 (standalone probe: `[1,1,1,1]·[-1,1,-2,0.5]` = −1.29e10 instead of −1.5); subnormal/inf/NaN also decode differently from the scalar tail. Dormant in production because F16 linears are online-quantised to Q8_0 at load and Rust's own unit test uses k=2 (< 64). | Reproduced verbatim (spec §3.5); pinned bit-exactly by `matmul_nt_f16_m1` (random signed weights) | Open — Rust is frozen during the port; fix both trees together after parity, or document as a limitation. The user decides. |
```

(c) Append to the `### Results` table (fill the hashes with `git rev-parse --short` of the `dump_kernels` commit and of HEAD, and the measured sgemm max errors from Task 8's report):
```markdown
| 2026-09-21 | 26 dense + 5 float-matmul backends-cpu unit tests ported by name (+ C++-only `parallel`/`sgemm`/`cpu_features`/error-text tests) | macOS arm64 (Apple M5; NEON) | — | `<cpp sha>` | pass |
| 2026-09-21 | Dense golden cases `rms_norm`, `layer_norm`, `softmax`, `softmax_axis0`, `log_softmax`, `silu`, `gelu_erf`, `gelu`, `reduce` (4 outputs), `apply_rope`, `apply_rope_partial`, `apply_rope_partial_scaled`, `attention_prefill`, `attention_decode`, `attention_masked`, `matmul_nt_f32_m1_k512`, `matmul_nt_f16_m1` vs the Rust kernels | macOS arm64 (Apple M5; NEON) | `<rust sha>` | `<cpp sha>` | bit-identical, 17/17 |
| 2026-09-21 | sgemm-backed cases vs `matrixmultiply`: `matmul_nt_f32_m1`, `matmul_nt_f32_m4`, `conv2d_s1`, `conv2d_s2` | macOS arm64 | `<rust sha>` | `<cpp sha>` | within 1e-5·max(1, max\|ref\|); measured max_err `<values>` |
```

- [ ] **Step 4: `docs/PROJECT_GUIDE.md`** — in the "### The C++ tree (in progress)" paragraph, replace `plus \`sapient::testing\`'s compare helpers — the golden-dump gates on it are bit-identical to the Rust oracle.` with `plus \`sapient::testing\`'s compare helpers, and plan C added \`sapient::backends_cpu\`'s dense kernels (\`cpp/libs/sapient-backends-cpu/\`: attention, RoPE, norms, softmax, reductions, conv2d and the float matmul paths, with a rayon stand-in and an own SGEMM) — the golden-dump gates on both are bit-identical to the Rust oracle, except the SGEMM-backed paths, which are held within a tolerance.`

- [ ] **Step 5: `CHANGELOG.md`** — under `## [Unreleased]`, after the plan-A section, add:
```markdown
### 🧱 C++ rewrite — sub-project 1a, plan C (sapient::backends_cpu dense kernels)
- `cpp/libs/sapient-backends-cpu`: attention, RoPE, LayerNorm/RMSNorm, softmax, reductions, element-wise ops, conv2d and the float matmul paths ported 1:1 with all 31 Rust tests; rayon stand-in (`parallel`), ISA probes, own SGEMM; `-fno-math-errno` joins the parity flags; 12 new golden cases (43 total) — 17 dense cases bit-identical, 4 SGEMM-backed within tolerance.
- Test-only Rust: `dump_kernels` gains the 12 dense cases. No product behaviour change. Recorded in `docs/PARITY.md`: a dormant defect in the Rust NEON F16 GEMV (negative f16 weights mis-decoded ×2^32), which the port reproduces on purpose.
```

- [ ] **Step 6: `justfile`** — after the `cpp-fmt` recipe add:

```make
# clang-tidy over the C++ sources with the CI-pinned checks (.clang-tidy, WarningsAsErrors: '*').
# Needs a configured build dir for compile_commands.json (`just cpp-configure`). Resolves the binary
# like cpp-fmt: $SAPIENT_CLANG_TIDY -> .superpowers/tools-venv/bin/clang-tidy -> clang-tidy-18 -> clang-tidy.
# NOTE: LLVM-18 clang-tidy cannot parse current macOS libc++ headers — on a Mac this gate is CI-only.
cpp-tidy preset="dev":
    CT="${SAPIENT_CLANG_TIDY:-}"; \
    if [ -z "$CT" ] && [ -x ".superpowers/tools-venv/bin/clang-tidy" ]; then CT=".superpowers/tools-venv/bin/clang-tidy"; fi; \
    if [ -z "$CT" ] && command -v clang-tidy-18 >/dev/null 2>&1; then CT="clang-tidy-18"; fi; \
    if [ -z "$CT" ] && command -v clang-tidy >/dev/null 2>&1; then CT="clang-tidy"; fi; \
    if [ -z "$CT" ]; then echo "cpp-tidy: no clang-tidy found (checked \$SAPIENT_CLANG_TIDY, .superpowers/tools-venv/bin/clang-tidy, clang-tidy-18, clang-tidy on PATH)" >&2; exit 1; fi; \
    [ -f "cpp/build/{{preset}}/compile_commands.json" ] || { echo "cpp-tidy: cpp/build/{{preset}}/compile_commands.json missing — run: just cpp-configure {{preset}}" >&2; exit 1; }; \
    case "$("$CT" --version)" in *" 18."*) ;; *) echo "cpp-tidy: warning — $CT reports $("$CT" --version | head -1); CI pins clang-tidy 18" >&2 ;; esac; \
    git ls-files -- 'cpp/libs/*.cpp' | xargs "$CT" -p cpp/build/{{preset}} -quiet
```
Check it parses: `just --list | grep cpp-tidy`. Running it on this Mac is expected to fail in libc++ headers (record that in the report; do not chase it).

- [ ] **Step 7: Final verification (record every summary line in the report)**

```bash
just cpp-lint && cd cpp && cmake --preset dev && cmake --build --preset dev \
  && ./tests/parity/golden_dump.sh /tmp/sapient-golden \
  && ctest --preset dev -N | tail -1 \
  && ctest --preset dev && SAPIENT_GOLDEN_DIR=/tmp/sapient-golden ctest --preset dev && cd .. \
  && cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings \
  && cargo test -p sapient-backends-cpu -- --test-threads=1 2>&1 | grep "^test result"
```
Expected, **derived** (report the actual numbers; a mismatch is reported in the hand-back, never "fixed" by editing tests): lint OK; `golden_dump: 43 cases`; `ctest -N` total = 79 (end of plan A) + 79 new = **158** (Task 1: 7, Task 2: 6, Task 3: 8, Task 4: 13, Task 5: 8, Task 6: 3, Task 7: 11, Task 8: 2 + 21); without dumps **127 passed / 31 skipped** (plan A's 10 + this plan's 21 golden tests); with dumps **157 passed / 1 skipped** (`Compare.golden_case_macro_skips_without_env`, by design); `cargo fmt`/`clippy` clean; Rust backends-cpu tests unchanged (`71 passed; 0 failed; 1 ignored`).

- [ ] **Step 8: Commit**

```bash
git add CLAUDE.md docs/ROADMAP.md docs/PARITY.md docs/PROJECT_GUIDE.md CHANGELOG.md justfile
git commit -m "docs(sp1a-C): dense kernels landed — CLAUDE.md, roadmap, parity ledger (+ known oracle defect), just cpp-tidy

CONTRIBUTING and README need no change for a library-internal plan.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

**Do NOT push.** The user pushes `feat/cpp-sp1a` (stacked on `feat/cpp-sp0-scaffold`).

---

## Self-review against the spec

- **§2.3 file table (plan C rows):** `cpu_features` (T1: sysctl / `getauxval` / `IsProcessorFeaturePresent` / cpuid+xgetbv, cached), `parallel` (T1: `RAYON_NUM_THREADS` rules, `par_chunks_mut` partition, `par_for`), `sgemm` (T2: same signature, packed 4-wide FMA strip, max-error gated + blocking-independent), `matmul` float paths + dispatcher skeleton (T7: F16 GEMV incl. the verbatim bit-surgery, k≥512 GEMV, sgemm row blocks with the `mblock` rule, `gemv_chunk` governed branch vs the rayon-domain count, `for_each_out_chunk` rayon branch, `gemm` with bias, quant arms stubbed), `attention` (T5: `dot_f32_neon`, `saxpby_neon`, `flash_attn_row` with the `-inf` `continue` and `1/EPSILON`, rank-4 guard, `kv_rep`, `mask=None ⇒ causal` with `kv_offset`, `causal_mask`, parallel over (batch, head)), `rope` (T4: three entry points + cache, `powf`/`sinf`/`cosf`), `elementwise` (T4→T3: scalar-broadcast rule, all activations, A&S `erf_approx`), `softmax`/`reduce`/`layernorm` (T4: sequential f32 sums in Rust order), `conv2d` (T6: parallel im2col, out-channel-block sgemm, two atomic counters). `thermal`/`spinpool` stubs with plan E's signatures (T1) — plan E rows.
- **§3 rules:** flags (T1 adds `-fno-math-errno`, a recorded ruling); ISA variants mirrored, no upgrades (x86: only `dot_f32_avx2` runtime-gated; `sgemm`'s strip kernel is a new function, not a mirrored one, and is max-error gated); accumulation orders and `vaddvq_f32`/AVX2 hsum (T5, T7); rounding/NaN semantics (`fmaxf`, `roundf`, `clamp`, `signum`) (T3, T4); software f16 everywhere except the verbatim bit-surgery (T7); libm per platform verified by the dumps (T8); chunk geometry identical on the rayon twin (T1) and pinned for `for_each_out_chunk` (T7); no exceptions across boundaries — `Result` + `panic()`, Rust index panics made explicit (every task lists its sites).
- **§4 verification:** 26 dense tests by name (attention 7, elementwise 7, rope 4, softmax 3, reduce 2, layernorm 2, conv2d 1) + 5 float matmul tests by name (T7) ✓; dump cases for plan C — `layer_norm` ✓, `conv2d` stride 1 and 2, groups 1 ✓, `reduce_*` ✓, `apply_rope_partial(_scaled)` ✓, masked attention ✓, `gelu` ✓, `softmax` axis 0 ✓, `log_softmax` ✓, `matmul_nt_f32` with F16 weights (m=1 GEMV) ✓, plus the f32 k=512 GEMV case; `golden_kernels_test.cpp` (T8) with `within_rel_of_max` for the sgemm-backed cases, `bit_identical` elsewhere; `exact_equal<T>` (plan D carry-over) landed with a test; hosts note unchanged (x86 via CI).
- **§5 row C doc updates:** T9 (CLAUDE.md, ROADMAP, PARITY incl. the known-defect table, PROJECT_GUIDE, CHANGELOG, `just cpp-tidy`).
- **Rulings recorded in the Global Constraints:** `-fno-math-errno`; `-0.0f` Sum seed; `from_f32_vec` moves; plan D/E stubs; the parallel pool is a partition, not rayon; the F16 mis-decode is reproduced and reported, not fixed.
- **Placeholder scan:** the `<cpp sha>`/`<rust sha>`/`<values>` tokens in T9 step 3 are deliberate fill-ins with the command to compute them; no TBD/TODO elsewhere.
- **Type consistency:** `parallel::{num_threads, par_for, par_chunks_mut}` and `env_usize` (T1) used in T2/T5/T6/T7; `sgemm(m,k,n,alpha,a,rsa,csa,b,rsb,csb,beta,c,rsc,csc)` (T2) called with `std::ptrdiff_t` strides in T6/T7; `thermal::{tick, effective_threads}`, `spinpool::{enabled, parallelism}` (T1) in T7; kernel signatures (`const Tensor*` for Rust `Option<&Tensor>`, `std::optional<float>` for `Option<f32>`, `std::span<const size_t>` positions, `std::span<const int64_t>` axes, `std::array` conv params) declared in T3–T7 and consumed identically in T8's golden test; `sapient::testing::{within_rel_of_max, exact_equal}` (T8) as declared. Test-count arithmetic: 79 → 158 (T1 7, T2 6, T3 8, T4 13, T5 8, T6 3, T7 11, T8 23).
