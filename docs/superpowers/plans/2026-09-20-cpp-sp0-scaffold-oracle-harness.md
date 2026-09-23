# Sub-project 0: C++ Scaffold + Oracle Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `cpp/` CMake tree, the lint/SPDX/shader-sync gates, the C++ CI jobs, and the two-language oracle harness (kernel golden dumps + greedy token parity) so that every later sub-project lands against a working verification backbone — with CI green on an essentially empty library.

**Architecture:** Side-by-side layout: `cpp/` is a CMake project (one target per future Rust crate, `sapient::<crate>` aliases) built with Clang, C++20 and `-ffp-contract=off`; `crates/` stays the untouched Rust oracle except for two *test-only* examples that emit oracle data. The harness has two halves: a Rust `dump_kernels` example writes seeded kernel inputs/outputs in a tiny self-describing binary format (`.sapd`) that a C++ `sapient::testing` reader consumes; a Rust `greedy_ids` example prints greedy token ids under a fixed text contract that a shell script diffs against the (future) C++ tool — until then the script runs in an explicit Rust-vs-Rust self-check mode, never a silent skip.

**Tech Stack:** CMake ≥ 3.24 + Ninja + CMakePresets; Clang (Apple clang on macOS, clang-18 on Ubuntu, clang-cl on Windows); GoogleTest v1.15.2 via FetchContent; Python 3 for portable lint scripts; bash for the parity scripts; Rust stable for the two oracle examples; GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-20-cpp-rewrite-design.md` — sections D1 (verification backbone), D2 (layout), D3 (conventions), D5 row 0, "Verification", "Risks". This plan also amends one spec sentence (Task 8): the Rust tree gains **two** test-only examples (`dump_kernels`, `greedy_ids`), not one.

## Global Constraints

- **Branch:** work on `feat/cpp-sp0-scaffold`, created from `feat/cpp-rewrite-spec` (Task 1 step 1). Commit after every task; **never push** — the user pushes and opens the PR after Task 8 (the pre-push hook then runs cargo fmt/clippy and clang-format).
- **Compiler:** Clang only for parity builds — `CMAKE_CXX_COMPILER_ID` must match `Clang` (covers `AppleClang` and clang-cl); CMake hard-fails otherwise unless `-DSAPIENT_ALLOW_NON_CLANG=ON` (spec D1/D3, decision 4).
- **Flags:** `-ffp-contract=off` (clang-cl: `/clang:-ffp-contract=off`) applied globally; configure fails if `CMAKE_CXX_FLAGS` contains `-march=native`, `-ffast-math` or `-Ofast`. Tests build at `-O1` (mirrors `[profile.test] opt-level = 1`).
- **Standard/build:** `CMAKE_CXX_STANDARD 20`, extensions OFF, CMake ≥ 3.24, Ninja generator in every preset.
- **Naming:** CMake target `sapient_<crate>` with alias `sapient::<crate>`; namespace `sapient::<crate>`; headers under `libs/sapient-<crate>/include/sapient/<crate>/`; one `.hpp/.cpp` per future Rust module with the same file stem.
- **SPDX header (verbatim, two lines) on every new file.** `//` form for `.hpp .cpp .h .c .mm .wgsl .rs`, `#` form (after an optional shebang) for `.cmake .py .sh CMakeLists.txt .clang-format .clang-tidy`. JSON files (`CMakePresets.json`) and `.md`/fixtures are exempt.
  ```
  // SPDX-License-Identifier: AGPL-3.0-only
  // Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
  ```
- **Rust tree is frozen** except: `crates/sapient-backends/cpu/examples/dump_kernels.rs`, `crates/sapient-generate/examples/greedy_ids.rs`, and one `[[example]]` entry in `crates/sapient-generate/Cargo.toml`. Both examples must pass `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets -- -D warnings` (examples are covered by `--all-targets`).
- **No vacuous passes:** a gate that cannot run must FAIL or `GTEST_SKIP()` with a visible reason — never `return` silently (spec "Verification inventory", tier T2 anti-pattern).
- **Golden dumps are never committed.** Only the fixed-content `format_sample.sapd` (constant values, no kernel output) is committed.
- **Docs rule:** Task 8 updates `docs/PROJECT_GUIDE.md`, `CLAUDE.md`, `CONTRIBUTING.md`, `README.md`, `docs/ROADMAP.md` (CLAUDE.md "Must follow").

## File structure (what this sub-project creates)

```
cpp/
  CMakeLists.txt                 project(), options, standard, includes cmake/*, add_subdirectory(libs/*), lint ctests
  CMakePresets.json              dev / release / ci-macos / ci-linux / ci-windows (configure+build+test)
  .gitignore                     build/
  .clang-format  .clang-tidy
  cmake/flags.cmake              Clang gate + fp-contract discipline (global)
  cmake/warnings.cmake           sapient_apply_warnings(<target>)
  cmake/deps.cmake               FetchContent pins (googletest v1.15.2)
  scripts/check_spdx.py          SPDX header gate (also runnable on crates/)
  scripts/shader_sync.py         WGSL copies == Rust originals
  third_party/LICENSES.md        third-party licence inventory
  libs/sapient-core/             CMakeLists.txt, include/sapient/core/version.hpp, src/version.cpp,
                                 tests/version_test.cpp, tests/build_flags_test.cpp
  libs/sapient-testing/          CMakeLists.txt, include/sapient/testing/golden.hpp, src/golden.cpp,
                                 tests/golden_test.cpp, tests/fixtures/format_sample.sapd
  libs/sapient-backends-wgpu/shaders/*.wgsl   20 byte-identical copies of crates/sapient-backends/wgpu/src/shaders/
  tests/parity/golden_dump.sh    regenerate kernel dumps on this host (Rust)
  tests/parity/greedy_parity.sh  diff greedy token ids Rust vs C++ (or --self-check)
crates/sapient-backends/cpu/examples/dump_kernels.rs   (new, test-only)
crates/sapient-generate/examples/greedy_ids.rs         (new, test-only) + [[example]] in Cargo.toml
docs/PARITY.md                   parity ledger (template + first records)
.github/workflows/ci.yml         + cpp-lint, cpp-test-macos, cpp-test-linux, cpp-build-windows, cpp-parity
.githooks/pre-push               + clang-format dry-run over cpp/
justfile                         + cpp-* recipes
NOTICE                           dependency-manifest sentence covers both trees
```

---

### Task 1: CMake scaffold, `sapient::core` skeleton, build-flag tests

**Files:**
- Create: `cpp/CMakeLists.txt`, `cpp/CMakePresets.json`, `cpp/.gitignore`
- Create: `cpp/cmake/flags.cmake`, `cpp/cmake/warnings.cmake`, `cpp/cmake/deps.cmake`
- Create: `cpp/libs/sapient-core/CMakeLists.txt`, `cpp/libs/sapient-core/include/sapient/core/version.hpp`, `cpp/libs/sapient-core/src/version.cpp`
- Test: `cpp/libs/sapient-core/tests/version_test.cpp`, `cpp/libs/sapient-core/tests/build_flags_test.cpp`

**Interfaces:**
- Consumes: `../Cargo.toml` `[workspace.package] version = "X.Y.Z"` (single source of truth for the version while the Rust tree exists).
- Produces: CMake function `sapient_apply_warnings(<target>)`; targets `sapient_core` / alias `sapient::core`, `sapient_core_tests`; `std::string_view sapient::core::version() noexcept`; compile definition `SAPIENT_VERSION_STRING`; presets `dev`, `release`, `ci-macos`, `ci-linux`, `ci-windows` for configure, build and test; GoogleTest targets `GTest::gtest_main` available to every later `libs/*/CMakeLists.txt`.

- [ ] **Step 1: Create the working branch**

```bash
cd /Users/yashchaurasia/Documents/Codes/sapient-cpp
git checkout feat/cpp-rewrite-spec && git checkout -b feat/cpp-sp0-scaffold
```

- [ ] **Step 2: Write the two failing tests**

`cpp/libs/sapient-core/tests/version_test.cpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <regex>
#include <string>

#include "sapient/core/version.hpp"

TEST(Version, MatchesCMakeProjectVersion) {
    // SAPIENT_VERSION_STRING is injected by CMake from ../Cargo.toml [workspace.package].
    EXPECT_EQ(sapient::core::version(), SAPIENT_VERSION_STRING);
}

TEST(Version, IsSemver) {
    const std::string v(sapient::core::version());
    EXPECT_TRUE(std::regex_match(v, std::regex(R"(\d+\.\d+\.\d+)"))) << v;
}
```

`cpp/libs/sapient-core/tests/build_flags_test.cpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
//
// Build-flag discipline (spec D1): bit-identity of the quantized kernels with the Rust oracle
// requires that the compiler never contracts `a*b + c` into a fused multiply-add.
#include <gtest/gtest.h>

#include <cmath>

#ifdef __FAST_MATH__
#error "-ffast-math is set: forbidden by spec D1 (breaks kernel bit-identity with the Rust build)"
#endif

static_assert(__cplusplus >= 202002L, "SAPIENT C++ requires C++20");

TEST(BuildFlags, FpContractIsOff) {
    // a*b = (1+2^-23)(1-2^-23) = 1 - 2^-46 exactly. Rounded to float that is 1.0f, so the
    // two-step evaluation gives (1.0f) + (-1.0f) == 0. A fused multiply-add keeps the exact
    // product and yields -2^-46 instead. `volatile` blocks constant folding.
    volatile float a = 1.0f + 0x1p-23f;
    volatile float b = 1.0f - 0x1p-23f;
    volatile float c = -1.0f;
    const float r = a * b + c;
    EXPECT_EQ(r, 0.0f) << "a*b+c was contracted into an FMA: the build must use -ffp-contract=off";
    // Sanity: a real FMA does differ on these inputs, so the assertion above is meaningful.
    EXPECT_NE(std::fma(static_cast<float>(a), static_cast<float>(b), static_cast<float>(c)), 0.0f);
}
```

- [ ] **Step 3: Write the CMake scaffold**

`cpp/CMakeLists.txt`:

```cmake
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
cmake_minimum_required(VERSION 3.24)

# Version: single source of truth is the Rust workspace manifest while crates/ exists.
set(_cargo_toml "${CMAKE_CURRENT_SOURCE_DIR}/../Cargo.toml")
if(EXISTS "${_cargo_toml}")
  file(READ "${_cargo_toml}" _cargo)
  # `version` is the first key after the [workspace.package] header in ../Cargo.toml.
  string(REGEX MATCH "\\[workspace\\.package\\][ \t\r\n]*version[ \t]*=[ \t]*\"([0-9]+\\.[0-9]+\\.[0-9]+)\"" _m "${_cargo}")
  set(SAPIENT_VERSION "${CMAKE_MATCH_1}")
endif()
if(NOT SAPIENT_VERSION)
  set(SAPIENT_VERSION "0.6.0")
endif()

project(sapient VERSION ${SAPIENT_VERSION} LANGUAGES C CXX)

option(SAPIENT_BUILD_TESTS "Build the GoogleTest suites and lint ctests" ON)
option(SAPIENT_MLX "Metal engine via the MLX C++ API (macOS, sub-project 4)" OFF)
option(SAPIENT_WGPU "Cross-platform GPU engine via wgpu-native (sub-project 4)" OFF)
option(SAPIENT_AUDIO_IO "Live mic/speaker via miniaudio (sub-project 5c)" ON)
option(SAPIENT_ACCELERATE "Apple Accelerate cblas for f32 GEMM (sub-project 1a)" OFF)
option(SAPIENT_ALLOW_NON_CLANG "Allow GCC/MSVC — NOT a parity build" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(flags)     # Clang gate + -ffp-contract=off (global)
include(warnings)  # sapient_apply_warnings(<target>)
include(deps)      # FetchContent pins

if(SAPIENT_BUILD_TESTS)
  enable_testing()
  include(GoogleTest)
endif()

add_subdirectory(libs/sapient-core)
```

`cpp/cmake/flags.cmake`:

```cmake
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
# Build-flag discipline for bit-identity with the Rust oracle (spec D1).

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  if(SAPIENT_ALLOW_NON_CLANG)
    message(WARNING "SAPIENT: ${CMAKE_CXX_COMPILER_ID} is not a parity compiler — kernel golden gates may fail")
  else()
    message(FATAL_ERROR
      "SAPIENT requires Clang (clang-cl on Windows) for parity builds; found ${CMAKE_CXX_COMPILER_ID}. "
      "Pass -DSAPIENT_ALLOW_NON_CLANG=ON for a non-parity build.")
  endif()
endif()

# Rust never contracts a*b+c into an FMA implicitly; Clang defaults to -ffp-contract=on.
if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  add_compile_options(/clang:-ffp-contract=off)
else()
  add_compile_options(-ffp-contract=off -fno-fast-math)
endif()

foreach(_bad IN ITEMS "-march=native" "-ffast-math" "-Ofast")
  if(CMAKE_CXX_FLAGS MATCHES "${_bad}" OR CMAKE_C_FLAGS MATCHES "${_bad}")
    message(FATAL_ERROR "SAPIENT: '${_bad}' breaks kernel bit-identity with the Rust build (spec D1)")
  endif()
endforeach()
```

`cpp/cmake/warnings.cmake`:

```cmake
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
# Per-target warnings-as-errors (the C++ analogue of `clippy -D warnings`). Not applied to
# FetchContent dependencies.
function(sapient_apply_warnings target)
  if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    target_compile_options(${target} PRIVATE /W4 /WX)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Werror)
  endif()
endfunction()
```

`cpp/cmake/deps.cmake`:

```cmake
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
# Third-party pins. Every entry must also be listed in third_party/LICENSES.md.
include(FetchContent)

if(SAPIENT_BUILD_TESTS)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
    GIT_SHALLOW    TRUE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)   # MSVC-ABI runtime match on Windows
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()
```

`cpp/libs/sapient-core/CMakeLists.txt`:

```cmake
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
add_library(sapient_core STATIC src/version.cpp)
add_library(sapient::core ALIAS sapient_core)
target_include_directories(sapient_core PUBLIC include)
target_compile_definitions(sapient_core PRIVATE SAPIENT_VERSION_STRING="${PROJECT_VERSION}")
sapient_apply_warnings(sapient_core)

if(SAPIENT_BUILD_TESTS)
  add_executable(sapient_core_tests tests/version_test.cpp tests/build_flags_test.cpp)
  target_link_libraries(sapient_core_tests PRIVATE sapient::core GTest::gtest_main)
  target_compile_definitions(sapient_core_tests PRIVATE SAPIENT_VERSION_STRING="${PROJECT_VERSION}")
  sapient_apply_warnings(sapient_core_tests)
  gtest_discover_tests(sapient_core_tests)
endif()
```

`cpp/libs/sapient-core/include/sapient/core/version.hpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once

#include <string_view>

namespace sapient::core {

/// Engine version — mirrors the Rust workspace `[workspace.package] version`.
std::string_view version() noexcept;

}  // namespace sapient::core
```

`cpp/libs/sapient-core/src/version.cpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/core/version.hpp"

namespace sapient::core {

std::string_view version() noexcept { return SAPIENT_VERSION_STRING; }

}  // namespace sapient::core
```

`cpp/CMakePresets.json`:

```json
{
  "version": 6,
  "cmakeMinimumRequired": { "major": 3, "minor": 24, "patch": 0 },
  "configurePresets": [
    { "name": "base", "hidden": true, "generator": "Ninja",
      "binaryDir": "${sourceDir}/build/${presetName}" },
    { "name": "dev", "inherits": "base",
      "displayName": "Dev — tests at -O1 like cargo's [profile.test]",
      "cacheVariables": { "CMAKE_BUILD_TYPE": "Debug",
                          "CMAKE_CXX_FLAGS_DEBUG": "-O1 -g", "CMAKE_C_FLAGS_DEBUG": "-O1 -g" } },
    { "name": "release", "inherits": "base",
      "cacheVariables": { "CMAKE_BUILD_TYPE": "Release" } },
    { "name": "ci-macos", "inherits": "dev",
      "condition": { "type": "equals", "lhs": "${hostSystemName}", "rhs": "Darwin" } },
    { "name": "ci-linux", "inherits": "dev",
      "condition": { "type": "equals", "lhs": "${hostSystemName}", "rhs": "Linux" },
      "cacheVariables": { "CMAKE_C_COMPILER": "clang", "CMAKE_CXX_COMPILER": "clang++" } },
    { "name": "ci-windows", "inherits": "base",
      "condition": { "type": "equals", "lhs": "${hostSystemName}", "rhs": "Windows" },
      "cacheVariables": { "CMAKE_BUILD_TYPE": "Debug",
                          "CMAKE_C_COMPILER": "clang-cl", "CMAKE_CXX_COMPILER": "clang-cl",
                          "CMAKE_CXX_FLAGS_DEBUG": "/clang:-O1 /Zi", "CMAKE_C_FLAGS_DEBUG": "/clang:-O1 /Zi" } }
  ],
  "buildPresets": [
    { "name": "dev", "configurePreset": "dev" },
    { "name": "release", "configurePreset": "release" },
    { "name": "ci-macos", "configurePreset": "ci-macos" },
    { "name": "ci-linux", "configurePreset": "ci-linux" },
    { "name": "ci-windows", "configurePreset": "ci-windows" }
  ],
  "testPresets": [
    { "name": "dev", "configurePreset": "dev", "output": { "outputOnFailure": true } },
    { "name": "ci-macos", "configurePreset": "ci-macos", "output": { "outputOnFailure": true } },
    { "name": "ci-linux", "configurePreset": "ci-linux", "output": { "outputOnFailure": true } },
    { "name": "ci-windows", "configurePreset": "ci-windows", "output": { "outputOnFailure": true } }
  ]
}
```

`cpp/.gitignore`:

```
build/
```

- [ ] **Step 4: Configure, build, run — expect both suites green**

Run (needs `cmake ≥ 3.24` and `ninja`; on this Mac `brew install cmake ninja` if missing):

```bash
cd cpp && cmake --preset dev && cmake --build --preset dev && ctest --preset dev
```

Expected: configure prints `-- The CXX compiler identification is AppleClang …`; ctest reports `100% tests passed, 0 tests failed out of 3` (Version.MatchesCMakeProjectVersion, Version.IsSemver, BuildFlags.FpContractIsOff).

- [ ] **Step 5: Commit**

```bash
git add cpp
git commit -m "cpp(sp0): CMake scaffold, sapient::core version, build-flag discipline tests

Clang-only gate, global -ffp-contract=off, presets dev/release/ci-*, GoogleTest
v1.15.2 pin, and a test that detects FMA contraction.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 6: Prove the fp-contract test actually detects contraction**

```bash
cd cpp && cmake --preset dev -DCMAKE_CXX_FLAGS="-ffp-contract=fast" 2>&1 | tail -2
```

Expected: configure succeeds (the forbidden-flag scan only lists `-march=native/-ffast-math/-Ofast`). Then:

```bash
cmake --build --preset dev && ctest --preset dev -R FpContract
```

Expected on Apple Silicon: the later global `-ffp-contract=off` from `flags.cmake` wins (it is added after `CMAKE_CXX_FLAGS` on the command line), so this still PASSES — which is the point: the flag file governs. Now bypass it to see the detector fire:

```bash
sed -i '' 's/-ffp-contract=off -fno-fast-math/-fno-fast-math/' cmake/flags.cmake
cmake --preset dev -DCMAKE_CXX_FLAGS="-ffp-contract=fast" >/dev/null && cmake --build --preset dev >/dev/null && ctest --preset dev -R FpContract; git checkout -- cmake/flags.cmake
```

Expected: `BuildFlags.FpContractIsOff` FAILS with "was contracted into an FMA". (On an x86-64 Mac without `-mfma` no FMA is emitted and the test passes either way; note that in the commit message if so.) The trailing `git checkout` restores the committed file (this is why the commit comes first); reconfigure clean and confirm nothing is left modified:

```bash
cmake --preset dev -UCMAKE_CXX_FLAGS && cmake --build --preset dev && ctest --preset dev && git status --short cpp
```

Expected: 3/3 pass and an empty `git status` for `cpp/`.

---

### Task 2: Rust `dump_kernels` example (golden-dump generator, `.sapd` v1 format)

**Files:**
- Create: `crates/sapient-backends/cpu/examples/dump_kernels.rs`

**Interfaces:**
- Consumes (existing Rust, verified signatures): `quant::{quantize_q8_0_block, quantize_q4_0_block, dot_q8_0_row_f32, dot_q4_0_row_f32, dot_q4_k_row_f32, dot_q5_k_row_f32, dot_q6_k_row_f32}`, `matmul::matmul_nt(&Tensor,&Tensor)->Result<Tensor>`, `layernorm::rms_norm(&Tensor, Option<&Tensor>, f32)`, `softmax::softmax(&Tensor, i64)`, `elementwise::{silu, gelu_erf}(&Tensor)`, `rope::apply_rope(&Tensor, &[usize], f32)` (x is `[batch, heads, seq, head_dim]`), `attention::scaled_dot_product_attention(q,k,v, Option<&Tensor>, Option<f32>, n_kv_heads)` (q `[B,H,Sq,D]`, k/v `[B,Hkv,Sk,D]`; `None` mask = causal with offset `Sk−Sq`), `Tensor::{from_f32(&[f32], Vec<usize>), from_quant_bytes(&[u8], Vec<usize>, DType), to_f32_vec(), shape().dims()}`, `DType::{Q4_K,Q5_K,Q6_K,Q8_0}`, `half::f16`.
- Produces: CLI `dump_kernels --out <dir> [--seed N]` → one `<case>.sapd` per case (list below); `dump_kernels --format-sample <file>` → the fixed-content sample. **`.sapd` v1 wire format (little-endian), the contract Task 3 implements:**
  ```
  magic "SAPD" | u32 version=1 | u32 name_len | name (UTF-8) | u32 n_arrays |
  per array: u32 name_len | name | u8 dtype | u32 ndim | u64 dims[ndim] | u64 byte_len | bytes
  dtype: 0=f32 1=u8 2=i8 3=i32 4=u32 5=u64
  ```
  Array names carry a role prefix: `in:` (kernel input), `param:` (scalar/config), `out:` (Rust output). Cases written by `--out`: `quantize_q8_0_block`, `quantize_q4_0_block`, `dot_q8_0_row_f32`, `dot_q4_0_row_f32`, `dot_q4_k_row_f32`, `dot_q5_k_row_f32`, `dot_q6_k_row_f32`, `matmul_nt_f32_m1`, `matmul_nt_f32_m4`, `matmul_nt_q8_0_m1`, `matmul_nt_q8_0_m3`, `matmul_nt_q4_k_m1`, `matmul_nt_q4_k_m3`, `matmul_nt_q6_k_m1`, `matmul_nt_q6_k_m3`, `rms_norm`, `softmax`, `silu`, `gelu_erf`, `apply_rope`, `attention_prefill`, `attention_decode` (22 cases). Format sample content is fixed in the code below and asserted by Task 3's test.

- [ ] **Step 1: Write the example**

`crates/sapient-backends/cpu/examples/dump_kernels.rs`:

```rust
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
//! Golden-dump generator for the Rust→C++ parity harness (TEST-ONLY; not part of the product).
//!
//! Writes one `.sapd` file per kernel case: seeded-random inputs plus the outputs the Rust
//! kernels produce ON THIS HOST. The public kernel entry points dispatch on the host ISA at
//! runtime (NEON/SDOT/SMMLA vs scalar/AVX2), so dumps are host-specific: regenerate them in the
//! same job that consumes them and never commit them (spec D1).
//!
//! Wire format v1 (little-endian):
//!   "SAPD" | u32 version=1 | u32 name_len | name | u32 n_arrays |
//!   per array: u32 name_len | name | u8 dtype | u32 ndim | u64 dims[ndim] | u64 byte_len | bytes
//!   dtype: 0=f32 1=u8 2=i8 3=i32 4=u32 5=u64.  Names are prefixed `in:` / `param:` / `out:`.
//!
//! Usage:
//!   cargo run --release -p sapient-backends-cpu --example dump_kernels -- --out <dir> [--seed N]
//!   cargo run --release -p sapient-backends-cpu --example dump_kernels -- --format-sample <file>

use std::error::Error;
use std::fs;
use std::path::PathBuf;

use half::f16;
use sapient_backends_cpu::kernels::{attention, elementwise, layernorm, matmul, quant, rope, softmax};
use sapient_core::{DType, Tensor};

const FORMAT_VERSION: u32 = 1;
const DT_F32: u8 = 0;
const DT_U8: u8 = 1;
const DT_I8: u8 = 2;
const DT_I32: u8 = 3;
const DT_U32: u8 = 4;
const DT_U64: u8 = 5;
const DEFAULT_SEED: u64 = 0x5A71_E47D_0000_0001;

const USAGE: &str = "usage: dump_kernels --out <dir> [--seed N] | --format-sample <file>";

/// xorshift64* — deterministic, dependency-free (same family as the sampler's RNG).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
    fn f32s(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| self.range(lo, hi)).collect()
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.next_u64() >> 56) as u8).collect()
    }
}

struct Array {
    name: String,
    dtype: u8,
    dims: Vec<u64>,
    bytes: Vec<u8>,
}

fn dtype_size(dtype: u8) -> usize {
    match dtype {
        DT_F32 | DT_I32 | DT_U32 => 4,
        DT_U8 | DT_I8 => 1,
        DT_U64 => 8,
        other => panic!("unknown dtype tag {other}"),
    }
}

impl Array {
    fn raw(name: &str, dtype: u8, dims: &[usize], bytes: Vec<u8>) -> Self {
        let numel: usize = dims.iter().product();
        assert_eq!(bytes.len(), numel * dtype_size(dtype), "{name}: byte length mismatch");
        Array {
            name: name.to_string(),
            dtype,
            dims: dims.iter().map(|&d| d as u64).collect(),
            bytes,
        }
    }
    fn f32(name: &str, dims: &[usize], v: &[f32]) -> Self {
        Self::raw(name, DT_F32, dims, v.iter().flat_map(|x| x.to_le_bytes()).collect())
    }
    fn u8(name: &str, dims: &[usize], v: &[u8]) -> Self {
        Self::raw(name, DT_U8, dims, v.to_vec())
    }
    fn i8(name: &str, dims: &[usize], v: &[i8]) -> Self {
        Self::raw(name, DT_I8, dims, v.iter().map(|&x| x as u8).collect())
    }
    fn i32(name: &str, dims: &[usize], v: &[i32]) -> Self {
        Self::raw(name, DT_I32, dims, v.iter().flat_map(|x| x.to_le_bytes()).collect())
    }
    fn u32(name: &str, dims: &[usize], v: &[u32]) -> Self {
        Self::raw(name, DT_U32, dims, v.iter().flat_map(|x| x.to_le_bytes()).collect())
    }
    fn u64(name: &str, dims: &[usize], v: &[u64]) -> Self {
        Self::raw(name, DT_U64, dims, v.iter().flat_map(|x| x.to_le_bytes()).collect())
    }
    /// An F32 tensor (any layout) as `f32` with its shape.
    fn tensor(name: &str, t: &Tensor) -> Self {
        Self::f32(name, t.shape().dims(), &t.to_f32_vec())
    }
}

fn put_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn encode_case(name: &str, arrays: &[Array]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"SAPD");
    buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    put_str(&mut buf, name);
    buf.extend_from_slice(&(arrays.len() as u32).to_le_bytes());
    for a in arrays {
        put_str(&mut buf, &a.name);
        buf.push(a.dtype);
        buf.extend_from_slice(&(a.dims.len() as u32).to_le_bytes());
        for d in &a.dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf.extend_from_slice(&(a.bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(&a.bytes);
    }
    buf
}

// ── quantized block builders ─────────────────────────────────────────────────────────────────
// K-quant blocks are random bits with FINITE f16 scales (random f16 bits could be NaN/Inf).

fn f16_bytes(v: f32) -> [u8; 2] {
    f16::from_f32(v).to_le_bytes()
}

fn q4_k_block(rng: &mut Rng) -> Vec<u8> {
    let mut b = Vec::with_capacity(144);
    b.extend_from_slice(&f16_bytes(rng.range(0.002, 0.05))); // d
    b.extend_from_slice(&f16_bytes(rng.range(0.0, 0.02))); // dmin
    b.extend(rng.bytes(12)); // packed 6-bit scales/mins
    b.extend(rng.bytes(128)); // nibbles
    b
}

fn q5_k_block(rng: &mut Rng) -> Vec<u8> {
    let mut b = Vec::with_capacity(176);
    b.extend_from_slice(&f16_bytes(rng.range(0.002, 0.05)));
    b.extend_from_slice(&f16_bytes(rng.range(0.0, 0.02)));
    b.extend(rng.bytes(12)); // scales
    b.extend(rng.bytes(32)); // qh (5th bits)
    b.extend(rng.bytes(128)); // qs
    b
}

fn q6_k_block(rng: &mut Rng) -> Vec<u8> {
    let mut b = Vec::with_capacity(210);
    b.extend(rng.bytes(128)); // ql
    b.extend(rng.bytes(64)); // qh
    b.extend(rng.bytes(16)); // 16 signed int8 scales (any bits are valid)
    b.extend_from_slice(&f16_bytes(rng.range(0.002, 0.05))); // d
    b
}

fn q8_0_row(w: &[f32]) -> Vec<u8> {
    w.chunks(32).flat_map(quant::quantize_q8_0_block).collect()
}

fn q4_0_row(w: &[f32]) -> Vec<u8> {
    w.chunks(32).flat_map(quant::quantize_q4_0_block).collect()
}

type BlockFn = fn(&mut Rng) -> Vec<u8>;
type DotFn = fn(&[u8], &[f32]) -> f32;

fn kquant_rows(rng: &mut Rng, rows: usize, k: usize, block: BlockFn) -> Vec<u8> {
    (0..rows * k / 256).flat_map(|_| block(rng)).collect()
}

type Case = (String, Vec<Array>);

fn case(name: impl Into<String>, arrays: Vec<Array>) -> Case {
    (name.into(), arrays)
}

// ── the cases ────────────────────────────────────────────────────────────────────────────────

fn build_cases(seed: u64) -> Result<Vec<Case>, Box<dyn Error>> {
    let mut rng = Rng::new(seed);
    let mut cases: Vec<Case> = Vec::new();

    // Block quantizers.
    let x32 = rng.f32s(32, -1.0, 1.0);
    let q8 = quant::quantize_q8_0_block(&x32);
    cases.push(case(
        "quantize_q8_0_block",
        vec![Array::f32("in:x", &[32], &x32), Array::u8("out:block", &[34], &q8)],
    ));
    let q4 = quant::quantize_q4_0_block(&x32);
    cases.push(case(
        "quantize_q4_0_block",
        vec![Array::f32("in:x", &[32], &x32), Array::u8("out:block", &[18], &q4)],
    ));

    // Row dot products (k = 256 for the 32-wide formats, 512 = two super-blocks for K-quants).
    let w = rng.f32s(256, -1.0, 1.0);
    let x = rng.f32s(256, -1.0, 1.0);
    let row = q8_0_row(&w);
    let y = quant::dot_q8_0_row_f32(&row, &x);
    cases.push(case(
        "dot_q8_0_row_f32",
        vec![
            Array::u8("in:row_blocks", &[row.len()], &row),
            Array::f32("in:x", &[256], &x),
            Array::f32("out:y", &[1], &[y]),
        ],
    ));
    let row = q4_0_row(&w);
    let y = quant::dot_q4_0_row_f32(&row, &x);
    cases.push(case(
        "dot_q4_0_row_f32",
        vec![
            Array::u8("in:row_blocks", &[row.len()], &row),
            Array::f32("in:x", &[256], &x),
            Array::f32("out:y", &[1], &[y]),
        ],
    ));
    let xk = rng.f32s(512, -1.0, 1.0);
    let kq: [(&str, BlockFn, DotFn); 3] = [
        ("dot_q4_k_row_f32", q4_k_block, quant::dot_q4_k_row_f32),
        ("dot_q5_k_row_f32", q5_k_block, quant::dot_q5_k_row_f32),
        ("dot_q6_k_row_f32", q6_k_block, quant::dot_q6_k_row_f32),
    ];
    for (name, block, dot) in kq {
        let row = kquant_rows(&mut rng, 1, 512, block);
        let y = dot(&row, &xk);
        cases.push(case(
            name,
            vec![
                Array::u8("in:row_blocks", &[row.len()], &row),
                Array::f32("in:x", &[512], &xk),
                Array::f32("out:y", &[1], &[y]),
            ],
        ));
    }

    // matmul_nt: f32 weights (GEMV m=1 and GEMM m=4).
    let (k, n) = (64usize, 16usize);
    let wt = Tensor::from_f32(&rng.f32s(n * k, -1.0, 1.0), vec![n, k])?;
    for m in [1usize, 4] {
        let xt = Tensor::from_f32(&rng.f32s(m * k, -1.0, 1.0), vec![m, k])?;
        let y = matmul::matmul_nt(&xt, &wt)?;
        cases.push(case(
            format!("matmul_nt_f32_m{m}"),
            vec![Array::tensor("in:x", &xt), Array::tensor("in:w", &wt), Array::tensor("out:y", &y)],
        ));
    }
    // matmul_nt: quantized weights [8, 512] (decode m=1 and prefill m=3 paths).
    let (rows, kq_len) = (8usize, 512usize);
    let wq8 = q8_0_row(&rng.f32s(rows * kq_len, -1.0, 1.0));
    let quants: [(&str, DType, Vec<u8>); 3] = [
        ("q8_0", DType::Q8_0, wq8),
        ("q4_k", DType::Q4_K, kquant_rows(&mut rng, rows, kq_len, q4_k_block)),
        ("q6_k", DType::Q6_K, kquant_rows(&mut rng, rows, kq_len, q6_k_block)),
    ];
    for (tag, dtype, wbytes) in quants {
        let wt = Tensor::from_quant_bytes(&wbytes, vec![rows, kq_len], dtype)?;
        for m in [1usize, 3] {
            let xt = Tensor::from_f32(&rng.f32s(m * kq_len, -1.0, 1.0), vec![m, kq_len])?;
            let y = matmul::matmul_nt(&xt, &wt)?;
            cases.push(case(
                format!("matmul_nt_{tag}_m{m}"),
                vec![
                    Array::tensor("in:x", &xt),
                    Array::u8("in:w_blocks", &[wbytes.len()], &wbytes),
                    Array::u32("param:w_shape", &[2], &[rows as u32, kq_len as u32]),
                    Array::tensor("out:y", &y),
                ],
            ));
        }
    }

    // Norm / activation / softmax.
    let xt = Tensor::from_f32(&rng.f32s(2 * 64, -2.0, 2.0), vec![2, 64])?;
    let wt = Tensor::from_f32(&rng.f32s(64, 0.5, 1.5), vec![64])?;
    let y = layernorm::rms_norm(&xt, Some(&wt), 1e-5)?;
    cases.push(case(
        "rms_norm",
        vec![
            Array::tensor("in:x", &xt),
            Array::tensor("in:weight", &wt),
            Array::f32("param:eps", &[1], &[1e-5]),
            Array::tensor("out:y", &y),
        ],
    ));
    let xt = Tensor::from_f32(&rng.f32s(2 * 3 * 8, -4.0, 4.0), vec![2, 3, 8])?;
    let y = softmax::softmax(&xt, -1)?;
    cases.push(case(
        "softmax",
        vec![Array::tensor("in:x", &xt), Array::i32("param:axis", &[1], &[-1]), Array::tensor("out:y", &y)],
    ));
    let xt = Tensor::from_f32(&rng.f32s(64, -6.0, 6.0), vec![64])?;
    cases.push(case(
        "silu",
        vec![Array::tensor("in:x", &xt), Array::tensor("out:y", &elementwise::silu(&xt)?)],
    ));
    cases.push(case(
        "gelu_erf",
        vec![Array::tensor("in:x", &xt), Array::tensor("out:y", &elementwise::gelu_erf(&xt)?)],
    ));

    // RoPE on [batch=1, heads=2, seq=4, head_dim=16] at cache offset 5.
    let xt = Tensor::from_f32(&rng.f32s(2 * 4 * 16, -1.0, 1.0), vec![1, 2, 4, 16])?;
    let positions: Vec<usize> = vec![5, 6, 7, 8];
    let y = rope::apply_rope(&xt, &positions, 10_000.0)?;
    cases.push(case(
        "apply_rope",
        vec![
            Array::tensor("in:x", &xt),
            Array::u64("param:positions", &[4], &positions.iter().map(|&p| p as u64).collect::<Vec<_>>()),
            Array::f32("param:base", &[1], &[10_000.0]),
            Array::tensor("out:y", &y),
        ],
    ));

    // GQA attention (4 heads over 2 kv heads), built-in causal mask (mask = None).
    let hd = 16usize;
    let attn = |rng: &mut Rng, seq_q: usize, seq_k: usize| -> Result<Vec<Array>, Box<dyn Error>> {
        let q = Tensor::from_f32(&rng.f32s(4 * seq_q * hd, -1.0, 1.0), vec![1, 4, seq_q, hd])?;
        let k = Tensor::from_f32(&rng.f32s(2 * seq_k * hd, -1.0, 1.0), vec![1, 2, seq_k, hd])?;
        let v = Tensor::from_f32(&rng.f32s(2 * seq_k * hd, -1.0, 1.0), vec![1, 2, seq_k, hd])?;
        let y = attention::scaled_dot_product_attention(&q, &k, &v, None, None, 2)?;
        Ok(vec![
            Array::tensor("in:q", &q),
            Array::tensor("in:k", &k),
            Array::tensor("in:v", &v),
            Array::u32("param:n_kv_heads", &[1], &[2]),
            Array::tensor("out:y", &y),
        ])
    };
    cases.push(case("attention_prefill", attn(&mut rng, 4, 4)?));
    cases.push(case("attention_decode", attn(&mut rng, 1, 5)?));

    Ok(cases)
}

/// Fixed-content sample for the C++ reader's unit test (committed as a fixture; contains no
/// kernel output, so it is host-independent).
fn format_sample() -> Case {
    case(
        "format_sample",
        vec![
            Array::f32("in:f32", &[5], &[0.0, 1.0, -1.0, 0.5, 3.25]),
            Array::u8("in:u8", &[2, 2], &[0, 1, 2, 255]),
            Array::i8("in:i8", &[2], &[-128, 127]),
            Array::i32("param:i32", &[1], &[-42]),
            Array::u32("param:u32", &[1], &[4_000_000_000]),
            Array::u64("param:u64", &[1], &[1 << 40]),
            Array::f32("out:empty", &[0], &[]),
        ],
    )
}

fn take(args: &[String], i: usize, flag: &str) -> String {
    args.get(i + 1).cloned().unwrap_or_else(|| {
        eprintln!("{flag} needs a value\n{USAGE}");
        std::process::exit(2)
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut out: Option<PathBuf> = None;
    let mut sample: Option<PathBuf> = None;
    let mut seed = DEFAULT_SEED;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                out = Some(PathBuf::from(take(&args, i, "--out")));
                i += 2;
            }
            "--format-sample" => {
                sample = Some(PathBuf::from(take(&args, i, "--format-sample")));
                i += 2;
            }
            "--seed" => {
                seed = take(&args, i, "--seed").parse()?;
                i += 2;
            }
            other => {
                eprintln!("unknown argument: {other}\n{USAGE}");
                std::process::exit(2);
            }
        }
    }

    if sample.is_none() && out.is_none() {
        eprintln!("{USAGE}");
        std::process::exit(2);
    }
    if let Some(path) = sample {
        let (name, arrays) = format_sample();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, encode_case(&name, &arrays))?;
        println!("wrote format sample to {}", path.display());
    }
    if let Some(dir) = out {
        fs::create_dir_all(&dir)?;
        let cases = build_cases(seed)?;
        for (name, arrays) in &cases {
            let path = dir.join(format!("{name}.sapd"));
            fs::write(&path, encode_case(name, arrays))?;
            println!("{}", path.display());
        }
        println!("wrote {} cases to {} (seed {seed:#x})", cases.len(), dir.display());
    }
    Ok(())
}
```

- [ ] **Step 2: Run it and inspect the output**

```bash
cargo run --release -q -p sapient-backends-cpu --example dump_kernels -- --out /tmp/sapient-golden --format-sample /tmp/format_sample.sapd
ls /tmp/sapient-golden | wc -l
xxd /tmp/format_sample.sapd | head -3
```

Expected: last line `wrote 22 cases to /tmp/sapient-golden (seed 0x5a71e47d00000001)`; `22`; the hexdump starts `5341 5044 0100 0000 0d00 0000 666f 726d` (`SAPD`, version 1, name length 13, `form…`).

- [ ] **Step 3: Determinism check — same seed, identical bytes**

```bash
cargo run --release -q -p sapient-backends-cpu --example dump_kernels -- --out /tmp/sapient-golden2 >/dev/null && diff -rq /tmp/sapient-golden /tmp/sapient-golden2 && echo IDENTICAL
```

Expected: `IDENTICAL` (no diff output). This also proves the Rust kernels are run-to-run deterministic on this host, including the rayon/spinpool-parallel `matmul_nt` cases.

- [ ] **Step 4: Rust lint gates**

```bash
cargo fmt --all && cargo clippy -p sapient-backends-cpu --all-targets -- -D warnings
```

Expected: no diff from fmt, clippy clean. (If clippy flags `needless_range_loop`-style lints in the arg parser, fix them rather than allow-listing.)

- [ ] **Step 5: Commit**

```bash
git add crates/sapient-backends/cpu/examples/dump_kernels.rs
git commit -m "test(oracle): dump_kernels example — seeded kernel golden dumps in .sapd v1

Test-only Rust tool for the C++ parity harness: 22 cases through the public
CPU kernel entry points (host-ISA dispatch exercised at runtime), plus a
fixed-content format sample. Dumps are host-specific and never committed.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: C++ `sapient::testing` golden reader + committed format fixture + `golden_dump.sh`

**Files:**
- Create: `cpp/libs/sapient-testing/CMakeLists.txt`, `cpp/libs/sapient-testing/include/sapient/testing/golden.hpp`, `cpp/libs/sapient-testing/src/golden.cpp`
- Create: `cpp/libs/sapient-testing/tests/fixtures/format_sample.sapd` (generated by Task 2's tool), `cpp/.gitattributes`
- Create: `cpp/tests/parity/golden_dump.sh`
- Modify: `cpp/CMakeLists.txt` (add `add_subdirectory(libs/sapient-testing)`)
- Test: `cpp/libs/sapient-testing/tests/golden_test.cpp`

**Interfaces:**
- Consumes: the `.sapd` v1 format from Task 2; CMake `sapient_apply_warnings`, `GTest::gtest_main` from Task 1.
- Produces (used by every kernel test from sub-project 1a on):
  ```cpp
  namespace sapient::testing {
  enum class GoldenDType : uint8_t { F32 = 0, U8 = 1, I8 = 2, I32 = 3, U32 = 4, U64 = 5 };
  size_t dtype_size(GoldenDType);
  struct GoldenArray { std::string name; GoldenDType dtype; std::vector<uint64_t> dims; std::vector<uint8_t> bytes;
                       size_t numel() const; template <class T> std::vector<T> as() const; /* throws std::logic_error on size mismatch */ };
  struct GoldenCase  { std::string name; std::vector<GoldenArray> arrays;
                       const GoldenArray* find(std::string_view) const; const GoldenArray& get(std::string_view) const; /* throws std::out_of_range */ };
  std::optional<GoldenCase> read_golden(const std::filesystem::path&, std::string* error = nullptr);
  std::vector<std::filesystem::path> list_golden(const std::filesystem::path& dir); // *.sapd, sorted
  }
  ```
  Env contract: `SAPIENT_GOLDEN_DIR` = directory of dumps produced by `cpp/tests/parity/golden_dump.sh <dir>`.

- [ ] **Step 1: Generate and commit the format fixture**

```bash
cargo run --release -q -p sapient-backends-cpu --example dump_kernels -- --format-sample cpp/libs/sapient-testing/tests/fixtures/format_sample.sapd
xxd cpp/libs/sapient-testing/tests/fixtures/format_sample.sapd | wc -l
```

Expected: `wrote format sample to …`; the file is 306 bytes (20 lines of xxd). It is deterministic (no RNG, no kernel), so it is safe to commit. Mark dumps as binary so the Windows runners' `core.autocrlf` can never rewrite a `0x0d` byte:

```bash
printf '*.sapd binary\n' > cpp/.gitattributes
```

- [ ] **Step 2: Write the failing tests**

`cpp/libs/sapient-testing/tests/golden_test.cpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "sapient/testing/golden.hpp"

namespace fs = std::filesystem;
using sapient::testing::GoldenDType;
using sapient::testing::list_golden;
using sapient::testing::read_golden;

namespace {

fs::path fixtures() { return fs::path(SAPIENT_TESTING_FIXTURES_DIR); }

fs::path temp_file(const char* stem) {
    return fs::temp_directory_path() / (std::string("sapient_golden_") + stem + ".sapd");
}

void write_bytes(const fs::path& p, const std::vector<uint8_t>& bytes) {
    std::ofstream out(p, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

std::vector<uint8_t> read_bytes(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

}  // namespace

TEST(Golden, ReadsFormatSampleWrittenByRust) {
    std::string err;
    auto c = read_golden(fixtures() / "format_sample.sapd", &err);
    ASSERT_TRUE(c.has_value()) << err;
    EXPECT_EQ(c->name, "format_sample");
    ASSERT_EQ(c->arrays.size(), 7u);

    const auto& f = c->get("in:f32");
    EXPECT_EQ(f.dtype, GoldenDType::F32);
    EXPECT_EQ(f.dims, (std::vector<uint64_t>{5}));
    EXPECT_EQ(f.numel(), 5u);
    EXPECT_EQ(f.as<float>(), (std::vector<float>{0.0f, 1.0f, -1.0f, 0.5f, 3.25f}));

    EXPECT_EQ(c->get("in:u8").dims, (std::vector<uint64_t>{2, 2}));
    EXPECT_EQ(c->get("in:u8").as<uint8_t>(), (std::vector<uint8_t>{0, 1, 2, 255}));
    EXPECT_EQ(c->get("in:i8").as<int8_t>(), (std::vector<int8_t>{-128, 127}));
    EXPECT_EQ(c->get("param:i32").as<int32_t>(), (std::vector<int32_t>{-42}));
    EXPECT_EQ(c->get("param:u32").as<uint32_t>(), (std::vector<uint32_t>{4000000000u}));
    EXPECT_EQ(c->get("param:u64").as<uint64_t>(), (std::vector<uint64_t>{uint64_t{1} << 40}));
    EXPECT_EQ(c->get("out:empty").numel(), 0u);
    EXPECT_EQ(c->find("does-not-exist"), nullptr);
    EXPECT_THROW(c->get("does-not-exist"), std::out_of_range);
    EXPECT_THROW(c->get("in:f32").as<uint8_t>(), std::logic_error);  // element size mismatch
}

TEST(Golden, RejectsBadMagic) {
    const auto p = temp_file("badmagic");
    write_bytes(p, {'X', 'X', 'X', 'X', 1, 0, 0, 0});
    std::string err;
    EXPECT_FALSE(read_golden(p, &err).has_value());
    EXPECT_NE(err.find("magic"), std::string::npos) << err;
    fs::remove(p);
}

TEST(Golden, RejectsTruncatedFile) {
    auto bytes = read_bytes(fixtures() / "format_sample.sapd");
    ASSERT_GT(bytes.size(), 40u);
    bytes.resize(bytes.size() / 2);
    const auto p = temp_file("truncated");
    write_bytes(p, bytes);
    std::string err;
    EXPECT_FALSE(read_golden(p, &err).has_value());
    EXPECT_NE(err.find("truncated"), std::string::npos) << err;
    fs::remove(p);
}

TEST(Golden, RejectsMissingFile) {
    std::string err;
    EXPECT_FALSE(read_golden(temp_file("nope-missing"), &err).has_value());
    EXPECT_FALSE(err.empty());
}

// Real gate when SAPIENT_GOLDEN_DIR points at dumps made on this host by
// cpp/tests/parity/golden_dump.sh; an explicit, visible skip otherwise (never a silent pass).
TEST(Golden, InventoryFromEnv) {
    const char* dir = std::getenv("SAPIENT_GOLDEN_DIR");
    if (dir == nullptr) {
        GTEST_SKIP() << "SAPIENT_GOLDEN_DIR not set — run cpp/tests/parity/golden_dump.sh <dir> and export it";
    }
    const auto files = list_golden(dir);
    ASSERT_FALSE(files.empty()) << "no *.sapd files in " << dir;
    for (const auto& f : files) {
        std::string err;
        auto c = read_golden(f, &err);
        ASSERT_TRUE(c.has_value()) << f << ": " << err;
        EXPECT_EQ(c->name, f.stem().string()) << f;
        bool has_out = false;
        for (const auto& a : c->arrays) has_out = has_out || a.name.starts_with("out:");
        EXPECT_TRUE(has_out) << f << " has no out: array";
    }
    std::printf("golden inventory: %zu cases in %s\n", files.size(), dir);
}
```

- [ ] **Step 3: Write the library**

`cpp/libs/sapient-testing/include/sapient/testing/golden.hpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
//
// Reader for the `.sapd` v1 golden-dump format written by the Rust oracle
// (crates/sapient-backends/cpu/examples/dump_kernels.rs). Test-support only.
//
//   "SAPD" | u32 version=1 | u32 name_len | name | u32 n_arrays |
//   per array: u32 name_len | name | u8 dtype | u32 ndim | u64 dims[ndim] | u64 byte_len | bytes
//   dtype: 0=f32 1=u8 2=i8 3=i32 4=u32 5=u64. All integers little-endian.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace sapient::testing {

enum class GoldenDType : uint8_t { F32 = 0, U8 = 1, I8 = 2, I32 = 3, U32 = 4, U64 = 5 };

/// Bytes per element for a dtype tag.
size_t dtype_size(GoldenDType dtype);

struct GoldenArray {
    std::string name;  // "in:x", "param:eps", "out:y", …
    GoldenDType dtype{GoldenDType::F32};
    std::vector<uint64_t> dims;
    std::vector<uint8_t> bytes;

    size_t numel() const;

    /// Decode the payload as a vector of T (copy; T must match the dtype's element size).
    template <class T>
    std::vector<T> as() const {
        if (sizeof(T) != dtype_size(dtype)) {
            throw std::logic_error("GoldenArray::as<T>: element size mismatch for " + name);
        }
        std::vector<T> out(bytes.size() / sizeof(T));
        if (!out.empty()) std::memcpy(out.data(), bytes.data(), bytes.size());
        return out;
    }
};

struct GoldenCase {
    std::string name;
    std::vector<GoldenArray> arrays;

    const GoldenArray* find(std::string_view array_name) const;
    /// Like find(), but throws std::out_of_range when absent.
    const GoldenArray& get(std::string_view array_name) const;
};

/// Parse one dump file. On failure returns nullopt and, if `error` is non-null, a reason.
std::optional<GoldenCase> read_golden(const std::filesystem::path& file, std::string* error = nullptr);

/// All `*.sapd` files in `dir`, sorted by path. Empty if `dir` does not exist.
std::vector<std::filesystem::path> list_golden(const std::filesystem::path& dir);

}  // namespace sapient::testing
```

`cpp/libs/sapient-testing/src/golden.cpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/testing/golden.hpp"

#include <algorithm>
#include <bit>
#include <fstream>
#include <iterator>

static_assert(std::endian::native == std::endian::little, "the .sapd reader assumes a little-endian host");

namespace sapient::testing {

namespace {

constexpr uint32_t kFormatVersion = 1;

struct Cursor {
    const std::vector<uint8_t>& buf;
    size_t pos = 0;

    bool has(size_t n) const { return pos + n <= buf.size(); }

    template <class T>
    bool read(T& out) {
        if (!has(sizeof(T))) return false;
        std::memcpy(&out, buf.data() + pos, sizeof(T));
        pos += sizeof(T);
        return true;
    }

    bool read_string(std::string& out) {
        uint32_t len = 0;
        if (!read(len) || !has(len)) return false;
        out.assign(reinterpret_cast<const char*>(buf.data() + pos), len);
        pos += len;
        return true;
    }

    bool read_bytes(std::vector<uint8_t>& out, uint64_t len) {
        if (!has(static_cast<size_t>(len))) return false;
        out.assign(buf.begin() + static_cast<std::ptrdiff_t>(pos),
                   buf.begin() + static_cast<std::ptrdiff_t>(pos + len));
        pos += static_cast<size_t>(len);
        return true;
    }
};

void set_error(std::string* error, std::string message) {
    if (error != nullptr) *error = std::move(message);
}

}  // namespace

size_t dtype_size(GoldenDType dtype) {
    switch (dtype) {
    case GoldenDType::F32:
    case GoldenDType::I32:
    case GoldenDType::U32:
        return 4;
    case GoldenDType::U8:
    case GoldenDType::I8:
        return 1;
    case GoldenDType::U64:
        return 8;
    }
    throw std::logic_error("unknown GoldenDType tag");
}

size_t GoldenArray::numel() const {
    size_t n = 1;
    for (const auto d : dims) n *= static_cast<size_t>(d);
    return dims.empty() ? 1 : n;
}

const GoldenArray* GoldenCase::find(std::string_view array_name) const {
    for (const auto& a : arrays) {
        if (a.name == array_name) return &a;
    }
    return nullptr;
}

const GoldenArray& GoldenCase::get(std::string_view array_name) const {
    const auto* a = find(array_name);
    if (a == nullptr) throw std::out_of_range("golden case '" + name + "' has no array '" + std::string(array_name) + "'");
    return *a;
}

std::optional<GoldenCase> read_golden(const std::filesystem::path& file, std::string* error) {
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        set_error(error, "cannot open " + file.string());
        return std::nullopt;
    }
    const std::vector<uint8_t> buf{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    Cursor cur{buf};

    if (buf.size() < 4 || std::memcmp(buf.data(), "SAPD", 4) != 0) {
        set_error(error, file.string() + ": bad magic (expected \"SAPD\")");
        return std::nullopt;
    }
    cur.pos = 4;
    uint32_t version = 0;
    if (!cur.read(version)) {
        set_error(error, file.string() + ": truncated before version");
        return std::nullopt;
    }
    if (version != kFormatVersion) {
        set_error(error, file.string() + ": unsupported .sapd version " + std::to_string(version));
        return std::nullopt;
    }

    GoldenCase c;
    uint32_t n_arrays = 0;
    if (!cur.read_string(c.name) || !cur.read(n_arrays)) {
        set_error(error, file.string() + ": truncated header");
        return std::nullopt;
    }
    c.arrays.reserve(n_arrays);
    for (uint32_t i = 0; i < n_arrays; ++i) {
        GoldenArray a;
        uint8_t tag = 0;
        uint32_t ndim = 0;
        uint64_t byte_len = 0;
        if (!cur.read_string(a.name) || !cur.read(tag) || !cur.read(ndim)) {
            set_error(error, file.string() + ": truncated array header #" + std::to_string(i));
            return std::nullopt;
        }
        if (tag > static_cast<uint8_t>(GoldenDType::U64)) {
            set_error(error, file.string() + ": unknown dtype tag " + std::to_string(tag) + " in " + a.name);
            return std::nullopt;
        }
        a.dtype = static_cast<GoldenDType>(tag);
        a.dims.resize(ndim);
        for (auto& d : a.dims) {
            if (!cur.read(d)) {
                set_error(error, file.string() + ": truncated dims in " + a.name);
                return std::nullopt;
            }
        }
        if (!cur.read(byte_len) || !cur.read_bytes(a.bytes, byte_len)) {
            set_error(error, file.string() + ": truncated payload in " + a.name);
            return std::nullopt;
        }
        if (a.bytes.size() != a.numel() * dtype_size(a.dtype)) {
            set_error(error, file.string() + ": byte length does not match dims×dtype in " + a.name);
            return std::nullopt;
        }
        c.arrays.push_back(std::move(a));
    }
    if (cur.pos != buf.size()) {
        set_error(error, file.string() + ": trailing bytes after last array");
        return std::nullopt;
    }
    return c;
}

std::vector<std::filesystem::path> list_golden(const std::filesystem::path& dir) {
    std::vector<std::filesystem::path> out;
    if (!std::filesystem::is_directory(dir)) return out;
    for (const auto& e : std::filesystem::directory_iterator(dir)) {
        if (e.is_regular_file() && e.path().extension() == ".sapd") out.push_back(e.path());
    }
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace sapient::testing
```

`cpp/libs/sapient-testing/CMakeLists.txt`:

```cmake
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
# Test-support library: reader for the Rust oracle's .sapd golden dumps. Not shipped.
add_library(sapient_testing STATIC src/golden.cpp)
add_library(sapient::testing ALIAS sapient_testing)
target_include_directories(sapient_testing PUBLIC include)
sapient_apply_warnings(sapient_testing)

if(SAPIENT_BUILD_TESTS)
  add_executable(sapient_testing_tests tests/golden_test.cpp)
  target_link_libraries(sapient_testing_tests PRIVATE sapient::testing GTest::gtest_main)
  target_compile_definitions(sapient_testing_tests PRIVATE
    SAPIENT_TESTING_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")
  sapient_apply_warnings(sapient_testing_tests)
  gtest_discover_tests(sapient_testing_tests)
endif()
```

Append to `cpp/CMakeLists.txt` after `add_subdirectory(libs/sapient-core)`:

```cmake
if(SAPIENT_BUILD_TESTS)
  add_subdirectory(libs/sapient-testing)
endif()
```

`cpp/tests/parity/golden_dump.sh` (make executable):

```bash
#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#
# Regenerate the kernel golden dumps ON THIS HOST with the Rust oracle, then point the C++
# tests at them:   cpp/tests/parity/golden_dump.sh <out-dir> [--seed N]
#                  SAPIENT_GOLDEN_DIR=<out-dir> ctest --preset dev
# Dumps are host-specific (ISA dispatch) — never commit them.
set -euo pipefail
out=${1:?usage: golden_dump.sh <out-dir> [--seed N]}
shift
repo=$(cd "$(dirname "$0")/../../.." && pwd)
mkdir -p "$out"
(cd "$repo" && cargo run --release -q -p sapient-backends-cpu --example dump_kernels -- --out "$out" "$@")
count=$(find "$out" -name '*.sapd' | wc -l | tr -d ' ')
echo "golden_dump: $count cases in $out — export SAPIENT_GOLDEN_DIR=$out before running ctest"
```

- [ ] **Step 4: Run the tests — first without dumps (visible skip), then with**

```bash
chmod +x cpp/tests/parity/golden_dump.sh
cd cpp && cmake --preset dev && cmake --build --preset dev && ctest --preset dev
```

Expected: `8 tests` total; `Golden.InventoryFromEnv` reports `***Skipped` with the SAPIENT_GOLDEN_DIR message; the other 7 pass. Then:

```bash
./tests/parity/golden_dump.sh /tmp/sapient-golden
SAPIENT_GOLDEN_DIR=/tmp/sapient-golden ctest --preset dev -R Golden.InventoryFromEnv -V | grep -E "golden inventory|Passed"
```

Expected: `golden inventory: 22 cases in /tmp/sapient-golden` and `Passed`.

- [ ] **Step 5: Commit**

```bash
git add cpp
git commit -m "cpp(sp0): sapient::testing golden-dump reader + committed format fixture

Reads the Rust oracle's .sapd v1 dumps; unit-tested against a fixed-content
sample written by dump_kernels (cross-language format contract), bad-magic and
truncation cases, and an env-driven inventory gate that skips visibly.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Rust `greedy_ids` example + `greedy_parity.sh`

**Files:**
- Create: `crates/sapient-generate/examples/greedy_ids.rs`
- Modify: `crates/sapient-generate/Cargo.toml` (add an `[[example]]` entry after `duplex_spike`)
- Create: `cpp/tests/parity/greedy_parity.sh`

**Interfaces:**
- Consumes (existing Rust, verified): `sapient_generate::{Pipeline, LoadOptions, GenerationBackend, SamplingStrategy}`; `Pipeline::from_pretrained_with_opts(&str, LoadOptions)` (async), `format_chat_prompt(&[ChatMessage])`, `tokenizer().encode(&str)`, `eos_token_ids_pub()`, `generate_token_ids(&[u32], usize, &[u32], SamplingStrategy) -> Result<Vec<u32>>`, `tokenizer().decode(&[u32], bool)`, `backend_display_label()`; `sapient_tokenizers::ChatMessage::user(impl Into<String>)`; `LoadOptions { hub: sapient_hub::LoadOptions { quiet, .. }, backend, .. }`.
- Produces: **the greedy-oracle output contract** (the C++ `greedy_ids` tool in sub-project 1b must print exactly this) — five tab-separated lines to stdout:
  ```
  model\t<alias as given>
  backend\t<free text, NOT compared>
  prompt_ids\t<space-separated u32>
  output_ids\t<space-separated u32>
  output_text\t<decoded text with \\ \n \t escaped>
  ```
  Invocation contract: `<tool> <model-alias> <prompt> <max_new>`; CPU backend, greedy, chat template applied to a single user turn, stop on any EOS id. `greedy_parity.sh --rust <bin> --cpp <bin> --model <alias> [--prompt …] [--max-new N]` exits 0 iff the `prompt_ids` and `output_ids` lines are identical; `--self-check` runs the Rust tool as both sides and says so.

- [ ] **Step 1: Write the example and register it**

`crates/sapient-generate/examples/greedy_ids.rs`:

```rust
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
//! Greedy token-id oracle for the Rust→C++ parity harness (TEST-ONLY; not part of the product).
//!
//! Loads a catalog model on the CPU backend, applies the chat template to one user turn,
//! greedy-decodes up to `max_new` tokens, and prints five tab-separated lines:
//!   model, backend, prompt_ids, output_ids, output_text
//! `cpp/tests/parity/greedy_parity.sh` diffs the `prompt_ids`/`output_ids` lines against the
//! C++ tool that ships with the same contract (sub-project 1b).
//!
//! Usage: cargo run --release -p sapient-generate --example greedy_ids -- <model-alias> "<prompt>" <max_new>

use sapient_generate::{GenerationBackend, LoadOptions, Pipeline, SamplingStrategy};
use sapient_tokenizers::ChatMessage;

fn join(ids: &[u32]) -> String {
    ids.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\n', "\\n").replace('\t', "\\t")
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        eprintln!("usage: greedy_ids <model-alias> <prompt> <max_new>");
        std::process::exit(2);
    }
    let (model, prompt) = (&args[0], &args[1]);
    let max_new: usize = args[2].parse()?;

    let mut opts = LoadOptions::default();
    opts.backend = GenerationBackend::Cpu;
    opts.hub.quiet = true;
    let pipeline = Pipeline::from_pretrained_with_opts(model, opts).await?;

    let text = pipeline.format_chat_prompt(&[ChatMessage::user(prompt.clone())])?;
    let prompt_ids = pipeline.tokenizer().encode(&text)?;
    let stop_ids = pipeline.eos_token_ids_pub();
    let out = pipeline.generate_token_ids(&prompt_ids, max_new, &stop_ids, SamplingStrategy::Greedy)?;
    let decoded = pipeline.tokenizer().decode(&out, true)?;

    println!("model\t{model}");
    println!("backend\t{}", pipeline.backend_display_label());
    println!("prompt_ids\t{}", join(&prompt_ids));
    println!("output_ids\t{}", join(&out));
    println!("output_text\t{}", escape(&decoded));
    Ok(())
}
```

Append to `crates/sapient-generate/Cargo.toml` (after the `duplex_spike` `[[example]]` block):

```toml
# Rust→C++ parity oracle (test-only): greedy token ids under a fixed output contract.
[[example]]
name = "greedy_ids"
path = "examples/greedy_ids.rs"
```

- [ ] **Step 2: Run it (downloads ~100 MB on first use) and check the contract**

```bash
cargo run --release -q -p sapient-generate --example greedy_ids -- smollm2-135m-q4 "Name three planets." 16 | cut -c1-80
```

Expected: exactly five lines starting `model`, `backend`, `prompt_ids`, `output_ids`, `output_text`; `output_ids` has ≤ 16 numbers; `output_text` is coherent English (e.g. mentions planets). Run it twice and confirm identical `output_ids` (CPU greedy is deterministic).

- [ ] **Step 3: Write the parity script**

`cpp/tests/parity/greedy_parity.sh` (make executable):

```bash
#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#
# Greedy token-id parity between two `greedy_ids` tools (Rust oracle vs C++ port).
#   greedy_parity.sh --rust <bin> --cpp <bin> --model <alias> [--prompt "<text>"] [--max-new N]
#   greedy_parity.sh --rust <bin> --self-check --model <alias>      # Rust vs Rust (determinism only)
# Both tools take `<model> <prompt> <max_new>` and print `prompt_ids\t…` / `output_ids\t…` lines.
# Exit 0 only when both lines are byte-identical. A missing binary is an ERROR, never a skip.
set -euo pipefail

RUST="" CPP="" MODEL="" PROMPT="Name three planets." MAX_NEW=64 SELF=0
while [ $# -gt 0 ]; do
  case "$1" in
    --rust) RUST=$2; shift 2 ;;
    --cpp) CPP=$2; shift 2 ;;
    --model) MODEL=$2; shift 2 ;;
    --prompt) PROMPT=$2; shift 2 ;;
    --max-new) MAX_NEW=$2; shift 2 ;;
    --self-check) SELF=1; shift ;;
    *) echo "greedy_parity: unknown argument '$1'" >&2; exit 2 ;;
  esac
done
[ -n "$RUST" ] && [ -x "$RUST" ] || { echo "greedy_parity: --rust binary missing or not executable: '$RUST'" >&2; exit 2; }
[ -n "$MODEL" ] || { echo "greedy_parity: --model is required" >&2; exit 2; }
if [ "$SELF" = 1 ]; then
  CPP="$RUST"
  echo "greedy_parity: SELF-CHECK — comparing the Rust oracle against itself (determinism only, NOT C++ parity)"
fi
[ -n "$CPP" ] && [ -x "$CPP" ] || { echo "greedy_parity: --cpp binary missing or not executable: '$CPP'" >&2; exit 2; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
"$RUST" "$MODEL" "$PROMPT" "$MAX_NEW" > "$tmp/rust.txt"
"$CPP" "$MODEL" "$PROMPT" "$MAX_NEW" > "$tmp/cpp.txt"
grep -E $'^(prompt_ids|output_ids)\t' "$tmp/rust.txt" > "$tmp/rust.ids" || true   # no match → the count check below reports it
grep -E $'^(prompt_ids|output_ids)\t' "$tmp/cpp.txt" > "$tmp/cpp.ids" || true
[ "$(wc -l < "$tmp/rust.ids")" -eq 2 ] || { echo "greedy_parity: Rust output lacks prompt_ids/output_ids lines" >&2; cat "$tmp/rust.txt" >&2; exit 2; }
[ "$(wc -l < "$tmp/cpp.ids")" -eq 2 ] || { echo "greedy_parity: C++ output lacks prompt_ids/output_ids lines" >&2; cat "$tmp/cpp.txt" >&2; exit 2; }

n=$(awk -F'\t' '/^output_ids/ { print split($2, a, " ") }' "$tmp/rust.ids")
if cmp -s "$tmp/rust.ids" "$tmp/cpp.ids"; then
  echo "greedy_parity: PARITY OK  model=$MODEL output_tokens=$n prompt=\"$PROMPT\""
  exit 0
fi

echo "greedy_parity: PARITY FAIL  model=$MODEL" >&2
diff "$tmp/rust.ids" "$tmp/cpp.ids" >&2 || true
awk -F'\t' '/^output_ids/ { print $2 }' "$tmp/rust.ids" | tr ' ' '\n' > "$tmp/r"
awk -F'\t' '/^output_ids/ { print $2 }' "$tmp/cpp.ids" | tr ' ' '\n' > "$tmp/c"
idx=$(paste "$tmp/r" "$tmp/c" | awk -F'\t' '$1 != $2 { print NR - 1; exit }')
[ -n "$idx" ] && echo "greedy_parity: first divergent output token index: $idx" >&2
exit 1
```

- [ ] **Step 4: Exercise the script — self-check passes, missing binary errors, mismatch fails**

```bash
chmod +x cpp/tests/parity/greedy_parity.sh
cpp/tests/parity/greedy_parity.sh --self-check --rust target/release/examples/greedy_ids --model smollm2-135m-q4 --max-new 32
```

Expected: the SELF-CHECK banner, then `greedy_parity: PARITY OK  model=smollm2-135m-q4 output_tokens=<n> …`, exit 0.

```bash
cpp/tests/parity/greedy_parity.sh --rust target/release/examples/greedy_ids --cpp /nonexistent --model smollm2-135m-q4; echo "exit=$?"
```

Expected: `--cpp binary missing…`, `exit=2` (an error, not a skip).

```bash
printf '#!/bin/sh\nprintf "prompt_ids\\t1 2 3\\noutput_ids\\t9 9 9\\n"\n' > /tmp/fake_ids && chmod +x /tmp/fake_ids
cpp/tests/parity/greedy_parity.sh --rust target/release/examples/greedy_ids --cpp /tmp/fake_ids --model smollm2-135m-q4 --max-new 4; echo "exit=$?"
```

Expected: `PARITY FAIL`, a diff, `first divergent output token index: 0`, `exit=1`.

- [ ] **Step 5: Rust lint gates, then commit**

```bash
cargo fmt --all && cargo clippy -p sapient-generate --all-targets -- -D warnings
git add crates/sapient-generate/examples/greedy_ids.rs crates/sapient-generate/Cargo.toml cpp/tests/parity/greedy_parity.sh
git commit -m "test(oracle): greedy_ids example + greedy_parity.sh

Fixed five-line output contract for greedy token ids (CPU, chat template,
EOS stop). The parity script diffs Rust vs C++ and errors on a missing
binary; --self-check runs the oracle against itself until the C++ tool
lands in sub-project 1b.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Lint & sync gates — clang-format, clang-tidy, SPDX check, shader copies + sync check

**Files:**
- Create: `cpp/.clang-format`, `cpp/.clang-tidy`
- Create: `cpp/scripts/check_spdx.py`, `cpp/scripts/shader_sync.py`
- Create: `cpp/libs/sapient-backends-wgpu/shaders/*.wgsl` (20 copies)
- Modify: `cpp/CMakeLists.txt` (register the two scripts as ctests)

**Interfaces:**
- Consumes: `crates/sapient-backends/wgpu/src/shaders/*.wgsl` (20 files, each already carrying the `//` SPDX header).
- Produces: `python3 cpp/scripts/check_spdx.py <root>` (exit 1 + list on violations; usable on `cpp/` AND `crates/`), `python3 cpp/scripts/shader_sync.py <repo-root>` (exit 1 on missing/extra/differing shader), ctest names `lint.spdx_headers`, `lint.shader_sync`; formatting/tidy configs used by Task 7's CI and pre-push.

- [ ] **Step 1: Write the SPDX gate**

`cpp/scripts/check_spdx.py` (make executable):

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
"""Fail if any source file under <root> lacks the two-line SAPIENT SPDX header.

`//` header: .hpp .cpp .h .c .mm .m .wgsl .metal .rs .swift .kt .ts .tsx
`#`  header: .cmake .py .sh .clang-format .clang-tidy CMakeLists.txt   (a shebang may precede it)
Skipped: build/, third_party/, _deps/, fixtures/, node_modules/, .git/, *.json, *.md
Usage: check_spdx.py <root> [<root> ...]
"""
import pathlib
import sys

SLASH = (
    "// SPDX-License-Identifier: AGPL-3.0-only",
    "// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)",
)
HASH = tuple(line.replace("//", "#", 1) for line in SLASH)
SLASH_EXT = {".hpp", ".cpp", ".h", ".c", ".mm", ".m", ".wgsl", ".metal", ".rs", ".swift", ".kt", ".ts", ".tsx"}
HASH_EXT = {".cmake", ".py", ".sh"}
HASH_NAMES = {"CMakeLists.txt", ".clang-format", ".clang-tidy"}
SKIP_DIRS = {"build", "third_party", "_deps", "fixtures", "node_modules", ".git", "target"}


def expected_header(path: pathlib.Path):
    if any(part in SKIP_DIRS for part in path.parts):
        return None
    if path.name in HASH_NAMES or path.suffix in HASH_EXT:
        return HASH
    if path.suffix in SLASH_EXT:
        return SLASH
    return None


def check_root(root: str):
    bad = []
    for path in sorted(pathlib.Path(root).rglob("*")):
        if not path.is_file():
            continue
        expected = expected_header(path)
        if expected is None:
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if lines and lines[0].startswith("#!"):
            lines = lines[1:]
        if tuple(lines[:2]) != expected:
            bad.append(path)
    return bad


def main(roots):
    bad = [p for root in roots for p in check_root(root)]
    for p in bad:
        print(f"check_spdx: missing/incorrect SPDX header: {p}", file=sys.stderr)
    print(f"check_spdx: {'FAIL' if bad else 'OK'} ({len(bad)} file(s) without the header; roots: {', '.join(roots)})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["."]))
```

- [ ] **Step 2: Run it on both trees — expect OK on `crates/`, and see it catch a violation**

```bash
python3 cpp/scripts/check_spdx.py cpp crates
```

Expected: `check_spdx: OK (0 file(s) without the header; roots: cpp, crates)` — this is the first time the Rust tree's 100 % header compliance is machine-checked. If it lists any `crates/` file, add the two header lines to that file (a comment-only change is within the "frozen Rust" rule) and mention it in the commit message; do not add exclusions.

```bash
printf 'int x;\n' > cpp/libs/sapient-core/src/bad.cpp && python3 cpp/scripts/check_spdx.py cpp; echo "exit=$?"; rm cpp/libs/sapient-core/src/bad.cpp
```

Expected: `check_spdx: missing/incorrect SPDX header: cpp/libs/sapient-core/src/bad.cpp`, `FAIL (1 …)`, `exit=1`.

- [ ] **Step 3: Copy the shaders and write the sync gate**

```bash
mkdir -p cpp/libs/sapient-backends-wgpu/shaders
cp crates/sapient-backends/wgpu/src/shaders/*.wgsl cpp/libs/sapient-backends-wgpu/shaders/
ls cpp/libs/sapient-backends-wgpu/shaders | wc -l
```

Expected: `20`.

`cpp/scripts/shader_sync.py` (make executable):

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
"""Byte-compare the WGSL shader copies under cpp/ with the Rust originals under crates/.

The C++ wgpu engine (sub-project 4) runs the same 20 shaders through wgpu-native; until the
Rust tree is removed (sub-project 9) the two copies must stay identical.
Usage: shader_sync.py <repo-root>
"""
import filecmp
import pathlib
import sys

RUST = "crates/sapient-backends/wgpu/src/shaders"
CPP = "cpp/libs/sapient-backends-wgpu/shaders"


def main(repo: str) -> int:
    root = pathlib.Path(repo)
    rust, cpp = root / RUST, root / CPP
    if not rust.is_dir():
        print("shader_sync: Rust shader directory absent (post-removal) — nothing to compare")
        return 0
    r = {p.name for p in rust.glob("*.wgsl")}
    c = {p.name for p in cpp.glob("*.wgsl")}
    problems = [f"missing in cpp/: {n}" for n in sorted(r - c)]
    problems += [f"extra in cpp/ (no Rust original): {n}" for n in sorted(c - r)]
    problems += [f"differs: {n}" for n in sorted(r & c) if not filecmp.cmp(rust / n, cpp / n, shallow=False)]
    for p in problems:
        print(f"shader_sync: {p}", file=sys.stderr)
    print(f"shader_sync: {'FAIL' if problems else 'OK'} ({len(r & c)} shader(s) identical, {len(problems)} problem(s))")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
```

```bash
python3 cpp/scripts/shader_sync.py .
echo "// drift" >> cpp/libs/sapient-backends-wgpu/shaders/rope.wgsl && python3 cpp/scripts/shader_sync.py .; echo "exit=$?"; git checkout -- cpp/libs/sapient-backends-wgpu/shaders/rope.wgsl 2>/dev/null || cp crates/sapient-backends/wgpu/src/shaders/rope.wgsl cpp/libs/sapient-backends-wgpu/shaders/rope.wgsl
python3 cpp/scripts/shader_sync.py .
```

Expected: `OK (20 shader(s) identical, 0 problem(s))`; then `differs: rope.wgsl` + `FAIL`, `exit=1`; then `OK` again.

- [ ] **Step 4: Formatting and tidy configs**

`cpp/.clang-format`:

```yaml
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
# Mirrors rustfmt's look (4-space indent, 100 columns) so Rust↔C++ diffs read side by side.
BasedOnStyle: LLVM
Standard: c++20
IndentWidth: 4
ColumnLimit: 100
AccessModifierOffset: -4
AllowShortFunctionsOnASingleLine: Inline
AllowShortIfStatementsOnASingleLine: WithoutElse
BinPackArguments: false
BinPackParameters: false
BreakBeforeBraces: Attach
DerivePointerAlignment: false
PointerAlignment: Left
IncludeBlocks: Regroup
SortIncludes: CaseSensitive
IncludeCategories:
  - Regex: '^<gtest/'
    Priority: 1
  - Regex: '^<[a-z_]+>$'
    Priority: 2
  - Regex: '^"sapient/'
    Priority: 4
  - Regex: '.*'
    Priority: 3
```

`cpp/.clang-tidy`:

```yaml
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
# The C++ analogue of `cargo clippy -- -D warnings`: a curated, low-noise set, all errors.
Checks: >
  -*,
  bugprone-*,
  -bugprone-easily-swappable-parameters,
  performance-*,
  modernize-use-nullptr,
  modernize-use-override,
  modernize-use-using,
  readability-container-size-empty,
  readability-redundant-string-cstr,
  misc-unused-using-decls
WarningsAsErrors: '*'
HeaderFilterRegex: '.*/cpp/(libs|apps|ffi)/.*'
FormatStyle: file
```

Format the tree and check tidy runs clean (clang-format/clang-tidy from Xcode CLT or `brew install llvm`):

```bash
git ls-files -- 'cpp/*.hpp' 'cpp/*.cpp' | xargs clang-format -i
git diff --stat   # commit any reflow the formatter made
cd cpp && cmake --preset dev >/dev/null && run-clang-tidy -p build/dev -quiet libs/ 2>&1 | tail -3
```

Expected: `run-clang-tidy` prints no `error:` lines (only its "Enabled checks" banner or nothing). If `run-clang-tidy` is not on PATH on this Mac use `$(brew --prefix llvm)/bin/run-clang-tidy`.

- [ ] **Step 5: Register both scripts as ctests**

Append to `cpp/CMakeLists.txt` (after the `sapient-testing` subdirectory block):

```cmake
if(SAPIENT_BUILD_TESTS)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_test(NAME lint.spdx_headers
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_spdx.py ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME lint.shader_sync
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/shader_sync.py ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()
```

```bash
cd cpp && cmake --preset dev && ctest --preset dev
```

Expected: `100% tests passed` over 10 tests (8 GoogleTests incl. the visible skip + `lint.spdx_headers` + `lint.shader_sync`).

- [ ] **Step 6: Commit**

```bash
chmod +x cpp/scripts/check_spdx.py cpp/scripts/shader_sync.py
git add cpp
git commit -m "cpp(sp0): lint gates — SPDX header check, WGSL shader-sync, clang-format/tidy

check_spdx.py machine-checks the header invariant on cpp/ and crates/ (it was
convention-only before); shader_sync.py keeps the 20 WGSL copies byte-identical
to the Rust originals until crates/ is removed.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Third-party licence inventory, `NOTICE`, and the parity ledger

**Files:**
- Create: `cpp/third_party/LICENSES.md`
- Modify: `NOTICE` (the sentence at lines 61–63 naming the dependency manifests)
- Create: `docs/PARITY.md`

**Interfaces:**
- Consumes: spec §D4 library map; Task 2/3/4 results for the first ledger rows.
- Produces: the inventory every later `cmake/deps.cmake` addition must extend; the ledger every later sub-project appends to (spec "Definition of done").

- [ ] **Step 1: Write the licence inventory**

`cpp/third_party/LICENSES.md`:

```markdown
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
```

- [ ] **Step 2: Update `NOTICE`**

Replace the sentence spanning `NOTICE` lines 61–63:

```
This product bundles third-party open-source components, each distributed under
its own license. Those licenses are listed in the dependency manifests
(Cargo.toml / Cargo.lock) and retained in the respective upstream sources.
```

with:

```
This product bundles third-party open-source components, each distributed under
its own license. Those licenses are listed in the dependency manifests — the Rust
tree's Cargo.toml / Cargo.lock and the C++ tree's cpp/third_party/LICENSES.md —
and retained in the respective upstream sources.
```

- [ ] **Step 3: Create the parity ledger**

`docs/PARITY.md`:

```markdown
# Parity ledger — Rust oracle vs C++ port

Every sub-project of the C++ rewrite (spec: `docs/superpowers/specs/2026-09-20-cpp-rewrite-design.md`,
§D5) records its gate results here. A row is only added after the gate ran on real hardware; a gate
that could not run is recorded as such, never as a pass. Commit hashes refer to this repository.

Conventions: **bit-identical** = byte-equal output; **token-identical** = equal `prompt_ids` and
`output_ids` from `cpp/tests/parity/greedy_parity.sh`; **max_err** = largest absolute difference.

## Harness self-validation (sub-project 0)

| Date | Gate | Host / ISA path | Rust commit | C++ commit | Result |
|---|---|---|---|---|---|
| YYYY-MM-DD | `.sapd` format round-trip: Rust `dump_kernels --format-sample` → C++ `sapient::testing::read_golden` (`Golden.ReadsFormatSampleWrittenByRust`) | macOS arm64 | `<sha>` | `<sha>` | pass — 7 arrays decoded exactly |
| YYYY-MM-DD | Kernel dump determinism: two `dump_kernels --out` runs, same seed | macOS arm64 (NEON/SDOT) | `<sha>` | — | bit-identical, 22 cases |
| YYYY-MM-DD | `Golden.InventoryFromEnv` over the 22 host dumps | macOS arm64 | `<sha>` | `<sha>` | pass |
| YYYY-MM-DD | `greedy_parity.sh --self-check` smollm2-135m-q4, 32 tokens | macOS arm64 | `<sha>` | — (Rust vs Rust) | token-identical |

## Sub-project 1a — core + IO + CPU kernels

_(no rows yet)_

## Sub-project 1b — CPU chat vertical slice

_(no rows yet)_
```

Fill the `YYYY-MM-DD` / `<sha>` placeholders with today's date and `git rev-parse --short HEAD` after the commits from Tasks 2–4 (the Rust and C++ commits differ; both are on this branch).

- [ ] **Step 4: Commit**

```bash
git add cpp/third_party/LICENSES.md NOTICE docs/PARITY.md
git commit -m "docs(sp0): third-party licence inventory, NOTICE manifest pointer, parity ledger

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: CI jobs, pre-push hook, justfile recipes

**Files:**
- Modify: `.github/workflows/ci.yml` (append five jobs at the end of `jobs:`)
- Modify: `.githooks/pre-push` (add a clang-format dry-run block before the final `echo "pre-push: lint clean ✓"`)
- Modify: `justfile` (append C++ recipes at the end)

**Interfaces:**
- Consumes: presets `ci-macos`/`ci-linux`/`ci-windows` (Task 1), `golden_dump.sh` + `SAPIENT_GOLDEN_DIR` (Task 3), `greedy_parity.sh --self-check` + `target/release/examples/greedy_ids` (Task 4), lint scripts (Task 5).
- Produces: CI job ids `cpp-lint`, `cpp-test-macos`, `cpp-test-linux`, `cpp-build-windows`, `cpp-parity` (matrix macos-14 + ubuntu-latest); `just cpp-configure|cpp-build|cpp-test|cpp-fmt|cpp-lint|cpp-golden`.

- [ ] **Step 1: Append the CI jobs**

Append to `.github/workflows/ci.yml` (same two-space indentation as the existing jobs):

```yaml
  # ── C++ tree (cpp/) — the Rust→C++ parity port ───────────────────────────────
  cpp-lint:
    name: C++ lint (format, tidy, SPDX, shader sync)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: sudo apt-get update -q && sudo apt-get install -y ninja-build clang-18 clang-tidy-18 clang-format-18
      - name: clang-format (dry run)
        run: git ls-files -- 'cpp/*.hpp' 'cpp/*.cpp' | xargs clang-format-18 --dry-run --Werror
      - name: SPDX headers (cpp/ and crates/)
        run: python3 cpp/scripts/check_spdx.py cpp crates
      - name: Shader copies match Rust originals
        run: python3 cpp/scripts/shader_sync.py .
      - name: clang-tidy
        working-directory: cpp
        run: |
          cmake --preset ci-linux -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18
          run-clang-tidy-18 -p build/ci-linux -quiet libs/

  cpp-test-macos:
    name: C++ build+test (macOS arm64)
    runs-on: macos-14
    steps:
      - uses: actions/checkout@v4
      - run: brew install ninja
      - working-directory: cpp
        run: cmake --preset ci-macos && cmake --build --preset ci-macos && ctest --preset ci-macos

  cpp-test-linux:
    name: C++ build+test (Linux x86_64, clang)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: sudo apt-get update -q && sudo apt-get install -y ninja-build clang
      - working-directory: cpp
        run: cmake --preset ci-linux && cmake --build --preset ci-linux && ctest --preset ci-linux

  cpp-build-windows:
    name: C++ build+test (Windows clang-cl)
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: ilammy/msvc-dev-cmd@v1
      - run: choco install ninja -y
      - working-directory: cpp
        run: cmake --preset ci-windows && cmake --build --preset ci-windows && ctest --preset ci-windows

  # Builds BOTH languages on the same host: regenerates the kernel golden dumps there, runs the
  # C++ tests against them, and greedy-diffs token ids. Until the C++ greedy tool lands
  # (sub-project 1b) the token check is an explicit Rust-vs-Rust self-check.
  cpp-parity:
    name: Parity (${{ matrix.os }})
    strategy:
      fail-fast: false
      matrix:
        include:
          - { os: macos-14, preset: ci-macos }
          - { os: ubuntu-latest, preset: ci-linux }
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - name: Install ninja + clang (Linux)
        if: runner.os == 'Linux'
        run: sudo apt-get update -q && sudo apt-get install -y ninja-build clang
      - name: Install ninja (macOS)
        if: runner.os == 'macOS'
        run: brew install ninja
      - name: Build the Rust oracle tools
        run: |
          cargo build --release -p sapient-backends-cpu --example dump_kernels
          cargo build --release -p sapient-generate --example greedy_ids
      - name: Kernel golden dumps (this host)
        run: cpp/tests/parity/golden_dump.sh "$RUNNER_TEMP/golden"
      - name: Configure + build C++
        working-directory: cpp
        run: cmake --preset ${{ matrix.preset }} && cmake --build --preset ${{ matrix.preset }}
      - name: C++ tests against the dumps
        working-directory: cpp
        env:
          SAPIENT_GOLDEN_DIR: ${{ runner.temp }}/golden
        run: ctest --preset ${{ matrix.preset }}
      - name: Cache the parity model
        uses: actions/cache@v4
        with:
          path: ~/.cache/huggingface
          key: hf-smollm2-135m-q4-${{ runner.os }}
      - name: Greedy token parity (Rust self-check until the C++ tool exists)
        run: cpp/tests/parity/greedy_parity.sh --self-check --rust target/release/examples/greedy_ids --model smollm2-135m-q4 --max-new 32
```

Validate the YAML locally:

```bash
python3 -c "import yaml,sys; d=yaml.safe_load(open('.github/workflows/ci.yml')); print(sorted(k for k in d['jobs'] if k.startswith('cpp-')))"
```

Expected: `['cpp-build-windows', 'cpp-lint', 'cpp-parity', 'cpp-test-linux', 'cpp-test-macos']`. (If PyYAML is missing: `pip3 install --user pyyaml`.)

- [ ] **Step 2: Extend the pre-push hook**

Insert into `.githooks/pre-push` immediately before the line `echo "pre-push: lint clean ✓"`:

```bash
# C++ tree: clang-format dry run (only when clang-format is installed).
if [ -d cpp ] && command -v clang-format >/dev/null 2>&1; then
  echo "pre-push: clang-format --dry-run over cpp/…"
  cpp_files=$(git ls-files -- 'cpp/*.hpp' 'cpp/*.cpp' 'cpp/*.h' 'cpp/*.mm')
  if [ -n "$cpp_files" ] && ! echo "$cpp_files" | xargs clang-format --dry-run --Werror; then
    echo "" >&2
    echo "pre-push: clang-format violations in cpp/ — run 'just cpp-fmt', commit, then push again." >&2
    exit 1
  fi
fi
```

Test the hook without pushing:

```bash
CI= SKIP_LINT= bash .githooks/pre-push; echo "exit=$?"
```

Expected: cargo fmt/clippy run as before, then `pre-push: clang-format --dry-run over cpp/…`, `pre-push: lint clean ✓`, `exit=0`.

- [ ] **Step 3: Append justfile recipes**

Append to `justfile`:

```make
# ── C++ tree (cpp/) — Rust→C++ parity port ─────────────────────────────────────
# Configure the C++ tree (presets: dev, release, ci-macos, ci-linux, ci-windows)
cpp-configure preset="dev":
    cd cpp && cmake --preset {{preset}}

# Build the C++ tree
cpp-build preset="dev":
    cd cpp && cmake --build --preset {{preset}}

# Run the C++ tests (set SAPIENT_GOLDEN_DIR to include the kernel golden gate)
cpp-test preset="dev":
    cd cpp && ctest --preset {{preset}}

# Format the C++ sources in place
cpp-fmt:
    git ls-files -- 'cpp/*.hpp' 'cpp/*.cpp' 'cpp/*.h' 'cpp/*.mm' | xargs clang-format -i

# Lint gates that need no build: SPDX headers (both trees) + WGSL shader sync
cpp-lint:
    python3 cpp/scripts/check_spdx.py cpp crates
    python3 cpp/scripts/shader_sync.py .

# Regenerate the kernel golden dumps on this host with the Rust oracle
cpp-golden out="/tmp/sapient-golden":
    cpp/tests/parity/golden_dump.sh {{out}}
```

```bash
just cpp-lint && just cpp-configure && just cpp-build && just cpp-test
```

Expected: both lint scripts print `OK`; configure/build/test succeed (10 tests, 1 visible skip).

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml .githooks/pre-push justfile
git commit -m "ci(sp0): C++ lint/test/parity jobs, pre-push clang-format, just cpp-* recipes

cpp-parity builds both languages per host (macos-14 arm64, ubuntu x86_64),
regenerates kernel golden dumps there, runs ctest against them, and greedy
self-checks the Rust oracle on smollm2-135m-q4.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Docs, spec amendment, changelog

**Files:**
- Modify: `CLAUDE.md` (Essential commands block + the "C++ rewrite programme" section), `CONTRIBUTING.md` ("C++ rewrite programme" section), `docs/PROJECT_GUIDE.md` (§6), `README.md` ("Build from Source"), `docs/ROADMAP.md` (Phase 7 row 0), `docs/superpowers/specs/2026-09-20-cpp-rewrite-design.md` (D2 comment), `CHANGELOG.md` (Unreleased entry)

**Interfaces:**
- Consumes: everything above. Produces: the documentation state sub-project 1a starts from.

- [ ] **Step 1: CLAUDE.md**

In `## Essential commands`, after the closing ``` of the existing bash block, add:

````markdown
```bash
# C++ tree (parity port, see the "C++ rewrite programme" section)
cd cpp && cmake --preset dev && cmake --build --preset dev && ctest --preset dev   # Clang + Ninja required
just cpp-lint                                   # SPDX headers (cpp/ AND crates/) + WGSL shader sync
just cpp-golden /tmp/sapient-golden             # Rust oracle → kernel golden dumps for this host
SAPIENT_GOLDEN_DIR=/tmp/sapient-golden just cpp-test
cargo run --release -p sapient-generate --example greedy_ids -- smollm2-135m-q4 "Hi" 16   # greedy oracle
cpp/tests/parity/greedy_parity.sh --self-check --rust target/release/examples/greedy_ids --model smollm2-135m-q4
```
````

In the `## C++ rewrite programme` section, replace the bullet sentence `Rust behaviour is frozen meanwhile — the only Rust addition allowed is the test-only kernel golden-dump example.` with `Rust behaviour is frozen meanwhile — the only Rust additions are two test-only examples: \`crates/sapient-backends/cpu/examples/dump_kernels.rs\` (kernel golden dumps, \`.sapd\` v1) and \`crates/sapient-generate/examples/greedy_ids.rs\` (greedy token-id oracle with a fixed five-line output contract).` and append this bullet:

```markdown
- **Sub-project 0 landed (scaffold + oracle harness):** `cpp/` builds `sapient::core` (version) and the test-support `sapient::testing` (`.sapd` golden reader); gates = `Golden.*` GoogleTests, `lint.spdx_headers`, `lint.shader_sync`, `BuildFlags.FpContractIsOff` (fails if the compiler contracts `a*b+c` into an FMA); CI jobs `cpp-lint`, `cpp-test-{macos,linux}`, `cpp-build-windows`, `cpp-parity` (both hosts: Rust builds dumps → C++ consumes; greedy `--self-check` until 1b). Golden dumps are host-specific and never committed; the one committed fixture is the fixed-content `format_sample.sapd`. Ledger: `docs/PARITY.md`.
```

- [ ] **Step 2: CONTRIBUTING.md**

Replace the last bullet of the `## C++ rewrite programme (in progress)` section (`The build/test/lint sections above describe the Rust tree; the C++ equivalents … are documented as sub-project 0 lands.`) with:

````markdown
- **C++ build/test/lint (sub-project 0 is in):** prerequisites are CMake ≥ 3.24, Ninja and **Clang** (Apple clang, clang ≥ 16, or clang-cl on Windows — GCC/MSVC are refused unless `-DSAPIENT_ALLOW_NON_CLANG=ON`, and such builds are not parity builds). Then:

  ```bash
  cd cpp && cmake --preset dev && cmake --build --preset dev && ctest --preset dev
  just cpp-fmt      # clang-format in place (CI runs --dry-run --Werror; the pre-push hook does too)
  just cpp-lint     # SPDX header gate on cpp/ and crates/, WGSL shader-sync gate
  ```

  Kernel golden gates need dumps from the Rust oracle **made on your machine**: `just cpp-golden /tmp/sapient-golden` then `SAPIENT_GOLDEN_DIR=/tmp/sapient-golden just cpp-test`. Never commit dump files. Every new C++ source file carries the SPDX header (the `lint.spdx_headers` ctest fails otherwise), lives under `cpp/libs/sapient-<crate>/` mirroring the Rust module it ports, and every parity result goes in `docs/PARITY.md`.
````

- [ ] **Step 3: PROJECT_GUIDE.md §6**

Append at the end of `## 6. How to build and run it yourself` (before `## 7.`):

```markdown
### The C++ tree (in progress)

The C++ port lives in `cpp/` and builds with CMake + Ninja + Clang: `cd cpp && cmake --preset dev && cmake --build --preset dev && ctest --preset dev`. Today it contains only the scaffolding and the "answer-key" machinery: a reader for the kernel test data the Rust build writes (`just cpp-golden`), header/shader lint gates, and CI jobs that build both languages on the same machine and compare them. Real inference code arrives with sub-projects 1a/1b (see `docs/ROADMAP.md` Phase 7 and `docs/PARITY.md`).
```

- [ ] **Step 4: README.md**

In `## Build from Source`, after the existing cargo instructions, add:

````markdown
**C++ tree (parity port, in progress):**

```bash
cd cpp && cmake --preset dev && cmake --build --preset dev && ctest --preset dev   # CMake ≥ 3.24, Ninja, Clang
```
````

- [ ] **Step 5: ROADMAP.md, spec, CHANGELOG**

In `docs/ROADMAP.md` Phase 7 table, change row 0's status cell from `planned` to `✅ done (branch feat/cpp-sp0-scaffold)`.

In the spec `docs/superpowers/specs/2026-09-20-cpp-rewrite-design.md`, D2 layout comment `crates/  ← Rust oracle, untouched except the dump example` → `crates/  ← Rust oracle, untouched except two test-only examples (dump_kernels, greedy_ids)`; and in D1's Kernel row, after `writes seeded-random inputs + outputs for each quant/matmul/attention/rope kernel` add ` (22 cases through the public entry points; \`.sapd\` v1 format documented in the example header)`.

In `CHANGELOG.md`, above `## [0.6.0] - 2026-07-14`, add:

```markdown
## [Unreleased]

### 🧱 C++ rewrite — sub-project 0 (scaffold + oracle harness)
- `cpp/` CMake tree (C++20, Clang-only, `-ffp-contract=off`), GoogleTest, presets, CI jobs `cpp-*`.
- Rust→C++ oracle harness: test-only `dump_kernels` (`.sapd` golden dumps) and `greedy_ids` examples, `sapient::testing` reader, `greedy_parity.sh`, SPDX and WGSL shader-sync gates, `docs/PARITY.md` ledger.
- No user-facing behaviour change; the Rust binaries are unaffected.
```

- [ ] **Step 6: Final verification of the whole sub-project, then commit**

```bash
just cpp-lint && cd cpp && cmake --preset dev && cmake --build --preset dev && ./tests/parity/golden_dump.sh /tmp/sapient-golden >/dev/null && SAPIENT_GOLDEN_DIR=/tmp/sapient-golden ctest --preset dev && cd .. && cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test -p sapient-backends-cpu -p sapient-generate -- --test-threads=1 2>&1 | tail -3
```

Expected: both lint scripts `OK`; ctest `100% tests passed` with **0 skipped** (the inventory gate is live); fmt clean; clippy clean; the Rust suites for the two touched crates still pass.

```bash
git add CLAUDE.md CONTRIBUTING.md docs/PROJECT_GUIDE.md README.md docs/ROADMAP.md docs/superpowers/specs/2026-09-20-cpp-rewrite-design.md CHANGELOG.md
git commit -m "docs(sp0): C++ build/test/lint instructions, spec amendment, roadmap row 0 done

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

**Do NOT push.** The user pushes `feat/cpp-sp0-scaffold` and opens the PR themselves; the five new CI jobs going green on that PR is sub-project 0's gate per spec §D5. Report the branch state and stop.

---

## Self-review against the spec

- **D1 verification backbone:** golden-dump harness (Tasks 2, 3), per-host generation in CI (Task 7 `cpp-parity`), greedy token parity script with the oracle contract (Task 4), fixtures reused (nothing to do yet — Whisper/Kokoro fixtures are consumed from 5a/5b), build-flag discipline enforced + tested (Task 1), `-O1` tests (presets), determinism note validated (Task 2 step 3, Task 4 step 2). Covered.
- **D2 layout:** `cpp/` root, `cmake/`, `third_party/` (manifest only), `libs/sapient-core`, `libs/sapient-backends-wgpu/shaders`, `tests/parity/` — Task 1/3/5/6. `sapient-testing` is an addition the spec did not name; it is test-support only and lives under `libs/` with the same conventions. Covered.
- **D3 conventions:** target/alias/namespace naming, warnings-as-errors, SPDX gate as ctest and CI. Covered.
- **D5 row 0:** skeleton ✓ presets ✓ dep pins ✓ SPDX gate ✓ clang-format/tidy ✓ GoogleTest ✓ parity scripts ✓ Rust dump example ✓ CI jobs incl. Windows clang-cl and `-ffp-contract=off` ✓ licence inventory ✓ (`NOTICE` updated). Gate "CI green on an empty lib" ✓.
- **Verification section:** all five CI job names present; `cpu_repack` RESOURCE_LOCK is 1a's concern (no such test yet). `docs/PARITY.md` created with the definition-of-done format.
- **Placeholder scan:** the only `YYYY-MM-DD`/`<sha>` tokens are in `docs/PARITY.md` with an explicit instruction to fill them in Task 6 step 3.
- **Type consistency:** `GoldenDType`, `GoldenArray::{numel, as<T>}`, `GoldenCase::{find, get}`, `read_golden`, `list_golden`, `dtype_size` match between header, source and test; CMake names `sapient_core`/`sapient::core`, `sapient_testing`/`sapient::testing`, `sapient_apply_warnings`, presets `dev/release/ci-macos/ci-linux/ci-windows` are used identically in Tasks 1, 3, 5, 7, 8; the `.sapd` dtype tags 0–5 and field order match between `dump_kernels.rs` and `golden.cpp`; the greedy contract lines (`prompt_ids`, `output_ids`, tab-separated) match between `greedy_ids.rs` and `greedy_parity.sh`.
