# Sub-project 1a, Plan A: `sapient::core` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port `crates/sapient-core` to C++ (`cpp/libs/sapient-core`, namespace `sapient::core`) with every one of its 22 unit tests reproduced by name, add the single shared dequantiser and the f16/bf16 conversions the rest of 1a builds on, add the `sapient::testing` comparison helpers, and prove `Tensor::to_f32_vec` bit-identical to Rust via golden dumps.

**Architecture:** One `.hpp/.cpp` pair per Rust module (`error`, `dtype`, `shape`, `buffer`, `tensor`) plus three small files Rust spreads elsewhere (`panic.hpp`, `f16.hpp`, `dequant.hpp`). `Result<T>` is `tl::expected<T, Error>`; Rust panics become `sapient::core::panic()` (abort). `Tensor` holds `{Shape, DType, element strides, shared_ptr<Buffer>, byte offset}` exactly like Rust, including the quant-bounded / float-unbounded `bytes()` asymmetry and the `slice_axis` element-size arithmetic. Dequantisation lives once, in `dequant.hpp`, and is validated bit-for-bit against Rust's `Tensor::to_f32_vec` through new `dump_kernels` cases.

**Tech Stack:** C++20, CMake presets from sub-project 0, GoogleTest, `tl::expected` v1.1.0 (new FetchContent pin, CC0), the Rust `dump_kernels` example (test-only Rust, allowed).

**Spec:** `docs/superpowers/specs/2026-09-21-cpp-sp1a-core-io-cpu-design.md` (§2.1, §3, §4, §5 row A) under `docs/superpowers/specs/2026-09-20-cpp-rewrite-design.md`. **Porting map (line-cited; read Part A before touching a module):** `docs/superpowers/notes/2026-09-21-sp1a-porting-map-core-io.md`.

## Global Constraints

- **Branch:** `feat/cpp-sp1a` (already created; stacked on `feat/cpp-sp0-scaffold`). Commit after every task. **Never push.**
- **Compiler/flags (programme spec D1):** Clang only; `-ffp-contract=off` is applied by `cpp/libs/CMakeLists.txt` to everything under `libs/`; never add `-march=native`/`-ffast-math`. Build and test with `cd cpp && cmake --preset dev && cmake --build --preset dev && ctest --preset dev`.
- **Warnings are errors** (`sapient_apply_warnings`: `-Wall -Wextra -Wpedantic -Wshadow -Werror`). No narrowing in braces, no unused parameters, no shadowing.
- **Naming (spec D3):** target `sapient_core` / alias `sapient::core`, namespace `sapient::core`, headers under `cpp/libs/sapient-core/include/sapient/core/`, sources under `src/`, tests under `tests/` named `<module>_test.cpp` with the Rust test names.
- **SPDX header verbatim** on every new `.hpp/.cpp` (`//` form) and on `CMakeLists.txt`/`.cmake` edits (`#` form); `check_spdx.py` runs as a ctest and fails otherwise:
  `// SPDX-License-Identifier: AGPL-3.0-only`
  `// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)`
- **Bit-identity rules (spec §3):** software f16/bf16 conversions (no F16C/`vcvt`); `(d * sc) * q` left-associative exactly as Rust wrote it; `int32_t` casts before subtracting 8/16/32 from nibbles; little-endian only (`static_assert`).
- **Rust tree frozen** except `crates/sapient-backends/cpu/examples/dump_kernels.rs` (Task 8 extends it). `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets -- -D warnings` must stay clean.
- **Error messages byte-identical** to the Rust `#[error]` strings listed in the porting map §A6 (including the U+2014 em-dashes inside three of them).
- **No exceptions across library boundaries**; recoverable failures are `Result`, unrecoverable ones are `panic()`.
- **Docs rule:** Task 9 updates CLAUDE.md, docs/ROADMAP.md, docs/PARITY.md, cpp/third_party/LICENSES.md, CHANGELOG.md (CONTRIBUTING/README/PROJECT_GUIDE need no change for a library-internal plan; say so in the commit).

## File structure

```
cpp/cmake/deps.cmake                              + tl::expected v1.1.0 pin (Task 1)
cpp/libs/sapient-core/CMakeLists.txt              sources + tests per task
cpp/libs/sapient-core/include/sapient/core/
  panic.hpp          [[noreturn]] panic(msg): stderr + abort  (Task 1)
  error.hpp          ErrorCode, Error{code, fields}, to_string(), factories, Result<T>, SAPIENT_TRY (Task 1)
  dtype.hpp          DType enum, block constants, element_size/alignment/block_*/byte_count/is_*/name/from_str/onnx (Task 2)
  shape.hpp          Shape{dims}: numel/strides/reshape/broadcast_with/expand_dims/squeeze/validate/flat_index (Task 3)
  f16.hpp            f16↔f32 (exact widen, RNE narrow), bf16↔f32 (Task 4)
  buffer.hpp         Buffer (abstract), CpuBuffer (aligned zeroed alloc / moved vector), BufferHandle (Task 5)
  dequant.hpp        get_scale_min_k4, q4_0/q8_0/q4_k/q5_k/q6_k block dequant, r4 depermute (Task 6)
  tensor.hpp         Tensor + TensorMeta + F32Cow (Task 7)
cpp/libs/sapient-core/src/{error,dtype,shape,f16,buffer,dequant,tensor}.cpp
cpp/libs/sapient-core/tests/{error,dtype,shape,f16,buffer,dequant,tensor,golden_dequant}_test.cpp
cpp/libs/sapient-testing/include/sapient/testing/compare.hpp + src/compare.cpp   (Task 8)
crates/sapient-backends/cpu/examples/dump_kernels.rs                                (Task 8: 9 dequant cases)
docs: CLAUDE.md, docs/ROADMAP.md, docs/PARITY.md, cpp/third_party/LICENSES.md, CHANGELOG.md, cpp/CMakeLists.txt comment (Task 9)
```

---

### Task 1: `panic.hpp`, `error.hpp/.cpp`, `Result<T>`, `tl::expected` pin

**Files:**
- Modify: `cpp/cmake/deps.cmake` (add the pin after the googletest block)
- Modify: `cpp/libs/sapient-core/CMakeLists.txt`
- Create: `cpp/libs/sapient-core/include/sapient/core/panic.hpp`, `include/sapient/core/error.hpp`, `src/error.cpp`
- Test: `cpp/libs/sapient-core/tests/error_test.cpp`

**Interfaces:**
- Produces: `namespace sapient::core { [[noreturn]] void panic(std::string_view msg); enum class ErrorCode; struct Error { ErrorCode code; ...; std::string to_string() const; static factories }; template <class T> using Result = tl::expected<T, Error>; }` and macros `SAPIENT_TRY(expr)` (for `Result<void>` or discard) and `SAPIENT_TRY_ASSIGN(lhs, expr)`. Every later task returns `Result<T>` and builds errors only through the factories below.

- [ ] **Step 1: Write the failing tests** — `cpp/libs/sapient-core/tests/error_test.cpp`

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <string>
#include <system_error>
#include <vector>

#include "sapient/core/error.hpp"

using sapient::core::Error;
using sapient::core::ErrorCode;
using sapient::core::Result;

// Rust: error_display
TEST(Error, error_display) {
    const auto e = Error::shape_mismatch({2, 3}, {2, 4});
    const std::string s = e.to_string();
    EXPECT_NE(s.find("Shape mismatch"), std::string::npos) << s;
    EXPECT_NE(s.find("[2, 3]"), std::string::npos) << s;
    EXPECT_EQ(s, "Shape mismatch: expected [2, 3], got [2, 4]");
}

// Rust: from_io_error
TEST(Error, from_io_error) {
    const auto e = Error::io(std::make_error_code(std::errc::no_such_file_or_directory), "file missing");
    EXPECT_EQ(e.code, ErrorCode::Io);
    EXPECT_EQ(e.to_string().rfind("IO error: ", 0), 0u) << e.to_string();
}

TEST(Error, messages_match_rust_format_strings) {
    EXPECT_EQ(Error::rank_mismatch(4, 3).to_string(), "Rank mismatch: expected 4, got 3");
    EXPECT_EQ(Error::type_mismatch("a quantized dtype", "f32").to_string(),
              "Type mismatch: expected a quantized dtype, got f32");
    EXPECT_EQ(Error::broadcast({2, 3}, {2, 4}).to_string(),
              "Incompatible shapes for broadcasting: [2, 3] and [2, 4]");
    EXPECT_EQ(Error::cyclic_graph().to_string(), "Graph contains a cycle — execution is impossible");
    EXPECT_EQ(Error::node_not_found("x").to_string(), "Node \"x\" not found in graph");
    EXPECT_EQ(Error::invalid_graph("Shape has zero dimension at axis 1").to_string(),
              "Graph validation failed: Shape has zero dimension at axis 1");
    EXPECT_EQ(Error::allocation_failed(32, 64).to_string(),
              "Allocation failed: requested 32 bytes (alignment 64)");
    EXPECT_EQ(Error::buffer_size_mismatch(10, 4).to_string(),
              "Buffer size mismatch: expected 10 bytes, got 4");
    EXPECT_EQ(Error::internal("t() requires a 2-D tensor").to_string(),
              "Internal error: t() requires a 2-D tensor");
    EXPECT_EQ(Error::gguf_parse("bad GGUF magic").to_string(), "GGUF parse error: bad GGUF magic");
    EXPECT_EQ(Error::model_not_found("/x").to_string(), "Model not found at path '/x'");
}

TEST(Error, try_macro_propagates) {
    auto inner = [](bool ok) -> Result<int> {
        if (!ok) return tl::unexpected(Error::internal("boom"));
        return 7;
    };
    auto outer = [&](bool ok) -> Result<int> {
        SAPIENT_TRY_ASSIGN(int v, inner(ok));
        return v + 1;
    };
    ASSERT_TRUE(outer(true).has_value());
    EXPECT_EQ(*outer(true), 8);
    ASSERT_FALSE(outer(false).has_value());
    EXPECT_EQ(outer(false).error().to_string(), "Internal error: boom");
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd cpp && cmake --preset dev 2>&1 | tail -2`
Expected: configure FAILS with `Cannot find source file: src/error.cpp` (or, once CMake is edited but the header is missing, a compile error `'sapient/core/error.hpp' file not found`).

- [ ] **Step 3: Add the `tl::expected` pin** — append to `cpp/cmake/deps.cmake` (outside the `if(SAPIENT_BUILD_TESTS)` block):

```cmake
# tl::expected — std::expected polyfill (C++23) used as sapient::core::Result<T>. CC0-1.0.
FetchContent_Declare(tl_expected
  GIT_REPOSITORY https://github.com/TartanLlama/expected.git
  GIT_TAG        v1.1.0
  GIT_SHALLOW    TRUE)
set(EXPECTED_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(EXPECTED_BUILD_PACKAGE OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(tl_expected)
```

- [ ] **Step 4: Write `panic.hpp`**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once

#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace sapient::core {

/// The C++ twin of a Rust `panic!` under `[profile.release] panic = "abort"`: print and abort.
/// Used only where the Rust code panics (e.g. `DType::block_bytes()` on a float dtype,
/// mutating an mmap buffer). Recoverable failures use `Result<T>` instead.
[[noreturn]] inline void panic(std::string_view msg) {
    std::fprintf(stderr, "sapient panic: %.*s\n", static_cast<int>(msg.size()), msg.data());
    std::fflush(stderr);
    std::abort();
}

}  // namespace sapient::core
```

- [ ] **Step 5: Write `error.hpp`**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
//
// Port of crates/sapient-core/src/error.rs. One error type for the whole C++ tree; the 25
// variants of Rust's `SapientError` become `ErrorCode` + the fields each variant carried.
// `to_string()` reproduces the `#[error("…")]` format strings byte for byte.

#include <cstddef>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <tl/expected.hpp>

namespace sapient::core {

enum class ErrorCode : uint8_t {
    ShapeMismatch,         // {expected: Vec<usize>, got: Vec<usize>}
    RankMismatch,          // {expected: usize, got: usize}
    TypeMismatch,          // {expected: String, got: String}
    BroadcastError,        // {lhs: Vec<usize>, rhs: Vec<usize>}
    CyclicGraph,
    NodeNotFound,          // (String)
    InvalidGraph,          // (String)
    ShapeInferenceFailed,  // {op, reason}
    UnsupportedOp,         // {backend, op}
    BackendError,          // {backend, message}
    NoBackendAvailable,
    AllocationFailed,      // {bytes, align}
    BufferSizeMismatch,    // {expected, got}
    PoolExhausted,
    OnnxParseError,        // (String)
    GgufParseError,        // (String)
    SafetensorsParseError, // (String)
    UnsupportedFormat,     // (String)
    ModelNotFound,         // (String)
    Io,                    // (std::io::Error) — stored as its message
    DeadlineExceeded,
    SchedulerShutdown,
    UninitializedRuntime,
    TelemetryError,        // (String)
    Internal,              // (String)
};

struct Error {
    ErrorCode code{ErrorCode::Internal};
    std::vector<size_t> expected_dims;  // ShapeMismatch.expected / BroadcastError.lhs
    std::vector<size_t> got_dims;       // ShapeMismatch.got / BroadcastError.rhs
    size_t expected_n{0};               // RankMismatch.expected / AllocationFailed.bytes / BufferSizeMismatch.expected
    size_t got_n{0};                    // RankMismatch.got / AllocationFailed.align / BufferSizeMismatch.got
    std::string a;                      // first string field (TypeMismatch.expected, ShapeInferenceFailed.op, UnsupportedOp.backend, BackendError.backend, single-string variants)
    std::string b;                      // second string field (TypeMismatch.got, ShapeInferenceFailed.reason, UnsupportedOp.op, BackendError.message)
    std::error_code io_code;            // Io only

    /// Byte-identical to Rust's `Display` (thiserror `#[error]` strings).
    std::string to_string() const;

    static Error shape_mismatch(std::vector<size_t> expected, std::vector<size_t> got);
    static Error rank_mismatch(size_t expected, size_t got);
    static Error type_mismatch(std::string expected, std::string got);
    static Error broadcast(std::vector<size_t> lhs, std::vector<size_t> rhs);
    static Error cyclic_graph();
    static Error node_not_found(std::string node);
    static Error invalid_graph(std::string msg);
    static Error shape_inference_failed(std::string op, std::string reason);
    static Error unsupported_op(std::string backend, std::string op);
    static Error backend(std::string backend, std::string message);
    static Error no_backend_available();
    static Error allocation_failed(size_t bytes, size_t align);
    static Error buffer_size_mismatch(size_t expected, size_t got);
    static Error pool_exhausted();
    static Error onnx_parse(std::string msg);
    static Error gguf_parse(std::string msg);
    static Error safetensors_parse(std::string msg);
    static Error unsupported_format(std::string fmt);
    static Error model_not_found(std::string path);
    /// Rust `From<std::io::Error>`: `message` is what `e.to_string()` would have printed.
    static Error io(std::error_code ec, std::string message);
    static Error deadline_exceeded();
    static Error scheduler_shutdown();
    static Error uninitialized_runtime();
    static Error telemetry(std::string msg);
    static Error internal(std::string msg);
};

template <class T>
using Result = tl::expected<T, Error>;

/// Rust's `Debug` for `Vec<usize>`: "[2, 3]" ("[]" when empty).
std::string debug_dims(const std::vector<size_t>& dims);

}  // namespace sapient::core

#define SAPIENT_CAT_(a, b) a##b
#define SAPIENT_CAT(a, b) SAPIENT_CAT_(a, b)

/// `expr?` for a Result whose value is discarded (or Result<void>).
#define SAPIENT_TRY(expr)                                                       \
    do {                                                                        \
        auto&& SAPIENT_CAT(sapient_try_, __LINE__) = (expr);                    \
        if (!SAPIENT_CAT(sapient_try_, __LINE__).has_value())                   \
            return ::tl::unexpected(std::move(SAPIENT_CAT(sapient_try_, __LINE__).error())); \
    } while (0)

/// `lhs = expr?;` — `lhs` may be a declaration (`int v`) or an existing lvalue.
#define SAPIENT_TRY_ASSIGN(lhs, expr)                                           \
    auto&& SAPIENT_CAT(sapient_try_, __LINE__) = (expr);                        \
    if (!SAPIENT_CAT(sapient_try_, __LINE__).has_value())                       \
        return ::tl::unexpected(std::move(SAPIENT_CAT(sapient_try_, __LINE__).error())); \
    lhs = std::move(*SAPIENT_CAT(sapient_try_, __LINE__))
```

- [ ] **Step 6: Write `src/error.cpp`**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/core/error.hpp"

#include <sstream>

namespace sapient::core {

std::string debug_dims(const std::vector<size_t>& dims) {
    std::string s = "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(dims[i]);
    }
    return s + "]";
}

namespace {
// Rust `{:?}` on a String: double-quoted with \" \\ \n \t \r escaped.
std::string debug_str(const std::string& s) {
    std::string out = "\"";
    for (const char c : s) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\t': out += "\\t"; break;
        case '\r': out += "\\r"; break;
        default: out += c;
        }
    }
    return out + "\"";
}
}  // namespace

std::string Error::to_string() const {
    switch (code) {
    case ErrorCode::ShapeMismatch:
        return "Shape mismatch: expected " + debug_dims(expected_dims) + ", got " + debug_dims(got_dims);
    case ErrorCode::RankMismatch:
        return "Rank mismatch: expected " + std::to_string(expected_n) + ", got " + std::to_string(got_n);
    case ErrorCode::TypeMismatch: return "Type mismatch: expected " + a + ", got " + b;
    case ErrorCode::BroadcastError:
        return "Incompatible shapes for broadcasting: " + debug_dims(expected_dims) + " and " + debug_dims(got_dims);
    case ErrorCode::CyclicGraph: return "Graph contains a cycle \xE2\x80\x94 execution is impossible";
    case ErrorCode::NodeNotFound: return "Node " + debug_str(a) + " not found in graph";
    case ErrorCode::InvalidGraph: return "Graph validation failed: " + a;
    case ErrorCode::ShapeInferenceFailed: return "Shape inference failed for op '" + a + "': " + b;
    case ErrorCode::UnsupportedOp: return "Backend '" + a + "' does not support op '" + b + "'";
    case ErrorCode::BackendError: return "Backend error from '" + a + "': " + b;
    case ErrorCode::NoBackendAvailable: return "No suitable backend found for execution";
    case ErrorCode::AllocationFailed:
        return "Allocation failed: requested " + std::to_string(expected_n) + " bytes (alignment " +
               std::to_string(got_n) + ")";
    case ErrorCode::BufferSizeMismatch:
        return "Buffer size mismatch: expected " + std::to_string(expected_n) + " bytes, got " +
               std::to_string(got_n);
    case ErrorCode::PoolExhausted:
        return "Memory pool exhausted \xE2\x80\x94 consider increasing pool capacity";
    case ErrorCode::OnnxParseError: return "ONNX parse error: " + a;
    case ErrorCode::GgufParseError: return "GGUF parse error: " + a;
    case ErrorCode::SafetensorsParseError: return "Safetensors parse error: " + a;
    case ErrorCode::UnsupportedFormat: return "Unsupported model format: " + a;
    case ErrorCode::ModelNotFound: return "Model not found at path '" + a + "'";
    case ErrorCode::Io: return "IO error: " + a;
    case ErrorCode::DeadlineExceeded: return "Request timed out (deadline exceeded)";
    case ErrorCode::SchedulerShutdown: return "Batch scheduler is shut down";
    case ErrorCode::UninitializedRuntime:
        return "Runtime is not initialized \xE2\x80\x94 call Session::new() first";
    case ErrorCode::TelemetryError: return "Telemetry export failed: " + a;
    case ErrorCode::Internal: return "Internal error: " + a;
    }
    return "Internal error: <unknown error code>";
}

namespace {
Error with_code(ErrorCode c) { Error e; e.code = c; return e; }
Error with_str(ErrorCode c, std::string s) { Error e = with_code(c); e.a = std::move(s); return e; }
Error with_two(ErrorCode c, std::string s, std::string t) { Error e = with_str(c, std::move(s)); e.b = std::move(t); return e; }
Error with_nums(ErrorCode c, size_t x, size_t y) { Error e = with_code(c); e.expected_n = x; e.got_n = y; return e; }
Error with_dims(ErrorCode c, std::vector<size_t> x, std::vector<size_t> y) {
    Error e = with_code(c); e.expected_dims = std::move(x); e.got_dims = std::move(y); return e;
}
}  // namespace

Error Error::shape_mismatch(std::vector<size_t> expected, std::vector<size_t> got) { return with_dims(ErrorCode::ShapeMismatch, std::move(expected), std::move(got)); }
Error Error::rank_mismatch(size_t expected, size_t got) { return with_nums(ErrorCode::RankMismatch, expected, got); }
Error Error::type_mismatch(std::string expected, std::string got) { return with_two(ErrorCode::TypeMismatch, std::move(expected), std::move(got)); }
Error Error::broadcast(std::vector<size_t> lhs, std::vector<size_t> rhs) { return with_dims(ErrorCode::BroadcastError, std::move(lhs), std::move(rhs)); }
Error Error::cyclic_graph() { return with_code(ErrorCode::CyclicGraph); }
Error Error::node_not_found(std::string node) { return with_str(ErrorCode::NodeNotFound, std::move(node)); }
Error Error::invalid_graph(std::string msg) { return with_str(ErrorCode::InvalidGraph, std::move(msg)); }
Error Error::shape_inference_failed(std::string op, std::string reason) { return with_two(ErrorCode::ShapeInferenceFailed, std::move(op), std::move(reason)); }
Error Error::unsupported_op(std::string backend_name, std::string op) { return with_two(ErrorCode::UnsupportedOp, std::move(backend_name), std::move(op)); }
Error Error::backend(std::string backend_name, std::string message) { return with_two(ErrorCode::BackendError, std::move(backend_name), std::move(message)); }
Error Error::no_backend_available() { return with_code(ErrorCode::NoBackendAvailable); }
Error Error::allocation_failed(size_t bytes, size_t align) { return with_nums(ErrorCode::AllocationFailed, bytes, align); }
Error Error::buffer_size_mismatch(size_t expected, size_t got) { return with_nums(ErrorCode::BufferSizeMismatch, expected, got); }
Error Error::pool_exhausted() { return with_code(ErrorCode::PoolExhausted); }
Error Error::onnx_parse(std::string msg) { return with_str(ErrorCode::OnnxParseError, std::move(msg)); }
Error Error::gguf_parse(std::string msg) { return with_str(ErrorCode::GgufParseError, std::move(msg)); }
Error Error::safetensors_parse(std::string msg) { return with_str(ErrorCode::SafetensorsParseError, std::move(msg)); }
Error Error::unsupported_format(std::string fmt) { return with_str(ErrorCode::UnsupportedFormat, std::move(fmt)); }
Error Error::model_not_found(std::string path) { return with_str(ErrorCode::ModelNotFound, std::move(path)); }
Error Error::io(std::error_code ec, std::string message) { Error e = with_str(ErrorCode::Io, std::move(message)); e.io_code = ec; return e; }
Error Error::deadline_exceeded() { return with_code(ErrorCode::DeadlineExceeded); }
Error Error::scheduler_shutdown() { return with_code(ErrorCode::SchedulerShutdown); }
Error Error::uninitialized_runtime() { return with_code(ErrorCode::UninitializedRuntime); }
Error Error::telemetry(std::string msg) { return with_str(ErrorCode::TelemetryError, std::move(msg)); }
Error Error::internal(std::string msg) { return with_str(ErrorCode::Internal, std::move(msg)); }

}  // namespace sapient::core
```

- [ ] **Step 7: Wire CMake** — replace `cpp/libs/sapient-core/CMakeLists.txt` with:

```cmake
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
add_library(sapient_core STATIC src/version.cpp src/error.cpp)
add_library(sapient::core ALIAS sapient_core)
target_include_directories(sapient_core PUBLIC include)
target_link_libraries(sapient_core PUBLIC tl::expected)
target_compile_definitions(sapient_core PRIVATE SAPIENT_VERSION_STRING="${PROJECT_VERSION}")
sapient_apply_warnings(sapient_core)

if(SAPIENT_BUILD_TESTS)
  add_executable(sapient_core_tests
    tests/version_test.cpp tests/build_flags_test.cpp tests/error_test.cpp)
  target_link_libraries(sapient_core_tests PRIVATE sapient::core GTest::gtest_main)
  target_compile_definitions(sapient_core_tests PRIVATE
    SAPIENT_VERSION_STRING="${PROJECT_VERSION}"
    SAPIENT_CARGO_TOML_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../../../Cargo.toml")
  sapient_apply_warnings(sapient_core_tests)
  gtest_discover_tests(sapient_core_tests)
endif()
```

Later tasks append their `src/*.cpp` and `tests/*_test.cpp` to these two lists.

- [ ] **Step 8: Build and run**

Run: `cd cpp && cmake --preset dev && cmake --build --preset dev && ctest --preset dev -R Error`
Expected: 4 tests pass (`Error.error_display`, `Error.from_io_error`, `Error.messages_match_rust_format_strings`, `Error.try_macro_propagates`). Then `ctest --preset dev` → 16 tests (12 prior + 4), 1 skip.

- [ ] **Step 9: Commit**

```bash
git add cpp/cmake/deps.cmake cpp/libs/sapient-core
git commit -m "cpp(core): Error/Result/panic — 25 error codes with Rust-identical messages, tl::expected pin

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `dtype.hpp/.cpp`

**Files:**
- Create: `cpp/libs/sapient-core/include/sapient/core/dtype.hpp`, `src/dtype.cpp`
- Modify: `cpp/libs/sapient-core/CMakeLists.txt` (add `src/dtype.cpp`, `tests/dtype_test.cpp`)
- Test: `cpp/libs/sapient-core/tests/dtype_test.cpp`

**Interfaces:**
- Consumes: `Error`, `Result`, `panic` (Task 1).
- Produces: `enum class DType : uint8_t { F32, F16, BF16, I32, I64, U8, Bool, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, Q4_K_R4, Q6_K_R4 }`; constants `QUANT_BLOCK_SIZE=32, K_QUANT_BLOCK_SIZE=256, Q4_0_BLOCK_BYTES=18, Q8_0_BLOCK_BYTES=34, Q4_K_BLOCK_BYTES=144, Q5_K_BLOCK_BYTES=176, Q6_K_BLOCK_BYTES=210`; `constexpr size_t element_size(DType)`, `alignment(DType)`, `size_t block_bytes(DType)` and `block_numel(DType)` (panic on non-quant), `size_t byte_count(DType, size_t numel)`, `constexpr bool is_quantized/is_float/is_integer(DType)`, `std::string_view name(DType)`, `Result<DType> dtype_from_str(std::string_view)`, `Result<DType> dtype_from_onnx(int32_t)`, `int32_t dtype_to_onnx(DType)`, `std::string to_string(DType)`, `operator<<`.

- [ ] **Step 1: Write the failing tests** — `tests/dtype_test.cpp`

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <string>

#include "sapient/core/dtype.hpp"

using namespace sapient::core;

// Rust: element_sizes
TEST(DType, element_sizes) {
    EXPECT_EQ(element_size(DType::F32), 4u);
    EXPECT_EQ(element_size(DType::I64), 8u);
    EXPECT_EQ(element_size(DType::Bool), 1u);
    EXPECT_EQ(element_size(DType::Q4_K), 0u);
}

// Rust: byte_count
TEST(DType, byte_count) {
    EXPECT_EQ(byte_count(DType::F32, 10), 40u);
    EXPECT_EQ(byte_count(DType::Q4_0, 64), 36u);
    EXPECT_EQ(byte_count(DType::Q4_K, 512), 288u);
    EXPECT_EQ(byte_count(DType::Q6_K_R4, 256), 210u);
    EXPECT_EQ(byte_count(DType::Q8_0, 40), 34u);  // truncating division, tail dropped like Rust
}

// Rust: from_str_roundtrip
TEST(DType, from_str_roundtrip) {
    const std::pair<const char*, DType> cases[] = {
        {"f32", DType::F32}, {"f16", DType::F16}, {"bf16", DType::BF16}, {"i32", DType::I32},
        {"i64", DType::I64}, {"u8", DType::U8},   {"bool", DType::Bool}};
    for (const auto& [s, dt] : cases) {
        auto r = dtype_from_str(s);
        ASSERT_TRUE(r.has_value()) << s;
        EXPECT_EQ(*r, dt) << s;
        EXPECT_EQ(name(dt), s);
    }
    EXPECT_EQ(*dtype_from_str("Float32"), DType::F32);  // lower-cased first
    EXPECT_EQ(*dtype_from_str("q4_k_m"), DType::Q4_K);
    EXPECT_FALSE(dtype_from_str("q4_k_r4").has_value());  // emitted by name(), not parseable (Rust asymmetry)
    EXPECT_EQ(dtype_from_str("nope").error().to_string(), "Type mismatch: expected a valid dtype, got nope");
}

// Rust: onnx_roundtrip
TEST(DType, onnx_roundtrip) {
    for (const DType dt : {DType::F32, DType::F16, DType::BF16, DType::I32, DType::I64, DType::U8, DType::Bool}) {
        auto r = dtype_from_onnx(dtype_to_onnx(dt));
        ASSERT_TRUE(r.has_value());
        EXPECT_EQ(*r, dt);
    }
    EXPECT_EQ(dtype_to_onnx(DType::Q4_K), 0);
    EXPECT_EQ(dtype_from_onnx(99).error().to_string(),
              "Type mismatch: expected a supported ONNX dtype, got ONNX code 99");
}

TEST(DType, block_constants_and_predicates) {
    EXPECT_EQ(block_bytes(DType::Q4_K_R4), Q4_K_BLOCK_BYTES);
    EXPECT_EQ(block_numel(DType::Q8_0), QUANT_BLOCK_SIZE);
    EXPECT_TRUE(is_quantized(DType::Q6_K_R4));
    EXPECT_TRUE(is_float(DType::BF16));
    EXPECT_TRUE(is_integer(DType::Bool));
    EXPECT_FALSE(is_float(DType::Q8_0));
    EXPECT_EQ(alignment(DType::Q4_0), 2u);
    EXPECT_EQ(alignment(DType::I64), 8u);
    EXPECT_EQ(to_string(DType::Q4_K_R4), "q4_k_r4");
}
```

- [ ] **Step 2: Run to verify it fails** — `cd cpp && cmake --preset dev` → `Cannot find source file: src/dtype.cpp` (after CMake edit: header not found).

- [ ] **Step 3: Write `dtype.hpp`**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-core/src/dtype.rs. Quantized dtypes store raw ggml block bytes;
// `element_size()` is 0 for them and `byte_count()` is the only valid size query.

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <string_view>

#include "sapient/core/error.hpp"

namespace sapient::core {

// Rust declaration order. New variants may be appended (Rust marks the enum #[non_exhaustive]).
enum class DType : uint8_t { F32, F16, BF16, I32, I64, U8, Bool, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, Q4_K_R4, Q6_K_R4 };

inline constexpr size_t QUANT_BLOCK_SIZE = 32;
inline constexpr size_t K_QUANT_BLOCK_SIZE = 256;
inline constexpr size_t Q4_0_BLOCK_BYTES = 18;
inline constexpr size_t Q8_0_BLOCK_BYTES = 34;
inline constexpr size_t Q4_K_BLOCK_BYTES = 144;
inline constexpr size_t Q5_K_BLOCK_BYTES = 176;
inline constexpr size_t Q6_K_BLOCK_BYTES = 210;

constexpr bool is_quantized(DType d) {
    switch (d) {
    case DType::Q4_0: case DType::Q8_0: case DType::Q4_K: case DType::Q4_K_R4:
    case DType::Q5_K: case DType::Q6_K: case DType::Q6_K_R4: return true;
    default: return false;
    }
}
constexpr bool is_float(DType d) { return d == DType::F32 || d == DType::F16 || d == DType::BF16; }
constexpr bool is_integer(DType d) { return d == DType::I32 || d == DType::I64 || d == DType::U8 || d == DType::Bool; }

/// Bytes per element; 0 for quantized dtypes (use byte_count()).
constexpr size_t element_size(DType d) {
    switch (d) {
    case DType::F32: return 4; case DType::F16: return 2; case DType::BF16: return 2;
    case DType::I32: return 4; case DType::I64: return 8; case DType::U8: return 1; case DType::Bool: return 1;
    default: return 0;
    }
}
constexpr size_t alignment(DType d) {
    switch (d) {
    case DType::F32: return 4; case DType::F16: case DType::BF16: return 2;
    case DType::I32: return 4; case DType::I64: return 8; case DType::U8: case DType::Bool: return 1;
    default: return 2;  // all quantized
    }
}
/// Panics on a non-quantized dtype (Rust: panic!("block_bytes() called on non-quantized dtype")).
size_t block_bytes(DType d);
size_t block_numel(DType d);
/// Truncating: a numel that is not a block multiple silently drops the tail (Rust parity).
size_t byte_count(DType d, size_t numel);

std::string_view name(DType d);
std::string to_string(DType d);
std::ostream& operator<<(std::ostream& os, DType d);
/// Rust `FromStr`: lower-cases, accepts the aliases ("float32", "q4_k_m", …); R4 names are NOT parseable.
Result<DType> dtype_from_str(std::string_view s);
Result<DType> dtype_from_onnx(int32_t code);
int32_t dtype_to_onnx(DType d);

}  // namespace sapient::core
```

- [ ] **Step 4: Write `src/dtype.cpp`**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/core/dtype.hpp"

#include <algorithm>
#include <cctype>

#include "sapient/core/panic.hpp"

namespace sapient::core {

size_t block_bytes(DType d) {
    switch (d) {
    case DType::Q4_0: return Q4_0_BLOCK_BYTES;
    case DType::Q8_0: return Q8_0_BLOCK_BYTES;
    case DType::Q4_K: case DType::Q4_K_R4: return Q4_K_BLOCK_BYTES;
    case DType::Q5_K: return Q5_K_BLOCK_BYTES;
    case DType::Q6_K: case DType::Q6_K_R4: return Q6_K_BLOCK_BYTES;
    default: panic("block_bytes() called on non-quantized dtype");
    }
}

size_t block_numel(DType d) {
    switch (d) {
    case DType::Q4_0: case DType::Q8_0: return QUANT_BLOCK_SIZE;
    case DType::Q4_K: case DType::Q4_K_R4: case DType::Q5_K: case DType::Q6_K: case DType::Q6_K_R4:
        return K_QUANT_BLOCK_SIZE;
    default: panic("block_numel() called on non-quantized dtype");
    }
}

size_t byte_count(DType d, size_t numel) {
    switch (d) {
    case DType::Q4_0: return numel / QUANT_BLOCK_SIZE * Q4_0_BLOCK_BYTES;
    case DType::Q8_0: return numel / QUANT_BLOCK_SIZE * Q8_0_BLOCK_BYTES;
    case DType::Q4_K: case DType::Q4_K_R4: return numel / K_QUANT_BLOCK_SIZE * Q4_K_BLOCK_BYTES;
    case DType::Q5_K: return numel / K_QUANT_BLOCK_SIZE * Q5_K_BLOCK_BYTES;
    case DType::Q6_K: case DType::Q6_K_R4: return numel / K_QUANT_BLOCK_SIZE * Q6_K_BLOCK_BYTES;
    default: return numel * element_size(d);
    }
}

std::string_view name(DType d) {
    switch (d) {
    case DType::F32: return "f32"; case DType::F16: return "f16"; case DType::BF16: return "bf16";
    case DType::I32: return "i32"; case DType::I64: return "i64"; case DType::U8: return "u8";
    case DType::Bool: return "bool"; case DType::Q4_0: return "q4_0"; case DType::Q8_0: return "q8_0";
    case DType::Q4_K: return "q4_k"; case DType::Q4_K_R4: return "q4_k_r4"; case DType::Q5_K: return "q5_k";
    case DType::Q6_K: return "q6_k"; case DType::Q6_K_R4: return "q6_k_r4";
    }
    return "?";
}
std::string to_string(DType d) { return std::string(name(d)); }
std::ostream& operator<<(std::ostream& os, DType d) { return os << name(d); }

Result<DType> dtype_from_str(std::string_view sv) {
    std::string s(sv);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (s == "f32" || s == "float32") return DType::F32;
    if (s == "f16" || s == "float16") return DType::F16;
    if (s == "bf16" || s == "bfloat16") return DType::BF16;
    if (s == "i32" || s == "int32") return DType::I32;
    if (s == "i64" || s == "int64") return DType::I64;
    if (s == "u8" || s == "uint8") return DType::U8;
    if (s == "bool") return DType::Bool;
    if (s == "q4_0") return DType::Q4_0;
    if (s == "q8_0") return DType::Q8_0;
    if (s == "q4_k" || s == "q4_k_m" || s == "q4_k_s") return DType::Q4_K;
    if (s == "q5_k" || s == "q5_k_m" || s == "q5_k_s") return DType::Q5_K;
    if (s == "q6_k") return DType::Q6_K;
    return tl::unexpected(Error::type_mismatch("a valid dtype", s));  // Rust reports the lower-cased input
}

Result<DType> dtype_from_onnx(int32_t code) {
    switch (code) {
    case 1: return DType::F32; case 2: return DType::U8; case 5: return DType::I32; case 7: return DType::I64;
    case 9: return DType::Bool; case 10: return DType::F16; case 16: return DType::BF16;
    default: return tl::unexpected(Error::type_mismatch("a supported ONNX dtype", "ONNX code " + std::to_string(code)));
    }
}

int32_t dtype_to_onnx(DType d) {
    switch (d) {
    case DType::F32: return 1; case DType::U8: return 2; case DType::I32: return 5; case DType::I64: return 7;
    case DType::Bool: return 9; case DType::F16: return 10; case DType::BF16: return 16;
    default: return 0;  // all quantized dtypes
    }
}

}  // namespace sapient::core
```

Note: Rust's `FromStr` lower-cases into a new string and its `TypeMismatch { got }` carries that lower-cased string (`crates/sapient-core/src/dtype.rs:207-229`), hence `s` not `sv` in the error above. Add to the test: `EXPECT_EQ(dtype_from_str("NoPe").error().to_string(), "Type mismatch: expected a valid dtype, got nope");`.

- [ ] **Step 5: CMake** — add `src/dtype.cpp` to `add_library(sapient_core …)` and `tests/dtype_test.cpp` to the test executable.

- [ ] **Step 6: Build and run** — `cd cpp && cmake --preset dev && cmake --build --preset dev && ctest --preset dev -R DType` → 5 pass.

- [ ] **Step 7: Commit**

```bash
git add cpp/libs/sapient-core
git commit -m "cpp(core): DType — block constants, sizes, names, FromStr aliases, ONNX mapping

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `shape.hpp/.cpp`

**Files:**
- Create: `include/sapient/core/shape.hpp`, `src/shape.cpp`; Test: `tests/shape_test.cpp`; Modify: `CMakeLists.txt`.

**Interfaces:**
- Consumes: `Error`, `Result`.
- Produces: `struct Shape { std::vector<size_t> dims; Shape(); Shape(std::vector<size_t>); Shape(std::initializer_list<size_t>); static Shape scalar(); size_t ndim() const; size_t numel() const; std::vector<size_t> strides() const; bool is_scalar() const; Result<Shape> reshape(std::vector<size_t>) const; Result<Shape> broadcast_with(const Shape&) const; Result<Shape> expand_dims(size_t axis) const; Shape squeeze() const; Result<void> validate() const; Result<size_t> flat_index(std::span<const size_t>) const; std::string to_string() const; bool operator==(const Shape&) const = default; }`.

- [ ] **Step 1: Failing tests** — `tests/shape_test.cpp`

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <vector>

#include "sapient/core/shape.hpp"

using sapient::core::Shape;

// Rust: numel
TEST(Shape, numel) {
    EXPECT_EQ(Shape({2, 3, 4}).numel(), 24u);
    EXPECT_EQ(Shape::scalar().numel(), 1u);
}
// Rust: strides_row_major
TEST(Shape, strides_row_major) {
    EXPECT_EQ(Shape({2, 3, 4}).strides(), (std::vector<size_t>{12, 4, 1}));
    EXPECT_TRUE(Shape::scalar().strides().empty());
}
// Rust: broadcast
TEST(Shape, broadcast) {
    auto r = Shape({1, 3}).broadcast_with(Shape({2, 3}));
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->dims, (std::vector<size_t>{2, 3}));
    auto r2 = Shape({3}).broadcast_with(Shape({2, 1, 3}));
    ASSERT_TRUE(r2.has_value());
    EXPECT_EQ(r2->dims, (std::vector<size_t>{2, 1, 3}));
}
// Rust: broadcast_fail
TEST(Shape, broadcast_fail) {
    auto r = Shape({2, 3}).broadcast_with(Shape({2, 4}));
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().to_string(), "Incompatible shapes for broadcasting: [2, 3] and [2, 4]");
}
// Rust: reshape
TEST(Shape, reshape) {
    auto r = Shape({2, 3}).reshape({6});
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->dims, (std::vector<size_t>{6}));
    EXPECT_EQ(Shape({2, 3}).reshape({5}).error().to_string(), "Shape mismatch: expected [2, 3], got [5]");
}
// Rust: flat_index
TEST(Shape, flat_index) {
    const size_t idx[] = {1, 2, 3};
    auto r = Shape({2, 3, 4}).flat_index(idx);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(*r, 23u);
    const size_t bad[] = {1, 3, 0};
    EXPECT_EQ(Shape({2, 3, 4}).flat_index(bad).error().to_string(),
              "Internal error: Index 3 out of bounds for dim 1 (size 3)");
    const size_t short_idx[] = {1};
    EXPECT_EQ(Shape({2, 3}).flat_index(short_idx).error().to_string(), "Rank mismatch: expected 2, got 1");
}
TEST(Shape, validate_expand_squeeze_display) {
    EXPECT_TRUE(Shape({2, 3}).validate().has_value());
    EXPECT_EQ(Shape({2, 0}).validate().error().to_string(),
              "Graph validation failed: Shape has zero dimension at axis 1");
    EXPECT_EQ(Shape({2, 3}).expand_dims(1)->dims, (std::vector<size_t>{2, 1, 3}));
    EXPECT_EQ(Shape({2, 3}).expand_dims(3).error().to_string(),
              "Internal error: expand_dims: axis 3 out of range for rank 2");
    EXPECT_EQ(Shape({1, 2, 1, 3}).squeeze().dims, (std::vector<size_t>{2, 3}));
    EXPECT_EQ(Shape({2, 3}).to_string(), "[2, 3]");
    EXPECT_EQ(Shape::scalar().to_string(), "[]");
}
```

- [ ] **Step 2: Run to verify it fails** — configure: `Cannot find source file: src/shape.cpp`.

- [ ] **Step 3: Write `shape.hpp`**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-core/src/shape.rs. `dims` is public like Rust's tuple field `.0`.

#include <cstddef>
#include <initializer_list>
#include <span>
#include <string>
#include <vector>

#include "sapient/core/error.hpp"

namespace sapient::core {

struct Shape {
    std::vector<size_t> dims;

    Shape() = default;
    explicit Shape(std::vector<size_t> d) : dims(std::move(d)) {}
    Shape(std::initializer_list<size_t> d) : dims(d) {}
    static Shape scalar() { return Shape(); }

    size_t ndim() const { return dims.size(); }
    /// Product of dims; the scalar shape (no dims) has numel 1.
    size_t numel() const;
    /// Row-major element strides; empty for the scalar shape.
    std::vector<size_t> strides() const;
    bool is_scalar() const { return dims.empty(); }
    Result<Shape> reshape(std::vector<size_t> new_dims) const;
    /// NumPy right-aligned broadcasting.
    Result<Shape> broadcast_with(const Shape& other) const;
    Result<Shape> expand_dims(size_t axis) const;
    Shape squeeze() const;  // drops every dim equal to 1
    /// Rejects zero dims (error text uses the InvalidGraph variant, as Rust does).
    Result<void> validate() const;
    /// Element (not byte) offset of a multi-index.
    Result<size_t> flat_index(std::span<const size_t> idx) const;
    std::string to_string() const;  // "[2, 3]"
    bool operator==(const Shape&) const = default;
};

}  // namespace sapient::core
```

- [ ] **Step 4: Write `src/shape.cpp`**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/core/shape.hpp"

#include <algorithm>

namespace sapient::core {

size_t Shape::numel() const {
    size_t n = 1;
    for (const size_t d : dims) n *= d;
    return n;
}

std::vector<size_t> Shape::strides() const {
    const size_t n = dims.size();
    std::vector<size_t> s(n);
    if (n == 0) return s;
    s[n - 1] = 1;
    for (size_t i = n - 1; i-- > 0;) s[i] = s[i + 1] * dims[i + 1];
    return s;
}

Result<Shape> Shape::reshape(std::vector<size_t> new_dims) const {
    Shape ns(std::move(new_dims));
    if (ns.numel() != numel()) return tl::unexpected(Error::shape_mismatch(dims, ns.dims));
    return ns;
}

Result<Shape> Shape::broadcast_with(const Shape& other) const {
    const size_t len = std::max(dims.size(), other.dims.size());
    std::vector<size_t> out(len);
    for (size_t i = 0; i < len; ++i) {
        const size_t ai = i < len - dims.size() ? 1 : dims[i - (len - dims.size())];
        const size_t bi = i < len - other.dims.size() ? 1 : other.dims[i - (len - other.dims.size())];
        if (ai == bi) out[i] = ai;
        else if (ai == 1) out[i] = bi;
        else if (bi == 1) out[i] = ai;
        else return tl::unexpected(Error::broadcast(dims, other.dims));
    }
    return Shape(std::move(out));
}

Result<Shape> Shape::expand_dims(size_t axis) const {
    if (axis > dims.size())
        return tl::unexpected(Error::internal("expand_dims: axis " + std::to_string(axis) +
                                              " out of range for rank " + std::to_string(dims.size())));
    std::vector<size_t> out = dims;
    out.insert(out.begin() + static_cast<std::ptrdiff_t>(axis), 1);
    return Shape(std::move(out));
}

Shape Shape::squeeze() const {
    std::vector<size_t> out;
    for (const size_t d : dims) if (d != 1) out.push_back(d);
    return Shape(std::move(out));
}

Result<void> Shape::validate() const {
    for (size_t i = 0; i < dims.size(); ++i)
        if (dims[i] == 0)
            return tl::unexpected(Error::invalid_graph("Shape has zero dimension at axis " + std::to_string(i)));
    return {};
}

Result<size_t> Shape::flat_index(std::span<const size_t> idx) const {
    if (idx.size() != dims.size()) return tl::unexpected(Error::rank_mismatch(dims.size(), idx.size()));
    const auto st = strides();
    size_t off = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (idx[i] >= dims[i])
            return tl::unexpected(Error::internal("Index " + std::to_string(idx[i]) + " out of bounds for dim " +
                                                  std::to_string(i) + " (size " + std::to_string(dims[i]) + ")"));
        off += idx[i] * st[i];
    }
    return off;
}

std::string Shape::to_string() const { return debug_dims(dims); }

}  // namespace sapient::core
```

- [ ] **Step 5: CMake** — add `src/shape.cpp` and `tests/shape_test.cpp`.
- [ ] **Step 6: Build and run** — `ctest --preset dev -R Shape` → 7 pass.
- [ ] **Step 7: Commit** — `git commit -m "cpp(core): Shape — strides, broadcast, reshape, flat_index with Rust error texts"` (+ trailer).

---

### Task 4: `f16.hpp/.cpp` — software half/bfloat conversions

**Files:**
- Create: `include/sapient/core/f16.hpp`, `src/f16.cpp`; Test: `tests/f16_test.cpp`; Modify: `CMakeLists.txt`.

**Interfaces:**
- Produces: `namespace sapient::core { float f16_bits_to_f32(uint16_t); uint16_t f32_to_f16_bits(float); float bf16_bits_to_f32(uint16_t); uint16_t f32_to_bf16_bits(float); inline float f16_le_to_f32(const uint8_t* p); inline float bf16_le_to_f32(const uint8_t* p); inline void f16_to_le(uint16_t bits, uint8_t* p); }`. These replace every `half::f16`/`half::bf16` call in the Rust tree (exact widening; round-to-nearest-even narrowing incl. subnormals; NaN → quiet NaN keeping the high payload bits — NaN policy is not parity-relevant, dumps never contain NaN).

- [ ] **Step 1: Failing tests** — `tests/f16_test.cpp`

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

#include "sapient/core/f16.hpp"

using namespace sapient::core;

TEST(F16, widening_is_exact) {
    EXPECT_EQ(f16_bits_to_f32(0x3C00), 1.0f);
    EXPECT_EQ(f16_bits_to_f32(0xC000), -2.0f);
    EXPECT_EQ(f16_bits_to_f32(0x3555), 0.333251953125f);
    EXPECT_EQ(f16_bits_to_f32(0x0001), 5.960464477539063e-08f);   // smallest subnormal 2^-24
    EXPECT_EQ(f16_bits_to_f32(0x03FF), 6.097555160522461e-05f);   // largest subnormal
    EXPECT_EQ(f16_bits_to_f32(0x0400), 6.103515625e-05f);         // smallest normal 2^-14
    EXPECT_EQ(f16_bits_to_f32(0x7BFF), 65504.0f);
    EXPECT_EQ(f16_bits_to_f32(0x7C00), std::numeric_limits<float>::infinity());
    EXPECT_EQ(f16_bits_to_f32(0xFC00), -std::numeric_limits<float>::infinity());
    EXPECT_EQ(std::bit_cast<uint32_t>(f16_bits_to_f32(0x8000)), 0x80000000u);  // -0
    EXPECT_TRUE(std::isnan(f16_bits_to_f32(0x7E00)));
}

TEST(F16, narrowing_rounds_to_nearest_even) {
    EXPECT_EQ(f32_to_f16_bits(1.0f), 0x3C00);
    EXPECT_EQ(f32_to_f16_bits(-2.0f), 0xC000);
    EXPECT_EQ(f32_to_f16_bits(65504.0f), 0x7BFF);
    EXPECT_EQ(f32_to_f16_bits(65520.0f), 0x7C00);              // tie at the top rounds to inf
    EXPECT_EQ(f32_to_f16_bits(1.0009765625f), 0x3C01);         // 1 + 2^-10 exact
    EXPECT_EQ(f32_to_f16_bits(1.00048828125f), 0x3C00);        // 1 + 2^-11: tie → even (0x3C00)
    EXPECT_EQ(f32_to_f16_bits(1.00146484375f), 0x3C02);        // 1 + 3·2^-11: tie → even (0x3C02)
    EXPECT_EQ(f32_to_f16_bits(5.960464477539063e-08f), 0x0001); // 2^-24
    EXPECT_EQ(f32_to_f16_bits(2.9802322387695312e-08f), 0x0000); // 2^-25: tie → even (0)
    EXPECT_EQ(f32_to_f16_bits(4.470348358154297e-08f), 0x0001);  // 3·2^-26 > half → 1
    EXPECT_EQ(f32_to_f16_bits(6.103515625e-05f), 0x0400);
    EXPECT_EQ(f32_to_f16_bits(0.0f), 0x0000);
    EXPECT_EQ(f32_to_f16_bits(-0.0f), 0x8000);
    EXPECT_EQ(f32_to_f16_bits(std::numeric_limits<float>::infinity()), 0x7C00);
    EXPECT_EQ(f32_to_f16_bits(1e-30f), 0x0000);
    EXPECT_EQ(f32_to_f16_bits(1e30f), 0x7C00);
    EXPECT_EQ(f32_to_f16_bits(std::numeric_limits<float>::quiet_NaN()) & 0x7E00, 0x7E00);
}

TEST(F16, every_non_nan_half_round_trips) {
    for (uint32_t b = 0; b < 0x10000; ++b) {
        const auto h = static_cast<uint16_t>(b);
        if ((h & 0x7C00) == 0x7C00 && (h & 0x03FF) != 0) continue;  // NaN payloads are not round-trip tested
        ASSERT_EQ(f32_to_f16_bits(f16_bits_to_f32(h)), h) << std::hex << b;
    }
}

TEST(F16, bf16) {
    EXPECT_EQ(bf16_bits_to_f32(0x3F80), 1.0f);
    EXPECT_EQ(f32_to_bf16_bits(1.0f), 0x3F80);
    EXPECT_EQ(f32_to_bf16_bits(std::bit_cast<float>(0x3F808000u)), 0x3F80);  // tie → even
    EXPECT_EQ(f32_to_bf16_bits(std::bit_cast<float>(0x3F808001u)), 0x3F81);  // above tie
    EXPECT_EQ(f32_to_bf16_bits(std::bit_cast<float>(0x3F818000u)), 0x3F82);  // tie, odd → up
    EXPECT_EQ(f32_to_bf16_bits(std::numeric_limits<float>::infinity()), 0x7F80);
    const uint8_t le[2] = {0x80, 0x3F};
    EXPECT_EQ(bf16_le_to_f32(le), 1.0f);
    const uint8_t hle[2] = {0x00, 0x3C};
    EXPECT_EQ(f16_le_to_f32(hle), 1.0f);
    uint8_t out[2];
    f16_to_le(0x3C00, out);
    EXPECT_EQ(out[0], 0x00);
    EXPECT_EQ(out[1], 0x3C);
}
```

- [ ] **Step 2: Run to verify it fails** — configure: `Cannot find source file: src/f16.cpp`.

- [ ] **Step 3: Write `f16.hpp`**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Software IEEE binary16 / bfloat16 conversions — the C++ twin of the `half` crate as SAPIENT
// uses it: exact widening, round-to-nearest-even narrowing (subnormals included). Software only:
// no F16C / NEON vcvt, so every host and every ISA path produces the same bits (spec §3 rule 5).

#include <bit>
#include <cstdint>
#include <cstring>

namespace sapient::core {

float f16_bits_to_f32(uint16_t h);
uint16_t f32_to_f16_bits(float f);

inline float bf16_bits_to_f32(uint16_t b) { return std::bit_cast<float>(static_cast<uint32_t>(b) << 16); }
uint16_t f32_to_bf16_bits(float f);

inline uint16_t load_le16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (static_cast<uint16_t>(p[1]) << 8)); }
inline float f16_le_to_f32(const uint8_t* p) { return f16_bits_to_f32(load_le16(p)); }
inline float bf16_le_to_f32(const uint8_t* p) { return bf16_bits_to_f32(load_le16(p)); }
inline void f16_to_le(uint16_t bits, uint8_t* p) { p[0] = static_cast<uint8_t>(bits & 0xFF); p[1] = static_cast<uint8_t>(bits >> 8); }

}  // namespace sapient::core
```

- [ ] **Step 4: Write `src/f16.cpp`**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/core/f16.hpp"

static_assert(std::endian::native == std::endian::little, "SAPIENT assumes a little-endian host");

namespace sapient::core {

float f16_bits_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;  // ±0
        } else {          // subnormal: value = mant · 2^-24 → normalise
            uint32_t m = mant;
            uint32_t s = 0;
            while ((m & 0x400u) == 0) { m <<= 1; ++s; }
            bits = sign | ((113u - s) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);  // inf, or NaN with the payload shifted up
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    return std::bit_cast<float>(bits);
}

uint16_t f32_to_f16_bits(float f) {
    const uint32_t x = std::bit_cast<uint32_t>(f);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t exp = (x >> 23) & 0xFFu;
    const uint32_t mant = x & 0x7FFFFFu;
    if (exp == 0xFF) {  // inf or NaN
        if (mant == 0) return static_cast<uint16_t>(sign | 0x7C00u);
        return static_cast<uint16_t>(sign | 0x7E00u | ((mant >> 13) & 0x1FFu));  // quiet NaN, high payload kept
    }
    const int32_t e = static_cast<int32_t>(exp) - 127 + 15;  // rebias to binary16
    if (e >= 31) return static_cast<uint16_t>(sign | 0x7C00u);  // overflow → inf
    if (e >= 1) {  // normal result; round-to-nearest-even on the 13 dropped bits; carry may overflow into inf
        uint32_t out = sign | (static_cast<uint32_t>(e) << 10) | (mant >> 13);
        const uint32_t round = mant & 0x1FFFu;
        if (round > 0x1000u || (round == 0x1000u && (out & 1u))) out += 1;
        return static_cast<uint16_t>(out);
    }
    if (e < -10) return static_cast<uint16_t>(sign);  // below half the smallest subnormal → ±0
    // subnormal result: shift the 24-bit significand (implicit 1) right by 14 - e, round to nearest even
    const uint32_t m = mant | 0x800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - e);
    uint32_t out = m >> shift;
    const uint32_t rem = m & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1);
    if (rem > halfway || (rem == halfway && (out & 1u))) out += 1;  // may carry into the smallest normal
    return static_cast<uint16_t>(sign | out);
}

uint16_t f32_to_bf16_bits(float f) {
    uint32_t x = std::bit_cast<uint32_t>(f);
    if ((x & 0x7F800000u) == 0x7F800000u && (x & 0x7FFFFFu) != 0) return static_cast<uint16_t>((x >> 16) | 0x40u);  // quiet NaN
    x += 0x7FFFu + ((x >> 16) & 1u);  // round to nearest even
    return static_cast<uint16_t>(x >> 16);
}

}  // namespace sapient::core
```

- [ ] **Step 5: CMake** — add `src/f16.cpp`, `tests/f16_test.cpp`.
- [ ] **Step 6: Build and run** — `ctest --preset dev -R F16` → 4 pass (the exhaustive round trip covers 65 536 values in well under a second).
- [ ] **Step 7: Commit** — `git commit -m "cpp(core): software f16/bf16 conversions, exhaustively round-trip tested"` (+ trailer).

---

### Task 5: `buffer.hpp/.cpp` — `Buffer`, `CpuBuffer`, `BufferHandle`

**Files:**
- Create: `include/sapient/core/buffer.hpp`, `src/buffer.cpp`; Test: `tests/buffer_test.cpp`; Modify: `CMakeLists.txt`.

**Interfaces:**
- Consumes: `DType`, `alignment()` (Task 2), `Error`/`Result` (Task 1), `panic`.
- Produces:
  ```cpp
  class Buffer { virtual std::span<const uint8_t> bytes() const = 0; virtual std::span<uint8_t> bytes_mut() = 0;
                 virtual size_t len() const = 0; virtual bool is_mmap() const { return false; } bool is_empty() const;
                 virtual size_t alignment() const = 0; virtual std::string_view device() const = 0; };
  using BufferHandle = std::shared_ptr<Buffer>;
  class CpuBuffer final : public Buffer {
    static Result<std::shared_ptr<CpuBuffer>> with_capacity(size_t bytes, size_t align);  // zero-filled
    static Result<std::shared_ptr<CpuBuffer>> zeros(size_t numel, DType dtype);           // align 64
    static Result<std::shared_ptr<CpuBuffer>> from_f32_slice(std::span<const float>);     // copy, align 64
    static Result<std::shared_ptr<CpuBuffer>> from_bytes_slice(std::span<const uint8_t>);  // copy, align 16
    static std::shared_ptr<CpuBuffer> from_f32_vec(std::vector<float>&&);                  // move, align 4
    std::span<const float> f32s() const; std::span<float> f32s_mut();  // panic unless len % 4 == 0
    const uint8_t* data() const; uint8_t* data();  /* + Buffer overrides; device() == "cpu" */ };
  ```

- [ ] **Step 1: Failing tests** — `tests/buffer_test.cpp`

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "sapient/core/buffer.hpp"
#include "sapient/core/dtype.hpp"

using namespace sapient::core;

// Rust: zeros_and_read
TEST(CpuBuffer, zeros_and_read) {
    auto b = CpuBuffer::zeros(4, DType::F32);
    ASSERT_TRUE(b.has_value());
    EXPECT_EQ((*b)->len(), 16u);
    for (const auto byte : (*b)->bytes()) EXPECT_EQ(byte, 0);
    EXPECT_EQ((*b)->alignment(), 64u);
    EXPECT_EQ((*b)->device(), "cpu");
    EXPECT_FALSE((*b)->is_mmap());
}
// Rust: from_f32_roundtrip
TEST(CpuBuffer, from_f32_roundtrip) {
    const float in[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto b = CpuBuffer::from_f32_slice(in);
    ASSERT_TRUE(b.has_value());
    const auto f = (*b)->f32s();
    ASSERT_EQ(f.size(), 4u);
    for (size_t i = 0; i < 4; ++i) EXPECT_EQ(f[i], in[i]);
}
// Rust: alignment_guarantee
TEST(CpuBuffer, alignment_guarantee) {
    auto b = CpuBuffer::with_capacity(32, 64);
    ASSERT_TRUE(b.has_value());
    EXPECT_EQ(reinterpret_cast<uintptr_t>((*b)->data()) % 64, 0u);
    EXPECT_EQ((*b)->len(), 32u);
}
TEST(CpuBuffer, from_f32_vec_moves_without_copy) {
    std::vector<float> v = {1.5f, 2.5f};
    const float* before = v.data();
    auto b = CpuBuffer::from_f32_vec(std::move(v));
    EXPECT_EQ(reinterpret_cast<const float*>(b->data()), before);
    EXPECT_EQ(b->len(), 8u);
    EXPECT_EQ(b->alignment(), 4u);
    EXPECT_EQ(b->f32s()[1], 2.5f);
}
TEST(CpuBuffer, from_bytes_slice_copies_with_align_16) {
    const uint8_t in[] = {1, 2, 3};
    auto b = CpuBuffer::from_bytes_slice(in);
    ASSERT_TRUE(b.has_value());
    EXPECT_EQ((*b)->alignment(), 16u);
    EXPECT_EQ((*b)->bytes()[2], 3);
    (*b)->bytes_mut()[0] = 9;
    EXPECT_EQ((*b)->bytes()[0], 9);
}
TEST(CpuBuffer, zero_length_is_valid) {
    auto b = CpuBuffer::with_capacity(0, 16);
    ASSERT_TRUE(b.has_value());
    EXPECT_EQ((*b)->len(), 0u);
    EXPECT_TRUE((*b)->is_empty());
    EXPECT_NE((*b)->data(), nullptr);  // Rust allocates 1 byte for a 0-length buffer
}
TEST(CpuBuffer, bad_alignment_is_an_error) {
    auto b = CpuBuffer::with_capacity(8, 3);  // not a power of two → Rust Layout error → AllocationFailed
    ASSERT_FALSE(b.has_value());
    EXPECT_EQ(b.error().to_string(), "Allocation failed: requested 8 bytes (alignment 3)");
}
```

- [ ] **Step 2: Run to verify it fails** — configure: `Cannot find source file: src/buffer.cpp`.

- [ ] **Step 3: Write `buffer.hpp`**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-core/src/buffer.rs. `Buffer` is the `dyn Buffer` trait; `CpuBuffer`
// owns a zero-filled aligned allocation (or a moved std::vector<float> — the safe twin of
// Rust's `from_f32_vec` layout trick). The mmap buffer lives in sapient::io, as in Rust.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "sapient/core/dtype.hpp"
#include "sapient/core/error.hpp"

namespace sapient::core {

class Buffer {
public:
    virtual ~Buffer() = default;
    virtual std::span<const uint8_t> bytes() const = 0;
    virtual std::span<uint8_t> bytes_mut() = 0;
    virtual size_t len() const = 0;
    virtual bool is_mmap() const { return false; }
    bool is_empty() const { return len() == 0; }
    virtual size_t alignment() const = 0;
    virtual std::string_view device() const = 0;  // "cpu" | "cpu-mmap"
};

/// Rust `BufferHandle(Arc<dyn Buffer>)`. Exclusive-mutation checks use `use_count() == 1`;
/// no code in the C++ tree ever takes a `weak_ptr` to a Buffer (keep it that way).
using BufferHandle = std::shared_ptr<Buffer>;

class CpuBuffer final : public Buffer {
public:
    /// Zero-filled, `align`-aligned allocation. `bytes == 0` still allocates one byte (Rust parity)
    /// but reports len 0. Non-power-of-two `align` → AllocationFailed (Rust: Layout error).
    static Result<std::shared_ptr<CpuBuffer>> with_capacity(size_t bytes, size_t align);
    static Result<std::shared_ptr<CpuBuffer>> zeros(size_t numel, DType dtype);        // align max(alignment(dtype), 64)
    static Result<std::shared_ptr<CpuBuffer>> from_f32_slice(std::span<const float> v);  // copy, align 64
    static Result<std::shared_ptr<CpuBuffer>> from_bytes_slice(std::span<const uint8_t> v);  // copy, align 16
    static std::shared_ptr<CpuBuffer> from_f32_vec(std::vector<float>&& v);              // move, align 4

    std::span<const float> f32s() const;  // panics unless len % 4 == 0 (Rust assert)
    std::span<float> f32s_mut();
    const uint8_t* data() const { return ptr_; }
    uint8_t* data() { return ptr_; }

    std::span<const uint8_t> bytes() const override { return {ptr_, len_}; }
    std::span<uint8_t> bytes_mut() override { return {ptr_, len_}; }
    size_t len() const override { return len_; }
    size_t alignment() const override { return align_; }
    std::string_view device() const override { return "cpu"; }

    CpuBuffer(const CpuBuffer&) = delete;
    CpuBuffer& operator=(const CpuBuffer&) = delete;
    ~CpuBuffer() override;

private:
    CpuBuffer() = default;
    uint8_t* ptr_{nullptr};          // points into raw_ or vec_
    void* raw_{nullptr};             // aligned allocation (freed with the matching aligned free)
    std::vector<float> vec_;         // storage for the moved-vector constructor
    size_t len_{0};
    size_t align_{0};
};

}  // namespace sapient::core
```

- [ ] **Step 4: Write `src/buffer.cpp`**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/core/buffer.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "sapient/core/panic.hpp"

#ifdef _WIN32
#include <malloc.h>
#endif

namespace sapient::core {

namespace {
bool is_pow2(size_t a) { return a != 0 && (a & (a - 1)) == 0; }

void* aligned_zeroed(size_t size, size_t align) {
    const size_t alloc_align = std::max(align, alignof(std::max_align_t));  // posix_memalign floor
#ifdef _WIN32
    void* p = _aligned_malloc(size, alloc_align);
    if (p == nullptr) return nullptr;
#else
    void* p = nullptr;
    if (posix_memalign(&p, alloc_align, size) != 0) return nullptr;
#endif
    std::memset(p, 0, size);
    return p;
}

void aligned_free(void* p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
}
}  // namespace

CpuBuffer::~CpuBuffer() {
    if (raw_ != nullptr) aligned_free(raw_);
}

Result<std::shared_ptr<CpuBuffer>> CpuBuffer::with_capacity(size_t bytes, size_t align) {
    if (!is_pow2(align)) return tl::unexpected(Error::allocation_failed(bytes, align));
    std::shared_ptr<CpuBuffer> b(new CpuBuffer());
    const size_t alloc_bytes = bytes == 0 ? 1 : bytes;  // Rust allocates 1 byte for an empty buffer
    b->raw_ = aligned_zeroed(alloc_bytes, align);
    if (b->raw_ == nullptr) return tl::unexpected(Error::allocation_failed(bytes, align));
    b->ptr_ = static_cast<uint8_t*>(b->raw_);
    b->len_ = bytes;
    b->align_ = align;
    return b;
}

Result<std::shared_ptr<CpuBuffer>> CpuBuffer::zeros(size_t numel, DType dtype) {
    return with_capacity(byte_count(dtype, numel), std::max(alignment(dtype), size_t{64}));
}

Result<std::shared_ptr<CpuBuffer>> CpuBuffer::from_f32_slice(std::span<const float> v) {
    SAPIENT_TRY_ASSIGN(auto b, with_capacity(v.size() * sizeof(float), 64));
    if (!v.empty()) std::memcpy(b->ptr_, v.data(), v.size() * sizeof(float));
    return b;
}

Result<std::shared_ptr<CpuBuffer>> CpuBuffer::from_bytes_slice(std::span<const uint8_t> v) {
    SAPIENT_TRY_ASSIGN(auto b, with_capacity(v.size(), 16));
    if (!v.empty()) std::memcpy(b->ptr_, v.data(), v.size());
    return b;
}

std::shared_ptr<CpuBuffer> CpuBuffer::from_f32_vec(std::vector<float>&& v) {
    std::shared_ptr<CpuBuffer> b(new CpuBuffer());
    b->vec_ = std::move(v);
    b->len_ = b->vec_.size() * sizeof(float);
    b->align_ = alignof(float);
    b->ptr_ = reinterpret_cast<uint8_t*>(b->vec_.data());
    if (b->ptr_ == nullptr) {  // an empty vector may have no storage; Rust falls back to a 1-byte allocation
        b->raw_ = aligned_zeroed(1, alignof(float));
        b->ptr_ = static_cast<uint8_t*>(b->raw_);
    }
    return b;
}

std::span<const float> CpuBuffer::f32s() const {
    if (len_ % sizeof(float) != 0) panic("CpuBuffer::f32s: length is not a multiple of 4");
    return {reinterpret_cast<const float*>(ptr_), len_ / sizeof(float)};
}

std::span<float> CpuBuffer::f32s_mut() {
    if (len_ % sizeof(float) != 0) panic("CpuBuffer::f32s_mut: length is not a multiple of 4");
    return {reinterpret_cast<float*>(ptr_), len_ / sizeof(float)};
}

}  // namespace sapient::core
```

- [ ] **Step 5: CMake** — add `src/buffer.cpp`, `tests/buffer_test.cpp`.
- [ ] **Step 6: Build and run** — `ctest --preset dev -R CpuBuffer` → 7 pass. `-Wpedantic` note: `reinterpret_cast` between `float*` and `uint8_t*` is fine; do not use `-Wcast-align`.
- [ ] **Step 7: Commit** — `git commit -m "cpp(core): Buffer/CpuBuffer — zeroed aligned allocation with Rust's per-constructor alignments"` (+ trailer).

---

### Task 6: `dequant.hpp/.cpp` — the single shared dequantiser

**Files:**
- Create: `include/sapient/core/dequant.hpp`, `src/dequant.cpp`; Test: `tests/dequant_test.cpp`; Modify: `CMakeLists.txt`.

**Interfaces:**
- Consumes: `f16_le_to_f32` (Task 4), block constants (Task 2).
- Produces (namespace `sapient::core::dequant`): `struct ScaleMin { uint8_t sc; uint8_t m; }; ScaleMin get_scale_min_k4(size_t j, const uint8_t* scales);` block functions writing exactly one block: `void q4_0_block(const uint8_t* block, float* out /*32*/)`, `q8_0_block` (32), `q4_k_block` (256), `q5_k_block` (256), `q6_k_block` (256); whole-tensor helpers `void q4_0(std::span<const uint8_t> bytes, size_t numel, float* out)`, `q8_0`, `q4_k`, `q5_k`, `q6_k` (iterate `bytes.size() / block_bytes` blocks; `numel` is the capacity of `out`, extra `out` stays 0); `void q4_k_r4(std::span<const uint8_t> bytes, size_t rows, size_t k, float* out)`, `q6_k_r4` (R4 de-permutation `off = ((g*4+r)*nb + b) * 256`). Plans B, C, D reuse these; the kernels' dot products do NOT (they keep their own accumulation order).

- [ ] **Step 1: Failing tests** — `tests/dequant_test.cpp`

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

#include "sapient/core/dequant.hpp"
#include "sapient/core/f16.hpp"

using namespace sapient::core;
using namespace sapient::core::dequant;

TEST(Dequant, get_scale_min_k4_unpacks_six_bit_pairs) {
    // scales[j] & 63 for j<4, scales[j+4] & 63 for the mins; j>=4 uses the split-nibble form.
    uint8_t s[12] = {0x3F, 0x01, 0x02, 0x03, 0x3F, 0x11, 0x12, 0x13, 0xAB, 0xCD, 0xEF, 0x21};
    EXPECT_EQ(get_scale_min_k4(0, s).sc, 63);
    EXPECT_EQ(get_scale_min_k4(0, s).m, 63);
    EXPECT_EQ(get_scale_min_k4(1, s).sc, 1);
    EXPECT_EQ(get_scale_min_k4(1, s).m, 0x11);
    // j = 4: sc = (s[8] & 0x0F) | ((s[0] >> 6) << 4) = 0x0B | 0 = 11; m = (s[8] >> 4) | ((s[4] >> 6) << 4) = 0x0A | 0 = 10
    EXPECT_EQ(get_scale_min_k4(4, s).sc, 11);
    EXPECT_EQ(get_scale_min_k4(4, s).m, 10);
    // j = 7: sc = (s[11] & 0x0F) | ((s[3] >> 6) << 4) = 1; m = (s[11] >> 4) | ((s[7] >> 6) << 4) = 2
    EXPECT_EQ(get_scale_min_k4(7, s).sc, 1);
    EXPECT_EQ(get_scale_min_k4(7, s).m, 2);
}

TEST(Dequant, q4_0_block_split_nibble_order) {
    uint8_t block[18] = {};
    f16_to_le(f32_to_f16_bits(0.5f), block);  // d = 0.5
    block[2] = 0xF0;                          // lo nibble 0 → elem 0 = (0-8)*0.5 = -4; hi nibble 15 → elem 16 = (15-8)*0.5 = 3.5
    float out[32];
    q4_0_block(block, out);
    EXPECT_EQ(out[0], -4.0f);
    EXPECT_EQ(out[16], 3.5f);
    EXPECT_EQ(out[1], -4.0f);  // zero nibble → -8 * 0.5
}

TEST(Dequant, q8_0_block) {
    uint8_t block[34] = {};
    f16_to_le(f32_to_f16_bits(2.0f), block);
    block[2] = static_cast<uint8_t>(int8_t{-3});
    block[33] = 127;
    float out[32];
    q8_0_block(block, out);
    EXPECT_EQ(out[0], -6.0f);
    EXPECT_EQ(out[31], 254.0f);
}

TEST(Dequant, q4_k_block_scale_and_min) {
    uint8_t block[144] = {};
    f16_to_le(f32_to_f16_bits(1.0f), block);      // d = 1
    f16_to_le(f32_to_f16_bits(0.5f), block + 2);  // dmin = 0.5
    block[4] = 2;  // sc for sub-block 0 = 2
    block[8] = 1;  // min for sub-block 0 = 1  → m1v = 0.5
    block[16] = 0x3A;  // qs[0]: lo nibble 0xA=10 → out[0] = 2*10 - 0.5 = 19.5 ; hi nibble 3 → out[32] uses (sc,m) of sub-block 1 = (0,0) → 0
    float out[256];
    q4_k_block(block, out);
    EXPECT_EQ(out[0], 19.5f);
    EXPECT_EQ(out[1], -0.5f);   // nibble 0 → 2*0 - 0.5
    EXPECT_EQ(out[32], 0.0f);   // sub-block 1: d*0*3 - 0.5*0
}

TEST(Dequant, q6_k_block_scale_indexing) {
    // d = 1; scales[0..16] = 1..16; all q = 32 + i via ql/qh so that (q-32) selects the scale index visibly.
    uint8_t block[210] = {};
    f16_to_le(f32_to_f16_bits(1.0f), block + 208);
    for (int i = 0; i < 16; ++i) block[192 + i] = static_cast<uint8_t>(i + 1);
    // Make q1 = 33 for l = 0 (ql low nibble 1, qh bits 10 = 0b10 → +32): ql[0] = 0x01, qh[0] = 0x02
    block[0] = 0x01;
    block[128] = 0x02;
    float out[256];
    q6_k_block(block, out);
    // out[0] = d * sc[sc_base + is + 0] * (q1 - 32) with is = 0 → sc[0] = 1 → 1 * 1 * 1 = 1
    EXPECT_EQ(out[0], 1.0f);
    // out[16]: l = 16 → is = 1 → sc[1] = 2; q1 for l=16: ql[16] = 0, qh[16] = 0 → q = 0 - 32 → 2 * -32 = -64
    EXPECT_EQ(out[16], -64.0f);
    // out[32] (sub 1, l=0): sc[0 + 0 + 2] = 3; q2 = ((ql[32] & 0xF) | (((qh[0] >> 2) & 3) << 4)) - 32 = -32 → -96
    EXPECT_EQ(out[32], -96.0f);
    // second 128-half: sc_base = 8 → out[128] uses sc[8] = 9, q = -32 → -288
    EXPECT_EQ(out[128], -288.0f);
}

TEST(Dequant, r4_depermutation_matches_row_major) {
    // Two rows, one super-block each (k = 256). Packed R4 layout for rows 0..3 requires 4 rows; use 4 rows.
    const size_t rows = 4, k = 256, nb = k / 256;
    std::vector<uint8_t> plain(rows * nb * 144, 0), packed(rows * nb * 144, 0);
    for (size_t r = 0; r < rows; ++r) {  // distinct d per row so rows are distinguishable
        f16_to_le(f32_to_f16_bits(static_cast<float>(r + 1)), plain.data() + r * 144);
        plain[r * 144 + 4] = 1;   // sc0 = 1
        plain[r * 144 + 16] = 1;  // qs[0] lo nibble = 1 → out[row*256] = (r+1)*1*1 - 0
    }
    // repack: dst block index g*4*nb + b*4 + r ← src (g*4 + r)*nb + b  (single group, nb = 1)
    for (size_t r = 0; r < rows; ++r) std::copy_n(plain.data() + r * 144, 144, packed.data() + r * 144);
    std::vector<float> a(rows * k), b(rows * k);
    q4_k(plain, rows * k, a.data());
    q4_k_r4(packed, rows, k, b.data());
    EXPECT_EQ(a, b);
    EXPECT_EQ(b[0], 1.0f);
    EXPECT_EQ(b[3 * 256], 4.0f);
}
```

- [ ] **Step 2: Run to verify it fails** — configure: `Cannot find source file: src/dequant.cpp`.

- [ ] **Step 3: Write `dequant.hpp`**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// The ONE dequantiser (spec §2.1): Rust keeps three hand-synced copies of this arithmetic
// (sapient-core tensor.rs, sapient-io gguf.rs, sapient-backends-cpu quant.rs); the C++ tree keeps
// this file and everyone calls it. Arithmetic is transcribed from tensor.rs:362-576 — keep the
// evaluation order (e.g. `(d * sc) * q`, `d1 * nib - m1v`), it decides bit-identity.

#include <cstddef>
#include <cstdint>
#include <span>

namespace sapient::core::dequant {

struct ScaleMin { uint8_t sc; uint8_t m; };
/// 6-bit (scale, min) pair `j` (0..8) from the 12 packed scale bytes of a Q4_K/Q5_K block.
ScaleMin get_scale_min_k4(size_t j, const uint8_t* scales);

void q4_0_block(const uint8_t* block, float* out);  // 18 B → 32 values, ggml split-nibble order
void q8_0_block(const uint8_t* block, float* out);  // 34 B → 32
void q4_k_block(const uint8_t* block, float* out);  // 144 B → 256
void q5_k_block(const uint8_t* block, float* out);  // 176 B → 256 (per-ELEMENT 5th bit)
void q6_k_block(const uint8_t* block, float* out);  // 210 B → 256 (+0/+2/+4/+6 scale indexing)

/// Whole tensors: `bytes.size() / block_bytes` blocks in order; `numel` is the capacity of `out`.
void q4_0(std::span<const uint8_t> bytes, size_t numel, float* out);
void q8_0(std::span<const uint8_t> bytes, size_t numel, float* out);
void q4_k(std::span<const uint8_t> bytes, size_t numel, float* out);
void q5_k(std::span<const uint8_t> bytes, size_t numel, float* out);
void q6_k(std::span<const uint8_t> bytes, size_t numel, float* out);
/// Row-interleaved layouts (2-D [rows, k]): packed block p → row g*4+r, block b; out[(row*nb + b)*256..].
void q4_k_r4(std::span<const uint8_t> bytes, size_t rows, size_t k, float* out);
void q6_k_r4(std::span<const uint8_t> bytes, size_t rows, size_t k, float* out);

}  // namespace sapient::core::dequant
```

- [ ] **Step 4: Write `src/dequant.cpp`**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/core/dequant.hpp"

#include "sapient/core/dtype.hpp"
#include "sapient/core/f16.hpp"
#include "sapient/core/panic.hpp"

namespace sapient::core::dequant {

ScaleMin get_scale_min_k4(size_t j, const uint8_t* s) {
    if (j < 4) return {static_cast<uint8_t>(s[j] & 63), static_cast<uint8_t>(s[j + 4] & 63)};
    return {static_cast<uint8_t>((s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4)),
            static_cast<uint8_t>((s[j + 4] >> 4) | ((s[j] >> 6) << 4))};
}

void q4_0_block(const uint8_t* block, float* out) {
    const float d = f16_le_to_f32(block);
    for (size_t j = 0; j < 16; ++j) {
        const uint8_t byte = block[2 + j];
        const int32_t lo = static_cast<int32_t>(byte & 0x0F) - 8;
        const int32_t hi = static_cast<int32_t>(byte >> 4) - 8;
        out[j] = static_cast<float>(lo) * d;
        out[j + 16] = static_cast<float>(hi) * d;
    }
}

void q8_0_block(const uint8_t* block, float* out) {
    const float d = f16_le_to_f32(block);
    for (size_t j = 0; j < 32; ++j) out[j] = static_cast<float>(static_cast<int8_t>(block[2 + j])) * d;
}

void q4_k_block(const uint8_t* block, float* out) {
    const float d = f16_le_to_f32(block);
    const float dmin = f16_le_to_f32(block + 2);
    const uint8_t* scales = block + 4;
    const uint8_t* qs = block + 16;
    size_t out_idx = 0, q_off = 0, is = 0;
    for (int g = 0; g < 4; ++g) {
        const ScaleMin a = get_scale_min_k4(is, scales);
        const ScaleMin b = get_scale_min_k4(is + 1, scales);
        const float d1 = d * static_cast<float>(a.sc), m1v = dmin * static_cast<float>(a.m);
        const float d2 = d * static_cast<float>(b.sc), m2v = dmin * static_cast<float>(b.m);
        for (size_t l = 0; l < 32; ++l) {
            out[out_idx + l] = d1 * static_cast<float>(qs[q_off + l] & 0x0F) - m1v;
            out[out_idx + l + 32] = d2 * static_cast<float>(qs[q_off + l] >> 4) - m2v;
        }
        out_idx += 64; q_off += 32; is += 2;
    }
}

void q5_k_block(const uint8_t* block, float* out) {
    const float d = f16_le_to_f32(block);
    const float dmin = f16_le_to_f32(block + 2);
    const uint8_t* scales = block + 4;
    const uint8_t* qh = block + 16;
    const uint8_t* ql = block + 48;
    size_t out_idx = 0, ql_off = 0, is = 0;
    uint8_t u1 = 1, u2 = 2;
    for (int g = 0; g < 4; ++g) {
        const ScaleMin a = get_scale_min_k4(is, scales);
        const ScaleMin b = get_scale_min_k4(is + 1, scales);
        const float d1 = d * static_cast<float>(a.sc), m1v = dmin * static_cast<float>(a.m);
        const float d2 = d * static_cast<float>(b.sc), m2v = dmin * static_cast<float>(b.m);
        for (size_t l = 0; l < 32; ++l) {
            const float hi = (qh[l] & u1) ? 16.0f : 0.0f;  // per-ELEMENT high bit (the fixed form)
            out[out_idx + l] = d1 * (static_cast<float>(ql[ql_off + l] & 0x0F) + hi) - m1v;
            const float hi2 = (qh[l] & u2) ? 16.0f : 0.0f;
            out[out_idx + l + 32] = d2 * (static_cast<float>(ql[ql_off + l] >> 4) + hi2) - m2v;
        }
        out_idx += 64; ql_off += 32; is += 2;
        if (is % 8 == 0) { u1 = 1; u2 = 2; } else { u1 = static_cast<uint8_t>(u1 << 2); u2 = static_cast<uint8_t>(u2 << 2); }
    }
}

void q6_k_block(const uint8_t* block, float* out) {
    const uint8_t* ql = block;
    const uint8_t* qh = block + 128;
    const uint8_t* sc = block + 192;
    const float d = f16_le_to_f32(block + 208);
    size_t out_idx = 0, ql_off = 0, qh_off = 0, sc_base = 0;
    for (int half = 0; half < 2; ++half) {
        for (size_t l = 0; l < 32; ++l) {
            const size_t is = l / 16;
            const int32_t q1 = static_cast<int32_t>((ql[ql_off + l] & 0x0F) | ((qh[qh_off + l] & 3) << 4)) - 32;
            const int32_t q2 = static_cast<int32_t>((ql[ql_off + l + 32] & 0x0F) | (((qh[qh_off + l] >> 2) & 3) << 4)) - 32;
            const int32_t q3 = static_cast<int32_t>((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) - 32;
            const int32_t q4 = static_cast<int32_t>((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) - 32;
            out[out_idx + l] = d * static_cast<float>(static_cast<int8_t>(sc[sc_base + is])) * static_cast<float>(q1);
            out[out_idx + l + 32] = d * static_cast<float>(static_cast<int8_t>(sc[sc_base + is + 2])) * static_cast<float>(q2);
            out[out_idx + l + 64] = d * static_cast<float>(static_cast<int8_t>(sc[sc_base + is + 4])) * static_cast<float>(q3);
            out[out_idx + l + 96] = d * static_cast<float>(static_cast<int8_t>(sc[sc_base + is + 6])) * static_cast<float>(q4);
        }
        out_idx += 128; ql_off += 64; qh_off += 32; sc_base += 8;
    }
}

namespace {
template <size_t BlockBytes, size_t BlockNumel, void (*Fn)(const uint8_t*, float*)>
void blocks(std::span<const uint8_t> bytes, size_t numel, float* out) {
    const size_t nblocks = bytes.size() / BlockBytes;
    if (nblocks * BlockNumel > numel) panic("dequant: block count exceeds output capacity");
    for (size_t b = 0; b < nblocks; ++b) Fn(bytes.data() + b * BlockBytes, out + b * BlockNumel);
}
template <size_t BlockBytes, void (*Fn)(const uint8_t*, float*)>
void r4(std::span<const uint8_t> bytes, size_t rows, size_t k, float* out) {
    const size_t nb = k / 256;
    const size_t nblocks = bytes.size() / BlockBytes;
    if (nblocks * 256 > rows * k) panic("dequant r4: block count exceeds output capacity");
    for (size_t p = 0; p < nblocks; ++p) {
        const size_t g = p / (4 * nb), rem = p % (4 * nb), b = rem / 4, r = rem % 4;
        const size_t row = g * 4 + r;
        Fn(bytes.data() + p * BlockBytes, out + (row * nb + b) * 256);
    }
}
}  // namespace

void q4_0(std::span<const uint8_t> bytes, size_t numel, float* out) { blocks<Q4_0_BLOCK_BYTES, 32, q4_0_block>(bytes, numel, out); }
void q8_0(std::span<const uint8_t> bytes, size_t numel, float* out) { blocks<Q8_0_BLOCK_BYTES, 32, q8_0_block>(bytes, numel, out); }
void q4_k(std::span<const uint8_t> bytes, size_t numel, float* out) { blocks<Q4_K_BLOCK_BYTES, 256, q4_k_block>(bytes, numel, out); }
void q5_k(std::span<const uint8_t> bytes, size_t numel, float* out) { blocks<Q5_K_BLOCK_BYTES, 256, q5_k_block>(bytes, numel, out); }
void q6_k(std::span<const uint8_t> bytes, size_t numel, float* out) { blocks<Q6_K_BLOCK_BYTES, 256, q6_k_block>(bytes, numel, out); }
void q4_k_r4(std::span<const uint8_t> bytes, size_t rows, size_t k, float* out) { r4<Q4_K_BLOCK_BYTES, q4_k_block>(bytes, rows, k, out); }
void q6_k_r4(std::span<const uint8_t> bytes, size_t rows, size_t k, float* out) { r4<Q6_K_BLOCK_BYTES, q6_k_block>(bytes, rows, k, out); }

}  // namespace sapient::core::dequant
```

- [ ] **Step 5: CMake** — add `src/dequant.cpp`, `tests/dequant_test.cpp`.
- [ ] **Step 6: Build and run** — `ctest --preset dev -R Dequant` → 6 pass. If `q6_k_block_scale_indexing` disagrees with your hand trace, re-derive from `tensor.rs:534-565` — the test values were derived from that code, not from ggml docs.
- [ ] **Step 7: Commit** — `git commit -m "cpp(core): shared dequantiser (Q4_0/Q8_0/Q4_K/Q5_K/Q6_K + R4) transcribed from tensor.rs"` (+ trailer).

---

### Task 7: `tensor.hpp/.cpp`

**Files:**
- Create: `include/sapient/core/tensor.hpp`, `src/tensor.cpp`; Test: `tests/tensor_test.cpp`; Modify: `CMakeLists.txt`.

**Interfaces:**
- Consumes: everything from Tasks 1–6.
- Produces:
  ```cpp
  struct F32Cow { std::span<const float> view; std::vector<float> owned; bool is_owned; std::span<const float> get() const; };
  class Tensor {
    static Result<Tensor> zeros(Shape, DType); from_f32_vec(std::vector<float>&&, Shape); from_f32(std::span<const float>, Shape);
    from_bf16_bytes(std::span<const uint8_t>, Shape); from_f16_bytes(std::span<const uint8_t>, Shape);
    from_quant_bytes(std::span<const uint8_t>, Shape, DType); scalar_f32(float); from_buffer(Shape, DType, BufferHandle, size_t offset);
    const Shape& shape() const; DType dtype() const; size_t ndim() const; size_t numel() const; std::span<const size_t> strides() const;
    const BufferHandle& buffer() const; size_t offset() const; bool is_scalar() const; bool is_contiguous() const; bool is_mmap() const;
    std::span<const uint8_t> bytes() const;          // quant: bounded by byte_count(numel); float: unbounded to buffer end
    std::span<const uint8_t> quant_blocks() const;   // panics unless quantized
    std::span<const float> f32_slice() const;        // panics unless F32; unbounded
    std::vector<float> to_contiguous_f32_vec() const; F32Cow to_f32_cow() const; std::vector<float> to_f32_vec() const;
    Result<Tensor> to_f32_tensor() const;
    Result<std::span<uint8_t>> bytes_mut();          // bounded for all dtypes; needs use_count()==1; mmap → panic inside the buffer
    Result<std::span<float>> f32_slice_mut();        // unbounded
    Result<Tensor> reshape(Shape) const; Result<Tensor> t() const; Result<Tensor> slice_axis(size_t axis, size_t start, size_t end) const;
    size_t byte_size() const; std::string to_string() const; };
  struct TensorMeta { Shape shape; DType dtype; static TensorMeta of(const Tensor&); };
  ```

- [ ] **Step 1: Failing tests** — `tests/tensor_test.cpp`

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "sapient/core/f16.hpp"
#include "sapient/core/tensor.hpp"

using namespace sapient::core;

// Rust: zeros_dtype_shape
TEST(Tensor, zeros_dtype_shape) {
    auto t = Tensor::zeros({2, 3}, DType::F32);
    ASSERT_TRUE(t.has_value());
    EXPECT_EQ(t->shape().dims, (std::vector<size_t>{2, 3}));
    EXPECT_EQ(t->dtype(), DType::F32);
    EXPECT_EQ(t->numel(), 6u);
}
// Rust: from_f32_roundtrip
TEST(Tensor, from_f32_roundtrip) {
    const float data[] = {1, 2, 3, 4, 5, 6};
    auto t = Tensor::from_f32(data, {2, 3});
    ASSERT_TRUE(t.has_value());
    const auto s = t->f32_slice();
    ASSERT_EQ(s.size(), 6u);
    for (size_t i = 0; i < 6; ++i) EXPECT_EQ(s[i], data[i]);
}
// Rust: reshape_preserves_data
TEST(Tensor, reshape_preserves_data) {
    const float data[] = {1, 2, 3, 4, 5, 6};
    auto t = Tensor::from_f32(data, {2, 3});
    auto r = t->reshape({3, 2});
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->shape().dims, (std::vector<size_t>{3, 2}));
    EXPECT_EQ(r->f32_slice()[5], 6.0f);
    EXPECT_EQ(r->buffer().get(), t->buffer().get());  // shared buffer, no copy
}
// Rust: reshape_wrong_numel
TEST(Tensor, reshape_wrong_numel) {
    auto t = Tensor::zeros({2, 3}, DType::F32);
    EXPECT_FALSE(t->reshape({5}).has_value());
}
// Rust: transpose_2d
TEST(Tensor, transpose_2d) {
    auto t = Tensor::zeros({3, 4}, DType::F32);
    auto tt = t->t();
    ASSERT_TRUE(tt.has_value());
    EXPECT_EQ(tt->shape().dims, (std::vector<size_t>{4, 3}));
    EXPECT_EQ(tt->strides()[0], 1u);
    EXPECT_EQ(tt->strides()[1], 4u);
    EXPECT_FALSE(tt->is_contiguous());
    EXPECT_EQ(Tensor::zeros({3}, DType::F32)->t().error().to_string(), "Internal error: t() requires a 2-D tensor");
}
// Rust: byte_size
TEST(Tensor, byte_size) {
    EXPECT_EQ(Tensor::zeros({4, 4}, DType::F32)->byte_size(), 64u);
}
// Rust: q5_k_dequant_high_bits_per_element
TEST(Tensor, q5_k_dequant_high_bits_per_element) {
    std::vector<uint8_t> block(176, 0);
    f16_to_le(f32_to_f16_bits(1.0f), block.data());      // d = 1
    f16_to_le(f32_to_f16_bits(0.0f), block.data() + 2);  // dmin = 0
    // Rust (tensor.rs:797-825) makes every (sc, m) pair unpack to sc = 1, m = 0:
    // scales[0..4] = 1 (sc for j < 4), scales[8..12] = 1 (low nibble → sc for j >= 4), scales[4..8] = 0 (mins).
    for (int j = 0; j < 4; ++j) { block[4 + j] = 1; block[12 + j] = 1; }
    block[16 + 5] = 0b0000'0001;  // qh[5] bit 0 → element 5 gets +16
    block[16 + 0] = 0b0000'0100;  // qh[0] bit 2 → element 64 (group 1, u1 = 4) gets +16
    block[48 + 3] = 0x02;         // ql[3] low nibble 2 → element 3 = 2
    auto t = Tensor::from_quant_bytes(block, {256}, DType::Q5_K);
    ASSERT_TRUE(t.has_value());
    const auto out = t->to_f32_vec();
    ASSERT_EQ(out.size(), 256u);
    EXPECT_EQ(out[3], 2.0f);
    EXPECT_EQ(out[5], 16.0f);
    EXPECT_EQ(out[64], 16.0f);
    for (size_t i = 0; i < 256; ++i) {
        if (i == 3 || i == 5 || i == 64) continue;
        EXPECT_EQ(out[i], 0.0f) << "element " << i;
    }
}
TEST(Tensor, bytes_bounded_for_quant_unbounded_for_float) {
    auto buf = *CpuBuffer::with_capacity(34 * 2 + 10, 16);  // 10 extra bytes after two Q8_0 blocks
    auto q = Tensor::from_buffer({64}, DType::Q8_0, buf, 0);
    ASSERT_TRUE(q.has_value());
    EXPECT_EQ(q->bytes().size(), 68u);        // bounded by byte_count(64)
    auto f = Tensor::from_buffer({4}, DType::F32, buf, 0);
    ASSERT_TRUE(f.has_value());
    EXPECT_EQ(f->bytes().size(), 78u);        // unbounded: to the end of the buffer (Rust parity)
    EXPECT_EQ(f->f32_slice().size(), 19u);    // 78 / 4 truncating
    EXPECT_EQ(Tensor::from_buffer({64}, DType::Q8_0, buf, 20).error().to_string(),
              "Buffer size mismatch: expected 88 bytes, got 78");
}
TEST(Tensor, bytes_mut_requires_exclusive_ownership) {
    auto t = *Tensor::zeros({4}, DType::F32);
    Tensor alias = t;  // shares the buffer
    EXPECT_EQ(t.bytes_mut().error().to_string(), "Internal error: Cannot mutate shared tensor buffer");
    alias = *Tensor::zeros({1}, DType::F32);
    auto m = t.bytes_mut();
    ASSERT_TRUE(m.has_value());
    EXPECT_EQ(m->size(), 16u);
    (*m)[0] = 1;
    EXPECT_EQ(t.bytes()[0], 1);
}
TEST(Tensor, slice_axis_and_contiguous_gather) {
    const float data[] = {1, 2, 3, 4, 5, 6};
    auto t = *Tensor::from_f32(data, {2, 3});
    auto s = t.slice_axis(0, 1, 2);
    ASSERT_TRUE(s.has_value());
    EXPECT_EQ(s->shape().dims, (std::vector<size_t>{1, 3}));
    EXPECT_EQ(s->offset(), 12u);  // 1 * stride(3) * 4 bytes
    EXPECT_FALSE(s->is_contiguous());
    EXPECT_EQ(s->to_contiguous_f32_vec(), (std::vector<float>{4, 5, 6}));
    auto tt = *t.t();
    EXPECT_EQ(tt.to_contiguous_f32_vec(), (std::vector<float>{1, 4, 2, 5, 3, 6}));
    EXPECT_EQ(t.slice_axis(2, 0, 1).error().to_string(), "Internal error: slice axis out of bounds");
}
TEST(Tensor, f16_bf16_and_cow) {
    const uint8_t h[] = {0x00, 0x3C, 0x00, 0xC0};  // 1.0, -2.0 as f16 LE
    auto t = *Tensor::from_f16_bytes(h, {2});
    EXPECT_EQ(t.to_f32_vec(), (std::vector<float>{1.0f, -2.0f}));
    EXPECT_EQ(t.dtype(), DType::F16);
    auto cow = t.to_f32_cow();
    EXPECT_TRUE(cow.is_owned);
    auto f = *Tensor::from_f32(std::vector<float>{3.0f}, {1});
    auto cow2 = f.to_f32_cow();
    EXPECT_FALSE(cow2.is_owned);
    EXPECT_EQ(cow2.get()[0], 3.0f);
    EXPECT_EQ(Tensor::from_f16_bytes(h, {3}).error().to_string(), "Shape mismatch: expected [3], got [2]");
    EXPECT_EQ(t.to_string(), "Tensor(shape=[2], dtype=f16, device=cpu)");
}
```

- [ ] **Step 2: Run to verify it fails** — configure: `Cannot find source file: src/tensor.cpp`.

- [ ] **Step 3: Write `tensor.hpp`**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-core/src/tensor.rs. Invariants that later layers rely on:
//   * `bytes()` is BOUNDED by byte_count(numel) for quantized dtypes and UNBOUNDED (to the end of the
//     buffer) for float dtypes — zero-copy MoE expert views depend on the quant bound (CLAUDE.md).
//   * `bytes_mut()` is bounded for all dtypes and needs exclusive ownership of the buffer.
//   * `reshape`/`t()`/`slice_axis` are views (shared buffer); `slice_axis` multiplies element
//     strides by element_size(), which is 0 for quantized dtypes (documented Rust trap, kept).
//   * `strides` are in elements, `offset` is in bytes.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "sapient/core/buffer.hpp"
#include "sapient/core/dtype.hpp"
#include "sapient/core/error.hpp"
#include "sapient/core/shape.hpp"

namespace sapient::core {

/// Rust `Cow<'_, [f32]>`: a borrowed view when the tensor is already F32, else an owned copy.
struct F32Cow {
    std::span<const float> view;
    std::vector<float> owned;
    bool is_owned{false};
    std::span<const float> get() const { return is_owned ? std::span<const float>(owned) : view; }
};

class Tensor {
public:
    static Result<Tensor> zeros(Shape shape, DType dtype);
    static Result<Tensor> from_f32_vec(std::vector<float>&& data, Shape shape);   // zero-copy (moved)
    static Result<Tensor> from_f32(std::span<const float> data, Shape shape);     // copy, align 64
    static Result<Tensor> from_bf16_bytes(std::span<const uint8_t> data, Shape shape);
    static Result<Tensor> from_f16_bytes(std::span<const uint8_t> data, Shape shape);
    static Result<Tensor> from_quant_bytes(std::span<const uint8_t> data, Shape shape, DType dtype);
    static Result<Tensor> scalar_f32(float v);
    static Result<Tensor> from_buffer(Shape shape, DType dtype, BufferHandle buffer, size_t offset);

    const Shape& shape() const { return shape_; }
    DType dtype() const { return dtype_; }
    size_t ndim() const { return shape_.ndim(); }
    size_t numel() const { return shape_.numel(); }
    std::span<const size_t> strides() const { return strides_; }
    const BufferHandle& buffer() const { return buffer_; }
    size_t offset() const { return offset_; }
    bool is_scalar() const { return shape_.is_scalar() || numel() == 1; }
    bool is_contiguous() const { return strides_ == shape_.strides() && offset_ == 0; }
    bool is_mmap() const { return buffer_->is_mmap(); }

    std::span<const uint8_t> bytes() const;
    std::span<const uint8_t> quant_blocks() const;  // panics unless is_quantized(dtype)
    std::span<const float> f32_slice() const;       // panics unless F32 and (len % 4 == 0)
    std::vector<float> to_contiguous_f32_vec() const;
    F32Cow to_f32_cow() const;
    std::vector<float> to_f32_vec() const;
    Result<Tensor> to_f32_tensor() const;

    Result<std::span<uint8_t>> bytes_mut();
    Result<std::span<float>> f32_slice_mut();

    Result<Tensor> reshape(Shape new_shape) const;
    Result<Tensor> t() const;
    Result<Tensor> slice_axis(size_t axis, size_t start, size_t end) const;

    size_t byte_size() const { return byte_count(dtype_, numel()); }
    std::string to_string() const;  // "Tensor(shape=[2, 3], dtype=f32, device=cpu)"

private:
    Tensor(Shape shape, DType dtype, std::vector<size_t> strides, BufferHandle buffer, size_t offset)
        : shape_(std::move(shape)), dtype_(dtype), strides_(std::move(strides)), buffer_(std::move(buffer)), offset_(offset) {}
    Shape shape_;
    DType dtype_{DType::F32};
    std::vector<size_t> strides_;  // elements
    BufferHandle buffer_;
    size_t offset_{0};             // bytes
};

struct TensorMeta {
    Shape shape;
    DType dtype;
    static TensorMeta of(const Tensor& t) { return {t.shape(), t.dtype()}; }
};

}  // namespace sapient::core
```

- [ ] **Step 4: Write `src/tensor.cpp`**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/core/tensor.hpp"

#include <algorithm>
#include <cstring>

#include "sapient/core/dequant.hpp"
#include "sapient/core/f16.hpp"
#include "sapient/core/panic.hpp"

namespace sapient::core {

Result<Tensor> Tensor::zeros(Shape shape, DType dtype) {
    SAPIENT_TRY(shape.validate());
    SAPIENT_TRY_ASSIGN(auto buf, CpuBuffer::zeros(shape.numel(), dtype));
    auto strides = shape.strides();
    return Tensor(std::move(shape), dtype, std::move(strides), std::move(buf), 0);
}

Result<Tensor> Tensor::from_f32_vec(std::vector<float>&& data, Shape shape) {
    SAPIENT_TRY(shape.validate());
    if (data.size() != shape.numel()) return tl::unexpected(Error::shape_mismatch(shape.dims, {data.size()}));
    auto strides = shape.strides();
    return Tensor(std::move(shape), DType::F32, std::move(strides), CpuBuffer::from_f32_vec(std::move(data)), 0);
}

Result<Tensor> Tensor::from_f32(std::span<const float> data, Shape shape) {
    SAPIENT_TRY(shape.validate());
    if (data.size() != shape.numel()) return tl::unexpected(Error::shape_mismatch(shape.dims, {data.size()}));
    SAPIENT_TRY_ASSIGN(auto buf, CpuBuffer::from_f32_slice(data));
    auto strides = shape.strides();
    return Tensor(std::move(shape), DType::F32, std::move(strides), std::move(buf), 0);
}

namespace {
Result<Tensor> from_half_bytes(std::span<const uint8_t> data, Shape shape, DType dtype) {
    SAPIENT_TRY(shape.validate());
    if (data.size() != shape.numel() * 2) return tl::unexpected(Error::shape_mismatch(shape.dims, {data.size() / 2}));
    SAPIENT_TRY_ASSIGN(auto buf, CpuBuffer::from_bytes_slice(data));
    return Tensor::from_buffer(std::move(shape), dtype, std::move(buf), 0);
}
}  // namespace

Result<Tensor> Tensor::from_bf16_bytes(std::span<const uint8_t> data, Shape shape) { return from_half_bytes(data, std::move(shape), DType::BF16); }
Result<Tensor> Tensor::from_f16_bytes(std::span<const uint8_t> data, Shape shape) { return from_half_bytes(data, std::move(shape), DType::F16); }

Result<Tensor> Tensor::from_quant_bytes(std::span<const uint8_t> data, Shape shape, DType dtype) {
    if (!is_quantized(dtype))
        return tl::unexpected(Error::type_mismatch("a quantized dtype (Q4_0, Q8_0, Q4_K, Q5_K, Q6_K)", to_string(dtype)));
    SAPIENT_TRY(shape.validate());
    const size_t expected = byte_count(dtype, shape.numel());
    if (data.size() != expected) return tl::unexpected(Error::shape_mismatch({expected}, {data.size()}));
    SAPIENT_TRY_ASSIGN(auto buf, CpuBuffer::from_bytes_slice(data));
    return from_buffer(std::move(shape), dtype, std::move(buf), 0);
}

Result<Tensor> Tensor::scalar_f32(float v) { return from_f32(std::span<const float>(&v, 1), Shape::scalar()); }

Result<Tensor> Tensor::from_buffer(Shape shape, DType dtype, BufferHandle buffer, size_t offset) {
    SAPIENT_TRY(shape.validate());
    const size_t required = byte_count(dtype, shape.numel());
    if (buffer->len() < offset + required) return tl::unexpected(Error::buffer_size_mismatch(offset + required, buffer->len()));
    auto strides = shape.strides();
    return Tensor(std::move(shape), dtype, std::move(strides), std::move(buffer), offset);
}

std::span<const uint8_t> Tensor::bytes() const {
    const auto all = buffer_->bytes();
    if (is_quantized(dtype_)) return all.subspan(offset_, byte_count(dtype_, numel()));
    return all.subspan(offset_);
}

std::span<const uint8_t> Tensor::quant_blocks() const {
    if (!is_quantized(dtype_)) panic("as_quant_blocks() called on a non-quantized tensor");
    return bytes();
}

std::span<const float> Tensor::f32_slice() const {
    if (dtype_ != DType::F32) panic("as_f32_slice() called on a non-F32 tensor");
    const auto b = bytes();
    if (b.size() % 4 != 0) panic("Buffer length not a multiple of 4");
    return {reinterpret_cast<const float*>(b.data()), b.size() / 4};
}

std::vector<float> Tensor::to_f32_vec() const {
    const size_t n = numel();
    switch (dtype_) {
    case DType::F32: { const auto s = f32_slice(); return {s.begin(), s.end()}; }  // whole remaining buffer (Rust parity)
    case DType::BF16: { const auto b = bytes(); std::vector<float> out; out.reserve(b.size() / 2);
        for (size_t i = 0; i + 1 < b.size(); i += 2) out.push_back(bf16_le_to_f32(b.data() + i)); return out; }
    case DType::F16: { const auto b = bytes(); std::vector<float> out; out.reserve(b.size() / 2);
        for (size_t i = 0; i + 1 < b.size(); i += 2) out.push_back(f16_le_to_f32(b.data() + i)); return out; }
    case DType::Q4_0: { std::vector<float> out(n, 0.0f); dequant::q4_0(bytes(), n, out.data()); return out; }
    case DType::Q8_0: { std::vector<float> out(n, 0.0f); dequant::q8_0(bytes(), n, out.data()); return out; }
    case DType::Q4_K: { std::vector<float> out(n, 0.0f); dequant::q4_k(bytes(), n, out.data()); return out; }
    case DType::Q5_K: { std::vector<float> out(n, 0.0f); dequant::q5_k(bytes(), n, out.data()); return out; }
    case DType::Q6_K: { std::vector<float> out(n, 0.0f); dequant::q6_k(bytes(), n, out.data()); return out; }
    case DType::Q4_K_R4: {
        if (shape_.dims.empty()) panic("Q4_K_R4 tensor must be 2-D");
        std::vector<float> out(n, 0.0f);
        dequant::q4_k_r4(bytes(), n / shape_.dims.back(), shape_.dims.back(), out.data());
        return out;
    }
    case DType::Q6_K_R4: {
        if (shape_.dims.empty()) panic("Q6_K_R4 tensor must be 2-D");
        std::vector<float> out(n, 0.0f);
        dequant::q6_k_r4(bytes(), n / shape_.dims.back(), shape_.dims.back(), out.data());
        return out;
    }
    default: panic("to_f32_vec: integer dtypes are not convertible (Rust panics here too)");
    }
}

F32Cow Tensor::to_f32_cow() const {
    F32Cow c;
    if (dtype_ == DType::F32) { c.view = f32_slice(); c.is_owned = false; }
    else { c.owned = to_f32_vec(); c.is_owned = true; }
    return c;
}

std::vector<float> Tensor::to_contiguous_f32_vec() const {
    const size_t n = numel();
    if (is_contiguous()) {
        if (dtype_ == DType::F32) { const auto s = f32_slice(); return {s.begin(), s.begin() + static_cast<std::ptrdiff_t>(n)}; }
        auto v = to_f32_vec();
        v.resize(std::min(n, v.size()));
        return v;
    }
    const std::vector<float> raw = to_f32_vec();  // F32: whole slice; others: dequantised
    const auto& dims = shape_.dims;
    std::vector<float> out(n, 0.0f);
    for (size_t flat = 0; flat < n; ++flat) {
        size_t rem = flat, src = 0;
        for (size_t d = dims.size(); d-- > 0;) { const size_t idx = rem % dims[d]; rem /= dims[d]; src += idx * strides_[d]; }
        out[flat] = src < raw.size() ? raw[src] : 0.0f;  // Rust: raw.get(src).unwrap_or(&0.0)
    }
    return out;
}

Result<Tensor> Tensor::to_f32_tensor() const {
    if (dtype_ == DType::F32) return *this;
    return from_f32(to_f32_vec(), shape_);
}

Result<std::span<uint8_t>> Tensor::bytes_mut() {
    if (buffer_.use_count() != 1) return tl::unexpected(Error::internal("Cannot mutate shared tensor buffer"));
    const size_t end = offset_ + byte_count(dtype_, numel());
    return buffer_->bytes_mut().subspan(offset_, end - offset_);
}

Result<std::span<float>> Tensor::f32_slice_mut() {
    if (dtype_ != DType::F32) return tl::unexpected(Error::internal("Tensor dtype is not F32"));
    if (buffer_.use_count() != 1) return tl::unexpected(Error::internal("Cannot mutate shared tensor buffer"));
    auto b = buffer_->bytes_mut().subspan(offset_);
    if (b.size() % 4 != 0) return tl::unexpected(Error::internal("Buffer length not a multiple of 4"));
    return std::span<float>(reinterpret_cast<float*>(b.data()), b.size() / 4);
}

Result<Tensor> Tensor::reshape(Shape new_shape) const {
    SAPIENT_TRY_ASSIGN(Shape ns, shape_.reshape(std::move(new_shape.dims)));
    auto strides = ns.strides();
    return Tensor(std::move(ns), dtype_, std::move(strides), buffer_, offset_);
}

Result<Tensor> Tensor::t() const {
    if (ndim() != 2) return tl::unexpected(Error::internal("t() requires a 2-D tensor"));
    Shape s({shape_.dims[1], shape_.dims[0]});
    std::vector<size_t> st = {strides_[1], strides_[0]};
    return Tensor(std::move(s), dtype_, std::move(st), buffer_, offset_);
}

Result<Tensor> Tensor::slice_axis(size_t axis, size_t start, size_t end) const {
    if (axis >= ndim()) return tl::unexpected(Error::internal("slice axis out of bounds"));
    if (start > end || end > shape_.dims[axis]) return tl::unexpected(Error::internal("slice range out of bounds"));
    Shape s = shape_;
    s.dims[axis] = end - start;
    // Rust parity: element_size() is 0 for quantized dtypes, so the offset does not move (documented trap).
    const size_t off = offset_ + start * strides_[axis] * element_size(dtype_);
    return Tensor(std::move(s), dtype_, strides_, buffer_, off);
}

std::string Tensor::to_string() const {
    return "Tensor(shape=" + shape_.to_string() + ", dtype=" + std::string(name(dtype_)) + ", device=" +
           std::string(buffer_->device()) + ")";
}

}  // namespace sapient::core
```

Notes for the implementer: (1) the `from_f32(std::vector<float>{3.0f}, {1})` call in the tests relies on `std::span<const float>` binding to a temporary vector — it does; (2) the R4 arms need a 2-D shape for `k`; Rust checks `dims.last()` exists and otherwise panics with the message shown; (3) `Tensor` is copyable (Rust `Clone` = shallow, shared buffer); (4) `-Wshadow`: the parameter named `shape` shadows the member accessor `shape()` — rename parameters to `shp` if Clang complains (it does under `-Wshadow` for member functions vs parameters? No: parameters shadow only variables. Keep `shape`, fix if it fires).

- [ ] **Step 5: CMake** — add `src/tensor.cpp`, `tests/tensor_test.cpp`.
- [ ] **Step 6: Build and run** — `ctest --preset dev -R Tensor` → 11 pass. Then the whole suite: `ctest --preset dev` → 12 (sp0) + 4 + 5 + 7 + 4 + 7 + 6 + 11 = **56 tests, 1 skip**.
- [ ] **Step 7: Commit** — `git commit -m "cpp(core): Tensor — views, bounded/unbounded bytes, exclusive mutation, to_f32_vec via the shared dequantiser"` (+ trailer).

---

### Task 8: `sapient::testing` comparison helpers, dequant golden dumps, golden test

**Files:**
- Create: `cpp/libs/sapient-testing/include/sapient/testing/compare.hpp`, `cpp/libs/sapient-testing/src/compare.cpp`
- Modify: `cpp/libs/sapient-testing/CMakeLists.txt` (add `src/compare.cpp`; link `GTest::gtest` PUBLIC)
- Modify: `crates/sapient-backends/cpu/examples/dump_kernels.rs` (9 new cases in `build_cases`)
- Create: `cpp/libs/sapient-core/tests/golden_dequant_test.cpp`; Modify: `cpp/libs/sapient-core/CMakeLists.txt` (add the test file; link `sapient::testing`)

**Interfaces:**
- Consumes: `read_golden`/`GoldenCase` (sub-project 0), `Tensor`, `f16` (Tasks 4, 7); the Rust `quant::repack_q4_k_rows4/repack_q6_k_rows4` (pub, `crates/sapient-backends/cpu/src/kernels/quant.rs:1229` and `:2000`).
- Produces (namespace `sapient::testing`): `std::optional<std::filesystem::path> golden_dir()` (from `SAPIENT_GOLDEN_DIR`); `std::optional<GoldenCase> load_golden_case(std::string_view name, std::string* why)`; `::testing::AssertionResult bit_identical(std::span<const float> got, std::span<const float> ref)` (compares `std::bit_cast<uint32_t>`, reports the first 5 mismatches with hex bits); `float max_abs_err(std::span<const float>, std::span<const float>)`; `::testing::AssertionResult within_abs(got, ref, float tol)`; macro `SAPIENT_GOLDEN_CASE(var, name)` (declares `const GoldenCase& var`, or `GTEST_SKIP()` with the reason). Plans C/D/E use these. Golden cases `dequant_q4_0`, `dequant_q8_0`, `dequant_q4_k`, `dequant_q5_k`, `dequant_q6_k`, `dequant_q4_k_r4`, `dequant_q6_k_r4` (arrays `in:bytes` u8, `param:shape` u32[2], `out:f32`), `dequant_f16`, `dequant_bf16` (arrays `in:src_f32`, `in:bytes`, `out:f32`). Dump total becomes 31.

- [ ] **Step 1: Write the helpers' failing test** — add to `cpp/libs/sapient-testing/tests/golden_test.cpp` (existing file; append):

```cpp
#include "sapient/testing/compare.hpp"

TEST(Compare, bit_identical_reports_first_mismatch_with_bits) {
    const float a[] = {1.0f, 2.0f, 0.0f};
    const float b[] = {1.0f, 2.0f, -0.0f};  // -0 differs bitwise
    EXPECT_TRUE(sapient::testing::bit_identical(a, a));
    const auto r = sapient::testing::bit_identical(a, b);
    EXPECT_FALSE(r);
    EXPECT_NE(std::string(r.message()).find("[2]"), std::string::npos) << r.message();
    EXPECT_NE(std::string(r.message()).find("0x80000000"), std::string::npos) << r.message();
    const float c[] = {1.0f};
    EXPECT_FALSE(sapient::testing::bit_identical(a, c));  // length mismatch
}

TEST(Compare, within_abs_and_max_abs_err) {
    const float a[] = {1.0f, 2.0f};
    const float b[] = {1.0f, 2.5f};
    EXPECT_EQ(sapient::testing::max_abs_err(a, b), 0.5f);
    EXPECT_TRUE(sapient::testing::within_abs(a, b, 0.5f));
    EXPECT_FALSE(sapient::testing::within_abs(a, b, 0.4f));
}

TEST(Compare, golden_case_macro_skips_without_env) {
    if (std::getenv("SAPIENT_GOLDEN_DIR") != nullptr) GTEST_SKIP() << "env set; the skip path is exercised without it";
    std::string why;
    EXPECT_FALSE(sapient::testing::load_golden_case("does_not_matter", &why).has_value());
    EXPECT_NE(why.find("SAPIENT_GOLDEN_DIR"), std::string::npos);
}
```

- [ ] **Step 2: Run to verify it fails** — `cd cpp && cmake --build --preset dev` → `'sapient/testing/compare.hpp' file not found`.

- [ ] **Step 3: Write `compare.hpp`**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Comparison helpers for the golden-dump gates (spec §4): bit-identical for quant/integer paths,
// absolute-error bounds for the sgemm-backed float paths. Test-support only.

#include <gtest/gtest.h>

#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "sapient/testing/golden.hpp"

namespace sapient::testing {

/// `SAPIENT_GOLDEN_DIR`, if set.
std::optional<std::filesystem::path> golden_dir();
/// `<golden_dir>/<name>.sapd`; on failure `*why` says whether the env var is unset, the file is
/// missing, or the parse failed.
std::optional<GoldenCase> load_golden_case(std::string_view name, std::string* why);

/// Exact comparison on the f32 bit patterns (so NaN payloads and -0 count as differences).
::testing::AssertionResult bit_identical(std::span<const float> got, std::span<const float> ref);
float max_abs_err(std::span<const float> got, std::span<const float> ref);
::testing::AssertionResult within_abs(std::span<const float> got, std::span<const float> ref, float tol);

}  // namespace sapient::testing

/// Bind `var` to the named golden case or skip the test with a visible reason.
#define SAPIENT_GOLDEN_CASE(var, name)                                                   \
    std::string sapient_golden_why_;                                                     \
    auto sapient_golden_opt_ = ::sapient::testing::load_golden_case((name), &sapient_golden_why_); \
    if (!sapient_golden_opt_.has_value()) GTEST_SKIP() << sapient_golden_why_;           \
    const ::sapient::testing::GoldenCase& var = *sapient_golden_opt_
```

- [ ] **Step 4: Write `src/compare.cpp`**

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/testing/compare.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <sstream>

namespace sapient::testing {

std::optional<std::filesystem::path> golden_dir() {
    const char* d = std::getenv("SAPIENT_GOLDEN_DIR");
    if (d == nullptr || *d == '\0') return std::nullopt;
    return std::filesystem::path(d);
}

std::optional<GoldenCase> load_golden_case(std::string_view name, std::string* why) {
    const auto dir = golden_dir();
    if (!dir) { if (why) *why = "SAPIENT_GOLDEN_DIR not set — run cpp/tests/parity/golden_dump.sh <dir> and export it"; return std::nullopt; }
    const auto file = *dir / (std::string(name) + ".sapd");
    if (!std::filesystem::exists(file)) { if (why) *why = "golden case missing: " + file.string() + " (regenerate the dumps with the current Rust tree)"; return std::nullopt; }
    std::string err;
    auto c = read_golden(file, &err);
    if (!c) { if (why) *why = err; return std::nullopt; }
    return c;
}

::testing::AssertionResult bit_identical(std::span<const float> got, std::span<const float> ref) {
    if (got.size() != ref.size())
        return ::testing::AssertionFailure() << "length " << got.size() << " != " << ref.size();
    size_t bad = 0;
    std::ostringstream os;
    for (size_t i = 0; i < got.size(); ++i) {
        const uint32_t g = std::bit_cast<uint32_t>(got[i]);
        const uint32_t r = std::bit_cast<uint32_t>(ref[i]);
        if (g == r) continue;
        if (bad < 5)
            os << "\n  [" << i << "] got " << got[i] << " (0x" << std::hex << std::setw(8) << std::setfill('0') << g
               << ") ref " << std::dec << ref[i] << " (0x" << std::hex << std::setw(8) << std::setfill('0') << r << ")" << std::dec;
        ++bad;
    }
    if (bad == 0) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << bad << " of " << got.size() << " values differ bitwise:" << os.str();
}

float max_abs_err(std::span<const float> got, std::span<const float> ref) {
    float m = 0.0f;
    const size_t n = got.size() < ref.size() ? got.size() : ref.size();
    for (size_t i = 0; i < n; ++i) m = std::fmax(m, std::fabs(got[i] - ref[i]));
    return m;
}

::testing::AssertionResult within_abs(std::span<const float> got, std::span<const float> ref, float tol) {
    if (got.size() != ref.size())
        return ::testing::AssertionFailure() << "length " << got.size() << " != " << ref.size();
    const float m = max_abs_err(got, ref);
    if (m <= tol) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << "max abs err " << m << " > " << tol;
}

}  // namespace sapient::testing
```

CMake (`cpp/libs/sapient-testing/CMakeLists.txt`): `add_library(sapient_testing STATIC src/golden.cpp src/compare.cpp)` and `target_link_libraries(sapient_testing PUBLIC GTest::gtest)`.

- [ ] **Step 5: Run the helper tests** — `cmake --build --preset dev && ctest --preset dev -R Compare` → 3 pass (the third passes or skips visibly depending on the env).

- [ ] **Step 6: Extend the Rust dump tool** — in `crates/sapient-backends/cpu/examples/dump_kernels.rs`, append inside `build_cases` right before `Ok(cases)`:

```rust
    // ── sapient-core dequantisation (Tensor::to_f32_vec) — sub-project 1a plan A gates ──────────
    let (dq_rows, dq_k) = (4usize, 512usize);
    let dq: [(&str, DType, Vec<u8>); 5] = [
        ("dequant_q4_0", DType::Q4_0, q4_0_row(&rng.f32s(dq_rows * dq_k, -1.0, 1.0))),
        ("dequant_q8_0", DType::Q8_0, q8_0_row(&rng.f32s(dq_rows * dq_k, -1.0, 1.0))),
        ("dequant_q4_k", DType::Q4_K, kquant_rows(&mut rng, dq_rows, dq_k, q4_k_block)),
        ("dequant_q5_k", DType::Q5_K, kquant_rows(&mut rng, dq_rows, dq_k, q5_k_block)),
        ("dequant_q6_k", DType::Q6_K, kquant_rows(&mut rng, dq_rows, dq_k, q6_k_block)),
    ];
    for (name, dtype, bytes) in dq {
        let t = Tensor::from_quant_bytes(&bytes, vec![dq_rows, dq_k], dtype)?;
        cases.push(case(
            name,
            vec![
                Array::u8("in:bytes", &[bytes.len()], &bytes),
                Array::u32("param:shape", &[2], &[dq_rows as u32, dq_k as u32]),
                Array::f32("out:f32", &[dq_rows * dq_k], &t.to_f32_vec()),
            ],
        ));
    }
    // Row-interleaved layouts: the backends-cpu repack is the only producer of R4 bytes.
    let plain = kquant_rows(&mut rng, dq_rows, dq_k, q4_k_block);
    let packed = quant::repack_q4_k_rows4(&plain, dq_rows, dq_k);
    let t = Tensor::from_quant_bytes(&packed, vec![dq_rows, dq_k], DType::Q4_K_R4)?;
    cases.push(case(
        "dequant_q4_k_r4",
        vec![
            Array::u8("in:bytes", &[packed.len()], &packed),
            Array::u32("param:shape", &[2], &[dq_rows as u32, dq_k as u32]),
            Array::f32("out:f32", &[dq_rows * dq_k], &t.to_f32_vec()),
        ],
    ));
    let plain = kquant_rows(&mut rng, dq_rows, dq_k, q6_k_block);
    let packed = quant::repack_q6_k_rows4(&plain, dq_rows, dq_k);
    let t = Tensor::from_quant_bytes(&packed, vec![dq_rows, dq_k], DType::Q6_K_R4)?;
    cases.push(case(
        "dequant_q6_k_r4",
        vec![
            Array::u8("in:bytes", &[packed.len()], &packed),
            Array::u32("param:shape", &[2], &[dq_rows as u32, dq_k as u32]),
            Array::f32("out:f32", &[dq_rows * dq_k], &t.to_f32_vec()),
        ],
    ));
    // Half-precision storage: finite values only (NaN payload policy is not a parity target).
    // `in:src_f32` lets the C++ side also check its f32→f16/bf16 narrowing against `half`.
    let src = rng.f32s(64, -100.0, 100.0);
    let f16_bytes: Vec<u8> = src.iter().flat_map(|v| f16::from_f32(*v).to_le_bytes()).collect();
    let t = Tensor::from_f16_bytes(&f16_bytes, vec![64])?;
    cases.push(case(
        "dequant_f16",
        vec![
            Array::f32("in:src_f32", &[64], &src),
            Array::u8("in:bytes", &[128], &f16_bytes),
            Array::f32("out:f32", &[64], &t.to_f32_vec()),
        ],
    ));
    let bf16_bytes: Vec<u8> = src.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect();
    let t = Tensor::from_bf16_bytes(&bf16_bytes, vec![64])?;
    cases.push(case(
        "dequant_bf16",
        vec![
            Array::f32("in:src_f32", &[64], &src),
            Array::u8("in:bytes", &[128], &bf16_bytes),
            Array::f32("out:f32", &[64], &t.to_f32_vec()),
        ],
    ));
```

Then: `cargo fmt --all && cargo clippy -p sapient-backends-cpu --all-targets -- -D warnings` (clean) and regenerate: `cpp/tests/parity/golden_dump.sh /tmp/sapient-golden` → `31 cases`. Update the doc comment at the top of `dump_kernels.rs` that lists the case families (add "sapient-core dequant: dequant_*").

- [ ] **Step 7: Write the C++ golden test** — `cpp/libs/sapient-core/tests/golden_dequant_test.cpp`

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
// Plan A gate: Tensor::to_f32_vec is bit-identical to the Rust oracle for every stored dtype.
#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "sapient/core/f16.hpp"
#include "sapient/core/tensor.hpp"
#include "sapient/testing/compare.hpp"

using namespace sapient::core;
using sapient::testing::bit_identical;

namespace {
struct QuantCase { const char* name; DType dtype; };
}  // namespace

class GoldenDequant : public ::testing::TestWithParam<QuantCase> {};

TEST_P(GoldenDequant, matches_rust_to_f32_vec) {
    SAPIENT_GOLDEN_CASE(c, GetParam().name);
    const auto bytes = c.get("in:bytes").as<uint8_t>();
    const auto shape = c.get("param:shape").as<uint32_t>();
    auto t = Tensor::from_quant_bytes(bytes, Shape({shape[0], shape[1]}), GetParam().dtype);
    ASSERT_TRUE(t.has_value()) << t.error().to_string();
    EXPECT_TRUE(bit_identical(t->to_f32_vec(), c.get("out:f32").as<float>()));
}

INSTANTIATE_TEST_SUITE_P(Quant, GoldenDequant,
                         ::testing::Values(QuantCase{"dequant_q4_0", DType::Q4_0}, QuantCase{"dequant_q8_0", DType::Q8_0},
                                           QuantCase{"dequant_q4_k", DType::Q4_K}, QuantCase{"dequant_q5_k", DType::Q5_K},
                                           QuantCase{"dequant_q6_k", DType::Q6_K}, QuantCase{"dequant_q4_k_r4", DType::Q4_K_R4},
                                           QuantCase{"dequant_q6_k_r4", DType::Q6_K_R4}),
                         [](const ::testing::TestParamInfo<QuantCase>& info) { return std::string(info.param.name); });

TEST(GoldenDequantHalf, f16_widening_and_narrowing_match_the_half_crate) {
    SAPIENT_GOLDEN_CASE(c, "dequant_f16");
    const auto bytes = c.get("in:bytes").as<uint8_t>();
    const auto src = c.get("in:src_f32").as<float>();
    auto t = Tensor::from_f16_bytes(bytes, Shape({64}));
    ASSERT_TRUE(t.has_value());
    EXPECT_TRUE(bit_identical(t->to_f32_vec(), c.get("out:f32").as<float>()));
    for (size_t i = 0; i < src.size(); ++i)
        EXPECT_EQ(f32_to_f16_bits(src[i]), load_le16(bytes.data() + 2 * i)) << "narrowing of " << src[i];
}

TEST(GoldenDequantHalf, bf16_widening_and_narrowing_match_the_half_crate) {
    SAPIENT_GOLDEN_CASE(c, "dequant_bf16");
    const auto bytes = c.get("in:bytes").as<uint8_t>();
    const auto src = c.get("in:src_f32").as<float>();
    auto t = Tensor::from_bf16_bytes(bytes, Shape({64}));
    ASSERT_TRUE(t.has_value());
    EXPECT_TRUE(bit_identical(t->to_f32_vec(), c.get("out:f32").as<float>()));
    for (size_t i = 0; i < src.size(); ++i)
        EXPECT_EQ(f32_to_bf16_bits(src[i]), load_le16(bytes.data() + 2 * i)) << "narrowing of " << src[i];
}
```

CMake (`cpp/libs/sapient-core/CMakeLists.txt`): add `tests/golden_dequant_test.cpp` and link `sapient::testing` into `sapient_core_tests`.

- [ ] **Step 8: Run without and with dumps**

`ctest --preset dev -R Golden` → the 9 new tests report **Skipped** with the `SAPIENT_GOLDEN_DIR` reason; then `SAPIENT_GOLDEN_DIR=/tmp/sapient-golden ctest --preset dev` → **all pass, 0 skipped** (56 + 3 + 9 = 68 tests). If a quant case fails, the report names the first differing index and both bit patterns: compare against the corresponding `dequant.cpp` line and the Rust arm before changing anything.

- [ ] **Step 9: Commit**

```bash
git add cpp/libs/sapient-testing cpp/libs/sapient-core crates/sapient-backends/cpu/examples/dump_kernels.rs
git commit -m "cpp(core): dequant golden gates — 9 Rust dump cases bit-identical incl. R4 layouts and f16/bf16; sapient::testing compare helpers

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Docs, parity ledger, deferred-residual cleanup, final verification

**Files:**
- Modify: `CLAUDE.md`, `docs/ROADMAP.md`, `docs/PARITY.md`, `cpp/third_party/LICENSES.md`, `CHANGELOG.md`, `cpp/CMakeLists.txt` (one comment line)

- [ ] **Step 1: CLAUDE.md** — in the `## C++ rewrite programme` section: (a) in the "Sub-project 0 landed" bullet, after `BuildFlags.FpContractIsOff (…)` add `and its x86-64 twin \`BuildFlags.FpContractIsOffUnderFmaTarget\` (a \`target("fma")\` probe, CI-only)`, and after the `.clang-tidy` mention (or at the end of the bullet) add `; \`.clang-tidy\` excludes \`bugprone-unchecked-optional-access\` (gtest false positive)`; (b) append a new bullet:

```markdown
- **Sub-project 1a, plan A landed (`sapient::core`):** `cpp/libs/sapient-core/` ports `dtype`, `shape`, `buffer`, `tensor`, `error` 1:1 (all 22 Rust tests by name) plus `f16.hpp` (software half/bfloat conversions, exhaustively round-trip tested) and **`dequant.hpp`, the single dequantiser** — Rust keeps three hand-synced copies of the Q5_K/Q6_K arithmetic, C++ keeps one. `Result<T>` is `tl::expected<T, Error>` (`SAPIENT_TRY`), Rust panics are `sapient::core::panic()` (abort). Invariants kept: `bytes()` bounded by `byte_count` for quant dtypes / unbounded for float; `bytes_mut()` needs `use_count()==1`; `slice_axis` uses `element_size()` (0 for quant — the documented trap). Gate: `dequant_*` golden cases (9, incl. R4 layouts and f16/bf16 narrowing vs the `half` crate) bit-identical; helpers `sapient::testing::{bit_identical, within_abs, SAPIENT_GOLDEN_CASE}`. Next: plan C (dense kernels).
```

- [ ] **Step 2: `cpp/CMakeLists.txt`** — change the stale comment on the `include(flags)` line to `include(flags)     # Clang gate (global) + sapient_apply_parity_flags_here() used by libs/`.

- [ ] **Step 3: `docs/ROADMAP.md`** — Phase 7 table row 1a status → `in progress — plan A (core) implemented on feat/cpp-sp1a; plans C, D, E, B pending`.

- [ ] **Step 4: `docs/PARITY.md`** — under "Sub-project 1a": mark gap (c) `closed by plan A (compare.hpp)`; under `### Results` replace `_(no rows yet)_` with:

```markdown
| Date | Gate | Host / ISA path | Rust commit | C++ commit | Result |
|---|---|---|---|---|---|
| 2026-09-DD | 22 sapient-core unit tests ported by name (+ f16 exhaustive round trip, dequant unit tests) | macOS arm64 (Apple M5) | `<rust sha>` | `<cpp sha>` | pass |
| 2026-09-DD | `dequant_{q4_0,q8_0,q4_k,q5_k,q6_k,q4_k_r4,q6_k_r4}` (4×512) + `dequant_{f16,bf16}` (64) vs Rust `Tensor::to_f32_vec` / `half` narrowing | macOS arm64 | `<sha>` | `<sha>` | bit-identical, 9/9 |
```
(fill the date and `git rev-parse --short` of the `dump_kernels` commit and of HEAD).

- [ ] **Step 5: `cpp/third_party/LICENSES.md`** — move `tl::expected` from "Planned" to "Pinned now": `| tl::expected (TartanLlama) | v1.1.0 | CC0-1.0 | sapient::core::Result<T> | cmake/deps.cmake |`.

- [ ] **Step 6: `CHANGELOG.md`** — under `## [Unreleased]` add `### 🧱 C++ rewrite — sub-project 1a, plan A (sapient::core)` with one bullet: `- \`cpp/libs/sapient-core\`: Tensor/DType/Shape/Buffer/Error ported 1:1 with all 22 Rust tests, software f16/bf16, the shared dequantiser, tl::expected-based Result; 9 golden dequant cases bit-identical to the Rust oracle.`

- [ ] **Step 7: Final verification (record every summary line in the report)**

```bash
just cpp-lint && cd cpp && cmake --preset dev && cmake --build --preset dev \
  && ./tests/parity/golden_dump.sh /tmp/sapient-golden \
  && SAPIENT_GOLDEN_DIR=/tmp/sapient-golden ctest --preset dev && cd .. \
  && cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings \
  && cargo test -p sapient-backends-cpu -- --test-threads=1 2>&1 | grep "^test result"
```
Expected: lint OK (SPDX file count grows by the new files); 31 golden cases; **68 tests passed, 0 skipped**; fmt/clippy clean; backends-cpu tests unchanged (71 passed, 1 ignored).

- [ ] **Step 8: Commit**

```bash
git add CLAUDE.md docs/ROADMAP.md docs/PARITY.md cpp/third_party/LICENSES.md CHANGELOG.md cpp/CMakeLists.txt
git commit -m "docs(sp1a-A): sapient::core landed — CLAUDE.md, roadmap, parity ledger, licence inventory

CONTRIBUTING, README and PROJECT_GUIDE need no change for a library-internal plan.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

**Do NOT push.** The user pushes `feat/cpp-sp1a` (stacked on `feat/cpp-sp0-scaffold`).

---

## Self-review against the spec

- **§2.1 file table:** dtype (T2), shape (T3), buffer (T5), tensor (T7), error (T1), f16 (T4), dequant (T6), panic (T1) — all present with the listed behaviours (quant-bounded `bytes()`, `use_count()==1`, `slice_axis` trap + debug note, integer-dtype panic, `from_f32_vec` move, per-constructor alignments 64/64/16/4, R4 names not parseable, `InvalidGraph` text for zero dims, `flat_index` element offset).
- **§3 rules:** software f16 (T4), left-associative `(d*sc)*q` (T6), `int32_t` casts before subtracting (T6), little-endian `static_assert` (T4), no exceptions across boundaries (Result/panic only), byte-identical messages (T1 tests).
- **§4:** 22 tests by name (T1 2, T2 4, T3 6, T5 3, T7 7 — matches the Rust split tensor 7 / shape 6 / dtype 4 / buffer 3 / error 2); dump cases A (T8, 9 cases); `sapient::testing` helpers incl. `expect_bit_identical`→`bit_identical`, `max_abs_err`, `SAPIENT_GOLDEN_CASE` (T8); host note (x86 via CI) unchanged.
- **§5 row A / §8 gaps:** docs updates and the three carried residuals (compare helpers, stale comment, CLAUDE.md omissions) — T9.
- **Placeholder scan:** the `<sha>`/`2026-09-DD` tokens in T9 step 4 are deliberate fill-ins with the command to compute them; no other TBD/TODO.
- **Type consistency:** `Error` factories (`shape_mismatch`, `rank_mismatch`, `type_mismatch`, `broadcast`, `internal`, `buffer_size_mismatch`, `allocation_failed`, `invalid_graph`) used in T2–T7 match T1; `Shape({…})`, `dims`, `strides()`, `validate()`, `reshape(vector)`; `CpuBuffer::{with_capacity, zeros, from_f32_slice, from_bytes_slice, from_f32_vec, f32s, data}`; `f16_le_to_f32`, `f32_to_f16_bits`, `load_le16`, `f16_to_le`; `dequant::{q4_0_block…, q4_0…, q4_k_r4, q6_k_r4}`; `Tensor` API as declared in T7; `bit_identical`, `load_golden_case`, `SAPIENT_GOLDEN_CASE` (T8) — consistent across tasks. Test-count arithmetic: sp0 12 → +4 +5 +7 +4 +7 +6 +11 = 56 → +3 (Compare) +9 (golden) = 68.
