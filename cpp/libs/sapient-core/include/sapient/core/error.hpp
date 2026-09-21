// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
//
// Port of crates/sapient-core/src/error.rs. One error type for the whole C++ tree; the 25
// variants of Rust's `SapientError` become `ErrorCode` + the fields each variant carried.
// `to_string()` reproduces the `#[error("…")]` format strings byte for byte.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <tl/expected.hpp>

namespace sapient::core {

enum class ErrorCode : uint8_t {
    ShapeMismatch,  // {expected: Vec<usize>, got: Vec<usize>}
    RankMismatch,   // {expected: usize, got: usize}
    TypeMismatch,   // {expected: String, got: String}
    BroadcastError, // {lhs: Vec<usize>, rhs: Vec<usize>}
    CyclicGraph,
    NodeNotFound,         // (String)
    InvalidGraph,         // (String)
    ShapeInferenceFailed, // {op, reason}
    UnsupportedOp,        // {backend, op}
    BackendError,         // {backend, message}
    NoBackendAvailable,
    AllocationFailed,   // {bytes, align}
    BufferSizeMismatch, // {expected, got}
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
    TelemetryError, // (String)
    Internal,       // (String)
};

struct Error {
    ErrorCode code{ErrorCode::Internal};
    std::vector<size_t> expected_dims; // ShapeMismatch.expected / BroadcastError.lhs
    std::vector<size_t> got_dims;      // ShapeMismatch.got / BroadcastError.rhs
    size_t expected_n{
        0};          // RankMismatch.expected / AllocationFailed.bytes / BufferSizeMismatch.expected
    size_t got_n{0}; // RankMismatch.got / AllocationFailed.align / BufferSizeMismatch.got
    std::string
        a; // first string field (TypeMismatch.expected, ShapeInferenceFailed.op, UnsupportedOp.backend, BackendError.backend, single-string variants)
    std::string
        b; // second string field (TypeMismatch.got, ShapeInferenceFailed.reason, UnsupportedOp.op, BackendError.message)
    std::error_code io_code; // Io only

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

template <class T> using Result = tl::expected<T, Error>;

/// Rust's `Debug` for `Vec<usize>`: "[2, 3]" ("[]" when empty).
std::string debug_dims(const std::vector<size_t>& dims);

} // namespace sapient::core

#define SAPIENT_CAT_(a, b) a##b
#define SAPIENT_CAT(a, b) SAPIENT_CAT_(a, b)

// `__COUNTER__` (unlike `__LINE__`) increments on every textual expansion, including repeats
// within one macro's own replacement list — so it must be captured into a macro ARGUMENT exactly
// once (here, by the outer SAPIENT_TRY/SAPIENT_TRY_ASSIGN passing it to the _IMPL macro below) and
// then referenced via that argument's already-expanded name, never written as `__COUNTER__`
// itself more than once per invocation.
#define SAPIENT_TRY_IMPL(var, expr)                                                                \
    do {                                                                                           \
        auto&& var = (expr);                                                                       \
        if (!var.has_value()) return ::tl::unexpected(std::move(var.error()));                     \
    } while (0)

/// `expr?` for a Result whose value is discarded (or Result<void>).
#define SAPIENT_TRY(expr) SAPIENT_TRY_IMPL(SAPIENT_CAT(sapient_try_, __COUNTER__), expr)

#define SAPIENT_TRY_ASSIGN_IMPL(var, lhs, expr)                                                    \
    auto&& var = (expr);                                                                           \
    if (!var.has_value()) return ::tl::unexpected(std::move(var.error()));                         \
    lhs = std::move(*var)

/// `lhs = expr?;` — `lhs` may be a declaration (`int v`) or an existing lvalue.
///
/// Must be used as its own statement inside braces (e.g. the body of a function or a `{ }`
/// block), never as the unbraced body of `if`/`for`/`while`: it intentionally declares a
/// temporary (`sapient_try_N`) in the enclosing scope, and an unbraced statement body is its own
/// scope, so the declaration would not outlive the following `lhs = ...` in the way the macro
/// relies on (and, depending on the surrounding code, may not even compile).
#define SAPIENT_TRY_ASSIGN(lhs, expr)                                                              \
    SAPIENT_TRY_ASSIGN_IMPL(SAPIENT_CAT(sapient_try_, __COUNTER__), lhs, expr)
