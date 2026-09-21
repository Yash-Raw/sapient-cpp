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
enum class DType : uint8_t {
    F32,
    F16,
    BF16,
    I32,
    I64,
    U8,
    Bool,
    Q4_0,
    Q8_0,
    Q4_K,
    Q5_K,
    Q6_K,
    Q4_K_R4,
    Q6_K_R4
};

inline constexpr size_t QUANT_BLOCK_SIZE = 32;
inline constexpr size_t K_QUANT_BLOCK_SIZE = 256;
inline constexpr size_t Q4_0_BLOCK_BYTES = 18;
inline constexpr size_t Q8_0_BLOCK_BYTES = 34;
inline constexpr size_t Q4_K_BLOCK_BYTES = 144;
inline constexpr size_t Q5_K_BLOCK_BYTES = 176;
inline constexpr size_t Q6_K_BLOCK_BYTES = 210;

constexpr bool is_quantized(DType d) {
    switch (d) {
    case DType::Q4_0:
    case DType::Q8_0:
    case DType::Q4_K:
    case DType::Q4_K_R4:
    case DType::Q5_K:
    case DType::Q6_K:
    case DType::Q6_K_R4:
        return true;
    default:
        return false;
    }
}
constexpr bool is_float(DType d) {
    return d == DType::F32 || d == DType::F16 || d == DType::BF16;
}
constexpr bool is_integer(DType d) {
    return d == DType::I32 || d == DType::I64 || d == DType::U8 || d == DType::Bool;
}

/// Bytes per element; 0 for quantized dtypes (use byte_count()).
constexpr size_t element_size(DType d) {
    switch (d) {
    case DType::F32:
        return 4;
    case DType::F16:
    case DType::BF16:
        return 2;
    case DType::I32:
        return 4;
    case DType::I64:
        return 8;
    case DType::U8:
    case DType::Bool:
        return 1;
    default:
        return 0;
    }
}
constexpr size_t alignment(DType d) {
    switch (d) {
    case DType::F32:
        return 4;
    case DType::F16:
    case DType::BF16:
        return 2;
    case DType::I32:
        return 4;
    case DType::I64:
        return 8;
    case DType::U8:
    case DType::Bool:
        return 1;
    default:
        return 2; // all quantized
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

} // namespace sapient::core
