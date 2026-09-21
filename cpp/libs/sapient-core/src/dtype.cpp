// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/core/dtype.hpp"

#include <algorithm>
#include <cctype>

#include "sapient/core/panic.hpp"

namespace sapient::core {

size_t block_bytes(DType d) {
    switch (d) {
    case DType::Q4_0:
        return Q4_0_BLOCK_BYTES;
    case DType::Q8_0:
        return Q8_0_BLOCK_BYTES;
    case DType::Q4_K:
    case DType::Q4_K_R4:
        return Q4_K_BLOCK_BYTES;
    case DType::Q5_K:
        return Q5_K_BLOCK_BYTES;
    case DType::Q6_K:
    case DType::Q6_K_R4:
        return Q6_K_BLOCK_BYTES;
    default:
        panic("block_bytes() called on non-quantized dtype");
    }
}

size_t block_numel(DType d) {
    switch (d) {
    case DType::Q4_0:
    case DType::Q8_0:
        return QUANT_BLOCK_SIZE;
    case DType::Q4_K:
    case DType::Q4_K_R4:
    case DType::Q5_K:
    case DType::Q6_K:
    case DType::Q6_K_R4:
        return K_QUANT_BLOCK_SIZE;
    default:
        panic("block_numel() called on non-quantized dtype");
    }
}

size_t byte_count(DType d, size_t numel) {
    switch (d) {
    case DType::Q4_0:
        return numel / QUANT_BLOCK_SIZE * Q4_0_BLOCK_BYTES;
    case DType::Q8_0:
        return numel / QUANT_BLOCK_SIZE * Q8_0_BLOCK_BYTES;
    case DType::Q4_K:
    case DType::Q4_K_R4:
        return numel / K_QUANT_BLOCK_SIZE * Q4_K_BLOCK_BYTES;
    case DType::Q5_K:
        return numel / K_QUANT_BLOCK_SIZE * Q5_K_BLOCK_BYTES;
    case DType::Q6_K:
    case DType::Q6_K_R4:
        return numel / K_QUANT_BLOCK_SIZE * Q6_K_BLOCK_BYTES;
    default:
        return numel * element_size(d);
    }
}

std::string_view name(DType d) {
    switch (d) {
    case DType::F32:
        return "f32";
    case DType::F16:
        return "f16";
    case DType::BF16:
        return "bf16";
    case DType::I32:
        return "i32";
    case DType::I64:
        return "i64";
    case DType::U8:
        return "u8";
    case DType::Bool:
        return "bool";
    case DType::Q4_0:
        return "q4_0";
    case DType::Q8_0:
        return "q8_0";
    case DType::Q4_K:
        return "q4_k";
    case DType::Q4_K_R4:
        return "q4_k_r4";
    case DType::Q5_K:
        return "q5_k";
    case DType::Q6_K:
        return "q6_k";
    case DType::Q6_K_R4:
        return "q6_k_r4";
    }
    return "?";
}
std::string to_string(DType d) {
    return std::string(name(d));
}
std::ostream& operator<<(std::ostream& os, DType d) {
    return os << name(d);
}

Result<DType> dtype_from_str(std::string_view sv) {
    std::string s(sv);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
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
    return tl::unexpected(
        Error::type_mismatch("a valid dtype", s)); // Rust reports the lower-cased input
}

Result<DType> dtype_from_onnx(int32_t code) {
    switch (code) {
    case 1:
        return DType::F32;
    case 2:
        return DType::U8;
    case 5:
        return DType::I32;
    case 7:
        return DType::I64;
    case 9:
        return DType::Bool;
    case 10:
        return DType::F16;
    case 16:
        return DType::BF16;
    default:
        return tl::unexpected(
            Error::type_mismatch("a supported ONNX dtype", "ONNX code " + std::to_string(code)));
    }
}

int32_t dtype_to_onnx(DType d) {
    switch (d) {
    case DType::F32:
        return 1;
    case DType::U8:
        return 2;
    case DType::I32:
        return 5;
    case DType::I64:
        return 7;
    case DType::Bool:
        return 9;
    case DType::F16:
        return 10;
    case DType::BF16:
        return 16;
    default:
        return 0; // all quantized dtypes
    }
}

} // namespace sapient::core
