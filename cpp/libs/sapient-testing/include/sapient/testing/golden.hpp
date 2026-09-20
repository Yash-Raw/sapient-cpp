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
