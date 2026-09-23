// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-io/src/safetensors.rs minus `load_as_graph` (sub-project 8). Despite the
// Rust module doc, nothing is zero-copy: load() maps the file, copies every tensor out, and drops
// the mapping. The reachable dtype set is {F32, F16, BF16}; F16/BF16 keep their raw bytes.

#include <cstdint>
#include <filesystem>
#include <span>

#include "sapient/core/error.hpp"
#include "sapient/io/gguf.hpp"

namespace sapient::io::safetensors {

using TensorMap = gguf::TensorMap;

class SafetensorsLoader {
public:
    static sapient::core::Result<TensorMap> load(const std::filesystem::path& path);
    static sapient::core::Result<TensorMap> from_bytes(std::span<const uint8_t> bytes);
    /// Alias for `load`.
    static sapient::core::Result<TensorMap> load_tensors(const std::filesystem::path& path);
};

} // namespace sapient::io::safetensors
