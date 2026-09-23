// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-io/src/lib.rs: the crate-root re-exports and the two convenience
// loaders. `load_graph` and `OnnxLoader` are sub-project 8.

#include <filesystem>

#include "sapient/core/error.hpp"
#include "sapient/io/gguf.hpp"
#include "sapient/io/safetensors.hpp"

namespace sapient::io {

using gguf::GgufLoader;
using gguf::GgufValue;
using safetensors::SafetensorsLoader;
using TensorMap = gguf::TensorMap;

/// `GgufLoader::load_tensors(path)` (the heap path).
sapient::core::Result<TensorMap> load_gguf(const std::filesystem::path& path);
/// `SafetensorsLoader::load_tensors(path)`.
sapient::core::Result<TensorMap> load_safetensors(const std::filesystem::path& path);

} // namespace sapient::io
