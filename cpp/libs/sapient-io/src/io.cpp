// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/io/io.hpp"

namespace sapient::io {

sapient::core::Result<TensorMap> load_gguf(const std::filesystem::path& path) {
    return GgufLoader::load_tensors(path);
}

sapient::core::Result<TensorMap> load_safetensors(const std::filesystem::path& path) {
    return SafetensorsLoader::load_tensors(path);
}

} // namespace sapient::io
