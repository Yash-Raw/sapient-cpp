// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/io/safetensors.hpp"

#include <array>
#include <bit>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "sapient/core/buffer.hpp"
#include "sapient/core/dtype.hpp"
#include "sapient/core/panic.hpp"
#include "sapient/core/shape.hpp"
#include "sapient/core/tensor.hpp"
#include "sapient/io/mmap.hpp"
#include "sapient/io/rust_std.hpp"

static_assert(std::endian::native == std::endian::little,
              "sapient::io assumes a little-endian host (as the Rust crate does)");

namespace sapient::io::safetensors {

namespace core = sapient::core;

namespace {

core::Error st_err(std::string msg) {
    return core::Error::safetensors_parse(std::move(msg));
}

// Rust `#[derive(Deserialize)] struct StMeta` (safetensors.rs:21-26).
struct StMeta {
    std::string dtype;
    std::vector<size_t> shape;
    std::array<size_t, 2> data_offsets{};
};

/// serde's strictness: a JSON unsigned integer only (no float coercion, no negatives).
std::optional<std::string> as_usize(const nlohmann::json& j, size_t& out) {
    if (!j.is_number_unsigned())
        return std::string("invalid type: ") + j.type_name() + ", expected usize";
    out = j.get<size_t>();
    return std::nullopt;
}

/// serde_json::from_value::<StMeta>. Unknown fields are ignored (no deny_unknown_fields). The
/// messages are C++-authored: only the caller's "tensor '{name}': " prefix is parity-bound.
std::optional<std::string> decode_meta(const nlohmann::json& v, StMeta& m) {
    if (!v.is_object())
        return std::string("invalid type: ") + v.type_name() + ", expected struct StMeta";
    const auto dt = v.find("dtype");
    if (dt == v.end()) return std::string("missing field `dtype`");
    if (!dt->is_string())
        return std::string("invalid type: ") + dt->type_name() + ", expected a string";
    m.dtype = dt->get<std::string>();
    const auto sh = v.find("shape");
    if (sh == v.end()) return std::string("missing field `shape`");
    if (!sh->is_array())
        return std::string("invalid type: ") + sh->type_name() + ", expected a sequence";
    for (const auto& e : *sh) {
        size_t d = 0;
        if (auto err = as_usize(e, d)) return err;
        m.shape.push_back(d);
    }
    const auto off = v.find("data_offsets");
    if (off == v.end()) return std::string("missing field `data_offsets`");
    if (!off->is_array())
        return std::string("invalid type: ") + off->type_name() + ", expected an array";
    if (off->size() != 2)
        return "invalid length " + std::to_string(off->size()) + ", expected an array of length 2";
    for (size_t i = 0; i < 2; ++i)
        if (auto err = as_usize((*off)[i], m.data_offsets[i])) return err;
    return std::nullopt;
}

/// Rust: `raw.chunks_exact(4)` → Vec<f32> → `Tensor::from_f32` (validate, then count check, then an
/// align-64 copy). One copy here with the same observables — incl. the dropped partial chunk.
core::Result<core::Tensor> f32_tensor(std::span<const uint8_t> raw, core::Shape shape) {
    SAPIENT_TRY(shape.validate());
    const size_t count = raw.size() / 4;
    if (count != shape.numel())
        return tl::unexpected(core::Error::shape_mismatch(shape.dims, {count}));
    SAPIENT_TRY_ASSIGN(auto buf, core::CpuBuffer::with_capacity(count * 4, 64));
    std::memcpy(buf->data(), raw.data(), count * 4);
    return core::Tensor::from_buffer(std::move(shape), core::DType::F32, std::move(buf), 0);
}

core::Result<core::Tensor> wrap(core::Result<core::Tensor> r) {
    if (!r) return tl::unexpected(st_err(r.error().to_string()));
    return r;
}

} // namespace

core::Result<TensorMap> SafetensorsLoader::load(const std::filesystem::path& path) {
    auto m = MappedFile::open(path);
    if (!m) {
        if (m.error().stage == MapStage::Open)
            return tl::unexpected(
                core::Error::model_not_found(display_path(path) + ": " + m.error().os.message));
        return tl::unexpected(st_err(m.error().os.message)); // no "mmap failed" prefix here
    }
    return from_bytes((*m)->bytes()); // every tensor is copied; the mapping dies with `m`
}

core::Result<TensorMap> SafetensorsLoader::from_bytes(std::span<const uint8_t> bytes) {
    if (bytes.size() < 8) return tl::unexpected(st_err("file too short"));
    uint64_t header_len64 = 0;
    std::memcpy(&header_len64, bytes.data(), 8);
    const auto header_len = static_cast<size_t>(header_len64);
    const size_t header_end = 8 + header_len; // wraps like Rust release
    if (header_end > bytes.size()) return tl::unexpected(st_err("header overflows file"));
    if (header_end < 8) // wrapped: Rust's check passed and `bytes[8..header_end]` panics
        core::panic("slice index starts at 8 but ends at " + std::to_string(header_end));
    const auto header = bytes.subspan(8, header_len);
    if (auto e = rust_std::utf8_error(header)) return tl::unexpected(st_err(*e));

    nlohmann::json root;
    try { // the only exception in sapient::io — caught here, never crosses the library boundary
        const char* first = reinterpret_cast<const char*>(header.data());
        root = nlohmann::json::parse(first, first + header.size());
    } catch (const nlohmann::json::exception& e) {
        return tl::unexpected(st_err(e.what()));
    }
    if (!root.is_object())
        return tl::unexpected(
            st_err(std::string("invalid type: ") + root.type_name() + ", expected a map"));

    const auto data = bytes.subspan(header_end);
    TensorMap tensors;
    for (const auto& [name, value] : root.items()) {
        if (name == "__metadata__") continue;
        StMeta meta;
        if (auto e = decode_meta(value, meta))
            return tl::unexpected(st_err("tensor '" + name + "': " + *e));
        core::DType dtype = core::DType::F32;
        if (meta.dtype == "F32")
            dtype = core::DType::F32;
        else if (meta.dtype == "F16")
            dtype = core::DType::F16;
        else if (meta.dtype == "BF16")
            dtype = core::DType::BF16;
        else if (meta.dtype == "I32")
            dtype = core::DType::I32;
        else if (meta.dtype == "I64")
            dtype = core::DType::I64;
        else if (meta.dtype == "U8")
            dtype = core::DType::U8;
        else if (meta.dtype == "BOOL")
            dtype = core::DType::Bool;
        else
            return tl::unexpected(st_err("unknown dtype '" + meta.dtype + "'"));

        const auto [start, end] = meta.data_offsets;
        if (end > data.size())
            return tl::unexpected(st_err("tensor '" + name + "' data out of bounds"));
        if (start > end) // Rust `&data_section[start..end]` panics
            core::panic("slice index starts at " + std::to_string(start) + " but ends at " +
                        std::to_string(end));
        const auto raw = data.subspan(start, end - start);
        core::Shape shape(meta.shape);

        core::Result<core::Tensor> t = tl::unexpected(core::Error::internal("unreachable"));
        switch (dtype) {
        case core::DType::F32:
            t = wrap(f32_tensor(raw, std::move(shape)));
            break;
        case core::DType::BF16:
            t = wrap(core::Tensor::from_bf16_bytes(raw, std::move(shape)));
            break;
        case core::DType::F16:
            t = wrap(core::Tensor::from_f16_bytes(raw, std::move(shape)));
            break;
        default:
            return tl::unexpected(st_err("unsupported safetensors dtype '" +
                                         core::to_string(dtype) + "' for tensor '" + name + "'"));
        }
        if (!t) return tl::unexpected(t.error());
        tensors.insert_or_assign(name, std::move(*t)); // last wins
    }
    return tensors;
}

core::Result<TensorMap> SafetensorsLoader::load_tensors(const std::filesystem::path& path) {
    return load(path);
}

} // namespace sapient::io::safetensors
