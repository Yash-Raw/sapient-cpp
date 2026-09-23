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

// Per-field decoders shared by both StMeta forms below (serde's derive(Deserialize) accepts a
// struct either as a JSON object with named fields, via visit_map, OR as a JSON array of exactly
// its fields in declaration order, via visit_seq — verified against real serde_json 1.0.150:
// `serde_json::from_value::<StMeta>(json!(["F32",[1],[0,4]]))` is `Ok`). The messages are
// C++-authored: only the caller's "tensor '{name}': " prefix is parity-bound.

std::optional<std::string> decode_dtype_value(const nlohmann::json& dt, std::string& out) {
    if (!dt.is_string())
        return std::string("invalid type: ") + dt.type_name() + ", expected a string";
    out = dt.get<std::string>();
    return std::nullopt;
}

std::optional<std::string> decode_shape_value(const nlohmann::json& sh, std::vector<size_t>& out) {
    if (!sh.is_array())
        return std::string("invalid type: ") + sh.type_name() + ", expected a sequence";
    for (const auto& e : sh) {
        size_t d = 0;
        if (auto err = as_usize(e, d)) return err;
        out.push_back(d);
    }
    return std::nullopt;
}

std::optional<std::string> decode_data_offsets_value(const nlohmann::json& off,
                                                     std::array<size_t, 2>& out) {
    if (!off.is_array())
        return std::string("invalid type: ") + off.type_name() + ", expected an array";
    if (off.size() != 2)
        return "invalid length " + std::to_string(off.size()) + ", expected an array of length 2";
    for (size_t i = 0; i < 2; ++i)
        if (auto err = as_usize(off[i], out[i])) return err;
    return std::nullopt;
}

/// serde_json::from_value::<StMeta>. Unknown fields on the object form are ignored (no
/// deny_unknown_fields). The sequence form decodes EXACTLY 3 elements positionally
/// (dtype, shape, data_offsets); any other length is Err.
std::optional<std::string> decode_meta(const nlohmann::json& v, StMeta& m) {
    if (v.is_array()) {
        if (v.size() != 3)
            return "invalid length " + std::to_string(v.size()) +
                   ", expected struct StMeta with 3 elements";
        if (auto err = decode_dtype_value(v[0], m.dtype)) return err;
        if (auto err = decode_shape_value(v[1], m.shape)) return err;
        if (auto err = decode_data_offsets_value(v[2], m.data_offsets)) return err;
        return std::nullopt;
    }
    if (!v.is_object())
        return std::string("invalid type: ") + v.type_name() + ", expected struct StMeta";
    const auto dt = v.find("dtype");
    if (dt == v.end()) return std::string("missing field `dtype`");
    if (auto err = decode_dtype_value(*dt, m.dtype)) return err;
    const auto sh = v.find("shape");
    if (sh == v.end()) return std::string("missing field `shape`");
    if (auto err = decode_shape_value(*sh, m.shape)) return err;
    const auto off = v.find("data_offsets");
    if (off == v.end()) return std::string("missing field `data_offsets`");
    if (auto err = decode_data_offsets_value(*off, m.data_offsets)) return err;
    return std::nullopt;
}

/// serde_json rejects a leading UTF-8 BOM outright ("expected value at line 1 column 1");
/// nlohmann's lexer silently skips it. Verified against real serde_json 1.0.150. Reject it
/// ourselves before handing the header to nlohmann.
bool starts_with_bom(std::span<const uint8_t> header) {
    return header.size() >= 3 && header[0] == 0xEF && header[1] == 0xBB && header[2] == 0xBF;
}

/// serde_json's lexer rejects a raw NUL byte anywhere in the header text — U+0000 is illegal
/// wherever it appears in JSON, so serde always returns an `Err` ("trailing characters at line 1
/// column N", the exact wording depending on position). nlohmann 3.11.3's lexer instead treats a
/// raw `\0` as end-of-input: `{}\0garbage` parses `Ok` (0 tensors) and a valid header padded with
/// trailing `\0` bytes also parses `Ok`. Verified against real serde_json 1.0.150. Reject any
/// embedded NUL ourselves before handing the header to nlohmann; the message text itself is not
/// parity-bound (it is the tail of an embedded serde_json error — see the "Exempt" rule).
bool contains_nul(std::span<const uint8_t> header) {
    for (const uint8_t c : header)
        if (c == 0) return true;
    return false;
}

/// serde_json's default (non "unbounded_depth") build enforces a recursion limit: the deepest
/// nesting of `{`/`[` compounds, counting the outermost container as depth 1, must stay <= 127 —
/// depth 128 is `Err("recursion limit exceeded …")`. Verified empirically against real
/// serde_json 1.0.150 (pure-array probe: 127 deep is Ok, 128 is Err). nlohmann has no such limit,
/// so this scans the header bytes (already UTF-8-validated) tracking depth outside string
/// literals, with backslash-escape handling — a full JSON syntax check is not needed, only
/// bracket/brace depth.
inline constexpr size_t MAX_JSON_DEPTH = 127;

bool exceeds_recursion_limit(std::span<const uint8_t> header) {
    size_t depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (const uint8_t c : header) {
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
        } else if (c == '{' || c == '[') {
            if (++depth > MAX_JSON_DEPTH) return true;
        } else if ((c == '}' || c == ']') && depth > 0) {
            --depth;
        }
    }
    return false;
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
    if (contains_nul(header)) return tl::unexpected(st_err("trailing characters"));
    if (starts_with_bom(header)) return tl::unexpected(st_err("expected value at line 1 column 1"));
    if (exceeds_recursion_limit(header)) return tl::unexpected(st_err("recursion limit exceeded"));

    nlohmann::json root;
    // The only exception in sapient::io — caught here, never crosses the library boundary. An
    // uncaught std::bad_alloc (nlohmann::json::exception's base is std::exception, not
    // std::bad_alloc) would instead terminate the process here — matching Rust, which aborts on
    // allocation failure rather than returning a Result.
    try {
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

        // Stores on success, else returns the error — directly from each case, no placeholder.
        auto store = [&](core::Result<core::Tensor> t) -> std::optional<core::Error> {
            if (!t) return t.error();
            tensors.insert_or_assign(name, std::move(*t)); // last wins
            return std::nullopt;
        };
        switch (dtype) {
        case core::DType::F32:
            if (auto e = store(wrap(f32_tensor(raw, std::move(shape))))) return tl::unexpected(*e);
            break;
        case core::DType::BF16:
            if (auto e = store(wrap(core::Tensor::from_bf16_bytes(raw, std::move(shape)))))
                return tl::unexpected(*e);
            break;
        case core::DType::F16:
            if (auto e = store(wrap(core::Tensor::from_f16_bytes(raw, std::move(shape)))))
                return tl::unexpected(*e);
            break;
        default:
            return tl::unexpected(st_err("unsupported safetensors dtype '" +
                                         core::to_string(dtype) + "' for tensor '" + name + "'"));
        }
    }
    return tensors;
}

core::Result<TensorMap> SafetensorsLoader::load_tensors(const std::filesystem::path& path) {
    return load(path);
}

} // namespace sapient::io::safetensors
