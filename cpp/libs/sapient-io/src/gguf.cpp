// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/io/gguf.hpp"

#include <bit>
#include <cstring>

#include "sapient/core/panic.hpp"
#include "sapient/io/rust_std.hpp"

static_assert(std::endian::native == std::endian::little,
              "sapient::io assumes a little-endian host (as the Rust crate does)");

namespace sapient::io::gguf {

// ── GgufValue accessors (gguf.rs:203-246) ────────────────────────────────────────────────────
std::optional<uint32_t> GgufValue::as_u32() const {
    if (const auto* v = get_if<uint32_t>()) return *v;
    if (const auto* v = get_if<uint64_t>()) return static_cast<uint32_t>(*v); // `as u32`
    if (const auto* v = get_if<int32_t>(); v != nullptr && *v >= 0)
        return static_cast<uint32_t>(*v);
    return std::nullopt;
}
std::optional<uint64_t> GgufValue::as_u64() const {
    if (const auto* v = get_if<uint64_t>()) return *v;
    if (const auto* v = get_if<uint32_t>()) return static_cast<uint64_t>(*v);
    return std::nullopt;
}
std::optional<float> GgufValue::as_f32() const {
    if (const auto* v = get_if<float>()) return *v;
    if (const auto* v = get_if<double>()) return static_cast<float>(*v);
    return std::nullopt;
}
std::optional<double> GgufValue::as_f64() const {
    if (const auto* v = get_if<double>()) return *v;
    if (const auto* v = get_if<float>()) return static_cast<double>(*v);
    return std::nullopt;
}
std::optional<bool> GgufValue::as_bool() const {
    if (const auto* v = get_if<bool>()) return *v;
    if (const auto* v = get_if<uint8_t>()) return *v != 0;
    return std::nullopt;
}
std::optional<std::string_view> GgufValue::as_str() const {
    if (const auto* v = get_if<std::string>()) return std::string_view(*v);
    return std::nullopt;
}

namespace detail {

// ── GgmlType (gguf.rs:110-178) ───────────────────────────────────────────────────────────────
std::optional<GgmlType> ggml_type_from_u32(uint32_t v) {
    switch (v) {
    case 0:
    case 1:
    case 2:
    case 3:
    case 6:
    case 7:
    case 8:
    case 9:
    case 10:
    case 11:
    case 12:
    case 13:
    case 14:
    case 30:
        return static_cast<GgmlType>(v);
    default:
        return std::nullopt;
    }
}

size_t block_size(GgmlType t) {
    switch (t) {
    case GgmlType::F32:
    case GgmlType::F16:
    case GgmlType::BF16:
        return 1;
    case GgmlType::Q4_0:
    case GgmlType::Q4_1:
    case GgmlType::Q5_0:
    case GgmlType::Q5_1:
    case GgmlType::Q8_0:
    case GgmlType::Q8_1:
        return 32;
    case GgmlType::Q2_K:
    case GgmlType::Q3_K:
    case GgmlType::Q4_K:
    case GgmlType::Q5_K:
    case GgmlType::Q6_K:
        return QK_K;
    }
    core::panic("block_size: invalid GgmlType");
}

size_t type_size(GgmlType t) {
    switch (t) {
    case GgmlType::F32:
        return 4;
    case GgmlType::F16:
    case GgmlType::BF16:
        return 2;
    case GgmlType::Q4_0:
        return 18;
    case GgmlType::Q4_1:
        return 20;
    case GgmlType::Q5_0:
        return 22;
    case GgmlType::Q5_1:
        return 24;
    case GgmlType::Q8_0:
        return 34;
    case GgmlType::Q8_1:
        return 36;
    case GgmlType::Q2_K:
        return 84;
    case GgmlType::Q3_K:
        return 110;
    case GgmlType::Q4_K:
        return 144;
    case GgmlType::Q5_K:
        return 176;
    case GgmlType::Q6_K:
        return 210;
    }
    core::panic("type_size: invalid GgmlType");
}

std::optional<core::DType> to_sapient_dtype(GgmlType t) {
    switch (t) {
    case GgmlType::Q4_0:
        return core::DType::Q4_0;
    case GgmlType::Q8_0:
        return core::DType::Q8_0;
    case GgmlType::Q4_K:
        return core::DType::Q4_K;
    case GgmlType::Q5_K:
        return core::DType::Q5_K;
    case GgmlType::Q6_K:
        return core::DType::Q6_K;
    default:
        return std::nullopt;
    }
}

std::string_view debug_name(GgmlType t) {
    switch (t) {
    case GgmlType::F32:
        return "F32";
    case GgmlType::F16:
        return "F16";
    case GgmlType::Q4_0:
        return "Q4_0";
    case GgmlType::Q4_1:
        return "Q4_1";
    case GgmlType::Q5_0:
        return "Q5_0";
    case GgmlType::Q5_1:
        return "Q5_1";
    case GgmlType::Q8_0:
        return "Q8_0";
    case GgmlType::Q8_1:
        return "Q8_1";
    case GgmlType::Q2_K:
        return "Q2_K";
    case GgmlType::Q3_K:
        return "Q3_K";
    case GgmlType::Q4_K:
        return "Q4_K";
    case GgmlType::Q5_K:
        return "Q5_K";
    case GgmlType::Q6_K:
        return "Q6_K";
    case GgmlType::BF16:
        return "BF16";
    }
    core::panic("debug_name: invalid GgmlType");
}

size_t tensor_byte_len(GgmlType kind, size_t numel) {
    if (kind == GgmlType::F32 || kind == GgmlType::F16 || kind == GgmlType::BF16)
        return numel * type_size(kind);
    return (numel / block_size(kind)) * type_size(kind);
}

// ── Low-level readers (gguf.rs:780-909) over a bounds-checked cursor ─────────────────────────
namespace {

class Cursor {
public:
    explicit Cursor(std::span<const uint8_t> b) : b_(b) {}
    size_t position() const { return pos_; }
    /// `read_exact`: all `n` bytes or Rust's EOF error. Checked BEFORE any caller allocates.
    core::Result<std::span<const uint8_t>> take(size_t n) {
        if (n > b_.size() - pos_)
            return tl::unexpected(core::Error::gguf_parse(std::string(rust_std::READ_EXACT_EOF)));
        const auto s = b_.subspan(pos_, n);
        pos_ += n;
        return s;
    }

private:
    std::span<const uint8_t> b_;
    size_t pos_{0};
};

template <class T> core::Result<T> read_le(Cursor& c) {
    SAPIENT_TRY_ASSIGN(const auto s, c.take(sizeof(T)));
    T v;
    std::memcpy(&v, s.data(), sizeof(T));
    return v;
}

core::Result<std::string> read_gguf_string(Cursor& c) {
    SAPIENT_TRY_ASSIGN(const uint64_t len, read_le<uint64_t>(c));
    // Hardening (plan B Global Constraints): bounds-check before allocating. Rust allocates
    // `vec![0; len]` first and aborts on an absurd len; for every len it survives, the text is
    // this same EOF error.
    SAPIENT_TRY_ASSIGN(const auto bytes, c.take(static_cast<size_t>(len)));
    if (auto e = rust_std::utf8_error(bytes)) return tl::unexpected(core::Error::gguf_parse(*e));
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

/// skip_value's byte width per item type; 0 = consumes nothing (Rust's `_ => {}` arm, incl. 9).
/// Strings (8) are variable-width and handled separately.
size_t skip_width(uint32_t vtype) {
    switch (vtype) {
    case 0:
    case 1:
    case 7:
        return 1;
    case 2:
    case 3:
        return 2;
    case 4:
    case 5:
    case 6:
        return 4;
    case 10:
    case 11:
    case 12:
        return 8;
    default:
        return 0;
    }
}

core::Result<void> skip_value(Cursor& c, uint32_t vtype) {
    if (vtype == 8) {
        SAPIENT_TRY(read_gguf_string(c));
        return {};
    }
    SAPIENT_TRY(c.take(skip_width(vtype)));
    return {};
}

core::Result<GgufValue> read_array(Cursor& c) {
    SAPIENT_TRY_ASSIGN(const uint32_t item_type, read_le<uint32_t>(c));
    SAPIENT_TRY_ASSIGN(const uint64_t count64, read_le<uint64_t>(c));
    const auto count = static_cast<size_t>(count64);
    switch (item_type) {
    case 4: {
        std::vector<uint32_t> v; // no reserve(count): untrusted
        for (size_t i = 0; i < count; ++i) {
            SAPIENT_TRY_ASSIGN(const uint32_t x, read_le<uint32_t>(c));
            v.push_back(x);
        }
        return GgufValue(std::move(v));
    }
    case 8: {
        std::vector<std::string> v;
        for (size_t i = 0; i < count; ++i) {
            SAPIENT_TRY_ASSIGN(std::string s, read_gguf_string(c));
            v.push_back(std::move(s));
        }
        return GgufValue(std::move(v));
    }
    case 6: {
        std::vector<float> v;
        for (size_t i = 0; i < count; ++i) {
            SAPIENT_TRY_ASSIGN(const float x, read_le<float>(c));
            v.push_back(x);
        }
        return GgufValue(std::move(v));
    }
    default:
        // Rust loops `count` times over skip_value; when the item type consumes nothing that
        // loop is a no-op, so skip it (a 2^60 count would otherwise hang — same bytes consumed).
        if (skip_width(item_type) == 0) return GgufValue(GgufOther{}); // 8 never reaches here
        for (size_t i = 0; i < count; ++i)
            SAPIENT_TRY(skip_value(c, item_type));
        return GgufValue(GgufOther{});
    }
}

core::Result<GgufValue> read_value(Cursor& c, uint32_t vtype) {
    switch (vtype) {
    case 0: {
        SAPIENT_TRY_ASSIGN(const uint8_t v, read_le<uint8_t>(c));
        return GgufValue(v);
    }
    case 1: {
        SAPIENT_TRY_ASSIGN(const uint8_t v, read_le<uint8_t>(c));
        return GgufValue(static_cast<int8_t>(v));
    }
    case 2: {
        SAPIENT_TRY_ASSIGN(const uint16_t v, read_le<uint16_t>(c));
        return GgufValue(v);
    }
    case 3: {
        SAPIENT_TRY_ASSIGN(const uint16_t v, read_le<uint16_t>(c));
        return GgufValue(static_cast<int16_t>(v));
    }
    case 4: {
        SAPIENT_TRY_ASSIGN(const uint32_t v, read_le<uint32_t>(c));
        return GgufValue(v);
    }
    case 5: {
        SAPIENT_TRY_ASSIGN(const int32_t v, read_le<int32_t>(c));
        return GgufValue(v);
    }
    case 6: {
        SAPIENT_TRY_ASSIGN(const float v, read_le<float>(c));
        return GgufValue(v);
    }
    case 7: {
        SAPIENT_TRY_ASSIGN(const uint8_t v, read_le<uint8_t>(c));
        return GgufValue(v != 0);
    }
    case 8: {
        SAPIENT_TRY_ASSIGN(std::string v, read_gguf_string(c));
        return GgufValue(std::move(v));
    }
    case 9:
        return read_array(c);
    case 10: {
        SAPIENT_TRY_ASSIGN(const uint64_t v, read_le<uint64_t>(c));
        return GgufValue(v);
    }
    case 11: {
        SAPIENT_TRY_ASSIGN(const int64_t v, read_le<int64_t>(c));
        return GgufValue(v);
    }
    case 12: {
        SAPIENT_TRY_ASSIGN(const double v, read_le<double>(c));
        return GgufValue(v);
    }
    default:
        return GgufValue(GgufOther{}); // consumes NOTHING (faithful; see Global Constraints)
    }
}

} // namespace

// ── parse_header (gguf.rs:517-578) ───────────────────────────────────────────────────────────
core::Result<ParsedHeader> parse_header(std::span<const uint8_t> bytes) {
    Cursor c(bytes);
    SAPIENT_TRY_ASSIGN(const uint32_t magic, read_le<uint32_t>(c));
    if (magic != GGUF_MAGIC) return tl::unexpected(core::Error::gguf_parse("bad GGUF magic"));
    SAPIENT_TRY_ASSIGN(const uint32_t version, read_le<uint32_t>(c));
    if (version < 1 || version > 3)
        return tl::unexpected(core::Error::gguf_parse("unsupported GGUF version " +
                                                      std::to_string(version) +
                                                      " (expected 1\xE2\x80\x93"
                                                      "3)"));
    SAPIENT_TRY_ASSIGN(const uint64_t tensor_count, read_le<uint64_t>(c));
    SAPIENT_TRY_ASSIGN(const uint64_t kv_count, read_le<uint64_t>(c));

    ParsedHeader h; // NO reserve(kv_count) / reserve(tensor_count): untrusted header fields
    for (uint64_t i = 0; i < kv_count; ++i) {
        SAPIENT_TRY_ASSIGN(std::string key, read_gguf_string(c));
        SAPIENT_TRY_ASSIGN(const uint32_t vtype, read_le<uint32_t>(c));
        SAPIENT_TRY_ASSIGN(GgufValue value, read_value(c, vtype));
        h.metadata.insert_or_assign(std::move(key), std::move(value)); // HashMap::insert: last wins
    }
    for (uint64_t i = 0; i < tensor_count; ++i) {
        SAPIENT_TRY_ASSIGN(std::string name, read_gguf_string(c));
        SAPIENT_TRY_ASSIGN(const uint32_t n_dims, read_le<uint32_t>(c));
        std::vector<size_t> dims; // no reserve(n_dims)
        for (uint32_t d = 0; d < n_dims; ++d) {
            SAPIENT_TRY_ASSIGN(const uint64_t dim, read_le<uint64_t>(c));
            dims.push_back(static_cast<size_t>(dim));
        }
        SAPIENT_TRY_ASSIGN(const uint32_t kind_raw, read_le<uint32_t>(c));
        const auto kind = ggml_type_from_u32(kind_raw);
        if (!kind)
            return tl::unexpected(
                core::Error::gguf_parse("unknown ggml type " + std::to_string(kind_raw)));
        SAPIENT_TRY_ASSIGN(const uint64_t offset, read_le<uint64_t>(c));
        h.tensor_infos.push_back(GgufTensorInfo{std::move(name), std::move(dims), *kind, offset});
    }

    uint64_t alignment = DEFAULT_ALIGNMENT;
    if (const auto it = h.metadata.find("general.alignment"); it != h.metadata.end())
        if (const auto a = it->second.as_u64()) alignment = *a;
    if (alignment == 0) core::panic("attempt to divide by zero"); // Rust u64::div_ceil(0)
    const uint64_t raw_pos = c.position();
    const uint64_t q = raw_pos / alignment + (raw_pos % alignment != 0 ? 1 : 0); // div_ceil
    h.data_start = static_cast<size_t>(q) * static_cast<size_t>(alignment);      // wraps like Rust
    return h;
}

} // namespace detail
} // namespace sapient::io::gguf
