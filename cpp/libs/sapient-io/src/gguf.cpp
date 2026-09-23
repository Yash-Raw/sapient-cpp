// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/io/gguf.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>

#include "sapient/core/dequant.hpp"
#include "sapient/core/f16.hpp"
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

// ── Dequantisation (gguf.rs:259-476) ─────────────────────────────────────────────────────────
namespace {

void require_bytes(std::span<const uint8_t> data, size_t need) {
    // Rust indexes `data[base..]` unchecked-by-us and panics past the end; never read OOB in C++.
    if (data.size() < need) core::panic("index out of bounds");
}

/// io's Q4_0/Q8_0 semantics: bytes/block_bytes blocks, writes past numel skipped.
template <size_t BlockBytes, void (*Block)(const uint8_t*, float*)>
std::vector<float> guarded_blocks(std::span<const uint8_t> data, size_t numel) {
    std::vector<float> out(numel, 0.0f);
    std::array<float, 32> tmp{};
    const size_t nblocks = data.size() / BlockBytes;
    for (size_t b = 0; b < nblocks; ++b) {
        Block(data.data() + b * BlockBytes, tmp.data());
        for (size_t j = 0; j < 32; ++j)
            if (b * 32 + j < numel) out[b * 32 + j] = tmp[j];
    }
    return out;
}

/// io's K-quant semantics: numel/256 blocks written straight into the output.
template <size_t BlockBytes, void (*Block)(const uint8_t*, float*)>
std::vector<float> k_blocks(std::span<const uint8_t> data, size_t numel) {
    const size_t nblocks = numel / QK_K;
    require_bytes(data, nblocks * BlockBytes);
    std::vector<float> out(numel, 0.0f);
    for (size_t b = 0; b < nblocks; ++b)
        Block(data.data() + b * BlockBytes, out.data() + b * QK_K);
    return out;
}

} // namespace

std::vector<float> dequantize_q4_0(std::span<const uint8_t> data, size_t numel) {
    return guarded_blocks<18, core::dequant::q4_0_block>(data, numel);
}
std::vector<float> dequantize_q8_0(std::span<const uint8_t> data, size_t numel) {
    return guarded_blocks<34, core::dequant::q8_0_block>(data, numel);
}

std::vector<float> dequantize_q5_0(std::span<const uint8_t> data, size_t numel) {
    const size_t nblocks = numel / 32;
    require_bytes(data, nblocks * 22);
    std::vector<float> out(numel, 0.0f);
    for (size_t b = 0; b < nblocks; ++b) {
        const uint8_t* base = data.data() + b * 22;
        const float scale = core::f16_le_to_f32(base);
        uint32_t qh = 0;
        std::memcpy(&qh, base + 2, 4);
        for (uint32_t j = 0; j < 16; ++j) {
            const uint8_t byte = base[6 + j];
            const uint32_t xh_0 = ((qh >> j) << 4) & 0x10u;
            const uint32_t xh_1 = (qh >> (j + 12)) & 0x10u;
            const int32_t x0 =
                static_cast<int32_t>(static_cast<uint32_t>(byte & 0x0Fu) | xh_0) - 16;
            const int32_t x1 = static_cast<int32_t>(static_cast<uint32_t>(byte >> 4) | xh_1) - 16;
            out[b * 32 + j] = static_cast<float>(x0) * scale;
            out[b * 32 + j + 16] = static_cast<float>(x1) * scale;
        }
    }
    return out;
}

std::vector<float> dequantize_q4_k(std::span<const uint8_t> data, size_t numel) {
    return k_blocks<144, core::dequant::q4_k_block>(data, numel);
}
std::vector<float> dequantize_q5_k(std::span<const uint8_t> data, size_t numel) {
    return k_blocks<176, core::dequant::q5_k_block>(data, numel);
}
std::vector<float> dequantize_q6_k(std::span<const uint8_t> data, size_t numel) {
    return k_blocks<210, core::dequant::q6_k_block>(data, numel);
}

std::vector<uint8_t> quantize_to_q8_0(std::span<const float> data) {
    std::vector<uint8_t> out;
    out.reserve(data.size() / 32 * 34);                  // sized from real data, not a header field
    for (size_t b = 0; b + 32 <= data.size(); b += 32) { // chunks_exact(32)
        float amax = 0.0f;
        for (size_t i = 0; i < 32; ++i)
            amax = std::fmax(amax, std::fabs(data[b + i])); // f32::max drops NaN
        const float d = amax / 127.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        uint8_t h[2];
        core::f16_to_le(core::f32_to_f16_bits(d), h); // RNE
        out.push_back(h[0]);
        out.push_back(h[1]);
        for (size_t i = 0; i < 32; ++i) {
            // `(v * id).round().clamp(-127.0, 127.0) as i8 as u8`: roundf, clamp, NaN → 0.
            const float q = std::clamp(std::roundf(data[b + i] * id), -127.0f, 127.0f);
            const int8_t v = std::isnan(q) ? int8_t{0} : static_cast<int8_t>(q);
            out.push_back(static_cast<uint8_t>(v));
        }
    }
    return out;
}

core::Result<std::vector<float>>
dequantize_to_f32(GgmlType kind, std::span<const uint8_t> bytes, size_t numel) {
    switch (kind) {
    case GgmlType::F32: {
        // `&bytes[..numel * 4]` (gguf.rs:478-492): the multiply WRAPS like Rust release, so a
        // huge `numel` can wrap `n` down to <= bytes.size() and decode fewer than `numel` floats
        // — the caller's `Tensor::from_f32` then reports ShapeMismatch, not a panic (I1 fix).
        const size_t n = numel * 4; // wraps like Rust release
        if (n > bytes.size())
            core::panic("range end index " + std::to_string(n) +
                        " out of range for slice of length " + std::to_string(bytes.size()));
        const size_t count = n / 4;
        std::vector<float> out(count);
        if (count > 0) std::memcpy(out.data(), bytes.data(), n);
        return out;
    }
    case GgmlType::F16:
    case GgmlType::BF16: {
        const size_t n = numel * 2; // wraps like Rust release
        if (n > bytes.size())
            core::panic("range end index " + std::to_string(n) +
                        " out of range for slice of length " + std::to_string(bytes.size()));
        const size_t count = n / 2;
        std::vector<float> out(count);
        for (size_t i = 0; i < count; ++i)
            out[i] = kind == GgmlType::F16 ? core::f16_le_to_f32(bytes.data() + 2 * i)
                                           : core::bf16_le_to_f32(bytes.data() + 2 * i);
        return out;
    }
    case GgmlType::Q4_0:
        return dequantize_q4_0(bytes, numel);
    case GgmlType::Q5_0:
        return dequantize_q5_0(bytes, numel);
    case GgmlType::Q8_0:
        return dequantize_q8_0(bytes, numel);
    case GgmlType::Q4_K:
        return dequantize_q4_k(bytes, numel);
    case GgmlType::Q5_K:
        return dequantize_q5_k(bytes, numel);
    case GgmlType::Q6_K:
        return dequantize_q6_k(bytes, numel);
    default:
        return tl::unexpected(core::Error::gguf_parse("unsupported GGUF quantization type " +
                                                      std::string(debug_name(kind))));
    }
}

// ── MmapBuffer / make_tensor(_mmap) (gguf.rs:38-78, 580-676) ─────────────────────────────────
std::span<const uint8_t> MmapBuffer::bytes() const {
    // `&self.mmap[self.offset..self.offset + self.len]` (gguf.rs:55-57): bounds-checked like
    // Rust's slice indexing (I2 fix) — the KEPT branch of make_tensor_mmap defers a wrapped
    // range's panic to here instead of panicking at load time.
    const size_t end = offset_ + len_; // wraps like Rust release
    if (end < offset_)
        core::panic("slice index starts at " + std::to_string(offset_) + " but ends at " +
                    std::to_string(end));
    if (end > mmap_->size())
        core::panic("range end index " + std::to_string(end) +
                    " out of range for slice of length " + std::to_string(mmap_->size()));
    return mmap_->bytes().subspan(offset_, len_);
}

std::span<uint8_t> MmapBuffer::bytes_mut() {
    core::panic("MmapBuffer is read-only \xE2\x80\x94 model weights cannot be mutated in-place");
}

namespace {

size_t numel_of(const std::vector<size_t>& dims) {
    size_t n = 1;
    for (const size_t d : dims)
        n *= d; // wraps like Rust release
    return std::max<size_t>(n, 1);
}

core::Shape shape_of(const std::vector<size_t>& dims) {
    return dims.empty() ? core::Shape{1} : core::Shape(dims);
}

/// `e.to_string()` wrapped as GgufParseError — Rust's `.map_err(|e| GgufParseError(e.to_string()))`.
core::Result<core::Tensor> wrap(core::Result<core::Tensor> r) {
    if (!r) return tl::unexpected(core::Error::gguf_parse(r.error().to_string()));
    return r;
}

struct Range {
    size_t start;
    size_t end;
};

/// `panic_on_wrap` is true for every caller that slices the bytes immediately after this check
/// (heap make_tensor, the mmap UNKEPT branch) — there, a wrapped `end` passes Rust's file-size
/// check and then the immediate slice panics (spec §3 rule 8). It is false for the mmap KEPT
/// branch, which never slices at load time in Rust (gguf.rs:580-615): that branch's wrapped-range
/// panic is deferred to `MmapBuffer::bytes()` instead (I2 fix).
core::Result<Range> data_range(const GgufTensorInfo& info,
                               size_t data_start,
                               size_t byte_len,
                               size_t file_len,
                               bool panic_on_wrap = true) {
    const size_t start = data_start + static_cast<size_t>(info.offset); // wraps like Rust release
    const size_t end = start + byte_len;                                // wraps like Rust release
    if (end > file_len)
        return tl::unexpected(core::Error::gguf_parse(
            "tensor '" + info.name + "': data range [" + std::to_string(start) + ".." +
            std::to_string(end) + "] exceeds file size " + std::to_string(file_len)));
    if (panic_on_wrap && end < start)
        core::panic("slice index starts at " + std::to_string(start) + " but ends at " +
                    std::to_string(end));
    return Range{start, end};
}

/// The identical else-branch of make_tensor / make_tensor_mmap: a type SAPIENT does not keep.
core::Result<core::Tensor>
convert_unkept(GgmlType kind, std::span<const uint8_t> raw, size_t numel, core::Shape shape) {
    SAPIENT_TRY_ASSIGN(const std::vector<float> f32_data, dequantize_to_f32(kind, raw, numel));
    if (block_size(kind) > 1 && numel % 32 == 0) { // in practice: Q5_0 only
        const std::vector<uint8_t> q8 = quantize_to_q8_0(f32_data);
        return wrap(core::Tensor::from_quant_bytes(q8, std::move(shape), core::DType::Q8_0));
    }
    return wrap(core::Tensor::from_f32(f32_data, std::move(shape))); // align 64 — not from_f32_vec
}

} // namespace

core::Result<core::Tensor>
make_tensor(const GgufTensorInfo& info, std::span<const uint8_t> bytes, size_t data_start) {
    const size_t numel = numel_of(info.dims);
    const size_t byte_len = tensor_byte_len(info.kind, numel);
    SAPIENT_TRY_ASSIGN(const Range r, data_range(info, data_start, byte_len, bytes.size()));
    const auto raw = bytes.subspan(r.start, byte_len);
    core::Shape shape = shape_of(info.dims);
    if (const auto dtype = to_sapient_dtype(info.kind))
        return wrap(
            core::Tensor::from_quant_bytes(raw, std::move(shape), *dtype)); // copy, align 16
    return convert_unkept(info.kind, raw, numel, std::move(shape));
}

core::Result<core::Tensor> make_tensor_mmap(const GgufTensorInfo& info,
                                            const std::shared_ptr<const MappedFile>& mmap,
                                            size_t data_start) {
    const size_t numel = numel_of(info.dims);
    const size_t byte_len = tensor_byte_len(info.kind, numel);
    const auto dtype = to_sapient_dtype(info.kind);
    // The KEPT branch (dtype has a value) never slices at load time in Rust, so a wrapped range
    // must not panic here — only the file-size Err is checked; the wrap panic is deferred to
    // MmapBuffer::bytes() (I2 fix). The UNKEPT branch slices `mmap->bytes()` immediately below,
    // like heap make_tensor, so it keeps the load-time wrap panic.
    SAPIENT_TRY_ASSIGN(const Range r,
                       data_range(info, data_start, byte_len, mmap->size(), !dtype.has_value()));
    core::Shape shape = shape_of(info.dims);
    if (dtype) {
        // Zero-copy: MmapBuffer.offset = data_start + info.offset, Tensor.offset = 0 (Rust's
        // two-level offset).
        auto buf = std::make_shared<MmapBuffer>(mmap, r.start, byte_len);
        return wrap(core::Tensor::from_buffer(std::move(shape), *dtype, std::move(buf), 0));
    }
    return convert_unkept(
        info.kind, mmap->bytes().subspan(r.start, byte_len), numel, std::move(shape));
}

} // namespace detail

// ── GgufLoader (gguf.rs:680-778) ─────────────────────────────────────────────────────────────
namespace {

core::Error open_or_map_error(const std::filesystem::path& path,
                              const MapError& e,
                              std::string_view map_prefix) {
    if (e.stage == MapStage::Open)
        return core::Error::model_not_found(display_path(path) + ": " + e.os.message);
    return core::Error::gguf_parse(std::string(map_prefix) + e.os.message);
}

core::Result<TensorMap> materialise(const detail::ParsedHeader& h, std::span<const uint8_t> bytes) {
    TensorMap tensors; // no reserve(tensor_count)
    for (const auto& info : h.tensor_infos) {
        SAPIENT_TRY_ASSIGN(core::Tensor t, detail::make_tensor(info, bytes, h.data_start));
        tensors.insert_or_assign(info.name, std::move(t)); // HashMap::insert: last wins
    }
    return tensors;
}

} // namespace

core::Result<GgufMetadata> GgufLoader::parse_metadata_only(const std::filesystem::path& path) {
    auto m = MappedFile::open(path);
    if (!m)
        return tl::unexpected(open_or_map_error(path, m.error(), "mmap failed for header read: "));
    SAPIENT_TRY_ASSIGN(detail::ParsedHeader h, detail::parse_header((*m)->bytes()));
    return std::move(h.metadata);
}

core::Result<std::pair<GgufMetadata, TensorMap>>
GgufLoader::load_tensors_mmap(const std::filesystem::path& path) {
    auto m = MappedFile::open(path);
    if (!m) return tl::unexpected(open_or_map_error(path, m.error(), "mmap failed: "));
    const std::shared_ptr<const MappedFile>& mmap = *m;
    SAPIENT_TRY_ASSIGN(detail::ParsedHeader h, detail::parse_header(mmap->bytes()));
    TensorMap tensors;
    for (const auto& info : h.tensor_infos) {
        SAPIENT_TRY_ASSIGN(core::Tensor t, detail::make_tensor_mmap(info, mmap, h.data_start));
        tensors.insert_or_assign(info.name, std::move(t));
    }
    return std::make_pair(std::move(h.metadata), std::move(tensors));
}

core::Result<TensorMap> GgufLoader::load_tensors(const std::filesystem::path& path) {
    SAPIENT_TRY_ASSIGN(auto both, load_tensors_with_metadata(path));
    return std::move(both.second);
}

core::Result<std::pair<GgufMetadata, TensorMap>>
GgufLoader::load_tensors_with_metadata(const std::filesystem::path& path) {
    auto bytes = read_file(path);
    if (!bytes)
        return tl::unexpected(
            core::Error::model_not_found(display_path(path) + ": " + bytes.error().message));
    SAPIENT_TRY_ASSIGN(detail::ParsedHeader h, detail::parse_header(*bytes));
    SAPIENT_TRY_ASSIGN(TensorMap tensors, materialise(h, *bytes));
    return std::make_pair(std::move(h.metadata), std::move(tensors));
}

core::Result<TensorMap> GgufLoader::tensors_from_bytes(std::span<const uint8_t> bytes) {
    SAPIENT_TRY_ASSIGN(const detail::ParsedHeader h, detail::parse_header(bytes));
    return materialise(h, bytes);
}

} // namespace sapient::io::gguf
