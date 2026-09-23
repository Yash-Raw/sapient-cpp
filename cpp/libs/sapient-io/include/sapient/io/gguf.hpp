// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-io/src/gguf.rs, minus the dead IR entry point `GgufLoader::load -> Graph`
// (sub-project 8). Dims stay in GGUF/ggml order ([in, out] for a linear weight): the flip to HF
// order is sapient-models' job (gguf_weights.rs), exactly as in Rust. Rust-private items that the
// ported tests call live in `gguf::detail`.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "sapient/core/buffer.hpp"
#include "sapient/core/dtype.hpp"
#include "sapient/core/error.hpp"
#include "sapient/core/tensor.hpp"
#include "sapient/io/mmap.hpp"

namespace sapient::io::gguf {

namespace core = sapient::core;

/// Rust `GgufValue::Other` — an unsupported value type or a skipped array.
struct GgufOther {
    bool operator==(const GgufOther&) const = default;
};

/// Rust `enum GgufValue` (gguf.rs:184-201): the 16 alternatives in declaration order.
class GgufValue {
public:
    using Storage = std::variant<uint8_t,
                                 int8_t,
                                 uint16_t,
                                 int16_t,
                                 uint32_t,
                                 int32_t,
                                 float,
                                 bool,
                                 std::string,
                                 uint64_t,
                                 int64_t,
                                 double,
                                 std::vector<uint32_t>,
                                 std::vector<std::string>,
                                 std::vector<float>,
                                 GgufOther>;
    enum class Kind : uint8_t {
        U8,
        I8,
        U16,
        I16,
        U32,
        I32,
        F32,
        Bool,
        Str,
        U64,
        I64,
        F64,
        ArrayU32,
        ArrayStr,
        ArrayF32,
        Other
    };

    GgufValue() : v_(GgufOther{}) {}
    /// Exact alternatives only — `GgufValue(1)` (an int) must not silently become I64.
    template <class T>
        requires(std::is_same_v<std::remove_cvref_t<T>, uint8_t> ||
                 std::is_same_v<std::remove_cvref_t<T>, int8_t> ||
                 std::is_same_v<std::remove_cvref_t<T>, uint16_t> ||
                 std::is_same_v<std::remove_cvref_t<T>, int16_t> ||
                 std::is_same_v<std::remove_cvref_t<T>, uint32_t> ||
                 std::is_same_v<std::remove_cvref_t<T>, int32_t> ||
                 std::is_same_v<std::remove_cvref_t<T>, float> ||
                 std::is_same_v<std::remove_cvref_t<T>, bool> ||
                 std::is_same_v<std::remove_cvref_t<T>, std::string> ||
                 std::is_same_v<std::remove_cvref_t<T>, uint64_t> ||
                 std::is_same_v<std::remove_cvref_t<T>, int64_t> ||
                 std::is_same_v<std::remove_cvref_t<T>, double> ||
                 std::is_same_v<std::remove_cvref_t<T>, std::vector<uint32_t>> ||
                 std::is_same_v<std::remove_cvref_t<T>, std::vector<std::string>> ||
                 std::is_same_v<std::remove_cvref_t<T>, std::vector<float>> ||
                 std::is_same_v<std::remove_cvref_t<T>, GgufOther>)
    explicit GgufValue(T&& v) : v_(std::forward<T>(v)) {}

    Kind kind() const { return static_cast<Kind>(v_.index()); }
    const Storage& storage() const { return v_; }
    template <class T> const T* get_if() const { return std::get_if<T>(&v_); }
    bool operator==(const GgufValue&) const = default;

    std::optional<uint32_t> as_u32() const;
    std::optional<uint64_t> as_u64() const;
    std::optional<float> as_f32() const;
    std::optional<double> as_f64() const;
    std::optional<bool> as_bool() const;
    std::optional<std::string_view> as_str() const;

private:
    Storage v_;
};

using GgufMetadata = std::unordered_map<std::string, GgufValue>;
using TensorMap = std::unordered_map<std::string, sapient::core::Tensor>;

namespace detail {

inline constexpr uint32_t GGUF_MAGIC = 0x46554747; // "GGUF"
inline constexpr uint64_t DEFAULT_ALIGNMENT = 32;
inline constexpr size_t QK_K = 256;

enum class GgmlType : uint8_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    BF16 = 30
};

std::optional<GgmlType> ggml_type_from_u32(uint32_t v);
size_t block_size(GgmlType t);
size_t type_size(GgmlType t);
/// The five types kept as packed blocks (zero-copy on the mmap path).
std::optional<sapient::core::DType> to_sapient_dtype(GgmlType t);
/// Rust's `{:?}` of the variant ("Q4_1", "BF16", …) — used in an error text.
std::string_view debug_name(GgmlType t);
/// `F32|F16|BF16 → numel * type_size`, else `(numel / block_size) * type_size` (truncating,
/// wrapping like Rust release).
size_t tensor_byte_len(GgmlType kind, size_t numel);

struct GgufTensorInfo {
    std::string name;
    std::vector<size_t> dims; // GGUF order, ne0 fastest-varying
    GgmlType kind{GgmlType::F32};
    uint64_t offset{0}; // relative to data_start
};

struct ParsedHeader {
    GgufMetadata metadata;
    std::vector<GgufTensorInfo> tensor_infos;
    size_t data_start{0}; // alignment-corrected
};

core::Result<ParsedHeader> parse_header(std::span<const uint8_t> bytes);

} // namespace detail

/// Rust `pub struct GgufLoader` (gguf.rs:680-778) minus `load -> Graph` (sub-project 8).
class GgufLoader {
public:
    /// Header KV only, via a mapping that is dropped before returning. Zero tensor allocation.
    static core::Result<GgufMetadata> parse_metadata_only(const std::filesystem::path& path);
    /// Q4_0/Q8_0/Q4_K/Q5_K/Q6_K are zero-copy views into the mapping (is_mmap()); F32/F16/BF16
    /// become F32 and Q5_0 becomes Q8_0 on the heap. (Rust's doc comment claims K-quants are
    /// dequantised here — stale; trust the code.)
    static core::Result<std::pair<GgufMetadata, TensorMap>>
    load_tensors_mmap(const std::filesystem::path& path);
    static core::Result<TensorMap> load_tensors(const std::filesystem::path& path);
    /// Reads the WHOLE file into the heap, then copies every tensor out (the ~2× peak-RSS path).
    static core::Result<std::pair<GgufMetadata, TensorMap>>
    load_tensors_with_metadata(const std::filesystem::path& path);
    static core::Result<TensorMap> tensors_from_bytes(std::span<const uint8_t> bytes);
};

namespace detail {

// io's own dequantisers (gguf.rs:259-476) — core's PER-BLOCK arithmetic with io's block
// counts: Q4_0/Q8_0 run bytes/block_bytes blocks and skip writes past numel; Q5_0 and the
// K-quants run numel/block_numel blocks. Q5_K is core's per-element form (io's own copy is stale
// and unreachable — spec §2.1). Returned vectors always have exactly `numel` elements.
std::vector<float> dequantize_q4_0(std::span<const uint8_t> data, size_t numel);
std::vector<float> dequantize_q8_0(std::span<const uint8_t> data, size_t numel);
std::vector<float> dequantize_q5_0(std::span<const uint8_t> data, size_t numel);
std::vector<float> dequantize_q4_k(std::span<const uint8_t> data, size_t numel);
std::vector<float> dequantize_q5_k(std::span<const uint8_t> data, size_t numel);
std::vector<float> dequantize_q6_k(std::span<const uint8_t> data, size_t numel);

/// F32 → ggml Q8_0 blocks (gguf.rs:289-301). A trailing partial chunk is dropped.
std::vector<uint8_t> quantize_to_q8_0(std::span<const float> data);

/// gguf.rs:478-505. Errors: "unsupported GGUF quantization type {Debug}".
core::Result<std::vector<float>>
dequantize_to_f32(GgmlType kind, std::span<const uint8_t> bytes, size_t numel);

/// A read-only window into a MappedFile (gguf.rs:38-78). Holds the mapping alive.
class MmapBuffer final : public core::Buffer {
public:
    MmapBuffer(std::shared_ptr<const MappedFile> mmap, size_t offset, size_t len)
        : mmap_(std::move(mmap)), offset_(offset), len_(len) {}
    std::span<const uint8_t> bytes() const override {
        return mmap_->bytes().subspan(offset_, len_);
    }
    std::span<uint8_t> bytes_mut() override;
    size_t len() const override { return len_; }
    bool is_mmap() const override { return true; }
    size_t alignment() const override { return 32; } // hard-coded GGUF default, as in Rust
    std::string_view device() const override { return "cpu-mmap"; }
    size_t offset() const { return offset_; }

private:
    std::shared_ptr<const MappedFile> mmap_;
    size_t offset_;
    size_t len_;
};

core::Result<core::Tensor>
make_tensor(const GgufTensorInfo& info, std::span<const uint8_t> bytes, size_t data_start);
core::Result<core::Tensor> make_tensor_mmap(const GgufTensorInfo& info,
                                            const std::shared_ptr<const MappedFile>& mmap,
                                            size_t data_start);

} // namespace detail
} // namespace sapient::io::gguf
