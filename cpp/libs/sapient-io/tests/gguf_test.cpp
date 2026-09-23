// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
// Port of crates/sapient-io/src/gguf.rs's behaviour: the two Rust unit tests by name (suite
// `Gguf`, Task 3) plus synthetic-file tests for every branch of the parser and both loaders.
// Every expected error literal is verbatim Rust output (probe crate, 2026-09-23).
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "io_test_util.hpp"

#include "sapient/backends_cpu/kernels/quant.hpp"
#include "sapient/core/dequant.hpp"
#include "sapient/core/f16.hpp"
#include "sapient/io/gguf.hpp"
#include "sapient/io/mmap.hpp"
#include "sapient/io/rust_std.hpp"

namespace gguf = sapient::io::gguf;
namespace detail = sapient::io::gguf::detail;
using gguf::GgufLoader;
using gguf::GgufValue;
using sapient::io::test::GgufBuilder;
using sapient::io::test::le_f32;
using sapient::io::test::le_u16;
using sapient::io::test::put_le;
using sapient::io::test::put_str;
using sapient::io::test::TempDir;

namespace {
std::string err_text(const sapient::core::Result<detail::ParsedHeader>& r) {
    return r.has_value() ? std::string("<ok>") : r.error().to_string();
}
const GgufValue& kv(const detail::ParsedHeader& h, const std::string& key) {
    return h.metadata.at(key); // a missing key throws; gtest reports the uncaught exception
}
} // namespace

// ── GgmlType table (gguf.rs:88-178) ───────────────────────────────────────────────────────────
TEST(GgufHeader, ggml_type_table) {
    using detail::GgmlType;
    struct Row {
        uint32_t code;
        GgmlType t;
        size_t block;
        size_t size;
        const char* name;
    };
    const Row rows[] = {
        {0, GgmlType::F32, 1, 4, "F32"},
        {1, GgmlType::F16, 1, 2, "F16"},
        {2, GgmlType::Q4_0, 32, 18, "Q4_0"},
        {3, GgmlType::Q4_1, 32, 20, "Q4_1"},
        {6, GgmlType::Q5_0, 32, 22, "Q5_0"},
        {7, GgmlType::Q5_1, 32, 24, "Q5_1"},
        {8, GgmlType::Q8_0, 32, 34, "Q8_0"},
        {9, GgmlType::Q8_1, 32, 36, "Q8_1"},
        {10, GgmlType::Q2_K, 256, 84, "Q2_K"},
        {11, GgmlType::Q3_K, 256, 110, "Q3_K"},
        {12, GgmlType::Q4_K, 256, 144, "Q4_K"},
        {13, GgmlType::Q5_K, 256, 176, "Q5_K"},
        {14, GgmlType::Q6_K, 256, 210, "Q6_K"},
        {30, GgmlType::BF16, 1, 2, "BF16"},
    };
    for (const auto& r : rows) {
        SCOPED_TRACE(r.name);
        ASSERT_EQ(detail::ggml_type_from_u32(r.code), r.t);
        EXPECT_EQ(detail::block_size(r.t), r.block);
        EXPECT_EQ(detail::type_size(r.t), r.size);
        EXPECT_EQ(detail::debug_name(r.t), r.name);
    }
    for (const uint32_t bad : {4u, 5u, 15u, 16u, 29u, 31u, 99u})
        EXPECT_EQ(detail::ggml_type_from_u32(bad), std::nullopt) << bad;
    using sapient::core::DType;
    EXPECT_EQ(detail::to_sapient_dtype(GgmlType::Q4_0), DType::Q4_0);
    EXPECT_EQ(detail::to_sapient_dtype(GgmlType::Q8_0), DType::Q8_0);
    EXPECT_EQ(detail::to_sapient_dtype(GgmlType::Q4_K), DType::Q4_K);
    EXPECT_EQ(detail::to_sapient_dtype(GgmlType::Q5_K), DType::Q5_K);
    EXPECT_EQ(detail::to_sapient_dtype(GgmlType::Q6_K), DType::Q6_K);
    for (const auto t : {GgmlType::F32,
                         GgmlType::F16,
                         GgmlType::BF16,
                         GgmlType::Q4_1,
                         GgmlType::Q5_0,
                         GgmlType::Q5_1,
                         GgmlType::Q8_1,
                         GgmlType::Q2_K,
                         GgmlType::Q3_K})
        EXPECT_EQ(detail::to_sapient_dtype(t), std::nullopt) << detail::debug_name(t);
    // tensor_byte_len: floats multiply, blocks truncate.
    EXPECT_EQ(detail::tensor_byte_len(GgmlType::F16, 3), 6u);
    EXPECT_EQ(detail::tensor_byte_len(GgmlType::F32, 5), 20u);
    EXPECT_EQ(detail::tensor_byte_len(GgmlType::Q8_0, 64), 68u);
    EXPECT_EQ(detail::tensor_byte_len(GgmlType::Q4_K, 300), 144u);
    EXPECT_EQ(detail::tensor_byte_len(GgmlType::Q5_0, 48), 22u);
}

// ── Scalar KV types (read_value codes 0-8, 10-12) ────────────────────────────────────────────
TEST(GgufHeader, parses_every_scalar_kv_type) {
    GgufBuilder b;
    b.kv_u8("u8", 200).kv_i8("i8", -5).kv_u16("u16", 60000).kv_i16("i16", -30000);
    b.kv_u32("u32", 4000000000u).kv_i32("i32", -7).kv_f32("f32", 1.25f).kv_bool_byte("b", 2);
    b.kv_str("s", "llama").kv_u64("u64", 1ull << 40).kv_i64("i64", -(1ll << 40));
    b.kv_f64("f64", 0.1);
    const auto h = detail::parse_header(b.header_bytes());
    ASSERT_TRUE(h.has_value()) << h.error().to_string();
    EXPECT_EQ(kv(*h, "u8").kind(), GgufValue::Kind::U8);
    EXPECT_EQ(*kv(*h, "u8").get_if<uint8_t>(), 200);
    EXPECT_EQ(*kv(*h, "i8").get_if<int8_t>(), -5);
    EXPECT_EQ(*kv(*h, "u16").get_if<uint16_t>(), 60000);
    EXPECT_EQ(*kv(*h, "i16").get_if<int16_t>(), -30000);
    EXPECT_EQ(*kv(*h, "u32").get_if<uint32_t>(), 4000000000u);
    EXPECT_EQ(*kv(*h, "i32").get_if<int32_t>(), -7);
    EXPECT_EQ(*kv(*h, "f32").get_if<float>(), 1.25f);
    EXPECT_EQ(kv(*h, "b").kind(), GgufValue::Kind::Bool);
    EXPECT_TRUE(*kv(*h, "b").get_if<bool>()); // any non-zero byte is true
    EXPECT_EQ(*kv(*h, "s").get_if<std::string>(), "llama");
    EXPECT_EQ(*kv(*h, "u64").get_if<uint64_t>(), 1ull << 40);
    EXPECT_EQ(*kv(*h, "i64").get_if<int64_t>(), -(1ll << 40));
    EXPECT_EQ(*kv(*h, "f64").get_if<double>(), 0.1);
    EXPECT_EQ(h->metadata.size(), 12u);
}

// ── Accessor widening rules (gguf.rs:203-246) ────────────────────────────────────────────────
TEST(GgufValue, accessor_widening_rules) {
    EXPECT_EQ(GgufValue(uint32_t{7}).as_u32(), 7u);
    EXPECT_EQ(GgufValue(uint64_t{(1ull << 32) + 5}).as_u32(), 5u); // `as u32` truncates
    EXPECT_EQ(GgufValue(int32_t{7}).as_u32(), 7u);
    EXPECT_EQ(GgufValue(int32_t{-1}).as_u32(), std::nullopt);
    EXPECT_EQ(GgufValue(uint16_t{7}).as_u32(), std::nullopt);
    EXPECT_EQ(GgufValue(uint64_t{9}).as_u64(), 9u);
    EXPECT_EQ(GgufValue(uint32_t{9}).as_u64(), 9u);
    EXPECT_EQ(GgufValue(int64_t{9}).as_u64(), std::nullopt);
    EXPECT_EQ(GgufValue(int32_t{9}).as_u64(), std::nullopt);
    EXPECT_EQ(GgufValue(0.1).as_f32(), static_cast<float>(0.1));
    EXPECT_EQ(GgufValue(1.5f).as_f32(), 1.5f);
    EXPECT_EQ(GgufValue(1.5f).as_f64(), 1.5);
    EXPECT_EQ(GgufValue(uint32_t{1}).as_f32(), std::nullopt);
    EXPECT_EQ(GgufValue(true).as_bool(), true);
    EXPECT_EQ(GgufValue(uint8_t{0}).as_bool(), false);
    EXPECT_EQ(GgufValue(uint8_t{3}).as_bool(), true);
    EXPECT_EQ(GgufValue(int8_t{1}).as_bool(), std::nullopt);
    EXPECT_EQ(GgufValue(std::string("x")).as_str(), "x");
    EXPECT_EQ(GgufValue(uint8_t{1}).as_str(), std::nullopt);
    EXPECT_EQ(GgufValue().kind(), GgufValue::Kind::Other);
}

// ── Arrays + the zero-byte skip quirks (read_value code 9, skip_value) ──────────────────────
TEST(GgufHeader, arrays_decode_or_become_other) {
    GgufBuilder b;
    {
        std::vector<uint8_t> items;
        for (const uint32_t v : {1u, 2u, 3u})
            put_le(items, v);
        b.kv_array("u32s", 4, 3, items);
    }
    b.kv_arr_str("strs", {"<s>", "a", "\xC3\xA9"});
    {
        std::vector<uint8_t> items;
        for (const float v : {0.5f, -1.0f})
            put_le(items, v);
        b.kv_array("f32s", 6, 2, items);
    }
    {
        std::vector<uint8_t> items; // item type 5 (i32): skipped 4 bytes each → Other
        for (const int32_t v : {1, -1, 3})
            put_le(items, v);
        b.kv_array("i32s", 5, 3, items);
    }
    {
        std::vector<uint8_t> items; // item type 0 (u8) → Other, 1 byte each
        items = {9, 8};
        b.kv_array("u8s", 0, 2, items);
    }
    b.kv_array("nested", 9, 3, {});         // skip_value(9) consumes NOTHING → no payload
    b.kv_array("unknown_items", 13, 5, {}); // likewise for an unknown item type
    b.kv_u32("after", 42);                  // proves the cursor stayed in sync
    const auto h = detail::parse_header(b.header_bytes());
    ASSERT_TRUE(h.has_value()) << h.error().to_string();
    EXPECT_EQ(*kv(*h, "u32s").get_if<std::vector<uint32_t>>(), (std::vector<uint32_t>{1, 2, 3}));
    EXPECT_EQ(*kv(*h, "strs").get_if<std::vector<std::string>>(),
              (std::vector<std::string>{"<s>", "a", "\xC3\xA9"}));
    EXPECT_EQ(*kv(*h, "f32s").get_if<std::vector<float>>(), (std::vector<float>{0.5f, -1.0f}));
    for (const char* k : {"i32s", "u8s", "nested", "unknown_items"})
        EXPECT_EQ(kv(*h, k).kind(), GgufValue::Kind::Other) << k;
    EXPECT_EQ(kv(*h, "after").as_u32(), 42u);
}

TEST(GgufHeader, unknown_value_type_consumes_nothing) {
    GgufBuilder b;
    b.kv_raw("weird", 13, {}); // type 13: Rust reads no payload → Other
    b.kv_u32("after", 7);
    const auto h = detail::parse_header(b.header_bytes());
    ASSERT_TRUE(h.has_value()) << h.error().to_string();
    EXPECT_EQ(kv(*h, "weird").kind(), GgufValue::Kind::Other);
    EXPECT_EQ(kv(*h, "after").as_u32(), 7u);
}

TEST(GgufHeader, duplicate_keys_last_wins) {
    GgufBuilder b;
    b.kv_u32("k", 1).kv_u32("k", 2);
    const auto h = detail::parse_header(b.header_bytes());
    ASSERT_TRUE(h.has_value());
    EXPECT_EQ(h->metadata.size(), 1u);
    EXPECT_EQ(kv(*h, "k").as_u32(), 2u);
}

// ── Tensor infos, data_start, alignment ─────────────────────────────────────────────────────
TEST(GgufHeader, tensor_infos_keep_gguf_dim_order) {
    GgufBuilder b;
    b.kv_str("general.architecture", "llama");
    b.tensor("w", {64, 32}, 8, std::vector<uint8_t>(size_t{64} * 32 / 32 * 34));
    b.tensor("n", {32}, 0, std::vector<uint8_t>(size_t{32} * 4));
    const auto hb = b.header_bytes();
    const auto h = detail::parse_header(hb);
    ASSERT_TRUE(h.has_value()) << h.error().to_string();
    ASSERT_EQ(h->tensor_infos.size(), 2u);
    EXPECT_EQ(h->tensor_infos[0].name, "w");
    EXPECT_EQ(h->tensor_infos[0].dims, (std::vector<size_t>{64, 32})); // [in, out] — no flip here
    EXPECT_EQ(h->tensor_infos[0].kind, detail::GgmlType::Q8_0);
    EXPECT_EQ(h->tensor_infos[0].offset, 0u);
    EXPECT_EQ(h->tensor_infos[1].offset, GgufBuilder::align_up(size_t{64} * 32 / 32 * 34, 32));
    EXPECT_EQ(h->data_start, GgufBuilder::align_up(hb.size(), 32));
}

TEST(GgufHeader, alignment_from_metadata) {
    for (const bool as_u64 : {false, true}) {
        GgufBuilder b;
        b.alignment = 64;
        if (as_u64)
            b.kv_u64("general.alignment", 64);
        else
            b.kv_u32("general.alignment", 64);
        b.tensor("t", {4}, 0, std::vector<uint8_t>(16));
        const auto hb = b.header_bytes();
        const auto h = detail::parse_header(hb);
        ASSERT_TRUE(h.has_value());
        EXPECT_EQ(h->data_start, GgufBuilder::align_up(hb.size(), 64)) << as_u64;
    }
}

TEST(GgufHeader, alignment_stored_as_i32_is_ignored) {
    GgufBuilder b; // as_u64() of an I32 is None → DEFAULT_ALIGNMENT (faithful Rust quirk)
    b.kv_i32("general.alignment", 64);
    const auto hb = b.header_bytes();
    const auto h = detail::parse_header(hb);
    ASSERT_TRUE(h.has_value());
    EXPECT_EQ(h->data_start, GgufBuilder::align_up(hb.size(), 32));
}

TEST(GgufHeader, zero_alignment_panics_like_rust_div_ceil) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    GgufBuilder b;
    b.kv_u32("general.alignment", 0);
    const auto hb = b.header_bytes();
    EXPECT_DEATH((void)detail::parse_header(hb), "attempt to divide by zero");
}

// ── Header-level errors (texts verbatim from Rust) ───────────────────────────────────────────
TEST(GgufHeader, versions_1_to_3_accepted_others_rejected) {
    for (const uint32_t v : {1u, 2u, 3u}) {
        GgufBuilder b;
        b.version = v;
        EXPECT_TRUE(detail::parse_header(b.header_bytes()).has_value()) << v;
    }
    for (const uint32_t v : {0u, 4u}) {
        GgufBuilder b;
        b.version = v;
        EXPECT_EQ(err_text(detail::parse_header(b.header_bytes())),
                  "GGUF parse error: unsupported GGUF version " + std::to_string(v) +
                      " (expected 1\xE2\x80\x93"
                      "3)");
    }
}

TEST(GgufHeader, bad_magic) {
    GgufBuilder b;
    b.magic = 0x58554747; // "GGUX"
    EXPECT_EQ(err_text(detail::parse_header(b.header_bytes())), "GGUF parse error: bad GGUF magic");
}

TEST(GgufHeader, truncation_at_every_prefix_is_eof) {
    GgufBuilder b;
    b.kv_str("general.architecture", "llama").kv_u32("general.alignment", 32);
    b.kv_arr_str("tokenizer.ggml.tokens", {"a", "bc"});
    b.tensor("w", {32, 2}, 8, std::vector<uint8_t>(68));
    const auto hb = b.header_bytes();
    ASSERT_TRUE(detail::parse_header(hb).has_value());
    for (size_t n = 0; n < hb.size(); ++n)
        ASSERT_EQ(err_text(detail::parse_header(std::span(hb).first(n))),
                  "GGUF parse error: failed to fill whole buffer")
            << "prefix " << n;
}

TEST(GgufHeader, huge_lengths_and_counts_are_eof_not_allocation) {
    const std::string eof = "GGUF parse error: failed to fill whole buffer";
    { // a 2^62-byte key: Rust would try to allocate it; C++ checks the remaining bytes first
        std::vector<uint8_t> v;
        put_le<uint32_t>(v, 0x46554747);
        put_le<uint32_t>(v, 3);
        put_le<uint64_t>(v, 0);
        put_le<uint64_t>(v, 1);
        put_le<uint64_t>(v, 1ull << 62);
        EXPECT_EQ(err_text(detail::parse_header(v)), eof);
    }
    { // 2^60 tensors, 2^60 KVs — no reserve() from header fields
        GgufBuilder b;
        b.tensor_count_override = 1ull << 60;
        EXPECT_EQ(err_text(detail::parse_header(b.header_bytes())), eof);
        GgufBuilder c;
        c.kv_count_override = 1ull << 60;
        EXPECT_EQ(err_text(detail::parse_header(c.header_bytes())), eof);
    }
    { // n_dims = 2^31
        std::vector<uint8_t> v;
        put_le<uint32_t>(v, 0x46554747);
        put_le<uint32_t>(v, 3);
        put_le<uint64_t>(v, 1);
        put_le<uint64_t>(v, 0);
        put_str(v, "w");
        put_le<uint32_t>(v, 1u << 31);
        EXPECT_EQ(err_text(detail::parse_header(v)), eof);
    }
    { // a u32 array claiming 2^60 items with 8 bytes present
        GgufBuilder b;
        const std::vector<uint8_t> items(8, 0);
        b.kv_array("a", 4, 1ull << 60, items);
        EXPECT_EQ(err_text(detail::parse_header(b.header_bytes())), eof);
    }
    { // a zero-width item type claiming 2^60 items: returns Other immediately, never hangs
        GgufBuilder b;
        b.kv_array("a", 9, 1ull << 60, {});
        b.kv_u32("after", 1);
        const auto h = detail::parse_header(b.header_bytes());
        ASSERT_TRUE(h.has_value()) << h.error().to_string();
        EXPECT_EQ(kv(*h, "a").kind(), GgufValue::Kind::Other);
        EXPECT_EQ(kv(*h, "after").as_u32(), 1u);
    }
}

TEST(GgufHeader, invalid_utf8_key_uses_rust_text) {
    GgufBuilder b;
    const std::string bad_key("\xFF", 1);
    b.kv_u32(bad_key, 1);
    EXPECT_EQ(err_text(detail::parse_header(b.header_bytes())),
              "GGUF parse error: invalid utf-8 sequence of 1 bytes from index 0");
}

TEST(GgufHeader, unknown_ggml_type) {
    for (const uint32_t code : {4u, 99u}) {
        GgufBuilder b;
        b.tensor("w", {32}, code, {});
        EXPECT_EQ(err_text(detail::parse_header(b.header_bytes())),
                  "GGUF parse error: unknown ggml type " + std::to_string(code));
    }
}

// ═════════════════════════════════════════ Task 3 ═══════════════════════════════════════════
namespace {

using sapient::core::DType;
using sapient::core::Tensor;

uint32_t lcg(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return s;
}
/// `n` blocks of `block_bytes` random bytes whose f16 fields at `f16_offsets` hold `f16_bits`
/// (so every scale is finite and the dequantised values stay sane).
std::vector<uint8_t> blocks(size_t n,
                            size_t block_bytes,
                            const std::vector<size_t>& f16_offsets,
                            uint16_t f16_bits,
                            uint32_t seed) {
    std::vector<uint8_t> out(n * block_bytes);
    for (auto& b : out)
        b = static_cast<uint8_t>(lcg(seed) >> 24);
    for (size_t k = 0; k < n; ++k)
        for (const size_t off : f16_offsets)
            sapient::core::f16_to_le(f16_bits, out.data() + k * block_bytes + off);
    return out;
}
/// A Q5_0 block with the given f16 scale and 32-bit high-bit mask; nibble byte j = j | (15-j)<<4.
std::vector<uint8_t> q5_0_block(uint16_t scale_bits, uint32_t qh) {
    std::vector<uint8_t> b;
    put_le<uint16_t>(b, scale_bits);
    put_le<uint32_t>(b, qh);
    for (uint32_t j = 0; j < 16; ++j)
        b.push_back(static_cast<uint8_t>(j | ((15u - j) << 4)));
    return b;
}
/// Closed form of the two q5_0_block variants the tests use (see q5_0_closed_form).
std::vector<float> q5_0_expected(uint32_t qh, float scale) {
    std::vector<float> out(32);
    for (int j = 0; j < 16; ++j) {
        if (qh == 0x0000FFFFu) {
            out[static_cast<size_t>(j)] = static_cast<float>(j) * scale;
            out[static_cast<size_t>(j) + 16] = static_cast<float>(-(j + 1)) * scale;
        } else { // 0xFFFF0000
            out[static_cast<size_t>(j)] = static_cast<float>(j - 16) * scale;
            out[static_cast<size_t>(j) + 16] = static_cast<float>(15 - j) * scale;
        }
    }
    return out;
}
bool same_bits(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::bit_cast<uint32_t>(a[i]) != std::bit_cast<uint32_t>(b[i])) return false;
    return true;
}
std::vector<float> f32_of(const Tensor& t) {
    const auto s = t.f32_slice();
    return {s.begin(), s.end()};
}
std::vector<uint8_t> bytes_of(const Tensor& t) {
    const auto s = t.bytes();
    return {s.begin(), s.end()};
}

} // namespace

// ── The 2 Rust tests (gguf.rs:911-935), names verbatim ──────────────────────────────────────
TEST(Gguf, q8_0_quantize_roundtrips_and_sizes) {
    std::vector<float> data(64);
    for (size_t i = 0; i < 64; ++i)
        data[i] = (static_cast<float>(i) - 32.0f) * 0.1f;
    const auto q = detail::quantize_to_q8_0(data);
    ASSERT_EQ(q.size(), 64u / 32u * 34u) << "Q8_0 = 34 bytes / 32 weights";
    const auto back = detail::dequantize_q8_0(q, 64);
    ASSERT_EQ(back.size(), 64u);
    for (size_t i = 0; i < 64; ++i)
        EXPECT_LT(std::fabs(data[i] - back[i]), 0.03f)
            << "roundtrip a=" << data[i] << " b=" << back[i];
}

TEST(Gguf, q8_0_quantize_handles_all_zeros) {
    const std::vector<float> zeros(32, 0.0f);
    const auto q = detail::quantize_to_q8_0(zeros);
    ASSERT_EQ(q.size(), 34u);
    for (const float v : detail::dequantize_q8_0(q, 32))
        EXPECT_EQ(v, 0.0f);
}

// ── quantize_to_q8_0 vs plan D's golden-gated quantize_q8_0_block (differential) ─────────────
TEST(GgufQ8, matches_backends_cpu_quantize_q8_0_block) {
    uint32_t seed = 0xC0FFEE;
    std::vector<float> data(5 * 32 + 7); // the 7-element tail is dropped (chunks_exact)
    for (auto& v : data)
        v = (static_cast<float>(lcg(seed) >> 8) / 16777216.0f - 0.5f) * 8.0f;
    data[3] = 0.0f;
    data[40] = -1e-30f;
    const auto q = detail::quantize_to_q8_0(data);
    ASSERT_EQ(q.size(), 5u * 34u);
    for (size_t b = 0; b < 5; ++b) {
        const auto ref = sapient::backends_cpu::kernels::quant::quantize_q8_0_block(
            std::span<const float>(data).subspan(b * 32, 32));
        EXPECT_TRUE(
            std::equal(ref.begin(), ref.end(), q.begin() + static_cast<std::ptrdiff_t>(b * 34)))
            << "block " << b;
    }
}

TEST(GgufQ8, non_finite_inputs_match_rust_casts) {
    // inf → d = inf, id = 1/inf = 0, inf·0 = NaN → Rust `as i8` = 0 (C++ must not static_cast NaN).
    std::vector<float> a(32, 1.0f);
    a[0] = std::numeric_limits<float>::infinity();
    // NaN is dropped by the f32::max fold; its own quant is NaN·id = NaN → 0.
    std::vector<float> b(32, 2.0f);
    b[5] = std::numeric_limits<float>::quiet_NaN();
    for (const auto* v : {&a, &b}) {
        const auto q = detail::quantize_to_q8_0(*v);
        const auto ref = sapient::backends_cpu::kernels::quant::quantize_q8_0_block(*v);
        EXPECT_TRUE(std::equal(ref.begin(), ref.end(), q.begin()));
    }
    const auto qa = detail::quantize_to_q8_0(a);
    EXPECT_EQ(qa[0], 0x00); // f16 +inf = 0x7C00, little-endian
    EXPECT_EQ(qa[1], 0x7C);
    for (size_t i = 2; i < 34; ++i)
        EXPECT_EQ(qa[i], 0) << i;
}

// ── dequant wrappers ─────────────────────────────────────────────────────────────────────────
TEST(GgufDequant, q5_0_closed_form) {
    auto bytes = q5_0_block(0x3C00, 0x0000FFFFu);        // scale 1.0
    const auto second = q5_0_block(0x3800, 0xFFFF0000u); // scale 0.5
    bytes.insert(bytes.end(), second.begin(), second.end());
    const auto out = detail::dequantize_q5_0(bytes, 64);
    auto expected = q5_0_expected(0x0000FFFFu, 1.0f);
    const auto e2 = q5_0_expected(0xFFFF0000u, 0.5f);
    expected.insert(expected.end(), e2.begin(), e2.end());
    EXPECT_TRUE(same_bits(out, expected));
    // numel not a multiple of 32: numel/32 blocks, the tail stays zero.
    const auto tail = detail::dequantize_q5_0(std::span(bytes).first(22), 48);
    ASSERT_EQ(tail.size(), 48u);
    EXPECT_TRUE(same_bits(std::span(tail).first(32), q5_0_expected(0x0000FFFFu, 1.0f)));
    for (size_t i = 32; i < 48; ++i)
        EXPECT_EQ(tail[i], 0.0f);
}

TEST(GgufDequant, kept_types_use_core_blocks_with_io_block_counts) {
    namespace dq = sapient::core::dequant;
    // Q4_0 / Q8_0: bytes/18 (bytes/34) blocks, writes past numel skipped.
    const auto q4 = blocks(2, 18, {0}, 0x3C00, 1);
    const auto out4 = detail::dequantize_q4_0(q4, 40);
    ASSERT_EQ(out4.size(), 40u);
    std::array<float, 32> b0{}, b1{};
    dq::q4_0_block(q4.data(), b0.data());
    dq::q4_0_block(q4.data() + 18, b1.data());
    EXPECT_TRUE(same_bits(std::span(out4).first(32), b0));
    EXPECT_TRUE(same_bits(std::span(out4).subspan(32), std::span(b1).first(8)));
    const auto q8 = blocks(2, 34, {0}, 0x2C00, 2);
    const auto out8 = detail::dequantize_q8_0(q8, 50);
    std::array<float, 32> c1{};
    dq::q8_0_block(q8.data() + 34, c1.data());
    EXPECT_TRUE(same_bits(std::span(out8).subspan(32), std::span(c1).first(18)));
    // K-quants: numel/256 blocks.
    struct K {
        detail::GgmlType t;
        size_t bb;
        std::vector<size_t> f16s; // not initializer_list: its backing array would dangle
        void (*block)(const uint8_t*, float*);
    };
    const K ks[] = {
        {detail::GgmlType::Q4_K, 144, {0, 2}, dq::q4_k_block},
        {detail::GgmlType::Q5_K, 176, {0, 2}, dq::q5_k_block}, // core's per-element form
        {detail::GgmlType::Q6_K, 210, {208}, dq::q6_k_block}};
    for (const auto& k : ks) {
        SCOPED_TRACE(std::string(detail::debug_name(k.t)));
        const auto raw = blocks(2, k.bb, k.f16s, 0x2C00, 3);
        const auto got = detail::dequantize_to_f32(k.t, raw, 512);
        ASSERT_TRUE(got.has_value());
        std::vector<float> want(512);
        k.block(raw.data(), want.data());
        k.block(raw.data() + k.bb, want.data() + 256);
        EXPECT_TRUE(same_bits(*got, want));
    }
}

TEST(GgufDequant, float_types_and_unsupported_types) {
    const auto f16 = le_u16({0x3C00, 0xC000, 0x7BFF});
    const auto a = detail::dequantize_to_f32(detail::GgmlType::F16, f16, 3);
    ASSERT_TRUE(a.has_value());
    EXPECT_EQ(*a, (std::vector<float>{1.0f, -2.0f, 65504.0f}));
    const auto bf16 = le_u16({0x3F80, 0xBFC0});
    const auto b = detail::dequantize_to_f32(detail::GgmlType::BF16, bf16, 2);
    ASSERT_TRUE(b.has_value());
    EXPECT_EQ(*b, (std::vector<float>{1.0f, -1.5f}));
    const auto f32 = le_f32({0.25f, -8.0f});
    const auto c = detail::dequantize_to_f32(detail::GgmlType::F32, f32, 2);
    ASSERT_TRUE(c.has_value());
    EXPECT_EQ(*c, (std::vector<float>{0.25f, -8.0f}));
    for (const auto t : {detail::GgmlType::Q4_1,
                         detail::GgmlType::Q5_1,
                         detail::GgmlType::Q8_1,
                         detail::GgmlType::Q2_K,
                         detail::GgmlType::Q3_K}) {
        const auto r = detail::dequantize_to_f32(t, {}, 0);
        ASSERT_FALSE(r.has_value());
        EXPECT_EQ(r.error().to_string(),
                  "GGUF parse error: unsupported GGUF quantization type " +
                      std::string(detail::debug_name(t)));
    }
}

// ── Tensor materialisation over all three routes ─────────────────────────────────────────────
namespace {

enum class Route : uint8_t { Heap, Mmap, Bytes };
const char* route_name(Route r) {
    return r == Route::Heap ? "heap" : r == Route::Mmap ? "mmap" : "bytes";
}

GgufBuilder fixture() {
    GgufBuilder b;
    b.kv_str("general.architecture", "llama").kv_u32("llama.block_count", 2);
    b.kv_f32("llama.rope.freq_base", 10000.0f);
    b.kv_arr_str("tokenizer.ggml.tokens", {"<s>", "a", "\xC3\xA9"});
    b.tensor("f32", {3}, 0, le_f32({1.5f, -2.0f, 0.25f}));
    b.tensor("f16", {2, 2}, 1, le_u16({0x3C00, 0xC000, 0x3800, 0x7BFF}));
    b.tensor("bf16", {4}, 30, le_u16({0x3F80, 0xBFC0, 0x3E80, 0x4040}));
    b.tensor("q4_0", {32, 2}, 2, blocks(2, 18, {0}, 0x3C00, 11));
    b.tensor("q8_0", {32}, 8, blocks(1, 34, {0}, 0x2C00, 12));
    b.tensor("q4_k", {256}, 12, blocks(1, 144, {0, 2}, 0x2C00, 13));
    b.tensor("q5_k", {256}, 13, blocks(1, 176, {0, 2}, 0x2C00, 14));
    b.tensor("q6_k", {256, 2}, 14, blocks(2, 210, {208}, 0x2C00, 15));
    auto q5 = q5_0_block(0x3C00, 0x0000FFFFu);
    const auto q5b = q5_0_block(0x3800, 0xFFFF0000u);
    q5.insert(q5.end(), q5b.begin(), q5b.end());
    b.tensor("q5_0", {32, 2}, 6, q5);
    b.tensor("q5_0_tail", {48}, 6, q5_0_block(0x3C00, 0x0000FFFFu)); // 48 % 32 != 0 → F32
    b.tensor("scalar", {}, 0, le_f32({7.0f}));                       // 0-dim → Shape{1}
    return b;
}

sapient::core::Result<gguf::TensorMap>
load(Route r, const std::filesystem::path& p, std::span<const uint8_t> bytes) {
    switch (r) {
    case Route::Heap:
        return GgufLoader::load_tensors(p);
    case Route::Mmap: {
        auto m = GgufLoader::load_tensors_mmap(p);
        if (!m) return tl::unexpected(m.error());
        return std::move(m->second);
    }
    case Route::Bytes:
        return GgufLoader::tensors_from_bytes(bytes);
    }
    return tl::unexpected(sapient::core::Error::internal("route"));
}

} // namespace

TEST(GgufTensors, every_route_materialises_the_fixture) {
    TempDir dir("fixture");
    const GgufBuilder b = fixture();
    const auto bytes = b.build();
    const auto p = dir.write("fixture.gguf", bytes);
    std::vector<float> q5_f32 = q5_0_expected(0x0000FFFFu, 1.0f);
    const auto q5_f32b = q5_0_expected(0xFFFF0000u, 0.5f);
    q5_f32.insert(q5_f32.end(), q5_f32b.begin(), q5_f32b.end());
    for (const Route r : {Route::Heap, Route::Mmap, Route::Bytes}) {
        SCOPED_TRACE(route_name(r));
        auto m = load(r, p, bytes);
        ASSERT_TRUE(m.has_value()) << m.error().to_string();
        ASSERT_EQ(m->size(), 11u);
        const auto& t = [&](const char* name) -> const Tensor& { return m->at(name); };
        // Float sources become F32 (align 64, heap) on every route.
        EXPECT_EQ(t("f32").dtype(), DType::F32);
        EXPECT_EQ(f32_of(t("f32")), (std::vector<float>{1.5f, -2.0f, 0.25f}));
        EXPECT_EQ(t("f16").shape().dims, (std::vector<size_t>{2, 2}));
        EXPECT_EQ(f32_of(t("f16")), (std::vector<float>{1.0f, -2.0f, 0.5f, 65504.0f}));
        EXPECT_EQ(f32_of(t("bf16")), (std::vector<float>{1.0f, -1.5f, 0.25f, 3.0f}));
        for (const char* f : {"f32", "f16", "bf16", "q5_0_tail", "scalar"}) {
            EXPECT_EQ(t(f).dtype(), DType::F32) << f;
            EXPECT_FALSE(t(f).is_mmap()) << f;
            EXPECT_EQ(t(f).buffer().alignment(), 64u) << f << ": from_f32, never from_f32_vec";
        }
        // Q5_0 → re-quantised Q8_0 (numel % 32 == 0); the tail case stays F32 with zeros.
        EXPECT_EQ(t("q5_0").dtype(), DType::Q8_0);
        EXPECT_FALSE(t("q5_0").is_mmap());
        EXPECT_EQ(bytes_of(t("q5_0")), detail::quantize_to_q8_0(q5_f32));
        const auto tail = f32_of(t("q5_0_tail"));
        EXPECT_TRUE(same_bits(std::span(tail).first(32), q5_0_expected(0x0000FFFFu, 1.0f)));
        for (size_t i = 32; i < 48; ++i)
            EXPECT_EQ(tail[i], 0.0f);
        EXPECT_EQ(t("scalar").shape().dims, (std::vector<size_t>{1}));
        // The five kept types: raw bytes, GGUF dim order, zero-copy on the mmap route only.
        struct Kept {
            const char* name;
            DType dtype;
            std::vector<size_t> dims;
        };
        const Kept kept[] = {{"q4_0", DType::Q4_0, {32, 2}},
                             {"q8_0", DType::Q8_0, {32}},
                             {"q4_k", DType::Q4_K, {256}},
                             {"q5_k", DType::Q5_K, {256}},
                             {"q6_k", DType::Q6_K, {256, 2}}};
        for (const auto& k : kept) {
            SCOPED_TRACE(k.name);
            const auto& src = *std::find_if(b.tensors.begin(), b.tensors.end(), [&](const auto& x) {
                return x.name == k.name;
            });
            EXPECT_EQ(t(k.name).dtype(), k.dtype);
            EXPECT_EQ(t(k.name).shape().dims, k.dims);
            EXPECT_EQ(bytes_of(t(k.name)), src.data);
            EXPECT_EQ(t(k.name).is_mmap(), r == Route::Mmap);
            EXPECT_EQ(t(k.name).buffer().alignment(), r == Route::Mmap ? 32u : 16u);
            EXPECT_EQ(t(k.name).buffer().device(), r == Route::Mmap ? "cpu-mmap" : "cpu");
        }
    }
}

TEST(GgufLoader, metadata_agrees_across_entry_points) {
    TempDir dir("metadata");
    const auto p = dir.write("m.gguf", fixture().build());
    auto heap = GgufLoader::load_tensors_with_metadata(p);
    auto mm = GgufLoader::load_tensors_mmap(p);
    auto md = GgufLoader::parse_metadata_only(p);
    ASSERT_TRUE(heap && mm && md);
    EXPECT_EQ(heap->first, mm->first);
    EXPECT_EQ(heap->first, *md);
    EXPECT_EQ(md->at("general.architecture").as_str(), "llama");
    EXPECT_EQ(*md->at("tokenizer.ggml.tokens").get_if<std::vector<std::string>>(),
              (std::vector<std::string>{"<s>", "a", "\xC3\xA9"}));
    auto plain = GgufLoader::load_tensors(p);
    ASSERT_TRUE(plain);
    EXPECT_EQ(plain->size(), heap->second.size());
}

TEST(GgufLoader, duplicate_tensor_names_last_wins) {
    TempDir dir("dup");
    GgufBuilder b;
    b.tensor("w", {1}, 0, le_f32({1.0f}));
    b.tensor("w", {1}, 0, le_f32({2.0f}));
    const auto bytes = b.build();
    const auto p = dir.write("d.gguf", bytes);
    for (const Route r : {Route::Heap, Route::Mmap, Route::Bytes}) {
        auto m = load(r, p, bytes);
        ASSERT_TRUE(m.has_value()) << route_name(r);
        EXPECT_EQ(m->size(), 1u);
        EXPECT_EQ(f32_of(m->at("w")), (std::vector<float>{2.0f})) << route_name(r);
    }
}

TEST(GgufLoader, mmap_tensor_outlives_the_loader) {
    // declared first → destroyed last (Windows cannot delete a mapped file)
    TempDir dir("outlives");
    const GgufBuilder b = fixture();
    const auto p = dir.write("o.gguf", b.build());
    std::optional<Tensor> keep;
    {
        auto m = GgufLoader::load_tensors_mmap(p);
        ASSERT_TRUE(m.has_value());
        keep = m->second.at("q6_k");
    } // the map, the metadata and every other tensor are gone; the MappedFile lives on in `keep`
    ASSERT_TRUE(keep->is_mmap());
    EXPECT_EQ(bytes_of(*keep), b.tensors[7].data); // tensors[7] is "q6_k" in fixture()
    keep.reset();
}

TEST(GgufLoader, mmap_buffer_is_read_only) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    TempDir dir("readonly");
    const auto p = dir.write("r.gguf", fixture().build());
    std::optional<Tensor> t;
    {
        auto m = GgufLoader::load_tensors_mmap(p);
        ASSERT_TRUE(m.has_value());
        t = m->second.at("q4_k");
    } // `t` now holds the only handle to its MmapBuffer, so bytes_mut reaches the buffer
    EXPECT_DEATH((void)t->bytes_mut(),
                 "MmapBuffer is read-only \xE2\x80\x94 model weights cannot be mutated in-place");
    t.reset();
}

// ── Materialisation errors (texts verbatim from Rust) ───────────────────────────────────────
TEST(GgufTensors, data_range_past_end_of_file) {
    TempDir dir("range");
    GgufBuilder b;
    b.tensor("w", {4}, 0, {});
    b.tensors[0].offset_override = 100;
    const auto bytes = b.build();
    const auto p = dir.write("r.gguf", bytes);
    const size_t ds = detail::parse_header(bytes)->data_start;
    const std::string want = "GGUF parse error: tensor 'w': data range [" +
                             std::to_string(ds + 100) + ".." + std::to_string(ds + 116) +
                             "] exceeds file size " + std::to_string(bytes.size());
    for (const Route r : {Route::Heap, Route::Mmap, Route::Bytes}) {
        const auto m = load(r, p, bytes);
        ASSERT_FALSE(m.has_value()) << route_name(r);
        EXPECT_EQ(m.error().to_string(), want) << route_name(r);
    }
}

TEST(GgufTensors, unsupported_quant_type_errors_on_every_route) {
    TempDir dir("q4_1");
    GgufBuilder b;
    b.tensor("w", {32}, 3, std::vector<uint8_t>(20)); // Q4_1
    const auto bytes = b.build();
    const auto p = dir.write("q.gguf", bytes);
    for (const Route r : {Route::Heap, Route::Mmap, Route::Bytes}) {
        const auto m = load(r, p, bytes);
        ASSERT_FALSE(m.has_value());
        EXPECT_EQ(m.error().to_string(),
                  "GGUF parse error: unsupported GGUF quantization type Q4_1")
            << route_name(r);
    }
}

TEST(GgufTensors, zero_dimension_is_wrapped_invalid_graph) {
    // numel = max(product, 1) = 1, so the byte range is valid; the Tensor constructor's
    // Shape::validate then rejects axis 0. Two files: the float branch (from_f32) and the kept
    // branch (from_quant_bytes on heap/bytes, from_buffer on mmap) — materialisation stops at the
    // FIRST bad tensor, so each needs its own file.
    struct Case {
        const char* name;
        std::vector<uint64_t> dims;
        uint32_t kind;
        std::vector<uint8_t> data;
    };
    const Case cases[] = {{"f", {0, 4}, 0, le_f32({1.0f})}, {"q", {0, 32}, 8, {}}};
    for (const auto& c : cases) {
        SCOPED_TRACE(c.name);
        TempDir dir("zero_dim");
        GgufBuilder b;
        b.tensor(c.name, c.dims, c.kind, c.data);
        const auto bytes = b.build();
        const auto p = dir.write("z.gguf", bytes);
        for (const Route r : {Route::Heap, Route::Mmap, Route::Bytes}) {
            const auto m = load(r, p, bytes);
            ASSERT_FALSE(m.has_value());
            EXPECT_EQ(
                m.error().to_string(),
                "GGUF parse error: Graph validation failed: Shape has zero dimension at axis 0")
                << route_name(r);
        }
    }
}

TEST(GgufTensors, wrapped_range_panics_like_rust_slice) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    GgufBuilder b;
    b.tensor("w", {1}, 0, {});
    const auto probe = b.build();
    const size_t ds = detail::parse_header(probe)->data_start;
    // start = ds + offset wraps to 2^64 - 2; end = start + 4 wraps to 2 <= file size → Rust's
    // bounds check passes and `&bytes[start..end]` panics.
    b.tensors[0].offset_override = std::numeric_limits<uint64_t>::max() - 1 - ds;
    const auto bytes = b.build();
    EXPECT_DEATH((void)GgufLoader::tensors_from_bytes(bytes), "slice index starts at");
}

// ── File-level errors per entry point ────────────────────────────────────────────────────────
TEST(GgufLoader, missing_file_is_model_not_found_everywhere) {
    TempDir dir("missing");
    const auto p = dir.path() / "missing.gguf";
    const std::string want = "Model not found at path '" + sapient::io::display_path(p) + ": " +
                             sapient::io::rust_std::os_error_message(2) + "'";
    EXPECT_EQ(GgufLoader::load_tensors(p).error().to_string(), want);
    EXPECT_EQ(GgufLoader::load_tensors_with_metadata(p).error().to_string(), want);
    EXPECT_EQ(GgufLoader::load_tensors_mmap(p).error().to_string(), want);
    EXPECT_EQ(GgufLoader::parse_metadata_only(p).error().to_string(), want);
}

TEST(GgufLoader, empty_file_is_eof_on_every_entry_point) {
    TempDir dir("empty");
    const auto p = dir.write("e.gguf", {});
    const std::string eof = "GGUF parse error: failed to fill whole buffer";
    EXPECT_EQ(GgufLoader::load_tensors(p).error().to_string(), eof);
    EXPECT_EQ(GgufLoader::load_tensors_mmap(p).error().to_string(), eof);
    EXPECT_EQ(GgufLoader::parse_metadata_only(p).error().to_string(), eof);
    EXPECT_EQ(GgufLoader::tensors_from_bytes({}).error().to_string(), eof);
}

#if !defined(_WIN32)
TEST(GgufLoader, directory_wrapping_per_entry_point) {
    TempDir dir("dir");
    const auto& p = dir.path();
    // Heap path: std::fs::read → open succeeds, read fails EISDIR → ModelNotFound.
    EXPECT_EQ(GgufLoader::load_tensors(p).error().to_string(),
              "Model not found at path '" + sapient::io::display_path(p) +
                  ": Is a directory (os error 21)'");
    // Mmap paths: open succeeds, mmap fails (macOS EINVAL / Linux ENODEV) → two different prefixes.
    const auto mm = GgufLoader::load_tensors_mmap(p).error().to_string();
    EXPECT_TRUE(mm.starts_with("GGUF parse error: mmap failed: ")) << mm;
    EXPECT_TRUE(mm.ends_with(")") && mm.find("(os error ") != std::string::npos) << mm;
    const auto md = GgufLoader::parse_metadata_only(p).error().to_string();
    EXPECT_TRUE(md.starts_with("GGUF parse error: mmap failed for header read: ")) << md;
    EXPECT_EQ(mm.substr(std::string("GGUF parse error: mmap failed: ").size()),
              md.substr(std::string("GGUF parse error: mmap failed for header read: ").size()));
}
#endif

// ═════════════════════════════════════ Fix round 1 ══════════════════════════════════════════
// I1: dequantize_to_f32's F32/F16/BF16 arms must return Err (ShapeMismatch via Tensor::from_f32),
// never abort, where Rust's wrapping-multiply range lands <= the byte slice — Rust only panics
// when the wrapped range truly exceeds the slice. `numel * type_size` wraps the SAME way in
// `tensor_byte_len`, so `raw`'s actual byte count always equals the wrapped `n` computed inside
// `dequantize_to_f32` for these three cases; the mismatch surfaces one level up, in `from_f32`.
TEST(GgufTensorsFix1, wrapped_float_numel_is_shape_mismatch_not_abort) {
    struct Case {
        const char* name;
        detail::GgmlType kind;
        uint32_t ggml_code;
        std::vector<uint64_t> dims;
        std::vector<uint8_t> data;
        std::string want; // full error text
    };
    const Case cases[] = {
        {"f32_tail_one",
         detail::GgmlType::F32,
         0,
         {(1ull << 62) + 1},
         le_f32({0.0f}),
         "GGUF parse error: Shape mismatch: expected [4611686018427387905], got [1]"},
        {"f32_tail_zero",
         detail::GgmlType::F32,
         0,
         {1ull << 62},
         {},
         "GGUF parse error: Shape mismatch: expected [4611686018427387904], got [0]"},
        {"f16_tail_one",
         detail::GgmlType::F16,
         1,
         {(1ull << 63) + 1},
         le_u16({0x3C00}),
         "GGUF parse error: Shape mismatch: expected [9223372036854775809], got [1]"},
    };
    for (const auto& c : cases) {
        SCOPED_TRACE(c.name);
        TempDir dir("wrapfloat");
        GgufBuilder b;
        b.tensor(c.name, c.dims, c.ggml_code, c.data);
        const auto bytes = b.build();
        const auto p = dir.write("w.gguf", bytes);
        for (const Route r : {Route::Heap, Route::Mmap, Route::Bytes}) {
            const auto m = load(r, p, bytes);
            ASSERT_FALSE(m.has_value()) << route_name(r);
            EXPECT_EQ(m.error().to_string(), c.want) << route_name(r);
        }
    }
}

TEST(GgufTensorsFix1, dequantize_to_f32_f32_empty_is_empty_vector) {
    const auto r = detail::dequantize_to_f32(detail::GgmlType::F32, {}, 0);
    ASSERT_TRUE(r.has_value()) << r.error().to_string();
    EXPECT_TRUE(r->empty());
}

// I2: make_tensor_mmap's KEPT branch (Q4_0/Q8_0/Q4_K/Q5_K/Q6_K) never slices at load time in
// Rust — a wrapped data range that lands <= file size loads Ok, and the wrap panic is deferred
// to the first read of the tensor's bytes (MmapBuffer::bytes()), not to load_tensors_mmap.
TEST(GgufTensorsFix2, kept_mmap_wrapped_range_panics_on_first_read_not_load) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    TempDir dir(
        "keptwrap"); // declared first → destroyed last (Windows cannot delete a mapped file)
    GgufBuilder b;
    b.tensor("w", {32}, 8, std::vector<uint8_t>(34)); // Q8_0, 1 block = 34 bytes
    const auto probe = b.build();
    const size_t ds = detail::parse_header(probe)->data_start;
    // Same technique as wrapped_range_panics_like_rust_slice: offset wraps `start` down near
    // 2^64, so start + 34 wraps back to <= file size — the load-time file-size check passes.
    b.tensors[0].offset_override = std::numeric_limits<uint64_t>::max() - 1 - ds;
    const auto bytes = b.build();
    const auto p = dir.write("kw.gguf", bytes);
    std::optional<Tensor> t;
    {
        auto m = GgufLoader::load_tensors_mmap(p);
        ASSERT_TRUE(m.has_value()) << m.error().to_string();
        t = m->second.at("w");
    } // `t` now holds the only handle to its MmapBuffer, so bytes() reaches the buffer
    ASSERT_TRUE(t->is_mmap());
    EXPECT_DEATH((void)t->bytes(), "slice index starts at");
    t.reset();
}
