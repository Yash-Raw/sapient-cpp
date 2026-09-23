// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
// Port of crates/sapient-io/src/gguf.rs's behaviour: the two Rust unit tests by name (suite
// `Gguf`, Task 3) plus synthetic-file tests for every branch of the parser and both loaders.
// Every expected error literal is verbatim Rust output (probe crate, 2026-09-23).
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "io_test_util.hpp"

#include "sapient/io/gguf.hpp"

namespace gguf = sapient::io::gguf;
namespace detail = sapient::io::gguf::detail;
using gguf::GgufValue;
using sapient::io::test::GgufBuilder;
using sapient::io::test::put_le;
using sapient::io::test::put_str;

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
    b.tensor("w", {64, 32}, 8, std::vector<uint8_t>(64 * 32 / 32 * 34));
    b.tensor("n", {32}, 0, std::vector<uint8_t>(32 * 4));
    const auto hb = b.header_bytes();
    const auto h = detail::parse_header(hb);
    ASSERT_TRUE(h.has_value()) << h.error().to_string();
    ASSERT_EQ(h->tensor_infos.size(), 2u);
    EXPECT_EQ(h->tensor_infos[0].name, "w");
    EXPECT_EQ(h->tensor_infos[0].dims, (std::vector<size_t>{64, 32})); // [in, out] — no flip here
    EXPECT_EQ(h->tensor_infos[0].kind, detail::GgmlType::Q8_0);
    EXPECT_EQ(h->tensor_infos[0].offset, 0u);
    EXPECT_EQ(h->tensor_infos[1].offset, GgufBuilder::align_up(64 * 32 / 32 * 34, 32));
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
