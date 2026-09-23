// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
// SafetensorsLoader (safetensors.rs) + io.hpp (lib.rs). Rust has no tests here; these cover every
// branch. Texts after "tensor '{name}': " (serde_json's in Rust) are C++-authored and only the
// prefix is asserted.
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "io_test_util.hpp"

#include "sapient/core/dtype.hpp"
#include "sapient/io/gguf.hpp"
#include "sapient/io/io.hpp"
#include "sapient/io/mmap.hpp"
#include "sapient/io/rust_std.hpp"
#include "sapient/io/safetensors.hpp"

using sapient::core::DType;
using sapient::io::safetensors::SafetensorsLoader;
using sapient::io::test::le_f32;
using sapient::io::test::le_u16;
using sapient::io::test::put_le;
using sapient::io::test::TempDir;

namespace {
std::vector<uint8_t> st(std::string_view header, const std::vector<uint8_t>& data) {
    std::vector<uint8_t> out;
    put_le<uint64_t>(out, header.size());
    out.insert(out.end(), header.begin(), header.end());
    out.insert(out.end(), data.begin(), data.end());
    return out;
}
std::string err(const std::vector<uint8_t>& file) {
    const auto r = SafetensorsLoader::from_bytes(file);
    return r.has_value() ? std::string("<ok>") : r.error().to_string();
}
std::vector<float> f32_of(const sapient::core::Tensor& t) {
    const auto s = t.f32_slice();
    return {s.begin(), s.end()};
}
} // namespace

TEST(Safetensors, loads_f32_and_keeps_half_types_raw) {
    std::vector<uint8_t> data = le_f32({1.5f, -2.0f});
    const auto h = le_u16({0x3C00, 0xC000});  // F16 1, -2
    const auto bh = le_u16({0x3F80, 0xBFC0}); // BF16 1, -1.5
    data.insert(data.end(), h.begin(), h.end());
    data.insert(data.end(), bh.begin(), bh.end());
    const auto file = st(R"({"__metadata__":{"format":"pt"},)"
                         R"("a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},)"
                         R"("h":{"dtype":"F16","shape":[2],"data_offsets":[8,12]},)"
                         R"("b":{"dtype":"BF16","shape":[1,2],"data_offsets":[12,16],"extra":1}})",
                         data);
    const auto m = SafetensorsLoader::from_bytes(file);
    ASSERT_TRUE(m.has_value()) << m.error().to_string();
    ASSERT_EQ(m->size(), 3u);
    const auto& a = m->at("a");
    EXPECT_EQ(a.dtype(), DType::F32);
    EXPECT_EQ(f32_of(a), (std::vector<float>{1.5f, -2.0f}));
    EXPECT_EQ(a.buffer().alignment(), 64u); // Rust Tensor::from_f32
    EXPECT_FALSE(a.is_mmap());
    const auto& ht = m->at("h");
    EXPECT_EQ(ht.dtype(), DType::F16);
    EXPECT_EQ(ht.to_f32_vec(), (std::vector<float>{1.0f, -2.0f}));
    const auto& bt = m->at("b");
    EXPECT_EQ(bt.dtype(), DType::BF16);
    EXPECT_EQ(bt.shape().dims, (std::vector<size_t>{1, 2}));
    EXPECT_EQ(bt.to_f32_vec(), (std::vector<float>{1.0f, -1.5f}));
}

TEST(Safetensors, half_types_stay_raw) {
    const auto file = st(R"({"h":{"dtype":"F16","shape":[2],"data_offsets":[0,4]},)"
                         R"("b":{"dtype":"BF16","shape":[2],"data_offsets":[4,8]}})",
                         [] {
                             auto v = le_u16({0x3C00, 0xC000});
                             const auto w = le_u16({0x3F80, 0xBFC0});
                             v.insert(v.end(), w.begin(), w.end());
                             return v;
                         }());
    const auto m = SafetensorsLoader::from_bytes(file);
    ASSERT_TRUE(m.has_value()) << m.error().to_string();
    const auto& h = m->at("h");
    EXPECT_EQ(h.dtype(), DType::F16); // NOT converted at load (unlike GGUF F16 → F32)
    EXPECT_EQ(h.buffer().alignment(), 16u);
    EXPECT_EQ(h.to_f32_vec(), (std::vector<float>{1.0f, -2.0f}));
    EXPECT_EQ(m->at("b").dtype(), DType::BF16);
    EXPECT_EQ(m->at("b").to_f32_vec(), (std::vector<float>{1.0f, -1.5f}));
}

TEST(Safetensors, f32_trailing_partial_chunk_is_dropped) {
    // Rust `raw.chunks_exact(4)`: 10 bytes → 2 floats, the 2 trailing bytes are ignored.
    auto data = le_f32({3.0f, 4.0f});
    data.push_back(0xAA);
    data.push_back(0xBB);
    const auto m = SafetensorsLoader::from_bytes(
        st(R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,10]}})", data));
    ASSERT_TRUE(m.has_value()) << m.error().to_string();
    EXPECT_EQ(f32_of(m->at("a")), (std::vector<float>{3.0f, 4.0f}));
}

TEST(Safetensors, scalar_shape_and_metadata_of_any_type) {
    const auto m = SafetensorsLoader::from_bytes(st(
        R"({"__metadata__":"not even an object","s":{"dtype":"F32","shape":[],"data_offsets":[0,4]}})",
        le_f32({9.0f})));
    ASSERT_TRUE(m.has_value()) << m.error().to_string();
    EXPECT_TRUE(m->at("s").shape().dims.empty());
    EXPECT_EQ(f32_of(m->at("s")), (std::vector<float>{9.0f}));
}

TEST(Safetensors, duplicate_keys_last_wins) {
    const auto m =
        SafetensorsLoader::from_bytes(st(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},)"
                                         R"("a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})",
                                         le_f32({1.0f, 2.0f})));
    ASSERT_TRUE(m.has_value()) << m.error().to_string();
    EXPECT_EQ(f32_of(m->at("a")), (std::vector<float>{2.0f}));
}

TEST(Safetensors, seq_form_stmeta_decodes_positionally) {
    // serde's derive(Deserialize) accepts StMeta as a JSON array of its 3 fields in declaration
    // order (dtype, shape, data_offsets), not just as an object — verified against real
    // serde_json 1.0.150 via `from_value::<StMeta>(json!(["F32",[1],[0,4]]))` => Ok.
    const auto m = SafetensorsLoader::from_bytes(st(R"({"a":["F32",[1],[0,4]]})", le_f32({7.5f})));
    ASSERT_TRUE(m.has_value()) << m.error().to_string();
    const auto& a = m->at("a");
    EXPECT_EQ(a.dtype(), DType::F32);
    EXPECT_EQ(a.shape().dims, (std::vector<size_t>{1}));
    EXPECT_EQ(f32_of(a), (std::vector<float>{7.5f}));
}

TEST(Safetensors, header_errors_match_rust) {
    const std::vector<uint8_t> four(4, 0);
    EXPECT_EQ(err({'a', 'b', 'c'}), "Safetensors parse error: file too short");
    {
        std::vector<uint8_t> f;
        put_le<uint64_t>(f, 100);
        f.push_back('{');
        f.push_back('}');
        EXPECT_EQ(err(f), "Safetensors parse error: header overflows file");
    }
    EXPECT_EQ(err(st(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,8]}})", four)),
              "Safetensors parse error: tensor 'a' data out of bounds");
    for (const auto& [dt, disp] : {std::pair{"I32", "i32"},
                                   std::pair{"I64", "i64"},
                                   std::pair{"U8", "u8"},
                                   std::pair{"BOOL", "bool"}})
        EXPECT_EQ(err(st(std::string(R"({"a":{"dtype":")") + dt +
                             R"(","shape":[1],"data_offsets":[0,4]}})",
                         four)),
                  std::string("Safetensors parse error: unsupported safetensors dtype '") + disp +
                      "' for tensor 'a'");
    EXPECT_EQ(err(st(R"({"a":{"dtype":"Q4","shape":[1],"data_offsets":[0,4]}})", four)),
              "Safetensors parse error: unknown dtype 'Q4'");
    EXPECT_EQ(err(st(R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}})", four)),
              "Safetensors parse error: Shape mismatch: expected [2], got [1]");
    EXPECT_EQ(err(st(R"({"a":{"dtype":"F16","shape":[2],"data_offsets":[0,2]}})", four)),
              "Safetensors parse error: Shape mismatch: expected [2], got [1]");
    EXPECT_EQ(
        err(st(R"({"a":{"dtype":"F32","shape":[0],"data_offsets":[0,4]}})", four)),
        "Safetensors parse error: Graph validation failed: Shape has zero dimension at axis 0");
    {
        std::vector<uint8_t> f;
        put_le<uint64_t>(f, 1);
        f.push_back(0xFF);
        EXPECT_EQ(err(f),
                  "Safetensors parse error: invalid utf-8 sequence of 1 bytes from index 0");
    }
}

TEST(Safetensors, strict_stmeta_decoding_prefix_only) {
    // serde rejects all of these; the text after the prefix is C++-authored (not parity-bound).
    const std::vector<uint8_t> four(4, 0);
    const char* bad_entries[] = {
        R"({"a":{"dtype":"F32","shape":[1.0],"data_offsets":[0,4]}})",
        R"({"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}})",
        R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4,4]}})",
        R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0]}})",
        R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4.5]}})",
        R"({"a":{"shape":[1],"data_offsets":[0,4]}})",
        R"({"a":{"dtype":5,"shape":[1],"data_offsets":[0,4]}})",
        R"({"a":{"dtype":"F32","shape":"1","data_offsets":[0,4]}})",
        R"({"a":["F32",[1]]})",         // seq form, 2 elements (too few)
        R"({"a":["F32",[1],[0,4],5]})", // seq form, 4 elements (too many)
    };
    for (const char* h : bad_entries)
        EXPECT_TRUE(err(st(h, four)).starts_with("Safetensors parse error: tensor 'a': ")) << h;
    for (const char* h : {"[1]", "{", "nope", "{\"a\":}"}) {
        const std::string e = err(st(h, four));
        EXPECT_TRUE(e.starts_with("Safetensors parse error: ")) << h;
        EXPECT_NE(e, "<ok>") << h;
    }
}

TEST(Safetensors, bom_prefixed_header_is_rejected) {
    // serde_json rejects a leading UTF-8 BOM outright ("expected value at line 1 column 1");
    // nlohmann's lexer silently skips it — verified against real serde_json 1.0.150.
    const std::string header =
        std::string("\xEF\xBB\xBF") + R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})";
    const auto four = std::vector<uint8_t>(4, 0);
    const auto e = err(st(header, four));
    EXPECT_TRUE(e.starts_with("Safetensors parse error: ")) << e;
    EXPECT_NE(e, "<ok>") << e;
}

TEST(Safetensors, recursion_depth_at_limit_is_ok_beyond_limit_is_rejected) {
    // serde_json's recursion limit: the deepest nesting of `{`/`[` compounds (the root container
    // counts as depth 1) must stay <= 127; depth 128 is Err. Verified empirically against real
    // serde_json 1.0.150 (a pure `[`*N`]`*N probe: N=127 is Ok, N=128 is Err). Here the nesting
    // sits under "__metadata__": root object (depth 1) + N brackets -> max depth N+1, so N=126 is
    // Ok (depth 127) and N=127 is Err (depth 128).
    auto build = [](size_t n) {
        std::string h = R"({"__metadata__":)";
        h.append(n, '[');
        h.append(n, ']');
        h += R"(,"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})";
        return h;
    };
    const auto four = std::vector<uint8_t>(4, 0);
    {
        const auto m = SafetensorsLoader::from_bytes(st(build(126), four));
        ASSERT_TRUE(m.has_value()) << m.error().to_string();
        EXPECT_EQ(f32_of(m->at("a")), (std::vector<float>{0.0f}));
    }
    {
        const auto e = err(st(build(127), four));
        EXPECT_TRUE(e.starts_with("Safetensors parse error: ")) << e;
        EXPECT_NE(e, "<ok>") << e;
    }
}

TEST(Safetensors, recursion_depth_inside_unknown_tensor_field_is_rejected) {
    // Same limit, but the nesting sits inside an unknown field of a tensor entry: root object
    // (depth 1) + tensor object "a" (depth 2) + N brackets -> max depth N+2, so N=125 is Ok
    // (depth 127) and N=126 is Err (depth 128).
    auto build = [](size_t n) {
        std::string h = R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"x":)";
        h.append(n, '[');
        h.append(n, ']');
        h += "}}";
        return h;
    };
    const auto four = std::vector<uint8_t>(4, 0);
    {
        const auto m = SafetensorsLoader::from_bytes(st(build(125), four));
        ASSERT_TRUE(m.has_value()) << m.error().to_string();
    }
    {
        const auto e = err(st(build(126), four));
        EXPECT_TRUE(e.starts_with("Safetensors parse error: ")) << e;
        EXPECT_NE(e, "<ok>") << e;
    }
}

TEST(Safetensors, malformed_ranges_panic_like_rust_slices) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    const std::vector<uint8_t> eight(8, 0);
    // start > end with end in range: Rust `&data_section[start..end]` panics.
    const auto f = st(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}})", eight);
    EXPECT_DEATH((void)SafetensorsLoader::from_bytes(f), "slice index starts at");
    // 8 + header_len wraps to 4 (<= file size): Rust's check passes, then `bytes[8..4]` panics.
    std::vector<uint8_t> g;
    put_le<uint64_t>(g, std::numeric_limits<uint64_t>::max() - 3);
    g.insert(g.end(), 8, 0);
    EXPECT_DEATH((void)SafetensorsLoader::from_bytes(g), "slice index starts at");
}

TEST(Safetensors, empty_file_is_too_short) {
    TempDir dir("st_empty");
    const auto p = dir.write("e.safetensors", {});
    const auto r = SafetensorsLoader::load(p);
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().to_string(), "Safetensors parse error: file too short");
}

TEST(Safetensors, missing_file_and_directory) {
    TempDir dir("st_missing");
    const auto p = dir.path() / "missing.safetensors";
    const auto r = SafetensorsLoader::load(p);
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().to_string(),
              "Model not found at path '" + sapient::io::display_path(p) + ": " +
                  sapient::io::rust_std::os_error_message(2) + "'");
#if !defined(_WIN32)
    // A directory opens and fails at mmap: SafetensorsParseError(e.to_string()) — NO prefix.
    const auto dr = SafetensorsLoader::load(dir.path());
    ASSERT_FALSE(dr.has_value());
    const auto e = dr.error().to_string();
    EXPECT_TRUE(e.starts_with("Safetensors parse error: ")) << e;
    EXPECT_NE(e.find("(os error "), std::string::npos) << e;
    EXPECT_EQ(e.find("mmap failed"), std::string::npos) << e;
#endif
}

TEST(Safetensors, load_copies_everything_out_of_the_mapping) {
    TempDir dir("st_load");
    const auto p = dir.write(
        "w.safetensors",
        st(R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})", le_f32({5.0f, 6.0f})));
    auto m = SafetensorsLoader::load(p);
    ASSERT_TRUE(m.has_value()) << m.error().to_string();
    EXPECT_FALSE(m->at("a").is_mmap()); // Rust drops the Mmap at the end of load()
    EXPECT_EQ(f32_of(m->at("a")), (std::vector<float>{5.0f, 6.0f}));
    auto again = SafetensorsLoader::load_tensors(p);
    ASSERT_TRUE(again.has_value());
    EXPECT_EQ(again->size(), 1u);
}

TEST(Io, load_gguf_and_load_safetensors_delegate) {
    TempDir dir("io_lib");
    const auto sp =
        dir.write("w.safetensors",
                  st(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})", le_f32({1.0f})));
    auto s = sapient::io::load_safetensors(sp);
    ASSERT_TRUE(s.has_value()) << s.error().to_string();
    EXPECT_EQ(s->size(), 1u);
    sapient::io::test::GgufBuilder b;
    b.tensor("w", {2}, 0, le_f32({1.0f, 2.0f}));
    const auto gp = dir.write("w.gguf", b.build());
    auto g = sapient::io::load_gguf(gp);
    ASSERT_TRUE(g.has_value()) << g.error().to_string();
    EXPECT_EQ(g->size(), 1u);
    // The lib.rs re-exports exist at crate root.
    static_assert(std::is_same_v<sapient::io::GgufLoader, sapient::io::gguf::GgufLoader>);
    static_assert(std::is_same_v<sapient::io::SafetensorsLoader, SafetensorsLoader>);
}
