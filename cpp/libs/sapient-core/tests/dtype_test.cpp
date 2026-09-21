// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <string>

#include "sapient/core/dtype.hpp"

using namespace sapient::core;

// Rust: element_sizes
TEST(DType, element_sizes) {
    EXPECT_EQ(element_size(DType::F32), 4u);
    EXPECT_EQ(element_size(DType::I64), 8u);
    EXPECT_EQ(element_size(DType::Bool), 1u);
    EXPECT_EQ(element_size(DType::Q4_K), 0u);
}

// Rust: byte_count
TEST(DType, byte_count) {
    EXPECT_EQ(byte_count(DType::F32, 10), 40u);
    EXPECT_EQ(byte_count(DType::Q4_0, 64), 36u);
    EXPECT_EQ(byte_count(DType::Q4_K, 512), 288u);
    EXPECT_EQ(byte_count(DType::Q6_K_R4, 256), 210u);
    EXPECT_EQ(byte_count(DType::Q8_0, 40), 34u); // truncating division, tail dropped like Rust
}

// Rust: from_str_roundtrip
TEST(DType, from_str_roundtrip) {
    const std::pair<const char*, DType> cases[] = {{"f32", DType::F32},
                                                   {"f16", DType::F16},
                                                   {"bf16", DType::BF16},
                                                   {"i32", DType::I32},
                                                   {"i64", DType::I64},
                                                   {"u8", DType::U8},
                                                   {"bool", DType::Bool}};
    for (const auto& [s, dt] : cases) {
        auto r = dtype_from_str(s);
        ASSERT_TRUE(r.has_value()) << s;
        EXPECT_EQ(*r, dt) << s;
        EXPECT_EQ(name(dt), s);
    }
    EXPECT_EQ(*dtype_from_str("Float32"), DType::F32); // lower-cased first
    EXPECT_EQ(*dtype_from_str("q4_k_m"), DType::Q4_K);
    EXPECT_FALSE(
        dtype_from_str("q4_k_r4").has_value()); // emitted by name(), not parseable (Rust asymmetry)
    EXPECT_EQ(dtype_from_str("nope").error().to_string(),
              "Type mismatch: expected a valid dtype, got nope");
    EXPECT_EQ(dtype_from_str("NoPe").error().to_string(),
              "Type mismatch: expected a valid dtype, got nope");
}

// Rust: onnx_roundtrip
TEST(DType, onnx_roundtrip) {
    for (const DType dt :
         {DType::F32, DType::F16, DType::BF16, DType::I32, DType::I64, DType::U8, DType::Bool}) {
        auto r = dtype_from_onnx(dtype_to_onnx(dt));
        ASSERT_TRUE(r.has_value());
        EXPECT_EQ(*r, dt);
    }
    EXPECT_EQ(dtype_to_onnx(DType::Q4_K), 0);
    EXPECT_EQ(dtype_from_onnx(99).error().to_string(),
              "Type mismatch: expected a supported ONNX dtype, got ONNX code 99");
}

TEST(DType, block_constants_and_predicates) {
    EXPECT_EQ(block_bytes(DType::Q4_K_R4), Q4_K_BLOCK_BYTES);
    EXPECT_EQ(block_numel(DType::Q8_0), QUANT_BLOCK_SIZE);
    EXPECT_TRUE(is_quantized(DType::Q6_K_R4));
    EXPECT_TRUE(is_float(DType::BF16));
    EXPECT_TRUE(is_integer(DType::Bool));
    EXPECT_FALSE(is_float(DType::Q8_0));
    EXPECT_EQ(alignment(DType::Q4_0), 2u);
    EXPECT_EQ(alignment(DType::I64), 8u);
    EXPECT_EQ(to_string(DType::Q4_K_R4), "q4_k_r4");
}
