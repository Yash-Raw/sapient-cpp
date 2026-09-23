// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
// Plan A gate: Tensor::to_f32_vec is bit-identical to the Rust oracle for every stored dtype.
#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "sapient/core/f16.hpp"
#include "sapient/core/tensor.hpp"
#include "sapient/testing/compare.hpp"

using namespace sapient::core;
using sapient::testing::bit_identical;

namespace {
struct QuantCase {
    const char* name;
    DType dtype;
};
} // namespace

class GoldenDequant : public ::testing::TestWithParam<QuantCase> {};

TEST_P(GoldenDequant, matches_rust_to_f32_vec) {
    SAPIENT_GOLDEN_CASE(c, GetParam().name);
    const auto bytes = c.get("in:bytes").as<uint8_t>();
    const auto shape = c.get("param:shape").as<uint32_t>();
    auto t = Tensor::from_quant_bytes(bytes, Shape({shape[0], shape[1]}), GetParam().dtype);
    ASSERT_TRUE(t.has_value()) << t.error().to_string();
    EXPECT_TRUE(bit_identical(t->to_f32_vec(), c.get("out:f32").as<float>()));
}

INSTANTIATE_TEST_SUITE_P(Quant,
                         GoldenDequant,
                         ::testing::Values(QuantCase{"dequant_q4_0", DType::Q4_0},
                                           QuantCase{"dequant_q8_0", DType::Q8_0},
                                           QuantCase{"dequant_q4_k", DType::Q4_K},
                                           QuantCase{"dequant_q5_k", DType::Q5_K},
                                           QuantCase{"dequant_q6_k", DType::Q6_K},
                                           QuantCase{"dequant_q4_k_r4", DType::Q4_K_R4},
                                           QuantCase{"dequant_q6_k_r4", DType::Q6_K_R4}),
                         [](const ::testing::TestParamInfo<QuantCase>& info) {
                             return std::string(info.param.name);
                         });

TEST(GoldenDequantHalf, f16_widening_and_narrowing_match_the_half_crate) {
    SAPIENT_GOLDEN_CASE(c, "dequant_f16");
    const auto bytes = c.get("in:bytes").as<uint8_t>();
    const auto src = c.get("in:src_f32").as<float>();
    auto t = Tensor::from_f16_bytes(bytes, Shape({64}));
    ASSERT_TRUE(t.has_value());
    EXPECT_TRUE(bit_identical(t->to_f32_vec(), c.get("out:f32").as<float>()));
    for (size_t i = 0; i < src.size(); ++i)
        EXPECT_EQ(f32_to_f16_bits(src[i]), load_le16(bytes.data() + 2 * i))
            << "narrowing of " << src[i];
}

TEST(GoldenDequantHalf, bf16_widening_and_narrowing_match_the_half_crate) {
    SAPIENT_GOLDEN_CASE(c, "dequant_bf16");
    const auto bytes = c.get("in:bytes").as<uint8_t>();
    const auto src = c.get("in:src_f32").as<float>();
    auto t = Tensor::from_bf16_bytes(bytes, Shape({64}));
    ASSERT_TRUE(t.has_value());
    EXPECT_TRUE(bit_identical(t->to_f32_vec(), c.get("out:f32").as<float>()));
    for (size_t i = 0; i < src.size(); ++i)
        EXPECT_EQ(f32_to_bf16_bits(src[i]), load_le16(bytes.data() + 2 * i))
            << "narrowing of " << src[i];
}
