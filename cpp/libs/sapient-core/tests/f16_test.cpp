// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

#include "sapient/core/f16.hpp"

using namespace sapient::core;

TEST(F16, widening_is_exact) {
    EXPECT_EQ(f16_bits_to_f32(0x3C00), 1.0f);
    EXPECT_EQ(f16_bits_to_f32(0xC000), -2.0f);
    EXPECT_EQ(f16_bits_to_f32(0x3555), 0.333251953125f);
    EXPECT_EQ(f16_bits_to_f32(0x0001), 5.960464477539063e-08f); // smallest subnormal 2^-24
    EXPECT_EQ(f16_bits_to_f32(0x03FF), 6.097555160522461e-05f); // largest subnormal
    EXPECT_EQ(f16_bits_to_f32(0x0400), 6.103515625e-05f);       // smallest normal 2^-14
    EXPECT_EQ(f16_bits_to_f32(0x7BFF), 65504.0f);
    EXPECT_EQ(f16_bits_to_f32(0x7C00), std::numeric_limits<float>::infinity());
    EXPECT_EQ(f16_bits_to_f32(0xFC00), -std::numeric_limits<float>::infinity());
    EXPECT_EQ(std::bit_cast<uint32_t>(f16_bits_to_f32(0x8000)), 0x80000000u); // -0
    EXPECT_TRUE(std::isnan(f16_bits_to_f32(0x7E00)));
}

TEST(F16, narrowing_rounds_to_nearest_even) {
    EXPECT_EQ(f32_to_f16_bits(1.0f), 0x3C00);
    EXPECT_EQ(f32_to_f16_bits(-2.0f), 0xC000);
    EXPECT_EQ(f32_to_f16_bits(65504.0f), 0x7BFF);
    EXPECT_EQ(f32_to_f16_bits(65520.0f), 0x7C00);       // tie at the top rounds to inf
    EXPECT_EQ(f32_to_f16_bits(1.0009765625f), 0x3C01);  // 1 + 2^-10 exact
    EXPECT_EQ(f32_to_f16_bits(1.00048828125f), 0x3C00); // 1 + 2^-11: tie → even (0x3C00)
    EXPECT_EQ(f32_to_f16_bits(1.00146484375f), 0x3C02); // 1 + 3·2^-11: tie → even (0x3C02)
    EXPECT_EQ(f32_to_f16_bits(5.960464477539063e-08f), 0x0001);  // 2^-24
    EXPECT_EQ(f32_to_f16_bits(2.9802322387695312e-08f), 0x0000); // 2^-25: tie → even (0)
    EXPECT_EQ(f32_to_f16_bits(4.470348358154297e-08f), 0x0001);  // 3·2^-26 > half → 1
    EXPECT_EQ(f32_to_f16_bits(6.103515625e-05f), 0x0400);
    EXPECT_EQ(f32_to_f16_bits(0.0f), 0x0000);
    EXPECT_EQ(f32_to_f16_bits(-0.0f), 0x8000);
    EXPECT_EQ(f32_to_f16_bits(std::numeric_limits<float>::infinity()), 0x7C00);
    EXPECT_EQ(f32_to_f16_bits(1e-30f), 0x0000);
    EXPECT_EQ(f32_to_f16_bits(1e30f), 0x7C00);
    EXPECT_EQ(f32_to_f16_bits(std::numeric_limits<float>::quiet_NaN()) & 0x7E00, 0x7E00);
}

TEST(F16, every_non_nan_half_round_trips) {
    for (uint32_t b = 0; b < 0x10000; ++b) {
        const auto h = static_cast<uint16_t>(b);
        if ((h & 0x7C00) == 0x7C00 && (h & 0x03FF) != 0)
            continue; // NaN payloads are not round-trip tested
        ASSERT_EQ(f32_to_f16_bits(f16_bits_to_f32(h)), h) << std::hex << b;
    }
}

TEST(F16, widening_quiets_signalling_nan) {
    const float h = f16_bits_to_f32(0x7C01); // f16 signalling NaN (exp all-1, mant bit 9 unset)
    EXPECT_TRUE(std::isnan(h));
    EXPECT_NE(std::bit_cast<uint32_t>(h) & 0x00400000u, 0u);

    const float b = bf16_bits_to_f32(0x7F81); // bf16 signalling NaN (exp all-1, mant bit 6 unset)
    EXPECT_TRUE(std::isnan(b));
    EXPECT_NE(std::bit_cast<uint32_t>(b) & 0x00400000u, 0u);
}

TEST(F16, bf16) {
    EXPECT_EQ(bf16_bits_to_f32(0x3F80), 1.0f);
    EXPECT_EQ(f32_to_bf16_bits(1.0f), 0x3F80);
    EXPECT_EQ(f32_to_bf16_bits(std::bit_cast<float>(0x3F808000u)), 0x3F80); // tie → even
    EXPECT_EQ(f32_to_bf16_bits(std::bit_cast<float>(0x3F808001u)), 0x3F81); // above tie
    EXPECT_EQ(f32_to_bf16_bits(std::bit_cast<float>(0x3F818000u)), 0x3F82); // tie, odd → up
    EXPECT_EQ(f32_to_bf16_bits(std::numeric_limits<float>::infinity()), 0x7F80);
    const uint8_t le[2] = {0x80, 0x3F};
    EXPECT_EQ(bf16_le_to_f32(le), 1.0f);
    const uint8_t hle[2] = {0x00, 0x3C};
    EXPECT_EQ(f16_le_to_f32(hle), 1.0f);
    uint8_t out[2];
    f16_to_le(0x3C00, out);
    EXPECT_EQ(out[0], 0x00);
    EXPECT_EQ(out[1], 0x3C);
}
