// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

#include "sapient/core/dequant.hpp"
#include "sapient/core/f16.hpp"

using namespace sapient::core;
using namespace sapient::core::dequant;

TEST(Dequant, get_scale_min_k4_unpacks_six_bit_pairs) {
    // scales[j] & 63 for j<4, scales[j+4] & 63 for the mins; j>=4 uses the split-nibble form.
    uint8_t s[12] = {0x3F, 0x01, 0x02, 0x03, 0x3F, 0x11, 0x12, 0x13, 0xAB, 0xCD, 0xEF, 0x21};
    EXPECT_EQ(get_scale_min_k4(0, s).sc, 63);
    EXPECT_EQ(get_scale_min_k4(0, s).m, 63);
    EXPECT_EQ(get_scale_min_k4(1, s).sc, 1);
    EXPECT_EQ(get_scale_min_k4(1, s).m, 0x11);
    // j = 4: sc = (s[8] & 0x0F) | ((s[0] >> 6) << 4) = 0x0B | 0 = 11; m = (s[8] >> 4) | ((s[4] >> 6) << 4) = 0x0A | 0 = 10
    EXPECT_EQ(get_scale_min_k4(4, s).sc, 11);
    EXPECT_EQ(get_scale_min_k4(4, s).m, 10);
    // j = 7: sc = (s[11] & 0x0F) | ((s[3] >> 6) << 4) = 1; m = (s[11] >> 4) | ((s[7] >> 6) << 4) = 2
    EXPECT_EQ(get_scale_min_k4(7, s).sc, 1);
    EXPECT_EQ(get_scale_min_k4(7, s).m, 2);
}

TEST(Dequant, q4_0_block_split_nibble_order) {
    uint8_t block[18] = {};
    f16_to_le(f32_to_f16_bits(0.5f), block); // d = 0.5
    block[2] =
        0xF0; // lo nibble 0 → elem 0 = (0-8)*0.5 = -4; hi nibble 15 → elem 16 = (15-8)*0.5 = 3.5
    float out[32];
    q4_0_block(block, out);
    EXPECT_EQ(out[0], -4.0f);
    EXPECT_EQ(out[16], 3.5f);
    EXPECT_EQ(out[1], -4.0f); // zero nibble → -8 * 0.5
}

TEST(Dequant, q8_0_block) {
    uint8_t block[34] = {};
    f16_to_le(f32_to_f16_bits(2.0f), block);
    block[2] = static_cast<uint8_t>(int8_t{-3});
    block[33] = 127;
    float out[32];
    q8_0_block(block, out);
    EXPECT_EQ(out[0], -6.0f);
    EXPECT_EQ(out[31], 254.0f);
}

TEST(Dequant, q4_k_block_scale_and_min) {
    uint8_t block[144] = {};
    f16_to_le(f32_to_f16_bits(1.0f), block);     // d = 1
    f16_to_le(f32_to_f16_bits(0.5f), block + 2); // dmin = 0.5
    block[4] = 2;                                // sc for sub-block 0 = 2
    block[8] = 1;                                // min for sub-block 0 = 1  → m1v = 0.5
    block[16] =
        0x3A; // qs[0]: lo nibble 0xA=10 → out[0] = 2*10 - 0.5 = 19.5 ; hi nibble 3 → out[32] uses (sc,m) of sub-block 1 = (0,0) → 0
    float out[256];
    q4_k_block(block, out);
    EXPECT_EQ(out[0], 19.5f);
    EXPECT_EQ(out[1], -0.5f); // nibble 0 → 2*0 - 0.5
    EXPECT_EQ(out[32], 0.0f); // sub-block 1: d*0*3 - 0.5*0
}

TEST(Dequant, q6_k_block_scale_indexing) {
    // d = 1; scales[0..16] = 1..16; all q = 32 + i via ql/qh so that (q-32) selects the scale index visibly.
    uint8_t block[210] = {};
    f16_to_le(f32_to_f16_bits(1.0f), block + 208);
    for (int i = 0; i < 16; ++i)
        block[192 + i] = static_cast<uint8_t>(i + 1);
    // Make q1 = 33 for l = 0 (ql low nibble 1, qh bits 10 = 0b10 → +32): ql[0] = 0x01, qh[0] = 0x02
    block[0] = 0x01;
    block[128] = 0x02;
    float out[256];
    q6_k_block(block, out);
    // out[0] = d * sc[sc_base + is + 0] * (q1 - 32) with is = 0 → sc[0] = 1 → 1 * 1 * 1 = 1
    EXPECT_EQ(out[0], 1.0f);
    // out[16]: l = 16 → is = 1 → sc[1] = 2; q1 for l=16: ql[16] = 0, qh[16] = 0 → q = 0 - 32 → 2 * -32 = -64
    EXPECT_EQ(out[16], -64.0f);
    // out[32] (sub 1, l=0): sc[0 + 0 + 2] = 3; q2 = ((ql[32] & 0xF) | (((qh[0] >> 2) & 3) << 4)) - 32 = -32 → -96
    EXPECT_EQ(out[32], -96.0f);
    // second 128-half: sc_base = 8 → out[128] uses sc[8] = 9, q = -32 → -288
    EXPECT_EQ(out[128], -288.0f);
}

TEST(Dequant, r4_depermutation_matches_row_major) {
    // Two rows, one super-block each (k = 256). Packed R4 layout for rows 0..3 requires 4 rows; use 4 rows.
    const size_t rows = 4, k = 256, nb = k / 256;
    std::vector<uint8_t> plain(rows * nb * 144, 0), packed(rows * nb * 144, 0);
    for (size_t r = 0; r < rows; ++r) { // distinct d per row so rows are distinguishable
        f16_to_le(f32_to_f16_bits(static_cast<float>(r + 1)), plain.data() + r * 144);
        plain[r * 144 + 4] = 1;  // sc0 = 1
        plain[r * 144 + 16] = 1; // qs[0] lo nibble = 1 → out[row*256] = (r+1)*1*1 - 0
    }
    // repack: dst block index g*4*nb + b*4 + r ← src (g*4 + r)*nb + b  (single group, nb = 1)
    for (size_t r = 0; r < rows; ++r)
        std::copy_n(plain.data() + r * 144, 144, packed.data() + r * 144);
    std::vector<float> a(rows * k), b(rows * k);
    q4_k(plain, rows * k, a.data());
    q4_k_r4(packed, rows, k, b.data());
    EXPECT_EQ(a, b);
    EXPECT_EQ(b[0], 1.0f);
    EXPECT_EQ(b[3 * 256], 4.0f);
}
