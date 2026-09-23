// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "sapient/core/dequant.hpp"
#include "sapient/core/f16.hpp"

using namespace sapient::core;
using namespace sapient::core::dequant;

TEST(Dequant, get_scale_min_k4_unpacks_six_bit_pairs) {
    // scales[j] & 63 for j<4, scales[j+4] & 63 for the mins; j>=4 uses the split-nibble form.
    uint8_t s[12] = {0x3F, 0x01, 0x02, 0x03, 0x3F, 0x11, 0x12, 0x13, 0xAB, 0xCD, 0xEF, 0x21};
    // Every `>> 6` source byte above has its top two bits clear, so j>=4's `<< 4` term would
    // silently contribute 0 without this: set the top two bits of s[0] and s[4] so the
    // high-bit assembly for j=4 is actually exercised. `& 63` masks off exactly those two
    // bits, so the j<4 (sc(0), m(0)) assertions below are unaffected.
    s[0] |= 0xC0; // 0x3F -> 0xFF
    s[4] |= 0x40; // 0x3F -> 0x7F
    EXPECT_EQ(get_scale_min_k4(0, s).sc, 63);
    EXPECT_EQ(get_scale_min_k4(0, s).m, 63);
    EXPECT_EQ(get_scale_min_k4(1, s).sc, 1);
    EXPECT_EQ(get_scale_min_k4(1, s).m, 0x11);
    // j = 4: sc = (s[8] & 0x0F) | ((s[0] >> 6) << 4) = 0x0B | ((0xFF >> 6) << 4) = 0x0B | 0x30 = 0x3B = 59
    // m  = (s[8] >> 4) | ((s[4] >> 6) << 4) = 0x0A | ((0x7F >> 6) << 4) = 0x0A | 0x10 = 0x1A = 26
    EXPECT_EQ(get_scale_min_k4(4, s).sc, 59);
    EXPECT_EQ(get_scale_min_k4(4, s).m, 26);
    // j = 7: sc = (s[11] & 0x0F) | ((s[3] >> 6) << 4) = 1; m = (s[11] >> 4) | ((s[7] >> 6) << 4) = 2
    // (s[3] and s[7] are untouched, so these are unchanged from before.)
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

TEST(Dequant, q5_k_high_bit_read_per_element) {
    // The historical bug read the 5th bit as `qh[is/8]` — ONE bit per 32-element sub-block —
    // instead of ggml's per-ELEMENT `qh[l]`. This fixture isolates g=1 (is=2, u1=4, ql_off=32,
    // out_idx=64) so the two forms diverge: the fixed form reads `qh[5]` (bit 2 set, since
    // l=5 lands on out[69]) while the historical bug would read `qh[is/8]` = `qh[0]` (unset)
    // for every l in this pass.
    uint8_t block[176] = {};
    f16_to_le(f32_to_f16_bits(1.0f), block);     // d = 1
    f16_to_le(f32_to_f16_bits(0.0f), block + 2); // dmin = 0 (keeps m1v = 0 regardless of .m)
    block[4] = 0x01;      // scales[0]: j=0 sc = 1 (g=0's `a` sub-block; m = scales[4]&63 = 0)
    block[6] = 0x01;      // scales[2]: j=2 sc = 1 (g=1's `a` sub-block, is=2; m = scales[6]&63 = 0)
    block[16 + 5] = 0x04; // qh[5] = 0x04 (bit 2 set); qh is block+16
    block[48 + 37] = 0x03; // ql[37] = 0x03 (lo nibble = 3); ql is block+48
    float out[256];
    q5_k_block(block, out);
    // g=1, l=5: out[69] = d1*(ql[37]&0x0F + hi) - m1v = 1*(3 + 16) - 0 = 19
    // (hi=16 because qh[5]&u1 = 0x04&0x04 != 0). The historical qh[is/8]=qh[0] bug would
    // read qh[0]=0 here (0 & 4 == 0 → hi=0) and give 3.0f instead.
    EXPECT_EQ(out[69], 19.0f);
    // g=1, l=4: ql[36] and qh[4] are both unset, so this lo-nibble slot is otherwise inert —
    // shows the bit is read per element (l=4 vs l=5), not applied to the whole sub-block.
    EXPECT_EQ(out[68], static_cast<float>(block[48 + 36] & 0x0F));
    // g=0, l=0: qh[0] is unset, so u1=1 does not trigger here either.
    EXPECT_EQ(out[0], 0.0f);
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
    // out[64] (sub 2, l=0): sc[0 + 0 + 4] = 5; q3 = ((ql[0] >> 4) | (((qh[0] >> 4) & 3) << 4)) - 32 = -32 → -160
    EXPECT_EQ(out[64], -32.0f * 5);
    // out[96] (sub 3, l=0): sc[0 + 0 + 6] = 7; q4 = ((ql[32] >> 4) | (((qh[0] >> 6) & 3) << 4)) - 32 = -32 → -224
    EXPECT_EQ(out[96], -32.0f * 7);
    // second 128-half: sc_base = 8 → out[128] uses sc[8] = 9, q = -32 → -288
    EXPECT_EQ(out[128], -288.0f);
}

TEST(Dequant, r4_depermutation_matches_row_major) {
    // rows=4, k=512 → nb=2 super-blocks per row: a REAL 4-row block-major repack (not the
    // k=256/nb=1 degenerate case, where packed==plain trivially and the test would pass even
    // with a wrong or missing permutation). Layout mirrors repack_q4_k_rows4
    // (crates/sapient-backends/cpu/src/kernels/quant.rs:1224-1240): single group (g=0, since
    // rows=4), packed block index `b*4 + r` <- plain (row-major) block index `r*nb + b`.
    const size_t rows = 4, k = 512, nb = k / 256;
    std::vector<uint8_t> plain(rows * nb * 144, 0), packed(rows * nb * 144, 0);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t bi = 0; bi < nb; ++bi) {
            const size_t src = r * nb + bi;                      // plain (row-major) block index
            const float d = static_cast<float>(10 * r + bi + 1); // unique per (row, block)
            f16_to_le(f32_to_f16_bits(d), plain.data() + src * 144);
            plain[src * 144 + 4] = 1;  // sc0 = 1
            plain[src * 144 + 16] = 1; // qs[0] lo nibble = 1 → block's out[0] = d*1*1 - 0 = d
        }
    }
    for (size_t r = 0; r < rows; ++r) {
        for (size_t bi = 0; bi < nb; ++bi) {
            const size_t src = r * nb + bi;
            const size_t dst = bi * 4 + r; // single group: dst = g*4*nb + bi*4 + r, g = 0
            std::copy_n(plain.data() + src * 144, 144, packed.data() + dst * 144);
        }
    }
    std::vector<float> a(rows * k), b(rows * k);
    q4_k(plain, rows * k, a.data());
    q4_k_r4(packed, rows, k, b.data());
    EXPECT_EQ(a, b);
    EXPECT_EQ(b[0], 1.0f);        // row 0, block 0: d = 10*0 + 0 + 1 = 1
    EXPECT_EQ(b[7 * 256], 32.0f); // row 3, block 1: d = 10*3 + 1 + 1 = 32, offset (3*2+1)*256
}

TEST(Dequant, r4_rejects_partial_row_groups) {
    std::vector<uint8_t> bytes(576, 0); // 4 Q4_K blocks (irrelevant content — panics before use)
    std::vector<float> out(1024, 0.0f);
    EXPECT_DEATH(q4_k_r4(bytes, 2, 512, out.data()), "dequant r4");
}

TEST(Dequant, r4_rejects_k_not_multiple_of_256) {
    std::vector<uint8_t> bytes(144, 0); // rows=4 passes the rows%4 check; k=128 must not
    std::vector<float> out(1024, 0.0f);
    EXPECT_DEATH(q4_k_r4(bytes, 4, 128, out.data()), "dequant r4");
}
