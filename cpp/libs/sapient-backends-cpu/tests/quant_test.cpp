// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
// Port of the `#[cfg(test)] mod tests` of crates/sapient-backends/cpu/src/kernels/quant.rs — the 24
// Rust tests by name (Tasks 1-5 of plan D) plus C++-only pins of the rounding/NaN rules the spec
// (§3.4) makes explicit. aarch64-only Rust tests are compiled only on aarch64; runtime `return`s
// without dotprod/i8mm become GTEST_SKIPs.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "sapient/backends_cpu/cpu_features.hpp"
#include "sapient/backends_cpu/kernels/quant.hpp"
#include "sapient/core/f16.hpp"

using namespace sapient::backends_cpu::kernels::quant;
#if defined(__aarch64__) || defined(_M_ARM64)
// has_dotprod() is only called from aarch64-only tests below; CI's clang-tidy host is x86_64,
// where an unguarded using-declaration would be unused (misc-unused-using-decls). has_i8mm() has
// no Task 1 caller at all — Task 3 adds its own using-declaration when it first calls it.
using sapient::backends_cpu::cpu_features::has_dotprod;
#endif

namespace {

// Deterministic pseudo-random f32 in roughly [-1, 1] (quant.rs `seq`, xorshift64).
std::vector<float> seq(size_t n) {
    uint64_t s = 0x9E3779B97F4A7C15ULL;
    std::vector<float> v(n);
    for (float& x : v) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        x = (static_cast<float>(s >> 40) / static_cast<float>(1u << 24)) * 2.0f - 1.0f;
    }
    return v;
}

// quant.rs `lcg_bytes` / `rand_x`: the 6364136223846793005 · s + 1442695040888963407 LCG.
uint64_t lcg_step(uint64_t& s) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return s;
}
// [[maybe_unused]]: Tasks 2-5 append tests that call this; Task 1 alone does not.
[[maybe_unused]] std::vector<uint8_t> lcg_bytes(uint64_t seed, size_t n) {
    std::vector<uint8_t> v(n);
    for (uint8_t& b : v)
        b = static_cast<uint8_t>(lcg_step(seed) >> 33);
    return v;
}
std::vector<float> rand_x(uint64_t seed, size_t n) {
    std::vector<float> v(n);
    for (float& x : v)
        x = (static_cast<float>(lcg_step(seed) >> 33) /
             static_cast<float>(std::numeric_limits<uint32_t>::max())) *
                3.0f -
            1.5f;
    return v;
}

// quant.rs `q8_0_weight_row`: quantize an f32 row into packed Q8_0 weight blocks.
std::vector<uint8_t> q8_0_weight_row(std::span<const float> w) {
    std::vector<uint8_t> out;
    out.reserve(w.size() / QK * Q8_0_BLOCK_BYTES);
    for (size_t b = 0; b + QK <= w.size(); b += QK) {
        const auto blk = quantize_q8_0_block(w.subspan(b, QK));
        out.insert(out.end(), blk.begin(), blk.end());
    }
    return out;
}

uint32_t bits(float f) {
    return std::bit_cast<uint32_t>(f);
}

} // namespace

// ── Q4_0 / Q8_0 (Task 1) ─────────────────────────────────────────────────────

// The Q8_0 SDOT path quantizes activations to int8. With a per-block scale it must stay close to
// the exact f32 path even when the activation row contains an outlier channel — a per-row scale
// (the old behaviour) diverges wildly here and produced garbage LLM output.
#if defined(__aarch64__) || defined(_M_ARM64)
TEST(Quant, sdot_q8_0_row_blockwise_survives_activation_outlier) {
    if (!has_dotprod()) GTEST_SKIP() << "dotprod not available";
    const size_t k = 256;
    const auto wf = seq(k);
    auto xf = seq(k);
    xf[100] = 60.0f; // outlier channel, ~60× the rest

    const auto w_blocks = q8_0_weight_row(wf);
    const float reference = dot_q8_0_row_f32(w_blocks, xf);

    const auto xq = quantize_row_to_i8_blocks(xf);
    const float blockwise = dot_q8_0_row_sdot(w_blocks, xq.q, xq.scales);
    const float rel = ::fabsf(blockwise - reference) / ::fmaxf(::fabsf(reference), 1e-3f);
    EXPECT_LT(rel, 0.05f) << "blockwise SDOT rel err " << rel << " too high (got " << blockwise
                          << ", ref " << reference << ")";

    // Old path: a single per-row scale set by the outlier collapses the rest.
    float max_abs = 0.0f;
    for (float v : xf)
        max_abs = ::fmaxf(max_abs, ::fabsf(v));
    const float row_scale = max_abs / 127.0f;
    const float inv = 1.0f / row_scale;
    std::vector<int8_t> x_row_i8(k);
    for (size_t i = 0; i < k; ++i)
        x_row_i8[i] = detail::round_clamp_i8(xf[i] * inv);
    const std::vector<float> per_row_scales(k / QK, row_scale);
    const float perrow = dot_q8_0_row_sdot(w_blocks, x_row_i8, per_row_scales);
    const float perrow_rel = ::fabsf(perrow - reference) / ::fmaxf(::fabsf(reference), 1e-3f);
    EXPECT_GT(perrow_rel, rel * 2.0f)
        << "per-row scale should be clearly worse than blockwise (per-row rel " << perrow_rel
        << ", blockwise rel " << rel << ")";
}

TEST(Quant, sdot_q8_0_block_matches_scalar_integer_dot) {
    if (!has_dotprod()) GTEST_SKIP() << "dotprod not available";
    const auto wf = seq(QK);
    const auto xf = seq(QK);
    std::vector<int8_t> w_i8(QK), x_i8(QK);
    for (size_t i = 0; i < QK; ++i) {
        w_i8[i] = detail::round_clamp_i8(wf[i] * 100.0f);
        x_i8[i] = detail::round_clamp_i8(xf[i] * 100.0f);
    }
    std::vector<uint8_t> block(Q8_0_BLOCK_BYTES, 0);
    sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(1.0f), block.data());
    for (size_t i = 0; i < QK; ++i)
        block[2 + i] = static_cast<uint8_t>(w_i8[i]);

    int32_t reference = 0;
    for (size_t i = 0; i < QK; ++i)
        reference += static_cast<int32_t>(w_i8[i]) * static_cast<int32_t>(x_i8[i]);
    EXPECT_EQ(detail::dot_q8_0_block_sdot(block, x_i8), reference)
        << "SDOT integer dot must match scalar reference";
}
#endif

TEST(Quant, q4_0_on_the_fly_dot_matches_dequantized_reference) {
    const size_t k = 256;
    const auto w = seq(k);
    auto x = seq(k);
    for (float& v : x)
        v *= 0.5f;

    const auto blocks = quantize_q4_0_row(w);
    // Storage: 18 bytes per 32 weights = 0.5625 B/weight vs 4 B for F32.
    ASSERT_EQ(blocks.size(), k / QK * Q4_0_BLOCK_BYTES);

    // Reference: dequantize fully, then dot.
    std::vector<float> w_hat(k, 0.0f);
    for (size_t b = 0; b < k / QK; ++b)
        dequantize_q4_0_block(
            std::span<const uint8_t>(blocks).subspan(b * Q4_0_BLOCK_BYTES, Q4_0_BLOCK_BYTES),
            std::span<float>(w_hat).subspan(b * QK, QK));
    float reference = -0.0f; // iter().zip().map().sum() seeds at -0.0
    for (size_t i = 0; i < k; ++i)
        reference += w_hat[i] * x[i];

    const float on_the_fly = dot_q4_0_row_f32(blocks, x);
    EXPECT_LT(::fabsf(on_the_fly - reference), 1e-3f)
        << "on-the-fly " << on_the_fly << " vs reference " << reference;
}

TEST(Quant, q4_0_quantization_error_is_bounded) {
    // Dequantized weights should track the originals within Q4 granularity.
    const auto w = seq(QK * 4);
    const auto blocks = quantize_q4_0_row(w);
    std::vector<float> w_hat(w.size(), 0.0f);
    for (size_t b = 0; b < w.size() / QK; ++b)
        dequantize_q4_0_block(
            std::span<const uint8_t>(blocks).subspan(b * Q4_0_BLOCK_BYTES, Q4_0_BLOCK_BYTES),
            std::span<float>(w_hat).subspan(b * QK, QK));
    float max_err = 0.0f;
    for (size_t i = 0; i < w.size(); ++i)
        max_err = ::fmaxf(max_err, ::fabsf(w[i] - w_hat[i]));
    EXPECT_LT(max_err, 0.2f) << "max quant error " << max_err << " too large";
}

// ── C++-only pins of spec §3.4 (Rust `as` casts, `round`, `f32::max`) ────────

TEST(Quant, nibble_truncates_toward_zero_and_clamps) {
    // ggml: MIN(15, (int)(x*id + 8.5)) — `as i32` truncates toward zero, then clamp 0..15.
    EXPECT_EQ(detail::nibble(0.6f), 9);   // 9.1 → 9
    EXPECT_EQ(detail::nibble(-0.6f), 7);  // 7.9 → 7 (not 8)
    EXPECT_EQ(detail::nibble(-9.0f), 0);  // -0.5 → 0
    EXPECT_EQ(detail::nibble(-9.4f), 0);  // -0.9 → 0 (truncation toward zero)
    EXPECT_EQ(detail::nibble(8.0f), 15);  // 16.5 → 16 → clamp 15
    EXPECT_EQ(detail::nibble(1e30f), 15); // saturating `as i32` → clamp 15
    EXPECT_EQ(detail::nibble(-1e30f), 0);
    EXPECT_EQ(detail::nibble(std::numeric_limits<float>::quiet_NaN()), 0); // NaN as i32 == 0
    EXPECT_EQ(detail::f32_to_i32_sat(3e9f), std::numeric_limits<int32_t>::max());
    EXPECT_EQ(detail::f32_to_i32_sat(-3e9f), std::numeric_limits<int32_t>::min());
    EXPECT_EQ(detail::f32_to_i32_sat(-2.9f), -2);
}

TEST(Quant, round_clamp_i8_rounds_half_away_from_zero_and_maps_nan_to_zero) {
    EXPECT_EQ(detail::round_clamp_i8(0.5f), 1);
    EXPECT_EQ(detail::round_clamp_i8(-0.5f), -1);
    EXPECT_EQ(detail::round_clamp_i8(2.5f), 3); // NOT banker's rounding
    EXPECT_EQ(detail::round_clamp_i8(-2.5f), -3);
    EXPECT_EQ(detail::round_clamp_i8(127.6f), 127);
    EXPECT_EQ(detail::round_clamp_i8(-200.0f), -127); // clamp, not -128
    EXPECT_EQ(detail::round_clamp_i8(std::numeric_limits<float>::infinity()), 127);
    EXPECT_EQ(detail::round_clamp_i8(std::numeric_limits<float>::quiet_NaN()), 0);
    EXPECT_EQ(detail::i8v(0xFF), -1);
    EXPECT_EQ(detail::i8v(0x80), -128);
    EXPECT_EQ(detail::i8v(0x7F), 127);
}

TEST(Quant, activation_quantizers_zero_row_and_nan_semantics) {
    // Zero row: the activation quantisers use scale 1.0 (not 0.0); the weight quantiser uses
    // scale 0 → f16(0) and inv 0 (quant.rs:223-240 vs :317-341, :366-392).
    const std::vector<float> zeros(QK_K, 0.0f);
    const auto i8 = quantize_row_to_i8_blocks(zeros);
    ASSERT_EQ(i8.scales.size(), QK_K / QK);
    for (float s : i8.scales)
        EXPECT_EQ(bits(s), bits(1.0f));
    const auto q8k = quantize_row_to_q8k(zeros);
    ASSERT_EQ(q8k.scales.size(), 1u);
    EXPECT_EQ(bits(q8k.scales[0]), bits(1.0f));
    ASSERT_EQ(q8k.sums.size(), QK_K / QK);
    for (int32_t s : q8k.sums)
        EXPECT_EQ(s, 0);
    const auto blk = quantize_q8_0_block(std::span<const float>(zeros).subspan(0, QK));
    EXPECT_EQ(sapient::core::f16_le_to_f32(blk.data()), 0.0f);
    for (size_t i = 2; i < Q8_0_BLOCK_BYTES; ++i)
        EXPECT_EQ(blk[i], 0);

    // A NaN element: fmaxf drops it from max_abs, and it quantizes to 0 (Rust `NaN as i8`).
    std::vector<float> row(QK, 0.5f);
    row[3] = std::numeric_limits<float>::quiet_NaN();
    const auto r = quantize_row_to_i8_blocks(row);
    EXPECT_EQ(bits(r.scales[0]), bits(0.5f / 127.0f));
    EXPECT_EQ(r.q[3], 0);
    EXPECT_EQ(r.q[0], 127);
    // Q8_K sums are the per-32 sums of the Q8_K quants, i.e. i8_block_sums(q).
    auto x = seq(QK_K);
    x[17] = 20.0f;
    const auto k8 = quantize_row_to_q8k(x);
    const auto sums = i8_block_sums(k8.q);
    ASSERT_EQ(sums.size(), k8.sums.size());
    for (size_t i = 0; i < sums.size(); ++i)
        EXPECT_EQ(sums[i], k8.sums[i]);
}

TEST(Quant, dot_q8_0_row_paths_agree) {
    // f32 path vs the two int8-activation paths: the scalar i8 dot and (under dotprod) SDOT
    // compute the same integer dots and combine in the same order → bit-identical to each other,
    // and both within activation-quantisation error of the f32 path.
    const size_t k = 128;
    const auto wf = seq(k);
    const auto xf = rand_x(0x1234, k);
    const auto w_blocks = q8_0_weight_row(wf);
    const float f32_dot = dot_q8_0_row_f32(w_blocks, xf);
    const auto xq = quantize_row_to_i8_blocks(xf);
    const float i8_dot = dot_q8_0_row_i8_scalar(w_blocks, xq.q, xq.scales);
    EXPECT_LT(::fabsf(f32_dot - i8_dot) / ::fmaxf(::fabsf(f32_dot), 1e-3f), 0.03f)
        << f32_dot << " vs " << i8_dot;
#if defined(__aarch64__) || defined(_M_ARM64)
    if (has_dotprod()) {
        const float sdot = dot_q8_0_row_sdot(w_blocks, xq.q, xq.scales);
        EXPECT_EQ(bits(sdot), bits(i8_dot)) << sdot << " vs " << i8_dot;
    }
#endif
}

TEST(QuantDeath, block_quantizers_panic_on_wrong_length) {
    const std::vector<float> x(31, 0.0f);
    EXPECT_DEATH((void)quantize_q8_0_block(x), "");
    EXPECT_DEATH((void)quantize_q4_0_block(x), "");
    const std::vector<float> x33(33, 0.0f);
    EXPECT_DEATH((void)quantize_row_to_i8_blocks(x33), "");
}
