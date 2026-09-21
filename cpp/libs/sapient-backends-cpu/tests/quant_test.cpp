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
#include "sapient/core/dequant.hpp"
#include "sapient/core/dtype.hpp"
#include "sapient/core/f16.hpp"
#include "sapient/core/shape.hpp"
#include "sapient/core/tensor.hpp"

using namespace sapient::backends_cpu::kernels::quant;
#if defined(__aarch64__) || defined(_M_ARM64)
// has_dotprod() is only called from aarch64-only tests below; CI's clang-tidy host is x86_64,
// where an unguarded using-declaration would be unused (misc-unused-using-decls). has_i8mm() has
// no Task 1 caller at all — Task 3 adds its own using-declaration when it first calls it.
using sapient::backends_cpu::cpu_features::has_dotprod;
using sapient::backends_cpu::cpu_features::has_i8mm;
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
std::vector<uint8_t> lcg_bytes(uint64_t seed, size_t n) {
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

// ── Q4_K (Task 2) ────────────────────────────────────────────────────────────

namespace {
// quant.rs: 8 pseudo-random Q4_K rows with SMALL positive f16 d/dmin (bytes 0x11,0x2c) so the
// magnitudes stay sane; the LCG only advances on non-header bytes (the `_ => nb()` arm).
std::vector<uint8_t> q4_k_test_rows(size_t n, size_t k, uint64_t seed) {
    const size_t row_bytes = k / 256 * Q4_K_BLOCK_BYTES;
    std::vector<uint8_t> rows(n * row_bytes);
    for (size_t i = 0; i < rows.size(); ++i) {
        switch (i % Q4_K_BLOCK_BYTES) {
        case 0:
        case 2:
            rows[i] = 0x11;
            break;
        case 1:
        case 3:
            rows[i] = 0x2c;
            break;
        default:
            rows[i] = static_cast<uint8_t>(lcg_step(seed) >> 33);
        }
    }
    return rows;
}
std::vector<float> ramp(size_t k, size_t mul, size_t mod, float sub, float step) {
    std::vector<float> x(k);
    for (size_t i = 0; i < k; ++i)
        x[i] = (static_cast<float>(i * mul % mod) - sub) * step;
    return x;
}
std::span<const uint8_t> row_of(const std::vector<uint8_t>& rows, size_t r, size_t row_bytes) {
    return std::span<const uint8_t>(rows).subspan(r * row_bytes, row_bytes);
}
} // namespace

TEST(Quant, q4_k_w4a8_matches_f32_path) {
    // The W4A8 (int8-activation) Q4_K dot must agree with the proven f32 path within
    // activation-quantization error. A layout/scale bug (the kind that produced Q6_K salad) shows
    // up as a wildly-wrong result, not a few %.
    uint64_t seed = 0x12345678ULL;
    auto next = [&seed]() { return static_cast<uint32_t>(lcg_step(seed) >> 33); };
    const size_t nblocks = 2; // 512 weights → 16 activation blocks of 32
    std::vector<uint8_t> row(nblocks * Q4_K_BLOCK_BYTES, 0);
    for (size_t b = 0; b < nblocks; ++b) {
        uint8_t* blk = row.data() + b * Q4_K_BLOCK_BYTES;
        sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(0.05f), blk);      // d
        sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(0.018f), blk + 2); // dmin
        for (size_t i = 4; i < Q4_K_BLOCK_BYTES; ++i)
            blk[i] = static_cast<uint8_t>(next() & 0xFF); // scales + packed nibbles
    }
    std::vector<float> x(nblocks * QK_K);
    for (float& v : x)
        v = (static_cast<float>(next()) /
             static_cast<float>(std::numeric_limits<uint32_t>::max())) *
                4.0f -
            2.0f;

    const float f32_dot = dot_q4_k_row_f32(row, x);
    const auto xq = quantize_row_to_i8_blocks(x);
    const auto xsums = i8_block_sums(xq.q);
    const float q8_dot = dot_q4_k_row_q8_scalar(row, xq.q, xq.scales, xsums);
    const float rel = ::fabsf(f32_dot - q8_dot) / ::fmaxf(::fabsf(f32_dot), 1e-3f);
    EXPECT_LT(rel, 0.03f) << "W4A8 mismatch: f32=" << f32_dot << " q8=" << q8_dot;

    // The NEON SDOT kernel must match the scalar W4A8 reference exactly (same integer dot; only
    // f32 reduction order differs → tiny tolerance).
#if defined(__aarch64__) || defined(_M_ARM64)
    if (has_dotprod()) {
        const float neon = dot_q4_k_row_q8_neon(row, xq.q, xq.scales, xsums);
        const float rel_n = ::fabsf(neon - q8_dot) / ::fmaxf(::fabsf(q8_dot), 1e-3f);
        EXPECT_LT(rel_n, 1e-4f) << "NEON≠scalar W4A8: neon=" << neon << " scalar=" << q8_dot;
    }
#endif
}

#if defined(__aarch64__) || defined(_M_ARM64)
TEST(Quant, q4_k_4rows_matches_single_row) {
    if (!has_dotprod()) GTEST_SKIP() << "dotprod not available";
    const size_t k = 512;
    const size_t row_bytes = k / 256 * Q4_K_BLOCK_BYTES;
    const auto rows = q4_k_test_rows(8, k, 0x5EEDULL);
    const auto x = ramp(k, 37, 97, 48.0f, 0.02f);
    const auto xq = quantize_row_to_i8_blocks(x);
    const auto x_sums = i8_block_sums(xq.q);
    for (size_t group = 0; group < 2; ++group) {
        const size_t j = group * 4;
        const std::array<std::span<const uint8_t>, 4> r4 = {row_of(rows, j, row_bytes),
                                                            row_of(rows, j + 1, row_bytes),
                                                            row_of(rows, j + 2, row_bytes),
                                                            row_of(rows, j + 3, row_bytes)};
        const auto got = dot_q4_k_4rows_q8_neon(r4, xq.q, xq.scales, x_sums);
        for (size_t o = 0; o < 4; ++o) {
            const float want = dot_q4_k_row_q8_neon(r4[o], xq.q, xq.scales, x_sums);
            EXPECT_EQ(bits(got[o]), bits(want))
                << "row " << j + o << " differs: " << got[o] << " vs " << want;
        }
    }
}
#endif

TEST(Quant, q4_k_r4_repack_roundtrips_through_dequant) {
    // to_f32_vec on a repacked tensor must equal to_f32_vec on the original (the de-interleave
    // map is the inverse of the repack permutation).
    using sapient::core::DType;
    using sapient::core::Shape;
    using sapient::core::Tensor;
    const size_t n = 8, k = 512;
    const auto blocks = q4_k_test_rows(n, k, 0x00D5ULL);
    auto orig = Tensor::from_quant_bytes(blocks, Shape{n, k}, DType::Q4_K);
    ASSERT_TRUE(orig.has_value()) << orig.error().to_string();
    const auto packed = repack_q4_k_rows4(blocks, n, k);
    auto r4 = Tensor::from_quant_bytes(packed, Shape{n, k}, DType::Q4_K_R4);
    ASSERT_TRUE(r4.has_value()) << r4.error().to_string();
    EXPECT_EQ(orig->to_f32_vec(), r4->to_f32_vec());
}

#if defined(__aarch64__) || defined(_M_ARM64)
TEST(Quant, q4_k_r4_kernel_matches_single_row) {
    if (!has_dotprod()) GTEST_SKIP() << "dotprod not available";
    const size_t n = 4, k = 512;
    const size_t row_bytes = k / 256 * Q4_K_BLOCK_BYTES;
    const auto blocks = q4_k_test_rows(n, k, 0x0B0BULL);
    const auto x = ramp(k, 53, 89, 44.0f, 0.02f);
    const auto xq = quantize_row_to_i8_blocks(x);
    const auto x_sums = i8_block_sums(xq.q);
    const auto packed = repack_q4_k_rows4(blocks, n, k);
    const auto got = dot_q4_k_4rows_r4_neon(packed, xq.q, xq.scales, x_sums);
    for (size_t r = 0; r < 4; ++r) {
        const float want =
            dot_q4_k_row_q8_neon(row_of(blocks, r, row_bytes), xq.q, xq.scales, x_sums);
        EXPECT_EQ(bits(got[r]), bits(want)) << "row " << r << ": " << got[r] << " vs " << want;
    }
}
#endif

TEST(QuantDeath, repack_q4_k_rows4_asserts_like_rust) {
    const std::vector<uint8_t> two_rows(2 * Q4_K_BLOCK_BYTES, 0);
    EXPECT_DEATH((void)repack_q4_k_rows4(two_rows, 2, 256), "multiple of 4");
    const std::vector<uint8_t> four_rows(4 * Q4_K_BLOCK_BYTES, 0);
    EXPECT_DEATH((void)repack_q4_k_rows4(four_rows, 4, 200), "");
    EXPECT_DEATH((void)repack_q4_k_rows4(two_rows, 4, 256), ""); // blocks.len() != n * row_bytes
}

// ── Q4_K × Q8_K, SMMLA (Task 3) ───────────────────────────────────────────────

namespace {
// quant.rs q4_k_q8k_scalar_matches_f32_path / *_q8k_kernels_match_single_row: rows filled with
// `(i*A + B) % M` bytes, then every block's d/dmin overwritten with fixed f16 values.
std::vector<uint8_t>
q4_k_mod_rows(size_t n, size_t k, size_t a, size_t b, size_t m, float d, float dmin) {
    const size_t row_bytes = k / QK_K * Q4_K_BLOCK_BYTES;
    std::vector<uint8_t> rows(n * row_bytes);
    for (size_t i = 0; i < rows.size(); ++i)
        rows[i] = static_cast<uint8_t>((i * a + b) % m);
    for (size_t r = 0; r < n; ++r)
        for (size_t blk = 0; blk < k / QK_K; ++blk) {
            uint8_t* base = rows.data() + r * row_bytes + blk * Q4_K_BLOCK_BYTES;
            sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(d), base);
            sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(dmin), base + 2);
        }
    return rows;
}
} // namespace

#if defined(__aarch64__) || defined(_M_ARM64)
TEST(Quant, q4_k_smmla_x2_matches_single_row) {
    if (!has_i8mm()) GTEST_SKIP() << "i8mm not available";
    const size_t n = 4, k = 512;
    const size_t row_bytes = k / 256 * Q4_K_BLOCK_BYTES;
    const auto rows = q4_k_test_rows(n, k, 0x18AAULL);
    const auto x0 = ramp(k, 37, 97, 48.0f, 0.02f);
    const auto x1 = ramp(k, 59, 101, 50.0f, 0.015f);
    const auto q0 = quantize_row_to_i8_blocks(x0);
    const auto q1 = quantize_row_to_i8_blocks(x1);
    const auto b0 = i8_block_sums(q0.q);
    const auto b1 = i8_block_sums(q1.q);
    const auto packed = repack_q4_k_rows4(rows, n, k);
    const auto got = dot_q4_k_4rows_r4_x2_smmla(packed, q0.q, q0.scales, b0, q1.q, q1.scales, b1);
    for (size_t r = 0; r < 4; ++r) {
        const auto row = row_of(rows, r, row_bytes);
        const float w0 = dot_q4_k_row_q8_neon(row, q0.q, q0.scales, b0);
        const float w1 = dot_q4_k_row_q8_neon(row, q1.q, q1.scales, b1);
        EXPECT_EQ(bits(got[r][0]), bits(w0)) << "row " << r << " x0: " << got[r][0] << " vs " << w0;
        EXPECT_EQ(bits(got[r][1]), bits(w1)) << "row " << r << " x1: " << got[r][1] << " vs " << w1;
    }
}
#endif

TEST(Quant, q4_k_q8k_scalar_matches_f32_path) {
    const size_t nblocks = 3;
    const auto row = q4_k_mod_rows(1, nblocks * QK_K, 197, 13, 251, 0.05f, 0.03f);
    uint64_t state = 0x2545F4914F6CDD1DULL;
    auto next = [&state]() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    };
    std::vector<float> x(nblocks * QK_K);
    for (float& v : x)
        v = (static_cast<float>(next()) /
             static_cast<float>(std::numeric_limits<uint64_t>::max())) *
                4.0f -
            2.0f;

    const float f32_dot = dot_q4_k_row_f32(row, x);
    const auto k8 = quantize_row_to_q8k(x);
    const float q8k_dot = dot_q4_k_row_q8k_scalar(row, k8.q, k8.scales, k8.sums);
    const float rel = ::fabsf(f32_dot - q8k_dot) / ::fmaxf(::fabsf(f32_dot), 1e-3f);
    EXPECT_LT(rel, 0.03f) << "Q8_K mismatch: f32=" << f32_dot << " q8k=" << q8k_dot;

    // The per-256 format must stay in the accuracy class of the accepted per-32 W4A8 path.
    const auto p = quantize_row_to_i8_blocks(x);
    const auto psum = i8_block_sums(p.q);
    const float w4a8 = dot_q4_k_row_q8_scalar(row, p.q, p.scales, psum);
    const float rel_vs = ::fabsf(w4a8 - q8k_dot) / ::fmaxf(::fabsf(w4a8), 1e-3f);
    EXPECT_LT(rel_vs, 0.03f) << "Q8_K vs W4A8 divergence: w4a8=" << w4a8 << " q8k=" << q8k_dot;

#if defined(__aarch64__) || defined(_M_ARM64)
    if (has_dotprod()) {
        const float neon = dot_q4_k_row_q8k_neon(row, k8.q, k8.scales, k8.sums);
        EXPECT_EQ(bits(neon), bits(q8k_dot)) << "NEON≠scalar Q8_K: " << neon << " vs " << q8k_dot;
    }
#endif
}

#if defined(__aarch64__) || defined(_M_ARM64)
TEST(Quant, q4_k_r4_q8k_kernels_match_single_row) {
    // Rust runs the dotprod kernels unconditionally here; the port skips on a non-dotprod host.
    if (!has_dotprod()) GTEST_SKIP() << "dotprod not available";
    const size_t n = 4, k = 512;
    const size_t row_bytes = k / QK_K * Q4_K_BLOCK_BYTES;
    const auto rows = q4_k_mod_rows(n, k, 149, 29, 249, 0.04f, 0.02f);
    const auto x0 = ramp(k, 37, 97, 48.0f, 0.02f);
    const auto x1 = ramp(k, 59, 101, 50.0f, 0.015f);
    const auto r0 = quantize_row_to_q8k(x0);
    const auto r1 = quantize_row_to_q8k(x1);
    const auto packed = repack_q4_k_rows4(rows, n, k);

    // 4-row R4 kernel vs single-row Q8_K kernel, exact bits.
    const auto got4 = dot_q4_k_4rows_r4_q8k_neon(packed, r0.q, r0.scales, r0.sums);
    for (size_t r = 0; r < 4; ++r) {
        const float want =
            dot_q4_k_row_q8k_neon(row_of(rows, r, row_bytes), r0.q, r0.scales, r0.sums);
        EXPECT_EQ(bits(got4[r]), bits(want)) << "r4 row " << r << ": " << got4[r] << " vs " << want;
    }
    // SMMLA x2 kernel vs single-row, exact bits over both x rows.
    if (has_i8mm()) {
        const auto got = dot_q4_k_4rows_r4_x2_q8k_smmla(
            packed, r0.q, r0.scales, r0.sums, r1.q, r1.scales, r1.sums);
        for (size_t r = 0; r < 4; ++r) {
            const auto row = row_of(rows, r, row_bytes);
            const float w0 = dot_q4_k_row_q8k_neon(row, r0.q, r0.scales, r0.sums);
            const float w1 = dot_q4_k_row_q8k_neon(row, r1.q, r1.scales, r1.sums);
            EXPECT_EQ(bits(got[r][0]), bits(w0)) << "smmla row " << r << " x0";
            EXPECT_EQ(bits(got[r][1]), bits(w1)) << "smmla row " << r << " x1";
        }
    }
}

TEST(Quant, q4_k_plain_4rows_q8k_matches_single_row) {
    if (!has_dotprod()) GTEST_SKIP() << "dotprod not available"; // port-added skip (see Task 3)
    const size_t n = 4, k = 512;
    const size_t row_bytes = k / QK_K * Q4_K_BLOCK_BYTES;
    const auto rows = q4_k_mod_rows(n, k, 167, 43, 247, 0.04f, 0.02f);
    const auto x = ramp(k, 41, 103, 51.0f, 0.02f);
    const auto r = quantize_row_to_q8k(x);
    const std::array<std::span<const uint8_t>, 4> r4 = {row_of(rows, 0, row_bytes),
                                                        row_of(rows, 1, row_bytes),
                                                        row_of(rows, 2, row_bytes),
                                                        row_of(rows, 3, row_bytes)};
    const auto got = dot_q4_k_4rows_q8k_neon(r4, r.q, r.scales, r.sums);
    for (size_t o = 0; o < 4; ++o) {
        const float want = dot_q4_k_row_q8k_neon(r4[o], r.q, r.scales, r.sums);
        EXPECT_EQ(bits(got[o]), bits(want)) << "row " << o << ": " << got[o] << " vs " << want;
    }
}

// ── Q5_K, Q6_K f32 (Task 4) ───────────────────────────────────────────────────

namespace {
// Q6_K must map weight i to scale i/16 (16 scales per 256-weight super-block), matching ggml
// dequantize_row_q6_K. Every 6-bit quant decodes to +1 (raw 33 = low nibble 1 | hi bits 2 << 4)
// and scales = 0..16, so with x = 1 and d = 1 the dot is Σ_i scale[i/16] = 16·(0+…+15) = 1920.
std::vector<uint8_t> canonical_q6_k_block() {
    std::vector<uint8_t> block(Q6_K_BLOCK_BYTES, 0);
    for (size_t i = 0; i < 128; ++i)
        block[i] = 0x11; // every low nibble = 1
    for (size_t i = 128; i < 192; ++i)
        block[i] = 0xAA; // every 2-bit hi field = 0b10 = 2
    for (size_t j = 0; j < 16; ++j)
        block[192 + j] = static_cast<uint8_t>(j); // scales 0..15
    sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(1.0f), block.data() + 208);
    return block;
}
std::vector<uint8_t> rand_q6_k_block(uint64_t seed) {
    auto blk = lcg_bytes(seed, Q6_K_BLOCK_BYTES);
    sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(0.04f), blk.data() + 208);
    return blk;
}
std::vector<uint8_t> rand_q5_k_block(uint64_t seed) {
    auto blk = lcg_bytes(seed, Q5_K_BLOCK_BYTES);
    sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(0.05f), blk.data());
    sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(0.02f), blk.data() + 2);
    return blk;
}
float q6_scale_of(const uint8_t* sc, size_t i) {
    return static_cast<float>(detail::i8v(sc[i]));
}
// Buggy Q6_K dot: one scale per 32-element sub-group (the shipped bug — sc[ib..ib+4], ib += 4 per
// 128-block), which only ever touches scales 0..7.
float dot_q6_k_buggy(std::span<const uint8_t> row_data, std::span<const float> x) {
    float acc = 0.0f;
    size_t x_off = 0;
    const size_t nb = row_data.size() / Q6_K_BLOCK_BYTES;
    for (size_t bi = 0; bi < nb; ++bi) {
        const uint8_t* block = row_data.data() + bi * Q6_K_BLOCK_BYTES;
        const uint8_t* ql = block;
        const uint8_t* qh = block + 128;
        const uint8_t* sc = block + 192;
        const float d = sapient::core::f16_le_to_f32(block + 208);
        size_t ql_off = 0, qh_off = 0, ib = 0;
        for (size_t half = 0; half < QK_K / 128; ++half) {
            for (size_t l = 0; l < 32; ++l) {
                const float q1 = static_cast<float>(
                    static_cast<int32_t>((ql[ql_off + l] & 0x0F) | ((qh[qh_off + l] & 3) << 4)) -
                    32);
                const float q2 =
                    static_cast<float>(static_cast<int32_t>((ql[ql_off + l + 32] & 0x0F) |
                                                            (((qh[qh_off + l] >> 2) & 3) << 4)) -
                                       32);
                const float q3 =
                    static_cast<float>(static_cast<int32_t>((ql[ql_off + l] >> 4) |
                                                            (((qh[qh_off + l] >> 4) & 3) << 4)) -
                                       32);
                const float q4 =
                    static_cast<float>(static_cast<int32_t>((ql[ql_off + l + 32] >> 4) |
                                                            (((qh[qh_off + l] >> 6) & 3) << 4)) -
                                       32);
                acc += d * q6_scale_of(sc, ib) * q1 * x[x_off + l];
                acc += d * q6_scale_of(sc, ib + 1) * q2 * x[x_off + l + 32];
                acc += d * q6_scale_of(sc, ib + 2) * q3 * x[x_off + l + 64];
                acc += d * q6_scale_of(sc, ib + 3) * q4 * x[x_off + l + 96];
            }
            x_off += 128;
            ql_off += 64;
            qh_off += 32;
            ib += 4;
        }
    }
    return acc;
}
// Buggy Q5_K dot: the 5th bit read from a single qh[is/8] byte per 32-element sub-block (the
// shipped bug) instead of the per-element qh[l].
float dot_q5_k_buggy(std::span<const uint8_t> row_data, std::span<const float> x) {
    float acc = 0.0f;
    size_t x_off = 0;
    const size_t nb = row_data.size() / Q5_K_BLOCK_BYTES;
    for (size_t bi = 0; bi < nb; ++bi) {
        const uint8_t* block = row_data.data() + bi * Q5_K_BLOCK_BYTES;
        const float d = sapient::core::f16_le_to_f32(block);
        const float dmin = sapient::core::f16_le_to_f32(block + 2);
        const uint8_t* scales = block + 4;
        const uint8_t* qh = block + 16;
        const uint8_t* ql = block + 48;
        size_t ql_off = 0, is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (size_t g = 0; g < QK_K / 64; ++g) {
            const auto [sc1, m1] = sapient::core::dequant::get_scale_min_k4(is, scales);
            const float d1 = d * static_cast<float>(sc1), m1v = dmin * static_cast<float>(m1);
            const auto [sc2, m2] = sapient::core::dequant::get_scale_min_k4(is + 1, scales);
            const float d2 = d * static_cast<float>(sc2), m2v = dmin * static_cast<float>(m2);
            const uint8_t qh_byte = qh[is / 8]; // BUG: one byte for all 32 elements
            for (size_t l = 0; l < 32; ++l) {
                const float hi1 = (qh_byte & u1) != 0 ? 16.0f : 0.0f;
                const float hi2 = (qh_byte & u2) != 0 ? 16.0f : 0.0f;
                acc +=
                    (d1 * (static_cast<float>(ql[ql_off + l] & 0x0F) + hi1) - m1v) * x[x_off + l];
                acc += (d2 * (static_cast<float>(ql[ql_off + l] >> 4) + hi2) - m2v) *
                       x[x_off + l + 32];
            }
            x_off += 64;
            ql_off += 32;
            is += 2;
            if (is % 8 == 0) {
                u1 = 1;
                u2 = 2;
            } else {
                u1 = static_cast<uint8_t>(u1 << 2);
                u2 = static_cast<uint8_t>(u2 << 2);
            }
        }
    }
    return acc;
}
float rel_err(float got, float reference) {
    return ::fabsf(got - reference) / ::fmaxf(::fabsf(reference), 1e-6f);
}
struct Stats {
    float mean, median, max;
};
Stats stats(std::vector<float>& v) {
    std::sort(v.begin(), v.end());
    float sum = -0.0f; // iter().sum::<f32>()
    for (float x : v)
        sum += x;
    return {sum / static_cast<float>(v.size()), v[v.size() / 2], v.back()};
}
// quant.rs q6_k_test_rows: random bytes with a small positive f16 d at [208..210).
std::vector<uint8_t> q6_k_test_rows(size_t n, size_t k, uint64_t seed) {
    const size_t row_bytes = k / 256 * Q6_K_BLOCK_BYTES;
    std::vector<uint8_t> rows(n * row_bytes);
    for (size_t i = 0; i < rows.size(); ++i) {
        switch (i % Q6_K_BLOCK_BYTES) {
        case 208:
            rows[i] = 0x11;
            break;
        case 209:
            rows[i] = 0x2c;
            break;
        default:
            rows[i] = static_cast<uint8_t>(lcg_step(seed) >> 33);
        }
    }
    return rows;
}
} // namespace

TEST(Quant, q6_k_scale_indexing_matches_ggml) {
    const auto block = canonical_q6_k_block();
    const std::vector<float> x(QK_K, 1.0f);
    const float got = dot_q6_k_row_f32(block, x);
    EXPECT_LT(::fabsf(got - 1920.0f), 1e-3f)
        << "Q6_K scale indexing wrong: got " << got << ", expected 1920 (old buggy code gives 896)";
}

// Corruption-magnitude benchmark (differential-verification methodology): each reconstruction of
// a historical silent-correctness bug is self-validated (Q6_K must reproduce the documented 896
// on the canonical block) before its error distribution is printed. Assertions: the two
// reconstruction fidelities; the rest is a report (run with --gtest_also_run_disabled_tests is
// not needed — it always runs, like `cargo test -- --nocapture`).
TEST(Quant, corruption_magnitude_report) {
    const auto canon = canonical_q6_k_block();
    const std::vector<float> xo(QK_K, 1.0f);
    const float buggy_canon = dot_q6_k_buggy(canon, xo);
    ASSERT_LT(::fabsf(buggy_canon - 896.0f), 1e-3f)
        << "Q6_K bug reconstruction infidelity: got " << buggy_canon << ", expected documented 896";
    const float correct_canon = dot_q6_k_row_f32(canon, xo);
    std::printf(
        "\n=== Corruption-magnitude benchmark (relative error vs verified reference) ===\n");
    std::printf("[validate] Q6_K canonical block: correct=%g buggy=%g rel_err=%.4f\n",
                correct_canon,
                buggy_canon,
                rel_err(buggy_canon, correct_canon));

    const size_t nblk = 256;
    std::vector<float> q6(nblk);
    for (size_t i = 0; i < nblk; ++i) {
        const auto blk = rand_q6_k_block(0xC0DE0000ULL + i);
        const auto x = rand_x(0xBEEF0000ULL + i, QK_K);
        q6[i] = rel_err(dot_q6_k_buggy(blk, x), dot_q6_k_row_f32(blk, x));
    }
    const Stats s6 = stats(q6);
    std::printf("Q6_K scale mis-index   (n=%zu): mean=%.3f median=%.3f max=%.3f\n",
                nblk,
                s6.mean,
                s6.median,
                s6.max);

    std::vector<float> q5(nblk);
    for (size_t i = 0; i < nblk; ++i) {
        const auto blk = rand_q5_k_block(0x5A5A0000ULL + i);
        const auto x = rand_x(0x13570000ULL + i, QK_K);
        q5[i] = rel_err(dot_q5_k_buggy(blk, x), dot_q5_k_row_f32(blk, x));
    }
    const Stats s5 = stats(q5);
    std::printf("Q5_K 5th-bit mis-index (n=%zu): mean=%.3f median=%.3f max=%.3f\n",
                nblk,
                s5.mean,
                s5.median,
                s5.max);

#if defined(__aarch64__) || defined(_M_ARM64)
    if (has_dotprod()) {
        const size_t k = 4096;
        const auto wf = rand_x(0xAAAA, k);
        const auto w_blocks = q8_0_weight_row(wf);
        std::printf("Activation quant (Q8_0 W8A8, K=%zu):  outlier   per-block   per-row\n", k);
        for (const float mag : {1.0f, 5.0f, 10.0f, 20.0f, 40.0f, 80.0f}) {
            auto xf = rand_x(0xBBBB, k);
            xf[k / 2] = mag; // single outlier channel
            const float reference = dot_q8_0_row_f32(w_blocks, xf);
            const auto xq = quantize_row_to_i8_blocks(xf);
            const float block = dot_q8_0_row_sdot(w_blocks, xq.q, xq.scales);
            float max_abs = 0.0f;
            for (float v : xf)
                max_abs = ::fmaxf(max_abs, ::fabsf(v));
            const float rs = max_abs / 127.0f;
            const float inv = 1.0f / rs;
            std::vector<int8_t> x_row(k);
            for (size_t i = 0; i < k; ++i)
                x_row[i] = detail::round_clamp_i8(xf[i] * inv);
            const std::vector<float> perrow_sc(k / QK, rs);
            const float perrow = dot_q8_0_row_sdot(w_blocks, x_row, perrow_sc);
            std::printf("  %5.0fx outlier:                %10.4f %10.4f\n",
                        static_cast<double>(mag),
                        static_cast<double>(rel_err(block, reference)),
                        static_cast<double>(rel_err(perrow, reference)));
        }
    }
#endif
    std::printf("===========================================================================\n\n");
}

TEST(Quant, q6_k_neon_matches_scalar) {
    // The vectorised Q6_K dot must equal the scalar reference (same f32 math, only reduction
    // order differs). A bit-layout/scale bug here = token-salad.
    uint64_t seed = 0x51EDC0DEULL;
    auto next = [&seed]() { return static_cast<uint32_t>(lcg_step(seed) >> 33); };
    const size_t nblocks = 3;
    std::vector<uint8_t> row(nblocks * Q6_K_BLOCK_BYTES);
    for (uint8_t& b : row)
        b = static_cast<uint8_t>(next() & 0xFF);
    for (size_t blk = 0; blk < nblocks; ++blk)
        sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(0.04f),
                                 row.data() + blk * Q6_K_BLOCK_BYTES + 208);
    std::vector<float> x(nblocks * QK_K);
    for (float& v : x)
        v = (static_cast<float>(next()) /
             static_cast<float>(std::numeric_limits<uint32_t>::max())) *
                3.0f -
            1.5f;
    const float scalar = detail::dot_q6_k_row_f32_scalar(row, x);
    const float got = dot_q6_k_row_f32(row, x); // dispatches to NEON on aarch64
    const float rel = ::fabsf(got - scalar) / ::fmaxf(::fabsf(scalar), 1e-3f);
    EXPECT_LT(rel, 1e-4f) << "Q6_K NEON≠scalar: neon=" << got << " scalar=" << scalar;
}

TEST(Quant, q5_k_neon_matches_scalar) {
    uint64_t seed = 0xA5A51234ULL;
    auto next = [&seed]() { return static_cast<uint32_t>(lcg_step(seed) >> 33); };
    const size_t nblocks = 3;
    std::vector<uint8_t> row(nblocks * Q5_K_BLOCK_BYTES);
    for (uint8_t& b : row)
        b = static_cast<uint8_t>(next() & 0xFF);
    for (size_t blk = 0; blk < nblocks; ++blk) {
        uint8_t* base = row.data() + blk * Q5_K_BLOCK_BYTES;
        sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(0.05f), base);
        sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(0.02f), base + 2);
    }
    std::vector<float> x(nblocks * QK_K);
    for (float& v : x)
        v = (static_cast<float>(next()) /
             static_cast<float>(std::numeric_limits<uint32_t>::max())) *
                3.0f -
            1.5f;
    const float scalar = detail::dot_q5_k_row_f32_scalar(row, x);
    const float got = dot_q5_k_row_f32(row, x);
    const float rel = ::fabsf(got - scalar) / ::fmaxf(::fabsf(scalar), 1e-3f);
    EXPECT_LT(rel, 1e-4f) << "Q5_K NEON≠scalar: neon=" << got << " scalar=" << scalar;
}

TEST(Quant, q6_k_r4_repack_roundtrips_through_dequant) {
    using sapient::core::DType;
    using sapient::core::Shape;
    using sapient::core::Tensor;
    const size_t n = 8, k = 512;
    const auto blocks = q6_k_test_rows(n, k, 0x6B6BULL);
    auto orig = Tensor::from_quant_bytes(blocks, Shape{n, k}, DType::Q6_K);
    ASSERT_TRUE(orig.has_value()) << orig.error().to_string();
    const auto packed = repack_q6_k_rows4(blocks, n, k);
    auto r4 = Tensor::from_quant_bytes(packed, Shape{n, k}, DType::Q6_K_R4);
    ASSERT_TRUE(r4.has_value()) << r4.error().to_string();
    EXPECT_EQ(orig->to_f32_vec(), r4->to_f32_vec());
}

#if defined(__aarch64__) || defined(_M_ARM64)
TEST(Quant, q6_k_r4_kernel_matches_single_row) {
    const size_t n = 4, k = 512;
    const size_t row_bytes = k / 256 * Q6_K_BLOCK_BYTES;
    const auto blocks = q6_k_test_rows(n, k, 0x6666ULL);
    const auto x = ramp(k, 41, 83, 41.0f, 0.02f);
    const auto packed = repack_q6_k_rows4(blocks, n, k);
    const auto got = dot_q6_k_4rows_r4_neon(packed, x);
    for (size_t r = 0; r < 4; ++r) {
        const float want = detail::dot_q6_k_row_f32_neon(row_of(blocks, r, row_bytes), x);
        EXPECT_EQ(bits(got[r]), bits(want)) << "row " << r << ": " << got[r] << " vs " << want;
    }
}
#endif

TEST(QuantDeath, repack_q6_k_rows4_asserts_like_rust) {
    const std::vector<uint8_t> two_rows(2 * Q6_K_BLOCK_BYTES, 0);
    EXPECT_DEATH((void)repack_q6_k_rows4(two_rows, 2, 256), "multiple of 4");
}

// ── Q6_K W6A8 / Q8_K / SMMLA (Task 5) ─────────────────────────────────────────

#if defined(__aarch64__) || defined(_M_ARM64)
TEST(Quant, q6_k_w6a8_neon_matches_scalar) {
    if (!has_dotprod()) GTEST_SKIP() << "dotprod not available";
    const size_t k = 512;
    const auto rows = q6_k_test_rows(2, k, 0x0666ULL);
    const size_t row_bytes = k / 256 * Q6_K_BLOCK_BYTES;
    const auto x = ramp(k, 29, 71, 35.0f, 0.03f);
    const auto xq = quantize_row_to_i8_blocks(x);
    for (size_t r = 0; r < 2; ++r) {
        const auto row = row_of(rows, r, row_bytes);
        const float want = dot_q6_k_row_q8_scalar(row, xq.q, xq.scales);
        const float got = dot_q6_k_row_q8_neon(row, xq.q, xq.scales);
        EXPECT_EQ(bits(got), bits(want)) << "row " << r << ": " << got << " vs " << want;
    }
}

TEST(Quant, q6_k_w6a8_r4_matches_single_row) {
    if (!has_dotprod()) GTEST_SKIP() << "dotprod not available";
    const size_t n = 4, k = 512;
    const auto rows = q6_k_test_rows(n, k, 0x0667ULL);
    const size_t row_bytes = k / 256 * Q6_K_BLOCK_BYTES;
    const auto x = ramp(k, 31, 67, 33.0f, 0.03f);
    const auto xq = quantize_row_to_i8_blocks(x);
    const auto packed = repack_q6_k_rows4(rows, n, k);
    const auto got = dot_q6_k_4rows_r4_q8_neon(packed, xq.q, xq.scales);
    for (size_t r = 0; r < 4; ++r) {
        const float want = dot_q6_k_row_q8_neon(row_of(rows, r, row_bytes), xq.q, xq.scales);
        EXPECT_EQ(bits(got[r]), bits(want)) << "row " << r << ": " << got[r] << " vs " << want;
    }
}
#endif

TEST(Quant, q6_k_w6a8_close_to_f32_path) {
    // Activation quantization is per-32-block int8 — same accuracy class as the accepted Q4_K
    // W4A8 path. Bound the relative error vs the exact f32-activation dot.
    const size_t k = 512;
    const auto rows = q6_k_test_rows(1, k, 0x0668ULL);
    const auto x = ramp(k, 43, 91, 45.0f, 0.02f);
    const auto xq = quantize_row_to_i8_blocks(x);
    const float exact = dot_q6_k_row_f32(rows, x);
    const float w6a8 = dot_q6_k_row_q8_scalar(rows, xq.q, xq.scales);
    const float rel = ::fabsf(w6a8 - exact) / ::fmaxf(::fabsf(exact), 1e-3f);
    EXPECT_LT(rel, 4e-2f) << "W6A8 vs f32: " << w6a8 << " vs " << exact << " (rel " << rel << ")";
}

#if defined(__aarch64__) || defined(_M_ARM64)
TEST(Quant, q6_k_smmla_x2_matches_single_row) {
    if (!has_i8mm()) GTEST_SKIP() << "i8mm not available";
    const size_t n = 4, k = 512;
    const auto rows = q6_k_test_rows(n, k, 0x68AAULL);
    const size_t row_bytes = k / 256 * Q6_K_BLOCK_BYTES;
    const auto x0 = ramp(k, 37, 97, 48.0f, 0.02f);
    const auto x1 = ramp(k, 61, 103, 51.0f, 0.015f);
    const auto q0 = quantize_row_to_i8_blocks(x0);
    const auto q1 = quantize_row_to_i8_blocks(x1);
    const auto packed = repack_q6_k_rows4(rows, n, k);
    const auto got = dot_q6_k_4rows_r4_x2_smmla(packed, q0.q, q0.scales, q1.q, q1.scales);
    for (size_t r = 0; r < 4; ++r) {
        const auto row = row_of(rows, r, row_bytes);
        const float w0 = dot_q6_k_row_q8_neon(row, q0.q, q0.scales);
        const float w1 = dot_q6_k_row_q8_neon(row, q1.q, q1.scales);
        EXPECT_EQ(bits(got[r][0]), bits(w0)) << "row " << r << " x0: " << got[r][0] << " vs " << w0;
        EXPECT_EQ(bits(got[r][1]), bits(w1)) << "row " << r << " x1: " << got[r][1] << " vs " << w1;
    }
}
#endif

TEST(Quant, q6_k_q8k_scalar_matches_f32_path) {
    const size_t nblocks = 2;
    std::vector<uint8_t> row(nblocks * Q6_K_BLOCK_BYTES);
    for (size_t i = 0; i < row.size(); ++i)
        row[i] = static_cast<uint8_t>((i * 181 + 17) % 251);
    for (size_t blk = 0; blk < nblocks; ++blk)
        sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(0.05f),
                                 row.data() + blk * Q6_K_BLOCK_BYTES + 208);
    const auto x = ramp(nblocks * QK_K, 43, 107, 53.0f, 0.02f);
    const float f32_dot = dot_q6_k_row_f32(row, x);
    const auto k8 = quantize_row_to_q8k(x);
    const float q8k = dot_q6_k_row_q8k_scalar(row, k8.q, k8.scales);
    const float rel = ::fabsf(f32_dot - q8k) / ::fmaxf(::fabsf(f32_dot), 1e-3f);
    EXPECT_LT(rel, 0.03f) << "Q6_K Q8_K vs f32: " << f32_dot << " vs " << q8k << " rel=" << rel;

    // Accuracy class vs the accepted per-32 W6A8 path.
    const auto p = quantize_row_to_i8_blocks(x);
    const float w6a8 = dot_q6_k_row_q8_scalar(row, p.q, p.scales);
    const float rel_vs = ::fabsf(w6a8 - q8k) / ::fmaxf(::fabsf(w6a8), 1e-3f);
    EXPECT_LT(rel_vs, 0.03f) << "Q6_K Q8_K vs W6A8: " << w6a8 << " vs " << q8k << " rel=" << rel_vs;

#if defined(__aarch64__) || defined(_M_ARM64)
    if (has_dotprod()) {
        const float neon = dot_q6_k_row_q8k_neon(row, k8.q, k8.scales);
        EXPECT_EQ(bits(neon), bits(q8k)) << "NEON≠scalar: " << neon << " vs " << q8k;
    }
#endif
}

#if defined(__aarch64__) || defined(_M_ARM64)
TEST(Quant, q6_k_r4_q8k_kernels_match_single_row) {
    if (!has_dotprod()) GTEST_SKIP() << "dotprod not available";
    const size_t n = 4, k = 512;
    const size_t row_bytes = k / QK_K * Q6_K_BLOCK_BYTES;
    std::vector<uint8_t> rows(n * row_bytes);
    for (size_t i = 0; i < rows.size(); ++i)
        rows[i] = static_cast<uint8_t>((i * 157 + 31) % 253);
    for (size_t r = 0; r < n; ++r)
        for (size_t blk = 0; blk < k / QK_K; ++blk)
            sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(0.04f),
                                     rows.data() + r * row_bytes + blk * Q6_K_BLOCK_BYTES + 208);
    const auto x0 = ramp(k, 37, 97, 48.0f, 0.02f);
    const auto x1 = ramp(k, 61, 89, 44.0f, 0.015f);
    const auto r0 = quantize_row_to_q8k(x0);
    const auto r1 = quantize_row_to_q8k(x1);
    const auto packed = repack_q6_k_rows4(rows, n, k);

    const auto got4 = dot_q6_k_4rows_r4_q8k_neon(packed, r0.q, r0.scales);
    for (size_t r = 0; r < 4; ++r) {
        const float want = dot_q6_k_row_q8k_neon(row_of(rows, r, row_bytes), r0.q, r0.scales);
        EXPECT_EQ(bits(got4[r]), bits(want)) << "r4 row " << r << ": " << got4[r] << " vs " << want;
    }
    if (has_i8mm()) {
        const auto got = dot_q6_k_4rows_r4_x2_q8k_smmla(packed, r0.q, r0.scales, r1.q, r1.scales);
        for (size_t r = 0; r < 4; ++r) {
            const auto row = row_of(rows, r, row_bytes);
            const float w0 = dot_q6_k_row_q8k_neon(row, r0.q, r0.scales);
            const float w1 = dot_q6_k_row_q8k_neon(row, r1.q, r1.scales);
            EXPECT_EQ(bits(got[r][0]), bits(w0)) << "smmla row " << r << " x0";
            EXPECT_EQ(bits(got[r][1]), bits(w1)) << "smmla row " << r << " x1";
        }
    }
}
#endif

#endif
