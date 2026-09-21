// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/backends_cpu/kernels/quant.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#if defined(__aarch64__) || defined(_M_ARM64)
#define SAPIENT_AARCH64 1
#include <arm_neon.h>
// Clang's target-guarded NEON intrinsics: a dotprod/i8mm intrinsic may only be used inside a
// function compiled with that feature. Every such kernel carries one of these (Rust:
// `#[target_feature(enable = "neon,dotprod")]` / `"neon,i8mm"`). i8mm does NOT imply dotprod.
#define SAPIENT_TARGET_DOTPROD __attribute__((target("dotprod")))
#define SAPIENT_TARGET_I8MM __attribute__((target("i8mm")))
#elif defined(__x86_64__) || defined(_M_X64)
#define SAPIENT_X86_64 1
#include <immintrin.h>
#endif

#include "sapient/backends_cpu/cpu_features.hpp"
#include "sapient/core/dequant.hpp"
#include "sapient/core/f16.hpp"
#include "sapient/core/panic.hpp"

namespace sapient::backends_cpu::kernels::quant {

using sapient::core::f16_le_to_f32;
using sapient::core::panic;

namespace detail {

int32_t f32_to_i32_sat(float v) {
    // Rust `as i32`: NaN → 0, ±inf and out-of-range saturate, otherwise truncate toward zero.
    if (std::isnan(v)) return 0;
    if (v >= 2147483648.0f) return std::numeric_limits<int32_t>::max();
    if (v <= -2147483648.0f) return std::numeric_limits<int32_t>::min();
    return static_cast<int32_t>(v);
}

int8_t round_clamp_i8(float v) {
    const float r = std::clamp(::roundf(v), -127.0f, 127.0f); // clamp lets NaN through, like Rust
    if (std::isnan(r)) return 0;                              // Rust `NaN as i8` == 0
    return static_cast<int8_t>(r);
}

uint8_t nibble(float scaled) {
    // ggml: MIN(15, (int)(x*id + 8.5)). Clamp into [0, 15].
    return static_cast<uint8_t>(std::clamp(f32_to_i32_sat(scaled + 8.5f), 0, 15));
}

int32_t i8v(uint8_t b) {
    return static_cast<int32_t>(static_cast<int8_t>(b)); // NOLINT(bugprone-signed-char-misuse)
}

} // namespace detail

namespace {

void check_block(std::span<const uint8_t> block, size_t bytes, const char* who) {
    if (block.size() != bytes) panic(who);
}
void check_len(size_t have, size_t need, const char* who) {
    if (have < need) panic(who);
}

// ── Q4_0 ─────────────────────────────────────────────────────────────────────

#if SAPIENT_AARCH64
// NEON Q4_0 block dot product: 16 packed bytes → lo nibbles (elements 0..15) and hi nibbles
// (16..31), minus 8, widened i8 → i16 → i32 → f32, FMA with the activations (quant.rs:123-175).
float dot_q4_0_block_neon(const uint8_t* block, const float* x) {
    const float scale = f16_le_to_f32(block);
    const uint8x16_t packed = vld1q_u8(block + 2);
    const uint8x16_t lo_u8 = vandq_u8(packed, vdupq_n_u8(0x0F));
    const uint8x16_t hi_u8 = vshrq_n_u8(packed, 4);
    const uint8x16_t eight = vdupq_n_u8(8);
    const int8x16_t lo_i8 = vreinterpretq_s8_u8(vsubq_u8(lo_u8, eight));
    const int8x16_t hi_i8 = vreinterpretq_s8_u8(vsubq_u8(hi_u8, eight));

    // to_f32x4!(v, vmovl_s8_low / vmovl_s8_high)
    const int16x8_t lo16a = vmovl_s8(vget_low_s8(lo_i8));
    const float32x4_t lo_f32_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16a)));
    const float32x4_t lo_f32_1 = vcvtq_f32_s32(vmovl_high_s16(lo16a));
    const int16x8_t lo16b = vmovl_high_s8(lo_i8);
    const float32x4_t lo_f32_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16b)));
    const float32x4_t lo_f32_3 = vcvtq_f32_s32(vmovl_high_s16(lo16b));
    const int16x8_t hi16a = vmovl_s8(vget_low_s8(hi_i8));
    const float32x4_t hi_f32_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16a)));
    const float32x4_t hi_f32_1 = vcvtq_f32_s32(vmovl_high_s16(hi16a));
    const int16x8_t hi16b = vmovl_high_s8(hi_i8);
    const float32x4_t hi_f32_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16b)));
    const float32x4_t hi_f32_3 = vcvtq_f32_s32(vmovl_high_s16(hi16b));

    const float32x4_t x0 = vld1q_f32(x);
    const float32x4_t x1 = vld1q_f32(x + 4);
    const float32x4_t x2 = vld1q_f32(x + 8);
    const float32x4_t x3 = vld1q_f32(x + 12);
    const float32x4_t x4 = vld1q_f32(x + 16);
    const float32x4_t x5 = vld1q_f32(x + 20);
    const float32x4_t x6 = vld1q_f32(x + 24);
    const float32x4_t x7 = vld1q_f32(x + 28);

    float32x4_t acc = vmulq_f32(lo_f32_0, x0);
    acc = vfmaq_f32(acc, lo_f32_1, x1);
    acc = vfmaq_f32(acc, lo_f32_2, x2);
    acc = vfmaq_f32(acc, lo_f32_3, x3);
    acc = vfmaq_f32(acc, hi_f32_0, x4);
    acc = vfmaq_f32(acc, hi_f32_1, x5);
    acc = vfmaq_f32(acc, hi_f32_2, x6);
    acc = vfmaq_f32(acc, hi_f32_3, x7);
    return vaddvq_f32(acc) * scale;
}

// NEON Q8_0 block dot: four groups of 8 i8 widened to f32, two vfmaq per group (quant.rs:266-290).
float dot_q8_0_block_neon(const uint8_t* block, const float* x) {
    const float scale = f16_le_to_f32(block);
    const int8_t* q_ptr = reinterpret_cast<const int8_t*>(block + 2);
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (size_t off = 0; off < QK; off += 8) { // fma_group!(0,0) (8,8) (16,16) (24,24)
        const int8x8_t q8 = vld1_s8(q_ptr + off);
        const int16x8_t q16 = vmovl_s8(q8);
        const float32x4_t qlo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16)));
        const float32x4_t qhi = vcvtq_f32_s32(vmovl_high_s16(q16));
        acc = vfmaq_f32(acc, qlo, vld1q_f32(x + off));
        acc = vfmaq_f32(acc, qhi, vld1q_f32(x + off + 4));
    }
    return vaddvq_f32(acc) * scale;
}
#endif

} // namespace

std::array<uint8_t, Q4_0_BLOCK_BYTES> quantize_q4_0_block(std::span<const float> x) {
    if (x.size() != QK) panic("quantize_q4_0_block: x must have 32 elements");
    // Scale from the value with the largest magnitude, preserving its sign (ggml derives `d`
    // this way, which is why d can be negative).
    float amax = 0.0f;
    float vmax = 0.0f;
    for (const float v : x) {
        if (::fabsf(v) > amax) {
            amax = ::fabsf(v);
            vmax = v;
        }
    }
    const float d = vmax / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;

    std::array<uint8_t, Q4_0_BLOCK_BYTES> out{};
    sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(d), out.data());
    for (size_t j = 0; j < QK / 2; ++j) {
        const uint8_t q0 = detail::nibble(x[j] * id);
        const uint8_t q1 = detail::nibble(x[j + QK / 2] * id);
        out[2 + j] = static_cast<uint8_t>(q0 | (q1 << 4));
    }
    return out;
}

void dequantize_q4_0_block(std::span<const uint8_t> block, std::span<float> out) {
    check_block(block, Q4_0_BLOCK_BYTES, "dequantize_q4_0_block: block must be 18 bytes");
    if (out.size() != QK) panic("dequantize_q4_0_block: out must have 32 elements");
    sapient::core::dequant::q4_0_block(block.data(), out.data()); // same arithmetic, same order
}

float detail::dot_q4_0_block_scalar(std::span<const uint8_t> block, std::span<const float> x) {
    const float d = f16_le_to_f32(block.data());
    float acc = 0.0f;
    for (size_t j = 0; j < QK / 2; ++j) {
        const uint8_t byte = block[2 + j];
        const int32_t lo = static_cast<int32_t>(byte & 0x0F) - 8;
        const int32_t hi = static_cast<int32_t>(byte >> 4) - 8;
        acc += static_cast<float>(lo) * x[j] + static_cast<float>(hi) * x[j + QK / 2];
    }
    return acc * d;
}

float dot_q4_0_block_f32(std::span<const uint8_t> block, std::span<const float> x) {
    check_block(block, Q4_0_BLOCK_BYTES, "dot_q4_0_block_f32: block must be 18 bytes");
    if (x.size() != QK) panic("dot_q4_0_block_f32: x must have 32 elements");
#if SAPIENT_AARCH64
    return dot_q4_0_block_neon(block.data(), x.data());
#else
    return detail::dot_q4_0_block_scalar(block, x);
#endif
}

float dot_q4_0_row_f32(std::span<const uint8_t> row_blocks, std::span<const float> x) {
    const size_t k = x.size();
    if (k % QK != 0) panic("dot_q4_0_row_f32: k must be a multiple of 32");
    const size_t nb = row_blocks.size() / Q4_0_BLOCK_BYTES; // chunks_exact
    check_len(k, nb * QK, "dot_q4_0_row_f32: x shorter than the row");
    float acc = 0.0f;
    for (size_t b = 0; b < nb; ++b)
        acc += dot_q4_0_block_f32(row_blocks.subspan(b * Q4_0_BLOCK_BYTES, Q4_0_BLOCK_BYTES),
                                  x.subspan(b * QK, QK));
    return acc;
}

std::vector<uint8_t> quantize_q4_0_row(std::span<const float> w) {
    if (w.size() % QK != 0) panic("quantize_q4_0_row: length must be a multiple of 32");
    std::vector<uint8_t> out;
    out.reserve(w.size() / QK * Q4_0_BLOCK_BYTES);
    for (size_t b = 0; b + QK <= w.size(); b += QK) {
        const auto blk = quantize_q4_0_block(w.subspan(b, QK));
        out.insert(out.end(), blk.begin(), blk.end());
    }
    return out;
}

// ── Q8_0 ─────────────────────────────────────────────────────────────────────

std::array<uint8_t, Q8_0_BLOCK_BYTES> quantize_q8_0_block(std::span<const float> x) {
    if (x.size() != QK) panic("quantize_q8_0_block: x must have 32 elements");
    float max_abs = 0.0f; // .map(|v| v.abs()).fold(0.0, f32::max) — NaN-dropping
    for (const float v : x)
        max_abs = ::fmaxf(max_abs, ::fabsf(v));
    const float scale = max_abs / 127.0f;
    const uint16_t d = sapient::core::f32_to_f16_bits(scale);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    std::array<uint8_t, Q8_0_BLOCK_BYTES> out{};
    sapient::core::f16_to_le(d, out.data());
    for (size_t i = 0; i < QK; ++i)
        out[2 + i] = static_cast<uint8_t>(detail::round_clamp_i8(x[i] * inv_scale));
    return out;
}

float detail::dot_q8_0_block_scalar(std::span<const uint8_t> block, std::span<const float> x) {
    const float d = f16_le_to_f32(block.data());
    float acc = 0.0f;
    for (size_t j = 0; j < QK; ++j)
        acc += static_cast<float>(detail::i8v(block[2 + j])) * x[j];
    return acc * d;
}

float dot_q8_0_block_f32(std::span<const uint8_t> block, std::span<const float> x) {
    check_block(block, Q8_0_BLOCK_BYTES, "dot_q8_0_block_f32: block must be 34 bytes");
    if (x.size() != QK) panic("dot_q8_0_block_f32: x must have 32 elements");
#if SAPIENT_AARCH64
    return dot_q8_0_block_neon(block.data(), x.data());
#else
    return detail::dot_q8_0_block_scalar(block, x);
#endif
}

// ── activation quantisers (quant.rs:317-392) ─────────────────────────────────

I8Blocks quantize_row_to_i8_blocks(std::span<const float> x) {
    if (x.size() % QK != 0) panic("quantize_row_to_i8_blocks: length must be a multiple of 32");
    const size_t nblocks = x.size() / QK;
    I8Blocks r{std::vector<int8_t>(x.size(), 0), std::vector<float>(nblocks, 0.0f)};
    for (size_t b = 0; b < nblocks; ++b) {
        const auto blk = x.subspan(b * QK, QK);
        float max_abs = 0.0f;
        for (const float v : blk)
            max_abs = ::fmaxf(max_abs, ::fabsf(v));
        const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        for (size_t i = 0; i < QK; ++i)
            r.q[b * QK + i] = detail::round_clamp_i8(blk[i] * inv);
        r.scales[b] = scale;
    }
    return r;
}

std::vector<int32_t> i8_block_sums(std::span<const int8_t> q) {
    if (q.size() % QK != 0) panic("i8_block_sums: length must be a multiple of 32");
    std::vector<int32_t> sums(q.size() / QK, 0);
    for (size_t b = 0; b < sums.size(); ++b) {
        int32_t s = 0;
        for (size_t i = 0; i < QK; ++i)
            s += static_cast<int32_t>(q[b * QK + i]);
        sums[b] = s;
    }
    return sums;
}

Q8kRow quantize_row_to_q8k(std::span<const float> x) {
    if (x.size() % QK_K != 0) panic("quantize_row_to_q8k: length must be a multiple of 256");
    const size_t nsuper = x.size() / QK_K;
    Q8kRow r{std::vector<int8_t>(x.size(), 0),
             std::vector<float>(nsuper, 0.0f),
             std::vector<int32_t>(x.size() / QK, 0)};
    for (size_t b = 0; b < nsuper; ++b) {
        const auto blk = x.subspan(b * QK_K, QK_K);
        float max_abs = 0.0f;
        for (const float v : blk)
            max_abs = ::fmaxf(max_abs, ::fabsf(v));
        const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        for (size_t i = 0; i < QK_K; ++i)
            r.q[b * QK_K + i] = detail::round_clamp_i8(blk[i] * inv);
        r.scales[b] = scale;
        for (size_t j = 0; j < QK_K / QK; ++j) {
            const size_t base = b * QK_K + j * QK;
            int32_t s = 0;
            for (size_t i = 0; i < QK; ++i)
                s += static_cast<int32_t>(r.q[base + i]);
            r.sums[b * (QK_K / QK) + j] = s;
        }
    }
    return r;
}

// ── SDOT (ARMv8.4-A dotprod) Q8_0 (quant.rs:442-500) ─────────────────────────

#if SAPIENT_AARCH64
SAPIENT_TARGET_DOTPROD int32_t detail::dot_q8_0_block_sdot(std::span<const uint8_t> block,
                                                           std::span<const int8_t> x_i8) {
    check_block(block, Q8_0_BLOCK_BYTES, "dot_q8_0_block_sdot: block must be 34 bytes");
    if (x_i8.size() != QK) panic("dot_q8_0_block_sdot: x_i8 must have 32 elements");
    const int8_t* w_ptr = reinterpret_cast<const int8_t*>(block.data() + 2);
    const int8_t* x_ptr = x_i8.data();
    const int8x16_t w0 = vld1q_s8(w_ptr);
    const int8x16_t x0 = vld1q_s8(x_ptr);
    const int8x16_t w1 = vld1q_s8(w_ptr + 16);
    const int8x16_t x1 = vld1q_s8(x_ptr + 16);
    int32x4_t acc = vdupq_n_s32(0);
    acc = vdotq_s32(acc, w0, x0); // sdot v_acc.4s, v_w.16b, v_x.16b
    acc = vdotq_s32(acc, w1, x1);
    return vaddvq_s32(acc);
}

SAPIENT_TARGET_DOTPROD float dot_q8_0_row_sdot(std::span<const uint8_t> row_blocks,
                                               std::span<const int8_t> x_i8,
                                               std::span<const float> x_scales) {
    const size_t nb = row_blocks.size() / Q8_0_BLOCK_BYTES;
    check_len(x_i8.size(), nb * QK, "dot_q8_0_row_sdot: x_i8 shorter than the row");
    check_len(x_scales.size(), nb, "dot_q8_0_row_sdot: x_scales shorter than the row");
    float acc = 0.0f;
    size_t x_off = 0;
    for (size_t bi = 0; bi < nb; ++bi) {
        const auto block = row_blocks.subspan(bi * Q8_0_BLOCK_BYTES, Q8_0_BLOCK_BYTES);
        const float w_scale = f16_le_to_f32(block.data());
        const int32_t dot = detail::dot_q8_0_block_sdot(block, x_i8.subspan(x_off, QK));
        acc += w_scale * x_scales[bi] * static_cast<float>(dot);
        x_off += QK;
    }
    return acc;
}
#endif

float dot_q8_0_row_i8_scalar(std::span<const uint8_t> row_blocks,
                             std::span<const int8_t> x_i8,
                             std::span<const float> x_scales) {
    const size_t nb = row_blocks.size() / Q8_0_BLOCK_BYTES;
    check_len(x_i8.size(), nb * QK, "dot_q8_0_row_i8_scalar: x_i8 shorter than the row");
    check_len(x_scales.size(), nb, "dot_q8_0_row_i8_scalar: x_scales shorter than the row");
    float acc = 0.0f;
    size_t x_off = 0;
    for (size_t bi = 0; bi < nb; ++bi) {
        const uint8_t* block = row_blocks.data() + bi * Q8_0_BLOCK_BYTES;
        const float w_scale = f16_le_to_f32(block);
        int32_t dot = 0; // zip().map().sum::<i32>()
        for (size_t j = 0; j < QK; ++j)
            dot += detail::i8v(block[2 + j]) * static_cast<int32_t>(x_i8[x_off + j]);
        acc += w_scale * x_scales[bi] * static_cast<float>(dot);
        x_off += QK;
    }
    return acc;
}

// ── AVX2+FMA Q8_0 row dot — the ONE x86 SIMD kernel (quant.rs:543-580) ───────

#if SAPIENT_X86_64
__attribute__((target("avx2,fma"))) float
detail::dot_q8_0_row_avx2(std::span<const uint8_t> row_blocks, std::span<const float> x) {
    const size_t k = x.size();
    if (k % QK != 0) panic("dot_q8_0_row_avx2: k must be a multiple of 32");
    const size_t nb = row_blocks.size() / Q8_0_BLOCK_BYTES;
    check_len(k, nb * QK, "dot_q8_0_row_avx2: x shorter than the row");
    __m256 row_acc = _mm256_setzero_ps();
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* block = row_blocks.data() + b * Q8_0_BLOCK_BYTES;
        const float scale = f16_le_to_f32(block);
        const uint8_t* q_ptr = block + 2; // Rust: `*const i32`, stepped 4 bytes per unit
        const float* xp = x.data() + b * QK;
        __m256 block_acc = _mm256_setzero_ps();
        for (size_t g = 0; g < 4; ++g) {
            // 4 bytes each (zero upper lanes) → 8 i32 lanes of which 4 carry quants.
            const __m128i q_i32_4 = _mm_loadu_si32(q_ptr + 8 * g);
            const __m128i q_i32_4b = _mm_loadu_si32(q_ptr + 8 * g + 4);
            const __m256i q_a = _mm256_cvtepi8_epi32(q_i32_4);
            const __m256i q_b = _mm256_cvtepi8_epi32(q_i32_4b);
            const __m256 xv_a = _mm256_loadu_ps(xp + g * 8); // x[8g, 8g+8): in-bounds
            // Rust loads 8 floats from xp + g*8 + 4 here — at g == 3 that runs 4 floats past
            // the block (past the end of x on the row's last block). Those lanes only multiply
            // q_b's zero lanes, so an in-bounds, zero-extended load is bit-identical for finite x
            // (plan D ruling; recorded in docs/PARITY.md as a known oracle defect).
            const __m256 xv_b =
                _mm256_insertf128_ps(_mm256_setzero_ps(), _mm_loadu_ps(xp + g * 8 + 4), 0);
            const __m256 qf_a = _mm256_cvtepi32_ps(q_a);
            const __m256 qf_b = _mm256_cvtepi32_ps(q_b);
            block_acc = _mm256_fmadd_ps(qf_a, xv_a, block_acc);
            block_acc = _mm256_fmadd_ps(qf_b, xv_b, block_acc);
        }
        const __m256 scale_v = _mm256_set1_ps(scale);
        row_acc = _mm256_fmadd_ps(block_acc, scale_v, row_acc);
    }
    // Horizontal sum of the 8-lane accumulator — the exact sequence of quant.rs:572-579.
    const __m128 lo = _mm256_castps256_ps128(row_acc);
    const __m128 hi = _mm256_extractf128_ps(row_acc, 1);
    const __m128 sum4 = _mm_add_ps(lo, hi);
    const __m128 shuf = _mm_movehdup_ps(sum4);
    const __m128 sum2 = _mm_add_ps(sum4, shuf);
    const __m128 sum1 = _mm_add_ss(sum2, _mm_movehl_ps(shuf, sum2));
    return _mm_cvtss_f32(sum1);
}
#endif

float dot_q8_0_row_f32(std::span<const uint8_t> row_blocks, std::span<const float> x) {
#if SAPIENT_X86_64
    if (cpu_features::has_avx2_fma()) return detail::dot_q8_0_row_avx2(row_blocks, x);
#endif
    const size_t k = x.size();
    if (k % QK != 0) panic("dot_q8_0_row_f32: k must be a multiple of 32");
    const size_t nb = row_blocks.size() / Q8_0_BLOCK_BYTES;
    check_len(k, nb * QK, "dot_q8_0_row_f32: x shorter than the row");
    float acc = 0.0f;
    for (size_t b = 0; b < nb; ++b)
        acc += dot_q8_0_block_f32(row_blocks.subspan(b * Q8_0_BLOCK_BYTES, Q8_0_BLOCK_BYTES),
                                  x.subspan(b * QK, QK));
    return acc;
}

// ── K-quants: Q4_K (quant.rs:588-1330) ───────────────────────────────────────

namespace {

using sapient::core::dequant::get_scale_min_k4;

// (d, dmin) of a Q4_K/Q5_K super-block header.
std::pair<float, float> q4k_header(const uint8_t* block) {
    return {f16_le_to_f32(block), f16_le_to_f32(block + 2)};
}

// Entry checks for the activation-side spans of a K-quant row of `nb` super-blocks.
void check_q8_row(size_t nb,
                  std::span<const int8_t> x_i8,
                  std::span<const float> x_scales,
                  size_t scales_per_block,
                  const char* who) {
    check_len(x_i8.size(), nb * QK_K, who);
    check_len(x_scales.size(), nb * scales_per_block, who);
}

#if SAPIENT_AARCH64
// Accumulate four 4-element i8 dot products into an i32x4 — Rust's `sdot_s32` inline asm.
SAPIENT_TARGET_DOTPROD inline int32x4_t sdot_s32(int32x4_t acc, int8x16_t w, int8x16_t x) {
    return vdotq_s32(acc, w, x);
}
#endif

} // namespace

float detail::dot_q4_k_row_f32_scalar(std::span<const uint8_t> row_data, std::span<const float> x) {
    const size_t nb = row_data.size() / Q4_K_BLOCK_BYTES;
    check_len(x.size(), nb * QK_K, "dot_q4_k_row_f32: x shorter than the row");
    float acc = 0.0f;
    size_t x_off = 0;
    for (size_t bi = 0; bi < nb; ++bi) {
        const uint8_t* block = row_data.data() + bi * Q4_K_BLOCK_BYTES;
        const auto [d, dmin] = q4k_header(block);
        const uint8_t* scales = block + 4;
        const uint8_t* qs = block + 16;
        size_t q_off = 0;
        size_t is = 0;
        for (size_t g = 0; g < QK_K / 64; ++g) {
            const auto [sc1, m1] = get_scale_min_k4(is, scales);
            const float d1 = d * static_cast<float>(sc1);
            const float m1v = dmin * static_cast<float>(m1);
            const auto [sc2, m2] = get_scale_min_k4(is + 1, scales);
            const float d2 = d * static_cast<float>(sc2);
            const float m2v = dmin * static_cast<float>(m2);
            for (size_t l = 0; l < 32; ++l) {
                acc += (d1 * static_cast<float>(qs[q_off + l] & 0x0F) - m1v) * x[x_off + l];
                acc += (d2 * static_cast<float>(qs[q_off + l] >> 4) - m2v) * x[x_off + l + 32];
            }
            x_off += 64;
            q_off += 32;
            is += 2;
        }
    }
    return acc;
}

#if SAPIENT_AARCH64
// NEON Q4_K row dot: 8 packed bytes (16 nibbles) per iteration, FMA for the lo- and hi-nibble
// sub-blocks, plus the Σx vectors for the min correction (quant.rs:661-748).
float detail::dot_q4_k_row_f32_neon(std::span<const uint8_t> row_data, std::span<const float> x) {
    const size_t nb = row_data.size() / Q4_K_BLOCK_BYTES;
    check_len(x.size(), nb * QK_K, "dot_q4_k_row_f32: x shorter than the row");
    float acc = 0.0f;
    size_t x_off = 0;
    const uint8x8_t mask4 = vdup_n_u8(0x0F);
    for (size_t bi = 0; bi < nb; ++bi) {
        const uint8_t* block = row_data.data() + bi * Q4_K_BLOCK_BYTES;
        const auto [d, dmin] = q4k_header(block);
        const uint8_t* scales = block + 4;
        const uint8_t* qs = block + 16;
        size_t q_off = 0;
        size_t is = 0;
        for (size_t g = 0; g < QK_K / 64; ++g) {
            const auto [sc1, m1] = get_scale_min_k4(is, scales);
            const auto [sc2, m2] = get_scale_min_k4(is + 1, scales);
            const float d1 = d * static_cast<float>(sc1);
            const float m1v = dmin * static_cast<float>(m1);
            const float d2 = d * static_cast<float>(sc2);
            const float m2v = dmin * static_cast<float>(m2);
            const float* x_lo = x.data() + x_off;
            const float* x_hi = x.data() + x_off + 32;

            float32x4_t vsum_lo = vdupq_n_f32(0.0f); // dot(lo_nibbles, x_lo)
            float32x4_t vsum_hi = vdupq_n_f32(0.0f); // dot(hi_nibbles, x_hi)
            float32x4_t vsum_xl = vdupq_n_f32(0.0f); // sum(x_lo) for the min correction
            float32x4_t vsum_xh = vdupq_n_f32(0.0f); // sum(x_hi)
            for (size_t chunk = 0; chunk < 4; ++chunk) {
                const uint8x8_t q8 = vld1_u8(qs + q_off + chunk * 8);
                const uint8x8_t lo8 = vand_u8(q8, mask4);
                const uint8x8_t hi8 = vshr_n_u8(q8, 4);
                const uint16x8_t lo16 = vmovl_u8(lo8);
                const float32x4_t lof0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16)));
                const float32x4_t lof1 = vcvtq_f32_u32(vmovl_high_u16(lo16));
                const uint16x8_t hi16 = vmovl_u8(hi8);
                const float32x4_t hif0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16)));
                const float32x4_t hif1 = vcvtq_f32_u32(vmovl_high_u16(hi16));
                const float32x4_t xl0 = vld1q_f32(x_lo + chunk * 8);
                const float32x4_t xl1 = vld1q_f32(x_lo + chunk * 8 + 4);
                const float32x4_t xh0 = vld1q_f32(x_hi + chunk * 8);
                const float32x4_t xh1 = vld1q_f32(x_hi + chunk * 8 + 4);
                vsum_lo = vfmaq_f32(vsum_lo, lof0, xl0);
                vsum_lo = vfmaq_f32(vsum_lo, lof1, xl1);
                vsum_hi = vfmaq_f32(vsum_hi, hif0, xh0);
                vsum_hi = vfmaq_f32(vsum_hi, hif1, xh1);
                vsum_xl = vaddq_f32(vsum_xl, vaddq_f32(xl0, xl1));
                vsum_xh = vaddq_f32(vsum_xh, vaddq_f32(xh0, xh1));
            }
            acc += d1 * vaddvq_f32(vsum_lo) - m1v * vaddvq_f32(vsum_xl);
            acc += d2 * vaddvq_f32(vsum_hi) - m2v * vaddvq_f32(vsum_xh);
            x_off += 64;
            q_off += 32;
            is += 2;
        }
    }
    return acc;
}
#endif

float dot_q4_k_row_f32(std::span<const uint8_t> row_data, std::span<const float> x) {
#if SAPIENT_AARCH64
    return detail::dot_q4_k_row_f32_neon(row_data, x);
#else
    return detail::dot_q4_k_row_f32_scalar(row_data, x);
#endif
}

float dot_q4_k_row_q8_scalar(std::span<const uint8_t> row_data,
                             std::span<const int8_t> x_i8,
                             std::span<const float> x_scales,
                             std::span<const int32_t> x_sums) {
    const size_t nb = row_data.size() / Q4_K_BLOCK_BYTES;
    check_q8_row(
        nb, x_i8, x_scales, QK_K / QK, "dot_q4_k_row_q8_scalar: activations shorter than the row");
    check_len(
        x_sums.size(), nb * (QK_K / QK), "dot_q4_k_row_q8_scalar: x_sums shorter than the row");
    float acc = 0.0f;
    size_t x_off = 0;
    for (size_t bi = 0; bi < nb; ++bi) {
        const uint8_t* block = row_data.data() + bi * Q4_K_BLOCK_BYTES;
        const auto [d, dmin] = q4k_header(block);
        const uint8_t* scales = block + 4;
        const uint8_t* qs = block + 16;
        size_t q_off = 0;
        size_t is = 0;
        for (size_t g = 0; g < QK_K / 64; ++g) {
            const auto [sc1, m1] = get_scale_min_k4(is, scales);
            const auto [sc2, m2] = get_scale_min_k4(is + 1, scales);
            const float d1 = d * static_cast<float>(sc1);
            const float m1v = dmin * static_cast<float>(m1);
            const float d2 = d * static_cast<float>(sc2);
            const float m2v = dmin * static_cast<float>(m2);
            // lo nibbles → activation block at x_off; hi nibbles → block at x_off + 32.
            const size_t blk_lo = x_off / QK;
            const size_t blk_hi = (x_off + 32) / QK;
            const int8_t* xlo = x_i8.data() + x_off;
            const int8_t* xhi = x_i8.data() + x_off + 32;
            int32_t dot_lo = 0;
            int32_t dot_hi = 0;
            for (size_t l = 0; l < 32; ++l) {
                const int32_t nlo = static_cast<int32_t>(qs[q_off + l] & 0x0F);
                const int32_t nhi = static_cast<int32_t>(qs[q_off + l] >> 4);
                dot_lo += nlo * static_cast<int32_t>(xlo[l]);
                dot_hi += nhi * static_cast<int32_t>(xhi[l]);
            }
            // Σx per sub-block comes precomputed (once per activation row).
            acc += x_scales[blk_lo] *
                   (d1 * static_cast<float>(dot_lo) - m1v * static_cast<float>(x_sums[blk_lo]));
            acc += x_scales[blk_hi] *
                   (d2 * static_cast<float>(dot_hi) - m2v * static_cast<float>(x_sums[blk_hi]));
            x_off += 64;
            q_off += 32;
            is += 2;
        }
    }
    return acc;
}

#if SAPIENT_AARCH64
// NEON W4A8 Q4_K row dot — the fast decode kernel: `sdot` on the nibbles vs int8 activations,
// Σx from the precomputed block sums (quant.rs:1084-1141). Bit-identical to the scalar W4A8.
SAPIENT_TARGET_DOTPROD float dot_q4_k_row_q8_neon(std::span<const uint8_t> row_data,
                                                  std::span<const int8_t> x_i8,
                                                  std::span<const float> x_scales,
                                                  std::span<const int32_t> x_sums) {
    const size_t nb = row_data.size() / Q4_K_BLOCK_BYTES;
    check_q8_row(
        nb, x_i8, x_scales, QK_K / QK, "dot_q4_k_row_q8_neon: activations shorter than the row");
    check_len(x_sums.size(), nb * (QK_K / QK), "dot_q4_k_row_q8_neon: x_sums shorter than the row");
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    float acc = 0.0f;
    size_t x_off = 0;
    for (size_t bi = 0; bi < nb; ++bi) {
        const uint8_t* block = row_data.data() + bi * Q4_K_BLOCK_BYTES;
        const auto [d, dmin] = q4k_header(block);
        const uint8_t* scales = block + 4;
        const uint8_t* qs = block + 16;
        size_t q_off = 0;
        size_t is = 0;
        for (size_t g = 0; g < QK_K / 64; ++g) {
            const auto [sc1, m1] = get_scale_min_k4(is, scales);
            const auto [sc2, m2] = get_scale_min_k4(is + 1, scales);
            const float d1 = d * static_cast<float>(sc1);
            const float m1v = dmin * static_cast<float>(m1);
            const float d2 = d * static_cast<float>(sc2);
            const float m2v = dmin * static_cast<float>(m2);

            // 32 packed bytes → 32 lo nibbles (sub-block 2g) + 32 hi nibbles (2g+1).
            const uint8x16_t q0 = vld1q_u8(qs + q_off);
            const uint8x16_t q1 = vld1q_u8(qs + q_off + 16);
            const int8x16_t lo0 = vreinterpretq_s8_u8(vandq_u8(q0, mask));
            const int8x16_t lo1 = vreinterpretq_s8_u8(vandq_u8(q1, mask));
            const int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q0, 4));
            const int8x16_t hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q1, 4));

            const int8x16_t xlo0 = vld1q_s8(x_i8.data() + x_off);
            const int8x16_t xlo1 = vld1q_s8(x_i8.data() + x_off + 16);
            const int8x16_t xhi0 = vld1q_s8(x_i8.data() + x_off + 32);
            const int8x16_t xhi1 = vld1q_s8(x_i8.data() + x_off + 48);

            const int32x4_t zero = vdupq_n_s32(0);
            const int32_t dot_lo = vaddvq_s32(sdot_s32(sdot_s32(zero, lo0, xlo0), lo1, xlo1));
            const int32_t dot_hi = vaddvq_s32(sdot_s32(sdot_s32(zero, hi0, xhi0), hi1, xhi1));

            const size_t blk_lo = x_off / QK;
            const size_t blk_hi = (x_off + 32) / QK;
            acc += x_scales[blk_lo] *
                   (d1 * static_cast<float>(dot_lo) - m1v * static_cast<float>(x_sums[blk_lo]));
            acc += x_scales[blk_hi] *
                   (d2 * static_cast<float>(dot_hi) - m2v * static_cast<float>(x_sums[blk_hi]));
            x_off += 64;
            q_off += 32;
            is += 2;
        }
    }
    return acc;
}

// Four Q4_K rows against ONE int8 activation vector — the multi-row GEMV core: the activation
// registers and sums are loaded once per 64-weight group and reused across the four rows; per-row
// arithmetic (values and order) is identical to dot_q4_k_row_q8_neon (quant.rs:1154-1225).
SAPIENT_TARGET_DOTPROD std::array<float, 4>
dot_q4_k_4rows_q8_neon(std::array<std::span<const uint8_t>, 4> rows,
                       std::span<const int8_t> x_i8,
                       std::span<const float> x_scales,
                       std::span<const int32_t> x_sums) {
    const size_t n_blocks = rows[0].size() / Q4_K_BLOCK_BYTES;
    for (const auto& r : rows)
        check_len(r.size(),
                  n_blocks * Q4_K_BLOCK_BYTES,
                  "dot_q4_k_4rows_q8_neon: row shorter than row 0");
    check_q8_row(n_blocks,
                 x_i8,
                 x_scales,
                 QK_K / QK,
                 "dot_q4_k_4rows_q8_neon: activations shorter than the row");
    check_len(x_sums.size(),
              n_blocks * (QK_K / QK),
              "dot_q4_k_4rows_q8_neon: x_sums shorter than the row");
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    std::array<float, 4> acc{};
    size_t x_off = 0;
    for (size_t bi = 0; bi < n_blocks; ++bi) {
        const size_t base = bi * Q4_K_BLOCK_BYTES;
        // Per-row super-block headers, hoisted once per block.
        float dv[4];
        float dminv[4];
        for (size_t r = 0; r < 4; ++r) {
            const auto [d, dmin] = q4k_header(rows[r].data() + base);
            dv[r] = d;
            dminv[r] = dmin;
        }
        size_t q_off = 0;
        size_t is = 0;
        for (size_t g = 0; g < QK_K / 64; ++g) {
            // Shared activation work: loaded ONCE for all 4 rows; sums precomputed.
            const int8x16_t xlo0 = vld1q_s8(x_i8.data() + x_off);
            const int8x16_t xlo1 = vld1q_s8(x_i8.data() + x_off + 16);
            const int8x16_t xhi0 = vld1q_s8(x_i8.data() + x_off + 32);
            const int8x16_t xhi1 = vld1q_s8(x_i8.data() + x_off + 48);
            const int32_t sum_lo = x_sums[x_off / QK];
            const int32_t sum_hi = x_sums[(x_off + 32) / QK];
            const float xs_lo = x_scales[x_off / QK];
            const float xs_hi = x_scales[(x_off + 32) / QK];

            for (size_t r = 0; r < 4; ++r) {
                const uint8_t* b = rows[r].data() + base;
                const uint8_t* scales = b + 4;
                const uint8_t* qs = b + 16;
                const auto [sc1, m1] = get_scale_min_k4(is, scales);
                const auto [sc2, m2] = get_scale_min_k4(is + 1, scales);
                const float d1 = dv[r] * static_cast<float>(sc1);
                const float m1v = dminv[r] * static_cast<float>(m1);
                const float d2 = dv[r] * static_cast<float>(sc2);
                const float m2v = dminv[r] * static_cast<float>(m2);

                const uint8x16_t q0 = vld1q_u8(qs + q_off);
                const uint8x16_t q1 = vld1q_u8(qs + q_off + 16);
                const int8x16_t lo0 = vreinterpretq_s8_u8(vandq_u8(q0, mask));
                const int8x16_t lo1 = vreinterpretq_s8_u8(vandq_u8(q1, mask));
                const int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q0, 4));
                const int8x16_t hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q1, 4));

                const int32x4_t zero = vdupq_n_s32(0);
                const int32_t dot_lo = vaddvq_s32(sdot_s32(sdot_s32(zero, lo0, xlo0), lo1, xlo1));
                const int32_t dot_hi = vaddvq_s32(sdot_s32(sdot_s32(zero, hi0, xhi0), hi1, xhi1));

                acc[r] +=
                    xs_lo * (d1 * static_cast<float>(dot_lo) - m1v * static_cast<float>(sum_lo));
                acc[r] +=
                    xs_hi * (d2 * static_cast<float>(dot_hi) - m2v * static_cast<float>(sum_hi));
            }
            x_off += 64;
            q_off += 32;
            is += 2;
        }
    }
    return acc;
}
#endif

std::vector<uint8_t> repack_q4_k_rows4(std::span<const uint8_t> blocks, size_t n, size_t k) {
    if (n % 4 != 0) panic("Q4_K_R4 repack: rows must be a multiple of 4");
    if (k % QK_K != 0) panic("Q4_K_R4 repack: k must be a multiple of 256");
    const size_t nb = k / QK_K;
    const size_t row_bytes = nb * Q4_K_BLOCK_BYTES;
    if (blocks.size() != n * row_bytes) panic("Q4_K_R4 repack: blocks.len() != n * row_bytes");
    std::vector<uint8_t> out(blocks.size(), 0);
    for (size_t g = 0; g < n / 4; ++g)
        for (size_t b = 0; b < nb; ++b)
            for (size_t r = 0; r < 4; ++r) {
                const size_t src = ((g * 4 + r) * nb + b) * Q4_K_BLOCK_BYTES;
                const size_t dst = (g * 4 * nb + b * 4 + r) * Q4_K_BLOCK_BYTES;
                std::copy_n(blocks.data() + src, Q4_K_BLOCK_BYTES, out.data() + dst);
            }
    return out;
}

#if SAPIENT_AARCH64
// Four Q4_K rows in the R4 layout: one contiguous stream `[r0.b, r1.b, r2.b, r3.b]` per block;
// per-row arithmetic identical to dot_q4_k_row_q8_neon (quant.rs:1259-1330).
SAPIENT_TARGET_DOTPROD std::array<float, 4>
dot_q4_k_4rows_r4_neon(std::span<const uint8_t> packed,
                       std::span<const int8_t> x_i8,
                       std::span<const float> x_scales,
                       std::span<const int32_t> x_sums) {
    const size_t nb = packed.size() / (4 * Q4_K_BLOCK_BYTES);
    check_q8_row(
        nb, x_i8, x_scales, QK_K / QK, "dot_q4_k_4rows_r4_neon: activations shorter than the row");
    check_len(
        x_sums.size(), nb * (QK_K / QK), "dot_q4_k_4rows_r4_neon: x_sums shorter than the row");
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    std::array<float, 4> acc{};
    size_t x_off = 0;
    for (size_t b = 0; b < nb; ++b) {
        const size_t gbase = b * 4 * Q4_K_BLOCK_BYTES;
        float dv[4];
        float dminv[4];
        for (size_t r = 0; r < 4; ++r) {
            const auto [d, dmin] = q4k_header(packed.data() + gbase + r * Q4_K_BLOCK_BYTES);
            dv[r] = d;
            dminv[r] = dmin;
        }
        size_t q_off = 0;
        size_t is = 0;
        for (size_t g = 0; g < QK_K / 64; ++g) {
            const int8x16_t xlo0 = vld1q_s8(x_i8.data() + x_off);
            const int8x16_t xlo1 = vld1q_s8(x_i8.data() + x_off + 16);
            const int8x16_t xhi0 = vld1q_s8(x_i8.data() + x_off + 32);
            const int8x16_t xhi1 = vld1q_s8(x_i8.data() + x_off + 48);
            const int32_t sum_lo = x_sums[x_off / QK];
            const int32_t sum_hi = x_sums[(x_off + 32) / QK];
            const float xs_lo = x_scales[x_off / QK];
            const float xs_hi = x_scales[(x_off + 32) / QK];

            for (size_t r = 0; r < 4; ++r) {
                const uint8_t* blk = packed.data() + gbase + r * Q4_K_BLOCK_BYTES;
                const uint8_t* scales = blk + 4;
                const uint8_t* qs = blk + 16;
                const auto [sc1, m1] = get_scale_min_k4(is, scales);
                const auto [sc2, m2] = get_scale_min_k4(is + 1, scales);
                const float d1 = dv[r] * static_cast<float>(sc1);
                const float m1v = dminv[r] * static_cast<float>(m1);
                const float d2 = dv[r] * static_cast<float>(sc2);
                const float m2v = dminv[r] * static_cast<float>(m2);

                const uint8x16_t q0 = vld1q_u8(qs + q_off);
                const uint8x16_t q1 = vld1q_u8(qs + q_off + 16);
                const int8x16_t lo0 = vreinterpretq_s8_u8(vandq_u8(q0, mask));
                const int8x16_t lo1 = vreinterpretq_s8_u8(vandq_u8(q1, mask));
                const int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q0, 4));
                const int8x16_t hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q1, 4));

                const int32x4_t zero = vdupq_n_s32(0);
                const int32_t dot_lo = vaddvq_s32(sdot_s32(sdot_s32(zero, lo0, xlo0), lo1, xlo1));
                const int32_t dot_hi = vaddvq_s32(sdot_s32(sdot_s32(zero, hi0, xhi0), hi1, xhi1));

                acc[r] +=
                    xs_lo * (d1 * static_cast<float>(dot_lo) - m1v * static_cast<float>(sum_lo));
                acc[r] +=
                    xs_hi * (d2 * static_cast<float>(dot_hi) - m2v * static_cast<float>(sum_hi));
            }
            x_off += 64;
            q_off += 32;
            is += 2;
        }
    }
    return acc;
}
#endif

// ── Q4_K × Q8_K, SMMLA (quant.rs:394-429, 829-1000, 1024-1082, 1335-1615) ───

float dot_q4_k_row_q8k_scalar(std::span<const uint8_t> row_data,
                              std::span<const int8_t> x_i8,
                              std::span<const float> x_scales,
                              std::span<const int32_t> x_sums) {
    const size_t nb = row_data.size() / Q4_K_BLOCK_BYTES;
    check_q8_row(
        nb, x_i8, x_scales, 1, "dot_q4_k_row_q8k_scalar: activations shorter than the row");
    check_len(
        x_sums.size(), nb * (QK_K / QK), "dot_q4_k_row_q8k_scalar: x_sums shorter than the row");
    float acc = 0.0f;
    size_t x_off = 0;
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* block = row_data.data() + b * Q4_K_BLOCK_BYTES;
        const auto [d, dmin] = q4k_header(block);
        const uint8_t* scales = block + 4;
        const uint8_t* qs = block + 16;
        size_t q_off = 0;
        size_t is = 0;
        int32_t isum = 0;
        int32_t imin = 0;
        for (size_t g = 0; g < QK_K / 64; ++g) {
            const auto [sc1, m1] = get_scale_min_k4(is, scales);
            const auto [sc2, m2] = get_scale_min_k4(is + 1, scales);
            const int8_t* xlo = x_i8.data() + x_off;
            const int8_t* xhi = x_i8.data() + x_off + 32;
            int32_t dot_lo = 0;
            int32_t dot_hi = 0;
            for (size_t l = 0; l < 32; ++l) {
                dot_lo += static_cast<int32_t>(qs[q_off + l] & 0x0F) * static_cast<int32_t>(xlo[l]);
                dot_hi += static_cast<int32_t>(qs[q_off + l] >> 4) * static_cast<int32_t>(xhi[l]);
            }
            isum += static_cast<int32_t>(sc1) * dot_lo + static_cast<int32_t>(sc2) * dot_hi;
            imin += static_cast<int32_t>(m1) * x_sums[x_off / QK] +
                    static_cast<int32_t>(m2) * x_sums[(x_off + 32) / QK];
            x_off += 64;
            q_off += 32;
            is += 2;
        }
        acc += x_scales[b] * (d * static_cast<float>(isum) - dmin * static_cast<float>(imin));
    }
    return acc;
}

#if SAPIENT_AARCH64
namespace {
// 2×2 int8 matrix-multiply-accumulate — Rust's `smmla_s32` inline asm: treats `a` and `b` as
// row-major 2×8 i8 matrices and accumulates a·bᵀ into the four lanes `[a0·b0, a0·b1, a1·b0, a1·b1]`.
SAPIENT_TARGET_I8MM inline int32x4_t smmla_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    return vmmlaq_s32(acc, a, b);
}
// TRN1 / TRN2 on the 64-bit halves of two i8 vectors: `[a.lo, b.lo]` / `[a.hi, b.hi]`.
inline int8x16_t vtrn1q_s64_s8(int8x16_t a, int8x16_t b) {
    return vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s8(a), vreinterpretq_s64_s8(b)));
}
inline int8x16_t vtrn2q_s64_s8(int8x16_t a, int8x16_t b) {
    return vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s8(a), vreinterpretq_s64_s8(b)));
}
} // namespace

SAPIENT_TARGET_DOTPROD float dot_q4_k_row_q8k_neon(std::span<const uint8_t> row_data,
                                                   std::span<const int8_t> x_i8,
                                                   std::span<const float> x_scales,
                                                   std::span<const int32_t> x_sums) {
    const size_t nb = row_data.size() / Q4_K_BLOCK_BYTES;
    check_q8_row(nb, x_i8, x_scales, 1, "dot_q4_k_row_q8k_neon: activations shorter than the row");
    check_len(
        x_sums.size(), nb * (QK_K / QK), "dot_q4_k_row_q8k_neon: x_sums shorter than the row");
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    float acc = 0.0f;
    size_t x_off = 0;
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* block = row_data.data() + b * Q4_K_BLOCK_BYTES;
        const auto [d, dmin] = q4k_header(block);
        const uint8_t* scales = block + 4;
        const uint8_t* qs = block + 16;
        size_t q_off = 0;
        size_t is = 0;
        int32_t isum = 0;
        int32_t imin = 0;
        for (size_t g = 0; g < QK_K / 64; ++g) {
            const auto [sc1, m1] = get_scale_min_k4(is, scales);
            const auto [sc2, m2] = get_scale_min_k4(is + 1, scales);
            const uint8x16_t q0 = vld1q_u8(qs + q_off);
            const uint8x16_t q1 = vld1q_u8(qs + q_off + 16);
            const int8x16_t lo0 = vreinterpretq_s8_u8(vandq_u8(q0, mask));
            const int8x16_t lo1 = vreinterpretq_s8_u8(vandq_u8(q1, mask));
            const int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q0, 4));
            const int8x16_t hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q1, 4));
            const int8x16_t xlo0 = vld1q_s8(x_i8.data() + x_off);
            const int8x16_t xlo1 = vld1q_s8(x_i8.data() + x_off + 16);
            const int8x16_t xhi0 = vld1q_s8(x_i8.data() + x_off + 32);
            const int8x16_t xhi1 = vld1q_s8(x_i8.data() + x_off + 48);
            const int32x4_t zero = vdupq_n_s32(0);
            const int32_t dot_lo = vaddvq_s32(sdot_s32(sdot_s32(zero, lo0, xlo0), lo1, xlo1));
            const int32_t dot_hi = vaddvq_s32(sdot_s32(sdot_s32(zero, hi0, xhi0), hi1, xhi1));
            isum += static_cast<int32_t>(sc1) * dot_lo + static_cast<int32_t>(sc2) * dot_hi;
            imin += static_cast<int32_t>(m1) * x_sums[x_off / QK] +
                    static_cast<int32_t>(m2) * x_sums[(x_off + 32) / QK];
            x_off += 64;
            q_off += 32;
            is += 2;
        }
        acc += x_scales[b] * (d * static_cast<float>(isum) - dmin * static_cast<float>(imin));
    }
    return acc;
}

SAPIENT_TARGET_DOTPROD std::array<float, 4>
dot_q4_k_4rows_q8k_neon(std::array<std::span<const uint8_t>, 4> rows,
                        std::span<const int8_t> x_i8,
                        std::span<const float> x_scales,
                        std::span<const int32_t> x_sums) {
    const size_t n_blocks = rows[0].size() / Q4_K_BLOCK_BYTES;
    for (const auto& r : rows)
        check_len(r.size(),
                  n_blocks * Q4_K_BLOCK_BYTES,
                  "dot_q4_k_4rows_q8k_neon: row shorter than row 0");
    const size_t nb_eff = std::min(n_blocks, x_scales.size()); // .take(n_blocks) over x_scales
    check_len(x_i8.size(), nb_eff * QK_K, "dot_q4_k_4rows_q8k_neon: x_i8 shorter than the row");
    check_len(x_sums.size(),
              nb_eff * (QK_K / QK),
              "dot_q4_k_4rows_q8k_neon: x_sums shorter than the row");
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    std::array<float, 4> acc{};
    size_t x_off = 0;
    for (size_t bi = 0; bi < nb_eff; ++bi) {
        const float db = x_scales[bi];
        const size_t base = bi * Q4_K_BLOCK_BYTES;
        float dv[4];
        float dminv[4];
        for (size_t r = 0; r < 4; ++r) {
            const auto [d, dmin] = q4k_header(rows[r].data() + base);
            dv[r] = d;
            dminv[r] = dmin;
        }
        size_t q_off = 0;
        size_t is = 0;
        int32_t isum[4] = {0, 0, 0, 0};
        int32_t imin[4] = {0, 0, 0, 0};
        for (size_t g = 0; g < QK_K / 64; ++g) {
            const int8x16_t xlo0 = vld1q_s8(x_i8.data() + x_off);
            const int8x16_t xlo1 = vld1q_s8(x_i8.data() + x_off + 16);
            const int8x16_t xhi0 = vld1q_s8(x_i8.data() + x_off + 32);
            const int8x16_t xhi1 = vld1q_s8(x_i8.data() + x_off + 48);
            const int32_t sum_lo = x_sums[x_off / QK];
            const int32_t sum_hi = x_sums[(x_off + 32) / QK];
            for (size_t r = 0; r < 4; ++r) {
                const uint8_t* b = rows[r].data() + base;
                const uint8_t* scales = b + 4;
                const uint8_t* qs = b + 16;
                const auto [sc1, m1] = get_scale_min_k4(is, scales);
                const auto [sc2, m2] = get_scale_min_k4(is + 1, scales);
                const uint8x16_t q0 = vld1q_u8(qs + q_off);
                const uint8x16_t q1 = vld1q_u8(qs + q_off + 16);
                const int8x16_t lo0 = vreinterpretq_s8_u8(vandq_u8(q0, mask));
                const int8x16_t lo1 = vreinterpretq_s8_u8(vandq_u8(q1, mask));
                const int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q0, 4));
                const int8x16_t hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q1, 4));
                const int32x4_t zero = vdupq_n_s32(0);
                const int32_t dot_lo = vaddvq_s32(sdot_s32(sdot_s32(zero, lo0, xlo0), lo1, xlo1));
                const int32_t dot_hi = vaddvq_s32(sdot_s32(sdot_s32(zero, hi0, xhi0), hi1, xhi1));
                isum[r] += static_cast<int32_t>(sc1) * dot_lo + static_cast<int32_t>(sc2) * dot_hi;
                imin[r] += static_cast<int32_t>(m1) * sum_lo + static_cast<int32_t>(m2) * sum_hi;
            }
            x_off += 64;
            q_off += 32;
            is += 2;
        }
        for (size_t r = 0; r < 4; ++r)
            acc[r] +=
                db * (dv[r] * static_cast<float>(isum[r]) - dminv[r] * static_cast<float>(imin[r]));
    }
    return acc;
}

SAPIENT_TARGET_DOTPROD std::array<float, 4>
dot_q4_k_4rows_r4_q8k_neon(std::span<const uint8_t> packed,
                           std::span<const int8_t> x_i8,
                           std::span<const float> x_scales,
                           std::span<const int32_t> x_sums) {
    const size_t nb = packed.size() / (4 * Q4_K_BLOCK_BYTES);
    const size_t nb_eff = std::min(nb, x_scales.size()); // .take(nb)
    check_len(x_i8.size(), nb_eff * QK_K, "dot_q4_k_4rows_r4_q8k_neon: x_i8 shorter than the row");
    check_len(x_sums.size(),
              nb_eff * (QK_K / QK),
              "dot_q4_k_4rows_r4_q8k_neon: x_sums shorter than the row");
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    std::array<float, 4> acc{};
    size_t x_off = 0;
    for (size_t b = 0; b < nb_eff; ++b) {
        const float db = x_scales[b];
        const size_t gbase = b * 4 * Q4_K_BLOCK_BYTES;
        float dv[4];
        float dminv[4];
        for (size_t r = 0; r < 4; ++r) {
            const auto [d, dmin] = q4k_header(packed.data() + gbase + r * Q4_K_BLOCK_BYTES);
            dv[r] = d;
            dminv[r] = dmin;
        }
        size_t q_off = 0;
        size_t is = 0;
        int32_t isum[4] = {0, 0, 0, 0};
        int32_t imin[4] = {0, 0, 0, 0};
        for (size_t g = 0; g < QK_K / 64; ++g) {
            const int8x16_t xlo0 = vld1q_s8(x_i8.data() + x_off);
            const int8x16_t xlo1 = vld1q_s8(x_i8.data() + x_off + 16);
            const int8x16_t xhi0 = vld1q_s8(x_i8.data() + x_off + 32);
            const int8x16_t xhi1 = vld1q_s8(x_i8.data() + x_off + 48);
            const int32_t sum_lo = x_sums[x_off / QK];
            const int32_t sum_hi = x_sums[(x_off + 32) / QK];
            for (size_t r = 0; r < 4; ++r) {
                const uint8_t* blk = packed.data() + gbase + r * Q4_K_BLOCK_BYTES;
                const uint8_t* scales = blk + 4;
                const uint8_t* qs = blk + 16;
                const auto [sc1, m1] = get_scale_min_k4(is, scales);
                const auto [sc2, m2] = get_scale_min_k4(is + 1, scales);
                const uint8x16_t q0 = vld1q_u8(qs + q_off);
                const uint8x16_t q1 = vld1q_u8(qs + q_off + 16);
                const int8x16_t lo0 = vreinterpretq_s8_u8(vandq_u8(q0, mask));
                const int8x16_t lo1 = vreinterpretq_s8_u8(vandq_u8(q1, mask));
                const int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q0, 4));
                const int8x16_t hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q1, 4));
                const int32x4_t zero = vdupq_n_s32(0);
                const int32_t dot_lo = vaddvq_s32(sdot_s32(sdot_s32(zero, lo0, xlo0), lo1, xlo1));
                const int32_t dot_hi = vaddvq_s32(sdot_s32(sdot_s32(zero, hi0, xhi0), hi1, xhi1));
                isum[r] += static_cast<int32_t>(sc1) * dot_lo + static_cast<int32_t>(sc2) * dot_hi;
                imin[r] += static_cast<int32_t>(m1) * sum_lo + static_cast<int32_t>(m2) * sum_hi;
            }
            x_off += 64;
            q_off += 32;
            is += 2;
        }
        for (size_t r = 0; r < 4; ++r)
            acc[r] +=
                db * (dv[r] * static_cast<float>(isum[r]) - dminv[r] * static_cast<float>(imin[r]));
    }
    return acc;
}

// Four Q4_K rows (R4) × TWO per-32 int8 activation rows via `smmla` (quant.rs:861-990). Each
// 16-weight segment-pair costs two `trn` shuffles + one `smmla` per weight-row pair; the dots come
// out in lane order [r0·x0, r0·x1, r1·x0, r1·x1] and the f32 combine is dot_q4_k_row_q8_neon's.
SAPIENT_TARGET_I8MM std::array<std::array<float, 2>, 4>
dot_q4_k_4rows_r4_x2_smmla(std::span<const uint8_t> packed,
                           std::span<const int8_t> x0_i8,
                           std::span<const float> x0_scales,
                           std::span<const int32_t> x0_sums,
                           std::span<const int8_t> x1_i8,
                           std::span<const float> x1_scales,
                           std::span<const int32_t> x1_sums) {
    const size_t nb = packed.size() / (4 * Q4_K_BLOCK_BYTES);
    check_q8_row(
        nb, x0_i8, x0_scales, QK_K / QK, "dot_q4_k_4rows_r4_x2_smmla: x0 shorter than the row");
    check_q8_row(
        nb, x1_i8, x1_scales, QK_K / QK, "dot_q4_k_4rows_r4_x2_smmla: x1 shorter than the row");
    check_len(x0_sums.size(),
              nb * (QK_K / QK),
              "dot_q4_k_4rows_r4_x2_smmla: x0_sums shorter than the row");
    check_len(x1_sums.size(),
              nb * (QK_K / QK),
              "dot_q4_k_4rows_r4_x2_smmla: x1_sums shorter than the row");
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    std::array<std::array<float, 2>, 4> acc{};
    size_t x_off = 0;
    for (size_t b = 0; b < nb; ++b) {
        const size_t gbase = b * 4 * Q4_K_BLOCK_BYTES;
        float dv[4];
        float dminv[4];
        for (size_t r = 0; r < 4; ++r) {
            const auto [d, dmin] = q4k_header(packed.data() + gbase + r * Q4_K_BLOCK_BYTES);
            dv[r] = d;
            dminv[r] = dmin;
        }
        size_t q_off = 0;
        size_t is = 0;
        for (size_t g = 0; g < QK_K / 64; ++g) {
            // Activation vectors for BOTH rows, once; per-sub-block sums precomputed.
            const int8x16_t x0lo0 = vld1q_s8(x0_i8.data() + x_off);
            const int8x16_t x0lo1 = vld1q_s8(x0_i8.data() + x_off + 16);
            const int8x16_t x0hi0 = vld1q_s8(x0_i8.data() + x_off + 32);
            const int8x16_t x0hi1 = vld1q_s8(x0_i8.data() + x_off + 48);
            const int8x16_t x1lo0 = vld1q_s8(x1_i8.data() + x_off);
            const int8x16_t x1lo1 = vld1q_s8(x1_i8.data() + x_off + 16);
            const int8x16_t x1hi0 = vld1q_s8(x1_i8.data() + x_off + 32);
            const int8x16_t x1hi1 = vld1q_s8(x1_i8.data() + x_off + 48);
            const int32_t sum_lo[2] = {x0_sums[x_off / QK], x1_sums[x_off / QK]};
            const int32_t sum_hi[2] = {x0_sums[(x_off + 32) / QK], x1_sums[(x_off + 32) / QK]};
            const float xs_lo[2] = {x0_scales[x_off / QK], x1_scales[x_off / QK]};
            const float xs_hi[2] = {x0_scales[(x_off + 32) / QK], x1_scales[(x_off + 32) / QK]};
            // Pair the two activation rows per 8-byte k-segment: [x0_seg, x1_seg].
            const int8x16_t xlo_a = vtrn1q_s64_s8(x0lo0, x1lo0);
            const int8x16_t xlo_b = vtrn2q_s64_s8(x0lo0, x1lo0);
            const int8x16_t xlo_c = vtrn1q_s64_s8(x0lo1, x1lo1);
            const int8x16_t xlo_d = vtrn2q_s64_s8(x0lo1, x1lo1);
            const int8x16_t xhi_a = vtrn1q_s64_s8(x0hi0, x1hi0);
            const int8x16_t xhi_b = vtrn2q_s64_s8(x0hi0, x1hi0);
            const int8x16_t xhi_c = vtrn1q_s64_s8(x0hi1, x1hi1);
            const int8x16_t xhi_d = vtrn2q_s64_s8(x0hi1, x1hi1);

            for (size_t pair = 0; pair < 2; ++pair) {
                const size_t r0 = pair * 2;
                const size_t r1 = pair * 2 + 1;
                const uint8_t* qs0 = packed.data() + gbase + r0 * Q4_K_BLOCK_BYTES + 16;
                const uint8_t* qs1 = packed.data() + gbase + r1 * Q4_K_BLOCK_BYTES + 16;
                const uint8x16_t q0a = vld1q_u8(qs0 + q_off);
                const uint8x16_t q0b = vld1q_u8(qs0 + q_off + 16);
                const uint8x16_t q1a = vld1q_u8(qs1 + q_off);
                const uint8x16_t q1b = vld1q_u8(qs1 + q_off + 16);
                const int8x16_t lo0a = vreinterpretq_s8_u8(vandq_u8(q0a, mask));
                const int8x16_t lo0b = vreinterpretq_s8_u8(vandq_u8(q0b, mask));
                const int8x16_t lo1a = vreinterpretq_s8_u8(vandq_u8(q1a, mask));
                const int8x16_t lo1b = vreinterpretq_s8_u8(vandq_u8(q1b, mask));
                const int8x16_t hi0a = vreinterpretq_s8_u8(vshrq_n_u8(q0a, 4));
                const int8x16_t hi0b = vreinterpretq_s8_u8(vshrq_n_u8(q0b, 4));
                const int8x16_t hi1a = vreinterpretq_s8_u8(vshrq_n_u8(q1a, 4));
                const int8x16_t hi1b = vreinterpretq_s8_u8(vshrq_n_u8(q1b, 4));
                // Weight-row pairs per 8-byte k-segment: [w_r0_seg, w_r1_seg].
                const int8x16_t wlo_a = vtrn1q_s64_s8(lo0a, lo1a);
                const int8x16_t wlo_b = vtrn2q_s64_s8(lo0a, lo1a);
                const int8x16_t wlo_c = vtrn1q_s64_s8(lo0b, lo1b);
                const int8x16_t wlo_d = vtrn2q_s64_s8(lo0b, lo1b);
                const int8x16_t whi_a = vtrn1q_s64_s8(hi0a, hi1a);
                const int8x16_t whi_b = vtrn2q_s64_s8(hi0a, hi1a);
                const int8x16_t whi_c = vtrn1q_s64_s8(hi0b, hi1b);
                const int8x16_t whi_d = vtrn2q_s64_s8(hi0b, hi1b);

                const int32x4_t zero = vdupq_n_s32(0);
                int32x4_t dlo = smmla_s32(zero, wlo_a, xlo_a);
                dlo = smmla_s32(dlo, wlo_b, xlo_b);
                dlo = smmla_s32(dlo, wlo_c, xlo_c);
                dlo = smmla_s32(dlo, wlo_d, xlo_d);
                int32x4_t dhi = smmla_s32(zero, whi_a, xhi_a);
                dhi = smmla_s32(dhi, whi_b, xhi_b);
                dhi = smmla_s32(dhi, whi_c, xhi_c);
                dhi = smmla_s32(dhi, whi_d, xhi_d);
                const int32_t dlo_arr[4] = {vgetq_lane_s32(dlo, 0),
                                            vgetq_lane_s32(dlo, 1),
                                            vgetq_lane_s32(dlo, 2),
                                            vgetq_lane_s32(dlo, 3)};
                const int32_t dhi_arr[4] = {vgetq_lane_s32(dhi, 0),
                                            vgetq_lane_s32(dhi, 1),
                                            vgetq_lane_s32(dhi, 2),
                                            vgetq_lane_s32(dhi, 3)};

                const size_t rows2[2] = {r0, r1};
                for (size_t ri = 0; ri < 2; ++ri) {
                    const size_t row = rows2[ri];
                    const uint8_t* scales = packed.data() + gbase + row * Q4_K_BLOCK_BYTES + 4;
                    const auto [sc1, m1] = get_scale_min_k4(is, scales);
                    const auto [sc2, m2] = get_scale_min_k4(is + 1, scales);
                    const float d1 = dv[row] * static_cast<float>(sc1);
                    const float m1v = dminv[row] * static_cast<float>(m1);
                    const float d2 = dv[row] * static_cast<float>(sc2);
                    const float m2v = dminv[row] * static_cast<float>(m2);
                    for (size_t xr = 0; xr < 2; ++xr) {
                        const int32_t dot_lo = dlo_arr[ri * 2 + xr];
                        const int32_t dot_hi = dhi_arr[ri * 2 + xr];
                        acc[row][xr] += xs_lo[xr] * (d1 * static_cast<float>(dot_lo) -
                                                     m1v * static_cast<float>(sum_lo[xr]));
                        acc[row][xr] += xs_hi[xr] * (d2 * static_cast<float>(dot_hi) -
                                                     m2v * static_cast<float>(sum_hi[xr]));
                    }
                }
            }
            x_off += 64;
            q_off += 32;
            is += 2;
        }
    }
    return acc;
}

// Same trn/smmla core with the integer-domain combine of dot_q4_k_row_q8k_neon (quant.rs:1487-1615).
SAPIENT_TARGET_I8MM std::array<std::array<float, 2>, 4>
dot_q4_k_4rows_r4_x2_q8k_smmla(std::span<const uint8_t> packed,
                               std::span<const int8_t> x0_i8,
                               std::span<const float> x0_scales,
                               std::span<const int32_t> x0_sums,
                               std::span<const int8_t> x1_i8,
                               std::span<const float> x1_scales,
                               std::span<const int32_t> x1_sums) {
    const size_t nb = packed.size() / (4 * Q4_K_BLOCK_BYTES);
    const size_t nb_eff = std::min({nb, x0_scales.size(), x1_scales.size()}); // zip().take(nb)
    check_len(
        x0_i8.size(), nb_eff * QK_K, "dot_q4_k_4rows_r4_x2_q8k_smmla: x0 shorter than the row");
    check_len(
        x1_i8.size(), nb_eff * QK_K, "dot_q4_k_4rows_r4_x2_q8k_smmla: x1 shorter than the row");
    check_len(x0_sums.size(),
              nb_eff * (QK_K / QK),
              "dot_q4_k_4rows_r4_x2_q8k_smmla: x0_sums shorter than the row");
    check_len(x1_sums.size(),
              nb_eff * (QK_K / QK),
              "dot_q4_k_4rows_r4_x2_q8k_smmla: x1_sums shorter than the row");
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    std::array<std::array<float, 2>, 4> acc{};
    size_t x_off = 0;
    for (size_t b = 0; b < nb_eff; ++b) {
        const float db[2] = {x0_scales[b], x1_scales[b]};
        const size_t gbase = b * 4 * Q4_K_BLOCK_BYTES;
        float dv[4];
        float dminv[4];
        for (size_t r = 0; r < 4; ++r) {
            const auto [d, dmin] = q4k_header(packed.data() + gbase + r * Q4_K_BLOCK_BYTES);
            dv[r] = d;
            dminv[r] = dmin;
        }
        size_t q_off = 0;
        size_t is = 0;
        int32_t isum[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
        int32_t imin[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
        for (size_t g = 0; g < QK_K / 64; ++g) {
            const int8x16_t x0lo0 = vld1q_s8(x0_i8.data() + x_off);
            const int8x16_t x0lo1 = vld1q_s8(x0_i8.data() + x_off + 16);
            const int8x16_t x0hi0 = vld1q_s8(x0_i8.data() + x_off + 32);
            const int8x16_t x0hi1 = vld1q_s8(x0_i8.data() + x_off + 48);
            const int8x16_t x1lo0 = vld1q_s8(x1_i8.data() + x_off);
            const int8x16_t x1lo1 = vld1q_s8(x1_i8.data() + x_off + 16);
            const int8x16_t x1hi0 = vld1q_s8(x1_i8.data() + x_off + 32);
            const int8x16_t x1hi1 = vld1q_s8(x1_i8.data() + x_off + 48);
            const int32_t sum_lo[2] = {x0_sums[x_off / QK], x1_sums[x_off / QK]};
            const int32_t sum_hi[2] = {x0_sums[(x_off + 32) / QK], x1_sums[(x_off + 32) / QK]};
            const int8x16_t xlo_a = vtrn1q_s64_s8(x0lo0, x1lo0);
            const int8x16_t xlo_b = vtrn2q_s64_s8(x0lo0, x1lo0);
            const int8x16_t xlo_c = vtrn1q_s64_s8(x0lo1, x1lo1);
            const int8x16_t xlo_d = vtrn2q_s64_s8(x0lo1, x1lo1);
            const int8x16_t xhi_a = vtrn1q_s64_s8(x0hi0, x1hi0);
            const int8x16_t xhi_b = vtrn2q_s64_s8(x0hi0, x1hi0);
            const int8x16_t xhi_c = vtrn1q_s64_s8(x0hi1, x1hi1);
            const int8x16_t xhi_d = vtrn2q_s64_s8(x0hi1, x1hi1);

            for (size_t pair = 0; pair < 2; ++pair) {
                const size_t r0 = pair * 2;
                const size_t r1 = pair * 2 + 1;
                const uint8_t* qs0 = packed.data() + gbase + r0 * Q4_K_BLOCK_BYTES + 16;
                const uint8_t* qs1 = packed.data() + gbase + r1 * Q4_K_BLOCK_BYTES + 16;
                const uint8x16_t q0a = vld1q_u8(qs0 + q_off);
                const uint8x16_t q0b = vld1q_u8(qs0 + q_off + 16);
                const uint8x16_t q1a = vld1q_u8(qs1 + q_off);
                const uint8x16_t q1b = vld1q_u8(qs1 + q_off + 16);
                const int8x16_t lo0a = vreinterpretq_s8_u8(vandq_u8(q0a, mask));
                const int8x16_t lo0b = vreinterpretq_s8_u8(vandq_u8(q0b, mask));
                const int8x16_t lo1a = vreinterpretq_s8_u8(vandq_u8(q1a, mask));
                const int8x16_t lo1b = vreinterpretq_s8_u8(vandq_u8(q1b, mask));
                const int8x16_t hi0a = vreinterpretq_s8_u8(vshrq_n_u8(q0a, 4));
                const int8x16_t hi0b = vreinterpretq_s8_u8(vshrq_n_u8(q0b, 4));
                const int8x16_t hi1a = vreinterpretq_s8_u8(vshrq_n_u8(q1a, 4));
                const int8x16_t hi1b = vreinterpretq_s8_u8(vshrq_n_u8(q1b, 4));
                const int8x16_t wlo_a = vtrn1q_s64_s8(lo0a, lo1a);
                const int8x16_t wlo_b = vtrn2q_s64_s8(lo0a, lo1a);
                const int8x16_t wlo_c = vtrn1q_s64_s8(lo0b, lo1b);
                const int8x16_t wlo_d = vtrn2q_s64_s8(lo0b, lo1b);
                const int8x16_t whi_a = vtrn1q_s64_s8(hi0a, hi1a);
                const int8x16_t whi_b = vtrn2q_s64_s8(hi0a, hi1a);
                const int8x16_t whi_c = vtrn1q_s64_s8(hi0b, hi1b);
                const int8x16_t whi_d = vtrn2q_s64_s8(hi0b, hi1b);

                const int32x4_t zero = vdupq_n_s32(0);
                int32x4_t dlo = smmla_s32(zero, wlo_a, xlo_a);
                dlo = smmla_s32(dlo, wlo_b, xlo_b);
                dlo = smmla_s32(dlo, wlo_c, xlo_c);
                dlo = smmla_s32(dlo, wlo_d, xlo_d);
                int32x4_t dhi = smmla_s32(zero, whi_a, xhi_a);
                dhi = smmla_s32(dhi, whi_b, xhi_b);
                dhi = smmla_s32(dhi, whi_c, xhi_c);
                dhi = smmla_s32(dhi, whi_d, xhi_d);
                const int32_t dlo_arr[4] = {vgetq_lane_s32(dlo, 0),
                                            vgetq_lane_s32(dlo, 1),
                                            vgetq_lane_s32(dlo, 2),
                                            vgetq_lane_s32(dlo, 3)};
                const int32_t dhi_arr[4] = {vgetq_lane_s32(dhi, 0),
                                            vgetq_lane_s32(dhi, 1),
                                            vgetq_lane_s32(dhi, 2),
                                            vgetq_lane_s32(dhi, 3)};

                const size_t rows2[2] = {r0, r1};
                for (size_t ri = 0; ri < 2; ++ri) {
                    const size_t row = rows2[ri];
                    const uint8_t* scales = packed.data() + gbase + row * Q4_K_BLOCK_BYTES + 4;
                    const auto [sc1, m1] = get_scale_min_k4(is, scales);
                    const auto [sc2, m2] = get_scale_min_k4(is + 1, scales);
                    for (size_t xr = 0; xr < 2; ++xr) {
                        const int32_t dot_lo = dlo_arr[ri * 2 + xr];
                        const int32_t dot_hi = dhi_arr[ri * 2 + xr];
                        isum[row][xr] +=
                            static_cast<int32_t>(sc1) * dot_lo + static_cast<int32_t>(sc2) * dot_hi;
                        imin[row][xr] += static_cast<int32_t>(m1) * sum_lo[xr] +
                                         static_cast<int32_t>(m2) * sum_hi[xr];
                    }
                }
            }
            x_off += 64;
            q_off += 32;
            is += 2;
        }
        for (size_t r = 0; r < 4; ++r)
            for (size_t xr = 0; xr < 2; ++xr)
                acc[r][xr] += db[xr] * (dv[r] * static_cast<float>(isum[r][xr]) -
                                        dminv[r] * static_cast<float>(imin[r][xr]));
    }
    return acc;
}
#endif

// ── Q5_K (quant.rs:1617-1755) ────────────────────────────────────────────────

float detail::dot_q5_k_row_f32_scalar(std::span<const uint8_t> row_data, std::span<const float> x) {
    const size_t nb = row_data.size() / Q5_K_BLOCK_BYTES;
    check_len(x.size(), nb * QK_K, "dot_q5_k_row_f32: x shorter than the row");
    float acc = 0.0f;
    size_t x_off = 0;
    for (size_t bi = 0; bi < nb; ++bi) {
        const uint8_t* block = row_data.data() + bi * Q5_K_BLOCK_BYTES;
        const auto [d, dmin] = q4k_header(block);
        const uint8_t* scales = block + 4;
        const uint8_t* qh = block + 16;
        const uint8_t* ql = block + 48;
        size_t ql_off = 0;
        size_t is = 0;
        uint8_t u1 = 1;
        uint8_t u2 = 2;
        for (size_t g = 0; g < QK_K / 64; ++g) {
            const auto [sc1, m1] = get_scale_min_k4(is, scales);
            const float d1 = d * static_cast<float>(sc1);
            const float m1v = dmin * static_cast<float>(m1);
            const auto [sc2, m2] = get_scale_min_k4(is + 1, scales);
            const float d2 = d * static_cast<float>(sc2);
            const float m2v = dmin * static_cast<float>(m2);
            // The 5th bit is PER-ELEMENT: ggml reads qh[l] (l = 0..32) and selects the active
            // bit-plane with u1/u2 (which shift by 2 each sub-block pair).
            for (size_t l = 0; l < 32; ++l) {
                const float hi1 = (qh[l] & u1) != 0 ? 16.0f : 0.0f;
                const float hi2 = (qh[l] & u2) != 0 ? 16.0f : 0.0f;
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

#if SAPIENT_AARCH64
namespace {
// Widen 16 u8 lanes to four f32x4 (`vf` of the Rust accum! macros).
inline void widen_u8x16_to_f32x4x4(uint8x16_t v, float32x4_t out[4]) {
    const uint16x8_t v16lo = vmovl_u8(vget_low_u8(v));
    const uint16x8_t v16hi = vmovl_high_u8(v);
    out[0] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v16lo)));
    out[1] = vcvtq_f32_u32(vmovl_high_u16(v16lo));
    out[2] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v16hi)));
    out[3] = vcvtq_f32_u32(vmovl_high_u16(v16hi));
}
// Q5_K accum!: acc += (d·val − m)·x over 16 lanes (4× f32x4).
inline float32x4_t q5_accum(float32x4_t acc, uint8x16_t val, float dd, float mm, const float* xb) {
    float32x4_t vf[4];
    widen_u8x16_to_f32x4x4(val, vf);
    const float32x4_t mneg = vdupq_n_f32(mm);
    for (size_t c = 0; c < 4; ++c) {
        const float32x4_t t = vsubq_f32(vmulq_n_f32(vf[c], dd), mneg);
        const float32x4_t xc = vld1q_f32(xb + c * 4);
        acc = vfmaq_f32(acc, t, xc);
    }
    return acc;
}
} // namespace

// NEON Q5_K row dot — the (fixed) scalar reference 16 lanes at a time; ONE accumulator across the
// whole row, reduced once at the end (quant.rs:1678-1755).
float detail::dot_q5_k_row_f32_neon(std::span<const uint8_t> row_data, std::span<const float> x) {
    const size_t nb = row_data.size() / Q5_K_BLOCK_BYTES;
    check_len(x.size(), nb * QK_K, "dot_q5_k_row_f32: x shorter than the row");
    const uint8x16_t mask0f = vdupq_n_u8(0x0F);
    const uint8x16_t sixteen = vdupq_n_u8(16);
    float32x4_t acc = vdupq_n_f32(0.0f);
    size_t x_off = 0;
    for (size_t bi = 0; bi < nb; ++bi) {
        const uint8_t* block = row_data.data() + bi * Q5_K_BLOCK_BYTES;
        const auto [d, dmin] = q4k_header(block);
        const uint8_t* scales = block + 4;
        const uint8_t* qh = block + 16;
        const uint8_t* ql = block + 48;
        size_t ql_off = 0;
        size_t is = 0;
        uint8_t u1 = 1;
        uint8_t u2 = 2;
        for (size_t g = 0; g < QK_K / 64; ++g) {
            const auto [sc1, m1] = get_scale_min_k4(is, scales);
            const auto [sc2, m2] = get_scale_min_k4(is + 1, scales);
            const float d1 = d * static_cast<float>(sc1);
            const float m1v = dmin * static_cast<float>(m1);
            const float d2 = d * static_cast<float>(sc2);
            const float m2v = dmin * static_cast<float>(m2);
            const uint8x16_t u1v = vdupq_n_u8(u1);
            const uint8x16_t u2v = vdupq_n_u8(u2);
            for (const size_t half : {size_t{0}, size_t{16}}) {
                const uint8x16_t qlv = vld1q_u8(ql + ql_off + half);
                const uint8x16_t qhv = vld1q_u8(qh + half);
                // 5th bit → 16 or 0 (per element): (qh & u) ? 16 : 0.
                const uint8x16_t hi1 = vandq_u8(vtstq_u8(qhv, u1v), sixteen);
                const uint8x16_t hi2 = vandq_u8(vtstq_u8(qhv, u2v), sixteen);
                const uint8x16_t val_lo = vaddq_u8(vandq_u8(qlv, mask0f), hi1); // 0..31
                const uint8x16_t val_hi = vaddq_u8(vshrq_n_u8(qlv, 4), hi2);
                acc = q5_accum(acc, val_lo, d1, m1v, x.data() + x_off + half);
                acc = q5_accum(acc, val_hi, d2, m2v, x.data() + x_off + 32 + half);
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
    return vaddvq_f32(acc);
}
#endif

float dot_q5_k_row_f32(std::span<const uint8_t> row_data, std::span<const float> x) {
#if SAPIENT_AARCH64
    return detail::dot_q5_k_row_f32_neon(row_data, x);
#else
    return detail::dot_q5_k_row_f32_scalar(row_data, x);
#endif
}

// ── Q6_K f32, repack, R4 f32 (quant.rs:2000-2018, 2208-2313, 2564-2710) ─────

namespace {
// `sc[i] as i8 as f32`.
inline float q6_scale(const uint8_t* sc, size_t i) {
    return static_cast<float>(detail::i8v(sc[i]));
}
#if SAPIENT_AARCH64
// The four 6-bit sub-positions of one 16-lane group (each 16× u8 in [0,63]) — shared by every
// NEON Q6_K kernel (quant.rs:2637-2652 and its five copies).
inline void q6_unpack(const uint8_t* ql,
                      const uint8_t* qh,
                      size_t ql_off,
                      size_t qh_off,
                      size_t l0,
                      uint8x16_t q[4]) {
    const uint8x16_t mask0f = vdupq_n_u8(0x0F);
    const uint8x16_t mask3 = vdupq_n_u8(0x03);
    const uint8x16_t ql_lo = vld1q_u8(ql + ql_off + l0);
    const uint8x16_t ql_hi = vld1q_u8(ql + ql_off + l0 + 32);
    const uint8x16_t qhv = vld1q_u8(qh + qh_off + l0);
    q[0] = vorrq_u8(vandq_u8(ql_lo, mask0f), vshlq_n_u8(vandq_u8(qhv, mask3), 4));
    q[1] = vorrq_u8(vandq_u8(ql_hi, mask0f), vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, 2), mask3), 4));
    q[2] = vorrq_u8(vshrq_n_u8(ql_lo, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, 4), mask3), 4));
    q[3] = vorrq_u8(vshrq_n_u8(ql_hi, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, 6), mask3), 4));
}
// Q6_K f32 accum!: acc += scale · Σ_lane (q − 32) · x over 16 lanes (4× f32x4).
inline float32x4_t q6_accum_f32(float32x4_t acc, uint8x16_t q, float scale, const float* xb) {
    float32x4_t qf[4];
    widen_u8x16_to_f32x4x4(q, qf);
    const float32x4_t m32 = vdupq_n_f32(32.0f);
    const float32x4_t sv = vdupq_n_f32(scale);
    for (size_t c = 0; c < 4; ++c) {
        const float32x4_t qm = vsubq_f32(qf[c], m32);
        const float32x4_t xc = vld1q_f32(xb + c * 4);
        acc = vfmaq_f32(acc, vmulq_f32(qm, sv), xc);
    }
    return acc;
}
#endif
} // namespace

float detail::dot_q6_k_row_f32_scalar(std::span<const uint8_t> row_data, std::span<const float> x) {
    const size_t nb = row_data.size() / Q6_K_BLOCK_BYTES;
    check_len(x.size(), nb * QK_K, "dot_q6_k_row_f32: x shorter than the row");
    float acc = 0.0f;
    size_t x_off = 0;
    for (size_t bi = 0; bi < nb; ++bi) {
        const uint8_t* block = row_data.data() + bi * Q6_K_BLOCK_BYTES;
        const uint8_t* ql = block;
        const uint8_t* qh = block + 128;
        const uint8_t* sc = block + 192;
        const float d = f16_le_to_f32(block + 208);
        size_t ql_off = 0;
        size_t qh_off = 0;
        // 16 i8 scales per super-block (one per 16-element group): within each 128-element half
        // the 4 sub-groups use offsets +0/+2/+4/+6, split again at l==16 (`is = l/16`), base +8
        // per 128-block (ggml dequantize_row_q6_K).
        size_t sc_base = 0;
        for (size_t half = 0; half < QK_K / 128; ++half) {
            for (size_t l = 0; l < 32; ++l) {
                const size_t is = l / 16;
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
                acc += d * q6_scale(sc, sc_base + is) * q1 * x[x_off + l];
                acc += d * q6_scale(sc, sc_base + is + 2) * q2 * x[x_off + l + 32];
                acc += d * q6_scale(sc, sc_base + is + 4) * q3 * x[x_off + l + 64];
                acc += d * q6_scale(sc, sc_base + is + 6) * q4 * x[x_off + l + 96];
            }
            x_off += 128;
            ql_off += 64;
            qh_off += 32;
            sc_base += 8;
        }
    }
    return acc;
}

#if SAPIENT_AARCH64
// NEON Q6_K row dot — the scalar reference 16 lanes at a time, ONE accumulator across the whole
// row (quant.rs:2623-2710). Same `sc_base + is + {0,2,4,6}` scale layout as the scalar.
float detail::dot_q6_k_row_f32_neon(std::span<const uint8_t> row_data, std::span<const float> x) {
    const size_t nb = row_data.size() / Q6_K_BLOCK_BYTES;
    check_len(x.size(), nb * QK_K, "dot_q6_k_row_f32: x shorter than the row");
    float32x4_t acc = vdupq_n_f32(0.0f);
    size_t x_off = 0;
    for (size_t bi = 0; bi < nb; ++bi) {
        const uint8_t* block = row_data.data() + bi * Q6_K_BLOCK_BYTES;
        const uint8_t* ql = block;
        const uint8_t* qh = block + 128;
        const uint8_t* sc = block + 192;
        const float d = f16_le_to_f32(block + 208);
        size_t ql_off = 0;
        size_t qh_off = 0;
        size_t sc_base = 0;
        for (size_t half = 0; half < QK_K / 128; ++half) {
            for (const size_t l0 : {size_t{0}, size_t{16}}) {
                const size_t is = l0 / 16;
                uint8x16_t q[4];
                q6_unpack(ql, qh, ql_off, qh_off, l0, q);
                const float s1 = d * q6_scale(sc, sc_base + is);
                const float s2 = d * q6_scale(sc, sc_base + is + 2);
                const float s3 = d * q6_scale(sc, sc_base + is + 4);
                const float s4 = d * q6_scale(sc, sc_base + is + 6);
                acc = q6_accum_f32(acc, q[0], s1, x.data() + x_off + l0);
                acc = q6_accum_f32(acc, q[1], s2, x.data() + x_off + 32 + l0);
                acc = q6_accum_f32(acc, q[2], s3, x.data() + x_off + 64 + l0);
                acc = q6_accum_f32(acc, q[3], s4, x.data() + x_off + 96 + l0);
            }
            x_off += 128;
            ql_off += 64;
            qh_off += 32;
            sc_base += 8;
        }
    }
    return vaddvq_f32(acc);
}

// Four R4 Q6_K rows × one f32 activation vector: one packed stream per row-group; per-row math
// identical to dot_q6_k_row_f32_neon, four accumulators carried across blocks (quant.rs:2208-2313).
std::array<float, 4> dot_q6_k_4rows_r4_neon(std::span<const uint8_t> packed,
                                            std::span<const float> x) {
    const size_t nb = packed.size() / (4 * Q6_K_BLOCK_BYTES);
    check_len(x.size(), nb * QK_K, "dot_q6_k_4rows_r4_neon: x shorter than the row");
    float32x4_t accv[4] = {
        vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    size_t x_off = 0;
    for (size_t b = 0; b < nb; ++b) {
        const size_t gbase = b * 4 * Q6_K_BLOCK_BYTES;
        for (size_t r = 0; r < 4; ++r) {
            const uint8_t* block = packed.data() + gbase + r * Q6_K_BLOCK_BYTES;
            const uint8_t* ql = block;
            const uint8_t* qh = block + 128;
            const uint8_t* sc = block + 192;
            const float d = f16_le_to_f32(block + 208);
            float32x4_t acc = accv[r];
            size_t xo = x_off;
            size_t ql_off = 0;
            size_t qh_off = 0;
            size_t sc_base = 0;
            for (size_t half = 0; half < QK_K / 128; ++half) {
                for (const size_t l0 : {size_t{0}, size_t{16}}) {
                    const size_t is = l0 / 16;
                    uint8x16_t q[4];
                    q6_unpack(ql, qh, ql_off, qh_off, l0, q);
                    const float s1 = d * q6_scale(sc, sc_base + is);
                    const float s2 = d * q6_scale(sc, sc_base + is + 2);
                    const float s3 = d * q6_scale(sc, sc_base + is + 4);
                    const float s4 = d * q6_scale(sc, sc_base + is + 6);
                    acc = q6_accum_f32(acc, q[0], s1, x.data() + xo + l0);
                    acc = q6_accum_f32(acc, q[1], s2, x.data() + xo + 32 + l0);
                    acc = q6_accum_f32(acc, q[2], s3, x.data() + xo + 64 + l0);
                    acc = q6_accum_f32(acc, q[3], s4, x.data() + xo + 96 + l0);
                }
                xo += 128;
                ql_off += 64;
                qh_off += 32;
                sc_base += 8;
            }
            accv[r] = acc;
        }
        x_off += QK_K;
    }
    return {vaddvq_f32(accv[0]), vaddvq_f32(accv[1]), vaddvq_f32(accv[2]), vaddvq_f32(accv[3])};
}
#endif

float dot_q6_k_row_f32(std::span<const uint8_t> row_data, std::span<const float> x) {
#if SAPIENT_AARCH64
    return detail::dot_q6_k_row_f32_neon(row_data, x);
#else
    return detail::dot_q6_k_row_f32_scalar(row_data, x);
#endif
}

std::vector<uint8_t> repack_q6_k_rows4(std::span<const uint8_t> blocks, size_t n, size_t k) {
    if (n % 4 != 0) panic("Q6_K_R4 repack: rows must be a multiple of 4");
    if (k % QK_K != 0) panic("Q6_K_R4 repack: k must be a multiple of 256");
    const size_t nb = k / QK_K;
    const size_t row_bytes = nb * Q6_K_BLOCK_BYTES;
    if (blocks.size() != n * row_bytes) panic("Q6_K_R4 repack: blocks.len() != n * row_bytes");
    std::vector<uint8_t> out(blocks.size(), 0);
    for (size_t g = 0; g < n / 4; ++g)
        for (size_t b = 0; b < nb; ++b)
            for (size_t r = 0; r < 4; ++r) {
                const size_t src = ((g * 4 + r) * nb + b) * Q6_K_BLOCK_BYTES;
                const size_t dst = (g * 4 * nb + b * 4 + r) * Q6_K_BLOCK_BYTES;
                std::copy_n(blocks.data() + src, Q6_K_BLOCK_BYTES, out.data() + dst);
            }
    return out;
}

// ── Q6_K W6A8 / Q8_K / SMMLA (Task 5) ────────────────────────────────────

} // namespace sapient::backends_cpu::kernels::quant
