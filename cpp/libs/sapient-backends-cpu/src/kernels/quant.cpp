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

// ── K-quants: Q4_K (Task 2) ──────────────────────────────────────────────────

} // namespace sapient::backends_cpu::kernels::quant
