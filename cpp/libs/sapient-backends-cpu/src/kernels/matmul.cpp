// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/backends_cpu/kernels/matmul.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "sapient/backends_cpu/cpu_features.hpp"
#include "sapient/backends_cpu/env.hpp"
#include "sapient/backends_cpu/kernels/quant.hpp"
#include "sapient/backends_cpu/parallel.hpp"
#include "sapient/backends_cpu/sgemm.hpp"
#include "sapient/backends_cpu/spinpool.hpp"
#include "sapient/backends_cpu/thermal.hpp"
#include "sapient/core/dtype.hpp"
#include "sapient/core/f16.hpp"
#include "sapient/core/panic.hpp"

namespace sapient::backends_cpu::kernels::matmul {

using sapient::core::DType;
using sapient::core::Error;
using sapient::core::Shape;

namespace {

// ── f32 dot products (matmul.rs:152-241) ─────────────────────────────────────
#if defined(__aarch64__) || defined(_M_ARM64)
// 16-element unroll: four vfmaq_f32 into ONE accumulator, then a 4-wide tail, vaddvq_f32, and
// the scalar tail added after the horizontal reduction.
float dot_f32_neon_fast(const float* a, const float* b, size_t n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        acc = vfmaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
        acc = vfmaq_f32(acc, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        acc = vfmaq_f32(acc, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        acc = vfmaq_f32(acc, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    for (; i + 4 <= n; i += 4)
        acc = vfmaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
    float s = vaddvq_f32(acc);
    for (; i < n; ++i)
        s += a[i] * b[i];
    return s;
}
float dot_f32_fast(const float* a, const float* b, size_t n) {
    return dot_f32_neon_fast(a, b, n);
}

#elif defined(__x86_64__) || defined(_M_X64)
// AVX2+FMA (the ONE runtime-gated x86 dot in this plan); the horizontal sum is the exact
// sequence from matmul.rs:206-213.
__attribute__((target("avx2,fma"))) float dot_f32_avx2(const float* a, const float* b, size_t n) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
    const __m128 lo = _mm256_castps256_ps128(acc);
    const __m128 hi = _mm256_extractf128_ps(acc, 1);
    const __m128 sum4 = _mm_add_ps(lo, hi);
    const __m128 shuf = _mm_movehdup_ps(sum4);
    const __m128 sum2 = _mm_add_ps(sum4, shuf);
    const __m128 sum1 = _mm_add_ss(sum2, _mm_movehl_ps(shuf, sum2));
    float s = _mm_cvtss_f32(sum1);
    for (; i < n; ++i)
        s += a[i] * b[i];
    return s;
}
float dot_f32_fast(const float* a, const float* b, size_t n) {
    if (cpu_features::has_avx2_fma()) return dot_f32_avx2(a, b, n);
    float s = -0.0f; // iter().zip().map().sum() seeds at -0.0
    for (size_t i = 0; i < n; ++i)
        s += a[i] * b[i];
    return s;
}

#else
float dot_f32_fast(const float* a, const float* b, size_t n) {
    float s = -0.0f;
    for (size_t i = 0; i < n; ++i)
        s += a[i] * b[i];
    return s;
}
#endif

// ── f32 × f16 dot (matmul.rs:252-315) ────────────────────────────────────────
#if defined(__aarch64__) || defined(_M_ARM64)
// F16→F32 by NEON integer bit surgery, VERBATIM from Rust (spec §3.5): valid for positive normal
// f16 values only. Subnormal/inf/NaN decode differently from the scalar tail below, and the
// unmasked `>> 10` carries the f16 sign bit into bit 5 of `exp16`, so negative weights end up with
// +32 in the exponent field (×2^32). Rust is frozen; the divergence is reproduced, recorded in
// docs/PARITY.md, and pinned bit-exactly by the `matmul_nt_f16_m1` golden case.
float dot_f32_x_f16_neon(const float* a_f32, const uint16_t* b_f16, size_t n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    size_t i = 0;
    const uint32x4_t mask_mant = vdupq_n_u32(0x000003FFu); // 10-bit mantissa mask
    const uint32x4_t mask_sign = vdupq_n_u32(0x00008000u); // sign bit in u16 position
    const uint32x4_t exp_bias = vdupq_n_u32(112u << 23);   // F32 bias 127 − F16 bias 15
    for (; i + 4 <= n; i += 4) {
        const float32x4_t av = vld1q_f32(a_f32 + i);
        const uint32x4_t u32x4 = vmovl_u16(vld1_u16(b_f16 + i)); // zero-extend u16 → u32
        const uint32x4_t sign = vshlq_n_u32(vandq_u32(u32x4, mask_sign), 16);
        const uint32x4_t exp16 = vshrq_n_u32(u32x4, 10);
        const uint32x4_t exp32 = vaddq_u32(vshlq_n_u32(exp16, 23), exp_bias);
        const uint32x4_t mant = vshlq_n_u32(vandq_u32(u32x4, mask_mant), 13);
        const float32x4_t bv = vreinterpretq_f32_u32(vorrq_u32(sign, vorrq_u32(exp32, mant)));
        acc = vfmaq_f32(acc, av, bv);
    }
    float s = vaddvq_f32(acc);
    for (; i < n; ++i)
        s += a_f32[i] * sapient::core::f16_bits_to_f32(b_f16[i]); // half::f16::from_bits().to_f32()
    return s;
}
float dot_f32_x_f16(const float* a, const uint16_t* b, size_t n) {
    return dot_f32_x_f16_neon(a, b, n);
}
#else
float dot_f32_x_f16(const float* a, const uint16_t* b, size_t n) {
    float s = -0.0f; // iter().zip().map().sum() seeds at -0.0
    for (size_t i = 0; i < n; ++i)
        s += a[i] * sapient::core::f16_bits_to_f32(b[i]);
    return s;
}
#endif

// ── float path of matmul_nt (matmul.rs:317-397) ──────────────────────────────
Result<Tensor> matmul_nt_float(const Tensor& x, const Tensor& w, size_t m, size_t k, size_t n) {
    // F16 GEMV decode: F16 weights widened per row inside NEON registers — no f32 copy of W.
    if (m == 1 && k >= 64 && w.dtype() == DType::F16) {
        const auto x_cow = x.to_f32_cow();
        const auto x_data = x_cow.get();
        const auto w_bytes = w.bytes();
        if (x_data.size() < k) sapient::core::panic("matmul_nt: x shorter than k");
        if (w_bytes.size() < 2 * n * k)
            sapient::core::panic("matmul_nt: F16 weight buffer shorter than n*k");
        // Rust: slice::from_raw_parts(bytes as *const u16, len/2) — F16 storage is packed
        // little-endian u16 (CpuBuffer alignment ≥ 2, F16 view offsets are multiples of 2).
        const auto* w_f16 = reinterpret_cast<const uint16_t*>(w_bytes.data());
        std::vector<float> out(n, 0.0f);
        const size_t chunk = detail::gemv_chunk(n);
        detail::for_each_out_chunk(out, chunk, [&](size_t chunk_idx, std::span<float> cs) {
            for (size_t local = 0; local < cs.size(); ++local) {
                const size_t j = chunk_idx * chunk + local;
                cs[local] = dot_f32_x_f16(x_data.data(), w_f16 + j * k, k);
            }
        });
        return Tensor::from_f32_vec(std::move(out), Shape{m, n});
    }

    const auto x_cow = x.to_f32_cow();
    const auto w_cow = w.to_f32_cow();
    const auto x_data = x_cow.get();
    const auto w_data = w_cow.get();
    if (x_data.size() < m * k || w_data.size() < n * k)
        sapient::core::panic("matmul_nt: operand shorter than its shape");
    std::vector<float> out(m * n, 0.0f);

    if (m == 1 && k >= 512) {
        // F32 GEMV decode — NEON/AVX2-vectorised dot products.
        const size_t chunk = detail::gemv_chunk(n);
        detail::for_each_out_chunk(out, chunk, [&](size_t chunk_idx, std::span<float> cs) {
            for (size_t local = 0; local < cs.size(); ++local) {
                const size_t j = chunk_idx * chunk + local;
                cs[local] = dot_f32_fast(x_data.data(), w_data.data() + j * k, k);
            }
        });
    } else {
        // Batched sgemm for prefill, split across X row blocks (each block an independent sgemm
        // over the same K reduction writing a disjoint output slice).
        const size_t flops = m * k * n;
        const size_t threads = std::max<size_t>(parallel::num_threads(), 1);
        const size_t mblock = (m >= 2 && flops >= (size_t{1} << 20))
                                  ? std::max<size_t>((m + threads - 1) / threads, 4)
                                  : m;
        parallel::par_chunks_mut(out, mblock * n, [&](size_t bi, std::span<float> out_block) {
            const size_t m0 = bi * mblock;
            const size_t mc = out_block.size() / n;
            sgemm(mc,
                  k,
                  n,
                  1.0f,
                  x_data.data() + m0 * k,
                  static_cast<std::ptrdiff_t>(k),
                  1,
                  w_data.data(),
                  1,
                  static_cast<std::ptrdiff_t>(k),
                  0.0f,
                  out_block.data(),
                  static_cast<std::ptrdiff_t>(n),
                  1);
        });
    }
    return Tensor::from_f32_vec(std::move(out), Shape{m, n});
}

// ── quantized arms (matmul.rs:538-1170) ──────────────────────────────────────

namespace quant = sapient::backends_cpu::kernels::quant;

// gemv_parallel!: n rows of a quantized GEMV, `dot(w_row_bytes, x_row)` per row, rows batched per
// task by gemv_chunk (matmul.rs:538-548).
template <class Dot>
void gemv_parallel(std::span<float> out_row,
                   size_t n,
                   size_t row_bytes,
                   std::span<const uint8_t> w_blocks,
                   std::span<const float> x_row,
                   Dot dot) {
    const size_t chunk = detail::gemv_chunk(n);
    detail::for_each_out_chunk(out_row, chunk, [&](size_t chunk_idx, std::span<float> cs) {
        for (size_t local = 0; local < cs.size(); ++local) {
            const size_t j = chunk_idx * chunk + local;
            cs[local] = dot(w_blocks.subspan(j * row_bytes, row_bytes), x_row);
        }
    });
}

// Rust's slice panics on `x_data[i*k..(i+1)*k]` / `w_blocks[j*row_bytes..]`, checked once.
void check_quant_operands(std::span<const float> x_data,
                          std::span<const uint8_t> w_blocks,
                          size_t m,
                          size_t k,
                          size_t n,
                          size_t row_bytes) {
    if (x_data.size() < m * k) sapient::core::panic("matmul_nt: x shorter than [m, k]");
    if (w_blocks.size() < n * row_bytes)
        sapient::core::panic("matmul_nt: quantized weight buffer shorter than [n, k]");
}

#if defined(__aarch64__) || defined(_M_ARM64)
// Per-32 int8 activations + their sums, or the Q8_K format — the `(x_i8, x_scales, x_sums)`
// triple every Q4_K SDOT path builds per activation row (matmul.rs:733-744, 811-815, 884-890).
quant::Q8kRow quantize_q4k_activations(std::span<const float> row, bool q8k) {
    if (q8k) return quant::quantize_row_to_q8k(row);
    auto q = quant::quantize_row_to_i8_blocks(row);
    auto sums = quant::i8_block_sums(q.q);
    return quant::Q8kRow{std::move(q.q), std::move(q.scales), std::move(sums)};
}
// The `(x_i8, x_scales)` pair of the Q6_K SDOT paths (Q8_K sums dropped) (matmul.rs:995-1003).
quant::I8Blocks quantize_q6k_activations(std::span<const float> row, bool q8k) {
    if (!q8k) return quant::quantize_row_to_i8_blocks(row);
    auto r = quant::quantize_row_to_q8k(row);
    return quant::I8Blocks{std::move(r.q), std::move(r.scales)};
}
#endif

Result<Tensor> matmul_nt_q4_0(const Tensor& x, const Tensor& w, size_t m, size_t k, size_t n) {
    if (k % quant::QK != 0)
        return tl::unexpected(
            Error::internal("Q4_0 matmul_nt: k must be a multiple of the block size (32)"));
    const auto x_cow = x.to_f32_cow();
    const auto x_data = x_cow.get();
    const auto w_blocks = w.quant_blocks();
    const size_t row_bytes = k / quant::QK * quant::Q4_0_BLOCK_BYTES;
    check_quant_operands(x_data, w_blocks, m, k, n, row_bytes);
    std::vector<float> out(m * n, 0.0f);
    for (size_t i = 0; i < m; ++i)
        gemv_parallel(std::span<float>(out).subspan(i * n, n),
                      n,
                      row_bytes,
                      w_blocks,
                      x_data.subspan(i * k, k),
                      quant::dot_q4_0_row_f32);
    return Tensor::from_f32_vec(std::move(out), Shape{m, n});
}

Result<Tensor> matmul_nt_q8_0(const Tensor& x, const Tensor& w, size_t m, size_t k, size_t n) {
    if (k % quant::QK != 0)
        return tl::unexpected(
            Error::internal("Q8_0 matmul_nt: k must be a multiple of the block size (32)"));
    const auto x_cow = x.to_f32_cow();
    const auto x_data = x_cow.get();
    const auto w_blocks = w.quant_blocks();
    const size_t row_bytes = k / quant::QK * quant::Q8_0_BLOCK_BYTES;
    check_quant_operands(x_data, w_blocks, m, k, n, row_bytes);
    std::vector<float> out(m * n, 0.0f);

#if defined(__aarch64__) || defined(_M_ARM64)
    // ── SDOT path (aarch64 dotprod): activations quantized to per-32 int8 ONCE per row ──
    if (cpu_features::has_dotprod()) {
        // Blocked W8A8 GEMM (m ≥ 8: prefill / vision towers): all rows quantize once, ONE parallel
        // region over weight-row chunks, output built [n, m] then flipped. Same kernel, same
        // scales as the per-row path → bit-identical (matmul.rs:608-650).
        if (m >= 8) {
            const size_t bpr = k / quant::QK; // activation blocks per row
            std::vector<int8_t> x_i8(m * k, 0);
            std::vector<float> x_scales(m * bpr, 0.0f);
            parallel::par_for(m, [&](size_t i) {
                const auto r = quant::quantize_row_to_i8_blocks(x_data.subspan(i * k, k));
                std::copy(
                    r.q.begin(), r.q.end(), x_i8.begin() + static_cast<std::ptrdiff_t>(i * k));
                std::copy(r.scales.begin(),
                          r.scales.end(),
                          x_scales.begin() + static_cast<std::ptrdiff_t>(i * bpr));
            });
            std::vector<float> out_t(n * m, 0.0f); // [n, m]
            const size_t wchunk = detail::gemv_chunk(n);
            parallel::par_chunks_mut(out_t, wchunk * m, [&](size_t ci, std::span<float> oc) {
                const size_t j0 = ci * wchunk;
                for (size_t jl = 0; jl * m < oc.size(); ++jl) { // oc.chunks_mut(m)
                    const size_t j = j0 + jl;
                    const auto wrow = w_blocks.subspan(j * row_bytes, row_bytes);
                    for (size_t i = 0; i < m; ++i)
                        oc[jl * m + i] = quant::dot_q8_0_row_sdot(
                            wrow,
                            std::span<const int8_t>(x_i8).subspan(i * k, k),
                            std::span<const float>(x_scales).subspan(i * bpr, bpr));
                }
            });
            // Transpose [n, m] → [m, n] (parallel over output rows).
            parallel::par_chunks_mut(out, n, [&](size_t i, std::span<float> orow) {
                for (size_t j = 0; j < orow.size(); ++j)
                    orow[j] = out_t[j * m + i];
            });
            return Tensor::from_f32_vec(std::move(out), Shape{m, n});
        }
        for (size_t i = 0; i < m; ++i) {
            // Per-block activation scales — a single per-row scale is destroyed by outlier
            // activation channels and yields incoherent output.
            const auto xq = quant::quantize_row_to_i8_blocks(x_data.subspan(i * k, k));
            const size_t chunk = detail::gemv_chunk(n);
            detail::for_each_out_chunk(
                std::span<float>(out).subspan(i * n, n),
                chunk,
                [&](size_t ci, std::span<float> cs) {
                    for (size_t local = 0; local < cs.size(); ++local) {
                        const size_t j = ci * chunk + local;
                        cs[local] = quant::dot_q8_0_row_sdot(
                            w_blocks.subspan(j * row_bytes, row_bytes), xq.q, xq.scales);
                    }
                });
        }
        return Tensor::from_f32_vec(std::move(out), Shape{m, n});
    }
#endif
    // ── Fallback: NEON widening or AVX2 ──
    for (size_t i = 0; i < m; ++i)
        gemv_parallel(std::span<float>(out).subspan(i * n, n),
                      n,
                      row_bytes,
                      w_blocks,
                      x_data.subspan(i * k, k),
                      quant::dot_q8_0_row_f32);
    return Tensor::from_f32_vec(std::move(out), Shape{m, n});
}

// Q4_K_R4 (row-interleaved) GEMV: weight rows come in groups of 4 whose super-blocks are
// block-interleaved into one contiguous stream (matmul.rs:711-860).
Result<Tensor> matmul_nt_q4_k_r4(const Tensor& x, const Tensor& w, size_t m, size_t k, size_t n) {
    if (k % 256 != 0 || n % 4 != 0)
        return tl::unexpected(
            Error::internal("Q4_K_R4: k must be a multiple of 256 and rows a multiple of 4"));
    const auto x_cow = x.to_f32_cow();
    const auto x_data = x_cow.get();
    const auto w_blocks = w.quant_blocks();
    const size_t row_bytes = k / 256 * quant::Q4_K_BLOCK_BYTES;
    check_quant_operands(x_data, w_blocks, m, k, n, row_bytes);
    std::vector<float> out(m * n, 0.0f);

#if defined(__aarch64__) || defined(_M_ARM64)
    const size_t group_bytes = 4 * row_bytes;
    // ── i8mm SMMLA prefill path (m ≥ 2, ARMv8.6): two activation rows per pass through each
    // weight group; output built group-major (transposed) so tasks own contiguous chunks ──
    if (m >= 2 && cpu_features::has_i8mm()) {
        const bool q8k = detail::q8k_activations();
        std::vector<quant::Q8kRow> quantized;
        quantized.reserve(m);
        for (size_t i = 0; i < m; ++i)
            quantized.push_back(quantize_q4k_activations(x_data.subspan(i * k, k), q8k));
        const size_t groups = n / 4;
        std::vector<float> out_t(n * m, 0.0f); // [group-rows][m]
        parallel::par_chunks_mut(out_t, 4 * m, [&](size_t g, std::span<float> chunk) {
            const auto group = w_blocks.subspan(g * group_bytes, group_bytes);
            size_t xi = 0;
            while (xi + 2 <= m) {
                const auto& a = quantized[xi];
                const auto& b = quantized[xi + 1];
                const auto v = q8k ? quant::dot_q4_k_4rows_r4_x2_q8k_smmla(
                                         group, a.q, a.scales, a.sums, b.q, b.scales, b.sums)
                                   : quant::dot_q4_k_4rows_r4_x2_smmla(
                                         group, a.q, a.scales, a.sums, b.q, b.scales, b.sums);
                for (size_t r = 0; r < 4; ++r) {
                    chunk[r * m + xi] = v[r][0];
                    chunk[r * m + xi + 1] = v[r][1];
                }
                xi += 2;
            }
            if (xi < m) {
                const auto& a = quantized[xi];
                const auto v = q8k ? quant::dot_q4_k_4rows_r4_q8k_neon(group, a.q, a.scales, a.sums)
                                   : quant::dot_q4_k_4rows_r4_neon(group, a.q, a.scales, a.sums);
                for (size_t r = 0; r < 4; ++r)
                    chunk[r * m + xi] = v[r];
            }
        });
        for (size_t g = 0; g < groups; ++g)
            for (size_t r = 0; r < 4; ++r)
                for (size_t i = 0; i < m; ++i)
                    out[i * n + g * 4 + r] = out_t[(g * 4 + r) * m + i];
        return Tensor::from_f32_vec(std::move(out), Shape{m, n});
    }

    // ── SDOT decode path: one contiguous stream per 4-row group ──
    if (cpu_features::has_dotprod()) {
        const bool q8k = detail::q8k_activations();
        for (size_t i = 0; i < m; ++i) {
            const auto xq = quantize_q4k_activations(x_data.subspan(i * k, k), q8k);
            const size_t gchunk = std::max<size_t>(detail::gemv_chunk(n) / 4, 1);
            detail::for_each_out_chunk(
                std::span<float>(out).subspan(i * n, n),
                gchunk * 4,
                [&](size_t ci, std::span<float> cs) {
                    const size_t g0 = ci * gchunk;
                    for (size_t gl = 0; gl * 4 < cs.size(); ++gl) { // cs.chunks_mut(4)
                        const size_t g = g0 + gl;
                        const auto group = w_blocks.subspan(g * group_bytes, group_bytes);
                        const auto v =
                            q8k ? quant::dot_q4_k_4rows_r4_q8k_neon(group, xq.q, xq.scales, xq.sums)
                                : quant::dot_q4_k_4rows_r4_neon(group, xq.q, xq.scales, xq.sums);
                        const auto slots =
                            cs.subspan(gl * 4, std::min<size_t>(4, cs.size() - gl * 4));
                        std::copy_n(v.begin(), slots.size(), slots.begin());
                    }
                });
        }
        return Tensor::from_f32_vec(std::move(out), Shape{m, n});
    }
#endif
    // Portable fallback (tests / x86): de-interleave each group's rows and use the scalar W4A8
    // dot — per-32 activations, never Q8_K (matmul.rs:841-859).
    for (size_t i = 0; i < m; ++i) {
        const auto xq = quant::quantize_row_to_i8_blocks(x_data.subspan(i * k, k));
        const auto x_sums = quant::i8_block_sums(xq.q);
        const size_t nb = k / 256;
        std::vector<uint8_t> row_buf(row_bytes, 0);
        for (size_t g = 0; g < n / 4; ++g)
            for (size_t r = 0; r < 4; ++r) {
                for (size_t b = 0; b < nb; ++b) {
                    const size_t src = (g * 4 * nb + b * 4 + r) * quant::Q4_K_BLOCK_BYTES;
                    std::copy_n(w_blocks.data() + src,
                                quant::Q4_K_BLOCK_BYTES,
                                row_buf.data() + b * quant::Q4_K_BLOCK_BYTES);
                }
                out[i * n + g * 4 + r] =
                    quant::dot_q4_k_row_q8_scalar(row_buf, xq.q, xq.scales, x_sums);
            }
    }
    return Tensor::from_f32_vec(std::move(out), Shape{m, n});
}

Result<Tensor> matmul_nt_q4_k(const Tensor& x, const Tensor& w, size_t m, size_t k, size_t n) {
    if (k % 256 != 0) return tl::unexpected(Error::internal("Q4_K: k must be a multiple of 256"));
    const auto x_cow = x.to_f32_cow();
    const auto x_data = x_cow.get();
    const auto w_blocks = w.quant_blocks();
    const size_t row_bytes = k / 256 * quant::Q4_K_BLOCK_BYTES;
    check_quant_operands(x_data, w_blocks, m, k, n, row_bytes);
    std::vector<float> out(m * n, 0.0f);

#if defined(__aarch64__) || defined(_M_ARM64)
    // ── W4A8 SDOT path: 4 weight rows share one pass over the activations; remainder rows use
    // the single-row kernel (per-row results bit-identical either way) (matmul.rs:879-929) ──
    if (cpu_features::has_dotprod()) {
        const bool q8k = detail::q8k_activations();
        for (size_t i = 0; i < m; ++i) {
            const auto xq = quantize_q4k_activations(x_data.subspan(i * k, k), q8k);
            const size_t chunk = detail::gemv_chunk(n);
            detail::for_each_out_chunk(
                std::span<float>(out).subspan(i * n, n),
                chunk,
                [&](size_t ci, std::span<float> cs) {
                    const size_t start = ci * chunk;
                    size_t local = 0;
                    while (local + 4 <= cs.size()) {
                        const size_t j = start + local;
                        const std::array<std::span<const uint8_t>, 4> rows = {
                            w_blocks.subspan(j * row_bytes, row_bytes),
                            w_blocks.subspan((j + 1) * row_bytes, row_bytes),
                            w_blocks.subspan((j + 2) * row_bytes, row_bytes),
                            w_blocks.subspan((j + 3) * row_bytes, row_bytes)};
                        const auto v =
                            q8k ? quant::dot_q4_k_4rows_q8k_neon(rows, xq.q, xq.scales, xq.sums)
                                : quant::dot_q4_k_4rows_q8_neon(rows, xq.q, xq.scales, xq.sums);
                        std::copy_n(v.begin(), 4, cs.begin() + static_cast<std::ptrdiff_t>(local));
                        local += 4;
                    }
                    for (; local < cs.size(); ++local) {
                        const size_t j = start + local;
                        const auto row = w_blocks.subspan(j * row_bytes, row_bytes);
                        cs[local] =
                            q8k ? quant::dot_q4_k_row_q8k_neon(row, xq.q, xq.scales, xq.sums)
                                : quant::dot_q4_k_row_q8_neon(row, xq.q, xq.scales, xq.sums);
                    }
                });
        }
        return Tensor::from_f32_vec(std::move(out), Shape{m, n});
    }
#endif
    for (size_t i = 0; i < m; ++i)
        gemv_parallel(std::span<float>(out).subspan(i * n, n),
                      n,
                      row_bytes,
                      w_blocks,
                      x_data.subspan(i * k, k),
                      quant::dot_q4_k_row_f32);
    return Tensor::from_f32_vec(std::move(out), Shape{m, n});
}

Result<Tensor> matmul_nt_q5_k(const Tensor& x, const Tensor& w, size_t m, size_t k, size_t n) {
    if (k % 256 != 0) return tl::unexpected(Error::internal("Q5_K: k must be a multiple of 256"));
    const auto x_cow = x.to_f32_cow();
    const auto x_data = x_cow.get();
    const auto w_blocks = w.quant_blocks();
    const size_t row_bytes = k / 256 * quant::Q5_K_BLOCK_BYTES;
    check_quant_operands(x_data, w_blocks, m, k, n, row_bytes);
    std::vector<float> out(m * n, 0.0f);
    // No SIMD/dotprod branch at all — Rust has none for Q5_K.
    for (size_t i = 0; i < m; ++i)
        gemv_parallel(std::span<float>(out).subspan(i * n, n),
                      n,
                      row_bytes,
                      w_blocks,
                      x_data.subspan(i * k, k),
                      quant::dot_q5_k_row_f32);
    return Tensor::from_f32_vec(std::move(out), Shape{m, n});
}

// Q6_K_R4 (row-interleaved) GEMV — same scheme as matmul_nt_q4_k_r4 (matmul.rs:971-1109).
Result<Tensor> matmul_nt_q6_k_r4(const Tensor& x, const Tensor& w, size_t m, size_t k, size_t n) {
    if (k % 256 != 0 || n % 4 != 0)
        return tl::unexpected(
            Error::internal("Q6_K_R4: k must be a multiple of 256 and rows a multiple of 4"));
    const auto x_cow = x.to_f32_cow();
    const auto x_data = x_cow.get();
    const auto w_blocks = w.quant_blocks();
    const size_t row_bytes = k / 256 * quant::Q6_K_BLOCK_BYTES;
    check_quant_operands(x_data, w_blocks, m, k, n, row_bytes);
    std::vector<float> out(m * n, 0.0f);

#if defined(__aarch64__) || defined(_M_ARM64)
    const size_t group_bytes = 4 * row_bytes;
    // ── i8mm SMMLA prefill path (m ≥ 2) ──
    if (m >= 2 && cpu_features::has_i8mm()) {
        const bool q8k = detail::q8k_activations();
        std::vector<quant::I8Blocks> quantized;
        quantized.reserve(m);
        for (size_t i = 0; i < m; ++i)
            quantized.push_back(quantize_q6k_activations(x_data.subspan(i * k, k), q8k));
        const size_t groups = n / 4;
        std::vector<float> out_t(n * m, 0.0f); // [group-rows][m]
        parallel::par_chunks_mut(out_t, 4 * m, [&](size_t g, std::span<float> chunk) {
            const auto group = w_blocks.subspan(g * group_bytes, group_bytes);
            size_t xi = 0;
            while (xi + 2 <= m) {
                const auto& a = quantized[xi];
                const auto& b = quantized[xi + 1];
                const auto v =
                    q8k ? quant::dot_q6_k_4rows_r4_x2_q8k_smmla(group, a.q, a.scales, b.q, b.scales)
                        : quant::dot_q6_k_4rows_r4_x2_smmla(group, a.q, a.scales, b.q, b.scales);
                for (size_t r = 0; r < 4; ++r) {
                    chunk[r * m + xi] = v[r][0];
                    chunk[r * m + xi + 1] = v[r][1];
                }
                xi += 2;
            }
            if (xi < m) {
                const auto& a = quantized[xi];
                const auto v = q8k ? quant::dot_q6_k_4rows_r4_q8k_neon(group, a.q, a.scales)
                                   : quant::dot_q6_k_4rows_r4_q8_neon(group, a.q, a.scales);
                for (size_t r = 0; r < 4; ++r)
                    chunk[r * m + xi] = v[r];
            }
        });
        for (size_t g = 0; g < groups; ++g)
            for (size_t r = 0; r < 4; ++r)
                for (size_t i = 0; i < m; ++i)
                    out[i * n + g * 4 + r] = out_t[(g * 4 + r) * m + i];
        return Tensor::from_f32_vec(std::move(out), Shape{m, n});
    }

    // ── NEON decode path: W6A8/Q8_K when dotprod is present, f32 activations otherwise ──
    {
        const bool dotprod = cpu_features::has_dotprod();
        for (size_t i = 0; i < m; ++i) {
            const auto x_row = x_data.subspan(i * k, k);
            const bool q8k = detail::q8k_activations();
            std::optional<quant::I8Blocks> quantized;
            if (dotprod) quantized = quantize_q6k_activations(x_row, q8k);
            const size_t gchunk = std::max<size_t>(detail::gemv_chunk(n) / 4, 1);
            detail::for_each_out_chunk(
                std::span<float>(out).subspan(i * n, n),
                gchunk * 4,
                [&](size_t ci, std::span<float> cs) {
                    const size_t g0 = ci * gchunk;
                    for (size_t gl = 0; gl * 4 < cs.size(); ++gl) {
                        const size_t g = g0 + gl;
                        const auto group = w_blocks.subspan(g * group_bytes, group_bytes);
                        std::array<float, 4> v{};
                        if (quantized.has_value())
                            v = q8k ? quant::dot_q6_k_4rows_r4_q8k_neon(
                                          group, quantized->q, quantized->scales)
                                    : quant::dot_q6_k_4rows_r4_q8_neon(
                                          group, quantized->q, quantized->scales);
                        else
                            v = quant::dot_q6_k_4rows_r4_neon(group, x_row); // f32 activations
                        const auto slots =
                            cs.subspan(gl * 4, std::min<size_t>(4, cs.size() - gl * 4));
                        std::copy_n(v.begin(), slots.size(), slots.begin());
                    }
                });
        }
        return Tensor::from_f32_vec(std::move(out), Shape{m, n});
    }
#else
    // Portable fallback (Rust's #[allow(unreachable_code)] block): de-interleave each group and
    // use the f32 dot (matmul.rs:1090-1108).
    for (size_t i = 0; i < m; ++i) {
        const auto x_row = x_data.subspan(i * k, k);
        const size_t nb = k / 256;
        std::vector<uint8_t> row_buf(row_bytes, 0);
        for (size_t g = 0; g < n / 4; ++g)
            for (size_t r = 0; r < 4; ++r) {
                for (size_t b = 0; b < nb; ++b) {
                    const size_t src = (g * 4 * nb + b * 4 + r) * quant::Q6_K_BLOCK_BYTES;
                    std::copy_n(w_blocks.data() + src,
                                quant::Q6_K_BLOCK_BYTES,
                                row_buf.data() + b * quant::Q6_K_BLOCK_BYTES);
                }
                out[i * n + g * 4 + r] = quant::dot_q6_k_row_f32(row_buf, x_row);
            }
    }
    return Tensor::from_f32_vec(std::move(out), Shape{m, n});
#endif
}

Result<Tensor> matmul_nt_q6_k(const Tensor& x, const Tensor& w, size_t m, size_t k, size_t n) {
    if (k % 256 != 0) return tl::unexpected(Error::internal("Q6_K: k must be a multiple of 256"));
    const auto x_cow = x.to_f32_cow();
    const auto x_data = x_cow.get();
    const auto w_blocks = w.quant_blocks();
    const size_t row_bytes = k / 256 * quant::Q6_K_BLOCK_BYTES;
    check_quant_operands(x_data, w_blocks, m, k, n, row_bytes);
    std::vector<float> out(m * n, 0.0f);

#if defined(__aarch64__) || defined(_M_ARM64)
    // ── W6A8 SDOT path: one `sdot` per 16-element scale group (matmul.rs:1126-1153) ──
    if (cpu_features::has_dotprod()) {
        const bool q8k = detail::q8k_activations();
        for (size_t i = 0; i < m; ++i) {
            const auto xq = quantize_q6k_activations(x_data.subspan(i * k, k), q8k);
            const size_t chunk = detail::gemv_chunk(n);
            detail::for_each_out_chunk(
                std::span<float>(out).subspan(i * n, n),
                chunk,
                [&](size_t ci, std::span<float> cs) {
                    for (size_t local = 0; local < cs.size(); ++local) {
                        const size_t j = ci * chunk + local;
                        const auto row = w_blocks.subspan(j * row_bytes, row_bytes);
                        cs[local] = q8k ? quant::dot_q6_k_row_q8k_neon(row, xq.q, xq.scales)
                                        : quant::dot_q6_k_row_q8_neon(row, xq.q, xq.scales);
                    }
                });
        }
        return Tensor::from_f32_vec(std::move(out), Shape{m, n});
    }
#endif
    for (size_t i = 0; i < m; ++i)
        gemv_parallel(std::span<float>(out).subspan(i * n, n),
                      n,
                      row_bytes,
                      w_blocks,
                      x_data.subspan(i * k, k),
                      quant::dot_q6_k_row_f32);
    return Tensor::from_f32_vec(std::move(out), Shape{m, n});
}

} // namespace

namespace detail {

size_t gemv_chunk(size_t n) {
    // The governed comparison is within RAYON's domain (the governor sheds rayon cores; the spin
    // pool is disabled entirely while governed) — matmul.rs:416-426.
    const size_t rayon_n = std::max<size_t>(parallel::num_threads(), 1);
    const size_t eff = thermal::effective_threads();
    if (eff < rayon_n)
        return std::max<size_t>(n / std::max<size_t>(eff, 1), 16); // governed: no ×4, no 512 cap
    const size_t ncpus = spinpool::enabled() ? spinpool::parallelism() : rayon_n;
    const std::optional<size_t> tpc = env_usize("SAPIENT_GEMV_TPC"); // read every call, like Rust
    if (tpc.has_value() && *tpc >= 1) return std::max<size_t>(n / (ncpus * *tpc), 16);
    return std::clamp<size_t>(n / (ncpus * 4), 16, 512);
}

void for_each_out_chunk(std::span<float> out,
                        size_t chunk,
                        const std::function<void(size_t, std::span<float>)>& f) {
    if (out.empty()) return;
    if (chunk == 0) sapient::core::panic("for_each_out_chunk: chunk size must not be zero");

    // SAPIENT_SPINPOOL_DEBUG=1: periodic dispatch-route census on stderr (matmul.rs:493-516).
    // Read via getenv on EVERY call, like Rust's uncached `std::env::var(..).is_ok()`.
    if (std::getenv("SAPIENT_SPINPOOL_DEBUG") != nullptr) {
        static std::atomic<uint64_t> spin_dispatches{0};
        static std::atomic<uint64_t> pool_dispatches{0}; // the `RAYON` counter's twin
        uint64_t s = 0;
        uint64_t r = 0;
        if (spinpool::enabled()) {
            s = spin_dispatches.fetch_add(1, std::memory_order_relaxed) + 1;
            r = pool_dispatches.load(std::memory_order_relaxed);
        } else {
            s = spin_dispatches.load(std::memory_order_relaxed);
            r = pool_dispatches.fetch_add(1, std::memory_order_relaxed) + 1;
        }
        if ((s + r) % 2000 == 0) {
            std::fprintf(stderr,
                         "[spinpool-debug] spin=%llu rayon=%llu chunk=%zu len=%zu n_chunks=%zu\n",
                         static_cast<unsigned long long>(s),
                         static_cast<unsigned long long>(r),
                         chunk,
                         out.size(),
                         (out.size() + chunk - 1) / chunk);
        }
    }

    if (spinpool::enabled()) {
        // Rust's SyncPtr: chunk geometry guarantees the spans built from `base` are disjoint, the
        // same contract par_chunks_mut relies on, and `out` outlives `run` (it blocks until every
        // chunk completes). The partition below is character-for-character the one in
        // parallel::par_chunks_mut — that identity is what makes the two routes bit-identical.
        const size_t len = out.size();
        const size_t n_chunks = (len + chunk - 1) / chunk;
        float* const base = out.data();
        spinpool::pool().run(n_chunks, [&](size_t ci) {
            const size_t start = ci * chunk;
            const size_t end = std::min(start + chunk, len);
            // A throwing callback would unwind through the pool's op slot (or std::terminate on a
            // worker); abort cleanly instead, exactly as parallel::invoke does on the other route.
            try {
                f(ci, std::span<float>(base + start, end - start));
            } catch (...) {
                sapient::core::panic("for_each_out_chunk: a callback threw an exception "
                                     "(callbacks must not throw)");
            }
        });
        return;
    }
    parallel::par_chunks_mut(out, chunk, f);
}

#if defined(__aarch64__) || defined(_M_ARM64)
bool q8k_activations() {
    // OnceLock twin: SAPIENT_Q8K_ACT read once; `v != "0"`, unset → true (matmul.rs:458-470).
    static const bool on = [] {
        const char* v = std::getenv("SAPIENT_Q8K_ACT");
        return v == nullptr || std::string_view(v) != "0";
    }();
    return on;
}
#endif

} // namespace detail

// ── matmul (matmul.rs:22-100) ────────────────────────────────────────────────
Result<Tensor> matmul(const Tensor& a, const Tensor& b) {
    const Shape& as = a.shape();
    const Shape& bs = b.shape();
    if (as.ndim() < 2 || bs.ndim() < 2)
        return tl::unexpected(Error::rank_mismatch(2, std::min(as.ndim(), bs.ndim())));
    const size_t a_rank = as.ndim();
    const size_t b_rank = bs.ndim();
    const size_t m = as.dims[a_rank - 2];
    const size_t k = as.dims[a_rank - 1];
    const size_t k2 = bs.dims[b_rank - 2];
    const size_t n = bs.dims[b_rank - 1];
    if (k != k2) return tl::unexpected(Error::shape_mismatch({m, k, n}, {m, k2, n}));

    size_t batch = 1;
    for (size_t i = 0; i + 2 < a_rank; ++i)
        batch *= as.dims[i];

    const auto a_cow = a.to_f32_cow();
    const auto a_data = a_cow.get();
    const auto b_cow = b.to_f32_cow();
    const auto b_data = b_cow.get();
    const size_t a_stride = m * k;
    const size_t b_stride = k * n;
    const size_t c_stride = m * n;
    if (a_data.size() < batch * a_stride || b_data.size() < batch * b_stride)
        sapient::core::panic(
            "matmul: operand data shorter than the batched shape"); // Rust: slice panic
    std::vector<float> out_data(batch * c_stride, 0.0f);
    for (size_t bi = 0; bi < batch; ++bi)
        sgemm(m,
              k,
              n,
              1.0f,
              a_data.data() + bi * a_stride,
              static_cast<std::ptrdiff_t>(k),
              1,
              b_data.data() + bi * b_stride,
              static_cast<std::ptrdiff_t>(n),
              1,
              0.0f,
              out_data.data() + bi * c_stride,
              static_cast<std::ptrdiff_t>(n),
              1);

    std::vector<size_t> out_dims(as.dims.begin(),
                                 as.dims.begin() + static_cast<std::ptrdiff_t>(a_rank - 2));
    out_dims.push_back(m);
    out_dims.push_back(n);
    return Tensor::from_f32_vec(std::move(out_data), Shape(out_dims));
}

// ── matmul_nt dispatcher (matmul.rs:114-145) ─────────────────────────────────
Result<Tensor> matmul_nt(const Tensor& x, const Tensor& w) {
    const auto& xd = x.shape().dims;
    const auto& wd = w.shape().dims;
    if (xd.size() != 2 || wd.size() != 2)
        return tl::unexpected(Error::internal("matmul_nt expects 2-D tensors"));
    const size_t m = xd[0], k = xd[1];
    const size_t n = wd[0], k2 = wd[1];
    if (k != k2) return tl::unexpected(Error::shape_mismatch({m, k}, {n, k2}));

    // Thermal governor sample point (rate-limited inside tick; plan E gives it a body).
    thermal::tick();

    switch (w.dtype()) {
    case DType::Q4_0:
        return matmul_nt_q4_0(x, w, m, k, n);
    case DType::Q8_0:
        return matmul_nt_q8_0(x, w, m, k, n);
    case DType::Q4_K:
        return matmul_nt_q4_k(x, w, m, k, n);
    case DType::Q4_K_R4:
        return matmul_nt_q4_k_r4(x, w, m, k, n);
    case DType::Q5_K:
        return matmul_nt_q5_k(x, w, m, k, n);
    case DType::Q6_K:
        return matmul_nt_q6_k(x, w, m, k, n);
    case DType::Q6_K_R4:
        return matmul_nt_q6_k_r4(x, w, m, k, n);
    default:
        return matmul_nt_float(x, w, m, k, n);
    }
}

// ── gemm (matmul.rs:1173-1256) ───────────────────────────────────────────────
Result<Tensor> gemm(const Tensor& a,
                    const Tensor& b,
                    const Tensor* bias,
                    float alpha,
                    float beta,
                    bool trans_a,
                    bool trans_b) {
    Tensor a2 = a;
    if (trans_a) {
        SAPIENT_TRY_ASSIGN(a2, a.t());
    }
    Tensor b2 = b;
    if (trans_b) {
        SAPIENT_TRY_ASSIGN(b2, b.t());
    }

    if (a2.ndim() < 2 || b2.ndim() < 2)
        sapient::core::panic("gemm: operands must be 2-D"); // Rust: dims()[1] panic
    const size_t m = a2.shape().dims[0];
    const size_t k = a2.shape().dims[1];
    const size_t k2 = b2.shape().dims[0];
    const size_t n = b2.shape().dims[1];
    if (k != k2) return tl::unexpected(Error::shape_mismatch({m, k}, {k2, n}));

    // A transposed view's data is the raw buffer; its strides do the transpose (as in Rust).
    const auto a_cow = a2.to_f32_cow();
    const auto a_data = a_cow.get();
    const auto b_cow = b2.to_f32_cow();
    const auto b_data = b_cow.get();
    const auto a_strides = a2.strides();
    const auto b_strides = b2.strides();
    std::vector<float> out(m * n, 0.0f);
    sgemm(m,
          k,
          n,
          alpha,
          a_data.data(),
          static_cast<std::ptrdiff_t>(a_strides[0]),
          static_cast<std::ptrdiff_t>(a_strides[1]),
          b_data.data(),
          static_cast<std::ptrdiff_t>(b_strides[0]),
          static_cast<std::ptrdiff_t>(b_strides[1]),
          0.0f,
          out.data(),
          static_cast<std::ptrdiff_t>(n),
          1);

    if (bias != nullptr) {
        const auto bias_data = bias->f32_slice(); // as_f32_slice: panics unless F32
        const size_t b_len = bias_data.size();
        if (b_len != n && b_len != 1) return tl::unexpected(Error::shape_mismatch({n}, {b_len}));
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j) {
                const float bv = b_len == 1 ? bias_data[0] : bias_data[j];
                out[i * n + j] += beta * bv;
            }
    }
    return Tensor::from_f32_vec(std::move(out), Shape{m, n});
}

} // namespace sapient::backends_cpu::kernels::matmul
