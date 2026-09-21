// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/backends_cpu/kernels/matmul.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "sapient/backends_cpu/cpu_features.hpp"
#include "sapient/backends_cpu/env.hpp"
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
    // PLAN E inserts here: the SAPIENT_SPINPOOL_DEBUG census and
    // `if (spinpool::enabled()) { spinpool::pool().run(n_chunks, …); return; }` — same partition.
    parallel::par_chunks_mut(out, chunk, f);
}

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
    case DType::Q8_0:
    case DType::Q4_K:
    case DType::Q4_K_R4:
    case DType::Q5_K:
    case DType::Q6_K:
    case DType::Q6_K_R4:
        // PLAN D: matmul_nt_q4_0 / q8_0 / q4_k / q4_k_r4 / q5_k / q6_k / q6_k_r4 replace this arm.
        return tl::unexpected(Error::internal("matmul_nt: quantized weights (" +
                                              sapient::core::to_string(w.dtype()) +
                                              ") land in plan D"));
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
