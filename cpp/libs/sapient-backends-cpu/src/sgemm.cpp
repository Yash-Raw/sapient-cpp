// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/backends_cpu/sgemm.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

#include "sapient/backends_cpu/cpu_features.hpp"
#endif

namespace sapient::backends_cpu {
namespace {

constexpr size_t KC = 256;  // k panel: the fixed p-order that makes results blocking-independent
constexpr size_t MC = 64;   // A rows packed per block
constexpr size_t NC = 1024; // B columns packed per panel

inline std::ptrdiff_t idx(size_t i, size_t j, std::ptrdiff_t rs, std::ptrdiff_t cs) {
    return static_cast<std::ptrdiff_t>(i) * rs + static_cast<std::ptrdiff_t>(j) * cs;
}

// acc[q] += Σ_p a[p] · bp[4p + q] for q in 0..4, where `bp` is one packed kc×4 strip of B.
// FMA where the ISA has it (single rounding per step), plain mul+add otherwise; the scalar
// fallback is only reached on hosts with neither NEON nor AVX2+FMA.
#if defined(__aarch64__) || defined(_M_ARM64)
void strip4(size_t kc, const float* a, const float* bp, float* acc) {
    float32x4_t v = vld1q_f32(acc);
    for (size_t p = 0; p < kc; ++p)
        v = vfmaq_f32(v, vld1q_f32(bp + 4 * p), vdupq_n_f32(a[p]));
    vst1q_f32(acc, v);
}
#else
void strip4_scalar(size_t kc, const float* a, const float* bp, float* acc) {
    for (size_t p = 0; p < kc; ++p)
        for (size_t q = 0; q < 4; ++q)
            acc[q] = acc[q] + a[p] * bp[4 * p + q];
}
#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx2,fma"))) void
strip4_fma(size_t kc, const float* a, const float* bp, float* acc) {
    __m128 v = _mm_loadu_ps(acc);
    for (size_t p = 0; p < kc; ++p)
        v = _mm_fmadd_ps(_mm_loadu_ps(bp + 4 * p), _mm_set1_ps(a[p]), v);
    _mm_storeu_ps(acc, v);
}
void strip4(size_t kc, const float* a, const float* bp, float* acc) {
    if (cpu_features::has_avx2_fma())
        strip4_fma(kc, a, bp, acc);
    else
        strip4_scalar(kc, a, bp, acc);
}
#else
void strip4(size_t kc, const float* a, const float* bp, float* acc) {
    strip4_scalar(kc, a, bp, acc);
}
#endif
#endif

} // namespace

void sgemm(size_t m,
           size_t k,
           size_t n,
           float alpha,
           const float* a,
           std::ptrdiff_t rsa,
           std::ptrdiff_t csa,
           const float* b,
           std::ptrdiff_t rsb,
           std::ptrdiff_t csb,
           float beta,
           float* c,
           std::ptrdiff_t rsc,
           std::ptrdiff_t csc) {
    if (m == 0 || n == 0) return;
    // beta pass: matrixmultiply never reads C when beta == 0.
    if (beta == 0.0f) {
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                c[idx(i, j, rsc, csc)] = 0.0f;
    } else if (beta != 1.0f) {
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                c[idx(i, j, rsc, csc)] *= beta;
    }
    if (k == 0 || alpha == 0.0f) return;

    std::vector<float> bp;
    std::vector<float> ap;
    for (size_t j0 = 0; j0 < n; j0 += NC) {
        const size_t nc = std::min(NC, n - j0);
        const size_t strips = (nc + 3) / 4;
        for (size_t p0 = 0; p0 < k; p0 += KC) {
            const size_t kc = std::min(KC, k - p0);
            // Pack B[p0..p0+kc, j0..j0+nc) as `strips` contiguous kc×4 strips, zero-padded past nc so
            // every column — including the ragged last strip — sees identical arithmetic.
            bp.assign(strips * kc * 4, 0.0f);
            for (size_t s = 0; s < strips; ++s)
                for (size_t p = 0; p < kc; ++p)
                    for (size_t q = 0; q < 4; ++q) {
                        const size_t j = 4 * s + q;
                        if (j < nc) bp[(s * kc + p) * 4 + q] = b[idx(p0 + p, j0 + j, rsb, csb)];
                    }
            for (size_t i0 = 0; i0 < m; i0 += MC) {
                const size_t mc = std::min(MC, m - i0);
                ap.resize(mc * kc);
                for (size_t i = 0; i < mc; ++i)
                    for (size_t p = 0; p < kc; ++p)
                        ap[i * kc + p] = a[idx(i0 + i, p0 + p, rsa, csa)];
                for (size_t i = 0; i < mc; ++i)
                    for (size_t s = 0; s < strips; ++s) {
                        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        strip4(kc, ap.data() + i * kc, bp.data() + s * kc * 4, acc);
                        for (size_t q = 0; q < 4; ++q) {
                            const size_t j = 4 * s + q;
                            if (j < nc) c[idx(i0 + i, j0 + j, rsc, csc)] += alpha * acc[q];
                        }
                    }
            }
        }
    }
}

} // namespace sapient::backends_cpu
