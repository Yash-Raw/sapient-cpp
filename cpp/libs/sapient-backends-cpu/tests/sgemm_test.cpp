// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "sapient/backends_cpu/sgemm.hpp"

using sapient::backends_cpu::sgemm;

namespace {

struct Lcg {
    uint64_t s;
    float next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return (static_cast<float>(s >> 40) / static_cast<float>(1ULL << 24)) * 2.0f - 1.0f;
    }
};

std::vector<float> rand_vec(size_t n, uint64_t seed) {
    Lcg g{seed};
    std::vector<float> v(n);
    for (float& x : v)
        x = g.next();
    return v;
}

// Reference in double: C = alpha·A·B + beta·C with explicit strides.
std::vector<float> naive(size_t m,
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
                         const std::vector<float>& c_in) {
    std::vector<float> out(m * n, 0.0f);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j) {
            double acc = 0.0;
            for (size_t p = 0; p < k; ++p)
                acc += static_cast<double>(a[static_cast<std::ptrdiff_t>(i) * rsa +
                                             static_cast<std::ptrdiff_t>(p) * csa]) *
                       static_cast<double>(b[static_cast<std::ptrdiff_t>(p) * rsb +
                                             static_cast<std::ptrdiff_t>(j) * csb]);
            const double prev = beta == 0.0f ? 0.0 : static_cast<double>(beta) * c_in[i * n + j];
            out[i * n + j] = static_cast<float>(static_cast<double>(alpha) * acc + prev);
        }
    return out;
}

float max_abs(const std::vector<float>& v) {
    float m = 0.0f;
    for (float x : v)
        m = std::max(m, std::fabs(x));
    return m;
}

void expect_close(const std::vector<float>& got, const std::vector<float>& ref, float rel) {
    ASSERT_EQ(got.size(), ref.size());
    const float tol = rel * std::max(1.0f, max_abs(ref));
    for (size_t i = 0; i < got.size(); ++i)
        ASSERT_NEAR(got[i], ref[i], tol) << "index " << i;
}

} // namespace

// C++-only tests (matrixmultiply has none we could port). Tolerance 1e-4·max(1, max|ref|) with k ≤ 300
// and |values| ≤ 1: a float re-ordering error is ~1e-6, an indexing/stride bug is O(1).
TEST(Sgemm, matches_naive_reference_over_shapes) {
    const size_t shapes[][3] = {
        {1, 1, 1}, {3, 5, 7}, {17, 300, 9}, {65, 257, 1030}, {2, 64, 16}, {4, 3, 1}};
    uint64_t seed = 1;
    for (const auto& s : shapes) {
        const size_t m = s[0], k = s[1], n = s[2];
        const auto a = rand_vec(m * k, seed++);
        const auto b = rand_vec(k * n, seed++);
        std::vector<float> c(m * n, 0.0f);
        sgemm(m,
              k,
              n,
              1.0f,
              a.data(),
              static_cast<std::ptrdiff_t>(k),
              1,
              b.data(),
              static_cast<std::ptrdiff_t>(n),
              1,
              0.0f,
              c.data(),
              static_cast<std::ptrdiff_t>(n),
              1);
        const auto ref = naive(m,
                               k,
                               n,
                               1.0f,
                               a.data(),
                               static_cast<std::ptrdiff_t>(k),
                               1,
                               b.data(),
                               static_cast<std::ptrdiff_t>(n),
                               1,
                               0.0f,
                               c);
        expect_close(c, ref, 1e-4f);
    }
}

// The four Rust call-site patterns: matmul (k,1)/(n,1); matmul_nt prefill B transposed via (1,k);
// gemm with arbitrary strides (here A transposed via (1,m)); conv2d (k,1)/(n2,1).
TEST(Sgemm, transposed_operands_via_strides) {
    const size_t m = 6, k = 70, n = 11;
    const auto at = rand_vec(k * m, 7); // A(i,p) = at[p*m + i]  → rsa = 1, csa = m
    const auto bt = rand_vec(n * k, 8); // B(p,j) = bt[j*k + p]  → rsb = 1, csb = k
    std::vector<float> c(m * n, 0.0f);
    sgemm(m,
          k,
          n,
          1.0f,
          at.data(),
          1,
          static_cast<std::ptrdiff_t>(m),
          bt.data(),
          1,
          static_cast<std::ptrdiff_t>(k),
          0.0f,
          c.data(),
          static_cast<std::ptrdiff_t>(n),
          1);
    const auto ref = naive(m,
                           k,
                           n,
                           1.0f,
                           at.data(),
                           1,
                           static_cast<std::ptrdiff_t>(m),
                           bt.data(),
                           1,
                           static_cast<std::ptrdiff_t>(k),
                           0.0f,
                           c);
    expect_close(c, ref, 1e-4f);

    // Output with a column stride (csc = 2, interleaved into a 2× buffer).
    std::vector<float> c2(m * n * 2, 0.0f);
    sgemm(m,
          k,
          n,
          1.0f,
          at.data(),
          1,
          static_cast<std::ptrdiff_t>(m),
          bt.data(),
          1,
          static_cast<std::ptrdiff_t>(k),
          0.0f,
          c2.data(),
          static_cast<std::ptrdiff_t>(2 * n),
          2);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            EXPECT_EQ(std::bit_cast<uint32_t>(c2[i * 2 * n + 2 * j]),
                      std::bit_cast<uint32_t>(c[i * n + j]));
}

// matmul_nt_float and conv2d split rows into thread-count-sized blocks; the result must not
// depend on the blocking (or RAYON_NUM_THREADS would change the model's numbers).
TEST(Sgemm, row_and_column_blocks_are_bit_identical_to_the_full_call) {
    const size_t m = 37, k = 300, n = 53;
    const auto a = rand_vec(m * k, 21);
    const auto b = rand_vec(k * n, 22);
    std::vector<float> full(m * n, 0.0f);
    sgemm(m,
          k,
          n,
          1.0f,
          a.data(),
          static_cast<std::ptrdiff_t>(k),
          1,
          b.data(),
          static_cast<std::ptrdiff_t>(n),
          1,
          0.0f,
          full.data(),
          static_cast<std::ptrdiff_t>(n),
          1);

    std::vector<float> rows(m * n, 0.0f);
    const size_t mblock = 8;
    for (size_t m0 = 0; m0 < m; m0 += mblock) {
        const size_t mc = std::min(mblock, m - m0);
        sgemm(mc,
              k,
              n,
              1.0f,
              a.data() + m0 * k,
              static_cast<std::ptrdiff_t>(k),
              1,
              b.data(),
              static_cast<std::ptrdiff_t>(n),
              1,
              0.0f,
              rows.data() + m0 * n,
              static_cast<std::ptrdiff_t>(n),
              1);
    }
    std::vector<float> cols(m * n, 0.0f);
    const size_t split = 20;
    sgemm(m,
          k,
          split,
          1.0f,
          a.data(),
          static_cast<std::ptrdiff_t>(k),
          1,
          b.data(),
          static_cast<std::ptrdiff_t>(n),
          1,
          0.0f,
          cols.data(),
          static_cast<std::ptrdiff_t>(n),
          1);
    sgemm(m,
          k,
          n - split,
          1.0f,
          a.data(),
          static_cast<std::ptrdiff_t>(k),
          1,
          b.data() + split,
          static_cast<std::ptrdiff_t>(n),
          1,
          0.0f,
          cols.data() + split,
          static_cast<std::ptrdiff_t>(n),
          1);
    for (size_t i = 0; i < m * n; ++i) {
        EXPECT_EQ(std::bit_cast<uint32_t>(rows[i]), std::bit_cast<uint32_t>(full[i]))
            << "row-blocked " << i;
        EXPECT_EQ(std::bit_cast<uint32_t>(cols[i]), std::bit_cast<uint32_t>(full[i]))
            << "col-blocked " << i;
    }
}

// matrixmultiply: beta == 0 means C is write-only (a NaN-filled C must not leak in).
TEST(Sgemm, beta_zero_never_reads_c) {
    const size_t m = 3, k = 9, n = 4;
    const auto a = rand_vec(m * k, 31);
    const auto b = rand_vec(k * n, 32);
    std::vector<float> c(m * n, std::numeric_limits<float>::quiet_NaN());
    sgemm(m,
          k,
          n,
          1.0f,
          a.data(),
          static_cast<std::ptrdiff_t>(k),
          1,
          b.data(),
          static_cast<std::ptrdiff_t>(n),
          1,
          0.0f,
          c.data(),
          static_cast<std::ptrdiff_t>(n),
          1);
    const auto ref = naive(m,
                           k,
                           n,
                           1.0f,
                           a.data(),
                           static_cast<std::ptrdiff_t>(k),
                           1,
                           b.data(),
                           static_cast<std::ptrdiff_t>(n),
                           1,
                           0.0f,
                           c);
    for (float v : c)
        EXPECT_TRUE(std::isfinite(v));
    expect_close(c, ref, 1e-4f);
}

TEST(Sgemm, alpha_and_beta_scale) {
    const size_t m = 5, k = 33, n = 6;
    const auto a = rand_vec(m * k, 41);
    const auto b = rand_vec(k * n, 42);
    const auto c0 = rand_vec(m * n, 43);
    auto c = c0;
    sgemm(m,
          k,
          n,
          2.0f,
          a.data(),
          static_cast<std::ptrdiff_t>(k),
          1,
          b.data(),
          static_cast<std::ptrdiff_t>(n),
          1,
          0.5f,
          c.data(),
          static_cast<std::ptrdiff_t>(n),
          1);
    const auto ref = naive(m,
                           k,
                           n,
                           2.0f,
                           a.data(),
                           static_cast<std::ptrdiff_t>(k),
                           1,
                           b.data(),
                           static_cast<std::ptrdiff_t>(n),
                           1,
                           0.5f,
                           c0);
    expect_close(c, ref, 1e-4f);
}

TEST(Sgemm, zero_k_or_zero_alpha_only_applies_beta) {
    const size_t m = 2, n = 3;
    const auto c0 = rand_vec(m * n, 51);
    const auto a = rand_vec(m * 4, 52);
    const auto b = rand_vec(4 * n, 53);
    auto c = c0;
    sgemm(m,
          0,
          n,
          1.0f,
          a.data(),
          4,
          1,
          b.data(),
          static_cast<std::ptrdiff_t>(n),
          1,
          0.5f,
          c.data(),
          static_cast<std::ptrdiff_t>(n),
          1);
    for (size_t i = 0; i < m * n; ++i)
        EXPECT_EQ(c[i], 0.5f * c0[i]);
    c = c0;
    sgemm(m,
          4,
          n,
          0.0f,
          a.data(),
          4,
          1,
          b.data(),
          static_cast<std::ptrdiff_t>(n),
          1,
          0.0f,
          c.data(),
          static_cast<std::ptrdiff_t>(n),
          1);
    for (float v : c)
        EXPECT_EQ(v, 0.0f);
}
