// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "sapient/backends_cpu/kernels/matmul.hpp"
#include "sapient/backends_cpu/parallel.hpp"
#include "sapient/core/dtype.hpp"
#include "sapient/core/f16.hpp"
#include "sapient/core/tensor.hpp"

using namespace sapient::backends_cpu::kernels::matmul;
using sapient::core::DType;
using sapient::core::Shape;
using sapient::core::Tensor;

namespace {
Tensor f32(std::vector<float> data, Shape shape) {
    auto r = Tensor::from_f32_vec(std::move(data), std::move(shape));
    if (!r) throw std::runtime_error(r.error().to_string());
    return std::move(*r);
}
std::vector<float> lcg(size_t n, uint64_t seed, float lo, float hi) {
    std::vector<float> v(n);
    for (float& x : v) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        x = lo + (hi - lo) * (static_cast<float>(seed >> 40) / static_cast<float>(1ULL << 24));
    }
    return v;
}
} // namespace

TEST(Matmul, matmul_2x2) {
    // [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]]
    const auto a = f32({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2});
    const auto b = f32({5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2});
    auto c = matmul(a, b);
    ASSERT_TRUE(c.has_value()) << c.error().to_string();
    const auto d = c->f32_slice();
    EXPECT_LT(std::fabs(d[0] - 19.0f), 1e-5f);
    EXPECT_LT(std::fabs(d[1] - 22.0f), 1e-5f);
    EXPECT_LT(std::fabs(d[2] - 43.0f), 1e-5f);
    EXPECT_LT(std::fabs(d[3] - 50.0f), 1e-5f);
}

TEST(Matmul, matmul_nt_linear) {
    // x = [1,2] (1x2); W = [[1,2],[3,4],[5,6]] shape [3,2]; y = x @ Wᵀ = [5, 11, 17].
    const auto x = f32({1.0f, 2.0f}, Shape{1, 2});
    const auto w = f32({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{3, 2});
    auto y = matmul_nt(x, w);
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    const auto d = y->f32_slice();
    EXPECT_EQ(y->shape().dims, (std::vector<size_t>{1, 3}));
    EXPECT_LT(std::fabs(d[0] - 5.0f), 1e-5f);
    EXPECT_LT(std::fabs(d[1] - 11.0f), 1e-5f);
    EXPECT_LT(std::fabs(d[2] - 17.0f), 1e-5f);
}

TEST(Matmul, matmul_nt_linear_f16_weight) {
    // Same as above but W is stored as F16 — must still be correct.
    const auto x = f32({1.0f, 2.0f}, Shape{1, 2});
    std::vector<uint8_t> bytes;
    for (float v : {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}) {
        uint8_t le[2];
        sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(v), le);
        bytes.push_back(le[0]);
        bytes.push_back(le[1]);
    }
    auto w = Tensor::from_f16_bytes(bytes, Shape{3, 2});
    ASSERT_TRUE(w.has_value());
    auto y = matmul_nt(x, *w);
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    const auto d = y->f32_slice();
    EXPECT_LT(std::fabs(d[0] - 5.0f), 1e-2f);
    EXPECT_LT(std::fabs(d[1] - 11.0f), 1e-2f);
    EXPECT_LT(std::fabs(d[2] - 17.0f), 1e-2f);
}

TEST(Matmul, matmul_rank_mismatch) {
    auto a = Tensor::zeros(Shape{4}, DType::F32);
    auto b = Tensor::zeros(Shape{4}, DType::F32);
    ASSERT_TRUE(a.has_value() && b.has_value());
    EXPECT_FALSE(matmul(*a, *b).has_value());
    EXPECT_EQ(matmul(*a, *b).error().to_string(), "Rank mismatch: expected 2, got 1");
}

TEST(Matmul, gemm_with_bias) {
    const auto a = f32({1.0f, 0.0f, 0.0f, 1.0f}, Shape{2, 2});
    const auto b = f32({2.0f, 3.0f, 4.0f, 5.0f}, Shape{2, 2});
    const auto bias = f32({1.0f, 1.0f}, Shape{2});
    auto c = gemm(a, b, &bias, 1.0f, 1.0f, false, false);
    ASSERT_TRUE(c.has_value()) << c.error().to_string();
    const auto d = c->f32_slice();
    // Identity × [[2,3],[4,5]] = [[2,3],[4,5]]; + bias [1,1] = [[3,4],[5,6]]
    EXPECT_LT(std::fabs(d[0] - 3.0f), 1e-5f) << "got " << d[0];
    EXPECT_LT(std::fabs(d[1] - 4.0f), 1e-5f) << "got " << d[1];
}

// ── C++-only ─────────────────────────────────────────────────────────────────

TEST(Matmul, gemv_chunk_follows_the_rust_formula) {
    if (std::getenv("SAPIENT_GEMV_TPC") != nullptr) GTEST_SKIP() << "SAPIENT_GEMV_TPC is set";
    // Plan C: spinpool disabled and thermal inert → ncpus = num_threads(); default clamp(n/(ncpus*4), 16, 512).
    const size_t ncpus = sapient::backends_cpu::parallel::num_threads();
    for (size_t n : {size_t{1}, size_t{16}, size_t{100}, size_t{4096}, size_t{151936}})
        EXPECT_EQ(detail::gemv_chunk(n), std::clamp<size_t>(n / (ncpus * 4), 16, 512)) << "n=" << n;
}

TEST(Matmul, for_each_out_chunk_partition_matches_rayon) {
    std::vector<float> out(100, -1.0f);
    detail::for_each_out_chunk(out, 16, [](size_t ci, std::span<float> cs) {
        for (float& v : cs)
            v = static_cast<float>(ci);
    });
    for (size_t i = 0; i < out.size(); ++i)
        EXPECT_EQ(out[i], static_cast<float>(i / 16)) << i;
    std::atomic<int> empty_calls{0};
    std::vector<float> empty;
    detail::for_each_out_chunk(
        empty, 16, [&](size_t, std::span<float>) { empty_calls.fetch_add(1); });
    EXPECT_EQ(empty_calls.load(), 0);
}

// m=1, k=512 takes the dot_f32_fast GEMV path (NEON / AVX2 / scalar): check it against a naive dot.
TEST(Matmul, f32_gemv_path_matches_naive_dot) {
    const size_t k = 512, n = 5;
    const auto xv = lcg(k, 3, -1.0f, 1.0f);
    const auto wv = lcg(n * k, 4, -1.0f, 1.0f);
    auto y = matmul_nt(f32(xv, Shape{1, k}), f32(wv, Shape{n, k}));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    for (size_t j = 0; j < n; ++j) {
        double ref = 0.0;
        for (size_t i = 0; i < k; ++i)
            ref += static_cast<double>(xv[i]) * static_cast<double>(wv[j * k + i]);
        EXPECT_NEAR(y->f32_slice()[j], static_cast<float>(ref), 1e-4f) << "col " << j;
    }
}

// m=1, k>=64, F16 weights takes the dot_f32_x_f16 GEMV path. POSITIVE normals only: the Rust NEON
// bit-surgery mis-decodes negative f16 values (sign bit leaks into the exponent), which this port
// reproduces on purpose — the golden case `matmul_nt_f16_m1` pins that reproduction bit-exactly.
TEST(Matmul, f16_gemv_path_matches_widened_reference_for_positive_normals) {
    const size_t k = 64, n = 3;
    const auto xv = lcg(k, 5, -1.0f, 1.0f);
    const auto wsrc = lcg(n * k, 6, 0.01f, 1.0f);
    std::vector<uint8_t> bytes;
    std::vector<float> widened;
    for (float v : wsrc) {
        const uint16_t h = sapient::core::f32_to_f16_bits(v);
        uint8_t le[2];
        sapient::core::f16_to_le(h, le);
        bytes.push_back(le[0]);
        bytes.push_back(le[1]);
        widened.push_back(sapient::core::f16_bits_to_f32(h));
    }
    auto w = Tensor::from_f16_bytes(bytes, Shape{n, k});
    ASSERT_TRUE(w.has_value());
    auto y = matmul_nt(f32(xv, Shape{1, k}), *w);
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    for (size_t j = 0; j < n; ++j) {
        double ref = 0.0;
        for (size_t i = 0; i < k; ++i)
            ref += static_cast<double>(xv[i]) * static_cast<double>(widened[j * k + i]);
        EXPECT_NEAR(y->f32_slice()[j], static_cast<float>(ref), 1e-4f) << "col " << j;
    }
}

// DELETE IN PLAN D (when the quant arms land). Pins the plan-C stub so quantized weights never
// yield a silent wrong result.
TEST(Matmul, quantized_weight_dtypes_are_stubbed_until_plan_d) {
    const auto x = f32(std::vector<float>(64, 0.5f), Shape{1, 64});
    auto w = Tensor::from_quant_bytes(std::vector<uint8_t>(2 * 34, 0), Shape{1, 64}, DType::Q8_0);
    ASSERT_TRUE(w.has_value()) << w.error().to_string();
    auto y = matmul_nt(x, *w);
    ASSERT_FALSE(y.has_value());
    EXPECT_EQ(y.error().to_string(),
              "Internal error: matmul_nt: quantized weights (" +
                  sapient::core::to_string(DType::Q8_0) + ") land in plan D");
}

// C++-only: the parity-bound Result-path texts of matmul_nt / matmul / gemm.
TEST(Matmul, error_messages_match_rust) {
    const auto x = f32({1.0f, 2.0f}, Shape{1, 2});
    const auto w3 = f32({1.0f, 2.0f, 3.0f}, Shape{3, 1});
    const auto x1 = f32({1.0f, 2.0f}, Shape{2});
    EXPECT_EQ(matmul_nt(x1, w3).error().to_string(),
              "Internal error: matmul_nt expects 2-D tensors");
    EXPECT_EQ(matmul_nt(x, w3).error().to_string(), "Shape mismatch: expected [1, 2], got [3, 1]");
    EXPECT_EQ(matmul(x, w3).error().to_string(),
              "Shape mismatch: expected [1, 2, 1], got [1, 3, 1]");
    EXPECT_EQ(gemm(x, w3, nullptr, 1.0f, 0.0f, false, false).error().to_string(),
              "Shape mismatch: expected [1, 2], got [3, 1]");
    const auto b = f32({1.0f, 2.0f}, Shape{2, 1});
    const auto bad_bias = f32({1.0f, 2.0f, 3.0f}, Shape{3});
    EXPECT_EQ(gemm(x, b, &bad_bias, 1.0f, 1.0f, false, false).error().to_string(),
              "Shape mismatch: expected [1], got [3]");
}
