// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sapient/backends_cpu/cpu_features.hpp"
#include "sapient/backends_cpu/kernels/matmul.hpp"
#include "sapient/backends_cpu/kernels/quant.hpp"
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

namespace {
namespace quant = sapient::backends_cpu::kernels::quant;
#if defined(__aarch64__) || defined(_M_ARM64)
using sapient::backends_cpu::cpu_features::has_dotprod;
using sapient::backends_cpu::cpu_features::has_i8mm;
#endif

// Rust: `.chunks_exact(32).flat_map(quantize_q8_0_block)` over a flat f32 array.
std::vector<uint8_t> q8_0_blocks(std::span<const float> w) {
    std::vector<uint8_t> out;
    for (size_t b = 0; b + 32 <= w.size(); b += 32) {
        const auto blk = quant::quantize_q8_0_block(w.subspan(b, 32));
        out.insert(out.end(), blk.begin(), blk.end());
    }
    return out;
}
Tensor quant_tensor(std::vector<uint8_t> bytes, Shape shape, DType dtype) {
    auto r = Tensor::from_quant_bytes(bytes, std::move(shape), dtype);
    if (!r) throw std::runtime_error(r.error().to_string());
    return std::move(*r);
}
#if defined(__aarch64__) || defined(_M_ARM64)
uint32_t bits(float f) {
    return std::bit_cast<uint32_t>(f);
}
#endif
} // namespace

// ── quantized arms (plan D) ──────────────────────────────────────────────────

TEST(Matmul, matmul_nt_q4_0_matches_float) {
    // 4 output rows, 64 input features (64 is a multiple of 32 = two blocks/row).
    const size_t n_out = 4, k = 64;
    std::vector<float> w_f32(n_out * k);
    for (size_t i = 0; i < w_f32.size(); ++i)
        w_f32[i] = (::fmodf(static_cast<float>(i), 16.0f) - 8.0f) * 0.05f;
    std::vector<float> x_f32(k);
    for (size_t i = 0; i < k; ++i)
        x_f32[i] = static_cast<float>(i) * 0.01f - 0.3f;

    const auto w_t = f32(w_f32, Shape{n_out, k});
    const auto x_t = f32(x_f32, Shape{1, k});
    auto ref_out = matmul_nt(x_t, w_t);
    ASSERT_TRUE(ref_out.has_value()) << ref_out.error().to_string();
    const auto ref_data = ref_out->to_f32_vec();

    // Quantize each row to Q4_0.
    std::vector<uint8_t> w_blocks;
    for (size_t r = 0; r < n_out; ++r) {
        const auto row = quant::quantize_q4_0_row(std::span<const float>(w_f32).subspan(r * k, k));
        w_blocks.insert(w_blocks.end(), row.begin(), row.end());
    }
    const auto w_q = quant_tensor(w_blocks, Shape{n_out, k}, DType::Q4_0);
    auto quant_out = matmul_nt(x_t, w_q);
    ASSERT_TRUE(quant_out.has_value()) << quant_out.error().to_string();
    const auto quant_data = quant_out->to_f32_vec();
    ASSERT_EQ(ref_data.size(), quant_data.size());
    for (size_t i = 0; i < ref_data.size(); ++i)
        EXPECT_LT(::fabsf(ref_data[i] - quant_data[i]), 5e-3f)
            << "row " << i << ": ref=" << ref_data[i] << " quant=" << quant_data[i];
}

#if defined(__aarch64__) || defined(_M_ARM64)
TEST(Matmul, q8_0_gemm_path_matches_per_row_path) {
    if (!has_dotprod()) GTEST_SKIP() << "dotprod not available";
    // m = 16 triggers the blocked GEMM; compare each row against the m = 1 (per-row GEMV) path —
    // same kernel, same scales → bit-identical.
    const size_t m = 16, k = 96, n = 24;
    uint64_t seed = 0x8A8AULL;
    auto nf = [&seed]() {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        return (static_cast<float>(seed >> 40) / static_cast<float>(1ULL << 24)) * 2.0f - 1.0f;
    };
    std::vector<float> xv(m * k);
    for (float& v : xv)
        v = nf() * 0.5f;
    std::vector<float> wv(n * k);
    for (float& v : wv)
        v = nf() * 0.2f;
    const auto w_q8 = quant_tensor(q8_0_blocks(wv), Shape{n, k}, DType::Q8_0);

    auto full = matmul_nt(f32(xv, Shape{m, k}), w_q8);
    ASSERT_TRUE(full.has_value()) << full.error().to_string();
    const auto fd = full->to_f32_vec();
    for (size_t i = 0; i < m; ++i) {
        const std::vector<float> x_row(xv.begin() + static_cast<std::ptrdiff_t>(i * k),
                                       xv.begin() + static_cast<std::ptrdiff_t>((i + 1) * k));
        auto want = matmul_nt(f32(x_row, Shape{1, k}), w_q8);
        ASSERT_TRUE(want.has_value());
        const auto wd = want->to_f32_vec();
        for (size_t j = 0; j < n; ++j)
            EXPECT_EQ(bits(fd[i * n + j]), bits(wd[j]))
                << "row " << i << " col " << j << ": " << fd[i * n + j] << " vs " << wd[j];
    }
}
#endif

TEST(Matmul, matmul_nt_q8_0_matches_float) {
    // Larger k so several 32-blocks per row exercise the SDOT/NEON path, and an activation
    // outlier to stress per-block activation quantization.
    const size_t n_out = 8, k = 256;
    std::vector<float> w_f32(n_out * k);
    for (size_t i = 0; i < w_f32.size(); ++i)
        w_f32[i] = (static_cast<float>(i * 7 % 31) - 15.0f) * 0.03f;
    std::vector<float> x_f32(k);
    for (size_t i = 0; i < k; ++i)
        x_f32[i] = ::sinf(static_cast<float>(i) * 0.013f) * 0.4f;
    x_f32[100] = 25.0f; // outlier channel

    const auto w_q = quant_tensor(q8_0_blocks(w_f32), Shape{n_out, k}, DType::Q8_0);
    const auto x_t = f32(x_f32, Shape{1, k});
    // Reference: dequantize the SAME Q8_0 weights to f32, then exact f32 matmul.
    const auto w_ref = f32(w_q.to_f32_vec(), Shape{n_out, k});
    auto ref_out = matmul_nt(x_t, w_ref);
    ASSERT_TRUE(ref_out.has_value());
    const auto ref_data = ref_out->to_f32_vec();
    auto quant_out = matmul_nt(x_t, w_q);
    ASSERT_TRUE(quant_out.has_value()) << quant_out.error().to_string();
    const auto quant_data = quant_out->to_f32_vec();
    ASSERT_EQ(ref_data.size(), quant_data.size());
    for (size_t i = 0; i < ref_data.size(); ++i) {
        const float tol = 0.02f * ::fmaxf(::fabsf(ref_data[i]), 1.0f);
        EXPECT_LT(::fabsf(ref_data[i] - quant_data[i]), tol)
            << "row " << i << ": ref=" << ref_data[i] << " quant=" << quant_data[i];
    }
}

// Replicates the GGUF load path for Q8_0 weights: the tensor is built with the ggml dim order
// [in, out] and then reshaped to HF [out, in] (exactly what map_gguf_tensors_to_hf does).
TEST(Matmul, matmul_nt_q8_0_gguf_dimflip_matches_float) {
    const size_t out_features = 64, in_features = 128;
    std::vector<float> w_f32(out_features * in_features);
    for (size_t i = 0; i < w_f32.size(); ++i)
        w_f32[i] = (static_cast<float>(i * 13 % 29) - 14.0f) * 0.02f;
    std::vector<float> x_f32(in_features);
    for (size_t i = 0; i < in_features; ++i)
        x_f32[i] = ::cosf(static_cast<float>(i) * 0.02f) * 0.5f;
    const auto x_t = f32(x_f32, Shape{1, in_features});

    // ggml stores ne[0]=in contiguous, so its flat byte order == row-major [out, in] == w_f32.
    auto w_gguf = quant_tensor(q8_0_blocks(w_f32), Shape{in_features, out_features}, DType::Q8_0)
                      .reshape(Shape{out_features, in_features});
    ASSERT_TRUE(w_gguf.has_value()) << w_gguf.error().to_string();
    const auto w_ref = f32(w_gguf->to_f32_vec(), Shape{out_features, in_features});
    auto ref_out = matmul_nt(x_t, w_ref);
    ASSERT_TRUE(ref_out.has_value());
    const auto ref_data = ref_out->to_f32_vec();
    auto got = matmul_nt(x_t, *w_gguf);
    ASSERT_TRUE(got.has_value()) << got.error().to_string();
    const auto gd = got->to_f32_vec();
    for (size_t i = 0; i < ref_data.size(); ++i)
        EXPECT_LT(::fabsf(ref_data[i] - gd[i]), 0.02f * ::fmaxf(::fabsf(ref_data[i]), 1.0f))
            << "out " << i << ": ref=" << ref_data[i] << " quant=" << gd[i];
}

#if defined(__aarch64__) || defined(_M_ARM64)
// Panel-blocked SMMLA prefill must be bit-identical to per-row GEMV: m = 259 spans two full
// 128-row panels plus an odd 3-row remainder (pair loop + odd final row on every group).
TEST(Matmul, q4_k_r4_prefill_matches_per_row) {
    const size_t n = 8, k = 256, m = 259;
    std::vector<uint8_t> blocks(n * quant::Q4_K_BLOCK_BYTES);
    for (size_t i = 0; i < blocks.size(); ++i)
        blocks[i] = static_cast<uint8_t>((i * 131 + 7) % 251);
    for (size_t r = 0; r < n; ++r) {
        uint8_t* base = blocks.data() + r * quant::Q4_K_BLOCK_BYTES;
        sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(0.05f), base);     // d
        sapient::core::f16_to_le(sapient::core::f32_to_f16_bits(0.03f), base + 2); // dmin
    }
    const auto packed = quant::repack_q4_k_rows4(blocks, n, k);
    const auto w_r4 = quant_tensor(packed, Shape{n, k}, DType::Q4_K_R4);

    std::vector<float> x_f32(m * k);
    for (size_t i = 0; i < x_f32.size(); ++i)
        x_f32[i] = (static_cast<float>(i * 37 % 97) - 48.0f) * 0.02f;
    auto panelled = matmul_nt(f32(x_f32, Shape{m, k}), w_r4);
    ASSERT_TRUE(panelled.has_value()) << panelled.error().to_string();
    const auto pd = panelled->to_f32_vec();
    for (size_t i = 0; i < m; ++i) {
        const std::vector<float> x_row(x_f32.begin() + static_cast<std::ptrdiff_t>(i * k),
                                       x_f32.begin() + static_cast<std::ptrdiff_t>((i + 1) * k));
        auto row_out = matmul_nt(f32(x_row, Shape{1, k}), w_r4);
        ASSERT_TRUE(row_out.has_value());
        const auto rd = row_out->to_f32_vec();
        for (size_t j = 0; j < n; ++j)
            EXPECT_EQ(bits(pd[i * n + j]), bits(rd[j]))
                << "row " << i << " col " << j << ": " << pd[i * n + j] << " vs " << rd[j];
    }
}
#endif

// C++-only: the seven parity-bound guard texts of the quantized arms (rule 7). byte_count
// truncates, so a [1, 48] Q4_0 tensor builds from one 18-byte block and still fails k % 32.
TEST(Matmul, quantized_guard_messages_match_rust) {
    auto q = [](size_t n, size_t k, DType dt, size_t nbytes) {
        return quant_tensor(std::vector<uint8_t>(nbytes, 0), Shape{n, k}, dt);
    };
    const auto x48 = f32(std::vector<float>(48, 0.0f), Shape{1, 48});
    EXPECT_EQ(matmul_nt(x48, q(1, 48, DType::Q4_0, 18)).error().to_string(),
              "Internal error: Q4_0 matmul_nt: k must be a multiple of the block size (32)");
    EXPECT_EQ(matmul_nt(x48, q(1, 48, DType::Q8_0, 34)).error().to_string(),
              "Internal error: Q8_0 matmul_nt: k must be a multiple of the block size (32)");
    const auto x128 = f32(std::vector<float>(128, 0.0f), Shape{1, 128});
    EXPECT_EQ(matmul_nt(x128, q(2, 128, DType::Q4_K, 144)).error().to_string(),
              "Internal error: Q4_K: k must be a multiple of 256");
    EXPECT_EQ(matmul_nt(x128, q(2, 128, DType::Q5_K, 176)).error().to_string(),
              "Internal error: Q5_K: k must be a multiple of 256");
    EXPECT_EQ(matmul_nt(x128, q(2, 128, DType::Q6_K, 210)).error().to_string(),
              "Internal error: Q6_K: k must be a multiple of 256");
    EXPECT_EQ(matmul_nt(x128, q(2, 128, DType::Q4_K_R4, 144)).error().to_string(),
              "Internal error: Q4_K_R4: k must be a multiple of 256 and rows a multiple of 4");
    EXPECT_EQ(matmul_nt(x128, q(2, 128, DType::Q6_K_R4, 210)).error().to_string(),
              "Internal error: Q6_K_R4: k must be a multiple of 256 and rows a multiple of 4");
    const auto x256 = f32(std::vector<float>(256, 0.0f), Shape{1, 256});
    EXPECT_EQ(matmul_nt(x256, q(2, 256, DType::Q4_K_R4, 2 * 144)).error().to_string(),
              "Internal error: Q4_K_R4: k must be a multiple of 256 and rows a multiple of 4");
}
