// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include "sapient/backends_cpu/kernels/attention.hpp"
#include "sapient/core/tensor.hpp"

using namespace sapient::backends_cpu::kernels::attention;
using sapient::core::Shape;
using sapient::core::Tensor;

namespace {
Tensor f32(std::vector<float> data, Shape shape) {
    auto r = Tensor::from_f32_vec(std::move(data), std::move(shape));
    if (!r) throw std::runtime_error(r.error().to_string());
    return std::move(*r);
}
std::vector<float> vec(const Tensor& x) {
    const auto s = x.f32_slice();
    return {s.begin(), s.end()};
}
} // namespace

TEST(Attention, mha_output_shape) {
    // batch=1, heads=2, seq=3, dim=4
    const auto q = f32(std::vector<float>(24, 0.1f), Shape{1, 2, 3, 4});
    const auto k = f32(std::vector<float>(24, 0.1f), Shape{1, 2, 3, 4});
    const auto v = f32(std::vector<float>(24, 0.1f), Shape{1, 2, 3, 4});
    auto out = scaled_dot_product_attention(q, k, v, nullptr, std::nullopt, 2);
    ASSERT_TRUE(out.has_value()) << out.error().to_string();
    EXPECT_EQ(out->shape().dims, (std::vector<size_t>{1, 2, 3, 4}));
}

TEST(Attention, gqa_kv_repeat) {
    // batch=1, n_heads=4, n_kv_heads=2, seq=2, dim=4
    const auto q = f32(std::vector<float>(32, 0.1f), Shape{1, 4, 2, 4});
    const auto k = f32(std::vector<float>(16, 0.1f), Shape{1, 2, 2, 4});
    const auto v = f32(std::vector<float>(16, 0.1f), Shape{1, 2, 2, 4});
    auto out = scaled_dot_product_attention(q, k, v, nullptr, std::nullopt, 2);
    ASSERT_TRUE(out.has_value()) << out.error().to_string();
    EXPECT_EQ(out->shape().dims, (std::vector<size_t>{1, 4, 2, 4}));
}

TEST(Attention, causal_mask_shape) {
    const Tensor m = causal_mask(3, 3);
    const auto d = m.f32_slice();
    // Position (0,1) should be -inf (index 1)
    EXPECT_TRUE(std::isinf(d[1]) && d[1] < 0.0f);
    // Position (1,0) should be 0 (index 3)
    EXPECT_EQ(d[3], 0.0f);
}

// Leading -inf mask positions (a sliding window that has scrolled past them) must not NaN the
// online softmax — regression for the Gemma3 "coherent until position 512, then salad" bug.
TEST(Attention, leading_masked_positions_do_not_nan) {
    const size_t seq_k = 8, head_dim = 4;
    const auto q = f32(std::vector<float>(head_dim, 1.0f), Shape{1, 1, 1, head_dim});
    std::vector<float> kv(seq_k * head_dim);
    for (size_t i = 0; i < kv.size(); ++i)
        kv[i] = static_cast<float>(i % 7) * 0.1f;
    const auto k = f32(kv, Shape{1, 1, seq_k, head_dim});
    const auto v = f32(kv, Shape{1, 1, seq_k, head_dim});
    // Window of 3: only the last 3 positions visible.
    std::vector<float> mask(seq_k, -std::numeric_limits<float>::infinity());
    for (size_t i = seq_k - 3; i < seq_k; ++i)
        mask[i] = 0.0f;
    const auto mask_t = f32(mask, Shape{1, seq_k});
    auto out = scaled_dot_product_attention(q, k, v, &mask_t, std::nullopt, 1);
    ASSERT_TRUE(out.has_value()) << out.error().to_string();
    const auto ov = out->to_f32_vec();
    for (float x : ov)
        EXPECT_TRUE(std::isfinite(x)) << "NaN/inf in output";
    // Reference: naive softmax over the visible 3 positions.
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores;
    for (size_t ki = seq_k - 3; ki < seq_k; ++ki) {
        float s = 0.0f;
        for (size_t d = 0; d < head_dim; ++d)
            s += kv[ki * head_dim + d];
        scores.push_back(s * scale);
    }
    float mx = -std::numeric_limits<float>::infinity();
    for (float s : scores)
        mx = std::max(mx, s);
    float sum = 0.0f;
    for (float& s : scores) {
        s = std::exp(s - mx);
        sum += s;
    }
    for (size_t d = 0; d < head_dim; ++d) {
        float want = 0.0f;
        for (size_t j = 0; j < 3; ++j)
            want += scores[j] / sum * kv[(seq_k - 3 + j) * head_dim + d];
        EXPECT_LT(std::fabs(ov[d] - want), 1e-5f) << "d" << d << ": " << ov[d] << " vs " << want;
    }
}

TEST(Attention, uniform_attention_recovers_v) {
    // Equal scores → uniform weights → the output is the mean of the V rows: (1+2+3+4)/4 = 2.5.
    const size_t seq = 4, dim = 8;
    std::vector<float> v_data(seq * dim);
    for (size_t i = 0; i < seq; ++i)
        for (size_t d = 0; d < dim; ++d)
            v_data[i * dim + d] = static_cast<float>(i + 1);
    const auto q = f32(std::vector<float>(seq * dim, 1.0f), Shape{1, 1, seq, dim});
    const auto k = f32(std::vector<float>(seq * dim, 1.0f), Shape{1, 1, seq, dim});
    const auto v = f32(v_data, Shape{1, 1, seq, dim});
    // Explicit all-zero mask so every key is attended (no causal masking).
    const auto mask = f32(std::vector<float>(seq * seq, 0.0f), Shape{seq, seq});
    auto out = scaled_dot_product_attention(q, k, v, &mask, std::nullopt, 1);
    ASSERT_TRUE(out.has_value()) << out.error().to_string();
    const float expected = (1.0f + 2.0f + 3.0f + 4.0f) / static_cast<float>(seq);
    for (float val : out->f32_slice())
        EXPECT_LT(std::fabs(val - expected), 1e-4f) << "Expected ~" << expected << ", got " << val;
}

// The online-softmax result matches a naive reference implementation.
TEST(Attention, flash_matches_naive) {
    const size_t batch = 1, n_heads = 2, seq_q = 4, seq_k = 4, head_dim = 8;
    const auto gen = [](size_t i) {
        return ::sinf(static_cast<float>(i) * 1.3f + 0.7f) * 0.5f + 0.5f;
    };
    std::vector<float> q_data(batch * n_heads * seq_q * head_dim),
        k_data(batch * n_heads * seq_k * head_dim), v_data(batch * n_heads * seq_k * head_dim);
    for (size_t i = 0; i < q_data.size(); ++i)
        q_data[i] = gen(i);
    for (size_t i = 0; i < k_data.size(); ++i)
        k_data[i] = gen(i + 100);
    for (size_t i = 0; i < v_data.size(); ++i)
        v_data[i] = gen(i + 200);
    const auto q = f32(q_data, Shape{batch, n_heads, seq_q, head_dim});
    const auto k = f32(k_data, Shape{batch, n_heads, seq_k, head_dim});
    const auto v = f32(v_data, Shape{batch, n_heads, seq_k, head_dim});
    const Tensor mask_t = causal_mask(seq_q, seq_k);
    auto flash_out = scaled_dot_product_attention(q, k, v, &mask_t, std::nullopt, n_heads);
    ASSERT_TRUE(flash_out.has_value()) << flash_out.error().to_string();

    // --- Naive reference ---
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const auto mask_data = mask_t.f32_slice();
    std::vector<float> ref_out(batch * n_heads * seq_q * head_dim, 0.0f);
    for (size_t b = 0; b < batch; ++b)
        for (size_t h = 0; h < n_heads; ++h) {
            const size_t q_off = (b * n_heads + h) * seq_q * head_dim;
            const size_t k_off = (b * n_heads + h) * seq_k * head_dim;
            const size_t v_off = k_off;
            const size_t o_off = q_off;
            for (size_t qi = 0; qi < seq_q; ++qi) {
                std::vector<float> scores(seq_k, 0.0f);
                for (size_t ki = 0; ki < seq_k; ++ki) {
                    float dot = 0.0f;
                    for (size_t d = 0; d < head_dim; ++d)
                        dot +=
                            q_data[q_off + qi * head_dim + d] * k_data[k_off + ki * head_dim + d];
                    scores[ki] = dot * scale + mask_data[qi * seq_k + ki];
                }
                float max_s = -std::numeric_limits<float>::infinity();
                for (float s : scores)
                    max_s = std::max(max_s, s);
                if (std::isinf(max_s)) max_s = 0.0f;
                float sum = 0.0f;
                for (float& s : scores) {
                    s = std::exp(s - max_s);
                    sum += s;
                }
                if (sum < std::numeric_limits<float>::epsilon())
                    sum = std::numeric_limits<float>::epsilon();
                for (size_t d = 0; d < head_dim; ++d) {
                    float acc = 0.0f;
                    for (size_t ki = 0; ki < seq_k; ++ki)
                        acc += scores[ki] / sum * v_data[v_off + ki * head_dim + d];
                    ref_out[o_off + qi * head_dim + d] = acc;
                }
            }
        }
    const auto flash_data = vec(*flash_out);
    ASSERT_EQ(flash_data.size(), ref_out.size());
    for (size_t i = 0; i < flash_data.size(); ++i) {
        const float diff = std::fabs(flash_data[i] - ref_out[i]);
        EXPECT_LT(diff, 1e-4f) << "Mismatch at index " << i << ": flash=" << flash_data[i]
                               << " ref=" << ref_out[i];
    }
}

// Decode-mode (seq_q=1): output shape and no NaN/Inf.
TEST(Attention, decode_mode_no_nan) {
    const size_t batch = 1, n_heads = 4, seq_q = 1, seq_k = 16, head_dim = 8;
    const auto q = f32(std::vector<float>(batch * n_heads * seq_q * head_dim, 0.1f),
                       Shape{batch, n_heads, seq_q, head_dim});
    const auto k = f32(std::vector<float>(batch * n_heads * seq_k * head_dim, 0.1f),
                       Shape{batch, n_heads, seq_k, head_dim});
    const auto v = f32(std::vector<float>(batch * n_heads * seq_k * head_dim, 0.2f),
                       Shape{batch, n_heads, seq_k, head_dim});
    auto out = scaled_dot_product_attention(q, k, v, nullptr, std::nullopt, n_heads);
    ASSERT_TRUE(out.has_value()) << out.error().to_string();
    EXPECT_EQ(out->shape().dims, (std::vector<size_t>{batch, n_heads, seq_q, head_dim}));
    for (float val : out->f32_slice())
        EXPECT_TRUE(std::isfinite(val)) << "NaN/Inf in decode output: " << val;
}

// C++-only: the one Result-path error and one Rust index panic.
TEST(Attention, rank_error_and_kv_panic) {
    const auto q3 = f32(std::vector<float>(8, 0.1f), Shape{1, 2, 4});
    const auto q = f32(std::vector<float>(8, 0.1f), Shape{1, 1, 2, 4});
    const auto k = f32(std::vector<float>(8, 0.1f), Shape{1, 1, 2, 4});
    EXPECT_EQ(scaled_dot_product_attention(q3, k, k, nullptr, std::nullopt, 1).error().to_string(),
              "Rank mismatch: expected 4, got 3");
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    const auto k2 = f32(std::vector<float>(8, 0.1f), Shape{2, 4}); // Rust: ks[2] index panic
    EXPECT_DEATH((void)scaled_dot_product_attention(q, k2, k2, nullptr, std::nullopt, 1), "rank");
}
