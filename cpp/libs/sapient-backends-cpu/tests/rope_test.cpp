// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "sapient/backends_cpu/kernels/rope.hpp"
#include "sapient/core/tensor.hpp"

using namespace sapient::backends_cpu::kernels::rope;
using sapient::core::Shape;
using sapient::core::Tensor;

namespace {
Tensor f32(std::vector<float> data, Shape shape) {
    auto r = Tensor::from_f32_vec(std::move(data), std::move(shape));
    if (!r) throw std::runtime_error(r.error().to_string());
    return std::move(*r);
}
} // namespace

TEST(Rope, rope_output_shape) {
    const auto x = f32(std::vector<float>(64, 0.1f), Shape{1, 2, 4, 8});
    const size_t positions[] = {0, 1, 2, 3};
    auto out = apply_rope(x, positions, 10000.0f);
    ASSERT_TRUE(out.has_value());
    EXPECT_EQ(out->shape().dims, (std::vector<size_t>{1, 2, 4, 8}));
}

TEST(Rope, rope_partial_leaves_tail_unchanged) {
    // head_dim=8, rotary_dim=4 → channels [4..8) pass through unchanged; at a non-zero position
    // the rotary channels [0..4) must change.
    std::vector<float> data;
    for (int v = 1; v <= 8; ++v)
        data.push_back(static_cast<float>(v));
    const auto x = f32(data, Shape{1, 1, 1, 8});
    const size_t positions[] = {3};
    auto out = apply_rope_partial(x, positions, 10000.0f, 4);
    ASSERT_TRUE(out.has_value());
    const auto o = out->f32_slice();
    for (size_t i = 4; i < 8; ++i)
        EXPECT_LT(std::fabs(o[i] - data[i]), 1e-6f) << "tail channel " << i << " changed";
    bool any_changed = false;
    for (size_t i = 0; i < 4; ++i)
        any_changed = any_changed || std::fabs(o[i] - data[i]) > 1e-6f;
    EXPECT_TRUE(any_changed);
}

TEST(Rope, rope_partial_full_matches_apply_rope) {
    std::vector<float> data;
    for (int v = 0; v < 16; ++v)
        data.push_back(static_cast<float>(v) * 0.1f);
    const auto x = f32(data, Shape{1, 1, 2, 8});
    const size_t positions[] = {2, 5};
    auto full = apply_rope(x, positions, 10000.0f);
    auto part = apply_rope_partial(x, positions, 10000.0f, 8);
    ASSERT_TRUE(full.has_value());
    ASSERT_TRUE(part.has_value());
    const auto a = full->f32_slice();
    const auto b = part->f32_slice();
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i)
        EXPECT_LT(std::fabs(a[i] - b[i]), 1e-6f) << i;
}

TEST(Rope, rope_position_zero_is_identity) {
    const std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    const auto x = f32(data, Shape{1, 1, 1, 4});
    const size_t positions[] = {0};
    auto out = apply_rope(x, positions, 10000.0f);
    ASSERT_TRUE(out.has_value());
    const auto o = out->f32_slice();
    for (size_t i = 0; i < data.size(); ++i)
        EXPECT_LT(std::fabs(data[i] - o[i]), 1e-6f) << "position 0 should be identity";
}

// C++-only: the four Result-path messages and the RankMismatch fields are parity-bound.
TEST(Rope, error_messages_match_rust) {
    const auto x3 = f32(std::vector<float>(8, 0.0f), Shape{1, 2, 4});
    const auto x4 = f32(std::vector<float>(8, 0.0f), Shape{1, 1, 1, 8});
    const auto x_odd = f32(std::vector<float>(7, 0.0f), Shape{1, 1, 1, 7});
    const size_t one[] = {0};
    const size_t two[] = {0, 1};
    EXPECT_EQ(apply_rope(x3, one, 1.0f).error().to_string(), "Rank mismatch: expected 4, got 3");
    EXPECT_EQ(apply_rope(x_odd, one, 1.0f).error().to_string(),
              "Internal error: RoPE requires even head_dim");
    EXPECT_EQ(apply_rope(x4, two, 1.0f).error().to_string(),
              "Internal error: positions length must match seq_len");
    EXPECT_EQ(apply_rope_partial(x4, one, 1.0f, 0).error().to_string(),
              "Internal error: rotary_dim must be in 1..=head_dim");
    EXPECT_EQ(apply_rope_partial(x4, one, 1.0f, 3).error().to_string(),
              "Internal error: RoPE requires even rotary_dim");
    EXPECT_EQ(apply_rope_partial(x4, two, 1.0f, 8).error().to_string(),
              "Internal error: positions length must match seq_len");
}
