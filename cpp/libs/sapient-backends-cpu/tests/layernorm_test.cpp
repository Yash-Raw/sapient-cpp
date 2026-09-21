// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "sapient/backends_cpu/kernels/layernorm.hpp"
#include "sapient/core/tensor.hpp"

using namespace sapient::backends_cpu::kernels::layernorm;
using sapient::core::Shape;
using sapient::core::Tensor;

namespace {
Tensor f32(std::vector<float> data, Shape shape) {
    auto r = Tensor::from_f32_vec(std::move(data), std::move(shape));
    if (!r) throw std::runtime_error(r.error().to_string());
    return std::move(*r);
}
} // namespace

TEST(LayerNorm, layernorm_zero_mean_unit_var) {
    const auto x = f32({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2});
    auto y = layer_norm(x, nullptr, nullptr, -1, 1e-5f);
    ASSERT_TRUE(y.has_value());
    const auto d = y->f32_slice();
    // Each pair: mean=1.5, var=0.25, std=0.5. (1-1.5)/0.5 = -1, (2-1.5)/0.5 = 1.
    EXPECT_LT(std::fabs(d[0] + 1.0f), 1e-4f) << "d[0]=" << d[0];
    EXPECT_LT(std::fabs(d[1] - 1.0f), 1e-4f) << "d[1]=" << d[1];
    EXPECT_LT(std::fabs(d[2] + 1.0f), 1e-4f) << "d[2]=" << d[2];
    EXPECT_LT(std::fabs(d[3] - 1.0f), 1e-4f) << "d[3]=" << d[3];
}

TEST(LayerNorm, rmsnorm_identity_weight) {
    // rms = sqrt((9 + 16) / 2) = sqrt(12.5); output = [3, 4] / sqrt(12.5).
    const auto x = f32({3.0f, 4.0f}, Shape{1, 2});
    const auto w = f32({1.0f, 1.0f}, Shape{2});
    auto y = rms_norm(x, &w, 0.0f);
    ASSERT_TRUE(y.has_value());
    const auto d = y->f32_slice();
    const float expected0 = 3.0f / std::sqrt(12.5f);
    const float expected1 = 4.0f / std::sqrt(12.5f);
    EXPECT_LT(std::fabs(d[0] - expected0), 1e-5f) << "d[0]=" << d[0] << " expected " << expected0;
    EXPECT_LT(std::fabs(d[1] - expected1), 1e-5f) << "d[1]=" << d[1] << " expected " << expected1;
}
