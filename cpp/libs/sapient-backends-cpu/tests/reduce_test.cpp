// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "sapient/backends_cpu/kernels/reduce.hpp"
#include "sapient/core/tensor.hpp"

using namespace sapient::backends_cpu::kernels::reduce;
using sapient::core::Shape;
using sapient::core::Tensor;

namespace {
Tensor f32(std::vector<float> data, Shape shape) {
    auto r = Tensor::from_f32_vec(std::move(data), std::move(shape));
    if (!r) throw std::runtime_error(r.error().to_string());
    return std::move(*r);
}
} // namespace

TEST(Reduce, sum_all) {
    const auto x = f32({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2});
    auto y = reduce_sum(x, {}, false);
    ASSERT_TRUE(y.has_value());
    EXPECT_TRUE(y->shape().dims.empty()) << "all-axes reduction is a scalar";
    EXPECT_LT(std::fabs(y->f32_slice()[0] - 10.0f), 1e-5f);
}

TEST(Reduce, mean_axis0) {
    const auto x = f32({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2});
    const int64_t axes[] = {0};
    auto y = reduce_mean(x, axes, false);
    ASSERT_TRUE(y.has_value());
    const auto d = y->f32_slice();
    EXPECT_LT(std::fabs(d[0] - 2.0f), 1e-5f) << "d[0]=" << d[0];
    EXPECT_LT(std::fabs(d[1] - 3.0f), 1e-5f) << "d[1]=" << d[1];
}
