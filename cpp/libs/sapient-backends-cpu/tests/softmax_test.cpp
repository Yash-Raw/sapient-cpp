// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "sapient/backends_cpu/kernels/softmax.hpp"
#include "sapient/core/tensor.hpp"

using namespace sapient::backends_cpu::kernels::softmax;
using sapient::core::Shape;
using sapient::core::Tensor;

namespace {
Tensor f32(std::vector<float> data, Shape shape) {
    auto r = Tensor::from_f32_vec(std::move(data), std::move(shape));
    if (!r) throw std::runtime_error(r.error().to_string());
    return std::move(*r);
}
} // namespace

TEST(Softmax, softmax_sums_to_one) {
    const auto x = f32({1.0f, 2.0f, 3.0f, 4.0f}, Shape{1, 4});
    auto y = softmax(x, 1);
    ASSERT_TRUE(y.has_value());
    float sum = 0.0f;
    for (float v : y->f32_slice())
        sum += v;
    EXPECT_LT(std::fabs(sum - 1.0f), 1e-6f) << "sum = " << sum;
}

TEST(Softmax, softmax_stable_large) {
    const auto x = f32({1000.0f, 1001.0f, 1002.0f}, Shape{1, 3});
    auto y = softmax(x, 1);
    ASSERT_TRUE(y.has_value());
    float sum = 0.0f;
    for (float v : y->f32_slice()) {
        EXPECT_TRUE(std::isfinite(v)) << "non-finite: " << v;
        sum += v;
    }
    EXPECT_LT(std::fabs(sum - 1.0f), 1e-5f) << "sum = " << sum;
}

TEST(Softmax, log_softmax_finite) {
    const auto x = f32({1.0f, 2.0f, 3.0f}, Shape{1, 3});
    auto y = log_softmax(x, 1);
    ASSERT_TRUE(y.has_value());
    for (float v : y->f32_slice())
        EXPECT_TRUE(std::isfinite(v));
}

// C++-only: the Result-path message is parity-bound; a very negative axis wraps like Rust's `as usize`.
TEST(Softmax, axis_error_message_matches_rust) {
    const auto x = f32({1.0f, 2.0f, 3.0f}, Shape{1, 3});
    auto e = softmax(x, 2);
    ASSERT_FALSE(e.has_value());
    EXPECT_EQ(e.error().to_string(), "Internal error: softmax axis 2 out of range for rank 2");
    auto w = softmax(x, -7);
    ASSERT_FALSE(w.has_value());
    EXPECT_EQ(w.error().to_string(), "Internal error: softmax axis -7 out of range for rank 2");
}
