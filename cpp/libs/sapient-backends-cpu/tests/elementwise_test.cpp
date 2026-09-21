// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "sapient/backends_cpu/kernels/elementwise.hpp"
#include "sapient/core/tensor.hpp"

using namespace sapient::backends_cpu::kernels::elementwise;
using sapient::core::Shape;
using sapient::core::Tensor;

namespace {
// Rust: fn t(data: &[f32]) -> Tensor { Tensor::from_f32(data, vec![data.len()]).unwrap() }
Tensor t(std::vector<float> data) {
    const size_t n = data.size();
    auto r = Tensor::from_f32_vec(std::move(data), Shape{n});
    if (!r) throw std::runtime_error(r.error().to_string());
    return std::move(*r);
}
std::vector<float> vec(const Tensor& x) {
    const auto s = x.f32_slice();
    return {s.begin(), s.end()};
}
} // namespace

TEST(Elementwise, test_add) {
    auto r = add(t({1.0f, 2.0f}), t({3.0f, 4.0f}));
    ASSERT_TRUE(r.has_value());
    EXPECT_LT(std::fabs(r->f32_slice()[0] - 4.0f), 1e-6f);
}

TEST(Elementwise, test_relu) {
    auto r = relu(t({-1.0f, 0.0f, 1.0f}));
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(vec(*r), (std::vector<float>{0.0f, 0.0f, 1.0f}));
}

TEST(Elementwise, test_sigmoid) {
    auto r = sigmoid(t({0.0f}));
    ASSERT_TRUE(r.has_value());
    EXPECT_LT(std::fabs(r->f32_slice()[0] - 0.5f), 1e-6f);
}

TEST(Elementwise, test_gelu) {
    auto r = gelu(t({0.0f}));
    ASSERT_TRUE(r.has_value());
    EXPECT_LT(std::fabs(r->f32_slice()[0]), 1e-5f);
}

TEST(Elementwise, test_erf) {
    const float v = erf_approx(0.0f);
    EXPECT_LT(std::fabs(v), 1e-6f) << "erf(0) should be ~0, got " << v;
}

TEST(Elementwise, test_gelu_erf) {
    // Exact GELU: g(0)=0, g(1)=0.8413447, g(-1)=-0.1586553.
    auto out = gelu_erf(t({0.0f, 1.0f, -1.0f}));
    ASSERT_TRUE(out.has_value());
    const auto v = vec(*out);
    EXPECT_LT(std::fabs(v[0]), 1e-6f);
    EXPECT_LT(std::fabs(v[1] - 0.8413447f), 1e-4f) << "g(1)=" << v[1];
    EXPECT_LT(std::fabs(v[2] - (-0.1586553f)), 1e-4f) << "g(-1)=" << v[2];
}

TEST(Elementwise, test_scalar_broadcast) {
    auto r = mul(t({1.0f, 2.0f, 3.0f}), t({2.0f}));
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(vec(*r), (std::vector<float>{2.0f, 4.0f, 6.0f}));
}

// C++-only: the two Result-path messages are parity-bound.
TEST(Elementwise, error_messages_match_rust) {
    auto bytes = std::vector<uint8_t>(4, 0);
    auto h = Tensor::from_f16_bytes(bytes, Shape{2});
    ASSERT_TRUE(h.has_value());
    auto e = neg(*h);
    ASSERT_FALSE(e.has_value());
    EXPECT_EQ(e.error().to_string(), "Type mismatch: expected f32, got f16");
    auto s = add(t({1.0f, 2.0f}), t({1.0f, 2.0f, 3.0f}));
    ASSERT_FALSE(s.has_value());
    EXPECT_EQ(s.error().to_string(), "Shape mismatch: expected [2], got [3]");
}

// C++-only: Rust f32::signum canonicalises NaN (f32::NAN = 0x7fc00000); the payload must not leak through.
TEST(Elementwise, erf_approx_canonicalises_nan_like_rust_signum) {
    const float payload_nan = std::bit_cast<float>(0x7fc12345u);
    EXPECT_EQ(std::bit_cast<uint32_t>(erf_approx(payload_nan)), 0x7fc00000u);
    EXPECT_LE(erf_approx(-0.0f), 0.0f);
}
