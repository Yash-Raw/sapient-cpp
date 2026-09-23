// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "sapient/backends_cpu/kernels/conv2d.hpp"
#include "sapient/core/tensor.hpp"

using namespace sapient::backends_cpu::kernels::conv2d;
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
std::vector<float> lcg(size_t n, uint64_t seed) {
    std::vector<float> v(n);
    for (float& x : v) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        x = (static_cast<float>(seed >> 40) / static_cast<float>(1ULL << 24)) * 2.0f - 1.0f;
    }
    return v;
}
} // namespace

TEST(Conv2d, conv2d_identity_kernel) {
    // 1×1 conv with identity weight.
    const auto x = f32({1.0f, 2.0f, 3.0f, 4.0f}, Shape{1, 1, 2, 2});
    const auto w = f32({1.0f}, Shape{1, 1, 1, 1});
    auto y = conv2d(x, w, nullptr, {1, 1}, {0, 0, 0, 0}, {1, 1}, {1, 1}, 1);
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_EQ(vec(*y), (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));
}

// C++-only: padding, stride, dilation, groups and bias against a naive double reference (the
// Rust unit test only covers the identity kernel; the golden dumps cover groups=1).
TEST(Conv2d, matches_naive_reference_with_padding_stride_dilation_groups) {
    const size_t n = 2, c_in = 4, h_in = 7, w_in = 6, c_out = 6, groups = 2, kh = 3, kw = 2;
    const size_t c_in_g = c_in / groups, c_out_g = c_out / groups;
    const std::array<size_t, 4> pads = {1, 0, 2, 1};
    const std::array<size_t, 2> strides = {2, 1};
    const std::array<size_t, 2> dilations = {1, 2};
    const auto xv = lcg(n * c_in * h_in * w_in, 11);
    const auto wv = lcg(c_out * c_in_g * kh * kw, 12);
    const auto bv = lcg(c_out, 13);
    const auto x = f32(xv, Shape{n, c_in, h_in, w_in});
    const auto w = f32(wv, Shape{c_out, c_in_g, kh, kw});
    const auto b = f32(bv, Shape{c_out});
    const uint64_t im2col_before = IM2COL_NS.load();
    const uint64_t gemm_before = GEMM_NS.load();
    auto y = conv2d(x, w, &b, {kh, kw}, pads, strides, dilations, groups);
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    const size_t h_out = (h_in + pads[0] + pads[2] - dilations[0] * (kh - 1) - 1) / strides[0] + 1;
    const size_t w_out = (w_in + pads[1] + pads[3] - dilations[1] * (kw - 1) - 1) / strides[1] + 1;
    EXPECT_EQ(y->shape().dims, (std::vector<size_t>{n, c_out, h_out, w_out}));
    EXPECT_GE(IM2COL_NS.load(), im2col_before);
    EXPECT_GE(GEMM_NS.load(), gemm_before);

    const auto got = vec(*y);
    float max_ref = 1.0f;
    std::vector<float> ref(got.size(), 0.0f);
    for (size_t bi = 0; bi < n; ++bi)
        for (size_t co = 0; co < c_out; ++co) {
            const size_t g = co / c_out_g;
            for (size_t oh = 0; oh < h_out; ++oh)
                for (size_t ow = 0; ow < w_out; ++ow) {
                    double acc = bv[co];
                    for (size_t cg = 0; cg < c_in_g; ++cg)
                        for (size_t ki = 0; ki < kh; ++ki)
                            for (size_t kj = 0; kj < kw; ++kj) {
                                const long ih =
                                    static_cast<long>(oh * strides[0] + ki * dilations[0]) -
                                    static_cast<long>(pads[0]);
                                const long iw =
                                    static_cast<long>(ow * strides[1] + kj * dilations[1]) -
                                    static_cast<long>(pads[1]);
                                if (ih < 0 || iw < 0 || ih >= static_cast<long>(h_in) ||
                                    iw >= static_cast<long>(w_in))
                                    continue;
                                const size_t ci = g * c_in_g + cg;
                                acc += static_cast<double>(
                                           xv[((bi * c_in + ci) * h_in + static_cast<size_t>(ih)) *
                                                  w_in +
                                              static_cast<size_t>(iw)]) *
                                       static_cast<double>(
                                           wv[((co * c_in_g + cg) * kh + ki) * kw + kj]);
                            }
                    const size_t idx = ((bi * c_out + co) * h_out + oh) * w_out + ow;
                    ref[idx] = static_cast<float>(acc);
                    max_ref = std::max(max_ref, std::fabs(ref[idx]));
                }
        }
    for (size_t i = 0; i < got.size(); ++i)
        ASSERT_NEAR(got[i], ref[i], 1e-4f * max_ref) << "index " << i;
}

// C++-only: the parity-bound InvalidGraph text and the rank check order.
TEST(Conv2d, error_messages_match_rust) {
    const auto x = f32(std::vector<float>(1 * 4 * 2 * 2, 0.0f), Shape{1, 4, 2, 2});
    const auto w = f32(std::vector<float>(3 * 2 * 1 * 1, 0.0f), Shape{3, 2, 1, 1});
    auto e = conv2d(x, w, nullptr, {1, 1}, {0, 0, 0, 0}, {1, 1}, {1, 1}, 3);
    ASSERT_FALSE(e.has_value());
    EXPECT_EQ(e.error().to_string(),
              "Graph validation failed: conv2d: groups=3, c_in=4, c_in/group=2: 2*3!=c_in");
    const auto x3 = f32(std::vector<float>(4, 0.0f), Shape{1, 2, 2});
    EXPECT_EQ(conv2d(x3, w, nullptr, {1, 1}, {0, 0, 0, 0}, {1, 1}, {1, 1}, 1).error().to_string(),
              "Rank mismatch: expected 4, got 3");
}
