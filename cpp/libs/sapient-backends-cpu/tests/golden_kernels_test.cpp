// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
// Plan C gate: the dense kernels vs the Rust oracle's dumps (SAPIENT_GOLDEN_DIR; unset → SKIP,
// set-but-missing → FAIL). Bit-identical everywhere except the sgemm-backed cases — matmul_nt_f32_m1
// (k=64 < 512 takes sgemm in Rust too), matmul_nt_f32_m4, conv2d_s1/s2 — which use the spec §4
// tolerance 1e-5·max(1, max|ref|). The six matmul_nt_q*_m{1,3} dumps are plan D's.
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sapient/backends_cpu/kernels.hpp"
#include "sapient/core/tensor.hpp"
#include "sapient/testing/compare.hpp"

using namespace sapient::backends_cpu::kernels;
using sapient::core::Shape;
using sapient::core::Tensor;
using sapient::testing::bit_identical;
using sapient::testing::GoldenCase;
using sapient::testing::within_rel_of_max;

namespace {

constexpr float SGEMM_REL = 1e-5f; // spec §4

Tensor tensor_of(const GoldenCase& c, std::string_view name) {
    const auto& a = c.get(name);
    std::vector<size_t> dims(a.dims.begin(), a.dims.end());
    auto t = Tensor::from_f32_vec(a.as<float>(), Shape(dims));
    if (!t) throw std::runtime_error(std::string(name) + ": " + t.error().to_string());
    return std::move(*t);
}
std::vector<float> ref(const GoldenCase& c, std::string_view name = "out:y") {
    return c.get(name).as<float>();
}
template <class T> T param(const GoldenCase& c, std::string_view name) {
    return c.get(name).as<T>().at(0);
}
std::vector<size_t> positions_of(const GoldenCase& c) {
    const auto p = c.get("param:positions").as<uint64_t>();
    return {p.begin(), p.end()};
}

} // namespace

// ── norms / activations / softmax ────────────────────────────────────────────

TEST(GoldenKernels, rms_norm) {
    SAPIENT_GOLDEN_CASE(c, "rms_norm");
    const auto x = tensor_of(c, "in:x");
    const auto w = tensor_of(c, "in:weight");
    auto y = layernorm::rms_norm(x, &w, param<float>(c, "param:eps"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

TEST(GoldenKernels, layer_norm) {
    SAPIENT_GOLDEN_CASE(c, "layer_norm");
    const auto x = tensor_of(c, "in:x");
    const auto w = tensor_of(c, "in:weight");
    const auto b = tensor_of(c, "in:bias");
    auto y = layernorm::layer_norm(x, &w, &b, -1, param<float>(c, "param:eps"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

TEST(GoldenKernels, softmax) {
    SAPIENT_GOLDEN_CASE(c, "softmax");
    auto y = softmax::softmax(tensor_of(c, "in:x"), param<int32_t>(c, "param:axis"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

TEST(GoldenKernels, softmax_axis0) {
    SAPIENT_GOLDEN_CASE(c, "softmax_axis0");
    auto y = softmax::softmax(tensor_of(c, "in:x"), param<int32_t>(c, "param:axis"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

TEST(GoldenKernels, log_softmax) {
    SAPIENT_GOLDEN_CASE(c, "log_softmax");
    auto y = softmax::log_softmax(tensor_of(c, "in:x"), param<int32_t>(c, "param:axis"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

TEST(GoldenKernels, silu) {
    SAPIENT_GOLDEN_CASE(c, "silu");
    auto y = elementwise::silu(tensor_of(c, "in:x"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

TEST(GoldenKernels, gelu_erf) {
    SAPIENT_GOLDEN_CASE(c, "gelu_erf");
    auto y = elementwise::gelu_erf(tensor_of(c, "in:x"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

TEST(GoldenKernels, gelu) {
    SAPIENT_GOLDEN_CASE(c, "gelu");
    auto y = elementwise::gelu(tensor_of(c, "in:x"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

TEST(GoldenKernels, reduce) {
    SAPIENT_GOLDEN_CASE(c, "reduce");
    const auto x = tensor_of(c, "in:x");
    const int64_t ax1[] = {1};
    const int64_t ax0[] = {0};
    const int64_t axn1[] = {-1};
    auto s = reduce::reduce_sum(x, ax1, false);
    ASSERT_TRUE(s.has_value()) << s.error().to_string();
    EXPECT_TRUE(bit_identical(s->to_f32_vec(), ref(c, "out:sum_axis1")));
    auto m = reduce::reduce_mean(x, {}, false);
    ASSERT_TRUE(m.has_value()) << m.error().to_string();
    EXPECT_TRUE(m->shape().dims.empty()) << "all-axes mean is a scalar";
    EXPECT_TRUE(bit_identical(m->to_f32_vec(), ref(c, "out:mean_all")));
    auto mx = reduce::reduce_max(x, ax0, true);
    ASSERT_TRUE(mx.has_value()) << mx.error().to_string();
    EXPECT_EQ(mx->shape().dims, (std::vector<size_t>{1, 3, 4}));
    EXPECT_TRUE(bit_identical(mx->to_f32_vec(), ref(c, "out:max_axis0_keep")));
    auto mn = reduce::reduce_min(x, axn1, false);
    ASSERT_TRUE(mn.has_value()) << mn.error().to_string();
    EXPECT_TRUE(bit_identical(mn->to_f32_vec(), ref(c, "out:min_axis_neg1")));
}

// ── RoPE (powf/sinf/cosf — the libm-portability risk, spec §7) ───────────────

TEST(GoldenKernels, apply_rope) {
    SAPIENT_GOLDEN_CASE(c, "apply_rope");
    auto y = rope::apply_rope(tensor_of(c, "in:x"), positions_of(c), param<float>(c, "param:base"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

TEST(GoldenKernels, apply_rope_partial) {
    SAPIENT_GOLDEN_CASE(c, "apply_rope_partial");
    auto y = rope::apply_rope_partial(tensor_of(c, "in:x"),
                                      positions_of(c),
                                      param<float>(c, "param:base"),
                                      param<uint32_t>(c, "param:rotary_dim"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

TEST(GoldenKernels, apply_rope_partial_scaled) {
    SAPIENT_GOLDEN_CASE(c, "apply_rope_partial_scaled");
    auto y = rope::apply_rope_partial_scaled(tensor_of(c, "in:x"),
                                             positions_of(c),
                                             param<float>(c, "param:base"),
                                             param<uint32_t>(c, "param:rotary_dim"),
                                             param<float>(c, "param:pos_scale"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

// ── attention ────────────────────────────────────────────────────────────────

TEST(GoldenKernels, attention_prefill) {
    SAPIENT_GOLDEN_CASE(c, "attention_prefill");
    auto y = attention::scaled_dot_product_attention(tensor_of(c, "in:q"),
                                                     tensor_of(c, "in:k"),
                                                     tensor_of(c, "in:v"),
                                                     nullptr,
                                                     std::nullopt,
                                                     param<uint32_t>(c, "param:n_kv_heads"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

TEST(GoldenKernels, attention_decode) {
    SAPIENT_GOLDEN_CASE(c, "attention_decode");
    auto y = attention::scaled_dot_product_attention(tensor_of(c, "in:q"),
                                                     tensor_of(c, "in:k"),
                                                     tensor_of(c, "in:v"),
                                                     nullptr,
                                                     std::nullopt,
                                                     param<uint32_t>(c, "param:n_kv_heads"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

TEST(GoldenKernels, attention_masked) {
    SAPIENT_GOLDEN_CASE(c, "attention_masked");
    const auto mask = tensor_of(c, "in:mask");
    auto y = attention::scaled_dot_product_attention(tensor_of(c, "in:q"),
                                                     tensor_of(c, "in:k"),
                                                     tensor_of(c, "in:v"),
                                                     &mask,
                                                     param<float>(c, "param:scale"),
                                                     param<uint32_t>(c, "param:n_kv_heads"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

// ── matmul_nt float paths ────────────────────────────────────────────────────

TEST(GoldenKernels, matmul_nt_f32_m1_k512) { // dot_f32_fast GEMV: bit-identical
    SAPIENT_GOLDEN_CASE(c, "matmul_nt_f32_m1_k512");
    auto y = matmul::matmul_nt(tensor_of(c, "in:x"), tensor_of(c, "in:w"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

TEST(GoldenKernels,
     matmul_nt_f16_m1) { // dot_f32_x_f16 GEMV incl. the reproduced mis-decode: bit-identical
    SAPIENT_GOLDEN_CASE(c, "matmul_nt_f16_m1");
    const auto shape = c.get("param:w_shape").as<uint32_t>();
    auto w = Tensor::from_f16_bytes(c.get("in:w_f16").as<uint8_t>(), Shape({shape[0], shape[1]}));
    ASSERT_TRUE(w.has_value()) << w.error().to_string();
    auto y = matmul::matmul_nt(tensor_of(c, "in:x"), *w);
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), ref(c)));
}

TEST(GoldenKernels, matmul_nt_f32_m1_sgemm_tolerance) { // k=64 < 512: sgemm in Rust too
    SAPIENT_GOLDEN_CASE(c, "matmul_nt_f32_m1");
    auto y = matmul::matmul_nt(tensor_of(c, "in:x"), tensor_of(c, "in:w"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(within_rel_of_max(y->to_f32_vec(), ref(c), SGEMM_REL));
}

TEST(GoldenKernels, matmul_nt_f32_m4_sgemm_tolerance) {
    SAPIENT_GOLDEN_CASE(c, "matmul_nt_f32_m4");
    auto y = matmul::matmul_nt(tensor_of(c, "in:x"), tensor_of(c, "in:w"));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(within_rel_of_max(y->to_f32_vec(), ref(c), SGEMM_REL));
}

// ── conv2d (sgemm-backed) ────────────────────────────────────────────────────

namespace {
void check_conv(const char* name) {
    SAPIENT_GOLDEN_CASE(c, name);
    const auto b = tensor_of(c, "in:bias");
    const auto p = c.get("param:pads").as<uint32_t>();
    const auto s = c.get("param:strides").as<uint32_t>();
    auto y = conv2d::conv2d(tensor_of(c, "in:x"),
                            tensor_of(c, "in:w"),
                            &b,
                            {3, 3},
                            {p[0], p[1], p[2], p[3]},
                            {s[0], s[1]},
                            {1, 1},
                            1);
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(within_rel_of_max(y->to_f32_vec(), ref(c), SGEMM_REL));
}
} // namespace

TEST(GoldenKernels, conv2d_s1_sgemm_tolerance) {
    check_conv("conv2d_s1");
}
TEST(GoldenKernels, conv2d_s2_sgemm_tolerance) {
    check_conv("conv2d_s2");
}
