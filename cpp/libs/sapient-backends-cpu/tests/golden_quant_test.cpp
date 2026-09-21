// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
// Plan D gate: the quantized kernels vs the Rust oracle's dumps (SAPIENT_GOLDEN_DIR; unset → SKIP,
// set-but-missing → FAIL). Everything here is BIT-IDENTICAL (integer arrays exactly, f32 arrays on
// their bit patterns) — no quantized path touches sgemm. The `_q8k_off` cases need
// SAPIENT_Q8K_ACT=0 in the process environment (the `sapient_backends_cpu_tests.q8k_off` ctest
// entry sets it); under the default environment they skip on aarch64 and simply run elsewhere
// (Rust's knob is aarch64-only, so the results are knob-independent on x86).
#include <gtest/gtest.h>

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sapient/backends_cpu/kernels/matmul.hpp"
#include "sapient/backends_cpu/kernels/quant.hpp"
#include "sapient/core/dtype.hpp"
#include "sapient/core/shape.hpp"
#include "sapient/core/tensor.hpp"
#include "sapient/testing/compare.hpp"

namespace quant = sapient::backends_cpu::kernels::quant;
namespace matmul = sapient::backends_cpu::kernels::matmul;
using sapient::core::DType;
using sapient::core::Shape;
using sapient::core::Tensor;
using sapient::testing::bit_identical;
using sapient::testing::exact_equal;
using sapient::testing::GoldenCase;

namespace {

Tensor tensor_of(const GoldenCase& c, std::string_view name) {
    const auto& a = c.get(name);
    std::vector<size_t> dims(a.dims.begin(), a.dims.end());
    auto t = Tensor::from_f32_vec(a.as<float>(), Shape(dims));
    if (!t) throw std::runtime_error(std::string(name) + ": " + t.error().to_string());
    return std::move(*t);
}
Tensor quant_tensor_of(const GoldenCase& c, DType dtype) {
    const auto bytes = c.get("in:w_blocks").as<uint8_t>();
    const auto shape = c.get("param:w_shape").as<uint32_t>();
    auto t = Tensor::from_quant_bytes(bytes, Shape{shape.at(0), shape.at(1)}, dtype);
    if (!t) throw std::runtime_error("in:w_blocks: " + t.error().to_string());
    return std::move(*t);
}
template <class T>
::testing::AssertionResult same(const std::vector<T>& got, const std::vector<T>& ref) {
    return exact_equal<T>(std::span<const T>(got), std::span<const T>(ref));
}
std::vector<float> f1(float v) {
    return {v};
}

bool q8k_on() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return matmul::detail::q8k_activations();
#else
    return false;
#endif
}

// One matmul_nt golden case: x from `in:x`, W from the raw quantized bytes, y bit-identical.
void check_matmul_nt(const char* name, DType dtype) {
    SAPIENT_GOLDEN_CASE(c, name);
    auto y = matmul::matmul_nt(tensor_of(c, "in:x"), quant_tensor_of(c, dtype));
    ASSERT_TRUE(y.has_value()) << y.error().to_string();
    EXPECT_TRUE(bit_identical(y->to_f32_vec(), c.get("out:y").as<float>()));
}
void check_matmul_nt_q8k_off(const char* name, DType dtype) {
    if (q8k_on())
        GTEST_SKIP() << "needs SAPIENT_Q8K_ACT=0 (run by the sapient_backends_cpu_tests.q8k_off "
                        "ctest entry)";
    check_matmul_nt(name, dtype);
}

} // namespace

// ── block quantizers and row dots (sub-project 0 cases, consumed here for the first time) ──

TEST(GoldenQuant, quantize_q8_0_block) {
    SAPIENT_GOLDEN_CASE(c, "quantize_q8_0_block");
    const auto blk = quant::quantize_q8_0_block(c.get("in:x").as<float>());
    EXPECT_TRUE(
        same(std::vector<uint8_t>(blk.begin(), blk.end()), c.get("out:block").as<uint8_t>()));
}
TEST(GoldenQuant, quantize_q4_0_block) {
    SAPIENT_GOLDEN_CASE(c, "quantize_q4_0_block");
    const auto blk = quant::quantize_q4_0_block(c.get("in:x").as<float>());
    EXPECT_TRUE(
        same(std::vector<uint8_t>(blk.begin(), blk.end()), c.get("out:block").as<uint8_t>()));
}
TEST(GoldenQuant, dot_q8_0_row_f32) {
    SAPIENT_GOLDEN_CASE(c, "dot_q8_0_row_f32");
    const float y =
        quant::dot_q8_0_row_f32(c.get("in:row_blocks").as<uint8_t>(), c.get("in:x").as<float>());
    EXPECT_TRUE(bit_identical(f1(y), c.get("out:y").as<float>()));
}
TEST(GoldenQuant, dot_q4_0_row_f32) {
    SAPIENT_GOLDEN_CASE(c, "dot_q4_0_row_f32");
    const float y =
        quant::dot_q4_0_row_f32(c.get("in:row_blocks").as<uint8_t>(), c.get("in:x").as<float>());
    EXPECT_TRUE(bit_identical(f1(y), c.get("out:y").as<float>()));
}
TEST(GoldenQuant, dot_q4_k_row_f32) {
    SAPIENT_GOLDEN_CASE(c, "dot_q4_k_row_f32");
    const float y =
        quant::dot_q4_k_row_f32(c.get("in:row_blocks").as<uint8_t>(), c.get("in:x").as<float>());
    EXPECT_TRUE(bit_identical(f1(y), c.get("out:y").as<float>()));
}
TEST(GoldenQuant, dot_q5_k_row_f32) {
    SAPIENT_GOLDEN_CASE(c, "dot_q5_k_row_f32");
    const float y =
        quant::dot_q5_k_row_f32(c.get("in:row_blocks").as<uint8_t>(), c.get("in:x").as<float>());
    EXPECT_TRUE(bit_identical(f1(y), c.get("out:y").as<float>()));
}
TEST(GoldenQuant, dot_q6_k_row_f32) {
    SAPIENT_GOLDEN_CASE(c, "dot_q6_k_row_f32");
    const float y =
        quant::dot_q6_k_row_f32(c.get("in:row_blocks").as<uint8_t>(), c.get("in:x").as<float>());
    EXPECT_TRUE(bit_identical(f1(y), c.get("out:y").as<float>()));
}

// ── activation quantisers and repacks (plan D cases) ──

TEST(GoldenQuant, quantize_row_to_i8_blocks) {
    SAPIENT_GOLDEN_CASE(c, "quantize_row_to_i8_blocks");
    const auto r = quant::quantize_row_to_i8_blocks(c.get("in:x").as<float>());
    EXPECT_TRUE(same(r.q, c.get("out:q").as<int8_t>()));
    EXPECT_TRUE(bit_identical(r.scales, c.get("out:scales").as<float>()));
}
TEST(GoldenQuant, i8_block_sums) {
    SAPIENT_GOLDEN_CASE(c, "i8_block_sums");
    EXPECT_TRUE(
        same(quant::i8_block_sums(c.get("in:q").as<int8_t>()), c.get("out:sums").as<int32_t>()));
}
TEST(GoldenQuant, quantize_row_to_q8k) {
    SAPIENT_GOLDEN_CASE(c, "quantize_row_to_q8k");
    const auto r = quant::quantize_row_to_q8k(c.get("in:x").as<float>());
    EXPECT_TRUE(same(r.q, c.get("out:q").as<int8_t>()));
    EXPECT_TRUE(bit_identical(r.scales, c.get("out:scales").as<float>()));
    EXPECT_TRUE(same(r.sums, c.get("out:sums").as<int32_t>()));
}
TEST(GoldenQuant, repack_q4_k_rows4) {
    SAPIENT_GOLDEN_CASE(c, "repack_q4_k_rows4");
    const auto shape = c.get("param:shape").as<uint32_t>();
    EXPECT_TRUE(
        same(quant::repack_q4_k_rows4(c.get("in:blocks").as<uint8_t>(), shape.at(0), shape.at(1)),
             c.get("out:packed").as<uint8_t>()));
}
TEST(GoldenQuant, repack_q6_k_rows4) {
    SAPIENT_GOLDEN_CASE(c, "repack_q6_k_rows4");
    const auto shape = c.get("param:shape").as<uint32_t>();
    EXPECT_TRUE(
        same(quant::repack_q6_k_rows4(c.get("in:blocks").as<uint8_t>(), shape.at(0), shape.at(1)),
             c.get("out:packed").as<uint8_t>()));
}

// ── matmul_nt over every quantized dtype (default SAPIENT_Q8K_ACT) ──

TEST(GoldenQuant, matmul_nt_q8_0_m1) {
    check_matmul_nt("matmul_nt_q8_0_m1", DType::Q8_0);
}
TEST(GoldenQuant, matmul_nt_q8_0_m3) {
    check_matmul_nt("matmul_nt_q8_0_m3", DType::Q8_0);
}
TEST(GoldenQuant, matmul_nt_q8_0_m8) {
    check_matmul_nt("matmul_nt_q8_0_m8", DType::Q8_0);
} // blocked GEMM
TEST(GoldenQuant, matmul_nt_q4_0_m1) {
    check_matmul_nt("matmul_nt_q4_0_m1", DType::Q4_0);
}
TEST(GoldenQuant, matmul_nt_q4_0_m3) {
    check_matmul_nt("matmul_nt_q4_0_m3", DType::Q4_0);
}
TEST(GoldenQuant, matmul_nt_q5_k_m1) {
    check_matmul_nt("matmul_nt_q5_k_m1", DType::Q5_K);
}
TEST(GoldenQuant, matmul_nt_q4_k_m1) {
    check_matmul_nt("matmul_nt_q4_k_m1", DType::Q4_K);
}
TEST(GoldenQuant, matmul_nt_q4_k_m3) {
    check_matmul_nt("matmul_nt_q4_k_m3", DType::Q4_K);
}
TEST(GoldenQuant, matmul_nt_q6_k_m1) {
    check_matmul_nt("matmul_nt_q6_k_m1", DType::Q6_K);
}
TEST(GoldenQuant, matmul_nt_q6_k_m3) {
    check_matmul_nt("matmul_nt_q6_k_m3", DType::Q6_K);
}
TEST(GoldenQuant, matmul_nt_q4_k_r4_m1) {
    check_matmul_nt("matmul_nt_q4_k_r4_m1", DType::Q4_K_R4);
}
TEST(GoldenQuant, matmul_nt_q4_k_r4_m2) {
    check_matmul_nt("matmul_nt_q4_k_r4_m2", DType::Q4_K_R4);
}
TEST(GoldenQuant, matmul_nt_q4_k_r4_m3) {
    check_matmul_nt("matmul_nt_q4_k_r4_m3", DType::Q4_K_R4);
}
TEST(GoldenQuant, matmul_nt_q4_k_r4_m8) {
    check_matmul_nt("matmul_nt_q4_k_r4_m8", DType::Q4_K_R4);
}
TEST(GoldenQuant, matmul_nt_q6_k_r4_m1) {
    check_matmul_nt("matmul_nt_q6_k_r4_m1", DType::Q6_K_R4);
}
TEST(GoldenQuant, matmul_nt_q6_k_r4_m2) {
    check_matmul_nt("matmul_nt_q6_k_r4_m2", DType::Q6_K_R4);
}
TEST(GoldenQuant, matmul_nt_q6_k_r4_m3) {
    check_matmul_nt("matmul_nt_q6_k_r4_m3", DType::Q6_K_R4);
}
TEST(GoldenQuant, matmul_nt_q6_k_r4_m8) {
    check_matmul_nt("matmul_nt_q6_k_r4_m8", DType::Q6_K_R4);
}

// ── the same twelve knob-sensitive cases under SAPIENT_Q8K_ACT=0 (per-32 W4A8/W6A8 kernels) ──

TEST(GoldenQuant, matmul_nt_q4_k_m1_q8k_off) {
    check_matmul_nt_q8k_off("matmul_nt_q4_k_m1_q8k_off", DType::Q4_K);
}
TEST(GoldenQuant, matmul_nt_q4_k_m3_q8k_off) {
    check_matmul_nt_q8k_off("matmul_nt_q4_k_m3_q8k_off", DType::Q4_K);
}
TEST(GoldenQuant, matmul_nt_q6_k_m1_q8k_off) {
    check_matmul_nt_q8k_off("matmul_nt_q6_k_m1_q8k_off", DType::Q6_K);
}
TEST(GoldenQuant, matmul_nt_q6_k_m3_q8k_off) {
    check_matmul_nt_q8k_off("matmul_nt_q6_k_m3_q8k_off", DType::Q6_K);
}
TEST(GoldenQuant, matmul_nt_q4_k_r4_m1_q8k_off) {
    check_matmul_nt_q8k_off("matmul_nt_q4_k_r4_m1_q8k_off", DType::Q4_K_R4);
}
TEST(GoldenQuant, matmul_nt_q4_k_r4_m2_q8k_off) {
    check_matmul_nt_q8k_off("matmul_nt_q4_k_r4_m2_q8k_off", DType::Q4_K_R4);
}
TEST(GoldenQuant, matmul_nt_q4_k_r4_m3_q8k_off) {
    check_matmul_nt_q8k_off("matmul_nt_q4_k_r4_m3_q8k_off", DType::Q4_K_R4);
}
TEST(GoldenQuant, matmul_nt_q4_k_r4_m8_q8k_off) {
    check_matmul_nt_q8k_off("matmul_nt_q4_k_r4_m8_q8k_off", DType::Q4_K_R4);
}
TEST(GoldenQuant, matmul_nt_q6_k_r4_m1_q8k_off) {
    check_matmul_nt_q8k_off("matmul_nt_q6_k_r4_m1_q8k_off", DType::Q6_K_R4);
}
TEST(GoldenQuant, matmul_nt_q6_k_r4_m2_q8k_off) {
    check_matmul_nt_q8k_off("matmul_nt_q6_k_r4_m2_q8k_off", DType::Q6_K_R4);
}
TEST(GoldenQuant, matmul_nt_q6_k_r4_m3_q8k_off) {
    check_matmul_nt_q8k_off("matmul_nt_q6_k_r4_m3_q8k_off", DType::Q6_K_R4);
}
TEST(GoldenQuant, matmul_nt_q6_k_r4_m8_q8k_off) {
    check_matmul_nt_q8k_off("matmul_nt_q6_k_r4_m8_q8k_off", DType::Q6_K_R4);
}
