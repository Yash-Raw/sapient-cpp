// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
//
// Build-flag discipline (spec D1): bit-identity of the quantized kernels with the Rust oracle
// requires that the compiler never contracts `a*b + c` into a fused multiply-add.
#include <gtest/gtest.h>

#include <cmath>

#ifdef __FAST_MATH__
#error "-ffast-math is set: forbidden by spec D1 (breaks kernel bit-identity with the Rust build)"
#endif

static_assert(__cplusplus >= 202002L, "SAPIENT C++ requires C++20");

TEST(BuildFlags, FpContractIsOff) {
    // a*b = (1+2^-23)(1-2^-23) = 1 - 2^-46 exactly. Rounded to float that is 1.0f, so the
    // two-step evaluation gives (1.0f) + (-1.0f) == 0. A fused multiply-add keeps the exact
    // product and yields -2^-46 instead. `volatile` blocks constant folding.
    volatile float a = 1.0f + 0x1p-23f;
    volatile float b = 1.0f - 0x1p-23f;
    volatile float c = -1.0f;
    const float r = a * b + c;
    EXPECT_EQ(r, 0.0f) << "a*b+c was contracted into an FMA: the build must use -ffp-contract=off";
    // Sanity: a real FMA does differ on these inputs, so the assertion above is meaningful.
    EXPECT_NE(std::fma(static_cast<float>(a), static_cast<float>(b), static_cast<float>(c)), 0.0f);
}

#if defined(__x86_64__) || defined(_M_X64)
// On x86-64 BASELINE (no target attribute), clang cannot contract a*b+c into an FMA even with
// -ffp-contract=on: the baseline ISA has no fma instruction to contract into, so the plain
// BuildFlags.FpContractIsOff test above would pass on x86-64 even with -ffp-contract=off
// missing from the build — it is not testing anything there. 1a's real AVX2/FMA kernels live in
// functions marked `__attribute__((target("avx2,fma")))`, where the fma instruction IS
// available, so THIS is the honest gate for those translation units: it reproduces the same
// target-attribute context and checks -ffp-contract=off still holds inside it. clang-cl accepts
// the same `__attribute__((target(...)))` syntax as clang, so no #ifdef is needed per frontend.
__attribute__((target("fma"))) float fma_target_probe() {
    volatile float a = 1.0f + 0x1p-23f;
    volatile float b = 1.0f - 0x1p-23f;
    volatile float c = -1.0f;
    return a * b + c;
}

TEST(BuildFlags, FpContractIsOffUnderFmaTarget) {
    if (!__builtin_cpu_supports("fma")) {
        GTEST_SKIP() << "host has no FMA";
    }
    EXPECT_EQ(fma_target_probe(), 0.0f)
        << "a*b+c was contracted into an FMA inside a target(\"fma\") function: "
           "-ffp-contract=off must hold even where the ISA has an fma instruction to use";
}
#endif
