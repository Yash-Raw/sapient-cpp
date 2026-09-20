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
