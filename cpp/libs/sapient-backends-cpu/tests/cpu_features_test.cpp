// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include "sapient/backends_cpu/cpu_features.hpp"

using namespace sapient::backends_cpu::cpu_features;

// C++-only: the probes are cached, consistent across calls, and platform-plausible. The values
// themselves are host facts (this Mac: dotprod=1, i8mm=1 via sysctl hw.optional.arm.FEAT_*).
TEST(CpuFeatures, probes_are_cached_and_platform_consistent) {
    EXPECT_EQ(has_dotprod(), has_dotprod());
    EXPECT_EQ(has_i8mm(), has_i8mm());
    EXPECT_EQ(has_avx2_fma(), has_avx2_fma());
#if defined(__aarch64__) || defined(_M_ARM64)
    EXPECT_FALSE(has_avx2_fma());
    if (has_i8mm()) EXPECT_TRUE(has_dotprod()) << "ARMv8.6 i8mm implies ARMv8.2 dotprod";
#else
    EXPECT_FALSE(has_dotprod());
    EXPECT_FALSE(has_i8mm());
#endif
    RecordProperty("dotprod", has_dotprod() ? 1 : 0);
    RecordProperty("i8mm", has_i8mm() ? 1 : 0);
    RecordProperty("avx2_fma", has_avx2_fma() ? 1 : 0);
}
