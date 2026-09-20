// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <regex>
#include <string>

#include "sapient/core/version.hpp"

TEST(Version, MatchesCMakeProjectVersion) {
    // SAPIENT_VERSION_STRING is injected by CMake from ../Cargo.toml [workspace.package].
    EXPECT_EQ(sapient::core::version(), SAPIENT_VERSION_STRING);
}

TEST(Version, IsSemver) {
    const std::string v(sapient::core::version());
    EXPECT_TRUE(std::regex_match(v, std::regex(R"(\d+\.\d+\.\d+)"))) << v;
}
