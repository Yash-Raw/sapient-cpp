// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <fstream>
#include <iterator>
#include <regex>
#include <string>

#include "sapient/core/version.hpp"

TEST(Version, MatchesCMakeProjectVersion) {
    // This only checks that the getter round-trips the value CMake configured it with
    // (SAPIENT_VERSION_STRING, injected from ../Cargo.toml [workspace.package] or from
    // -DSAPIENT_VERSION). It does NOT exercise the Cargo.toml extraction path itself —
    // see Version.MatchesWorkspaceCargoToml below for that.
    EXPECT_EQ(sapient::core::version(), SAPIENT_VERSION_STRING);
}

TEST(Version, IsSemver) {
    const std::string v(sapient::core::version());
    EXPECT_TRUE(std::regex_match(v, std::regex(R"(\d+\.\d+\.\d+)"))) << v;
}

// Exercises the real extraction path: reads the same file CMake read and compares.
TEST(Version, MatchesWorkspaceCargoToml) {
    std::ifstream in(SAPIENT_CARGO_TOML_PATH);
    if (!in) GTEST_SKIP() << "no ../Cargo.toml (Rust tree removed) — version now comes from -DSAPIENT_VERSION";
    const std::string toml((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    // Custom raw-string delimiter (`re`): the pattern's own trailing `)"`  (capture-group
    // close + literal closing quote) would otherwise collide with the default `)"` raw-string
    // terminator and truncate the literal early.
    const std::regex re(R"re(\[workspace\.package\]\s*version\s*=\s*"(\d+\.\d+\.\d+)")re");
    std::smatch m;
    ASSERT_TRUE(std::regex_search(toml, m, re)) << "regex found no [workspace.package] version";
    EXPECT_EQ(sapient::core::version(), m[1].str());
}
