// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
// Port of the `#[cfg(test)] mod tests` of crates/sapient-backends/cpu/src/thermal.rs — all 6 Rust
// tests by name. The four governor tests build a fake sysfs root and hand it to the constructor
// directly; they never set SAPIENT_THERMAL_PATH (that variable only feeds the process-global
// singleton, which must stay untouched so the rest of the binary sees a real, inert governor).
// `external_level_caps_effective_threads` is the ONLY test that touches process-global state.
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

#include "sapient/backends_cpu/parallel.hpp"
#include "sapient/backends_cpu/thermal.hpp"

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;
namespace thermal = sapient::backends_cpu::thermal;
using thermal::ThermalGovernor;

namespace {

int current_pid() {
#if defined(_WIN32)
    return _getpid();
#else
    return static_cast<int>(::getpid());
#endif
}

// Rust `fake_sysfs`: `<dir>/thermal_zone{zone}/temp` containing "{millideg}\n".
void fake_sysfs(const fs::path& dir, size_t zone, int64_t millideg) {
    const fs::path z = dir / ("thermal_zone" + std::to_string(zone));
    std::error_code ec;
    fs::create_directories(z, ec);
    ASSERT_FALSE(ec) << "create_directories " << z.string() << ": " << ec.message();
    std::ofstream f(z / "temp", std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(f.is_open()) << "open " << (z / "temp").string();
    f << millideg << "\n";
}

// Rust `tmp(name)`: a unique fake-sysfs root per test (gtest runs tests in one process, but a
// shared directory would still let one test's zone files leak into another's governor).
fs::path tmp(const char* name) {
    std::error_code ec;
    const fs::path d =
        fs::temp_directory_path(ec) /
        ("sapient-thermal-test-" + std::to_string(current_pid()) + "-" + std::string(name));
    fs::remove_all(d, ec);
    ec.clear();
    fs::create_directories(d, ec);
    EXPECT_FALSE(ec) << "create_directories " << d.string() << ": " << ec.message();
    return d;
}

// Restores the process-global external level whatever a test body does (including a failed
// assertion that returns early). Without this, a mid-test failure would leave the level at 3 and
// every later `spinpool::enabled()` in this binary would return false.
struct ExternalLevelGuard {
    ~ExternalLevelGuard() { thermal::set_external_thermal_level(0); }
};

} // namespace

TEST(Thermal, no_zones_is_inert) {
    const fs::path d = tmp("inert");
    const ThermalGovernor g(d, 80, 70, 8);
    EXPECT_FALSE(g.is_active());
    EXPECT_EQ(g.sample(), 8u) << "no zones → full threads";
}

TEST(Thermal, hot_steps_down_to_floor_and_cool_restores) {
    const fs::path d = tmp("steps");
    fake_sysfs(d, 0, 85000);
    const ThermalGovernor g(d, 80, 70, 4);
    ASSERT_TRUE(g.is_active());
    EXPECT_EQ(g.sample(), 3u);
    EXPECT_EQ(g.sample(), 2u);
    EXPECT_EQ(g.sample(), 2u) << "floor at half the cores — never collapses";

    fake_sysfs(d, 0, 60000);
    EXPECT_EQ(g.sample(), 3u);
    EXPECT_EQ(g.sample(), 4u);
    EXPECT_EQ(g.sample(), 4u) << "capped at full threads";
}

TEST(Thermal, hysteresis_holds_between_thresholds) {
    const fs::path d = tmp("hysteresis");
    fake_sysfs(d, 0, 85000);
    const ThermalGovernor g(d, 80, 70, 4);
    g.sample();
    EXPECT_EQ(g.effective(), 3u);
    fake_sysfs(d, 0, 75000); // between cool (70) and hot (80)
    EXPECT_EQ(g.sample(), 3u) << "holds inside the hysteresis band";
}

TEST(Thermal, hottest_zone_wins) {
    const fs::path d = tmp("hottest");
    fake_sysfs(d, 0, 50000);
    fake_sysfs(d, 1, 90000);
    const ThermalGovernor g(d, 80, 70, 4);
    ASSERT_TRUE(g.max_temp_mc().has_value());
    EXPECT_EQ(*g.max_temp_mc(), 90000);
    EXPECT_EQ(g.sample(), 3u) << "backs off on the hottest zone";
}

TEST(Thermal, external_cap_mapping) {
    // 8 cores: nominal full, fair 6, serious 4, critical 2.
    EXPECT_EQ(thermal::detail::external_cap(0, 8), 8u);
    EXPECT_EQ(thermal::detail::external_cap(1, 8), 6u);
    EXPECT_EQ(thermal::detail::external_cap(2, 8), 4u);
    EXPECT_EQ(thermal::detail::external_cap(3, 8), 2u);
    // Never below one thread, even on tiny core counts.
    EXPECT_EQ(thermal::detail::external_cap(3, 1), 1u);
    EXPECT_EQ(thermal::detail::external_cap(2, 1), 1u);
    // Levels past critical clamp to the critical cap.
    EXPECT_EQ(thermal::detail::external_cap(7, 8), 2u);
}

// The global external level caps `effective_threads()` and releasing it restores full
// parallelism. Serialized within this one test (the level is process-global); no other test in
// this binary reads `effective_threads()`.
TEST(Thermal, external_level_caps_effective_threads) {
    const ExternalLevelGuard restore;
    const size_t max = std::max<size_t>(sapient::backends_cpu::parallel::num_threads(), 1);
    thermal::set_external_thermal_level(0);
    EXPECT_EQ(thermal::effective_threads(), max);
    thermal::set_external_thermal_level(3);
    EXPECT_EQ(thermal::effective_threads(), std::max<size_t>(max / 4, 1));
    EXPECT_EQ(thermal::external_thermal_level(), 3);
    // Clamped, not wrapped.
    thermal::set_external_thermal_level(200);
    EXPECT_EQ(thermal::external_thermal_level(), 3);
    thermal::set_external_thermal_level(0);
    EXPECT_EQ(thermal::effective_threads(), max) << "release restores full threads";
}
