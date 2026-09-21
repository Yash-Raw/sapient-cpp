// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/testing/compare.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <sstream>

namespace sapient::testing {

std::optional<std::filesystem::path> golden_dir() {
    const char* d = std::getenv("SAPIENT_GOLDEN_DIR");
    if (d == nullptr || *d == '\0') return std::nullopt;
    return std::filesystem::path(d);
}

std::optional<GoldenCase> load_golden_case(std::string_view name, std::string* why) {
    const auto dir = golden_dir();
    if (!dir) {
        if (why)
            *why = "SAPIENT_GOLDEN_DIR not set — run cpp/tests/parity/golden_dump.sh <dir> and "
                   "export it";
        return std::nullopt;
    }
    const auto file = *dir / (std::string(name) + ".sapd");
    if (!std::filesystem::exists(file)) {
        if (why)
            *why = "golden case missing: " + file.string() +
                   " (regenerate the dumps with the current Rust tree)";
        return std::nullopt;
    }
    std::string err;
    auto c = read_golden(file, &err);
    if (!c) {
        if (why) *why = err;
        return std::nullopt;
    }
    return c;
}

::testing::AssertionResult bit_identical(std::span<const float> got, std::span<const float> ref) {
    if (got.size() != ref.size())
        return ::testing::AssertionFailure() << "length " << got.size() << " != " << ref.size();
    size_t bad = 0;
    std::ostringstream os;
    for (size_t i = 0; i < got.size(); ++i) {
        const uint32_t g = std::bit_cast<uint32_t>(got[i]);
        const uint32_t r = std::bit_cast<uint32_t>(ref[i]);
        if (g == r) continue;
        if (bad < 5)
            os << "\n  [" << i << "] got " << got[i] << " (0x" << std::hex << std::setw(8)
               << std::setfill('0') << g << ") ref " << std::dec << ref[i] << " (0x" << std::hex
               << std::setw(8) << std::setfill('0') << r << ")" << std::dec;
        ++bad;
    }
    if (bad == 0) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure()
           << bad << " of " << got.size() << " values differ bitwise:" << os.str();
}

float max_abs_err(std::span<const float> got, std::span<const float> ref) {
    float m = 0.0f;
    const size_t n = got.size() < ref.size() ? got.size() : ref.size();
    for (size_t i = 0; i < n; ++i)
        m = std::fmax(m, std::fabs(got[i] - ref[i]));
    return m;
}

::testing::AssertionResult
within_abs(std::span<const float> got, std::span<const float> ref, float tol) {
    if (got.size() != ref.size())
        return ::testing::AssertionFailure() << "length " << got.size() << " != " << ref.size();
    const float m = max_abs_err(got, ref);
    if (m <= tol) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << "max abs err " << m << " > " << tol;
}

} // namespace sapient::testing
