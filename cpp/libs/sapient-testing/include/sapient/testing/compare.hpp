// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Comparison helpers for the golden-dump gates (spec §4): bit-identical for quant/integer paths,
// absolute-error bounds for the sgemm-backed float paths. Test-support only.

#include <gtest/gtest.h>

#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "sapient/testing/golden.hpp"

namespace sapient::testing {

/// `SAPIENT_GOLDEN_DIR`, if set.
std::optional<std::filesystem::path> golden_dir();
/// `<golden_dir>/<name>.sapd`; on failure `*why` says whether the env var is unset, the file is
/// missing, or the parse failed.
std::optional<GoldenCase> load_golden_case(std::string_view name, std::string* why);

/// Exact comparison on the f32 bit patterns (so NaN payloads and -0 count as differences).
::testing::AssertionResult bit_identical(std::span<const float> got, std::span<const float> ref);
/// Max |got[i] - ref[i]| over the shorter of the two spans (mismatched lengths are not an error
/// here — `within_abs` below rejects that separately). NaN rule: a NaN in one span where the
/// other is finite at the same index is a genuine mismatch and returns +infinity (IEEE `fmax`
/// would silently drop the NaN operand and let it contribute 0 error — not used here for exactly
/// that reason); a NaN in BOTH spans at the same index is treated as equal (0 error).
float max_abs_err(std::span<const float> got, std::span<const float> ref);
/// `max_abs_err(got, ref) <= tol`, plus a length check. On a NaN-vs-finite mismatch, fails with
/// the first offending index and both values (e.g. "index 17: got nan, ref 0.25") rather than
/// the generic "max abs err inf > tol", since the source of an infinite error is otherwise
/// opaque.
::testing::AssertionResult
within_abs(std::span<const float> got, std::span<const float> ref, float tol);

} // namespace sapient::testing

/// Bind `var` to the named golden case. Two distinct bail-out behaviours:
///   - `SAPIENT_GOLDEN_DIR` unset: dumps were never generated for this host — SKIP (visible,
///     not a silent pass).
///   - `SAPIENT_GOLDEN_DIR` set but the named case is missing/unreadable: FAIL. A stale local
///     dump directory (e.g. after this task added new cases) must not silently turn a
///     bit-identity gate into a skip.
#define SAPIENT_GOLDEN_CASE(var, name)                                                             \
    std::string sapient_golden_why_;                                                               \
    auto sapient_golden_opt_ = ::sapient::testing::load_golden_case((name), &sapient_golden_why_); \
    if (!sapient_golden_opt_.has_value()) {                                                        \
        if (::sapient::testing::golden_dir().has_value()) {                                        \
            FAIL() << sapient_golden_why_;                                                         \
        } else {                                                                                   \
            GTEST_SKIP() << sapient_golden_why_;                                                   \
        }                                                                                          \
    }                                                                                              \
    const ::sapient::testing::GoldenCase& var = *sapient_golden_opt_
