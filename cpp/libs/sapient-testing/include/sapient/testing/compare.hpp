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

// Local counterpart of sapient/core/error.hpp's SAPIENT_CAT/SAPIENT_CAT_ (not included here to
// keep sapient::testing independent of sapient::core) — used below to give SAPIENT_GOLDEN_CASE's
// temporaries __COUNTER__-unique names, like SAPIENT_TRY, so two cases can share a test body.
#define SAPIENT_TESTING_CAT_(a, b) a##b
#define SAPIENT_TESTING_CAT(a, b) SAPIENT_TESTING_CAT_(a, b)

/// Bind `var` to the named golden case. Two distinct bail-out behaviours:
///   - `SAPIENT_GOLDEN_DIR` unset: dumps were never generated for this host — SKIP (visible,
///     not a silent pass).
///   - `SAPIENT_GOLDEN_DIR` set but the named case is missing/unreadable: FAIL. A stale local
///     dump directory (e.g. after this task added new cases) must not silently turn a
///     bit-identity gate into a skip.
// `var`, `why_var`, `opt_var` below are declarator names, never expressions — parenthesizing them
// (bugprone-macro-parentheses' usual advice) is inapplicable, the same reasoning as
// sapient/core/error.hpp's SAPIENT_TRY_IMPL.
// NOLINTBEGIN(bugprone-macro-parentheses)
#define SAPIENT_GOLDEN_CASE_IMPL(var, why_var, opt_var, name)                                      \
    std::string why_var;                                                                           \
    auto opt_var = ::sapient::testing::load_golden_case((name), &why_var);                         \
    if (!opt_var.has_value()) {                                                                    \
        if (::sapient::testing::golden_dir().has_value()) {                                        \
            FAIL() << why_var;                                                                     \
        } else {                                                                                   \
            GTEST_SKIP() << why_var;                                                               \
        }                                                                                          \
    }                                                                                              \
    const ::sapient::testing::GoldenCase& var = *opt_var

#define SAPIENT_GOLDEN_CASE(var, name)                                                             \
    SAPIENT_GOLDEN_CASE_IMPL(var,                                                                  \
                             SAPIENT_TESTING_CAT(sapient_golden_why_, __COUNTER__),                \
                             SAPIENT_TESTING_CAT(sapient_golden_opt_, __COUNTER__),                \
                             name)
// NOLINTEND(bugprone-macro-parentheses)
