// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Rust-compatible env-knob parsing, shared by the rayon twin (RAYON_NUM_THREADS), matmul
// (SAPIENT_GEMV_TPC) and plan E's spinpool knobs — the C++ spelling of
// `std::env::var(name).ok().and_then(|v| v.parse::<usize>().ok())`.

#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <optional>
#include <string_view>
#include <system_error>

namespace sapient::backends_cpu {

/// `usize::from_str` applied to `getenv(name)`: an optional leading '+', then decimal digits only
/// (no whitespace, no '-'). nullopt when the variable is unset, empty or unparsable.
inline std::optional<size_t> env_usize(const char* name) {
    const char* s = std::getenv(name);
    if (s == nullptr) return std::nullopt;
    std::string_view v(s);
    if (!v.empty() && v.front() == '+') v.remove_prefix(1);
    if (v.empty()) return std::nullopt;
    size_t out = 0;
    const auto r = std::from_chars(v.data(), v.data() + v.size(), out);
    if (r.ec != std::errc{} || r.ptr != v.data() + v.size()) return std::nullopt;
    return out;
}

} // namespace sapient::backends_cpu
