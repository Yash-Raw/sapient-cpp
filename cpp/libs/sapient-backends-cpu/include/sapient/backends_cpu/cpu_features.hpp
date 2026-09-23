// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Cached runtime ISA probes — the twins of Rust's `is_aarch64_feature_detected!("dotprod"/"i8mm")`
// and `is_x86_feature_detected!("avx2") && ("fma")` in sapient-backends-cpu (porting map §2.1).
// NEON is compile-time on aarch64 and never probed, exactly as in Rust.

namespace sapient::backends_cpu::cpu_features {

/// aarch64 `dotprod` (ARMv8.2 SDOT); always false on other architectures.
bool has_dotprod();
/// aarch64 `i8mm` (ARMv8.6 SMMLA); always false elsewhere — including Windows/arm64, where
/// std_detect exposes no i8mm probe (not a release target).
bool has_i8mm();
/// x86_64 `avx2 && fma`, gated like std_detect on OS support for the YMM state (OSXSAVE and
/// XCR0 bits 1|2); always false elsewhere.
bool has_avx2_fma();

} // namespace sapient::backends_cpu::cpu_features
