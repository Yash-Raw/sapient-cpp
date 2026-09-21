// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-backends/cpu/src/thermal.rs — PLAN E. Plan C ships only the two entry
// points `matmul` needs, with inert bodies that equal Rust's behaviour on a host without thermal
// zones and no external level: plan E replaces the bodies (governor, external cap, tick sampling)
// and adds ThermalGovernor, set_external_thermal_level, external_thermal_level.

#include <cstddef>

namespace sapient::backends_cpu::thermal {

/// Rust `thermal::effective_threads()`: the stricter of the sysfs governor and the external level.
/// Plan C: always `parallel::num_threads()` (no governor, level 0).
size_t effective_threads();

/// Rust `thermal::tick()`: rate-limited governor sample at the top of `matmul_nt`.
/// Plan C: no-op (no governor).
void tick();

} // namespace sapient::backends_cpu::thermal
