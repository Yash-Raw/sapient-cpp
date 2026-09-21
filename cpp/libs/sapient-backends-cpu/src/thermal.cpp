// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/backends_cpu/thermal.hpp"

#include "sapient/backends_cpu/parallel.hpp"

namespace sapient::backends_cpu::thermal {

size_t effective_threads() {
    return parallel::num_threads();
} // plan E: governor/external min

void tick() {} // plan E: 500 ms rate-limited sysfs sample

} // namespace sapient::backends_cpu::thermal
