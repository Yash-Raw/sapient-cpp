// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/backends_cpu/spinpool.hpp"

#include "sapient/backends_cpu/parallel.hpp"

namespace sapient::backends_cpu::spinpool {

bool enabled() {
    return false;
} // plan E: SAPIENT_SPINPOOL / platform default && thermal check

size_t parallelism() {
    return parallel::num_threads();
} // plan E: workers + 1

} // namespace sapient::backends_cpu::spinpool
