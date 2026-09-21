// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-backends/cpu/src/spinpool.rs — PLAN E. Plan C ships only the two entry
// points `matmul::gemv_chunk` needs, with the values Rust yields under `SAPIENT_SPINPOOL=0`:
// plan E replaces the bodies and adds SpinPool, pool(), and the env knobs.

#include <cstddef>

namespace sapient::backends_cpu::spinpool {

/// Rust `spinpool::enabled()`: env/platform default AND thermal::effective_threads() >=
/// parallel::num_threads(). Plan C: always false (the pool does not exist yet).
bool enabled();

/// Rust `spinpool::parallelism()` = workers + 1. Plan C: `parallel::num_threads()`.
size_t parallelism();

} // namespace sapient::backends_cpu::spinpool
