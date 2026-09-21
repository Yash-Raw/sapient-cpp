// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// The rayon stand-in (spec §2.3 `parallel`). rayon is not ported: what parity depends on is the
// chunk→range PARTITION of `par_chunks_mut`, which is deterministic and thread-independent, so a
// persistent pool that hands out chunk indices reproduces every kernel's per-slot result exactly.
// Work stealing, `join`, and rayon's sleep protocol are deliberately absent.

#include <cstddef>
#include <functional>
#include <span>

namespace sapient::backends_cpu::parallel {

/// `rayon::current_num_threads()` of the global pool (rayon-core 1.13 rules): `RAYON_NUM_THREADS`
/// parsed as usize — ≥1 → that; 0 → the default; unset/unparsable → `RAYON_RS_NUM_CPUS` with the
/// same rule; else the logical CPU count (≥1). Computed once, at first call.
size_t num_threads();

/// Calls `f(i)` for every `i` in `[0, n)` exactly once, on the calling thread and the pool's
/// workers, and returns when all have completed. Re-entrant: `f` may itself call `par_for`.
void par_for(size_t n, const std::function<void(size_t)>& f);

/// rayon `out.par_chunks_mut(chunk).enumerate().for_each(|(ci, cs)| f(ci, cs))`: chunk `ci` is
/// `out[ci*chunk, min((ci+1)*chunk, out.size()))`. An empty `out` makes no calls. `chunk == 0`
/// panics (rayon: "chunk size must not be zero").
void par_chunks_mut(std::span<float> out,
                    size_t chunk,
                    const std::function<void(size_t, std::span<float>)>& f);

} // namespace sapient::backends_cpu::parallel
