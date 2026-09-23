// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-backends/cpu/src/thermal.rs (plan E). Passively-cooled boards hit their
// firmware trip point under sustained decode and every core hard-throttles; backing off BEFORE the
// trip sustains a higher steady state. This module reads the Linux thermal zones
// (`/sys/class/thermal/thermal_zone*/temp`, millidegrees) and lowers the effective parallelism
// target the GEMV chunker sizes tasks for — fewer, larger tasks than there are threads leaves the
// surplus idle (a pool cannot be resized at runtime), cutting package power so the clocks stay up.
//
// One sample per 500 ms at most; every other `tick()` is a single atomic compare. Hysteresis steps
// the target down one core at/above the hot threshold (default 80 °C) and back up at/below the cool
// threshold (default 70 °C); the floor is half the cores — graceful, never collapse. On machines
// with no thermal zones (macOS, Windows, containers) the governor is inert and `effective_threads()`
// is exactly `parallel::num_threads()`.
//
// Mobile (roadmap 11.3) has no sysfs: the host app feeds the OS thermal signal in through
// `set_external_thermal_level`, which caps the same target. When both sources are active the
// STRICTER one wins.
//
// Env: `SAPIENT_THERMAL=off` disables both mechanisms; `SAPIENT_THERMAL_HOT`/`_COOL` set the
// thresholds in °C; `SAPIENT_THERMAL_PATH` overrides the sysfs root (the process-global singleton
// only — the tests construct `ThermalGovernor` directly).

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

namespace sapient::backends_cpu::thermal {

/// Hysteresis governor over a set of sysfs thermal zones. Pure state machine — constructed
/// directly in tests with a fake sysfs root; the process-wide singleton (built from env) lives
/// behind `tick()`/`effective_threads()`.
///
/// `sample()` is `const` with a mutable atomic, mirroring Rust's `&self` receiver: the singleton
/// is handed out as a `const ThermalGovernor*` and still has to step its own state.
class ThermalGovernor {
public:
    /// Scan `root` for `thermal_zone*/temp` files. `hot_c`/`cool_c` are in °C; `max_threads` is the
    /// full parallelism to restore to when cool. A missing or unreadable `root` yields no zones and
    /// an inert governor — never an exception (see the `std::error_code` rule in the plan).
    ThermalGovernor(const std::filesystem::path& root,
                    int64_t hot_c,
                    int64_t cool_c,
                    size_t max_threads);

    ThermalGovernor(const ThermalGovernor&) = delete;
    ThermalGovernor& operator=(const ThermalGovernor&) = delete;

    /// True when the machine exposes at least one thermal zone.
    bool is_active() const;

    /// Hottest zone in millidegrees, or nullopt when nothing is readable.
    std::optional<int64_t> max_temp_mc() const;

    /// Current effective thread target.
    size_t effective() const;

    /// Take one temperature sample and step the target: −1 core at/above hot (floored at half the
    /// cores), +1 at/below cool (capped at full). Between the thresholds the target holds
    /// (hysteresis). Returns the new target.
    size_t sample() const;

private:
    std::vector<std::filesystem::path> zones_;
    int64_t hot_mc_;
    int64_t cool_mc_;
    size_t max_threads_;
    size_t min_threads_;
    mutable std::atomic<size_t> effective_;
    mutable std::atomic<bool> warned_;
};

/// The parallelism target GEMV chunking should size tasks for: the full thread count, reduced while
/// the sysfs governor is backing off and/or a host-fed external level caps it (the stricter source
/// wins). Cheap (two atomic loads) — called per matmul.
size_t effective_threads();

/// Rate-limited thermal sample: at most one sysfs read per 500 ms; all other calls are a single
/// atomic compare. Called from `matmul_nt`'s entry.
void tick();

/// Feed the host OS's thermal state into the governor: 0 nominal, 1 fair/moderate, 2
/// serious/severe, 3 critical (levels > 3 clamp to critical). Cheap and thread-safe — call it
/// straight from the OS callback. No-op while `SAPIENT_THERMAL=off`.
void set_external_thermal_level(uint8_t level);

/// The last level fed to `set_external_thermal_level` (0 when never set).
uint8_t external_thermal_level();

namespace detail {
/// Thread cap for an external thermal level over `max` cores. Nominal runs full; fair sheds a
/// quarter; serious halves (the sysfs governor's floor); critical quarters — on mobile, critical
/// means the OS is about to act, so dropping below the sysfs floor is deliberate. Private in Rust;
/// exposed here because `external_cap_mapping` tests it.
size_t external_cap(uint8_t level, size_t max);
} // namespace detail

} // namespace sapient::backends_cpu::thermal
