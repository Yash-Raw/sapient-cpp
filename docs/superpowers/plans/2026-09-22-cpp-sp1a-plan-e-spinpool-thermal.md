# Sub-project 1a, Plan E: `sapient::backends_cpu` spin pool + thermal governor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the two inert plan-C stubs with the real ports of `crates/sapient-backends/cpu/src/thermal.rs` (347 lines) and `crates/sapient-backends/cpu/src/spinpool.rs` (541 lines), wire the spin-pool branch of `matmul::detail::for_each_out_chunk` (plus the `SAPIENT_SPINPOOL_DEBUG` census), reproduce all 6 thermal + 5 spinpool Rust unit tests (and the one `#[ignore]` perf probe) by name, and prove through two new ctest entries that the whole golden suite is bit-identical with the pool ON and with the pool OFF.

**Architecture:** Two `.hpp/.cpp` pairs mirroring the Rust modules. `thermal` is a pure hysteresis state machine over sorted `thermal_zone*/temp` files (constructed directly in tests against a fake sysfs root) plus a process-global singleton, an external 4-level cap, and a 500 ms rate-limited `tick()`. `spinpool` is a direct port of the seqlock op-handoff protocol: a leaked `SpinPool` with 128-byte-padded hot atomics, `workers` detached threads that spin then park on a condvar, guided contiguous block claiming, and a publisher that participates in its own op. `matmul::detail::for_each_out_chunk` gains the `spinpool::enabled()` branch; both branches produce the identical `(chunk index → [start, end))` partition, which is the entire basis of bit-identity.

**Tech Stack:** C++20, CMake presets from sub-project 0, GoogleTest, `sapient::core` (`panic.hpp`), `sapient::backends_cpu` (plan C: `parallel`, `env.hpp`, the `matmul` dispatcher), `std::atomic`/`std::mutex`/`std::condition_variable`/`std::thread`, `std::filesystem` (error-code overloads only), macOS `pthread_set_qos_class_self_np`. No new third-party dependencies.

**Spec:** `docs/superpowers/specs/2026-09-21-cpp-sp1a-core-io-cpu-design.md` (§2.3 `spinpool` + `thermal` rows, §3.7, §4, §5 row E) under `docs/superpowers/specs/2026-09-20-cpp-rewrite-design.md`. **Porting map (read §2.10, §3.1, §3.2, §6a, §6b, §7 before touching a file):** `docs/superpowers/notes/2026-09-21-sp1a-porting-map-cpu-kernels.md`. **Plan D** (the conventions and the golden suite this plan re-runs): `docs/superpowers/plans/2026-09-21-cpp-sp1a-plan-d-quant-kernels.md`.

## Global Constraints

- **Branch:** `feat/cpp-sp1a` (stacked on `feat/cpp-sp0-scaffold`; plan D is complete at 39742bf). Worktree `.claude/worktrees/feat-cpp-sp0-scaffold`. Commit after every task, with a blank line before the `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>` trailer (copy it byte-for-byte). **Never push** — the user pushes and opens PRs.
- **Compiler/flags (programme spec D1 + spec §3.1):** Clang only; `-ffp-contract=off -fno-math-errno` are applied by `cpp/libs/CMakeLists.txt`; never add `-march=native`/`-ffast-math`. Build and test with `cd cpp && cmake --preset dev && cmake --build --preset dev && ctest --preset dev`. **Never run ctest from the repo root** (it writes a stray `Testing/` directory; plan D's lesson).
- **BEHAVIOURAL FLAG — read before Task 3.** Rust's `spinpool::enabled()` defaults to **ON** on macOS, but `enabled()` returning true changes nothing until `for_each_out_chunk` consults it. **Task 2 makes the pool exist and be correct; Task 3 is where the route actually flips** and every existing test on this Mac (quant, matmul, attention, golden — all 242) starts dispatching its GEMVs through the pool instead of `parallel::par_chunks_mut`. That is intended and Rust-faithful, and it makes the existing suite the pool's regression net. **Consequence: every task's verification step is the FULL `ctest --preset dev`, never a filtered subset.** A task that only runs its own new tests has not been verified.
- **`SAPIENT_THERMAL=off` breaks one thermal test, faithfully.** Rust's `set_external_thermal_level` early-returns when the mechanism is disabled, so `external_level_caps_effective_threads` asserts a cap that can never be applied under that variable — the Rust test would fail the same way. Never run the WHOLE suite with `SAPIENT_THERMAL=off`; the gate entry that needs it (Task 3's `spinpool_on`) filters to `Golden*` and the route probe, which is why it is unaffected.
- **`std::filesystem` throwing overloads are forbidden in this plan.** Rust's `if let Ok(entries) = std::fs::read_dir(root)` silently yields no zones when the root does not exist — which is the normal case on macOS/Windows (`/sys/class/thermal` is absent). `std::filesystem::directory_iterator(root)` without an `std::error_code` **throws**, and that exception would escape the function-local `static` initializer inside `governor()`, propagate through `effective_threads()` into `matmul_nt`, find no handler, and `std::terminate` the process on the first matmul. Every `directory_iterator`, `increment`, `is_regular_file`, `create_directories` and `remove_all` in `thermal.cpp` and `thermal_test.cpp` **must** take the `std::error_code&` overload. Do not "clean up" those parameters.
- **Memory orderings are copied verbatim from `spinpool.rs`; do not relitigate them.** Acquire on every park-loop `generation` load; **SeqCst** on `active.fetch_add`/`fetch_sub`, on the worker's authoritative `generation` load, on both publisher `generation.fetch_add`s, on the publisher's `active` drain load and on the `parked` add/sub/load; Release on `completed.fetch_add`; Acquire on the publisher's `completed` wait; Relaxed on `next_block.fetch_add` and the two counter resets. **Order matters — the odd bump goes strictly BEFORE the `active` drain** (porting map §6a; draining first leaves the window that produced the measured SIGSEGV, and `rapid_ops_with_constant_parking` is the test that pins it).
- **Warnings are errors** (`-Wall -Wextra -Wpedantic -Wshadow -Werror`). `spinpool.rs` shadows deliberately (`let g = loop { let g = … }`, and `g` again for the sleep-loop reload); C++ cannot, so those three reads get distinct names (`cur`, `woke`, and the outer `g`). Do not "fix" the shadowing by reusing one variable — the outer `g` must survive the inner loop.
- **CI clang-tidy gate** (`.clang-tidy`, `WarningsAsErrors: '*'`; CI-only on macOS; `just cpp-tidy`): widen before multiplying (`size_t{3} * participants`, `max * size_t{3} / 4`), take `std::function` by `const&`, explicit `static_cast` for every narrowing, no dead `using` declarations (`misc-unused-using-decls`), and no owning raw pointers outside the two documented leaks (`SpinPool::create`, whose comment says why).
- **Cross-compile the two new library TUs for x86_64** (plan D's rule; `enabled()`'s platform branches make x86 the "else → off" path that nothing on this Mac exercises). From `cpp/`:
  `clang++ -std=c++20 --target=x86_64-apple-macos -Wall -Wextra -Wpedantic -Wshadow -Werror -ffp-contract=off -fno-math-errno -Ilibs/sapient-core/include -Ilibs/sapient-backends-cpu/include -c libs/sapient-backends-cpu/src/thermal.cpp -o /dev/null`
  and the same for `src/spinpool.cpp`. `spinpool.cpp` additionally needs the `-DSAPIENT_TARGET_MACOS=0` probe build (both natively and cross) to reach its non-macOS arms, because an `x86_64-apple-macos` target still reports `TARGET_OS_OSX == 1` — see Task 2 Step 8. Unlike plan D, the new **test** files contain no architecture `#if` blocks at all — that whole trap class (an unbalanced `#endif` hiding tests from x86) does not apply here; keep it that way.
- **Windows is compile-only in CI and has never compiled any of these branches.** Keep `#ifdef _WIN32` code minimal and follow the include pattern already in `src/cpu_features.cpp` (`<windows.h>` inside the platform arm). `SetThreadDescription` for thread names and `_getpid()` from `<process.h>` for the test temp directory are the only Windows-specific calls; both are best-effort and neither is parity-bound.
- **Portability of the test binaries:** include every std header you use (`<algorithm>`, `<atomic>`, `<chrono>`, `<cstdint>`, `<cstdio>`, `<cstdlib>`, `<filesystem>`, `<fstream>`, `<string>`, `<string_view>`, `<system_error>`, `<thread>`, `<vector>`); **no gtest `ASSERT_*` inside a worker thread or inside a closure handed to `pool().run`/`par_for`** — accumulate into an atomic and assert on the main thread (Windows runs ctest, and `ASSERT_*` expands to `return`).
- **Formatting:** run the CI-pinned clang-format 18 (`.superpowers/tools-venv/bin/clang-format -i`, or `just cpp-fmt`) on every new/changed `.hpp/.cpp` **after `git add`**, then re-add.
- **Naming (spec D3):** `include/sapient/backends_cpu/{thermal,spinpool}.hpp`, `src/{thermal,spinpool}.cpp`, `tests/{thermal,spinpool}_test.cpp`; namespaces `sapient::backends_cpu::thermal` and `…::spinpool`; Rust's private-but-tested `external_cap` lives in `thermal::detail`; gtest suites `Thermal` and `Spinpool`; test names are the Rust test names verbatim.
- **SPDX header verbatim** on every new `.hpp/.cpp` (`//` form) and on `CMakeLists.txt` edits (`#` form):
  `// SPDX-License-Identifier: AGPL-3.0-only`
  `// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)`
- **Env-knob parsing differs per knob — do not reach for one helper.** `SAPIENT_SPINPOOL`: `v != "0"` (an *empty* value is ON, matching Rust's `map(|v| v != "0")`). `SAPIENT_THERMAL`: `eq_ignore_ascii_case("off")`. `SAPIENT_THERMAL_HOT`/`_COOL`: `v.trim().parse::<i64>()` — trims whitespace and accepts a leading `+`/`-`, neither of which `env_usize` does, so Task 1 writes a local `parse_i64_trimmed`. `SAPIENT_THERMAL_PATH`: raw path. `SAPIENT_SPINPOOL_WORKERS`/`_SPINS`/`_BLOCK`: `env_usize` (`_SPINS` cast to `uint64_t`, `_BLOCK` additionally filtered `>= 1`). `SAPIENT_SPINPOOL_DEBUG`: **presence only, read via `getenv` on every call** (Rust's `.is_ok()` is not cached). Caching rules mirror Rust exactly: `OnceLock` → `static const` initialised once; everything else re-read per call.
- **`std::hint::spin_loop()` twin:** `_mm_pause()` on x86_64, `__builtin_arm_isb(0xF)` on aarch64 (what rustc lowers `spin_loop` to there), nothing elsewhere. **Never `std::this_thread::yield()`** — it is a syscall and changes the parking dynamics the 4 000-iteration budget was measured against. Not parity-bound for results, but load-bearing for the behaviour the tests exercise.
- **Rust's `tracing::warn!`/`info!` become `std::fprintf(stderr, …)`** with the same message text (precedent: `cpp/libs/sapient-core/include/sapient/core/panic.hpp`). Not parity-bound — there is no logging framework in the C++ tree yet, and introducing one is out of scope for 1a.
- **Rust tree frozen.** Plan E adds **no** `dump_kernels.rs` cases and no Rust changes of any kind — the gate re-runs plan D's existing dumps in two process environments. `cargo test -p sapient-backends-cpu` must stay `71 passed; 1 ignored`; `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets -- -D warnings` stay clean.
- **No exceptions across library boundaries**; recoverable failures are `Result`, unrecoverable ones `sapient::core::panic()`. A `std::thread` constructor failure in `SpinPool::create` is Rust's `.expect("spawn spinpool worker")` → `panic("spawn spinpool worker")`. No new death tests: nothing in this plan panics observably on a supported path.
- **Docs rule:** Task 4 updates CLAUDE.md, docs/ROADMAP.md, docs/PARITY.md, docs/PROJECT_GUIDE.md, CHANGELOG.md and the spec's §4/§5 "as built" notes. CONTRIBUTING/README need no change for a library-internal plan; say so in the commit. `cpp/third_party/LICENSES.md` unchanged (no third-party code added).

## Execution notes (model selection)

- **Task 1 (thermal):** standard-tier implementer, mid-tier reviewer. Mechanical state machine; the only trap is the `error_code` rule.
- **Task 2 (spinpool):** **mid-tier implementer, most-capable-tier task reviewer.** The plan contains the complete code, so it *looks* like transcription — but a transcription error in a memory ordering is silent and passes the stress test most of the time. The reviewer's brief must say: *diff every `std::memory_order` argument against `spinpool.rs` line by line, and confirm the odd generation bump precedes the `active` drain.*
- **Task 3 (wiring + gate):** standard-tier implementer, mid-tier reviewer.
- **Task 4 (docs):** cheap-tier implementer, cheap-tier reviewer.

## File structure

```
cpp/libs/sapient-backends-cpu/include/sapient/backends_cpu/thermal.hpp    (Task 1: replaces the stub header)
cpp/libs/sapient-backends-cpu/src/thermal.cpp                              (Task 1: replaces the stub body)
cpp/libs/sapient-backends-cpu/tests/thermal_test.cpp                       (Task 1: new, 6 Rust tests)
cpp/libs/sapient-backends-cpu/include/sapient/backends_cpu/spinpool.hpp   (Task 2: replaces the stub header)
cpp/libs/sapient-backends-cpu/src/spinpool.cpp                             (Task 2: replaces the stub body)
cpp/libs/sapient-backends-cpu/tests/spinpool_test.cpp                      (Task 2: new, 5 Rust tests + the ignored probe; Task 3 appends the 2 route probes)
cpp/libs/sapient-backends-cpu/src/kernels/matmul.cpp                       (Task 3: the for_each_out_chunk pool branch + the debug census)
cpp/libs/sapient-backends-cpu/CMakeLists.txt                               (Task 1 + Task 2: new test sources; Task 3: the two gate entries)
docs: CLAUDE.md, docs/ROADMAP.md, docs/PARITY.md, docs/PROJECT_GUIDE.md, CHANGELOG.md, spec §4/§5 notes (Task 4)
```

Nothing under `crates/` is touched by any task.

---

### Task 1: `thermal.hpp/.cpp` — the hysteresis governor, the external level cap, `effective_threads()` and `tick()`

**Files:**
- Modify (replace the plan-C stub bodies wholesale): `cpp/libs/sapient-backends-cpu/include/sapient/backends_cpu/thermal.hpp`, `cpp/libs/sapient-backends-cpu/src/thermal.cpp`
- Create: `cpp/libs/sapient-backends-cpu/tests/thermal_test.cpp`
- Modify: `cpp/libs/sapient-backends-cpu/CMakeLists.txt` (add `tests/thermal_test.cpp` to `sapient_backends_cpu_tests`, and drop "thermal/spinpool are plan-E stubs until plan E lands" from the file's header comment once Task 2 lands — for now narrow it to spinpool only)

**Interfaces:**
- Consumes: `sapient::backends_cpu::parallel::num_threads()` (the `rayon::current_num_threads()` twin).
- Produces (namespace `sapient::backends_cpu::thermal`): `class ThermalGovernor` with `ThermalGovernor(const std::filesystem::path& root, int64_t hot_c, int64_t cool_c, size_t max_threads)`, `bool is_active() const`, `std::optional<int64_t> max_temp_mc() const`, `size_t effective() const`, `size_t sample() const`; free functions `size_t effective_threads()`, `void tick()`, `void set_external_thermal_level(uint8_t)`, `uint8_t external_thermal_level()`; `detail::external_cap(uint8_t level, size_t max) -> size_t`. Task 2's `spinpool::enabled()` consumes `effective_threads()`; `matmul::detail::gemv_chunk` already consumes `effective_threads()` and `matmul_nt` already calls `tick()` (plan C wired both against the stubs — no call-site change is needed here).

Rust reference: `crates/sapient-backends/cpu/src/thermal.rs` (all 347 lines). Porting map §6b and §2.10.

- [ ] **Step 1: Write the failing tests** — create `cpp/libs/sapient-backends-cpu/tests/thermal_test.cpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
// Port of the `#[cfg(test)] mod tests` of crates/sapient-backends/cpu/src/thermal.rs — all 6 Rust
// tests by name. The four governor tests build a fake sysfs root and hand it to the constructor
// directly; they never set SAPIENT_THERMAL_PATH (that variable only feeds the process-global
// singleton, which must stay untouched so the rest of the binary sees a real, inert governor).
// `external_level_caps_effective_threads` is the ONLY test that touches process-global state.
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

#include "sapient/backends_cpu/parallel.hpp"
#include "sapient/backends_cpu/thermal.hpp"

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;
namespace thermal = sapient::backends_cpu::thermal;
using thermal::ThermalGovernor;

namespace {

int current_pid() {
#if defined(_WIN32)
    return _getpid();
#else
    return static_cast<int>(::getpid());
#endif
}

// Rust `fake_sysfs`: `<dir>/thermal_zone{zone}/temp` containing "{millideg}\n".
void fake_sysfs(const fs::path& dir, size_t zone, int64_t millideg) {
    const fs::path z = dir / ("thermal_zone" + std::to_string(zone));
    std::error_code ec;
    fs::create_directories(z, ec);
    ASSERT_FALSE(ec) << "create_directories " << z.string() << ": " << ec.message();
    std::ofstream f(z / "temp", std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(f.is_open()) << "open " << (z / "temp").string();
    f << millideg << "\n";
}

// Rust `tmp(name)`: a unique fake-sysfs root per test (gtest runs tests in one process, but a
// shared directory would still let one test's zone files leak into another's governor).
fs::path tmp(const char* name) {
    std::error_code ec;
    const fs::path d =
        fs::temp_directory_path(ec) /
        ("sapient-thermal-test-" + std::to_string(current_pid()) + "-" + std::string(name));
    fs::remove_all(d, ec);
    ec.clear();
    fs::create_directories(d, ec);
    EXPECT_FALSE(ec) << "create_directories " << d.string() << ": " << ec.message();
    return d;
}

// Restores the process-global external level whatever a test body does (including a failed
// assertion that returns early). Without this, a mid-test failure would leave the level at 3 and
// every later `spinpool::enabled()` in this binary would return false.
struct ExternalLevelGuard {
    ~ExternalLevelGuard() { thermal::set_external_thermal_level(0); }
};

} // namespace

TEST(Thermal, no_zones_is_inert) {
    const fs::path d = tmp("inert");
    const ThermalGovernor g(d, 80, 70, 8);
    EXPECT_FALSE(g.is_active());
    EXPECT_EQ(g.sample(), 8u) << "no zones → full threads";
}

TEST(Thermal, hot_steps_down_to_floor_and_cool_restores) {
    const fs::path d = tmp("steps");
    fake_sysfs(d, 0, 85000);
    const ThermalGovernor g(d, 80, 70, 4);
    ASSERT_TRUE(g.is_active());
    EXPECT_EQ(g.sample(), 3u);
    EXPECT_EQ(g.sample(), 2u);
    EXPECT_EQ(g.sample(), 2u) << "floor at half the cores — never collapses";

    fake_sysfs(d, 0, 60000);
    EXPECT_EQ(g.sample(), 3u);
    EXPECT_EQ(g.sample(), 4u);
    EXPECT_EQ(g.sample(), 4u) << "capped at full threads";
}

TEST(Thermal, hysteresis_holds_between_thresholds) {
    const fs::path d = tmp("hysteresis");
    fake_sysfs(d, 0, 85000);
    const ThermalGovernor g(d, 80, 70, 4);
    g.sample();
    EXPECT_EQ(g.effective(), 3u);
    fake_sysfs(d, 0, 75000); // between cool (70) and hot (80)
    EXPECT_EQ(g.sample(), 3u) << "holds inside the hysteresis band";
}

TEST(Thermal, hottest_zone_wins) {
    const fs::path d = tmp("hottest");
    fake_sysfs(d, 0, 50000);
    fake_sysfs(d, 1, 90000);
    const ThermalGovernor g(d, 80, 70, 4);
    ASSERT_TRUE(g.max_temp_mc().has_value());
    EXPECT_EQ(*g.max_temp_mc(), 90000);
    EXPECT_EQ(g.sample(), 3u) << "backs off on the hottest zone";
}

TEST(Thermal, external_cap_mapping) {
    // 8 cores: nominal full, fair 6, serious 4, critical 2.
    EXPECT_EQ(thermal::detail::external_cap(0, 8), 8u);
    EXPECT_EQ(thermal::detail::external_cap(1, 8), 6u);
    EXPECT_EQ(thermal::detail::external_cap(2, 8), 4u);
    EXPECT_EQ(thermal::detail::external_cap(3, 8), 2u);
    // Never below one thread, even on tiny core counts.
    EXPECT_EQ(thermal::detail::external_cap(3, 1), 1u);
    EXPECT_EQ(thermal::detail::external_cap(2, 1), 1u);
    // Levels past critical clamp to the critical cap.
    EXPECT_EQ(thermal::detail::external_cap(7, 8), 2u);
}

// The global external level caps `effective_threads()` and releasing it restores full
// parallelism. Serialized within this one test (the level is process-global); no other test in
// this binary reads `effective_threads()`.
TEST(Thermal, external_level_caps_effective_threads) {
    const ExternalLevelGuard restore;
    const size_t max = std::max<size_t>(sapient::backends_cpu::parallel::num_threads(), 1);
    thermal::set_external_thermal_level(0);
    EXPECT_EQ(thermal::effective_threads(), max);
    thermal::set_external_thermal_level(3);
    EXPECT_EQ(thermal::effective_threads(), std::max<size_t>(max / 4, 1));
    EXPECT_EQ(thermal::external_thermal_level(), 3);
    // Clamped, not wrapped.
    thermal::set_external_thermal_level(200);
    EXPECT_EQ(thermal::external_thermal_level(), 3);
    thermal::set_external_thermal_level(0);
    EXPECT_EQ(thermal::effective_threads(), max) << "release restores full threads";
}
```

Add `tests/thermal_test.cpp` to the `add_executable(sapient_backends_cpu_tests …)` source list in `cpp/libs/sapient-backends-cpu/CMakeLists.txt` (append after `tests/cpu_features_test.cpp`, keeping the existing single-line style), and narrow the file's header comment from "thermal/spinpool are plan-E stubs until plan E lands" to "spinpool is a plan-E stub until Task 2 lands".

Note `<algorithm>` is needed for `std::max` — add it to the include list above if your compiler does not pull it in transitively (it must be explicit; the Global Constraints require every used header).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd cpp && cmake --build --preset dev 2>&1 | tail -20`
Expected: a COMPILE failure — `no member named 'ThermalGovernor' in namespace 'sapient::backends_cpu::thermal'` (the stub header declares only `effective_threads` and `tick`).

- [ ] **Step 3: Write the header**

Replace `cpp/libs/sapient-backends-cpu/include/sapient/backends_cpu/thermal.hpp` entirely:

```cpp
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
```

- [ ] **Step 4: Write the implementation**

Replace `cpp/libs/sapient-backends-cpu/src/thermal.cpp` entirely:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/backends_cpu/thermal.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <system_error>

#include "sapient/backends_cpu/parallel.hpp"

namespace fs = std::filesystem;

namespace sapient::backends_cpu::thermal {
namespace {

/// Minimum interval between temperature reads (Rust `TICK_MS`).
constexpr uint64_t kTickMs = 500;
/// Default backoff threshold, °C (Pi firmware throttles at 85) — Rust `DEFAULT_HOT_C`.
constexpr int64_t kDefaultHotC = 80;
/// Default recovery threshold, °C — Rust `DEFAULT_COOL_C`.
constexpr int64_t kDefaultCoolC = 70;

/// Rust `static EXTERNAL_LEVEL: AtomicU8`.
std::atomic<uint8_t> g_external_level{0};

/// Rust `v.trim().parse::<i64>().ok()`. `std::from_chars` accepts a leading '-' but rejects '+',
/// and trims nothing — both gaps are closed here. Rust trims Unicode whitespace; the ASCII set is
/// the reachable one for a sysfs file and an env var.
std::optional<int64_t> parse_i64_trimmed(std::string_view v) {
    constexpr std::string_view kWs = " \t\n\r\f\v";
    const auto b = v.find_first_not_of(kWs);
    if (b == std::string_view::npos) return std::nullopt;
    v = v.substr(b, v.find_last_not_of(kWs) - b + 1);
    if (v.front() == '+') v.remove_prefix(1);
    if (v.empty()) return std::nullopt;
    int64_t out = 0;
    const auto r = std::from_chars(v.data(), v.data() + v.size(), out);
    if (r.ec != std::errc{} || r.ptr != v.data() + v.size()) return std::nullopt;
    return out;
}

/// Rust's `parse_c` closure in `governor()`.
int64_t env_c(const char* name, int64_t dflt) {
    const char* v = std::getenv(name);
    if (v == nullptr) return dflt;
    return parse_i64_trimmed(v).value_or(dflt);
}

/// `SAPIENT_THERMAL=off` disables BOTH the sysfs governor and the external level — one escape
/// hatch for the whole mechanism. Rust `OnceLock` twin: read once per process.
bool thermal_disabled() {
    static const bool off = [] {
        const char* v = std::getenv("SAPIENT_THERMAL");
        if (v == nullptr) return false;
        const std::string_view s(v);
        if (s.size() != 3) return false; // eq_ignore_ascii_case("off")
        const auto lower = [](char c) {
            return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
        };
        return lower(s[0]) == 'o' && lower(s[1]) == 'f' && lower(s[2]) == 'f';
    }();
    return off;
}

/// Rust `governor() -> Option<&'static ThermalGovernor>`: OnceLock, None when disabled or when the
/// machine exposes no zones. `std::optional` + `emplace` because ThermalGovernor holds atomics and
/// is therefore neither copyable nor movable; the storage is a function-local static, so its
/// lifetime is the process's and handing out a pointer is safe.
const ThermalGovernor* governor() {
    static std::optional<ThermalGovernor> gov;
    static const bool initialised = [] {
        if (thermal_disabled()) return true;
        const char* p = std::getenv("SAPIENT_THERMAL_PATH");
        gov.emplace(p != nullptr ? fs::path(p) : fs::path("/sys/class/thermal"),
                    env_c("SAPIENT_THERMAL_HOT", kDefaultHotC),
                    env_c("SAPIENT_THERMAL_COOL", kDefaultCoolC),
                    std::max<size_t>(parallel::num_threads(), 1));
        if (!gov->is_active()) gov.reset(); // Rust: `gov.is_active().then_some(gov)`
        return true;
    }();
    (void)initialised;
    return gov.has_value() ? &*gov : nullptr;
}

} // namespace

ThermalGovernor::ThermalGovernor(const fs::path& root,
                                 int64_t hot_c,
                                 int64_t cool_c,
                                 size_t max_threads)
    : hot_mc_(hot_c * 1000),
      cool_mc_(cool_c * 1000),
      max_threads_(std::max<size_t>(max_threads, 1)),
      min_threads_(std::max<size_t>(std::max<size_t>(max_threads, 1) / 2, 1)),
      effective_(std::max<size_t>(max_threads, 1)),
      warned_(false) {
    // Rust `if let Ok(entries) = std::fs::read_dir(root)`: a missing root simply yields no zones.
    // The error_code overloads are MANDATORY — the throwing ones would terminate the process on
    // macOS/Windows, where /sys/class/thermal does not exist (see the plan's Global Constraints).
    std::error_code ec;
    fs::directory_iterator it(root, ec);
    const fs::directory_iterator end;
    for (; !ec && it != end; it.increment(ec)) {
        const std::string name = it->path().filename().string();
        if (name.rfind("thermal_zone", 0) != 0) continue;
        fs::path temp = it->path() / "temp";
        std::error_code fec;
        if (fs::is_regular_file(temp, fec)) zones_.push_back(std::move(temp));
    }
    std::sort(zones_.begin(), zones_.end()); // Rust `zones.sort()` (PathBuf Ord is lexicographic)
}

bool ThermalGovernor::is_active() const {
    return !zones_.empty();
}

std::optional<int64_t> ThermalGovernor::max_temp_mc() const {
    // Rust: read_to_string → trim → parse::<i64> → .max(); unreadable/unparsable zones are skipped.
    std::optional<int64_t> best;
    for (const auto& p : zones_) {
        std::ifstream f(p, std::ios::binary);
        if (!f.is_open()) continue;
        const std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        const auto v = parse_i64_trimmed(s);
        if (!v.has_value()) continue;
        if (!best.has_value() || *v > *best) best = v;
    }
    return best;
}

size_t ThermalGovernor::effective() const {
    return effective_.load(std::memory_order_relaxed);
}

size_t ThermalGovernor::sample() const {
    const auto t = max_temp_mc();
    if (!t.has_value()) return effective();
    const size_t cur = effective();
    if (*t >= hot_mc_ && cur > min_threads_) {
        const size_t next = cur - 1;
        effective_.store(next, std::memory_order_relaxed);
        if (!warned_.exchange(true, std::memory_order_relaxed)) { // Rust's one-shot `warned.swap`
            std::fprintf(stderr,
                         "thermal: %.1f °C ≥ %lld °C — backing decode off to %zu/%zu threads to "
                         "sustain clocks (set SAPIENT_THERMAL=off to disable)\n",
                         static_cast<double>(*t) / 1000.0,
                         static_cast<long long>(hot_mc_ / 1000), next, max_threads_);
        }
        return next;
    }
    if (*t <= cool_mc_ && cur < max_threads_) {
        const size_t next = cur + 1;
        effective_.store(next, std::memory_order_relaxed);
        return next;
    }
    return cur; // hysteresis band
}

namespace detail {
size_t external_cap(uint8_t level, size_t max) {
    switch (level) {
        case 0:
            return max;
        case 1:
            return std::max<size_t>(max * size_t{3} / 4, 1);
        case 2:
            return std::max<size_t>(max / 2, 1);
        default:
            return std::max<size_t>(max / 4, 1);
    }
}
} // namespace detail

void set_external_thermal_level(uint8_t level) {
    if (thermal_disabled()) return;
    const uint8_t lv = std::min<uint8_t>(level, 3);
    const uint8_t prev = g_external_level.exchange(lv, std::memory_order_relaxed);
    if (prev != lv) {
        std::fprintf(stderr,
                     "thermal: external level %u → %u; effective decode threads now %zu\n",
                     static_cast<unsigned>(prev), static_cast<unsigned>(lv), effective_threads());
    }
}

uint8_t external_thermal_level() {
    return g_external_level.load(std::memory_order_relaxed);
}

size_t effective_threads() {
    const size_t max = std::max<size_t>(parallel::num_threads(), 1);
    const ThermalGovernor* g = governor();
    const size_t base = (g != nullptr) ? g->effective() : max;
    return std::min(base,
                    detail::external_cap(g_external_level.load(std::memory_order_relaxed), max));
}

void tick() {
    const ThermalGovernor* g = governor();
    if (g == nullptr) return;
    static const std::chrono::steady_clock::time_point epoch = std::chrono::steady_clock::now();
    static std::atomic<uint64_t> last_ms{0};
    const auto now_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                                                              epoch)
            .count());
    uint64_t last = last_ms.load(std::memory_order_relaxed);
    if (now_ms <= last || now_ms - last < kTickMs) return; // Rust `now_ms.saturating_sub(last)`
    // Only the thread that wins the exchange samples; the others fall through (Rust's
    // compare_exchange guard). `compare_exchange_strong` writes `last` on failure — harmless, we
    // return either way.
    if (last_ms.compare_exchange_strong(last, now_ms, std::memory_order_relaxed,
                                        std::memory_order_relaxed)) {
        g->sample();
    }
}

} // namespace sapient::backends_cpu::thermal
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd cpp && cmake --build --preset dev && ctest --preset dev`
Expected: all previously-passing tests still pass, plus the 6 new `Thermal.*` tests — **248 tests, 100%**. `effective_threads()` on this Mac still returns `parallel::num_threads()` (no `/sys/class/thermal`, external level 0), so every existing `gemv_chunk` result is unchanged.

- [ ] **Step 6: Verify the inert path really is inert (the `error_code` rule)**

Run: `cd cpp && ./build/dev/libs/sapient-backends-cpu/sapient_backends_cpu_tests --gtest_filter=Matmul.*:Thermal.* 2>&1 | tail -5`
Expected: PASSED, no `terminate called after throwing an instance of 'std::filesystem::filesystem_error'`. (Adjust the binary path if the preset puts it elsewhere; `ctest --preset dev -V -R Thermal` is an equivalent check.)

- [ ] **Step 7: Cross-compile the TU for x86_64**

Run, from `cpp/`:
```bash
clang++ -std=c++20 --target=x86_64-apple-macos -Wall -Wextra -Wpedantic -Wshadow -Werror \
  -ffp-contract=off -fno-math-errno \
  -Ilibs/sapient-core/include -Ilibs/sapient-backends-cpu/include \
  -c libs/sapient-backends-cpu/src/thermal.cpp -o /dev/null
```
Expected: no output, exit 0.

- [ ] **Step 8: Confirm the Rust oracle is untouched**

Run: `git status --short crates/ && cargo test -p sapient-backends-cpu 2>&1 | tail -3`
Expected: no output from `git status`; `test result: ok. 71 passed; 0 failed; 1 ignored`.

- [ ] **Step 9: Format and commit**

```bash
git add cpp/libs/sapient-backends-cpu/include/sapient/backends_cpu/thermal.hpp \
        cpp/libs/sapient-backends-cpu/src/thermal.cpp \
        cpp/libs/sapient-backends-cpu/tests/thermal_test.cpp \
        cpp/libs/sapient-backends-cpu/CMakeLists.txt
.superpowers/tools-venv/bin/clang-format -i \
        cpp/libs/sapient-backends-cpu/include/sapient/backends_cpu/thermal.hpp \
        cpp/libs/sapient-backends-cpu/src/thermal.cpp \
        cpp/libs/sapient-backends-cpu/tests/thermal_test.cpp
git add -u
git commit -m "$(cat <<'EOF'
cpp(backends-cpu): port thermal.rs — hysteresis governor, external level cap, tick()

Replaces plan C's inert stubs with the full port of thermal.rs: ThermalGovernor
(sorted thermal_zone*/temp scan, 80/70 °C hysteresis, floor at half the cores,
one-shot warning), the 4-level external cap, effective_threads() as the stricter
of the two sources, and the 500 ms rate-limited tick(). All 6 Rust tests ported
by name; the four governor tests build a fake sysfs root and never touch the
process-global singleton.

Every std::filesystem call takes the std::error_code overload: the throwing
overloads would escape governor()'s static initialiser and terminate the process
on the first matmul on macOS/Windows, where /sys/class/thermal does not exist.

EOF
)"
```

Append the trailer with a blank line before it, exactly:
`Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`

---

### Task 2: `spinpool.hpp/.cpp` — the seqlock op-handoff pool, `pool()`, `parallelism()`, `enabled()`

**Files:**
- Modify (replace the plan-C stub bodies wholesale): `cpp/libs/sapient-backends-cpu/include/sapient/backends_cpu/spinpool.hpp`, `cpp/libs/sapient-backends-cpu/src/spinpool.cpp`
- Create: `cpp/libs/sapient-backends-cpu/tests/spinpool_test.cpp`
- Modify: `cpp/libs/sapient-backends-cpu/CMakeLists.txt` (add `tests/spinpool_test.cpp`; drop the now-stale "spinpool is a plan-E stub" line from the header comment)

**Interfaces:**
- Consumes: `sapient::backends_cpu::parallel::num_threads()`, `sapient::backends_cpu::thermal::effective_threads()` (Task 1), `sapient::backends_cpu::env_usize` (`env.hpp`), `sapient::core::panic`.
- Produces (namespace `sapient::backends_cpu::spinpool`): `inline constexpr uint64_t kDefaultSpinIters = 4000`; `class SpinPool` with `static SpinPool& create(size_t workers, uint64_t spin_iters)`, `template <class F> void run(size_t n_chunks, const F& f)`, `size_t workers() const`; free functions `SpinPool& pool()`, `size_t parallelism()`, `bool enabled()`. Task 3's `matmul::detail::for_each_out_chunk` consumes `enabled()` and `pool().run(...)`; `matmul::detail::gemv_chunk` already consumes `enabled()`/`parallelism()` (plan C wired them against the stubs — no call-site change is needed here).

Rust reference: `crates/sapient-backends/cpu/src/spinpool.rs` (all 541 lines). Porting map §6a and §2.10. **Read the module doc-comment at `spinpool.rs:1-44` before writing anything** — it is the safety argument for the ordering this task must reproduce.

**Ruling — `run` is a template over an erased core, not `std::function`.** Rust's `run<F: Fn(usize)+Sync>(&self, n_chunks, f: &F)` builds an `OpSlot { call: thunk::<F>, ctx: f as *const F as *const () }`. The C++ twin is a header-inline `template <class F> void run(size_t, const F&)` forwarding to `run_erased(n, &thunk<F>, &f)`. A `std::function` parameter would heap-allocate the `for_each_out_chunk` lambda (four captured words — past libc++'s small-object buffer) on **every GEMV**, which is precisely the per-dispatch tax the pool exists to remove. Costs if wrong: a later performance plan would have to undo it.

**Ruling — thread names on Windows are omitted.** macOS (`pthread_setname_np(name)`, self-only) and Linux (`pthread_setname_np(self, name)`, 15-char cap) get Rust's `sapient-spin-{w}` names. Windows is compile-only in CI, `SetThreadDescription`'s availability depends on the SDK's `NTDDI_VERSION`, and a thread name is a debugging aid with no behavioural or parity role — so the `#else` arm is an explicitly-commented no-op rather than a construct nobody here can compile-test. Costs if wrong: Windows debuggers show unnamed threads.

- [ ] **Step 1: Write the failing tests** — create `cpp/libs/sapient-backends-cpu/tests/spinpool_test.cpp`:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
// Port of the `tests`, `perf_probe` and `stress` modules of
// crates/sapient-backends/cpu/src/spinpool.rs — all 5 Rust tests by name plus the one #[ignore]
// probe (gtest's DISABLED_ prefix). No gtest assertion runs on a worker thread or inside a chunk
// closure: failures are accumulated into atomics and asserted on the main thread.
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

#include "sapient/backends_cpu/spinpool.hpp"

namespace spinpool = sapient::backends_cpu::spinpool;

TEST(Spinpool, runs_every_chunk_exactly_once) {
    spinpool::SpinPool& pool = spinpool::pool();
    for (size_t round = 0; round < 200; ++round) {
        const size_t n = 1 + (round % 61);
        std::vector<std::atomic<uint32_t>> hits(n);
        for (auto& h : hits) h.store(0, std::memory_order_seq_cst);
        pool.run(n, [&hits](size_t c) { hits[c].fetch_add(1, std::memory_order_seq_cst); });
        for (size_t c = 0; c < n; ++c) {
            ASSERT_EQ(hits[c].load(std::memory_order_seq_cst), 1u)
                << "round " << round << " chunk " << c;
        }
    }
}

TEST(Spinpool, concurrent_publishers_serialize) {
    spinpool::SpinPool& pool = spinpool::pool();
    std::atomic<uint64_t> failures{0};
    std::vector<std::thread> threads;
    threads.reserve(4);
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&pool, &failures] {
            for (int i = 0; i < 100; ++i) {
                std::vector<std::atomic<uint32_t>> hits(37);
                for (auto& h : hits) h.store(0, std::memory_order_seq_cst);
                pool.run(37, [&hits](size_t c) { hits[c].fetch_add(1, std::memory_order_seq_cst); });
                for (auto& h : hits) {
                    if (h.load(std::memory_order_seq_cst) != 1)
                        failures.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(failures.load(std::memory_order_relaxed), 0u)
        << "a chunk ran zero or multiple times while publishers overlapped";
}

TEST(Spinpool, survives_park_and_wake) {
    spinpool::SpinPool& pool = spinpool::pool();
    std::vector<std::atomic<uint32_t>> hits(16);
    for (auto& h : hits) h.store(0, std::memory_order_seq_cst);
    pool.run(16, [&hits](size_t c) { hits[c].fetch_add(1, std::memory_order_seq_cst); });
    // Sleep well past any spin budget so the workers park, then dispatch again.
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    std::vector<std::atomic<uint32_t>> hits2(16);
    for (auto& h : hits2) h.store(0, std::memory_order_seq_cst);
    pool.run(16, [&hits2](size_t c) { hits2[c].fetch_add(1, std::memory_order_seq_cst); });
    for (size_t c = 0; c < 16; ++c) {
        EXPECT_EQ(hits[c].load(std::memory_order_seq_cst), 1u) << "pre-park chunk " << c;
        EXPECT_EQ(hits2[c].load(std::memory_order_seq_cst), 1u) << "post-wake chunk " << c;
    }
}

// Ground-truth probe: parallel speedup of the pool itself, outside the engine. Rust marks it
// #[ignore]; gtest's twin is the DISABLED_ prefix. Run:
//   SAPIENT_SPINPOOL_WORKERS=9 ./sapient_backends_cpu_tests \
//     --gtest_also_run_disabled_tests --gtest_filter=Spinpool.DISABLED_pool_speedup_probe
TEST(Spinpool, DISABLED_pool_speedup_probe) {
    spinpool::SpinPool& pool = spinpool::pool();
    constexpr size_t kNChunks = 40;
    constexpr uint64_t kWorkPerChunk = 400000;
    std::vector<std::atomic<uint64_t>> sink(kNChunks);
    for (auto& s : sink) s.store(0, std::memory_order_relaxed);
    const auto busy = [&sink](size_t c) {
        uint64_t x = static_cast<uint64_t>(c) ^ 0x9e3779b97f4a7c15ULL;
        for (uint64_t i = 0; i < kWorkPerChunk; ++i) x = x * 6364136223846793005ULL + i;
        sink[c].store(x, std::memory_order_relaxed);
    };
    pool.run(kNChunks, busy); // warm both paths once
    for (size_t c = 0; c < kNChunks; ++c) busy(c);

    auto t = std::chrono::steady_clock::now();
    for (int r = 0; r < 50; ++r)
        for (size_t c = 0; c < kNChunks; ++c) busy(c);
    const auto serial = std::chrono::steady_clock::now() - t;

    t = std::chrono::steady_clock::now();
    for (int r = 0; r < 50; ++r) pool.run(kNChunks, busy);
    const auto par = std::chrono::steady_clock::now() - t;

    const double serial_s = std::chrono::duration<double>(serial).count();
    const double par_s = std::chrono::duration<double>(par).count();
    std::printf("workers=%zu serial=%.3fs pool=%.3fs speedup=%.2fx\n", pool.workers(), serial_s,
                par_s, serial_s / par_s);

    t = std::chrono::steady_clock::now();
    for (int r = 0; r < 10000; ++r)
        pool.run(kNChunks, [&sink](size_t c) {
            sink[c].store(static_cast<uint64_t>(c), std::memory_order_relaxed);
        });
    const double tiny_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
    std::printf("10k near-empty ops: %.3fs (%.1f µs/op)\n", tiny_s, tiny_s * 1e6 / 10000.0);
    SUCCEED();
}

// Reproducer class for the op-boundary ABA: a tiny spin budget makes workers park/wake around
// every op, and back-to-back ops with heap-owned closure state make a stale or torn slot read
// fatal (the SIGSEGV this test exists to prevent regressing). Builds its OWN pool, like Rust.
TEST(Spinpool, rapid_ops_with_constant_parking) {
    spinpool::SpinPool& pool = spinpool::SpinPool::create(4, 50); // parks after ~50 spins
    for (size_t round = 0; round < 5000; ++round) {
        const size_t n = 2 + (round % 13);
        std::vector<uint64_t> payload(n);
        for (size_t v = 0; v < n; ++v) payload[v] = static_cast<uint64_t>(v + round);
        std::vector<std::atomic<uint64_t>> acc(n);
        for (auto& a : acc) a.store(0, std::memory_order_seq_cst);
        pool.run(n, [&acc, &payload](size_t c) {
            acc[c].store(payload[c] * 2, std::memory_order_seq_cst);
        });
        for (size_t c = 0; c < n; ++c) {
            ASSERT_EQ(acc[c].load(std::memory_order_seq_cst), payload[c] * 2)
                << "round " << round << " chunk " << c;
        }
    }
}
```

Add `tests/spinpool_test.cpp` to the `add_executable(sapient_backends_cpu_tests …)` source list.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd cpp && cmake --build --preset dev 2>&1 | tail -20`
Expected: a COMPILE failure — `no type named 'SpinPool' in namespace 'sapient::backends_cpu::spinpool'`.

- [ ] **Step 3: Write the header**

Replace `cpp/libs/sapient-backends-cpu/include/sapient/backends_cpu/spinpool.hpp` entirely:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-backends/cpu/src/spinpool.rs (plan E). Persistent spin-wait worker pool
// for the decode hot path (llama.cpp-style).
//
// One decoded token runs ~200+ GEMV parallel regions — one per matmul call — and every fork/join
// region pays worker wake + park latency (µs-scale futex round-trips), ~230 barriers per token.
// Here the workers stay HOT for the duration of a generation: between ops they spin (bounded
// iterations, then park on a condvar), so dispatching an op during decode costs a few atomic
// operations instead of thread wakeups.
//
// ## Contract
// - `run(n_chunks, f)` executes `f(0..n_chunks)` exactly once each, in parallel (the caller
//   participates), returning only after ALL chunks complete. Chunks must touch disjoint data —
//   the same contract as `parallel::par_chunks_mut`.
// - Concurrent publishers serialize on a lock: two overlapping matmuls degrade to two
//   back-to-back fully-parallel ops (never a deadlock, and each still uses the whole pool).
//   `f` must therefore NOT call `run` again — the publish mutex is not recursive, exactly as in
//   Rust. No kernel nests parallel regions.
// - `enabled()` is false while the thermal governor is shedding cores (spinning workers would
//   defeat the backoff) and when `SAPIENT_SPINPOOL=0`. Callers fall back to `parallel`.
//
// ## Op-handoff protocol (seqlock-style)
// The generation counter is EVEN when an op is published and ODD while the slot is being
// rewritten. A publisher (holding `publish_`):
//   1. bumps `generation_` to ODD — closes the door: no worker can newly join,
//   2. waits for `active_ == 0` (workers still inside the previous op leave),
//   3. rewrites the op slot + chunk counters,
//   4. bumps `generation_` to EVEN and wakes parked workers.
// A worker joins by registering in `active_` FIRST and then making its authoritative
// `generation_` read (both SeqCst): if that read saw the old even generation it happened before
// the odd bump, so the publisher's drain in step 2 observes the registration and waits the worker
// out; if it happened after, the worker sees ODD and backs out. Either way a worker can never
// copy the slot while it is being rewritten. (The original order — drain BEFORE the odd bump —
// left exactly that window, and it segfaulted in practice under rapid park/wake cycling.
// `Spinpool.rapid_ops_with_constant_parking` pins the fixed behaviour.)
//
// The plain (non-atomic) `op_` member is not a data race: the publisher's writes are ordered
// before the SeqCst store that makes `generation_` even, and a worker only reads the slot after
// an acquiring/SeqCst load that observed that value — a textbook release/acquire pair.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace sapient::backends_cpu::spinpool {

/// Spin iterations before a worker parks on the condvar. MEASURED optimum on M4 (llama-1B Q4_K_M
/// decode sweep: 0 → 54.2 tok/s, 4k → 63.6, 16k → 59.1, 50k → 52.3, 200k hot-spin → 38.6): ~4k
/// iterations (~10–20 µs) catches the back-to-back GEMVs inside a layer, while parking through the
/// longer serial phases (attention, sampling). Long spins actively HURT — workers burning cores
/// during serial phases steal the package power budget / scheduler slots from the critical-path
/// thread. Overridable via `SAPIENT_SPINPOOL_SPINS`.
inline constexpr uint64_t kDefaultSpinIters = 4000;

class SpinPool {
public:
    /// Rust `SpinPool::new` — allocates a pool, spawns `workers` detached threads and returns a
    /// reference that lives for the rest of the process (Rust's `Box::leak`). Deliberately leaked:
    /// the detached workers must never outlive the mutexes and condition variables they wait on,
    /// which a static destructor running at exit would destroy underneath them. Reached by `pool()`
    /// and, as in Rust, directly by the `rapid_ops_with_constant_parking` stress test.
    static SpinPool& create(size_t workers, uint64_t spin_iters);

    SpinPool(const SpinPool&) = delete;
    SpinPool& operator=(const SpinPool&) = delete;

    /// Execute `f(0..n_chunks)` in parallel across the pool + this thread. Returns after every
    /// chunk has run. Chunks must write disjoint data. `f` must not throw (an escaping exception
    /// aborts via `sapient::core::panic`, mirroring a Rust panic under `panic = "abort"`), and must
    /// not itself call `run`.
    template <class F>
    void run(size_t n_chunks, const F& f) {
        run_erased(n_chunks, &thunk<F>, static_cast<const void*>(&f));
    }

    /// Worker threads (excludes the participating publisher).
    size_t workers() const { return workers_; }

private:
    using CallFn = void (*)(const void*, size_t);

    /// Rust's `thunk::<F>` trampoline: `ctx` is the `&F` handed to `run`, alive until `run` returns
    /// (the publisher blocks until every block completes).
    template <class F>
    static void thunk(const void* ctx, size_t c) {
        (*static_cast<const F*>(ctx))(c);
    }

    struct OpSlot {
        CallFn call;
        const void* ctx;
        size_t n_chunks;
        /// Chunks per claimed block (guided scheduling granularity).
        size_t block;
    };

    /// Pad each hot atomic to its own cache line (128 B covers Apple Silicon's line pairs). Without
    /// this, `generation_` — which every idle worker spins on — shares a line with
    /// `completed_`/`next_block_`, so every completion invalidates the spinners' line and every
    /// spin-load contends the completer's store: measured ~2× decode REGRESSION on M4 before
    /// padding. This is why llama.cpp's threadpool pads its counters.
    template <class T>
    struct alignas(128) Pad {
        T v;
    };

    SpinPool(size_t workers, uint64_t spin_iters);

    void run_erased(size_t n_chunks, CallFn call, const void* ctx);
    void execute_blocks(const OpSlot& op);
    void worker_loop();

    /// Seqlock generation: even = published, odd = slot being rewritten.
    Pad<std::atomic<uint64_t>> generation_{};
    OpSlot op_;
    /// GUIDED claiming: participants grab contiguous BLOCKS of chunks off this counter (~3 blocks
    /// per participant off macOS). v1 per-chunk claiming load-balanced the M4's P/E cores (+5% vs
    /// the fork/join path) but its ~112 RMWs/op on two hot lines was a 2× regression on 14-core
    /// Thor; v2 static shares fixed Thor but made every op wait for the slowest E-core on M4.
    /// Guided blocks keep v2's contiguity at ~4× less claim traffic than v1 while letting fast
    /// cores take more blocks.
    Pad<std::atomic<size_t>> next_block_{};
    /// Completed BLOCKS (one increment per block, not per chunk).
    Pad<std::atomic<size_t>> completed_{};
    /// Workers currently inside an op (validated slot copy → last share).
    Pad<std::atomic<size_t>> active_{};
    std::mutex publish_;
    std::mutex sleep_;
    std::condition_variable wake_;
    Pad<std::atomic<size_t>> parked_{};
    size_t workers_;
    uint64_t spin_iters_;
};

/// The process-global pool: `parallel::num_threads() - 1` workers (the publishing thread
/// participates), matching the task budget `gemv_chunk` computes from the same figure. Lazily
/// created on first use. Env: `SAPIENT_SPINPOOL_WORKERS`, `SAPIENT_SPINPOOL_SPINS`.
SpinPool& pool();

/// Worker threads + the participating publisher — the parallelism the task count should be sized
/// for when the pool is active (`gemv_chunk` uses it). Instantiates the pool.
size_t parallelism();

/// Whether the spin pool should be used for this dispatch. Off when `SAPIENT_SPINPOOL=0` (A/B
/// lever / escape hatch) and while the thermal governor or an external level is shedding cores —
/// parked workers shed heat, spinning workers do not, so governed decode must stay on the
/// `parallel` path. Does NOT instantiate the pool.
bool enabled();

} // namespace sapient::backends_cpu::spinpool
```

- [ ] **Step 4: Write the implementation**

Replace `cpp/libs/sapient-backends-cpu/src/spinpool.cpp` entirely:

```cpp
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/backends_cpu/spinpool.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <optional>
#include <string_view>
#include <thread>

#include "sapient/backends_cpu/env.hpp"
#include "sapient/backends_cpu/parallel.hpp"
#include "sapient/backends_cpu/thermal.hpp"
#include "sapient/core/panic.hpp"

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

// Rust's `cfg!(target_os = "macos")` — macOS proper, not iOS (which sub-project 7 brings in and
// which has neither this QoS story nor the measurement behind the ON default).
//
// The `#ifndef` is a COMPILE-CHECK HOOK, not a configuration knob. `--target=x86_64-apple-macos`
// still reports TARGET_OS_OSX == 1, so no cross-compile available on this host reaches the
// non-macOS arms of `enabled()` and `set_worker_thread_name()`; a probe build passing
// `-DSAPIENT_TARGET_MACOS=0` does. Never set it from CMake or anywhere else.
#ifndef SAPIENT_TARGET_MACOS
#if defined(__APPLE__) && defined(TARGET_OS_OSX) && TARGET_OS_OSX
#define SAPIENT_TARGET_MACOS 1
#else
#define SAPIENT_TARGET_MACOS 0
#endif
#endif

#if SAPIENT_TARGET_MACOS
#include <pthread.h>
#include <pthread/qos.h>
#elif defined(__linux__)
#include <pthread.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace sapient::backends_cpu::spinpool {
namespace {

/// Rust `std::hint::spin_loop()`. Not parity-bound for results, but load-bearing for the parking
/// dynamics the 4 000-iteration budget was measured against — never substitute `yield()`, which is
/// a syscall.
inline void spin_hint() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#elif defined(__aarch64__) || defined(_M_ARM64)
    __builtin_arm_isb(0xF); // what rustc lowers core::hint::spin_loop to on aarch64 (`isb sy`)
#endif
}

/// macOS demotes CPU-burning threads (priority decay) and prefers E-cores for them; a demoted
/// worker holding a claimed block stalls the whole op barrier — measured as the pool scaling
/// BACKWARDS with thread count (10 threads slower than 4). Pin the QoS class so the scheduler
/// treats spin-waiting threads as latency-sensitive, the same thing ggml's threadpool does.
inline void pin_qos_user_interactive() {
#if SAPIENT_TARGET_MACOS
    ::pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

/// Rust names each worker `sapient-spin-{w}` at spawn. macOS's pthread_setname_np is self-only;
/// Linux's takes a handle and caps the name at 15 characters + NUL (so indices past 99 truncate).
/// Windows is left unnamed on purpose — see the plan's ruling; a thread name is a debugging aid
/// with no behavioural role, and SetThreadDescription's availability is SDK-version dependent in a
/// CI job that is compile-only.
void set_worker_thread_name([[maybe_unused]] size_t w) {
#if SAPIENT_TARGET_MACOS
    char name[32];
    std::snprintf(name, sizeof(name), "sapient-spin-%zu", w);
    ::pthread_setname_np(name);
#elif defined(__linux__)
    char name[16];
    std::snprintf(name, sizeof(name), "sapient-spin-%zu", w);
    ::pthread_setname_np(::pthread_self(), name);
#endif
}

/// `SAPIENT_SPINPOOL_BLOCK`: fixed chunks-per-claimed-block override for the guided scheduler
/// (topology experiments). Rust `OnceLock`, filtered `>= 1`.
std::optional<size_t> block_size_override() {
    static const std::optional<size_t> v = [] {
        const auto x = env_usize("SAPIENT_SPINPOOL_BLOCK");
        return (x.has_value() && *x >= 1) ? x : std::optional<size_t>{};
    }();
    return v;
}

} // namespace

SpinPool::SpinPool(size_t workers, uint64_t spin_iters)
    : op_{[](const void*, size_t) {}, nullptr, 0, 1}, // Rust `impl Default for OpSlot`
      workers_(workers),
      spin_iters_(spin_iters) {}

SpinPool& SpinPool::create(size_t workers, uint64_t spin_iters) {
    // Leaked on purpose (Rust `Box::leak`): the detached workers below outlive every static
    // destructor, and destroying `publish_`/`sleep_`/`wake_` underneath a waiting worker at exit
    // is undefined behaviour. One allocation per process (plus one per stress-test run).
    auto* p = new SpinPool(workers, spin_iters); // NOLINT(cppcoreguidelines-owning-memory)
    for (size_t w = 0; w < workers; ++w) {
        try {
            std::thread t([p, w] {
                set_worker_thread_name(w);
                p->worker_loop();
            });
            t.detach();
        } catch (...) {
            sapient::core::panic("spawn spinpool worker"); // Rust `.expect("spawn spinpool worker")`
        }
    }
    return *p;
}

void SpinPool::execute_blocks(const OpSlot& op) {
    // Claim contiguous blocks of `op.block` chunks off the shared counter until none remain.
    // Dynamic (fast cores take more blocks — the M4 P/E balance) yet contiguous within each block
    // (the Thor prefetch locality). One completion increment per BLOCK. A parked worker simply
    // claims nothing — no deadlock.
    for (;;) {
        const size_t b = next_block_.v.fetch_add(1, std::memory_order_relaxed);
        const size_t lo = b * op.block;
        if (lo >= op.n_chunks) break;
        const size_t hi = std::min(lo + op.block, op.n_chunks);
        for (size_t c = lo; c < hi; ++c) op.call(op.ctx, c);
        completed_.v.fetch_add(1, std::memory_order_release);
    }
}

void SpinPool::worker_loop() {
    pin_qos_user_interactive();
    uint64_t seen = 0; // last even generation this worker ran
    for (;;) {
        // ── Wait for a new published (even) generation ───────────────────────────────────────
        // Rust shadows `g` here; -Wshadow forbids that, so the inner reads are `cur` and `woke`.
        uint64_t spins = 0;
        uint64_t g = 0;
        for (;;) {
            const uint64_t cur = generation_.v.load(std::memory_order_acquire);
            if (cur % 2 == 0 && cur != seen) {
                g = cur;
                break;
            }
            ++spins;
            if (spins > spin_iters_) {
                std::unique_lock<std::mutex> guard(sleep_);
                parked_.v.fetch_add(1, std::memory_order_seq_cst);
                for (;;) {
                    const uint64_t woke = generation_.v.load(std::memory_order_acquire);
                    if (woke % 2 == 0 && woke != seen) break;
                    wake_.wait(guard);
                }
                parked_.v.fetch_sub(1, std::memory_order_seq_cst);
                g = generation_.v.load(std::memory_order_acquire);
                break;
            }
            spin_hint();
        }
        if (g % 2 != 0 || g == seen) continue; // woke on an odd/stale gen — re-enter the wait loop

        // ── Enter the op: register FIRST, then the authoritative read ────────────────────────
        // The registration must be visible to the publisher's drain before we commit to reading
        // the slot; SeqCst on both sides gives the total order the safety argument needs.
        active_.v.fetch_add(1, std::memory_order_seq_cst);
        if (generation_.v.load(std::memory_order_seq_cst) != g) {
            active_.v.fetch_sub(1, std::memory_order_seq_cst); // door closed (odd) or a newer op
            continue;
        }
        // Gen is even and unchanged since we registered in `active_`; the next publisher waits for
        // active_ == 0 before touching the slot, so this copy is of a stable, published op.
        const OpSlot op = op_;
        execute_blocks(op);
        active_.v.fetch_sub(1, std::memory_order_seq_cst);
        seen = g;
    }
}

void SpinPool::run_erased(size_t n_chunks, CallFn call, const void* ctx) {
    if (n_chunks == 0) return;
    if (n_chunks == 1 || workers_ == 0) {
        for (size_t c = 0; c < n_chunks; ++c) call(ctx, c);
        return;
    }

    // The publisher runs the token's SERIAL phases (norms, RoPE, sampling) between ops. If the
    // workers are QoS-pinned but the publisher is not, macOS runs the one thread doing
    // critical-path work at the LOWEST priority in the process — pin it too, once per thread.
#if SAPIENT_TARGET_MACOS
    {
        static thread_local bool qos_pinned = false;
        if (!qos_pinned) {
            pin_qos_user_interactive();
            qos_pinned = true;
        }
    }
#endif

    const std::lock_guard<std::mutex> publisher(publish_);
    // ORDER MATTERS — odd bump BEFORE the active drain. A worker joins by registering in `active_`
    // and THEN reading `generation_` (its authoritative check): if that read precedes this bump it
    // saw the old even gen and its registration is visible to the drain below (we wait it out); if
    // it follows the bump it sees odd and backs out. Draining FIRST left a window — sample
    // active_ == 0, a late worker registers, validates the still-unchanged gen, and copies the slot
    // WHILE we rewrite it (torn copy → dangling ctx → the measured SIGSEGV).
    generation_.v.fetch_add(1, std::memory_order_seq_cst); // → odd: door closed
    while (active_.v.load(std::memory_order_seq_cst) != 0) spin_hint();

    // Block size is TOPOLOGY-dependent (both directions measured): heterogeneous P/E cores
    // (M-series) need block = 1 — an E-core claiming a late multi-chunk block adds a straggler tail
    // to every op (llama-1B lm_head block ≈ 8 measured 62 → 44 tok/s on M4); homogeneous server ARM
    // wants ~3 blocks/participant — per-chunk claim+completion RMW traffic was Thor's 2×
    // regression, and guided blocks took it to +8% OVER the fork/join path.
    const size_t participants = workers_ + 1;
#if SAPIENT_TARGET_MACOS
    const size_t default_block = 1;
#else
    const size_t default_block = std::max<size_t>(n_chunks / (size_t{3} * participants), 1);
#endif
    const size_t block = block_size_override().value_or(default_block);
    const size_t n_blocks = (n_chunks + block - 1) / block; // div_ceil
    const OpSlot op{call, ctx, n_chunks, block};
    op_ = op;
    next_block_.v.store(0, std::memory_order_relaxed);
    completed_.v.store(0, std::memory_order_relaxed);
    generation_.v.fetch_add(1, std::memory_order_seq_cst); // → even: published
    if (parked_.v.load(std::memory_order_seq_cst) > 0) {
        const std::lock_guard<std::mutex> sleeper(sleep_);
        wake_.notify_all();
    }

    // Participate from the calling thread.
    execute_blocks(op);
    // Block writes become visible via the Release increments in execute_blocks.
    while (completed_.v.load(std::memory_order_acquire) < n_blocks) spin_hint();
}

SpinPool& pool() {
    // Rust `OnceLock` twin; the reference is bound once and the pool lives for the process.
    static SpinPool& instance = [] () -> SpinPool& {
        const auto spins_env = env_usize("SAPIENT_SPINPOOL_SPINS");
        const uint64_t spins =
            spins_env.has_value() ? static_cast<uint64_t>(*spins_env) : kDefaultSpinIters;
        // SAPIENT_SPINPOOL_WORKERS decouples the pool size from the parallel pool's — the
        // diagnostic lever for isolating pool-internal cost from two-pool mixing (e.g.
        // RAYON_NUM_THREADS=1 + WORKERS=9 runs all GEMV parallelism on the spin pool alone).
        const auto workers_env = env_usize("SAPIENT_SPINPOOL_WORKERS");
        const size_t workers = workers_env.has_value()
                                   ? *workers_env
                                   : std::max<size_t>(parallel::num_threads(), 1) - 1;
        return SpinPool::create(workers, spins);
    }();
    return instance;
}

size_t parallelism() {
    return pool().workers() + 1;
}

bool enabled() {
    static const bool on = [] {
        const char* v = std::getenv("SAPIENT_SPINPOOL");
        if (v != nullptr) return std::string_view(v) != "0"; // Rust `map(|v| v != "0")`
        // Measurement-driven default (2026-07-10, guided v4): the win scales with thread count —
        // more per-token fork/join tax to reclaim. M4 +7.7% (llama-1B) / +0.9% (qwen); Thor
        // 14-core +5.3%; Pi 5 4-core −4% (at ~100 ms/token there is no tax to reclaim). So: ON for
        // macOS and for Linux/aarch64 at ≥ 8 threads (Thor in, Pi out); everything else (x86,
        // Windows — unmeasured) stays opt-in. SAPIENT_SPINPOOL=1/0 overrides.
#if SAPIENT_TARGET_MACOS
        return true;
#elif defined(__linux__) && (defined(__aarch64__) || defined(_M_ARM64))
        return parallel::num_threads() >= 8;
#else
        return false;
#endif
    }();
    return on && thermal::effective_threads() >= parallel::num_threads();
}

} // namespace sapient::backends_cpu::spinpool
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd cpp && cmake --build --preset dev && ctest --preset dev`
Expected: **253 tests, 100%** (248 after Task 1, plus 5 `Spinpool.*`; the `DISABLED_` probe is not counted). Note what this does and does not prove: `enabled()` now returns Rust's default, but `for_each_out_chunk` still dispatches unconditionally through `parallel::par_chunks_mut` until Task 3, so a green suite here shows the pool is **correct and coexists safely**, not that any GEMV uses it. If any of the previously-green 242 now fails, the pool is wrong; do not adjust a kernel.

- [ ] **Step 6: Run the stress and concurrency tests repeatedly**

The op-boundary race is timing-dependent; one green run proves little.

Run: `cd cpp && for i in $(seq 1 20); do ./build/dev/libs/sapient-backends-cpu/sapient_backends_cpu_tests --gtest_filter='Spinpool.*' --gtest_brief=1 || { echo "FAILED on iteration $i"; break; }; done; echo done`
Expected: 20 clean iterations, no `FAILED on iteration`, no crash, no hang. (Adjust the binary path to wherever the `dev` preset writes it; `ctest --preset dev -R 'Spinpool' --repeat until-fail:20` is the equivalent.)

- [ ] **Step 7: Verify the whole golden suite under an explicitly-forced pool**

Run: `cd cpp && SAPIENT_SPINPOOL=1 SAPIENT_THERMAL=off ctest --preset dev -E 'Thermal\.external_level_caps_effective_threads' 2>&1 | tail -5` then `SAPIENT_SPINPOOL=0 ctest --preset dev 2>&1 | tail -5`
Expected: 100% both ways. The `-E` exclusion is required and is not a workaround: `SAPIENT_THERMAL=off` makes `set_external_thermal_level` a no-op (Rust does the same), so that one test asserts a cap that cannot be applied. Only the first command needs it — the second sets no thermal variable. (Task 3 makes this a permanent ctest gate rather than a manual step; running it here catches a broken pool before the wiring task builds on it.)

- [ ] **Step 8: Cross-compile the TU for x86_64**

Run, from `cpp/`:
```bash
clang++ -std=c++20 --target=x86_64-apple-macos -Wall -Wextra -Wpedantic -Wshadow -Werror \
  -ffp-contract=off -fno-math-errno \
  -Ilibs/sapient-core/include -Ilibs/sapient-backends-cpu/include \
  -c libs/sapient-backends-cpu/src/spinpool.cpp -o /dev/null
```
Expected: no output, exit 0. This compiles `spin_hint()`'s `_mm_pause` arm — but **not** `enabled()`'s `#else → false` arm: an `x86_64-apple-macos` target still reports `TARGET_OS_OSX == 1` (verified 2026-09-22), so `SAPIENT_TARGET_MACOS` stays 1 and the macOS arm is taken. Compile the non-macOS arms with the probe hook, natively and cross:

```bash
for tgt in "" "--target=x86_64-apple-macos"; do
  clang++ -std=c++20 $tgt -Wall -Wextra -Wpedantic -Wshadow -Werror \
    -ffp-contract=off -fno-math-errno -DSAPIENT_TARGET_MACOS=0 \
    -Ilibs/sapient-core/include -Ilibs/sapient-backends-cpu/include \
    -c libs/sapient-backends-cpu/src/spinpool.cpp -o /dev/null || echo "FAILED: $tgt"
done; echo done
```
Expected: no `FAILED` line. This is what compiles `enabled()`'s `#else → false` arm and `set_worker_thread_name`'s empty `#else`. The `__linux__` arms (the two-argument `pthread_setname_np`, the `num_threads() >= 8` default) cannot be compiled without a Linux sysroot and remain CI-only — say so in the task report rather than claiming coverage you do not have.

- [ ] **Step 9: Confirm the Rust oracle is untouched, then format and commit**

Run: `git status --short crates/` (expect no output).

```bash
git add cpp/libs/sapient-backends-cpu/include/sapient/backends_cpu/spinpool.hpp \
        cpp/libs/sapient-backends-cpu/src/spinpool.cpp \
        cpp/libs/sapient-backends-cpu/tests/spinpool_test.cpp \
        cpp/libs/sapient-backends-cpu/CMakeLists.txt
.superpowers/tools-venv/bin/clang-format -i \
        cpp/libs/sapient-backends-cpu/include/sapient/backends_cpu/spinpool.hpp \
        cpp/libs/sapient-backends-cpu/src/spinpool.cpp \
        cpp/libs/sapient-backends-cpu/tests/spinpool_test.cpp
git add -u
git commit -m "$(cat <<'EOF'
cpp(backends-cpu): port spinpool.rs — seqlock op handoff, guided blocks, park/wake

Replaces plan C's inert stubs with the full port: SpinPool with 128-byte-padded
hot atomics, detached workers that spin `SAPIENT_SPINPOOL_SPINS` (default 4000)
iterations then park on a condvar, guided contiguous block claiming (block 1 on
macOS's P/E topology, ~3 blocks/participant elsewhere), a publisher that
participates in its own op, and the macOS QoS pinning for workers and publishers.
`run` is a template over an erased core (Rust's thunk::<F> + ctx), so no
allocation happens per dispatch.

The seqlock order is load-bearing and copied verbatim: generation bumps to ODD
strictly BEFORE the `active` drain. Draining first leaves the window that
produced the measured SIGSEGV; `rapid_ops_with_constant_parking` pins it.

`enabled()` now returns Rust's default — ON for macOS and Linux/aarch64 with at
least 8 threads — so from this commit the whole CPU test suite dispatches its
GEMVs through the pool. All 242 earlier tests stay green, pool on and pool off.

EOF
)"
```

Append the trailer with a blank line before it, exactly:
`Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`

---

### Task 3: the `for_each_out_chunk` pool branch, the `SAPIENT_SPINPOOL_DEBUG` census, and the pool-on/pool-off golden gate

**Files:**
- Modify: `cpp/libs/sapient-backends-cpu/src/kernels/matmul.cpp` (the `detail::for_each_out_chunk` body and the include list)
- Modify: `cpp/libs/sapient-backends-cpu/include/sapient/backends_cpu/kernels/matmul.hpp` (the `for_each_out_chunk` doc comment: drop "plan E adds the spin-pool branch")
- Modify: `cpp/libs/sapient-backends-cpu/tests/spinpool_test.cpp` (append the two route probes)
- Modify: `cpp/libs/sapient-backends-cpu/CMakeLists.txt` (the two gate entries)

**Interfaces:**
- Consumes: `spinpool::enabled()`, `spinpool::pool()` (Task 2), `parallel::par_chunks_mut` (plan C), `sapient::core::panic`.
- Produces: no new public API. `detail::for_each_out_chunk` keeps its signature; only its body changes. Two new gtests `Spinpool.route_is_on_under_env` / `Spinpool.route_is_off_under_env` exist solely to make the two ctest entries non-vacuous.

Rust reference: `crates/sapient-backends/cpu/src/kernels/matmul.rs:471-534` (`SyncPtr`, the census, both branches). Porting map §3.2.

**Ruling — the gate needs an anti-vacuous probe of its own.** Plan D's `q8k_off` entry could detect a non-applied `ENVIRONMENT` property because the `_q8k_off` gtests skip with a distinctive text. Here the vacuous failure is *silent*: if `ENVIRONMENT` were not applied, the `spinpool_on` entry would simply run with the ambient default, still be bit-identical, and pass — proving nothing about the pool branch. So each entry additionally runs a route probe that `GTEST_SKIP()`s with the phrase `spinpool route probe` unless the variable holds exactly the expected value, and each entry carries `FAIL_REGULAR_EXPRESSION "spinpool route probe"`. Costs if wrong: two gtests and two CMake lines.

**Ruling — `SAPIENT_THERMAL=off` on the pool-ON entry.** `enabled()` is `ON && thermal::effective_threads() >= parallel::num_threads()`. On a Linux CI runner with real `/sys/class/thermal` zones, a hot box sheds a core mid-run and `enabled()` flips false — the route probe would fail for a reason that has nothing to do with this plan. Pinning `SAPIENT_THERMAL=off` on that entry makes the assertion deterministic everywhere. The pool-OFF entry needs no such pin (`SAPIENT_SPINPOOL=0` short-circuits the `ON` term). Costs if wrong: a flaky CI job on a hot runner.

**Ruling — the gate filter covers `GoldenKernels.*`, not just `GoldenQuant.*`.** The float GEMV paths (`matmul.cpp:155` and `:175`, the F16 and f32 `m == 1` arms) route through `for_each_out_chunk` exactly as the quantized arms do, so plan C's dense golden cases are part of what the pool branch must not perturb. The `GoldenQuant.*_q8k_off` cases are excluded — they need their own `SAPIENT_Q8K_ACT=0` process and are already gated by plan D's entry.

- [ ] **Step 1: Write the failing tests** — append to `cpp/libs/sapient-backends-cpu/tests/spinpool_test.cpp`:

```cpp
// ── Route probes: the anti-vacuous half of the pool-on/pool-off golden gate ─────────────────
// These exist only so the two ctest entries below can prove their ENVIRONMENT property was
// actually applied. Under the ambient environment they skip; the skip text is what
// FAIL_REGULAR_EXPRESSION matches, so a silently-unset variable fails the entry instead of
// passing it vacuously. Keep the phrase "spinpool route probe" in both texts and nowhere else.

TEST(Spinpool, route_is_on_under_env) {
    const char* v = std::getenv("SAPIENT_SPINPOOL");
    if (v == nullptr || std::string_view(v) != "1") {
        GTEST_SKIP() << "spinpool route probe: needs SAPIENT_SPINPOOL=1 in the environment (the "
                        "sapient_backends_cpu_tests.spinpool_on ctest entry sets it)";
    }
    ASSERT_TRUE(spinpool::enabled())
        << "SAPIENT_SPINPOOL=1 must route for_each_out_chunk through the spin pool";
}

TEST(Spinpool, route_is_off_under_env) {
    const char* v = std::getenv("SAPIENT_SPINPOOL");
    if (v == nullptr || std::string_view(v) != "0") {
        GTEST_SKIP() << "spinpool route probe: needs SAPIENT_SPINPOOL=0 in the environment (the "
                        "sapient_backends_cpu_tests.spinpool_off ctest entry sets it)";
    }
    ASSERT_FALSE(spinpool::enabled())
        << "SAPIENT_SPINPOOL=0 must route for_each_out_chunk through parallel::par_chunks_mut";
}
```

Add `#include <cstdlib>` and `#include <string_view>` to that file's include list.

Then add the two entries to `cpp/libs/sapient-backends-cpu/CMakeLists.txt`, immediately after the existing `q8k_off` block:

```cmake
  # `spinpool::enabled()` is a once-per-process read (Rust OnceLock twin), so the pool-on and
  # pool-off halves of plan E's gate need their own processes. Both run the FULL golden suite: the
  # float GEMV arms of matmul_nt route through for_each_out_chunk too, not just the quantized ones.
  # The `_q8k_off` cases are excluded — they belong to the q8k_off entry's environment.
  #
  # Unlike q8k_off, a non-applied ENVIRONMENT property here would fail SILENTLY (the suite would
  # run on the ambient default and still be bit-identical), so each entry also runs a route probe
  # that skips with the text "spinpool route probe" unless its variable is set — and that text is
  # what FAIL_REGULAR_EXPRESSION rejects.
  add_test(NAME sapient_backends_cpu_tests.spinpool_on
           COMMAND sapient_backends_cpu_tests
                   "--gtest_filter=Golden*.*:Spinpool.route_is_on_under_env-GoldenQuant.*_q8k_off")
  # SAPIENT_THERMAL=off keeps the route assertion deterministic on a Linux runner with real
  # thermal zones: a hot box would shed a core and flip enabled() to false mid-run.
  set_tests_properties(sapient_backends_cpu_tests.spinpool_on PROPERTIES
                        ENVIRONMENT "SAPIENT_SPINPOOL=1;SAPIENT_THERMAL=off"
                        FAIL_REGULAR_EXPRESSION "spinpool route probe")

  add_test(NAME sapient_backends_cpu_tests.spinpool_off
           COMMAND sapient_backends_cpu_tests
                   "--gtest_filter=Golden*.*:Spinpool.route_is_off_under_env-GoldenQuant.*_q8k_off")
  set_tests_properties(sapient_backends_cpu_tests.spinpool_off PROPERTIES
                        ENVIRONMENT "SAPIENT_SPINPOOL=0"
                        FAIL_REGULAR_EXPRESSION "spinpool route probe")
```

- [ ] **Step 2: Run the tests to verify the gate fails for the right reason**

Run: `cd cpp && cmake --build --preset dev && ctest --preset dev -R 'spinpool_(on|off)' -V 2>&1 | tail -20`
Expected: `sapient_backends_cpu_tests.spinpool_on` **passes** (Task 2 already made `enabled()` true on macOS, so the probe's assertion holds) and `sapient_backends_cpu_tests.spinpool_off` **passes** too — but both are still exercising only ONE code path, because `for_each_out_chunk` ignores `spinpool::enabled()` until Step 3. Record in the ledger that this step's real verification is Step 4's before/after comparison, not a red test: the branch being added is a *route*, and the only honest red signal available is the `SAPIENT_SPINPOOL_DEBUG` census printing `spin=0` in Step 5.

- [ ] **Step 3: Wire the pool branch**

In `cpp/libs/sapient-backends-cpu/src/kernels/matmul.cpp`, add `#include <atomic>`, `#include <cstdio>` and `#include <cstdlib>` to the include list if absent, and replace the `detail::for_each_out_chunk` body:

```cpp
void for_each_out_chunk(std::span<float> out,
                        size_t chunk,
                        const std::function<void(size_t, std::span<float>)>& f) {
    if (out.empty()) return;
    if (chunk == 0) sapient::core::panic("for_each_out_chunk: chunk size must not be zero");

    // SAPIENT_SPINPOOL_DEBUG=1: periodic dispatch-route census on stderr (matmul.rs:493-516).
    // Read via getenv on EVERY call, like Rust's uncached `std::env::var(..).is_ok()`.
    if (std::getenv("SAPIENT_SPINPOOL_DEBUG") != nullptr) {
        static std::atomic<uint64_t> spin_dispatches{0};
        static std::atomic<uint64_t> pool_dispatches{0}; // the `RAYON` counter's twin
        uint64_t s = 0;
        uint64_t r = 0;
        if (spinpool::enabled()) {
            s = spin_dispatches.fetch_add(1, std::memory_order_relaxed) + 1;
            r = pool_dispatches.load(std::memory_order_relaxed);
        } else {
            s = spin_dispatches.load(std::memory_order_relaxed);
            r = pool_dispatches.fetch_add(1, std::memory_order_relaxed) + 1;
        }
        if ((s + r) % 2000 == 0) {
            std::fprintf(stderr,
                         "[spinpool-debug] spin=%llu rayon=%llu chunk=%zu len=%zu n_chunks=%zu\n",
                         static_cast<unsigned long long>(s), static_cast<unsigned long long>(r),
                         chunk, out.size(), (out.size() + chunk - 1) / chunk);
        }
    }

    if (spinpool::enabled()) {
        // Rust's SyncPtr: chunk geometry guarantees the spans built from `base` are disjoint, the
        // same contract par_chunks_mut relies on, and `out` outlives `run` (it blocks until every
        // chunk completes). The partition below is character-for-character the one in
        // parallel::par_chunks_mut — that identity is what makes the two routes bit-identical.
        const size_t len = out.size();
        const size_t n_chunks = (len + chunk - 1) / chunk;
        float* const base = out.data();
        spinpool::pool().run(n_chunks, [&](size_t ci) {
            const size_t start = ci * chunk;
            const size_t end = std::min(start + chunk, len);
            // A throwing callback would unwind through the pool's op slot (or std::terminate on a
            // worker); abort cleanly instead, exactly as parallel::invoke does on the other route.
            try {
                f(ci, std::span<float>(base + start, end - start));
            } catch (...) {
                sapient::core::panic("for_each_out_chunk: a callback threw an exception "
                                     "(callbacks must not throw)");
            }
        });
        return;
    }
    parallel::par_chunks_mut(out, chunk, f);
}
```

Update the declaration's comment in `kernels/matmul.hpp` from "Plan C runs the rayon-twin branch only; plan E adds the spin-pool branch (identical partition)." to "Both routes — the spin pool when `spinpool::enabled()`, `parallel::par_chunks_mut` otherwise — produce the identical partition; that is what makes them bit-identical." and drop "plan E adds the spin-pool branch of `for_each_out_chunk`" from the file's top-of-file comment.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd cpp && cmake --build --preset dev && ctest --preset dev`
Expected: **257 tests, 100%** (253 after Task 2, plus the 2 route probes as discovered tests and the 2 new gate entries). No test may change its result relative to Task 2.

- [ ] **Step 5: Prove the route actually switches (the census)**

Run:
```bash
cd cpp && SAPIENT_SPINPOOL=1 SAPIENT_THERMAL=off SAPIENT_SPINPOOL_DEBUG=1 \
  ./build/dev/libs/sapient-backends-cpu/sapient_backends_cpu_tests \
  --gtest_filter='Matmul.*:Quant.*' 2>&1 | grep -c 'spinpool-debug.*rayon=0'
```
Expected: at least 1 — a census line with a non-zero `spin=` and `rayon=0`, proving the dispatches took the pool route. Then the mirror:
```bash
cd cpp && SAPIENT_SPINPOOL=0 SAPIENT_SPINPOOL_DEBUG=1 \
  ./build/dev/libs/sapient-backends-cpu/sapient_backends_cpu_tests \
  --gtest_filter='Matmul.*:Quant.*' 2>&1 | grep -c 'spinpool-debug.*spin=0'
```
Expected: at least 1 — `spin=0` with a non-zero `rayon=`. If either count is 0, the suite did not reach 2 000 dispatches on that filter; widen it to `--gtest_filter='*'` before concluding the wiring is wrong. Paste both numbers into the task report — this is the step that distinguishes a working branch from a dead one.

- [ ] **Step 6: Verify the gate rejects a vacuous run**

Temporarily change the `spinpool_on` entry's `ENVIRONMENT` to `"SAPIENT_THERMAL=off"` (dropping `SAPIENT_SPINPOOL=1`), re-run `cmake --build --preset dev && ctest --preset dev -R spinpool_on`, and confirm it **FAILS** on the `spinpool route probe` text. Then restore the line and confirm it passes again. Report both outcomes — an anti-vacuous guard that has never been seen to fire is not a guard.

- [ ] **Step 7: Confirm the Rust oracle is untouched, then format and commit**

Run: `git status --short crates/` (expect no output) and `cargo test -p sapient-backends-cpu 2>&1 | tail -3` (expect `71 passed; 0 failed; 1 ignored`).

```bash
git add cpp/libs/sapient-backends-cpu/src/kernels/matmul.cpp \
        cpp/libs/sapient-backends-cpu/include/sapient/backends_cpu/kernels/matmul.hpp \
        cpp/libs/sapient-backends-cpu/tests/spinpool_test.cpp \
        cpp/libs/sapient-backends-cpu/CMakeLists.txt
.superpowers/tools-venv/bin/clang-format -i \
        cpp/libs/sapient-backends-cpu/src/kernels/matmul.cpp \
        cpp/libs/sapient-backends-cpu/include/sapient/backends_cpu/kernels/matmul.hpp \
        cpp/libs/sapient-backends-cpu/tests/spinpool_test.cpp
git add -u
git commit -m "$(cat <<'EOF'
cpp(backends-cpu): route for_each_out_chunk through the spin pool, gate both routes

Adds the spinpool branch and the SAPIENT_SPINPOOL_DEBUG census to
matmul::detail::for_each_out_chunk. Both routes build the identical
(chunk index -> [start, end)) partition, which is the whole basis of their
bit-identity; the pool route wraps the callback so a throw aborts cleanly
instead of unwinding through the op slot.

The gate is two new ctest entries running the full golden suite in their own
processes (SAPIENT_SPINPOOL is a once-per-process read): spinpool_on with the
pool forced on, spinpool_off with it forced off. A non-applied ENVIRONMENT
property would fail silently here, so each entry also runs a route probe that
skips with the text "spinpool route probe" unless its variable is set, and
FAIL_REGULAR_EXPRESSION rejects that text.

EOF
)"
```

Append the trailer with a blank line before it, exactly:
`Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`

---

### Task 4: docs, parity ledger, final verification

**Files:**
- Modify: `CLAUDE.md` (the "C++ rewrite programme" section), `docs/ROADMAP.md` (the sub-project 1a progress row), `docs/PARITY.md` (plan E's gate rows + close the carried-in open gaps that plan E resolves), `docs/PROJECT_GUIDE.md` (the C++ tree's module list), `CHANGELOG.md` (the unreleased section), `docs/superpowers/specs/2026-09-21-cpp-sp1a-core-io-cpu-design.md` (§4 and §5 "as built" notes)

**Interfaces:** none — documentation only.

- [ ] **Step 1: Record the parity results in `docs/PARITY.md`**

Add plan E's rows following the format plans A/C/D already use in that file (read them first and match the column layout exactly). The rows must carry: the gate name, the host (`macOS arm64, Apple M5`), both trees' commit hashes, what was compared, and the result. At minimum:

- `sp1a plan E — spinpool/thermal unit tests`: 6 thermal + 5 spinpool Rust tests ported by name, all passing; 1 `#[ignore]` probe ported as `DISABLED_`.
- `sp1a plan E — golden suite, pool ON`: full `Golden*` suite under `SAPIENT_SPINPOOL=1 SAPIENT_THERMAL=off`, bit-identical to the Rust dumps.
- `sp1a plan E — golden suite, pool OFF`: same suite under `SAPIENT_SPINPOOL=0`, bit-identical — i.e. pool-on == pool-off == Rust.
- `sp1a plan E — stress`: `rapid_ops_with_constant_parking` × 20 consecutive runs clean.

Also update the **open gaps** section: plan E closes "the `for_each_out_chunk` pool branch" and "`thermal`/`spinpool` are inert stubs". The two **known-defect** rows carried since plans C and D (the NEON f16 mis-decode in `dot_f32_x_f16_neon`, the AVX2 out-of-bounds read in `dot_q8_0_row_avx2`) are **untouched by this plan** — leave them exactly as they are, still awaiting the user's decision. Do not mark them resolved.

New gaps to record, honestly:
- The spin pool's behaviour on x86_64 and Windows is **compile-verified only**; `enabled()` defaults to off there, so CI exercises the `parallel` route on those hosts. The pool route's only real-hardware coverage is macOS arm64 until a Linux/aarch64 runner exists.
- `SAPIENT_SPINPOOL_BLOCK`'s non-macOS default (`n_chunks / (3 · participants)`) is untested on real hardware for the same reason.
- No performance claim is made or measured: `DISABLED_pool_speedup_probe` is available but is not part of any gate, and spec §2.3 explicitly makes performance a post-1a concern.

- [ ] **Step 2: Update `CLAUDE.md`**

In the "C++ rewrite programme" section, record that sub-project 1a's plan E is complete: `sapient::backends_cpu` now has the real `thermal` governor and `spinpool`, `for_each_out_chunk` dispatches on `spinpool::enabled()`, and the golden suite is gated in both routes. Add the lessons worth carrying into plan B and sub-project 1b:
- **`std::filesystem`'s throwing overloads must never appear in a path reachable from a static initialiser** — `/sys/class/thermal` is absent on macOS/Windows, and the throwing `directory_iterator` would have terminated the process on the first matmul.
- **`enabled()` defaults ON on macOS**, so the whole CPU suite runs on the spin pool by default; any future kernel work must keep both routes green (`ctest --preset dev` already runs the OFF route as a separate entry).
- **A gate whose environment silently fails to apply is not a gate** — the route probes plus `FAIL_REGULAR_EXPRESSION` are the pattern to reuse whenever a ctest entry depends on a once-per-process env read.
- The spin pool's publish mutex is **not recursive**: a kernel that nested parallel regions would deadlock. No kernel does today; a future one must use `parallel` for the inner region.

Keep the existing "Must follow" doc rules intact.

- [ ] **Step 3: Update `docs/ROADMAP.md`, `docs/PROJECT_GUIDE.md` and `CHANGELOG.md`**

- ROADMAP: mark plan E complete in the sub-project 1a row; the remaining 1a work is plan B (io).
- PROJECT_GUIDE: in the C++ tree's module list, `thermal` and `spinpool` are no longer stubs — describe them in one line each.
- CHANGELOG: add an entry under the unreleased section noting the C++ port's spin pool and thermal governor. Keep the `## [X.Y.Z] - date` heading contract intact (`release.yml` awk-extracts by it).

- [ ] **Step 4: Update the spec's "as built" notes**

In `docs/superpowers/specs/2026-09-21-cpp-sp1a-core-io-cpu-design.md`:
- §4 "Dump cases added by plan", row E: record that the gate landed as two ctest entries re-running plan D's existing dumps in two process environments, plus two route probes — **no new dump cases and no Rust changes** were needed.
- §5, plan E row: mark the gate satisfied.
- §2.3 `spinpool` row: note the two approved deviations — `run` is a template over an erased core (Rust's `thunk::<F>`), and Windows thread naming is omitted.

- [ ] **Step 5: Full verification**

Run each and paste the output into the task report:
```bash
cd cpp && cmake --build --preset dev && ctest --preset dev 2>&1 | tail -5
cd cpp && ctest --preset dev -R 'spinpool_(on|off)|q8k_off' -V 2>&1 | tail -15
cargo test -p sapient-backends-cpu 2>&1 | tail -3
cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -3
git status --short crates/
just cpp-fmt && git diff --stat
```
Expected: 257/257 ctest; the three env-gated entries pass; `71 passed; 0 failed; 1 ignored`; clippy clean; no `crates/` changes; no formatting diff.

Also re-run the golden dump end to end, since plan E is the first plan that changes *how* the kernels are dispatched rather than what they compute:
```bash
just cpp-golden /tmp/sapient-golden-e && SAPIENT_GOLDEN_DIR=/tmp/sapient-golden-e just cpp-test 2>&1 | tail -5
```
Expected: 100%, with the golden cases comparing rather than skipping.

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md docs/ROADMAP.md docs/PARITY.md docs/PROJECT_GUIDE.md CHANGELOG.md \
        docs/superpowers/specs/2026-09-21-cpp-sp1a-core-io-cpu-design.md
git commit -m "$(cat <<'EOF'
docs(sp1a plan E): parity rows, CLAUDE.md/ROADMAP/PROJECT_GUIDE/CHANGELOG, spec as-built

Records plan E's gates in docs/PARITY.md: the 11 ported unit tests, the golden
suite bit-identical with the spin pool ON and OFF, and 20 clean stress runs. New
honest gaps recorded: the pool route is real-hardware-verified on macOS arm64
only (enabled() defaults off on x86/Windows), and no performance claim is made.

The two known oracle defects carried from plans C and D (the NEON f16 mis-decode
and the AVX2 out-of-bounds read) are untouched and still await a decision.

CONTRIBUTING.md and README.md need no change — plan E is library-internal and
adds no build step, dependency or user-facing surface.

EOF
)"
```

Append the trailer with a blank line before it, exactly:
`Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`

---

## Self-review against the spec

**Spec coverage (§2.3 `spinpool` and `thermal` rows, §5 row E):**

| Spec requirement | Task |
|---|---|
| `OpSlot`, `alignas(128) Pad<T>`, same atomic orderings | 2 |
| `std::mutex`/`std::condition_variable` (Rust used std, not parking_lot) | 2 |
| worker threads named `sapient-spin-N` | 2 (macOS/Linux; Windows omitted by ruling) |
| macOS QoS pinning, workers and publishers (once per thread) | 2 |
| seqlock protocol, generation ODD **before** draining `active` | 2 (Global Constraints + code comment + stress test) |
| guided block claiming with the per-OS block-size rule | 2 |
| `run()` serial fast paths (`n_chunks == 0`, `== 1`, `workers == 0`) | 2 |
| process-lifetime singleton `pool()`, `parallelism()` | 2 |
| `enabled()` = env/platform default AND `thermal::effective_threads() >= parallel::num_threads()` | 2 |
| Env `SAPIENT_SPINPOOL`, `_WORKERS`, `_SPINS` (default 4000), `_BLOCK`, `_DEBUG` | 2 (first four), 3 (`_DEBUG`) |
| `ThermalGovernor` sorted zones, 80/70 hysteresis, floor `max/2`, one-shot warning | 1 |
| external level cap (0..3 → full/¾/½/¼, floor 1) | 1 |
| `effective_threads()` = stricter of the two | 1 |
| `tick()` rate-limited to 500 ms with the compare-exchange winner sampling | 1 |
| Env `SAPIENT_THERMAL`, `_PATH`, `_HOT`, `_COOL`; `std::filesystem` | 1 |
| §3.7 identical partition on both routes | 3 |
| §5 gate: 5+1 spinpool and 6 thermal tests | 1 (6), 2 (5+1) |
| §5 gate: D golden suite pool-on == pool-off | 3 |
| Doc updates | 4 |

**Placeholder scan:** every step carries its command or its complete code; no "TBD", no "similar to Task N", no "add appropriate error handling".

**Type consistency:** `ThermalGovernor::sample()` is `const` in both the header (Step 3) and the implementation (Step 4) and is called on `const ThermalGovernor` objects in the tests (Step 1) and through the `const ThermalGovernor*` singleton (`tick`). `SpinPool::create` returns `SpinPool&` in the header, the implementation, `pool()`, and the stress test. `detail::external_cap(uint8_t, size_t)` matches between header, implementation and `external_cap_mapping`. `for_each_out_chunk`'s signature is unchanged from plan C, so every one of its eight call sites in `matmul.cpp` keeps compiling untouched.

**Ordering check:** Task 1 must precede Task 2 (`spinpool::enabled()` calls `thermal::effective_threads()`); Task 2 must precede Task 3 (`for_each_out_chunk` calls `pool()`); Task 4 is last because it records results the first three produce.
