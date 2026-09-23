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
    : hot_mc_(hot_c * 1000), cool_mc_(cool_c * 1000),
      max_threads_(std::max<size_t>(max_threads, 1)),
      min_threads_(std::max<size_t>(std::max<size_t>(max_threads, 1) / 2, 1)),
      effective_(std::max<size_t>(max_threads, 1)), warned_(false) {
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
                         static_cast<long long>(hot_mc_ / 1000),
                         next,
                         max_threads_);
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
                     static_cast<unsigned>(prev),
                     static_cast<unsigned>(lv),
                     effective_threads());
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
    const auto now_ms = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                                  std::chrono::steady_clock::now() - epoch)
                                                  .count());
    uint64_t last = last_ms.load(std::memory_order_relaxed);
    if (now_ms <= last || now_ms - last < kTickMs) return; // Rust `now_ms.saturating_sub(last)`
    // Only the thread that wins the exchange samples; the others fall through (Rust's
    // compare_exchange guard). `compare_exchange_strong` writes `last` on failure — harmless, we
    // return either way.
    if (last_ms.compare_exchange_strong(
            last, now_ms, std::memory_order_relaxed, std::memory_order_relaxed)) {
        g->sample();
    }
}

} // namespace sapient::backends_cpu::thermal
