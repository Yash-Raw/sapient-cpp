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
      workers_(workers), spin_iters_(spin_iters) {}

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
            sapient::core::panic(
                "spawn spinpool worker"); // Rust `.expect("spawn spinpool worker")`
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
        for (size_t c = lo; c < hi; ++c)
            op.call(op.ctx, c);
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
        for (size_t c = 0; c < n_chunks; ++c)
            call(ctx, c);
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
    while (active_.v.load(std::memory_order_seq_cst) != 0)
        spin_hint();

    // Block size is TOPOLOGY-dependent (both directions measured): heterogeneous P/E cores
    // (M-series) need block = 1 — an E-core claiming a late multi-chunk block adds a straggler tail
    // to every op (llama-1B lm_head block ≈ 8 measured 62 → 44 tok/s on M4); homogeneous server ARM
    // wants ~3 blocks/participant — per-chunk claim+completion RMW traffic was Thor's 2×
    // regression, and guided blocks took it to +8% OVER the fork/join path.
    [[maybe_unused]] const size_t participants = workers_ + 1;
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
    while (completed_.v.load(std::memory_order_acquire) < n_blocks)
        spin_hint();
}

SpinPool& pool() {
    // Rust `OnceLock` twin; the reference is bound once and the pool lives for the process.
    static SpinPool& instance = []() -> SpinPool& {
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
