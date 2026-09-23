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
#include <cstdlib>
#include <string_view>
#include <thread>
#include <vector>

#include "sapient/backends_cpu/spinpool.hpp"

namespace spinpool = sapient::backends_cpu::spinpool;

TEST(Spinpool, runs_every_chunk_exactly_once) {
    spinpool::SpinPool& pool = spinpool::pool();
    for (size_t round = 0; round < 200; ++round) {
        const size_t n = 1 + (round % 61);
        std::vector<std::atomic<uint32_t>> hits(n);
        for (auto& h : hits)
            h.store(0, std::memory_order_seq_cst);
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
                for (auto& h : hits)
                    h.store(0, std::memory_order_seq_cst);
                pool.run(37,
                         [&hits](size_t c) { hits[c].fetch_add(1, std::memory_order_seq_cst); });
                for (auto& h : hits) {
                    if (h.load(std::memory_order_seq_cst) != 1)
                        failures.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto& th : threads)
        th.join();
    EXPECT_EQ(failures.load(std::memory_order_relaxed), 0u)
        << "a chunk ran zero or multiple times while publishers overlapped";
}

TEST(Spinpool, survives_park_and_wake) {
    spinpool::SpinPool& pool = spinpool::pool();
    std::vector<std::atomic<uint32_t>> hits(16);
    for (auto& h : hits)
        h.store(0, std::memory_order_seq_cst);
    pool.run(16, [&hits](size_t c) { hits[c].fetch_add(1, std::memory_order_seq_cst); });
    // Sleep well past any spin budget so the workers park, then dispatch again.
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    std::vector<std::atomic<uint32_t>> hits2(16);
    for (auto& h : hits2)
        h.store(0, std::memory_order_seq_cst);
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
    for (auto& s : sink)
        s.store(0, std::memory_order_relaxed);
    const auto busy = [&sink](size_t c) {
        uint64_t x = static_cast<uint64_t>(c) ^ 0x9e3779b97f4a7c15ULL;
        for (uint64_t i = 0; i < kWorkPerChunk; ++i)
            x = x * 6364136223846793005ULL + i;
        sink[c].store(x, std::memory_order_relaxed);
    };
    pool.run(kNChunks, busy); // warm both paths once
    for (size_t c = 0; c < kNChunks; ++c)
        busy(c);

    auto t = std::chrono::steady_clock::now();
    for (int r = 0; r < 50; ++r)
        for (size_t c = 0; c < kNChunks; ++c)
            busy(c);
    const auto serial = std::chrono::steady_clock::now() - t;

    t = std::chrono::steady_clock::now();
    for (int r = 0; r < 50; ++r)
        pool.run(kNChunks, busy);
    const auto par = std::chrono::steady_clock::now() - t;

    const double serial_s = std::chrono::duration<double>(serial).count();
    const double par_s = std::chrono::duration<double>(par).count();
    std::printf("workers=%zu serial=%.3fs pool=%.3fs speedup=%.2fx\n",
                pool.workers(),
                serial_s,
                par_s,
                serial_s / par_s);

    t = std::chrono::steady_clock::now();
    for (int r = 0; r < 10000; ++r)
        pool.run(kNChunks, [&sink](size_t c) {
            sink[c].store(static_cast<uint64_t>(c), std::memory_order_relaxed);
        });
    const double tiny_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
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
        for (size_t v = 0; v < n; ++v)
            payload[v] = static_cast<uint64_t>(v + round);
        std::vector<std::atomic<uint64_t>> acc(n);
        for (auto& a : acc)
            a.store(0, std::memory_order_seq_cst);
        pool.run(n, [&acc, &payload](size_t c) {
            acc[c].store(payload[c] * 2, std::memory_order_seq_cst);
        });
        for (size_t c = 0; c < n; ++c) {
            ASSERT_EQ(acc[c].load(std::memory_order_seq_cst), payload[c] * 2)
                << "round " << round << " chunk " << c;
        }
    }
}

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
