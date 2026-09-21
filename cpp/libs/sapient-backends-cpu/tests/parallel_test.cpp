// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include "sapient/backends_cpu/parallel.hpp"

using namespace sapient::backends_cpu::parallel;

// C++-only tests: `parallel` replaces rayon; what parity depends on is the chunk→range partition.

TEST(Parallel, num_threads_is_at_least_one_and_stable) {
    EXPECT_GE(num_threads(), 1u);
    EXPECT_EQ(num_threads(), num_threads());
}

TEST(Parallel, par_for_visits_each_index_exactly_once) {
    const size_t n = 1000;
    std::vector<std::atomic<int>> hits(n);
    for (auto& h : hits)
        h.store(0);
    par_for(n, [&](size_t i) { hits[i].fetch_add(1); });
    for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(hits[i].load(), 1) << "index " << i;
}

TEST(Parallel, par_for_zero_makes_no_calls) {
    std::atomic<int> calls{0};
    par_for(0, [&](size_t) { calls.fetch_add(1); });
    EXPECT_EQ(calls.load(), 0);
}

// rayon: out.par_chunks_mut(64).enumerate() over 1000 elements → chunk ci covers
// [ci*64, min((ci+1)*64, 1000)); 16 chunks, the last one 40 long. No gtest assertions inside the
// lambda (it runs on pool threads; gtest assertions are not thread-safe on Windows) — record, then assert.
TEST(Parallel, par_chunks_mut_partition_matches_rayon) {
    std::vector<float> out(1000, -1.0f);
    std::vector<std::atomic<int>> seen(16);
    for (auto& s : seen)
        s.store(0);
    std::vector<size_t> lens(16, 0);
    std::atomic<int> out_of_range{0};
    par_chunks_mut(out, 64, [&](size_t ci, std::span<float> cs) {
        if (ci >= 16) {
            out_of_range.fetch_add(1);
            return;
        }
        seen[ci].fetch_add(1);
        lens[ci] = cs.size();
        for (float& v : cs)
            v = static_cast<float>(ci);
    });
    EXPECT_EQ(out_of_range.load(), 0);
    for (size_t ci = 0; ci < 16; ++ci) {
        EXPECT_EQ(seen[ci].load(), 1) << "chunk " << ci;
        EXPECT_EQ(lens[ci], ci == 15 ? 40u : 64u) << "chunk " << ci;
    }
    for (size_t i = 0; i < out.size(); ++i)
        EXPECT_EQ(out[i], static_cast<float>(i / 64)) << i;

    // Exact multiple: one chunk that is the whole slice.
    std::atomic<int> calls{0};
    std::atomic<int> bad{0};
    par_chunks_mut(out, 1000, [&](size_t ci, std::span<float> cs) {
        calls.fetch_add(1);
        if (ci != 0 || cs.size() != 1000) bad.fetch_add(1);
    });
    EXPECT_EQ(calls.load(), 1);
    EXPECT_EQ(bad.load(), 0);

    // Empty slice: no calls (rayon yields no chunks).
    std::atomic<int> empty_calls{0};
    std::vector<float> empty;
    par_chunks_mut(empty, 8, [&](size_t, std::span<float>) { empty_calls.fetch_add(1); });
    EXPECT_EQ(empty_calls.load(), 0);
}

TEST(Parallel, nested_par_for_completes) {
    std::atomic<int> count{0};
    par_for(8, [&](size_t) { par_for(8, [&](size_t) { count.fetch_add(1); }); });
    EXPECT_EQ(count.load(), 64);
}

// rayon panics on `par_chunks_mut(0)` ("chunk size must not be zero"); the twin panics too.
TEST(ParallelDeath, par_chunks_mut_zero_chunk_panics) {
    GTEST_FLAG_SET(death_test_style, "threadsafe"); // pool threads may already exist
    std::vector<float> out(4, 0.0f);
    EXPECT_DEATH(par_chunks_mut(out, 0, [](size_t, std::span<float>) {}), "chunk");
}

// A callback that throws must not unwind through the pool (Job would be destroyed while a
// worker/queue still references it) — it aborts instead, like a rayon closure panic under
// panic=abort.
TEST(ParallelDeath, throwing_callback_aborts) {
    GTEST_FLAG_SET(death_test_style, "threadsafe"); // pool threads may already exist
    EXPECT_DEATH(par_for(4, [](size_t) { throw std::runtime_error("boom"); }), "callback threw");
}
