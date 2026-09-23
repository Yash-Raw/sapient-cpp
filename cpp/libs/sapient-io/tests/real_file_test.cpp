// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
// Env-gated real-file checks (spec §4 plan B gate). Unset → SKIP with the exact text the CI step
// greps for; set but unreadable → FAIL (a stale path must not silently downgrade the gate).
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>

#include "sapient/core/dtype.hpp"
#include "sapient/io/gguf.hpp"
#include "sapient/io/safetensors.hpp"

using sapient::io::gguf::GgufLoader;

TEST(RealFile, gguf_heap_and_mmap_agree) {
    const char* env = std::getenv("SAPIENT_TEST_GGUF");
    if (env == nullptr || *env == '\0') GTEST_SKIP() << "SAPIENT_TEST_GGUF unset";
    const std::filesystem::path p(env);
    std::error_code ec;
    ASSERT_TRUE(std::filesystem::is_regular_file(p, ec))
        << "SAPIENT_TEST_GGUF=" << env << " is not a readable file";
    auto heap = GgufLoader::load_tensors_with_metadata(p);
    ASSERT_TRUE(heap.has_value()) << heap.error().to_string();
    auto mm = GgufLoader::load_tensors_mmap(p);
    ASSERT_TRUE(mm.has_value()) << mm.error().to_string();
    auto md = GgufLoader::parse_metadata_only(p);
    ASSERT_TRUE(md.has_value()) << md.error().to_string();

    EXPECT_EQ(heap->first, mm->first);
    EXPECT_EQ(heap->first, *md);
    ASSERT_FALSE(heap->second.empty());
    ASSERT_EQ(heap->second.size(), mm->second.size());
    size_t n_mmap = 0;
    size_t n_requant = 0;
    for (const auto& [name, h] : heap->second) {
        SCOPED_TRACE(name);
        const auto it = mm->second.find(name);
        ASSERT_NE(it, mm->second.end());
        const auto& m = it->second;
        EXPECT_EQ(h.dtype(), m.dtype());
        EXPECT_EQ(h.shape(), m.shape());
        EXPECT_FALSE(h.is_mmap());
        const auto hb = h.bytes();
        const auto mb = m.bytes();
        ASSERT_EQ(hb.size(), mb.size());
        EXPECT_TRUE(std::equal(hb.begin(), hb.end(), mb.begin()));
        if (m.is_mmap()) {
            ++n_mmap;
            EXPECT_TRUE(sapient::core::is_quantized(m.dtype()));
        } else if (sapient::core::is_quantized(m.dtype())) {
            ++n_requant; // only a Q5_0 source re-quantised to Q8_0 is quantized AND not mmap'd
            EXPECT_EQ(m.dtype(), sapient::core::DType::Q8_0);
        } else {
            EXPECT_EQ(m.dtype(), sapient::core::DType::F32);
        }
    }
    EXPECT_GT(n_mmap, 0u);
    std::printf("[real-file] %s: %zu tensors, %zu zero-copy, %zu requantised Q5_0->Q8_0, %zu KVs\n",
                env,
                heap->second.size(),
                n_mmap,
                n_requant,
                heap->first.size());
}

TEST(RealFile, safetensors_loads) {
    const char* env = std::getenv("SAPIENT_TEST_SAFETENSORS");
    if (env == nullptr || *env == '\0') GTEST_SKIP() << "SAPIENT_TEST_SAFETENSORS unset";
    const std::filesystem::path p(env);
    std::error_code ec;
    ASSERT_TRUE(std::filesystem::is_regular_file(p, ec))
        << "SAPIENT_TEST_SAFETENSORS=" << env << " is not a readable file";
    auto m = sapient::io::safetensors::SafetensorsLoader::load(p);
    ASSERT_TRUE(m.has_value()) << m.error().to_string();
    ASSERT_FALSE(m->empty());
    for (const auto& [name, t] : *m) {
        SCOPED_TRACE(name);
        EXPECT_FALSE(t.is_mmap());
        const auto dt = t.dtype();
        EXPECT_TRUE(dt == sapient::core::DType::F32 || dt == sapient::core::DType::F16 ||
                    dt == sapient::core::DType::BF16);
        EXPECT_EQ(t.bytes().size(), sapient::core::byte_count(dt, t.numel()));
    }
    std::printf("[real-file] %s: %zu tensors\n", env, m->size());
}
