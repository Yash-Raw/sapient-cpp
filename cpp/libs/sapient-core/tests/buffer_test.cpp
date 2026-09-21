// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "sapient/core/buffer.hpp"
#include "sapient/core/dtype.hpp"

using namespace sapient::core;

// Rust: zeros_and_read
TEST(CpuBuffer, zeros_and_read) {
    auto b = CpuBuffer::zeros(4, DType::F32);
    ASSERT_TRUE(b.has_value());
    EXPECT_EQ((*b)->len(), 16u);
    for (const auto byte : (*b)->bytes())
        EXPECT_EQ(byte, 0);
    EXPECT_EQ((*b)->alignment(), 64u);
    EXPECT_EQ((*b)->device(), "cpu");
    EXPECT_FALSE((*b)->is_mmap());
}
// Rust: from_f32_roundtrip
TEST(CpuBuffer, from_f32_roundtrip) {
    const float in[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto b = CpuBuffer::from_f32_slice(in);
    ASSERT_TRUE(b.has_value());
    const auto f = (*b)->f32s();
    ASSERT_EQ(f.size(), 4u);
    for (size_t i = 0; i < 4; ++i)
        EXPECT_EQ(f[i], in[i]);
}
// Rust: alignment_guarantee
TEST(CpuBuffer, alignment_guarantee) {
    auto b = CpuBuffer::with_capacity(32, 64);
    ASSERT_TRUE(b.has_value());
    EXPECT_EQ(reinterpret_cast<uintptr_t>((*b)->data()) % 64, 0u);
    EXPECT_EQ((*b)->len(), 32u);
}
TEST(CpuBuffer, from_f32_vec_moves_without_copy) {
    std::vector<float> v = {1.5f, 2.5f};
    const float* before = v.data();
    auto b = CpuBuffer::from_f32_vec(std::move(v));
    EXPECT_EQ(reinterpret_cast<const float*>(b->data()), before);
    EXPECT_EQ(b->len(), 8u);
    EXPECT_EQ(b->alignment(), 4u);
    EXPECT_EQ(b->f32s()[1], 2.5f);
}
TEST(CpuBuffer, from_bytes_slice_copies_with_align_16) {
    const uint8_t in[] = {1, 2, 3};
    auto b = CpuBuffer::from_bytes_slice(in);
    ASSERT_TRUE(b.has_value());
    EXPECT_EQ((*b)->alignment(), 16u);
    EXPECT_EQ((*b)->bytes()[2], 3);
    (*b)->bytes_mut()[0] = 9;
    EXPECT_EQ((*b)->bytes()[0], 9);
}
TEST(CpuBuffer, zero_length_is_valid) {
    auto b = CpuBuffer::with_capacity(0, 16);
    ASSERT_TRUE(b.has_value());
    EXPECT_EQ((*b)->len(), 0u);
    EXPECT_TRUE((*b)->is_empty());
    EXPECT_NE((*b)->data(), nullptr); // Rust allocates 1 byte for a 0-length buffer
}
TEST(CpuBuffer, bad_alignment_is_an_error) {
    auto b =
        CpuBuffer::with_capacity(8, 3); // not a power of two → Rust Layout error → AllocationFailed
    ASSERT_FALSE(b.has_value());
    EXPECT_EQ(b.error().to_string(), "Allocation failed: requested 8 bytes (alignment 3)");
}
