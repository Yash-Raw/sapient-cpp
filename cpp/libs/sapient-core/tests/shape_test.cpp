// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <vector>

#include "sapient/core/shape.hpp"

using sapient::core::Shape;

// Rust: numel
TEST(Shape, numel) {
    EXPECT_EQ(Shape({2, 3, 4}).numel(), 24u);
    EXPECT_EQ(Shape::scalar().numel(), 1u);
}
// Rust: strides_row_major
TEST(Shape, strides_row_major) {
    EXPECT_EQ(Shape({2, 3, 4}).strides(), (std::vector<size_t>{12, 4, 1}));
    EXPECT_TRUE(Shape::scalar().strides().empty());
}
// Rust: broadcast
TEST(Shape, broadcast) {
    auto r = Shape({1, 3}).broadcast_with(Shape({2, 3}));
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->dims, (std::vector<size_t>{2, 3}));
    auto r2 = Shape({3}).broadcast_with(Shape({2, 1, 3}));
    ASSERT_TRUE(r2.has_value());
    EXPECT_EQ(r2->dims, (std::vector<size_t>{2, 1, 3}));
}
// Rust: broadcast_fail
TEST(Shape, broadcast_fail) {
    auto r = Shape({2, 3}).broadcast_with(Shape({2, 4}));
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().to_string(), "Incompatible shapes for broadcasting: [2, 3] and [2, 4]");
}
// Rust: reshape
TEST(Shape, reshape) {
    auto r = Shape({2, 3}).reshape({6});
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->dims, (std::vector<size_t>{6}));
    EXPECT_EQ(Shape({2, 3}).reshape({5}).error().to_string(),
              "Shape mismatch: expected [2, 3], got [5]");
}
// Rust: flat_index
TEST(Shape, flat_index) {
    const size_t idx[] = {1, 2, 3};
    auto r = Shape({2, 3, 4}).flat_index(idx);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(*r, 23u);
    const size_t bad[] = {1, 3, 0};
    EXPECT_EQ(Shape({2, 3, 4}).flat_index(bad).error().to_string(),
              "Internal error: Index 3 out of bounds for dim 1 (size 3)");
    const size_t short_idx[] = {1};
    EXPECT_EQ(Shape({2, 3}).flat_index(short_idx).error().to_string(),
              "Rank mismatch: expected 2, got 1");
}
TEST(Shape, validate_expand_squeeze_display) {
    EXPECT_TRUE(Shape({2, 3}).validate().has_value());
    EXPECT_EQ(Shape({2, 0}).validate().error().to_string(),
              "Graph validation failed: Shape has zero dimension at axis 1");
    EXPECT_EQ(Shape({2, 3}).expand_dims(1)->dims, (std::vector<size_t>{2, 1, 3}));
    EXPECT_EQ(Shape({2, 3}).expand_dims(3).error().to_string(),
              "Internal error: expand_dims: axis 3 out of range for rank 2");
    EXPECT_EQ(Shape({1, 2, 1, 3}).squeeze().dims, (std::vector<size_t>{2, 3}));
    EXPECT_EQ(Shape({2, 3}).to_string(), "[2, 3]");
    EXPECT_EQ(Shape::scalar().to_string(), "[]");
}
