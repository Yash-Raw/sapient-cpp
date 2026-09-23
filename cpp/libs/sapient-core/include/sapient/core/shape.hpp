// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-core/src/shape.rs. `dims` is public like Rust's tuple field `.0`.

#include <cstddef>
#include <initializer_list>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "sapient/core/error.hpp"

namespace sapient::core {

struct Shape {
    std::vector<size_t> dims;

    Shape() = default;
    explicit Shape(std::vector<size_t> d) : dims(std::move(d)) {}
    Shape(std::initializer_list<size_t> d) : dims(d) {}
    static Shape scalar() { return Shape(); }

    size_t ndim() const { return dims.size(); }
    /// Product of dims; the scalar shape (no dims) has numel 1.
    size_t numel() const;
    /// Row-major element strides; empty for the scalar shape.
    std::vector<size_t> strides() const;
    bool is_scalar() const { return dims.empty(); }
    Result<Shape> reshape(std::vector<size_t> new_dims) const;
    /// NumPy right-aligned broadcasting.
    Result<Shape> broadcast_with(const Shape& other) const;
    Result<Shape> expand_dims(size_t axis) const;
    Shape squeeze() const; // drops every dim equal to 1
    /// Rejects zero dims (error text uses the InvalidGraph variant, as Rust does).
    Result<void> validate() const;
    /// Element (not byte) offset of a multi-index.
    Result<size_t> flat_index(std::span<const size_t> idx) const;
    std::string to_string() const; // "[2, 3]"
    bool operator==(const Shape&) const = default;
};

} // namespace sapient::core
