// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/core/shape.hpp"

#include <algorithm>

namespace sapient::core {

size_t Shape::numel() const {
    size_t n = 1;
    for (const size_t d : dims) n *= d;
    return n;
}

std::vector<size_t> Shape::strides() const {
    const size_t n = dims.size();
    std::vector<size_t> s(n);
    if (n == 0) return s;
    s[n - 1] = 1;
    for (size_t i = n - 1; i-- > 0;) s[i] = s[i + 1] * dims[i + 1];
    return s;
}

Result<Shape> Shape::reshape(std::vector<size_t> new_dims) const {
    Shape ns(std::move(new_dims));
    if (ns.numel() != numel()) return tl::unexpected(Error::shape_mismatch(dims, ns.dims));
    return ns;
}

Result<Shape> Shape::broadcast_with(const Shape& other) const {
    const size_t len = std::max(dims.size(), other.dims.size());
    std::vector<size_t> out(len);
    for (size_t i = 0; i < len; ++i) {
        const size_t ai = i < len - dims.size() ? 1 : dims[i - (len - dims.size())];
        const size_t bi = i < len - other.dims.size() ? 1 : other.dims[i - (len - other.dims.size())];
        if (ai == bi) out[i] = ai;
        else if (ai == 1) out[i] = bi;
        else if (bi == 1) out[i] = ai;
        else return tl::unexpected(Error::broadcast(dims, other.dims));
    }
    return Shape(std::move(out));
}

Result<Shape> Shape::expand_dims(size_t axis) const {
    if (axis > dims.size())
        return tl::unexpected(Error::internal("expand_dims: axis " + std::to_string(axis) +
                                              " out of range for rank " + std::to_string(dims.size())));
    std::vector<size_t> out = dims;
    out.insert(out.begin() + static_cast<std::ptrdiff_t>(axis), 1);
    return Shape(std::move(out));
}

Shape Shape::squeeze() const {
    std::vector<size_t> out;
    for (const size_t d : dims) if (d != 1) out.push_back(d);
    return Shape(std::move(out));
}

Result<void> Shape::validate() const {
    for (size_t i = 0; i < dims.size(); ++i)
        if (dims[i] == 0)
            return tl::unexpected(Error::invalid_graph("Shape has zero dimension at axis " + std::to_string(i)));
    return {};
}

Result<size_t> Shape::flat_index(std::span<const size_t> idx) const {
    if (idx.size() != dims.size()) return tl::unexpected(Error::rank_mismatch(dims.size(), idx.size()));
    const auto st = strides();
    size_t off = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (idx[i] >= dims[i])
            return tl::unexpected(Error::internal("Index " + std::to_string(idx[i]) + " out of bounds for dim " +
                                                  std::to_string(i) + " (size " + std::to_string(dims[i]) + ")"));
        off += idx[i] * st[i];
    }
    return off;
}

std::string Shape::to_string() const { return debug_dims(dims); }

}  // namespace sapient::core
