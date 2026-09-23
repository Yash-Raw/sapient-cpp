// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/backends_cpu/kernels/reduce.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "sapient/core/panic.hpp"

namespace sapient::backends_cpu::kernels::reduce {

using sapient::core::Shape;

namespace {

std::vector<size_t> normalise_axes(std::span<const int64_t> axes, size_t ndim) {
    std::vector<size_t> out;
    if (axes.empty()) {
        for (size_t i = 0; i < ndim; ++i)
            out.push_back(i);
        return out;
    }
    for (const int64_t a : axes)
        out.push_back(a < 0 ? static_cast<size_t>(static_cast<int64_t>(ndim) + a)
                            : static_cast<size_t>(a)); // Rust: `(ndim as i64 + a) as usize`
    return out;
}

/// Rust `fn reduce<F>(x, axes, keep_dims, init, f)`.
template <class F>
Result<Tensor>
reduce_impl(const Tensor& x, std::span<const int64_t> axes, bool keep_dims, float init, F f) {
    const Shape& shape = x.shape();
    const auto cow = x.to_f32_cow();
    const auto data = cow.get(); // unbounded for F32 (rule 7): Rust iterates data.len(), so do we
    const std::vector<size_t> norm_axes = normalise_axes(axes, shape.ndim());
    const auto reduced = [&](size_t i) {
        return std::find(norm_axes.begin(), norm_axes.end(), i) != norm_axes.end();
    };

    std::vector<size_t> out_dims;
    for (size_t i = 0; i < shape.ndim(); ++i) {
        if (reduced(i)) {
            if (keep_dims) out_dims.push_back(1);
        } else {
            out_dims.push_back(shape.dims[i]);
        }
    }
    size_t out_numel = 1;
    for (const size_t d : out_dims)
        out_numel *= d;
    out_numel = std::max<size_t>(out_numel, 1);
    std::vector<float> out_data(out_numel, init);

    // Rust rebuilds `Shape(out_dims.clone()).strides()` inside the per-element loop; it is a pure
    // function of out_dims, so it is hoisted here (behaviour-identical).
    const std::vector<size_t> out_strides = Shape(out_dims).strides();
    std::vector<size_t> multi(shape.ndim(), 0);
    for (size_t flat = 0; flat < data.size(); ++flat) {
        size_t r = flat;
        for (size_t i = shape.ndim(); i-- > 0;) {
            if (shape.dims[i] == 0)
                sapient::core::panic("reduce: zero dimension"); // Rust: `% 0` panic
            multi[i] = r % shape.dims[i];
            r /= shape.dims[i];
        }
        size_t out_flat = 0;
        size_t oi = 0;
        for (size_t i = 0; i < multi.size(); ++i) {
            if (!reduced(i)) {
                out_flat += multi[i] * (oi < out_strides.size() ? out_strides[oi] : 1);
                ++oi;
            } else if (keep_dims) {
                ++oi; // dim = 1, stride may still be 1.
            }
        }
        if (out_flat >= out_data.size()) sapient::core::panic("reduce: output index out of range");
        out_data[out_flat] = f(out_data[out_flat], data[flat]);
    }
    return Tensor::from_f32_vec(std::move(out_data), Shape(out_dims));
}

} // namespace

Result<Tensor> reduce_sum(const Tensor& x, std::span<const int64_t> axes, bool keep_dims) {
    return reduce_impl(x, axes, keep_dims, 0.0f, [](float acc, float v) { return acc + v; });
}

Result<Tensor> reduce_mean(const Tensor& x, std::span<const int64_t> axes, bool keep_dims) {
    auto sum = reduce_sum(x, axes, keep_dims);
    if (!sum.has_value()) return tl::unexpected(std::move(sum.error()));
    const std::vector<size_t> norm_axes = normalise_axes(axes, x.ndim());
    size_t count = 1;
    for (const size_t a : norm_axes) {
        if (a >= x.ndim())
            sapient::core::panic("reduce_mean: axis out of range"); // Rust: dims()[a] panic
        count *= x.shape().dims[a];
    }
    const auto s = sum->f32_slice();
    std::vector<float> d(s.size());
    for (size_t i = 0; i < s.size(); ++i)
        d[i] = s[i] / static_cast<float>(count);
    return Tensor::from_f32_vec(std::move(d), sum->shape());
}

Result<Tensor> reduce_max(const Tensor& x, std::span<const int64_t> axes, bool keep_dims) {
    return reduce_impl(
        x, axes, keep_dims, -std::numeric_limits<float>::infinity(), [](float acc, float v) {
            return ::fmaxf(acc, v);
        }); // f32::max
}

Result<Tensor> reduce_min(const Tensor& x, std::span<const int64_t> axes, bool keep_dims) {
    return reduce_impl(
        x, axes, keep_dims, std::numeric_limits<float>::infinity(), [](float acc, float v) {
            return ::fminf(acc, v);
        }); // f32::min
}

} // namespace sapient::backends_cpu::kernels::reduce
