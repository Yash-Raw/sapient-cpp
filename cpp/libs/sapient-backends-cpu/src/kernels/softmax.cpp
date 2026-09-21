// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/backends_cpu/kernels/softmax.hpp"

#include <cfloat>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "sapient/core/panic.hpp"

namespace sapient::backends_cpu::kernels::softmax {

using sapient::core::Error;
using sapient::core::Shape;

namespace {

// Rust `(ndim as i64 + axis) as usize`: a still-negative sum wraps to a huge index, which the
// range check in the caller then rejects. The unsigned conversion here wraps identically.
size_t normalise_axis(int64_t axis, size_t ndim) {
    return axis < 0 ? static_cast<size_t>(static_cast<int64_t>(ndim) + axis)
                    : static_cast<size_t>(axis);
}

Result<Tensor> apply_softmax_impl(const Tensor& x, int64_t axis, bool log_mode) {
    const Shape& shape = x.shape();
    const size_t ndim = shape.ndim();
    const size_t ax = normalise_axis(axis, ndim);
    if (ax >= ndim)
        return tl::unexpected(Error::internal("softmax axis " + std::to_string(axis) +
                                              " out of range for rank " + std::to_string(ndim)));

    const auto cow = x.to_f32_cow();
    const auto data = cow.get();
    if (data.size() < shape.numel()) sapient::core::panic("softmax: data shorter than shape");
    std::vector<float> out(data.size(),
                           0.0f); // Rust: vec![0.0; data.len()] (unbounded view, rule 7)

    size_t outer = 1;
    for (size_t i = 0; i < ax; ++i)
        outer *= shape.dims[i];
    const size_t dim_size = shape.dims[ax];
    size_t inner = 1;
    for (size_t i = ax + 1; i < ndim; ++i)
        inner *= shape.dims[i];

    std::vector<float> slice(dim_size);
    std::vector<float> exps(dim_size);
    for (size_t o = 0; o < outer; ++o) {
        for (size_t i = 0; i < inner; ++i) {
            for (size_t d = 0; d < dim_size; ++d)
                slice[d] = data[(o * dim_size + d) * inner + i];
            // fold(NEG_INFINITY, f32::max) — fmaxf drops a NaN operand exactly like f32::max.
            float max_v = -std::numeric_limits<float>::infinity();
            for (size_t d = 0; d < dim_size; ++d)
                max_v = ::fmaxf(max_v, slice[d]);
            if (max_v == -std::numeric_limits<float>::infinity()) max_v = 0.0f;
            for (size_t d = 0; d < dim_size; ++d)
                exps[d] = ::expf(slice[d] - max_v);
            float sum_e = -0.0f; // iter().sum::<f32>() seeds at -0.0
            for (size_t d = 0; d < dim_size; ++d)
                sum_e += exps[d];
            if (sum_e == 0.0f) sum_e = FLT_EPSILON;
            for (size_t d = 0; d < dim_size; ++d) {
                const size_t idx = (o * dim_size + d) * inner + i;
                out[idx] = log_mode ? (slice[d] - max_v) - ::logf(sum_e) : exps[d] / sum_e;
            }
        }
    }
    return Tensor::from_f32_vec(std::move(out), shape);
}

} // namespace

Result<Tensor> softmax(const Tensor& x, int64_t axis) {
    return apply_softmax_impl(x, axis, false);
}
Result<Tensor> log_softmax(const Tensor& x, int64_t axis) {
    return apply_softmax_impl(x, axis, true);
}

} // namespace sapient::backends_cpu::kernels::softmax
