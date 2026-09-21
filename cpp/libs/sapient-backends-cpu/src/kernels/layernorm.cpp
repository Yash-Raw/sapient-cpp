// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/backends_cpu/kernels/layernorm.hpp"

#include <cmath>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "sapient/core/panic.hpp"

namespace sapient::backends_cpu::kernels::layernorm {

using sapient::core::F32Cow;
using sapient::core::Shape;

Result<Tensor>
layer_norm(const Tensor& x, const Tensor* weight, const Tensor* bias, int64_t axis, float epsilon) {
    const Shape& shape = x.shape();
    const size_t ndim = shape.ndim();
    const size_t ax = axis < 0 ? static_cast<size_t>(static_cast<int64_t>(ndim) + axis)
                               : static_cast<size_t>(axis);
    if (ax > ndim)
        sapient::core::panic("layer_norm: axis out of range"); // Rust: dims()[..ax] panic

    size_t outer = 1;
    for (size_t i = 0; i < ax; ++i)
        outer *= shape.dims[i];
    size_t norm_size = 1;
    for (size_t i = ax; i < ndim; ++i)
        norm_size *= shape.dims[i];

    const auto cow = x.to_f32_cow();
    const auto data = cow.get();
    if (outer * norm_size > data.size())
        sapient::core::panic("layer_norm: data shorter than shape");
    std::vector<float> out(data.size(), 0.0f);

    std::optional<F32Cow> w_cow;
    std::span<const float> w;
    if (weight != nullptr) {
        w_cow = weight->to_f32_cow();
        w = w_cow->get();
        if (w.size() < norm_size)
            sapient::core::panic("layer_norm: weight shorter than the normalised size");
    }
    std::optional<F32Cow> b_cow;
    std::span<const float> b;
    if (bias != nullptr) {
        b_cow = bias->to_f32_cow();
        b = b_cow->get();
        if (b.size() < norm_size)
            sapient::core::panic("layer_norm: bias shorter than the normalised size");
    }

    for (size_t o = 0; o < outer; ++o) {
        const size_t base = o * norm_size;
        const float* slice = data.data() + base;

        float sum = -0.0f; // iter().sum::<f32>() seeds at -0.0
        for (size_t i = 0; i < norm_size; ++i)
            sum += slice[i];
        const float mean = sum / static_cast<float>(norm_size);

        float var_sum = -0.0f;
        for (size_t i = 0; i < norm_size; ++i)
            var_sum += (slice[i] - mean) * (slice[i] - mean);
        const float var = var_sum / static_cast<float>(norm_size);

        const float inv_std = 1.0f / ::sqrtf(var + epsilon);

        for (size_t i = 0; i < norm_size; ++i) {
            const float normed = (slice[i] - mean) * inv_std;
            float y = normed;
            if (weight != nullptr && bias != nullptr)
                y = normed * w[i] + b[i];
            else if (weight != nullptr)
                y = normed * w[i];
            else if (bias != nullptr)
                y = normed + b[i];
            out[base + i] = y;
        }
    }
    return Tensor::from_f32_vec(std::move(out), shape);
}

Result<Tensor> rms_norm(const Tensor& x, const Tensor* weight, float epsilon) {
    const Shape& shape = x.shape();
    const size_t ndim = shape.ndim();

    size_t outer = 1;
    for (size_t i = 0; i + 1 < ndim; ++i)
        outer *= shape.dims[i]; // dims[..ndim.saturating_sub(1)]
    const size_t dim = ndim > 0 ? shape.dims[ndim - 1] : 1;

    const auto cow = x.to_f32_cow();
    const auto data = cow.get();
    if (outer * dim > data.size()) sapient::core::panic("rms_norm: data shorter than shape");
    std::vector<float> out(data.size(), 0.0f);

    std::optional<F32Cow> w_cow;
    std::span<const float> w;
    if (weight != nullptr) {
        w_cow = weight->to_f32_cow();
        w = w_cow->get();
        if (w.size() < dim) sapient::core::panic("rms_norm: weight shorter than the last dim");
    }

    for (size_t o = 0; o < outer; ++o) {
        const size_t base = o * dim;
        const float* slice = data.data() + base;

        float sq = -0.0f; // iter().map(v*v).sum::<f32>() seeds at -0.0
        for (size_t i = 0; i < dim; ++i)
            sq += slice[i] * slice[i];
        const float rms_sq = sq / static_cast<float>(dim);
        const float inv_rms = 1.0f / ::sqrtf(rms_sq + epsilon);

        for (size_t i = 0; i < dim; ++i) {
            const float wv = weight != nullptr ? w[i] : 1.0f; // w.map_or(1.0, |ww| ww[i])
            out[base + i] = slice[i] * inv_rms * wv;
        }
    }
    return Tensor::from_f32_vec(std::move(out), shape);
}

} // namespace sapient::backends_cpu::kernels::layernorm
