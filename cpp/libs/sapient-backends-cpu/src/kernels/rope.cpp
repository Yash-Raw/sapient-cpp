// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/backends_cpu/kernels/rope.hpp"

#include <cmath>

#include "sapient/core/panic.hpp"

namespace sapient::backends_cpu::kernels::rope {

using sapient::core::Error;
using sapient::core::Shape;

Result<Tensor> apply_rope(const Tensor& x, std::span<const size_t> positions, float base) {
    const auto& dims = x.shape().dims;
    if (dims.size() != 4) return tl::unexpected(Error::rank_mismatch(4, dims.size()));
    const size_t batch = dims[0], n_heads = dims[1], seq_len = dims[2], head_dim = dims[3];
    if (head_dim % 2 != 0) return tl::unexpected(Error::internal("RoPE requires even head_dim"));
    if (positions.size() != seq_len)
        return tl::unexpected(Error::internal("positions length must match seq_len"));

    const size_t half = head_dim / 2;
    const auto cow = x.to_f32_cow();
    const auto x_data = cow.get();
    std::vector<float> out(x_data.begin(), x_data.end()); // x_data.to_vec() (unbounded, rule 7)
    if (x_data.size() < batch * n_heads * seq_len * head_dim)
        sapient::core::panic("apply_rope: data shorter than shape");

    for (size_t b = 0; b < batch; ++b)
        for (size_t h = 0; h < n_heads; ++h)
            for (size_t s = 0; s < seq_len; ++s) {
                const size_t pos = positions[s];
                const size_t base_idx = ((b * n_heads + h) * seq_len + s) * head_dim;
                for (size_t i = 0; i < half; ++i) {
                    // θ_i = pos / base^(2i / head_dim)
                    const float freq =
                        static_cast<float>(pos) /
                        ::powf(base, 2.0f * static_cast<float>(i) / static_cast<float>(head_dim));
                    const float sin_f = ::sinf(freq);
                    const float cos_f = ::cosf(freq);
                    const float x0 = x_data[base_idx + i];
                    const float x1 = x_data[base_idx + i + half];
                    out[base_idx + i] = x0 * cos_f - x1 * sin_f;
                    out[base_idx + i + half] = x1 * cos_f + x0 * sin_f;
                }
            }
    return Tensor::from_f32_vec(std::move(out), Shape{batch, n_heads, seq_len, head_dim});
}

Result<Tensor> apply_rope_partial(const Tensor& x,
                                  std::span<const size_t> positions,
                                  float base,
                                  size_t rotary_dim) {
    return apply_rope_partial_scaled(x, positions, base, rotary_dim, 1.0f);
}

Result<Tensor> apply_rope_partial_scaled(const Tensor& x,
                                         std::span<const size_t> positions,
                                         float base,
                                         size_t rotary_dim,
                                         float pos_scale) {
    const auto& dims = x.shape().dims;
    if (dims.size() != 4) return tl::unexpected(Error::rank_mismatch(4, dims.size()));
    const size_t batch = dims[0], n_heads = dims[1], seq_len = dims[2], head_dim = dims[3];
    if (rotary_dim == 0 || rotary_dim > head_dim)
        return tl::unexpected(Error::internal("rotary_dim must be in 1..=head_dim"));
    if (rotary_dim % 2 != 0)
        return tl::unexpected(Error::internal("RoPE requires even rotary_dim"));
    if (positions.size() != seq_len)
        return tl::unexpected(Error::internal("positions length must match seq_len"));

    // The rotary half-split is over rotary_dim; channels [rotary_dim, head_dim) pass through.
    const size_t half = rotary_dim / 2;
    const auto cow = x.to_f32_cow();
    const auto x_data = cow.get();
    std::vector<float> out(x_data.begin(), x_data.end());
    if (x_data.size() < batch * n_heads * seq_len * head_dim)
        sapient::core::panic("apply_rope_partial_scaled: data shorter than shape");

    for (size_t b = 0; b < batch; ++b)
        for (size_t h = 0; h < n_heads; ++h)
            for (size_t s = 0; s < seq_len; ++s) {
                const size_t pos = positions[s];
                const size_t base_idx = ((b * n_heads + h) * seq_len + s) * head_dim;
                for (size_t i = 0; i < half; ++i) {
                    const float freq =
                        (static_cast<float>(pos) / pos_scale) /
                        ::powf(base, 2.0f * static_cast<float>(i) / static_cast<float>(rotary_dim));
                    const float sin_f = ::sinf(freq);
                    const float cos_f = ::cosf(freq);
                    const float x0 = x_data[base_idx + i];
                    const float x1 = x_data[base_idx + i + half];
                    out[base_idx + i] = x0 * cos_f - x1 * sin_f;
                    out[base_idx + i + half] = x1 * cos_f + x0 * sin_f;
                }
            }
    return Tensor::from_f32_vec(std::move(out), Shape{batch, n_heads, seq_len, head_dim});
}

std::pair<std::vector<float>, std::vector<float>>
rope_cos_sin_cache(size_t max_seq_len, size_t head_dim, float base) {
    const size_t half = head_dim / 2;
    std::vector<float> cos_table(max_seq_len * half, 0.0f);
    std::vector<float> sin_table(max_seq_len * half, 0.0f);
    for (size_t pos = 0; pos < max_seq_len; ++pos)
        for (size_t i = 0; i < half; ++i) {
            const float freq =
                static_cast<float>(pos) /
                ::powf(base, 2.0f * static_cast<float>(i) / static_cast<float>(head_dim));
            cos_table[pos * half + i] = ::cosf(freq);
            sin_table[pos * half + i] = ::sinf(freq);
        }
    return {std::move(cos_table), std::move(sin_table)};
}

} // namespace sapient::backends_cpu::kernels::rope
