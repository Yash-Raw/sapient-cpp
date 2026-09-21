// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-backends/cpu/src/kernels/rope.rs — NEOX rotate-half RoPE on
// [batch, n_heads, seq, head_dim]. Scalar; `powf`/`sinf`/`cosf` from the platform libm, exactly
// the calls rustc's LLVM intrinsics lower to (the biggest libm-portability risk in the crate).

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include "sapient/core/error.hpp"
#include "sapient/core/tensor.hpp"

namespace sapient::backends_cpu::kernels::rope {

using sapient::core::Result;
using sapient::core::Tensor;

/// θ_i = pos / base^(2i/head_dim); [x0, x1] → [x0·cos − x1·sin, x1·cos + x0·sin] over the two halves.
Result<Tensor> apply_rope(const Tensor& x, std::span<const size_t> positions, float base);
/// Rotates only the first `rotary_dim` channels of each head (Phi partial RoPE).
Result<Tensor> apply_rope_partial(const Tensor& x,
                                  std::span<const size_t> positions,
                                  float base,
                                  size_t rotary_dim);
/// `apply_rope_partial` with linear position scaling (effective position = pos / pos_scale).
Result<Tensor> apply_rope_partial_scaled(const Tensor& x,
                                         std::span<const size_t> positions,
                                         float base,
                                         size_t rotary_dim,
                                         float pos_scale);
/// (cos, sin) tables of shape [max_seq_len, head_dim/2].
std::pair<std::vector<float>, std::vector<float>>
rope_cos_sin_cache(size_t max_seq_len, size_t head_dim, float base);

} // namespace sapient::backends_cpu::kernels::rope
