// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-backends/cpu/src/kernels/attention.rs — Flash-Edge online-softmax
// attention (never materialises the seq_q × seq_k score matrix), causal masking via -inf,
// grouped-query attention by KV-head repeat, NEON dot/saxpby on aarch64.

#include <cstddef>
#include <optional>

#include "sapient/core/error.hpp"
#include "sapient/core/tensor.hpp"

namespace sapient::backends_cpu::kernels::attention {

using sapient::core::Result;
using sapient::core::Tensor;

/// q: [batch, n_heads, seq_q, head_dim]; k, v: [batch, n_kv_heads, seq_k, head_dim];
/// mask: optional additive [seq_q, seq_k] (−inf = masked). `mask == nullptr` means BUILT-IN CAUSAL
/// masking with the KV-cache offset (seq_k − seq_q) — pass an explicit all-zeros mask for
/// non-causal attention (the Whisper/SigLIP trap). `scale` defaults to 1/sqrt(head_dim).
/// Output: [batch, n_heads, seq_q, head_dim]. Parallel over (batch, head).
Result<Tensor> scaled_dot_product_attention(const Tensor& q,
                                            const Tensor& k,
                                            const Tensor& v,
                                            const Tensor* mask,
                                            std::optional<float> scale,
                                            size_t n_kv_heads);

/// Additive causal mask [seq_q, seq_k]: 0 where ki <= qi + (seq_k − seq_q), −inf beyond.
Tensor causal_mask(size_t seq_q, size_t seq_k);

} // namespace sapient::backends_cpu::kernels::attention
