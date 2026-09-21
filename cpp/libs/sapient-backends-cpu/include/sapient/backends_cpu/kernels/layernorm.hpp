// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-backends/cpu/src/kernels/layernorm.rs — LayerNorm over the axes from
// `axis` to the end, RMSNorm over the last axis. Sequential f32 sums, single-threaded.

#include <cstdint>

#include "sapient/core/error.hpp"
#include "sapient/core/tensor.hpp"

namespace sapient::backends_cpu::kernels::layernorm {

using sapient::core::Result;
using sapient::core::Tensor;

/// y = (x - mean) / sqrt(var + eps) * weight + bias; `axis` is the first normalised axis
/// (typically -1). `weight`/`bias` may be null.
Result<Tensor>
layer_norm(const Tensor& x, const Tensor* weight, const Tensor* bias, int64_t axis, float epsilon);

/// y = x / sqrt(mean(x²) + eps) * weight over the last axis. `weight` may be null (= 1).
Result<Tensor> rms_norm(const Tensor& x, const Tensor* weight, float epsilon);

} // namespace sapient::backends_cpu::kernels::layernorm
