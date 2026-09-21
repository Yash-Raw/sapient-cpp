// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-backends/cpu/src/kernels/reduce.rs — sum/mean/max/min over one or more
// axes (empty `axes` = all), element-wise scatter-accumulate, single-threaded.

#include <cstdint>
#include <span>

#include "sapient/core/error.hpp"
#include "sapient/core/tensor.hpp"

namespace sapient::backends_cpu::kernels::reduce {

using sapient::core::Result;
using sapient::core::Tensor;

Result<Tensor> reduce_sum(const Tensor& x, std::span<const int64_t> axes, bool keep_dims);
Result<Tensor> reduce_mean(const Tensor& x, std::span<const int64_t> axes, bool keep_dims);
Result<Tensor> reduce_max(const Tensor& x, std::span<const int64_t> axes, bool keep_dims);
Result<Tensor> reduce_min(const Tensor& x, std::span<const int64_t> axes, bool keep_dims);

} // namespace sapient::backends_cpu::kernels::reduce
