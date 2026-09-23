// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-backends/cpu/src/kernels/softmax.rs — max-subtracted softmax /
// log-softmax along one axis, single-threaded.

#include <cstdint>

#include "sapient/core/error.hpp"
#include "sapient/core/tensor.hpp"

namespace sapient::backends_cpu::kernels::softmax {

using sapient::core::Result;
using sapient::core::Tensor;

/// Numerically stable softmax along `axis` (negative counts from the end).
Result<Tensor> softmax(const Tensor& x, int64_t axis);
/// Numerically stable log-softmax along `axis`.
Result<Tensor> log_softmax(const Tensor& x, int64_t axis);

} // namespace sapient::backends_cpu::kernels::softmax
