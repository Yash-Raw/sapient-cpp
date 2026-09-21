// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-backends/cpu/src/kernels/elementwise.rs. All kernels are scalar f32.
// Binary ops zip equal-length operands or broadcast a numel-1 side; unary ops require F32 input.

#include <optional>

#include "sapient/core/error.hpp"
#include "sapient/core/tensor.hpp"

namespace sapient::backends_cpu::kernels::elementwise {

using sapient::core::Result;
using sapient::core::Tensor;

Result<Tensor> add(const Tensor& a, const Tensor& b);
Result<Tensor> sub(const Tensor& a, const Tensor& b);
Result<Tensor> mul(const Tensor& a, const Tensor& b);
Result<Tensor> div(const Tensor& a, const Tensor& b);
Result<Tensor> pow(const Tensor& a, const Tensor& b);

Result<Tensor> neg(const Tensor& x);
Result<Tensor> abs(const Tensor& x);
Result<Tensor> sqrt(const Tensor& x);
Result<Tensor> exp(const Tensor& x);
Result<Tensor> log(const Tensor& x);
Result<Tensor> erf(const Tensor& x);
Result<Tensor> floor(const Tensor& x);
Result<Tensor> ceil(const Tensor& x);
Result<Tensor> round(const Tensor& x);

Result<Tensor> relu(const Tensor& x);
Result<Tensor> sigmoid(const Tensor& x);
Result<Tensor> tanh_act(const Tensor& x);
/// GELU tanh approximation: 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))).
Result<Tensor> gelu(const Tensor& x);
/// Exact (erf-based) GELU 0.5·x·(1 + erf(x/√2)) — the Whisper variant; uses erf_approx, not erff.
Result<Tensor> gelu_erf(const Tensor& x);
Result<Tensor> silu(const Tensor& x);
Result<Tensor> hard_swish(const Tensor& x);
Result<Tensor> leaky_relu(const Tensor& x, float alpha);
Result<Tensor> clip(const Tensor& x, std::optional<float> min, std::optional<float> max);

/// Abramowitz & Stegun 7.1.26 rational approximation (max error ~1.5e-7), verbatim from Rust —
/// NEVER replaced by `::erff` (bit-identity). Rust-private; exposed for the ported `test_erf`.
float erf_approx(float x);

} // namespace sapient::backends_cpu::kernels::elementwise
