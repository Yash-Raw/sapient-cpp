// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-backends/cpu/src/kernels/conv2d.rs — 2-D convolution as im2col (parallel
// per row) + an out-channel-blocked sgemm (max-error gated vs Rust, spec §2.3).

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

#include "sapient/core/error.hpp"
#include "sapient/core/tensor.hpp"

namespace sapient::backends_cpu::kernels::conv2d {

using sapient::core::Result;
using sapient::core::Tensor;

/// Per-op profiling counters (accumulated nanoseconds of the im2col build vs the GEMM); the
/// Kokoro decoder profile drains these. Relaxed atomics, like Rust's `AtomicU64`.
extern std::atomic<uint64_t> IM2COL_NS;
extern std::atomic<uint64_t> GEMM_NS;

/// (N, C_in, H, W) × weight (C_out, C_in/groups, kh, kw) → (N, C_out, H_out, W_out).
/// `kernel_shape` is accepted but unused (Rust `_kernel_shape`: the kernel dims come from `weight`).
/// `pads` = [top, left, bottom, right].
Result<Tensor> conv2d(const Tensor& x,
                      const Tensor& weight,
                      const Tensor* bias,
                      std::array<size_t, 2> kernel_shape,
                      std::array<size_t, 4> pads,
                      std::array<size_t, 2> strides,
                      std::array<size_t, 2> dilations,
                      size_t groups);

} // namespace sapient::backends_cpu::kernels::conv2d
