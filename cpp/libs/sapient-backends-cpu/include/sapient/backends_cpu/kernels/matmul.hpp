// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-backends/cpu/src/kernels/matmul.rs. Plan C: `matmul`, the `matmul_nt`
// dispatcher with its FLOAT paths (F16 GEMV bit-surgery, f32 GEMV, sgemm prefill row blocks),
// `gemm`, `gemv_chunk`, `for_each_out_chunk`. Plan D adds the seven quantized arms (today they
// return an explicit Error); plan E adds the spin-pool branch of `for_each_out_chunk`.

#include <cstddef>
#include <functional>
#include <span>

#include "sapient/core/error.hpp"
#include "sapient/core/tensor.hpp"

namespace sapient::backends_cpu::kernels::matmul {

using sapient::core::Result;
using sapient::core::Tensor;

/// (…, M, K) × (…, K, N) → (…, M, N); batch = product of `a`'s leading dims. sgemm-backed.
Result<Tensor> matmul(const Tensor& a, const Tensor& b);

/// Linear projection x [M, K] · Wᵀ with W stored [N, K] (PyTorch nn.Linear layout) → [M, N].
/// Dispatches on W's dtype without expanding quantized weights (plan D); float weights take the
/// F16 GEMV (m == 1, k ≥ 64, F16), the f32 GEMV (m == 1, k ≥ 512) or the blocked sgemm path.
Result<Tensor> matmul_nt(const Tensor& x, const Tensor& w);

/// C = alpha · op(A) × op(B) + beta · bias (bias [n] or [1], broadcast over rows).
Result<Tensor> gemm(const Tensor& a,
                    const Tensor& b,
                    const Tensor* bias,
                    float alpha,
                    float beta,
                    bool trans_a,
                    bool trans_b);

namespace detail {
/// Rust `gemv_chunk(n)`: the rows-per-task size for a GEMV over n output rows (porting map §3.1).
/// Thermal-governed → exactly `effective_threads()` tasks, no cap; else `SAPIENT_GEMV_TPC` (read
/// every call) → max(n / (ncpus·tpc), 16); default clamp(n / (ncpus·4), 16, 512).
size_t gemv_chunk(size_t n);
/// Rust `for_each_out_chunk`: `f(ci, out[ci*chunk, min((ci+1)*chunk, len)))` for every chunk.
/// Plan C runs the rayon-twin branch only; plan E adds the spin-pool branch (identical partition).
void for_each_out_chunk(std::span<float> out,
                        size_t chunk,
                        const std::function<void(size_t, std::span<float>)>& f);
} // namespace detail

} // namespace sapient::backends_cpu::kernels::matmul
