// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// The `matrixmultiply::sgemm` stand-in (spec §2.3; approved deviation: MAX-ERROR gated against the
// Rust build, never bit-identical — matrixmultiply's blocking is not reproducible without porting
// it, and performance is not a 1a goal). What IS required: C[i][j] must not depend on m, n, or
// which rows/columns a call covers, because matmul_nt_float and conv2d split rows into blocks
// sized by the thread count and the model's numbers must not vary with RAYON_NUM_THREADS.

#include <cstddef>

namespace sapient::backends_cpu {

/// C = alpha·A·B + beta·C for A (m×k), B (k×n), C (m×n) addressed as a[i*rsa + p*csa],
/// b[p*rsb + j*csb], c[i*rsc + j*csc]. `beta == 0` ⇒ C is write-only (may hold NaN on entry).
/// `alpha == 0` or `k == 0` ⇒ C = beta·C.
void sgemm(size_t m,
           size_t k,
           size_t n,
           float alpha,
           const float* a,
           std::ptrdiff_t rsa,
           std::ptrdiff_t csa,
           const float* b,
           std::ptrdiff_t rsb,
           std::ptrdiff_t csb,
           float beta,
           float* c,
           std::ptrdiff_t rsc,
           std::ptrdiff_t csc);

} // namespace sapient::backends_cpu
