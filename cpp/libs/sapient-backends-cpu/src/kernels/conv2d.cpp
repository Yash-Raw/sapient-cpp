// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/backends_cpu/kernels/conv2d.hpp"

#include <algorithm>
#include <chrono>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "sapient/backends_cpu/parallel.hpp"
#include "sapient/backends_cpu/sgemm.hpp"
#include "sapient/core/panic.hpp"

namespace sapient::backends_cpu::kernels::conv2d {

using sapient::core::Error;
using sapient::core::F32Cow;
using sapient::core::Shape;

std::atomic<uint64_t> IM2COL_NS{0};
std::atomic<uint64_t> GEMM_NS{0};

namespace {
uint64_t elapsed_ns(std::chrono::steady_clock::time_point t0) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0)
            .count());
}
} // namespace

Result<Tensor> conv2d(const Tensor& x,
                      const Tensor& weight,
                      const Tensor* bias,
                      [[maybe_unused]] std::array<size_t, 2> kernel_shape,
                      std::array<size_t, 4> pads,
                      std::array<size_t, 2> strides,
                      std::array<size_t, 2> dilations,
                      size_t groups) {
    const Shape& xs = x.shape();
    const Shape& ws = weight.shape();
    if (xs.ndim() != 4) return tl::unexpected(Error::rank_mismatch(4, xs.ndim()));
    if (ws.ndim() != 4) return tl::unexpected(Error::rank_mismatch(4, ws.ndim()));

    const size_t n = xs.dims[0], c_in = xs.dims[1], h_in = xs.dims[2], w_in = xs.dims[3];
    const size_t c_out = ws.dims[0], c_in_g = ws.dims[1], kh = ws.dims[2], kw = ws.dims[3];

    const size_t g = groups;
    if (c_in != c_in_g * g)
        return tl::unexpected(Error::invalid_graph(
            "conv2d: groups=" + std::to_string(g) + ", c_in=" + std::to_string(c_in) +
            ", c_in/group=" + std::to_string(c_in_g) + ": " + std::to_string(c_in_g) + "*" +
            std::to_string(g) + "!=c_in"));

    // Rust: `(h_in + pads[0] + pads[2] - dilations[0]*(kh-1) - 1) / strides[0] + 1` — usize
    // underflow and division by zero both panic there (debug assertions on under cargo test).
    if (kh == 0 || kw == 0) sapient::core::panic("conv2d: zero kernel size");
    if (strides[0] == 0 || strides[1] == 0) sapient::core::panic("conv2d: zero stride");
    const size_t h_span = h_in + pads[0] + pads[2];
    const size_t w_span = w_in + pads[1] + pads[3];
    const size_t h_need = dilations[0] * (kh - 1) + 1;
    const size_t w_need = dilations[1] * (kw - 1) + 1;
    if (h_span < h_need || w_span < w_need)
        sapient::core::panic("conv2d: kernel larger than the padded input");
    const size_t h_out = (h_span - h_need) / strides[0] + 1;
    const size_t w_out = (w_span - w_need) / strides[1] + 1;
    if (g == 0) sapient::core::panic("conv2d: groups must be non-zero"); // Rust: c_out / g

    const auto x_cow = x.to_f32_cow();
    const auto x_data = x_cow.get();
    const auto w_cow = weight.to_f32_cow();
    const auto w_data = w_cow.get();
    std::optional<F32Cow> b_cow;
    std::span<const float> b_data;
    if (bias != nullptr) {
        b_cow = bias->to_f32_cow();
        b_data = b_cow->get();
    }

    const size_t col_rows = c_in_g * kh * kw;
    const size_t col_cols = h_out * w_out;
    const size_t c_out_g = c_out / g;
    if (x_data.size() < n * c_in * h_in * w_in)
        sapient::core::panic("conv2d: x data shorter than shape");
    if (w_data.size() < c_out * col_rows)
        sapient::core::panic("conv2d: weight data shorter than shape");
    if (bias != nullptr && b_data.size() < c_out)
        sapient::core::panic("conv2d: bias shorter than c_out");

    std::vector<float> out_data(n * c_out * h_out * w_out, 0.0f);
    std::vector<float> col;
    std::vector<float> gemm_out;

    for (size_t batch = 0; batch < n; ++batch) {
        for (size_t group = 0; group < g; ++group) {
            // ── im2col for this (batch, group): one parallel task per row ──
            const auto t_col = std::chrono::steady_clock::now();
            col.assign(col_rows * col_cols, 0.0f);
            const size_t c_start = group * c_in_g;
            parallel::par_chunks_mut(col, col_cols, [&](size_t row, std::span<float> dst) {
                const size_t kj = row % kw;
                const size_t ki = (row / kw) % kh;
                const size_t ci = row / (kh * kw);
                const size_t c = c_start + ci;
                const size_t base = batch * (c_in * h_in * w_in) + c * (h_in * w_in);
                for (size_t oh = 0; oh < h_out; ++oh) {
                    const auto ih = static_cast<std::ptrdiff_t>(oh * strides[0]) +
                                    static_cast<std::ptrdiff_t>(ki * dilations[0]) -
                                    static_cast<std::ptrdiff_t>(pads[0]);
                    float* drow = dst.data() + oh * w_out;
                    if (ih < 0 || ih >= static_cast<std::ptrdiff_t>(h_in)) {
                        std::fill(drow, drow + w_out, 0.0f);
                        continue;
                    }
                    const float* xrow = x_data.data() + base + static_cast<size_t>(ih) * w_in;
                    const auto off = static_cast<std::ptrdiff_t>(kj * dilations[1]) -
                                     static_cast<std::ptrdiff_t>(pads[1]);
                    if (strides[1] == 1) { // contiguous middle, zero edges
                        for (size_t ow = 0; ow < w_out; ++ow) {
                            const std::ptrdiff_t iw = static_cast<std::ptrdiff_t>(ow) + off;
                            drow[ow] =
                                (iw >= 0 && static_cast<size_t>(iw) < w_in) ? xrow[iw] : 0.0f;
                        }
                    } else {
                        for (size_t ow = 0; ow < w_out; ++ow) {
                            const std::ptrdiff_t iw =
                                static_cast<std::ptrdiff_t>(ow * strides[1]) + off;
                            drow[ow] =
                                (iw >= 0 && static_cast<size_t>(iw) < w_in) ? xrow[iw] : 0.0f;
                        }
                    }
                }
            });
            IM2COL_NS.fetch_add(elapsed_ns(t_col), std::memory_order_relaxed);

            // ── GEMM: W_group (c_out_g × col_rows) · col (col_rows × col_cols), split over
            //    out-channel row blocks (each an independent sgemm over the same K) ──
            const auto t_gemm = std::chrono::steady_clock::now();
            const size_t w_off = group * c_out_g * (c_in_g * kh * kw);
            const size_t m = c_out_g;
            const size_t k = col_rows;
            const size_t n2 = col_cols;
            gemm_out.assign(m * n2, 0.0f);
            const size_t flops = m * k * n2;
            const size_t threads = std::max<size_t>(parallel::num_threads(), 1);
            const size_t mblock =
                flops >= (size_t{1} << 20) ? std::max<size_t>((m + threads - 1) / threads, 8) : m;
            parallel::par_chunks_mut(
                gemm_out, mblock * n2, [&](size_t bi, std::span<float> out_block) {
                    const size_t m0 = bi * mblock;
                    const size_t mc = out_block.size() / n2;
                    sgemm(mc,
                          k,
                          n2,
                          1.0f,
                          w_data.data() + w_off + m0 * k,
                          static_cast<std::ptrdiff_t>(k),
                          1,
                          col.data(),
                          static_cast<std::ptrdiff_t>(n2),
                          1,
                          0.0f,
                          out_block.data(),
                          static_cast<std::ptrdiff_t>(n2),
                          1);
                });
            GEMM_NS.fetch_add(elapsed_ns(t_gemm), std::memory_order_relaxed);

            // ── copy-out with bias ──
            const size_t c_out_start = group * c_out_g;
            for (size_t co = 0; co < c_out_g; ++co) {
                const float bias_v = bias != nullptr ? b_data[c_out_start + co] : 0.0f;
                for (size_t hw = 0; hw < col_cols; ++hw) {
                    const size_t out_idx =
                        batch * (c_out * h_out * w_out) + (c_out_start + co) * (h_out * w_out) + hw;
                    out_data[out_idx] = gemm_out[co * n2 + hw] + bias_v;
                }
            }
        }
    }
    return Tensor::from_f32_vec(std::move(out_data), Shape{n, c_out, h_out, w_out});
}

} // namespace sapient::backends_cpu::kernels::conv2d
