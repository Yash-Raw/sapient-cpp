// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/backends_cpu/kernels/elementwise.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "sapient/core/dtype.hpp"

// Inside this namespace the module's own `exp`, `log`, `abs`, `sqrt`, `floor`, `ceil`, `round`,
// `pow` (const Tensor&) HIDE the unqualified libm names — every libm call below is spelled with
// its global float name (`::expf`, …). That is also the bit-identity rule (float, never double).
namespace sapient::backends_cpu::kernels::elementwise {

using sapient::core::DType;
using sapient::core::Error;
using sapient::core::Shape;

namespace {

// Rust `unary_f32`: F32 only (TypeMismatch otherwise); reads the UNBOUNDED f32 view, so a sliced
// F32 view yields data.len() != numel and from_f32 returns ShapeMismatch — as in Rust.
template <class F> Result<Tensor> unary_f32(const Tensor& x, F f) {
    if (x.dtype() != DType::F32)
        return tl::unexpected(Error::type_mismatch("f32", sapient::core::to_string(x.dtype())));
    const auto cow = x.to_f32_cow();
    const auto src = cow.get();
    std::vector<float> data(src.size());
    for (size_t i = 0; i < src.size(); ++i)
        data[i] = f(src[i]);
    return Tensor::from_f32_vec(std::move(data), x.shape());
}

// Rust `binary_f32`: same length → zip; else numel-1 side broadcasts; else ShapeMismatch.
template <class F> Result<Tensor> binary_f32(const Tensor& a, const Tensor& b, F f) {
    const auto a_cow = a.to_f32_cow();
    const auto a_data = a_cow.get();
    const auto b_cow = b.to_f32_cow();
    const auto b_data = b_cow.get();
    std::vector<float> out;
    Shape shape;
    if (a_data.size() == b_data.size()) {
        out.resize(a_data.size());
        for (size_t i = 0; i < out.size(); ++i)
            out[i] = f(a_data[i], b_data[i]);
        shape = a.shape();
    } else if (b_data.size() == 1) {
        const float scalar = b_data[0];
        out.resize(a_data.size());
        for (size_t i = 0; i < out.size(); ++i)
            out[i] = f(a_data[i], scalar);
        shape = a.shape();
    } else if (a_data.size() == 1) {
        const float scalar = a_data[0];
        out.resize(b_data.size());
        for (size_t i = 0; i < out.size(); ++i)
            out[i] = f(scalar, b_data[i]);
        shape = b.shape();
    } else {
        return tl::unexpected(Error::shape_mismatch(a.shape().dims, b.shape().dims));
    }
    return Tensor::from_f32_vec(std::move(out), std::move(shape));
}

} // namespace

// ── Arithmetic ────────────────────────────────────────────────────────────────
Result<Tensor> add(const Tensor& a, const Tensor& b) {
    return binary_f32(a, b, [](float x, float y) { return x + y; });
}
Result<Tensor> sub(const Tensor& a, const Tensor& b) {
    return binary_f32(a, b, [](float x, float y) { return x - y; });
}
Result<Tensor> mul(const Tensor& a, const Tensor& b) {
    return binary_f32(a, b, [](float x, float y) { return x * y; });
}
Result<Tensor> div(const Tensor& a, const Tensor& b) {
    return binary_f32(a, b, [](float x, float y) { return x / y; });
}
Result<Tensor> pow(const Tensor& a, const Tensor& b) {
    return binary_f32(a, b, [](float x, float y) { return ::powf(x, y); });
}

Result<Tensor> neg(const Tensor& x) {
    return unary_f32(x, [](float v) { return -v; });
}
Result<Tensor> abs(const Tensor& x) {
    return unary_f32(x, [](float v) { return ::fabsf(v); });
}
Result<Tensor> sqrt(const Tensor& x) {
    return unary_f32(x, [](float v) { return ::sqrtf(v); });
}
Result<Tensor> exp(const Tensor& x) {
    return unary_f32(x, [](float v) { return ::expf(v); });
}
Result<Tensor> log(const Tensor& x) {
    return unary_f32(x, [](float v) { return ::logf(v); }); // Rust `ln`
}
Result<Tensor> erf(const Tensor& x) {
    return unary_f32(x, erf_approx);
}
Result<Tensor> floor(const Tensor& x) {
    return unary_f32(x, [](float v) { return ::floorf(v); });
}
Result<Tensor> ceil(const Tensor& x) {
    return unary_f32(x, [](float v) { return ::ceilf(v); });
}
Result<Tensor> round(const Tensor& x) {
    return unary_f32(x,
                     [](float v) { return ::roundf(v); }); // half away from zero, like f32::round
}

// ── Activations ───────────────────────────────────────────────────────────────
Result<Tensor> relu(const Tensor& x) {
    return unary_f32(x, [](float v) { return ::fmaxf(v, 0.0f); }); // f32::max
}
Result<Tensor> sigmoid(const Tensor& x) {
    return unary_f32(x, [](float v) { return 1.0f / (1.0f + ::expf(-v)); });
}
Result<Tensor> tanh_act(const Tensor& x) {
    return unary_f32(x, [](float v) { return ::tanhf(v); });
}
Result<Tensor> gelu(const Tensor& x) {
    constexpr float SQRT_2_OVER_PI = 0.79788456f; // Rust 0.797_884_56
    constexpr float COEF = 0.044715f;             // Rust 0.044_715
    return unary_f32(x, [](float v) {
        const float inner = SQRT_2_OVER_PI * (v + COEF * v * v * v);
        return 0.5f * v * (1.0f + ::tanhf(inner));
    });
}
Result<Tensor> gelu_erf(const Tensor& x) {
    constexpr float INV_SQRT_2 =
        0.70710678118654752440f; // std::f32::consts::FRAC_1_SQRT_2 (0x3F3504F3)
    return unary_f32(x, [](float v) { return 0.5f * v * (1.0f + erf_approx(v * INV_SQRT_2)); });
}
Result<Tensor> silu(const Tensor& x) {
    return unary_f32(x, [](float v) { return v / (1.0f + ::expf(-v)); });
}
Result<Tensor> hard_swish(const Tensor& x) {
    // Rust: v * (v + 3.0).clamp(0.0, 6.0) / 6.0 — std::clamp has the same NaN pass-through.
    return unary_f32(x, [](float v) { return v * std::clamp(v + 3.0f, 0.0f, 6.0f) / 6.0f; });
}
Result<Tensor> leaky_relu(const Tensor& x, float alpha) {
    return unary_f32(x, [alpha](float v) { return v >= 0.0f ? v : alpha * v; });
}
Result<Tensor> clip(const Tensor& x, std::optional<float> min, std::optional<float> max) {
    return unary_f32(x, [min, max](float v) {
        const float lo = min.has_value() ? ::fmaxf(v, *min) : v; // f32::max
        return max.has_value() ? ::fminf(lo, *max) : lo;         // f32::min
    });
}

// ── Erf approximation (Abramowitz & Stegun), coefficients verbatim from Rust ─────────────
float erf_approx(float x) {
    // f32::signum: NaN stays NaN; otherwise ±1 by sign bit (+0.0 → 1, -0.0 → -1).
    const float sign = std::isnan(x) ? x : ::copysignf(1.0f, x);
    x = ::fabsf(x);
    const float t = 1.0f / (1.0f + 0.3275911f * x);
    const float y =
        1.0f - (0.25482959f +
                (-0.28449674f + (1.42141374f + (-1.45315203f + 1.06140543f * t) * t) * t) * t) *
                   t * ::expf(-x * x);
    return sign * y;
}

} // namespace sapient::backends_cpu::kernels::elementwise
