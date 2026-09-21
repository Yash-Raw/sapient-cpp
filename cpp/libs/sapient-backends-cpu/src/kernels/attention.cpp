// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/backends_cpu/kernels/attention.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

#include "sapient/backends_cpu/parallel.hpp"
#include "sapient/core/panic.hpp"

namespace sapient::backends_cpu::kernels::attention {

using sapient::core::Error;
using sapient::core::F32Cow;
using sapient::core::Shape;

namespace {

// ── SIMD helpers (attention.rs:29-85) ────────────────────────────────────────
#if defined(__aarch64__) || defined(_M_ARM64)
// 4-wide vfmaq_f32 into ONE accumulator, vaddvq_f32, then the scalar tail added after the
// horizontal reduction — this exact shape is what the golden dumps pin.
float dot_f32_neon(const float* a, const float* b, size_t n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4)
        acc = vfmaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
    float s = vaddvq_f32(acc);
    for (; i < n; ++i)
        s += a[i] * b[i];
    return s;
}

// o[i] = alpha * o[i] + beta * v[i]: vfmaq_f32(vmulq_f32(vo, va), vv, vb).
void saxpby_neon(float* o, const float* v, size_t n, float alpha, float beta) {
    const float32x4_t va = vdupq_n_f32(alpha);
    const float32x4_t vb = vdupq_n_f32(beta);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t vo = vld1q_f32(o + i);
        const float32x4_t vv = vld1q_f32(v + i);
        vst1q_f32(o + i, vfmaq_f32(vmulq_f32(vo, va), vv, vb));
    }
    for (; i < n; ++i)
        o[i] = alpha * o[i] + beta * v[i];
}
#else
// Scalar fallbacks (attention.rs:50, :78). The dot is `iter().zip().map().sum()` — seed -0.0.
float dot_f32_neon(const float* a, const float* b, size_t n) {
    float s = -0.0f;
    for (size_t i = 0; i < n; ++i)
        s += a[i] * b[i];
    return s;
}
void saxpby_neon(float* o, const float* v, size_t n, float alpha, float beta) {
    for (size_t i = 0; i < n; ++i)
        o[i] = alpha * o[i] + beta * v[i];
}
#endif

// ── Flash-Edge online-softmax kernel for one query row (attention.rs:98) ──────
void flash_attn_row(const float* q_row,  // [head_dim]
                    const float* k_head, // [seq_k * head_dim] contiguous
                    const float* v_head, // [seq_k * head_dim] contiguous
                    float* o_row,        // [head_dim], written in place
                    float scale,
                    size_t head_dim,
                    size_t attend_len,    // k/v positions to visit (causal: qi + offset + 1)
                    const float* mask_row // optional additive mask row, length seq_k
) {
    float m = -std::numeric_limits<float>::infinity(); // running max
    float l = 0.0f;                                    // running sum of exp weights
    for (size_t d = 0; d < head_dim; ++d)
        o_row[d] = 0.0f;

    for (size_t ki = 0; ki < attend_len; ++ki) {
        const float* k_row = k_head + ki * head_dim;
        const float raw_s = dot_f32_neon(q_row, k_row, head_dim) * scale;
        // Rust: raw_s + mask_row.map(|m| m[ki]).unwrap_or(0.0) — the add happens either way.
        const float s = raw_s + (mask_row != nullptr ? mask_row[ki] : 0.0f);

        // Fully-masked position: skip — while m is still -inf the update below would compute
        // exp(-inf - -inf) = NaN and poison the row (Gemma3 sliding-window regression).
        if (s == -std::numeric_limits<float>::infinity()) continue;

        const float m_new = s > m ? s : m;
        const float p = ::expf(s - m_new);
        const float correction = ::expf(m - m_new);
        saxpby_neon(o_row, v_head + ki * head_dim, head_dim, correction, p); // O = c·O + p·v[ki]
        l = correction * l + p;
        m = m_new;
    }

    const float inv_l = l == 0.0f ? 1.0f / FLT_EPSILON : 1.0f / l;
    for (size_t d = 0; d < head_dim; ++d)
        o_row[d] *= inv_l;
}

} // namespace

Result<Tensor> scaled_dot_product_attention(const Tensor& q,
                                            const Tensor& k,
                                            const Tensor& v,
                                            const Tensor* mask,
                                            std::optional<float> scale,
                                            size_t n_kv_heads) {
    const std::vector<size_t> qs = q.shape().dims;
    const std::vector<size_t> ks = k.shape().dims;
    if (qs.size() != 4) return tl::unexpected(Error::rank_mismatch(4, qs.size()));
    if (ks.size() < 3)
        sapient::core::panic("scaled_dot_product_attention: k must have rank >= 3"); // Rust: ks[2]

    const size_t batch = qs[0], n_heads = qs[1], seq_q = qs[2], head_dim = qs[3];
    const size_t seq_k = ks[2];
    const float sc = scale.has_value() ? *scale : 1.0f / ::sqrtf(static_cast<float>(head_dim));
    if (n_kv_heads == 0)
        sapient::core::panic("scaled_dot_product_attention: n_kv_heads must be non-zero");
    const size_t kv_rep = n_heads / n_kv_heads; // 1 for MHA, >1 for GQA

    // K and V contiguous f32 once (they may be strided KV-cache views); Q zero-copy when possible.
    const std::vector<float> k_data = k.to_contiguous_f32_vec();
    const std::vector<float> v_data = v.to_contiguous_f32_vec();
    const F32Cow q_cow = q.to_f32_cow();
    const std::span<const float> q_data = q_cow.get();
    const std::span<const size_t> q_strides = q.strides();
    std::optional<F32Cow> mask_cow;
    std::span<const float> mask_data;
    if (mask != nullptr) {
        mask_cow = mask->to_f32_cow();
        mask_data = mask_cow->get();
    }

    const size_t kv_offset = seq_k >= seq_q ? seq_k - seq_q : 0; // saturating_sub: cached prefix
    const size_t head_out_size = seq_q * head_dim;
    const size_t kv_head_size = seq_k * head_dim;
    std::vector<float> out(batch * n_heads * head_out_size, 0.0f);

    parallel::par_chunks_mut(out, head_out_size, [&](size_t bh, std::span<float> out_chunk) {
        const size_t b = bh / n_heads;
        const size_t h = bh % n_heads;
        if (kv_rep == 0)
            sapient::core::panic(
                "scaled_dot_product_attention: n_heads < n_kv_heads"); // Rust: h / 0
        const size_t kv_h = h / kv_rep;
        const size_t kv_base = (b * n_kv_heads + kv_h) * kv_head_size;
        if (kv_base + kv_head_size > k_data.size() || kv_base + kv_head_size > v_data.size())
            sapient::core::panic("scaled_dot_product_attention: k/v shorter than [batch, "
                                 "n_kv_heads, seq_k, head_dim]");
        const float* k_head = k_data.data() + kv_base;
        const float* v_head = v_data.data() + kv_base;

        std::vector<float> q_row_owned;
        for (size_t qi = 0; qi < seq_q; ++qi) {
            const size_t q_base_elem = b * q_strides[0] + h * q_strides[1] + qi * q_strides[2];
            const float* q_row = nullptr;
            if (q_strides[3] == 1) { // contiguous along head_dim: zero-copy
                if (q_base_elem + head_dim > q_data.size())
                    sapient::core::panic("scaled_dot_product_attention: q row out of range");
                q_row = q_data.data() + q_base_elem;
            } else {
                q_row_owned.resize(head_dim);
                for (size_t d = 0; d < head_dim; ++d) {
                    const size_t idx = q_base_elem + d * q_strides[3];
                    if (idx >= q_data.size())
                        sapient::core::panic(
                            "scaled_dot_product_attention: q element out of range");
                    q_row_owned[d] = q_data[idx];
                }
                q_row = q_row_owned.data();
            }
            // An explicit mask governs (it may already encode causality); otherwise built-in causal.
            const size_t attend_len = mask != nullptr ? seq_k : std::min(qi + kv_offset + 1, seq_k);
            const float* mask_row = nullptr;
            if (mask != nullptr) {
                if ((qi + 1) * seq_k > mask_data.size())
                    sapient::core::panic(
                        "scaled_dot_product_attention: mask shorter than [seq_q, seq_k]");
                mask_row = mask_data.data() + qi * seq_k;
            }
            flash_attn_row(q_row,
                           k_head,
                           v_head,
                           out_chunk.data() + qi * head_dim,
                           sc,
                           head_dim,
                           attend_len,
                           mask_row);
        }
    });

    return Tensor::from_f32_vec(std::move(out), Shape{batch, n_heads, seq_q, head_dim});
}

Tensor causal_mask(size_t seq_q, size_t seq_k) {
    std::vector<float> data(seq_q * seq_k, 0.0f);
    const size_t offset = seq_k >= seq_q ? seq_k - seq_q : 0;
    for (size_t qi = 0; qi < seq_q; ++qi)
        for (size_t ki = 0; ki < seq_k; ++ki)
            if (ki > qi + offset) data[qi * seq_k + ki] = -std::numeric_limits<float>::infinity();
    auto t = Tensor::from_f32_vec(std::move(data), Shape{seq_q, seq_k});
    if (!t.has_value())
        sapient::core::panic("causal_mask: " + t.error().to_string()); // Rust: unwrap
    return std::move(*t);
}

} // namespace sapient::backends_cpu::kernels::attention
