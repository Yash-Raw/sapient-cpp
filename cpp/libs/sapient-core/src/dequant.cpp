// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/core/dequant.hpp"

#include "sapient/core/dtype.hpp"
#include "sapient/core/f16.hpp"
#include "sapient/core/panic.hpp"

namespace sapient::core::dequant {

ScaleMin get_scale_min_k4(size_t j, const uint8_t* s) {
    if (j < 4) return {static_cast<uint8_t>(s[j] & 63), static_cast<uint8_t>(s[j + 4] & 63)};
    return {static_cast<uint8_t>((s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4)),
            static_cast<uint8_t>((s[j + 4] >> 4) | ((s[j] >> 6) << 4))};
}

void q4_0_block(const uint8_t* block, float* out) {
    const float d = f16_le_to_f32(block);
    for (size_t j = 0; j < 16; ++j) {
        const uint8_t byte = block[2 + j];
        const int32_t lo = static_cast<int32_t>(byte & 0x0F) - 8;
        const int32_t hi = static_cast<int32_t>(byte >> 4) - 8;
        out[j] = static_cast<float>(lo) * d;
        out[j + 16] = static_cast<float>(hi) * d;
    }
}

void q8_0_block(const uint8_t* block, float* out) {
    const float d = f16_le_to_f32(block);
    for (size_t j = 0; j < 32; ++j)
        out[j] = static_cast<float>(static_cast<int8_t>(block[2 + j])) * d;
}

void q4_k_block(const uint8_t* block, float* out) {
    const float d = f16_le_to_f32(block);
    const float dmin = f16_le_to_f32(block + 2);
    const uint8_t* scales = block + 4;
    const uint8_t* qs = block + 16;
    size_t out_idx = 0, q_off = 0, is = 0;
    for (int g = 0; g < 4; ++g) {
        const ScaleMin a = get_scale_min_k4(is, scales);
        const ScaleMin b = get_scale_min_k4(is + 1, scales);
        const float d1 = d * static_cast<float>(a.sc), m1v = dmin * static_cast<float>(a.m);
        const float d2 = d * static_cast<float>(b.sc), m2v = dmin * static_cast<float>(b.m);
        for (size_t l = 0; l < 32; ++l) {
            out[out_idx + l] = d1 * static_cast<float>(qs[q_off + l] & 0x0F) - m1v;
            out[out_idx + l + 32] = d2 * static_cast<float>(qs[q_off + l] >> 4) - m2v;
        }
        out_idx += 64;
        q_off += 32;
        is += 2;
    }
}

void q5_k_block(const uint8_t* block, float* out) {
    const float d = f16_le_to_f32(block);
    const float dmin = f16_le_to_f32(block + 2);
    const uint8_t* scales = block + 4;
    const uint8_t* qh = block + 16;
    const uint8_t* ql = block + 48;
    size_t out_idx = 0, ql_off = 0, is = 0;
    uint8_t u1 = 1, u2 = 2;
    for (int g = 0; g < 4; ++g) {
        const ScaleMin a = get_scale_min_k4(is, scales);
        const ScaleMin b = get_scale_min_k4(is + 1, scales);
        const float d1 = d * static_cast<float>(a.sc), m1v = dmin * static_cast<float>(a.m);
        const float d2 = d * static_cast<float>(b.sc), m2v = dmin * static_cast<float>(b.m);
        for (size_t l = 0; l < 32; ++l) {
            const float hi = (qh[l] & u1) ? 16.0f : 0.0f; // per-ELEMENT high bit (the fixed form)
            out[out_idx + l] = d1 * (static_cast<float>(ql[ql_off + l] & 0x0F) + hi) - m1v;
            const float hi2 = (qh[l] & u2) ? 16.0f : 0.0f;
            out[out_idx + l + 32] = d2 * (static_cast<float>(ql[ql_off + l] >> 4) + hi2) - m2v;
        }
        out_idx += 64;
        ql_off += 32;
        is += 2;
        if (is % 8 == 0) {
            u1 = 1;
            u2 = 2;
        } else {
            u1 = static_cast<uint8_t>(u1 << 2);
            u2 = static_cast<uint8_t>(u2 << 2);
        }
    }
}

void q6_k_block(const uint8_t* block, float* out) {
    const uint8_t* ql = block;
    const uint8_t* qh = block + 128;
    const uint8_t* sc = block + 192;
    const float d = f16_le_to_f32(block + 208);
    size_t out_idx = 0, ql_off = 0, qh_off = 0, sc_base = 0;
    for (int half = 0; half < 2; ++half) {
        for (size_t l = 0; l < 32; ++l) {
            const size_t is = l / 16;
            const int32_t q1 =
                static_cast<int32_t>((ql[ql_off + l] & 0x0F) | ((qh[qh_off + l] & 3) << 4)) - 32;
            const int32_t q2 = static_cast<int32_t>((ql[ql_off + l + 32] & 0x0F) |
                                                    (((qh[qh_off + l] >> 2) & 3) << 4)) -
                               32;
            const int32_t q3 =
                static_cast<int32_t>((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) -
                32;
            const int32_t q4 = static_cast<int32_t>((ql[ql_off + l + 32] >> 4) |
                                                    (((qh[qh_off + l] >> 6) & 3) << 4)) -
                               32;
            out[out_idx + l] = d * static_cast<float>(static_cast<int8_t>(sc[sc_base + is])) *
                               static_cast<float>(q1);
            out[out_idx + l + 32] = d *
                                    static_cast<float>(static_cast<int8_t>(sc[sc_base + is + 2])) *
                                    static_cast<float>(q2);
            out[out_idx + l + 64] = d *
                                    static_cast<float>(static_cast<int8_t>(sc[sc_base + is + 4])) *
                                    static_cast<float>(q3);
            out[out_idx + l + 96] = d *
                                    static_cast<float>(static_cast<int8_t>(sc[sc_base + is + 6])) *
                                    static_cast<float>(q4);
        }
        out_idx += 128;
        ql_off += 64;
        qh_off += 32;
        sc_base += 8;
    }
}

namespace {
template <size_t BlockBytes, size_t BlockNumel, void (*Fn)(const uint8_t*, float*)>
void blocks(std::span<const uint8_t> bytes, size_t numel, float* out) {
    const size_t nblocks = bytes.size() / BlockBytes;
    if (nblocks * BlockNumel > numel) panic("dequant: block count exceeds output capacity");
    for (size_t b = 0; b < nblocks; ++b)
        Fn(bytes.data() + b * BlockBytes, out + b * BlockNumel);
}
template <size_t BlockBytes, void (*Fn)(const uint8_t*, float*)>
void r4(std::span<const uint8_t> bytes, size_t rows, size_t k, float* out) {
    // Mirrors the three preconditions `repack_q4_k_rows4`/`repack_q6_k_rows4`
    // assert (crates/sapient-backends/cpu/src/kernels/quant.rs:1230-1234,
    // :2001-2005): without them a bad (rows, k, bytes) triple either
    // out-of-bounds-writes `out` (an aggregate size check alone would not
    // catch a wrong rows/k split of the same total) or divides by zero below
    // (`nb == 0` when `k < 256`). Checked in the same order as the Rust
    // asserts, before `nb` is used for anything.
    if (rows % 4 != 0) panic("dequant r4: rows must be a multiple of 4");
    if (k % 256 != 0) panic("dequant r4: k must be a multiple of 256");
    const size_t nb = k / 256;
    const size_t row_bytes = nb * BlockBytes;
    if (bytes.size() != rows * row_bytes) panic("dequant r4: byte length does not match rows * k");
    const size_t nblocks = bytes.size() / BlockBytes;
    for (size_t p = 0; p < nblocks; ++p) {
        const size_t g = p / (4 * nb), rem = p % (4 * nb), b = rem / 4, r = rem % 4;
        const size_t row = g * 4 + r;
        Fn(bytes.data() + p * BlockBytes, out + (row * nb + b) * 256);
    }
}
} // namespace

void q4_0(std::span<const uint8_t> bytes, size_t numel, float* out) {
    blocks<Q4_0_BLOCK_BYTES, 32, q4_0_block>(bytes, numel, out);
}
void q8_0(std::span<const uint8_t> bytes, size_t numel, float* out) {
    blocks<Q8_0_BLOCK_BYTES, 32, q8_0_block>(bytes, numel, out);
}
void q4_k(std::span<const uint8_t> bytes, size_t numel, float* out) {
    blocks<Q4_K_BLOCK_BYTES, 256, q4_k_block>(bytes, numel, out);
}
void q5_k(std::span<const uint8_t> bytes, size_t numel, float* out) {
    blocks<Q5_K_BLOCK_BYTES, 256, q5_k_block>(bytes, numel, out);
}
void q6_k(std::span<const uint8_t> bytes, size_t numel, float* out) {
    blocks<Q6_K_BLOCK_BYTES, 256, q6_k_block>(bytes, numel, out);
}
void q4_k_r4(std::span<const uint8_t> bytes, size_t rows, size_t k, float* out) {
    r4<Q4_K_BLOCK_BYTES, q4_k_block>(bytes, rows, k, out);
}
void q6_k_r4(std::span<const uint8_t> bytes, size_t rows, size_t k, float* out) {
    r4<Q6_K_BLOCK_BYTES, q6_k_block>(bytes, rows, k, out);
}

} // namespace sapient::core::dequant
