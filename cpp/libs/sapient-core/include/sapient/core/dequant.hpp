// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// The ONE dequantiser (spec §2.1): Rust keeps three hand-synced copies of this arithmetic
// (sapient-core tensor.rs, sapient-io gguf.rs, sapient-backends-cpu quant.rs); the C++ tree keeps
// this file and everyone calls it. Arithmetic is transcribed from tensor.rs:362-576 — keep the
// evaluation order (e.g. `(d * sc) * q`, `d1 * nib - m1v`), it decides bit-identity.

#include <cstddef>
#include <cstdint>
#include <span>

namespace sapient::core::dequant {

struct ScaleMin {
    uint8_t sc;
    uint8_t m;
};
/// 6-bit (scale, min) pair `j` (0..=7) from the 12 packed scale bytes of a Q4_K/Q5_K block.
ScaleMin get_scale_min_k4(size_t j, const uint8_t* scales);

void q4_0_block(const uint8_t* block, float* out); // 18 B → 32 values, ggml split-nibble order
void q8_0_block(const uint8_t* block, float* out); // 34 B → 32
void q4_k_block(const uint8_t* block, float* out); // 144 B → 256
void q5_k_block(const uint8_t* block, float* out); // 176 B → 256 (per-ELEMENT 5th bit)
void q6_k_block(const uint8_t* block, float* out); // 210 B → 256 (+0/+2/+4/+6 scale indexing)

/// Whole tensors: `bytes.size() / block_bytes` blocks in order; `numel` is the capacity of `out`.
/// The CALLER must zero-initialise `out` — any capacity beyond `nblocks * block_numel` is left
/// untouched, not memset here.
void q4_0(std::span<const uint8_t> bytes, size_t numel, float* out);
void q8_0(std::span<const uint8_t> bytes, size_t numel, float* out);
void q4_k(std::span<const uint8_t> bytes, size_t numel, float* out);
void q5_k(std::span<const uint8_t> bytes, size_t numel, float* out);
void q6_k(std::span<const uint8_t> bytes, size_t numel, float* out);
/// Row-interleaved layouts (2-D [rows, k]): packed block p → row g*4+r, block b; out[(row*nb + b)*256..].
void q4_k_r4(std::span<const uint8_t> bytes, size_t rows, size_t k, float* out);
void q6_k_r4(std::span<const uint8_t> bytes, size_t rows, size_t k, float* out);

} // namespace sapient::core::dequant
