// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-backends/cpu/src/kernels/quant.rs: quantized weight blocks (ggml layouts)
// and the on-the-fly dequantizing dot products that keep weights quantized in memory. Standalone
// like the Rust module — no Tensor dependency; byte/float/int8 spans in, f32 out.
//
// ISA variants are mirrored, never upgraded (spec §3.2): scalar everywhere; on aarch64 plain NEON
// (compile-time baseline), NEON+dotprod (`vdotq_s32` = Rust's `sdot` inline asm) and NEON+i8mm
// (`vmmlaq_s32` = `smmla`), each dotprod/i8mm kernel carrying `__attribute__((target(...)))`;
// on x86_64 exactly one AVX2+FMA kernel (`detail::dot_q8_0_row_avx2`), reached only through
// `dot_q8_0_row_f32`'s runtime gate. Rust's `unsafe fn` preconditions ("caller verified dotprod /
// i8mm") are preconditions here too: matmul.cpp gates on cpu_features before calling them.
//
// Every accumulation order, reduction intrinsic and rounding rule follows the Rust source (porting
// map §4); Rust index panics are entry checks that `sapient::core::panic()`.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace sapient::backends_cpu::kernels::quant {

/// Weights per Q4_0/Q8_0 block.
inline constexpr size_t QK = 32;
/// Bytes per Q4_0 block: 2 (f16 scale) + 16 (packed nibbles).
inline constexpr size_t Q4_0_BLOCK_BYTES = 18;
/// Bytes per Q8_0 block: 2 (f16 scale) + 32 (i8 quants).
inline constexpr size_t Q8_0_BLOCK_BYTES = 34;
/// Weights per K-quant super-block.
inline constexpr size_t QK_K = 256;
inline constexpr size_t Q4_K_BLOCK_BYTES = 144;
inline constexpr size_t Q5_K_BLOCK_BYTES = 176;
inline constexpr size_t Q6_K_BLOCK_BYTES = 210;

/// Rust `(Vec<i8>, Vec<f32>)` from `quantize_row_to_i8_blocks`: int8 quants + ONE scale per 32.
struct I8Blocks {
    std::vector<int8_t> q;
    std::vector<float> scales;
};
/// Rust `(Vec<i8>, Vec<f32>, Vec<i32>)` from `quantize_row_to_q8k`: int8 quants, ONE scale per
/// 256-element super-block, per-32 sums (llama.cpp `block_q8_K`).
struct Q8kRow {
    std::vector<int8_t> q;
    std::vector<float> scales;
    std::vector<int32_t> sums;
};

// ── Q4_0 ────────────────────────────────────────────────────────────────────────
/// Quantize 32 f32 into one Q4_0 block (ggml: `d = vmax / -8`, sign preserved; nibbles truncated).
std::array<uint8_t, Q4_0_BLOCK_BYTES> quantize_q4_0_block(std::span<const float> x);
/// One Q4_0 block → 32 values (delegates to the shared core::dequant::q4_0_block).
void dequantize_q4_0_block(std::span<const uint8_t> block, std::span<float> out);
/// One block · 32 activations (NEON on aarch64, scalar elsewhere).
float dot_q4_0_block_f32(std::span<const uint8_t> block, std::span<const float> x);
/// Whole row: f32 `acc += block_dot` sequentially over the 18-byte blocks.
float dot_q4_0_row_f32(std::span<const uint8_t> row_blocks, std::span<const float> x);
/// Quantize a full f32 row (`w.size() % 32 == 0`) into packed Q4_0 blocks.
std::vector<uint8_t> quantize_q4_0_row(std::span<const float> w);

// ── Q8_0 ────────────────────────────────────────────────────────────────────────
/// Quantize 32 f32 into one Q8_0 block (`scale = max_abs/127`, f16 scale, roundf then clamp).
std::array<uint8_t, Q8_0_BLOCK_BYTES> quantize_q8_0_block(std::span<const float> x);
/// One block · 32 activations (NEON on aarch64, scalar elsewhere).
float dot_q8_0_block_f32(std::span<const uint8_t> block, std::span<const float> x);
/// Whole row · f32 activations: AVX2+FMA on x86_64 with `has_avx2_fma()`, else per-block.
float dot_q8_0_row_f32(std::span<const uint8_t> row_blocks, std::span<const float> x);
/// Whole row · int8 activations with per-32 scales — the portable twin of `dot_q8_0_row_sdot`
/// (same integer dots, same f32 combine order).
float dot_q8_0_row_i8_scalar(std::span<const uint8_t> row_blocks,
                             std::span<const int8_t> x_i8,
                             std::span<const float> x_scales);

// ── activation quantisers ───────────────────────────────────────────────────────
/// Per-32-block int8 activations (`x.size() % 32 == 0`); a zero block gets scale 1.0.
I8Blocks quantize_row_to_i8_blocks(std::span<const float> x);
/// Per-32 Σq (i32) of an int8 row — the precomputed `x_sums` every W4A8 kernel takes.
std::vector<int32_t> i8_block_sums(std::span<const int8_t> q);
/// Q8_K-style activations (`x.size() % 256 == 0`): one scale per 256, sums per 32.
Q8kRow quantize_row_to_q8k(std::span<const float> x);

#if defined(__aarch64__) || defined(_M_ARM64)
/// Q8_0 row · int8 activations via `sdot`. Precondition: `cpu_features::has_dotprod()`.
float dot_q8_0_row_sdot(std::span<const uint8_t> row_blocks,
                        std::span<const int8_t> x_i8,
                        std::span<const float> x_scales);
#endif

namespace detail {
/// Rust `f as i32`: truncation toward zero, saturating, NaN → 0.
int32_t f32_to_i32_sat(float v);
/// Rust `(v).round().clamp(-127.0, 127.0) as i8`: half away from zero, NaN → 0.
int8_t round_clamp_i8(float v);
/// ggml nibble: `clamp((scaled + 8.5) as i32, 0, 15)`.
uint8_t nibble(float scaled);
/// Rust `byte as i8 as i32` — the one place bytes are sign-extended.
int32_t i8v(uint8_t b);
float dot_q4_0_block_scalar(std::span<const uint8_t> block, std::span<const float> x);
float dot_q8_0_block_scalar(std::span<const uint8_t> block, std::span<const float> x);
#if defined(__aarch64__) || defined(_M_ARM64)
/// Integer dot of one Q8_0 block with 32 int8 activations (two `sdot`, `vaddvq_s32`).
int32_t dot_q8_0_block_sdot(std::span<const uint8_t> block, std::span<const int8_t> x_i8);
#endif
#if defined(__x86_64__) || defined(_M_X64)
/// The one x86 SIMD kernel. Precondition: `cpu_features::has_avx2_fma()`.
float dot_q8_0_row_avx2(std::span<const uint8_t> row_blocks, std::span<const float> x);
#endif
} // namespace detail

// ── Q4_K (Task 2) ───────────────────────────────────────────────────────────────
// ── Q4_K × Q8_K, SMMLA (Task 3) ─────────────────────────────────────────────────
// ── Q5_K, Q6_K f32, Q6_K repack/R4 f32 (Task 4) ─────────────────────────────────
// ── Q6_K W6A8 / Q8_K / SMMLA (Task 5) ───────────────────────────────────────────

} // namespace sapient::backends_cpu::kernels::quant
