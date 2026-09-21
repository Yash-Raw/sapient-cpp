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

// ── Q4_K ────────────────────────────────────────────────────────────────────────
// Block: [0..2) d f16 | [2..4) dmin f16 | [4..16) 12 packed 6-bit (scale,min) pairs | [16..144) 128
// nibble bytes. Per 64-weight group g: lo nibbles ↔ x[64g..64g+32) with (sc,m) pair 2g, hi nibbles
// ↔ x[64g+32..64g+64) with pair 2g+1.
/// Row · f32 activations (NEON on aarch64, scalar elsewhere).
float dot_q4_k_row_f32(std::span<const uint8_t> row_data, std::span<const float> x);
/// W4A8: row · per-32 int8 activations; `x_sums` = `i8_block_sums(x_i8)` (precomputed, never
/// re-reduced). Scalar reference for the SDOT kernels (same integer dot, same f32 combine order).
float dot_q4_k_row_q8_scalar(std::span<const uint8_t> row_data,
                             std::span<const int8_t> x_i8,
                             std::span<const float> x_scales,
                             std::span<const int32_t> x_sums);
/// Repack `n` Q4_K rows into the Q4_K_R4 layout: groups of 4 rows, super-blocks block-major within
/// the group (`[r0.b0, r1.b0, r2.b0, r3.b0, r0.b1, …]`). Panics unless `n % 4 == 0`, `k % 256 == 0`
/// and `blocks.size() == n · k/256 · 144`.
std::vector<uint8_t> repack_q4_k_rows4(std::span<const uint8_t> blocks, size_t n, size_t k);
#if defined(__aarch64__) || defined(_M_ARM64)
/// NEON W4A8 row dot via `sdot`; bit-identical to `dot_q4_k_row_q8_scalar`. Precondition: dotprod.
float dot_q4_k_row_q8_neon(std::span<const uint8_t> row_data,
                           std::span<const int8_t> x_i8,
                           std::span<const float> x_scales,
                           std::span<const int32_t> x_sums);
/// Four row-major Q4_K rows against ONE int8 activation row (activations loaded once per 64-weight
/// group); each lane bit-identical to `dot_q4_k_row_q8_neon`. Precondition: dotprod.
std::array<float, 4> dot_q4_k_4rows_q8_neon(std::array<std::span<const uint8_t>, 4> rows,
                                            std::span<const int8_t> x_i8,
                                            std::span<const float> x_scales,
                                            std::span<const int32_t> x_sums);
/// Four Q4_K rows in the R4 layout (`packed` = one whole row-group) against one int8 activation
/// row; each lane bit-identical to `dot_q4_k_row_q8_neon`. Precondition: dotprod.
std::array<float, 4> dot_q4_k_4rows_r4_neon(std::span<const uint8_t> packed,
                                            std::span<const int8_t> x_i8,
                                            std::span<const float> x_scales,
                                            std::span<const int32_t> x_sums);
#endif
namespace detail {
float dot_q4_k_row_f32_scalar(std::span<const uint8_t> row_data, std::span<const float> x);
#if defined(__aarch64__) || defined(_M_ARM64)
float dot_q4_k_row_f32_neon(std::span<const uint8_t> row_data, std::span<const float> x);
#endif
} // namespace detail

// ── Q4_K × Q8_K activations (integer-domain sub-scale combine), SMMLA prefill ───
/// Scalar oracle: per super-block `isum = Σ sc·dot`, `imin = Σ mn·bsum` in i32, then ONE
/// `acc += x_scales[b] · (d·isum − dmin·imin)`. `x_scales` has one f32 per 256, `x_sums` one i32 per 32.
float dot_q4_k_row_q8k_scalar(std::span<const uint8_t> row_data,
                              std::span<const int8_t> x_i8,
                              std::span<const float> x_scales,
                              std::span<const int32_t> x_sums);
#if defined(__aarch64__) || defined(_M_ARM64)
/// `sdot` core + integer-domain combine; bit-identical to `dot_q4_k_row_q8k_scalar`. Precondition: dotprod.
float dot_q4_k_row_q8k_neon(std::span<const uint8_t> row_data,
                            std::span<const int8_t> x_i8,
                            std::span<const float> x_scales,
                            std::span<const int32_t> x_sums);
/// Four row-major rows × one Q8_K row; lanes bit-identical to `dot_q4_k_row_q8k_neon`. Iterates
/// `min(n_blocks, x_scales.size())` blocks (Rust `.take(n_blocks)`). Precondition: dotprod.
std::array<float, 4> dot_q4_k_4rows_q8k_neon(std::array<std::span<const uint8_t>, 4> rows,
                                             std::span<const int8_t> x_i8,
                                             std::span<const float> x_scales,
                                             std::span<const int32_t> x_sums);
/// Four R4 rows × one Q8_K row; same lane identity and `.take` rule. Precondition: dotprod.
std::array<float, 4> dot_q4_k_4rows_r4_q8k_neon(std::span<const uint8_t> packed,
                                                std::span<const int8_t> x_i8,
                                                std::span<const float> x_scales,
                                                std::span<const int32_t> x_sums);
/// Four R4 rows × TWO per-32 int8 activation rows via `smmla` — the prefill kernel. Returns
/// `[[row0·x0, row0·x1], …, [row3·x0, row3·x1]]`, every lane bit-identical to
/// `dot_q4_k_row_q8_neon`. Precondition: i8mm.
std::array<std::array<float, 2>, 4> dot_q4_k_4rows_r4_x2_smmla(std::span<const uint8_t> packed,
                                                               std::span<const int8_t> x0_i8,
                                                               std::span<const float> x0_scales,
                                                               std::span<const int32_t> x0_sums,
                                                               std::span<const int8_t> x1_i8,
                                                               std::span<const float> x1_scales,
                                                               std::span<const int32_t> x1_sums);
/// Four R4 rows × TWO Q8_K rows via `smmla`; lanes bit-identical to `dot_q4_k_row_q8k_neon`;
/// iterates `min(nb, x0_scales.size(), x1_scales.size())` blocks. Precondition: i8mm.
std::array<std::array<float, 2>, 4>
dot_q4_k_4rows_r4_x2_q8k_smmla(std::span<const uint8_t> packed,
                               std::span<const int8_t> x0_i8,
                               std::span<const float> x0_scales,
                               std::span<const int32_t> x0_sums,
                               std::span<const int8_t> x1_i8,
                               std::span<const float> x1_scales,
                               std::span<const int32_t> x1_sums);
#endif

// ── Q5_K ────────────────────────────────────────────────────────────────────────
// Block: [0..2) d | [2..4) dmin | [4..16) scales | [16..48) qh (per-ELEMENT 5th bits, bit-plane
// selected by u1/u2) | [48..176) ql nibbles.
/// Row · f32 activations (NEON on aarch64, scalar elsewhere). No int8 variant exists (Rust has none).
float dot_q5_k_row_f32(std::span<const uint8_t> row_data, std::span<const float> x);

// ── Q6_K ────────────────────────────────────────────────────────────────────────
// Block: [0..128) ql | [128..192) qh (two 2-bit fields per byte) | [192..208) 16 SIGNED i8 scales,
// one per 16 weights (offsets +0/+2/+4/+6 within a 128-half, `is = l/16`, base +8 per half — the
// historical token-salad bug) | [208..210) d f16.
/// Row · f32 activations (NEON on aarch64, scalar elsewhere).
float dot_q6_k_row_f32(std::span<const uint8_t> row_data, std::span<const float> x);
/// Repack `n` Q6_K rows into the Q6_K_R4 layout (same 4-row block-major interleave as Q4_K_R4,
/// over 210-byte blocks). Same panics as `repack_q4_k_rows4`.
std::vector<uint8_t> repack_q6_k_rows4(std::span<const uint8_t> blocks, size_t n, size_t k);
#if defined(__aarch64__) || defined(_M_ARM64)
/// Four R4 Q6_K rows against one f32 activation vector (plain NEON; the decode path when dotprod
/// is absent); each lane bit-identical to `detail::dot_q6_k_row_f32_neon`.
std::array<float, 4> dot_q6_k_4rows_r4_neon(std::span<const uint8_t> packed,
                                            std::span<const float> x);
#endif
namespace detail {
float dot_q5_k_row_f32_scalar(std::span<const uint8_t> row_data, std::span<const float> x);
float dot_q6_k_row_f32_scalar(std::span<const uint8_t> row_data, std::span<const float> x);
#if defined(__aarch64__) || defined(_M_ARM64)
float dot_q5_k_row_f32_neon(std::span<const uint8_t> row_data, std::span<const float> x);
float dot_q6_k_row_f32_neon(std::span<const uint8_t> row_data, std::span<const float> x);
#endif
} // namespace detail

// ── Q6_K W6A8 / Q8_K / SMMLA (Task 5) ───────────────────────────────────────────
/// W6A8 scalar reference: per 16-element scale group `acc += ((d · sc) · xs) · dot` with the −32
/// folded into the integer dot; `x_scales` one per 32.
float dot_q6_k_row_q8_scalar(std::span<const uint8_t> row_data,
                             std::span<const int8_t> x_i8,
                             std::span<const float> x_scales);
/// Q8_K scalar oracle: `isum += sc · dot` (i32) across the super-block, then
/// `acc += x_scales[b] · d · isum`; `x_scales` one per 256.
float dot_q6_k_row_q8k_scalar(std::span<const uint8_t> row_data,
                              std::span<const int8_t> x_i8,
                              std::span<const float> x_scales);
#if defined(__aarch64__) || defined(_M_ARM64)
/// One `sdot` per 16-element group; bit-identical to `dot_q6_k_row_q8_scalar`. Precondition: dotprod.
float dot_q6_k_row_q8_neon(std::span<const uint8_t> row_data,
                           std::span<const int8_t> x_i8,
                           std::span<const float> x_scales);
/// Bit-identical to `dot_q6_k_row_q8k_scalar`. Precondition: dotprod.
float dot_q6_k_row_q8k_neon(std::span<const uint8_t> row_data,
                            std::span<const int8_t> x_i8,
                            std::span<const float> x_scales);
/// Four R4 rows, W6A8; lanes bit-identical to `dot_q6_k_row_q8_neon`. Precondition: dotprod.
std::array<float, 4> dot_q6_k_4rows_r4_q8_neon(std::span<const uint8_t> packed,
                                               std::span<const int8_t> x_i8,
                                               std::span<const float> x_scales);
/// Four R4 rows × one Q8_K row; lanes bit-identical to `dot_q6_k_row_q8k_neon`; iterates
/// `min(nb, x_scales.size())` blocks. Precondition: dotprod.
std::array<float, 4> dot_q6_k_4rows_r4_q8k_neon(std::span<const uint8_t> packed,
                                                std::span<const int8_t> x_i8,
                                                std::span<const float> x_scales);
/// Four R4 rows × TWO per-32 int8 rows via `smmla`; lanes bit-identical to `dot_q6_k_row_q8_neon`
/// (no sums — Q6_K has no min term). Precondition: i8mm.
std::array<std::array<float, 2>, 4> dot_q6_k_4rows_r4_x2_smmla(std::span<const uint8_t> packed,
                                                               std::span<const int8_t> x0_i8,
                                                               std::span<const float> x0_scales,
                                                               std::span<const int8_t> x1_i8,
                                                               std::span<const float> x1_scales);
/// Four R4 rows × TWO Q8_K rows via `smmla`; lanes bit-identical to `dot_q6_k_row_q8k_neon`;
/// iterates `min(nb, x0_scales.size(), x1_scales.size())` blocks. Precondition: i8mm.
std::array<std::array<float, 2>, 4>
dot_q6_k_4rows_r4_x2_q8k_smmla(std::span<const uint8_t> packed,
                               std::span<const int8_t> x0_i8,
                               std::span<const float> x0_scales,
                               std::span<const int8_t> x1_i8,
                               std::span<const float> x1_scales);
#endif

} // namespace sapient::backends_cpu::kernels::quant
