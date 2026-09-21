# Porting map: sapient-backends-cpu kernels, spinpool, thermal (Rust → C++)

> Reference for sub-project 1a plans C (dense kernels), D (quant kernels) and E (spinpool/thermal). Generated 2026-09-21 from a full read of the crate at commit 3cd6997; verify line numbers against the tree before relying on them.


Root: `/Users/yashchaurasia/Documents/Codes/sapient-cpp/.claude/worktrees/feat-cpp-sp0-scaffold/crates/sapient-backends/cpu/src/`
All line numbers are `file:line` in that tree. `backend.rs` / `pool.rs` excluded as instructed.

Module tree (`lib.rs:13-20`, `kernels/mod.rs:6-14`): `backend`, `kernels::{attention, conv2d, elementwise, layernorm, matmul, quant, reduce, rope, softmax}`, `pool`, `spinpool`, `thermal`.
Sizes: quant 3787, matmul 1536, attention 530, spinpool 541, thermal 347, elementwise 232, conv2d 216, rope 206, layernorm 150, reduce 140, softmax 120.

---

## 0. Cross-cutting facts the C++ port must fix first

* **`quant.rs` has ZERO sapient-core dependency** — only `half::f16` (`quant.rs:19`) and raw `&[u8]/&[f32]/&[i8]`. Port it as a standalone header/TU with an fp16 helper. Everything else (`matmul`, `attention`, `rope`, …) takes `Tensor`.
* **No `f32::mul_add` anywhere** in the crate (grep: 0 hits). All fusion is explicit SIMD (`vfmaq_f32`, `_mm256_fmadd_ps`) or plain `a*b + c` in scalar code. So `-ffp-contract=off` + *explicit* intrinsic FMA where the Rust used one, plain mul/add where it did not.
* **f16→f32 is always `half::f16::from_le_bytes([b0,b1]).to_f32()`** — IEEE-correct, handles subnormal/inf/NaN. `vcvt_f32_f16` is never used. The ONE exception is `matmul.rs:252-298` (`dot_f32_x_f16_neon`), which does manual bit-surgery valid for **normals only** (subnormal/inf/NaN F16 weights decode differently there than in the scalar tail at `matmul.rs:294`, which uses `half::f16::from_bits().to_f32()`). Reproduce both branches verbatim, including the divergence.
* ISA groups present: **scalar**, **aarch64 NEON** (`target_feature="neon"`, unconditional on aarch64 — never runtime-detected), **aarch64 NEON+dotprod** (`sdot` via `asm!`), **aarch64 NEON+i8mm** (`smmla` via `asm!`), **x86_64 AVX2+FMA** (runtime `is_x86_feature_detected!`). **There are NO x86 K-quant kernels** — Q4_K/Q5_K/Q6_K on x86 fall to the scalar reference. AVX2 exists only for `dot_q8_0_row_f32` and `dot_f32_fast`.

---

## 1. `kernels/quant.rs` — function inventory (3787 lines)

Constants: `QK=32` (:32), `Q4_0_BLOCK_BYTES=18` (:34), `Q8_0_BLOCK_BYTES=34` (:36), `QK_K=256` (:588), `Q4_K_BLOCK_BYTES=144` (:589), `Q5_K_BLOCK_BYTES=176` (:590), `Q6_K_BLOCK_BYTES=210` (:591).

### 1a. Scalar / portable (no cfg, no unsafe unless noted)
| line | signature | notes |
|---|---|---|
| :41 | `pub fn quantize_q4_0_block(x:&[f32]) -> [u8;18]` | amax/vmax scan, `d = vmax/-8.0`, `id = d!=0 ? 1/d : 0` |
| :67 | `fn nibble(scaled:f32) -> u8` | `((scaled + 8.5) as i32).clamp(0,15)` — **truncation toward zero**, not round |
| :74 | `pub fn dequantize_q4_0_block(block:&[u8], out:&mut [f32])` | |
| :93 | `#[inline] pub fn dot_q4_0_block_f32(block,&x) -> f32` | cfg dispatch (see §2) |
| :104 | `#[inline(always)] #[allow(dead_code)] fn dot_q4_0_block_scalar` | `acc += lo*x[j] + hi*x[j+16]` sequential, `* d` at end |
| :197 | `pub fn dot_q4_0_row_f32(row_blocks,&x) -> f32` | `acc += block_dot` per 18-byte block, f32 sequential |
| :208 | `pub fn quantize_q4_0_row(w:&[f32]) -> Vec<u8>` | |
| :223 | `pub fn quantize_q8_0_block(x:&[f32]) -> [u8;34]` | `max_abs = fold(0.0, f32::max)`; `scale = max_abs/127`; `d = f16(scale)`; `inv = scale>0 ? 1/scale : 0`; `(v*inv).round().clamp(-127,127) as i8` |
| :242 | `#[inline] pub fn dot_q8_0_block_f32` | cfg dispatch |
| :253 | `fn dot_q8_0_block_scalar` | sequential f32, `*d` at end |
| :317 | `pub fn quantize_row_to_i8_blocks(x) -> (Vec<i8>, Vec<f32>)` | **per-32 scales**; `scale = max_abs>0 ? max_abs/127 : 1.0`; `inv = scale>0 ? 1/scale : 0`; `(v*inv).round().clamp(-127,127) as i8` |
| :343 | `pub fn i8_block_sums(q:&[i8]) -> Vec<i32>` | per-32 Σ in i32, iterator order |
| :366 | `pub fn quantize_row_to_q8k(x) -> (Vec<i8>, Vec<f32>, Vec<i32>)` | **one scale per 256**; same rounding; sums still per-32 |
| :394 | `pub fn dot_q4_k_row_q8k_scalar(row_data,x_i8,x_scales,x_sums) -> f32` | integer-domain oracle |
| :502 | `pub fn dot_q8_0_row_i8_scalar(row_blocks,x_i8,x_scales) -> f32` | |
| :523 | `pub fn dot_q8_0_row_f32(row_blocks,&x) -> f32` | runtime AVX2 check (see §2) |
| :596 | `#[inline(always)] fn get_scale_min_k4(j, scales) -> (u8,u8)` | 6-bit unpack, §5 |
| :616 | `pub fn dot_q4_k_row_f32(row_data,&x)` | cfg dispatch |
| :625 | `#[allow(dead_code)] fn dot_q4_k_row_f32_scalar` | |
| :750 | `pub fn dot_q4_k_row_q8_scalar(row_data,x_i8,x_scales,x_sums) -> f32` | W4A8, per-32 |
| :1229 | `pub fn repack_q4_k_rows4(blocks,n,k) -> Vec<u8>` | asserts `n%4==0`, `k%256==0` |
| :1617 | `pub fn dot_q5_k_row_f32` | cfg dispatch |
| :1626 | `#[allow(dead_code)] fn dot_q5_k_row_f32_scalar` | per-element 5th bit (`qh[l]`, bit-plane u1/u2) |
| :1757 | `pub fn dot_q6_k_row_q8_scalar(row_data,x_i8,x_scales) -> f32` | W6A8 oracle |
| :1878 | `pub fn dot_q6_k_row_q8k_scalar(row_data,x_i8,x_scales) -> f32` | integer-domain oracle |
| :2000 | `pub fn repack_q6_k_rows4(blocks,n,k) -> Vec<u8>` | |
| :2564 | `pub fn dot_q6_k_row_f32` | cfg dispatch |
| :2573 | `#[allow(dead_code)] fn dot_q6_k_row_f32_scalar` | |

### 1b. aarch64 NEON only (`#[cfg(target_arch="aarch64")] #[target_feature(enable="neon")]`, all `unsafe`)
| line | signature | intrinsics |
|---|---|---|
| :123 | `unsafe fn dot_q4_0_block_neon(block,&x)->f32` | `vld1q_u8, vandq_u8, vdupq_n_u8, vshrq_n_u8, vsubq_u8, vreinterpretq_s8_u8, vmovl_s8/vmovl_high_s8, vmovl_s16/vmovl_high_s16, vget_low_s16, vcvtq_f32_s32, vld1q_f32, vmulq_f32, vfmaq_f32, vaddvq_f32` |
| :181 | `#[inline(always)] unsafe fn vmovl_s8_low(int8x16_t)->int16x8_t` | no `target_feature` attr |
| :188 | `#[inline(always)] unsafe fn vmovl_s8_high` | |
| :266 | `unsafe fn dot_q8_0_block_neon` | `vld1_s8, vmovl_s8, vmovl_s16, vmovl_high_s16, vcvtq_f32_s32, vfmaq_f32, vaddvq_f32` |
| :661 | `unsafe fn dot_q4_k_row_f32_neon` | `vdup_n_u8, vld1_u8, vand_u8, vshr_n_u8::<4>, vmovl_u8, vmovl_u16, vmovl_high_u16, vcvtq_f32_u32, vld1q_f32, vfmaq_f32, vaddq_f32, vaddvq_f32` |
| :992 | `#[inline] unsafe fn vtrn1q_s64_s8(a,b)` | `vtrn1q_s64` + reinterprets |
| :1004 | `#[inline] unsafe fn vtrn2q_s64_s8(a,b)` | `vtrn2q_s64` |
| :1678 | `unsafe fn dot_q5_k_row_f32_neon` | `vtstq_u8, vandq_u8, vaddq_u8, vshrq_n_u8::<4>, vmovl_u8/high, vcvtq_f32_u32, vmulq_n_f32, vsubq_f32, vfmaq_f32, vaddvq_f32` |
| :2208 | `pub unsafe fn dot_q6_k_4rows_r4_neon(packed,&x)->[f32;4]` | f32 activations, R4 layout; `vorrq_u8, vshlq_n_u8::<4>, vsubq_f32, vmulq_f32, vfmaq_f32, vaddvq_f32` |
| :2623 | `unsafe fn dot_q6_k_row_f32_neon` | same family |

### 1c. aarch64 NEON+dotprod (`#[target_feature(enable="neon,dotprod")]`, all `unsafe`) — `sdot` via inline asm
Core primitive `:804 #[inline] unsafe fn sdot_s32(acc:int32x4_t, w:int8x16_t, x:int8x16_t) -> int32x4_t` = `core::arch::asm!("sdot {0:v}.4s, {1:v}.16b, {2:v}.16b", inout(vreg) a, in(vreg) w, in(vreg) x, options(nomem,nostack))` (`quant.rs:810-817`).
| line | signature |
|---|---|
| :442 | `unsafe fn dot_q8_0_block_sdot(block,x_i8) -> i32` (two raw `asm!("sdot …")` at :458 and :465, then `vaddvq_s32`) |
| :486 | `pub unsafe fn dot_q8_0_row_sdot(row_blocks,x_i8,x_scales) -> f32` |
| :1024 | `pub unsafe fn dot_q4_k_row_q8k_neon(row_data,x_i8,x_scales,x_sums) -> f32` |
| :1084 | `pub unsafe fn dot_q4_k_row_q8_neon(row_data,x_i8,x_scales,x_sums) -> f32` |
| :1154 | `pub unsafe fn dot_q4_k_4rows_q8_neon(rows:[&[u8];4], x_i8, x_scales, x_sums) -> [f32;4]` |
| :1259 | `pub unsafe fn dot_q4_k_4rows_r4_neon(packed, x_i8, x_scales, x_sums) -> [f32;4]` |
| :1335 | `pub unsafe fn dot_q4_k_4rows_q8k_neon(rows:[&[u8];4], …) -> [f32;4]` |
| :1410 | `pub unsafe fn dot_q4_k_4rows_r4_q8k_neon(packed, …) -> [f32;4]` |
| :1809 | `pub unsafe fn dot_q6_k_row_q8_neon(row_data,x_i8,x_scales) -> f32` |
| :1932 | `pub unsafe fn dot_q6_k_row_q8k_neon(row_data,x_i8,x_scales) -> f32` |
| :2030 | `pub unsafe fn dot_q6_k_4rows_r4_q8_neon(packed,x_i8,x_scales) -> [f32;4]` |
| :2119 | `pub unsafe fn dot_q6_k_4rows_r4_q8k_neon(packed,x_i8,x_scales) -> [f32;4]` |

### 1d. aarch64 NEON+i8mm (`#[target_feature(enable="neon,i8mm")]`, all `unsafe`) — `smmla` via inline asm
Core primitive `:829 #[inline] unsafe fn smmla_s32(acc,a,b) -> int32x4_t` = `asm!("smmla {0:v}.4s, {1:v}.16b, {2:v}.16b", …)` (`quant.rs:835-841`); result lanes `[a0·b0, a0·b1, a1·b0, a1·b1]`.
| line | signature |
|---|---|
| :861 | `pub unsafe fn dot_q4_k_4rows_r4_x2_smmla(packed, x0_i8,x0_scales,x0_sums, x1_i8,x1_scales,x1_sums) -> [[f32;2];4]` (`#[allow(clippy::too_many_arguments)]`) |
| :1487 | `pub unsafe fn dot_q4_k_4rows_r4_x2_q8k_smmla(packed, x0…, x1…) -> [[f32;2];4]` |
| :2315 | `pub unsafe fn dot_q6_k_4rows_r4_x2_smmla(packed, x0_i8,x0_scales, x1_i8,x1_scales) -> [[f32;2];4]` (no sums — Q6_K has no min term) |
| :2446 | `pub unsafe fn dot_q6_k_4rows_r4_x2_q8k_smmla(packed, x0_i8,x0_scales, x1_i8,x1_scales) -> [[f32;2];4]` |

### 1e. x86_64 AVX2+FMA
| line | signature | intrinsics |
|---|---|---|
| :543 | `#[cfg(target_arch="x86_64")] #[target_feature(enable="avx2,fma")] unsafe fn dot_q8_0_row_avx2(row_blocks,&x)->f32` | `_mm256_setzero_ps, _mm_loadu_si32, _mm256_cvtepi8_epi32, _mm256_loadu_ps, _mm256_cvtepi32_ps, _mm256_fmadd_ps`; hsum = `_mm256_castps256_ps128, _mm256_extractf128_ps, _mm_add_ps, _mm_movehdup_ps, _mm_add_ss, _mm_movehl_ps, _mm_cvtss_f32` |

---

## 2. RUNTIME DISPATCH TABLE (the thing the port must reproduce exactly)

### 2.1 Leaf dispatch inside `quant.rs` (compile-time `cfg`, not runtime)
```
dot_q4_0_block_f32   (:93)   : aarch64 -> dot_q4_0_block_neon     | else dot_q4_0_block_scalar
dot_q8_0_block_f32   (:242)  : aarch64 -> dot_q8_0_block_neon     | else dot_q8_0_block_scalar
dot_q4_k_row_f32     (:616)  : aarch64 -> dot_q4_k_row_f32_neon   | else dot_q4_k_row_f32_scalar
dot_q5_k_row_f32     (:1617) : aarch64 -> dot_q5_k_row_f32_neon   | else dot_q5_k_row_f32_scalar
dot_q6_k_row_f32     (:2564) : aarch64 -> dot_q6_k_row_f32_neon   | else dot_q6_k_row_f32_scalar
dot_q8_0_row_f32     (:523)  : if x86_64 && is_x86_feature_detected!("avx2") && ...!("fma")
                                   -> dot_q8_0_row_avx2
                               else per-block loop over dot_q8_0_block_f32   // NEON or scalar
dot_f32_fast (matmul.rs)     : aarch64 (:225) -> dot_f32_neon_fast (unconditional, no detect)
                               x86_64 (:231)  -> avx2&&fma ? dot_f32_avx2 : scalar iterator sum
                               other  (:240)  -> scalar iterator sum
dot_f32_x_f16 (matmul.rs)    : aarch64 (:303) -> dot_f32_x_f16_neon | else (:310) scalar half::f16
```

### 2.2 `matmul_nt(x, w)` — `matmul.rs:114`
```
crate::thermal::tick();                       // matmul.rs:131  (rate-limited, see §6b)
match w.dtype() {                             // matmul.rs:135-144
  Q4_0    -> matmul_nt_q4_0    (:550)
  Q8_0    -> matmul_nt_q8_0    (:575)
  Q4_K    -> matmul_nt_q4_k    (:862)
  Q4_K_R4 -> matmul_nt_q4_k_r4 (:711)
  Q5_K    -> matmul_nt_q5_k    (:945)
  Q6_K    -> matmul_nt_q6_k    (:1111)
  Q6_K_R4 -> matmul_nt_q6_k_r4 (:971)
  _       -> matmul_nt_float   (:317)
}
```
Shape guard first (`:117-127`): both 2-D, `k == k2`, else `ShapeMismatch`.

### 2.3 `matmul_nt_float` — `matmul.rs:317`
```
if m == 1 && k >= 64 && w.dtype() == F16:                       // :323
    view w bytes as &[u16]; out = vec![0; n]
    chunk = gemv_chunk(n); for_each_out_chunk(out, chunk, |ci,cs| slot = dot_f32_x_f16(x, w_row))
    return [1,n]
if m == 1 && k >= 512:                                          // :348   (F32/BF16 after to_f32_cow)
    chunk = gemv_chunk(n); for_each_out_chunk(...) with dot_f32_fast
else:                                                           // :357  batched SGEMM
    flops = m*k*n
    mblock = (m >= 2 && flops >= (1<<20)) ? max(ceil(m / max(rayon_threads,1)), 4) : m   // :364
    out.par_chunks_mut(mblock*n).enumerate().for_each(|(bi, blk)|
        sgemm(mc=blk.len()/n, k, n, 1.0,
              x_ptr + bi*mblock*k, rsa=k, csa=1,
              w_ptr,               rsb=1, csb=k,      // W read transposed via strides
              0.0, blk_ptr, rsc=n, csc=1))            // :375-391
```

### 2.4 `matmul_nt_q4_0` — `matmul.rs:550`
```
guard k % 32 == 0 else internal error "Q4_0 matmul_nt: k must be a multiple of the block size (32)"
row_bytes = k/32*18
for i in 0..m:  gemv_parallel!(out[i*n..], n, row_bytes, w_blocks, x_row, quant::dot_q4_0_row_f32)
```
`gemv_parallel!` (`:538-548`) = `chunk = gemv_chunk(n); for_each_out_chunk(slice, chunk, |ci,cs| for local: j = ci*chunk+local; slot = dot(w_row_j, x_row))`.

### 2.5 `matmul_nt_q8_0` — `matmul.rs:575`
```
guard k % 32 == 0; row_bytes = k/32*34
#[cfg(aarch64)] if is_aarch64_feature_detected!("dotprod"):              // :599
    if m >= 8:                                                            // :608  blocked W8A8 GEMM
        bpr = k/32
        x_i8[m*k], x_scales[m*bpr] filled by: x_i8.par_chunks_mut(k).zip(x_scales.par_chunks_mut(bpr))
                                              .enumerate().for_each(quantize_row_to_i8_blocks)   // :612
        out_t[n*m]; wchunk = gemv_chunk(n)
        out_t.par_chunks_mut(wchunk*m).enumerate().for_each(|(ci,oc)|     // :624
            for jl,orow in oc.chunks_mut(m): j = ci*wchunk+jl
                for i,slot in orow: slot = dot_q8_0_row_sdot(w_row_j, x_i8[i], x_scales[i]))
        transpose [n,m] -> [m,n] via out.par_chunks_mut(n)                // :644
        return
    for i in 0..m:                                                        // :652 per-row GEMV
        (x_i8, x_scales) = quantize_row_to_i8_blocks(x_row)
        chunk = gemv_chunk(n); for_each_out_chunk(out_row, chunk, dot_q8_0_row_sdot)
    return
// fallback (non-aarch64, or aarch64 without dotprod):                     // :677
for i in 0..m: gemv_parallel!(..., quant::dot_q8_0_row_f32)   // -> AVX2 on x86 w/ avx2+fma, else NEON/scalar
```

### 2.6 `matmul_nt_q4_k` — `matmul.rs:862`
```
guard k % 256 == 0; row_bytes = k/256*144
#[cfg(aarch64)] if is_aarch64_feature_detected!("dotprod"):              // :879
    q8k = q8k_activations()                                              // env, see 2.10
    for i in 0..m:
        (x_i8, x_scales, x_sums) = q8k ? quantize_row_to_q8k(x_row)
                                       : { (q,s)=quantize_row_to_i8_blocks; (q,s,i8_block_sums(q)) }
        chunk = gemv_chunk(n)
        for_each_out_chunk(out_row, chunk, |ci,cs|:
            local=0
            while local+4 <= cs.len():                                    // :898 multi-row 4
                rows = w rows [j..j+4]
                v = q8k ? dot_q4_k_4rows_q8k_neon(rows,…) : dot_q4_k_4rows_q8_neon(rows,…)
                cs[local..local+4] = v; local += 4
            for remaining slots:                                          // :913 single-row
                slot = q8k ? dot_q4_k_row_q8k_neon(row,…) : dot_q4_k_row_q8_neon(row,…))
    return
for i in 0..m: gemv_parallel!(..., quant::dot_q4_k_row_f32)              // :931
```

### 2.7 `matmul_nt_q4_k_r4` — `matmul.rs:711`
```
guard k % 256 == 0 && n % 4 == 0; row_bytes = k/256*144; group_bytes = 4*row_bytes
#[cfg(aarch64)] if m >= 2 && is_aarch64_feature_detected!("i8mm"):       // :731  PREFILL
    q8k = q8k_activations()
    quantized[i] = q8k ? quantize_row_to_q8k(row) : (i8_blocks + i8_block_sums)   // :733-744
    out_t[n*m]; out_t.par_chunks_mut(4*m).enumerate().for_each(|(g,chunk)|        // :748
        group = w_blocks[g*group_bytes ..]
        xi = 0
        while xi+2 <= m:  v = q8k ? dot_q4_k_4rows_r4_x2_q8k_smmla(group,x0,s0,b0,x1,s1,b1)
                                  : dot_q4_k_4rows_r4_x2_smmla(...)
                          chunk[r*m+xi]=v[r][0]; chunk[r*m+xi+1]=v[r][1]; xi+=2
        if xi < m:        v = q8k ? dot_q4_k_4rows_r4_q8k_neon(group,x0,s0,b0)
                                  : dot_q4_k_4rows_r4_neon(...)
                          chunk[r*m+xi]=v[r])
    de-transpose out[i*n + g*4 + r] = out_t[(g*4+r)*m + i]                        // :793
    return
#[cfg(aarch64)] if is_aarch64_feature_detected!("dotprod"):              // :804  DECODE
    q8k = q8k_activations(); per activation row:
    groups = n/4;  gchunk = max(gemv_chunk(n)/4, 1)                       // :816
    for_each_out_chunk(out_row, gchunk*4, |ci,cs|:                        // :817  chunk is in ELEMENTS
        g0 = ci*gchunk
        for gl, slots in cs.chunks_mut(4): g = g0+gl
            v = q8k ? dot_q4_k_4rows_r4_q8k_neon(group,…) : dot_q4_k_4rows_r4_neon(group,…)
            slots.copy_from_slice(&v[..slots.len()]))
    return
// portable fallback (tests / x86):                                       // :841
per row: (x_i8,x_scales) = quantize_row_to_i8_blocks; x_sums = i8_block_sums
de-interleave each group's 4 rows into row_buf, then dot_q4_k_row_q8_scalar   // NOTE: per-32, never q8k
```

### 2.8 `matmul_nt_q6_k` — `matmul.rs:1111`
```
guard k % 256 == 0; row_bytes = k/256*210
#[cfg(aarch64)] if is_aarch64_feature_detected!("dotprod"):              // :1126
    q8k = q8k_activations()
    per row: (x_i8,x_scales) = q8k ? drop-sums(quantize_row_to_q8k) : quantize_row_to_i8_blocks
    chunk = gemv_chunk(n)
    for_each_out_chunk(out_row, chunk, |ci,cs| for local: j = ci*chunk+local;
        slot = q8k ? dot_q6_k_row_q8k_neon(row,…) : dot_q6_k_row_q8_neon(row,…))
    return
for i in 0..m: gemv_parallel!(..., quant::dot_q6_k_row_f32)              // :1155
```

### 2.9 `matmul_nt_q6_k_r4` — `matmul.rs:971`  /  `matmul_nt_q5_k` — `matmul.rs:945`
```
Q6_K_R4: guard k%256==0 && n%4==0; group_bytes = 4*row_bytes
#[cfg(aarch64)] if m >= 2 && i8mm:                                        // :987 PREFILL
    quantized[i] = q8k ? (q,s) from quantize_row_to_q8k (sums dropped) : quantize_row_to_i8_blocks
    out_t.par_chunks_mut(4*m).enumerate(): pairs -> dot_q6_k_4rows_r4_x2_q8k_smmla / _x2_smmla
                                           odd tail -> dot_q6_k_4rows_r4_q8k_neon / _q8_neon
    de-transpose identical to Q4_K_R4; return
#[cfg(aarch64)] { dotprod = is_aarch64_feature_detected!("dotprod");      // :1051 DECODE
    per row: quantized = dotprod.then(q8k ? q8k-drop-sums : per-32)
    gchunk = max(gemv_chunk(n)/4,1); for_each_out_chunk(out_row, gchunk*4, |ci,cs|
        per 4-slot group: Some -> q8k ? dot_q6_k_4rows_r4_q8k_neon : dot_q6_k_4rows_r4_q8_neon
                          None -> dot_q6_k_4rows_r4_neon(group, x_row)    // f32 activations!
    ); return }
portable fallback (#[allow(unreachable_code)] :1090): de-interleave -> dot_q6_k_row_f32

Q5_K (:945): NO SIMD/dotprod branch at all — always gemv_parallel!(dot_q5_k_row_f32)
```

### 2.10 Env-var knobs and defaults (complete list for this subtree)
| var | read at | parse | default |
|---|---|---|---|
| `SAPIENT_Q8K_ACT` | `matmul.rs:463` (`q8k_activations`, `OnceLock`, aarch64 only) | `v != "0"` | **true** (Q8_K activations ON) |
| `SAPIENT_GEMV_TPC` | `matmul.rs:440` (read **every call**, not cached) | `usize`, filtered `>= 1` | unset → `(n/(ncpus*4)).clamp(16,512)`; set → `(n/(ncpus*tpc)).max(16)` (no 512 cap) |
| `SAPIENT_SPINPOOL_DEBUG` | `matmul.rs:494` (`is_ok()`, every call) | presence | off; prints census every 2000 dispatches |
| `SAPIENT_SPINPOOL` | `spinpool.rs:374` (`OnceLock`) | `v != "0"` | macOS **on**; linux+aarch64 && `rayon::current_num_threads() >= 8` **on**; else off |
| `SAPIENT_SPINPOOL_WORKERS` | `spinpool.rs:341` (`OnceLock` in `pool()`) | `usize` | `rayon::current_num_threads().saturating_sub(1)` |
| `SAPIENT_SPINPOOL_SPINS` | `spinpool.rs:333` | `u64` | `DEFAULT_SPIN_ITERS = 4_000` (`spinpool.rs:58`) |
| `SAPIENT_SPINPOOL_BLOCK` | `spinpool.rs:354` (`OnceLock`) | `usize`, `>= 1` | macOS → 1; else `(n_chunks/(3*participants)).max(1)` |
| `SAPIENT_THERMAL` | `thermal.rs:186` (`OnceLock`) | `eq_ignore_ascii_case("off")` | enabled |
| `SAPIENT_THERMAL_PATH` | `thermal.rs:196` | path | `/sys/class/thermal` |
| `SAPIENT_THERMAL_HOT` | `thermal.rs:207` | `i64` °C | 80 |
| `SAPIENT_THERMAL_COOL` | `thermal.rs:208` | `i64` °C | 70 |
| `RAYON_NUM_THREADS` | indirect, via `rayon::current_num_threads()` | — | rayon default (logical cores) |

`SAPIENT_NO_REPACK` / `SAPIENT_REPACK_Q6K` are **not** in this crate — they gate the repack decision in `sapient-models/src/forward/common.rs`; this crate only exposes `repack_q4_k_rows4` / `repack_q6_k_rows4`.

---

## 3. Parallelism sites (exact chunk geometry)

### 3.1 `gemv_chunk(n)` — `matmul.rs:416-449` (verbatim logic)
```
rayon_n = max(rayon::current_num_threads(), 1)
eff     = thermal::effective_threads()
if eff < rayon_n:                      // GOVERNED branch
    return max(n / max(eff,1), 16)     // exactly eff tasks, NO 512 cap, no x4
ncpus = spinpool::enabled() ? spinpool::parallelism() : rayon_n
match env SAPIENT_GEMV_TPC (usize, >=1):
    Some(tpc) => max(n / (ncpus*tpc), 16)
    None      => clamp(n / (ncpus*4), 16, 512)
```

### 3.2 `for_each_out_chunk(out, chunk, f)` — `matmul.rs:486-534`
```
if out.is_empty(): return
if env SAPIENT_SPINPOOL_DEBUG set: bump SPIN/RAYON counters, eprintln every 2000   // :494-516
if spinpool::enabled():                                                            // :517
    len = out.len(); n_chunks = len.div_ceil(chunk)
    spinpool::pool().run(n_chunks, &|ci| {
        start = ci*chunk; end = min(start+chunk, len);
        f(ci, &mut out[start..end]) })                                             // raw ptr + SyncPtr
else:
    out.par_chunks_mut(chunk).enumerate().for_each(|(ci,cs)| f(ci,cs))             // :530
```
Both branches produce the **same** `(ci, [start,end))` partition → identical output rows per task. `SyncPtr` (`matmul.rs:473-479`) is the `*mut f32` wrapper with `unsafe impl Sync`.

### 3.3 All `rayon::` / `pool().run` sites
| site | geometry |
|---|---|
| `matmul.rs:369` `out.par_chunks_mut(mblock*n)` | float prefill SGEMM row-blocks, `mblock` per §2.3 |
| `matmul.rs:521` `spinpool::pool().run(n_chunks, …)` | **the only spin-pool call site in the whole crate** |
| `matmul.rs:530` `out.par_chunks_mut(chunk)` | rayon twin of the above |
| `matmul.rs:612` `x_i8.par_chunks_mut(k).zip(x_scales.par_chunks_mut(bpr))` | Q8_0 m≥8 activation quantization, one task per activation row |
| `matmul.rs:624` `out_t.par_chunks_mut(wchunk*m)` | Q8_0 m≥8, `wchunk = gemv_chunk(n)` weight rows × m |
| `matmul.rs:644` `out.par_chunks_mut(n)` | Q8_0 m≥8 transpose, one task per output row |
| `matmul.rs:748` `out_t.par_chunks_mut(4*m)` | Q4_K_R4 SMMLA prefill: **one task per 4-row group** |
| `matmul.rs:1003` `out_t.par_chunks_mut(4*m)` | Q6_K_R4 SMMLA prefill: same |
| `matmul.rs:365`, `conv2d.rs:152` | `rayon::current_num_threads()` for block sizing |
| `attention.rs:232` `out.par_chunks_mut(head_out_size)` | one task per `(batch, head)`; `head_out_size = seq_q*head_dim` |
| `conv2d.rs:88` `col.par_chunks_mut(col_cols)` | one task per im2col row |
| `conv2d.rs:157` `gemm_out.par_chunks_mut(mblock*n2)` | conv GEMM out-channel blocks, `mblock = flops>=1<<20 ? max(ceil(m/threads),8) : m` |
| `spinpool.rs:344`, `:386`, `:389`, `thermal.rs:209`, `:221` | `rayon::current_num_threads()` only |
No `rayon::join` and no `par_iter` in these files.

---

## 4. Numerics that affect bit-identity

**Accumulation orders (per kernel, must be mirrored exactly):**
* `dot_q4_0_row_f32` (:197): f32 `acc += block_dot` sequentially over blocks; each block dot = NEON 8× `vfmaq_f32` into ONE `float32x4_t`, then `vaddvq_f32(acc) * scale` (:166-175). `vaddvq_f32` reduces lanes as `(l0+l1)+(l2+l3)` per ARM `faddp` pairing — C++ must use `vaddvq_f32`, not a hand-rolled serial lane sum.
* `dot_q8_0_block_neon` (:266): 4 groups × 2 `vfmaq_f32` into one accumulator, `vaddvq_f32 * scale`.
* `dot_q8_0_row_avx2` (:543): per-block `block_acc` (8 lanes) then `row_acc = _mm256_fmadd_ps(block_acc, scale_v, row_acc)` — scale folded via FMA, hsum ONLY at the end (:572-579). This differs structurally from the NEON path (which reduces per block) — the two are *not* bit-identical to each other; keep each.
* W4A8 per-32 kernels (`dot_q4_k_row_q8_neon` :1084): per 64-weight group, `dot_lo = vaddvq_s32(sdot(sdot(0,lo0,xlo0),lo1,xlo1))` (exact i32), then two f32 statements `acc += xs_lo*(d1*dot_lo - m1v*sum_lo); acc += xs_hi*(d2*dot_hi - m2v*sum_hi)` in that order.
* Q8_K integer-domain kernels (`dot_q4_k_row_q8k_neon` :1024): `isum += sc1*dot_lo + sc2*dot_hi`, `imin += m1*x_sums[..] + m2*x_sums[..]` over the 4 groups (i32), then ONE `acc += x_scales[b] * (d*isum - dmin*imin)` per 256.
* Q6_K W6A8 (:1809): per 16-element group `acc += d * sc_i8 * xs * dot` — four f32 multiplies, group order `(sc_off 0, x+0), (2, +32), (4, +64), (6, +96)`, outer `l0 ∈ {0,16}`, outer-outer 2 half-blocks.
* Q6_K Q8_K (:1932): `isum += sc_i8 * dot` (i32) across the whole super-block, then `acc += x_scales[bi] * d * isum`.
* 4-row kernels: per-row accumulators `acc[r]`, activation loaded once per group, per-row arithmetic identical to the single-row kernel — lane-for-lane bit-identical by design (gated by tests).
* SMMLA x2 kernels: dots extracted with `vgetq_lane_s32::<0..3>` in lane order `[r0·x0, r0·x1, r1·x0, r1·x1]`, then the same f32/integer combine as the single-row kernel.
* `dot_q6_k_row_f32_neon` (:2623) / `dot_q5_k_row_f32_neon` (:1678): accumulate into ONE `float32x4_t` for the **entire row** (across all super-blocks), single `vaddvq_f32` at the very end — different from the scalar reference's sequential adds (tests assert closeness, not bit-equality, for these two).
* `dot_q6_k_4rows_r4_neon` (:2208): four row accumulators `accv[r]` carried across blocks, `vaddvq_f32` per row at the end.
* `dot_f32_neon_fast` (:157): 16-wide unroll, 4 `vfmaq_f32` into one accumulator, then 4-wide tail, then `vaddvq_f32`, then **scalar tail added after the horizontal reduction** (:185-189). Same shape for AVX2 (:214-218).
* `flash_attn_row` (`attention.rs:98`): online softmax, `m_new = max(s,m)`, `p = exp(s-m_new)`, `correction = exp(m-m_new)`, `o = correction*o + p*v` via `saxpby_neon` (`vfmaq_f32(vmulq_f32(vo,va), vv, vb)` :66), `l = correction*l + p`, final `*= 1/l` (or `1/f32::EPSILON` when `l==0`, :157-161). Fully-masked (`s == -inf`) positions `continue` (:134) — the NaN guard.

**Quantization rounding (exact):**
* `nibble` (:67): `(scaled + 8.5) as i32` → C++ `static_cast<int32_t>` (truncation toward zero), then clamp [0,15].
* Q8_0 weights (:232) / per-32 activations (:327) / Q8_K activations (:377): `(v * inv).round().clamp(-127.0,127.0) as i8`. Rust `f32::round` = **round half away from zero** → C++ `std::roundf`, not `rintf`/`nearbyint`.
* `max_abs` via `.fold(0.0f32, f32::max)` (:225, :323, :373) — `f32::max` returns the non-NaN operand; a NaN in the row yields `0.0`-seeded propagation semantics that differ from `std::max`. Use `fmaxf`.
* Zero-row guards differ: `quantize_q8_0_block` uses `scale = max_abs/127` and `inv = scale>0 ? 1/scale : 0`; the activation quantizers use `scale = max_abs>0 ? max_abs/127 : 1.0` (so a zero row gets scale 1.0, not 0.0).
* Q4_0 `d = vmax / -8.0` (sign preserved, d may be negative, :53).
* `i8_block_sums` / in-kernel `x_sums`: plain i32 sums; Q4_K's `imin` uses these precomputed sums, never a re-reduction (`vaddlvq_s8` was removed).

---

## 5. Data layouts (as the code reads them)

```
Q4_0  18 B / 32 w :  [0..2)  d   (f16 LE)
                     [2..18) qs[16]; byte j: lo nibble -> elem j, hi nibble -> elem j+16
                     value = (nib - 8) * d                                  (quant.rs:74-85,104-114)

Q8_0  34 B / 32 w :  [0..2)  d (f16 LE); [2..34) 32 × i8;  value = qi * d   (quant.rs:253-259)

Q4_K 144 B / 256 w:  [0..2)   d    (f16 LE)
                     [2..4)   dmin (f16 LE)
                     [4..16)  scales[12]  (8 × 6-bit (sc,min) pairs)
                     [16..144) qs[128]
   iteration (all Q4_K kernels, quant.rs:635-649 / 1101-1136):
     x_off=0; q_off=0; is=0
     repeat 4 times (QK_K/64):
        (sc1,m1)=get_scale_min_k4(is,   scales)
        (sc2,m2)=get_scale_min_k4(is+1, scales)
        lo nibbles qs[q_off + l] & 0x0F  (l=0..32) pair with x[x_off + l]
        hi nibbles qs[q_off + l] >> 4                 pair with x[x_off + 32 + l]
        value_lo = d*sc1*nib - dmin*m1 ;  value_hi = d*sc2*nib - dmin*m2
        x_off += 64; q_off += 32; is += 2

   get_scale_min_k4(j, s)  (quant.rs:596-605):
     if j < 4 : return ( s[j] & 63 ,  s[j+4] & 63 )
     else     : return ( (s[j+4] & 0x0F) | ((s[j-4] >> 6) << 4),
                         (s[j+4] >> 4)   | ((s[j]   >> 6) << 4) )

Q5_K 176 B / 256 w:  [0..2) d, [2..4) dmin, [4..16) scales[12],
                     [16..48)  qh[32]   (per-ELEMENT 5th bit),
                     [48..176) ql[128]
   iteration (quant.rs:1635-1667): u1=1, u2=2 at block start
     repeat 4 times:
        hi1 = (qh[l] & u1) ? 16 : 0        // l = 0..32, PER ELEMENT
        hi2 = (qh[l] & u2) ? 16 : 0
        acc += (d*sc1*((ql[ql_off+l] & 0x0F) + hi1) - dmin*m1) * x[x_off+l]
        acc += (d*sc2*((ql[ql_off+l] >> 4)   + hi2) - dmin*m2) * x[x_off+l+32]
        x_off += 64; ql_off += 32; is += 2
        if is % 8 == 0 { u1=1; u2=2 } else { u1 <<= 2; u2 <<= 2 }

Q6_K 210 B / 256 w:  [0..128)   ql[128]
                     [128..192) qh[64]
                     [192..208) scales[16]  (SIGNED i8, one per 16 weights)
                     [208..210) d (f16 LE)
   iteration (quant.rs:1768-1795 / 2589-2610): ql_off=0; qh_off=0; sc_base=0
     repeat 2 times (QK_K/128):
       for l0 in {0, 16}:  is = l0/16
         for (sub, sc_off, x_add) in [(0,0,0), (1,2,32), (2,4,64), (3,6,96)]:
            for l in 0..16: b = l0 + l
              q = sub==0 ? (ql[ql_off+b]    & 0x0F) | (( qh[qh_off+b]       & 3) << 4)
                : sub==1 ? (ql[ql_off+b+32] & 0x0F) | (((qh[qh_off+b] >> 2) & 3) << 4)
                : sub==2 ? (ql[ql_off+b]    >> 4)   | (((qh[qh_off+b] >> 4) & 3) << 4)
                :          (ql[ql_off+b+32] >> 4)   | (((qh[qh_off+b] >> 6) & 3) << 4)
              value = d * (int8)scales[sc_base + is + sc_off] * (q - 32)
              paired with x[x_off + x_add + l0 + l]
       x_off += 128; ql_off += 64; qh_off += 32; sc_base += 8
   (the +0/+2/+4/+6 scale offsets and the l0 split are the historical token-salad bug;
    test q6_k_scale_indexing_matches_ggml pins it, expected 1920 vs buggy 896)

R4 repack (quant.rs:1229-1247 Q4_K, :2000-2018 Q6_K — identical shape):
   nb = k/256; BB = 144 (Q4_K) or 210 (Q6_K); rows n, groups g = n/4
   for g in 0..n/4: for b in 0..nb: for r in 0..4:
       src_block_index = (g*4 + r)*nb + b
       dst_block_index = g*4*nb + b*4 + r
   i.e. within a group the super-blocks are BLOCK-MAJOR: [r0.b0, r1.b0, r2.b0, r3.b0, r0.b1, …]
   Pure permutation of whole blocks, same total size. Kernels index row r of block b at
   packed[b*4*BB + r*BB .. +BB]  (quant.rs:1271-1278, 2039-2062).
```

---

## 6. `spinpool.rs` and `thermal.rs` internals

### 6a. `spinpool.rs` (541 lines)
* `const DEFAULT_SPIN_ITERS: u64 = 4_000` (:58).
* `struct OpSlot { call: unsafe fn(*const (), usize), ctx: *const (), n_chunks: usize, block: usize }` (:60-67); `Default` = no-op fn, null ctx, 0, 1 (:69-79).
* `#[repr(align(128))] struct Pad<T>(T)` (:87-88) — **128-byte** alignment, mandatory (unpadded measured ~2× decode regression).
* `pub struct SpinPool` (:90-115): `generation: Pad<AtomicU64>`, `op: UnsafeCell<OpSlot>`, `next_block: Pad<AtomicUsize>`, `completed: Pad<AtomicUsize>`, `active: Pad<AtomicUsize>`, `publish: Mutex<()>`, `sleep: Mutex<()>`, `wake: Condvar`, `parked: Pad<AtomicUsize>`, `workers: usize`, `spin_iters: u64`. `unsafe impl Sync` (:120). **std `Mutex`/`Condvar`, not parking_lot, not futex** (`parking_lot` is used only in `pool.rs:11`).
* `fn new(workers, spin_iters) -> &'static SpinPool` (:123): `Box::leak`, then spawns `workers` threads named `sapient-spin-{w}` running `worker_loop`.
* `fn execute_blocks(&self, op:&OpSlot)` (:152-167): loop `b = next_block.fetch_add(1, Relaxed)`; `lo = b*op.block`; break if `lo >= n_chunks`; `hi = min(lo+block, n_chunks)`; call `op.call(ctx, c)` for `c in lo..hi`; `completed.fetch_add(1, Release)` — **one increment per BLOCK**.
* `fn worker_loop(&self)` (:169-227):
  * macOS: `libc::pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)` once at thread start (:176-179).
  * spin: `g = generation.load(Acquire)`; accept when `g % 2 == 0 && g != seen`; `spins > spin_iters` → lock `sleep`, `parked.fetch_add(1, SeqCst)`, `wake.wait` loop, `parked.fetch_sub(1, SeqCst)`, re-load gen; else `std::hint::spin_loop()`.
  * join: `active.fetch_add(1, SeqCst)` **FIRST**, then `generation.load(SeqCst) != g` → back out (`active.fetch_sub(1, SeqCst)`, continue); else copy `*op.get()`, `execute_blocks`, `active.fetch_sub(1, SeqCst)`, `seen = g`.
* `pub fn run<F: Fn(usize)+Sync>(&self, n_chunks, f:&F)` (:231-324):
  * `thunk::<F>` trampoline (:232-235). `n_chunks == 0` → return. `n_chunks == 1 || workers == 0` → run serially on the caller (:239-244).
  * macOS: publisher QoS-pins itself once per thread via a `thread_local! Cell<bool> QOS_PINNED` (:251-267).
  * `let _g = publish.lock()` (:269) — serializes publishers.
  * **Seqlock order (load-bearing, a SIGSEGV fix):** `generation.fetch_add(1, SeqCst)` → ODD **before** `while active.load(SeqCst) != 0 { spin_loop() }` (:278-281).
  * `participants = workers + 1` (:285); `block = block_size_override().unwrap_or(if cfg!(macos) {1} else {(n_chunks/(3*participants)).max(1)})` (:293-299); `n_blocks = n_chunks.div_ceil(block)`.
  * write slot, `next_block.store(0, Relaxed)`, `completed.store(0, Relaxed)`, `generation.fetch_add(1, SeqCst)` → EVEN (:307-312); if `parked.load(SeqCst) > 0` lock `sleep` + `wake.notify_all()` (:313-316).
  * publisher participates: `execute_blocks(&op)`, then `while completed.load(Acquire) < n_blocks { spin_loop() }` (:319-323).
* `pub fn pool() -> &'static SpinPool` (:330) — `OnceLock`, spins/workers from env (see §2.10).
* `fn block_size_override() -> Option<usize>` (:351), `pub fn parallelism() -> usize = workers + 1` (:363).
* `pub fn enabled() -> bool` (:371-390): `ON` (`OnceLock`, env/platform default) **&&** `crate::thermal::effective_threads() >= rayon::current_num_threads()` — i.e. the pool is disabled whenever the thermal governor or an external level has shed a single core.

### 6b. `thermal.rs` (347 lines)
* `TICK_MS = 500` (:43), `DEFAULT_HOT_C = 80` (:45), `DEFAULT_COOL_C = 70` (:47).
* `pub struct ThermalGovernor { zones: Vec<PathBuf>, hot_mc: i64, cool_mc: i64, max_threads: usize, min_threads: usize, effective: AtomicUsize, warned: AtomicBool }` (:52-61).
* `pub fn new(root:&Path, hot_c, cool_c, max_threads) -> Self` (:66-90): `read_dir(root)`, entries whose file name starts with `"thermal_zone"`, take `<entry>/temp` if `is_file()`, **`zones.sort()`**; `hot_mc = hot_c*1000`; `min_threads = (max_threads/2).max(1)`; `effective = max_threads`.
* `pub fn is_active(&self) -> bool` (:93) = `!zones.is_empty()`.
* `pub fn max_temp_mc(&self) -> Option<i64>` (:98): read each file, `trim().parse::<i64>()`, `.max()` — unreadable/unparsable zones silently skipped.
* `pub fn effective(&self) -> usize` (:107) — `Relaxed`.
* `pub fn sample(&self) -> usize` (:114-139): `None` temp → return current; `t >= hot_mc && cur > min_threads` → `cur-1` (store Relaxed) + one-shot `tracing::warn!` guarded by `warned.swap(true, Relaxed)`; `t <= cool_mc && cur < max_threads` → `cur+1`; else hold (hysteresis band).
* External override: `static EXTERNAL_LEVEL: AtomicU8` (:145); `fn external_cap(level, max)` (:151-158) = `0→max`, `1→(max*3/4).max(1)`, `2→(max/2).max(1)`, `_→(max/4).max(1)`; `pub fn set_external_thermal_level(level: u8)` (:162) no-ops when disabled, clamps to 3, `swap(Relaxed)`, logs on change; `pub fn external_thermal_level() -> u8` (:177).
* `fn thermal_disabled()` (:183) — `OnceLock`, `SAPIENT_THERMAL` equals "off" case-insensitively.
* `fn governor() -> Option<&'static ThermalGovernor>` (:190-214) — `OnceLock<Option<…>>`; `None` if disabled or no zones; built with `rayon::current_num_threads().max(1)`.
* `pub fn effective_threads() -> usize` (:220-227): `max = rayon threads`; `base = governor().map(effective).unwrap_or(max)`; return `min(base, external_cap(EXTERNAL_LEVEL, max))` — **stricter source wins**.
* `pub fn tick()` (:232-247): return if no governor; `EPOCH: OnceLock<Instant>`, `LAST_MS: AtomicU64`; skip if `now_ms - last < 500`; else `compare_exchange(last, now_ms, Relaxed, Relaxed)` and only the winner calls `sample()`.

---

## 7. Tests (counts confirmed; `#[ignore]` and gates noted)

| file | count | names (line) | gates |
|---|---|---|---|
| quant.rs | **24** | `q6_k_scale_indexing_matches_ggml`(2715), `corruption_magnitude_report`(2891), `sdot_q8_0_row_blockwise_survives_activation_outlier`(2993), `sdot_q8_0_block_matches_scalar_integer_dot`(3036), `q4_0_on_the_fly_dot_matches_dequantized_reference`(3074), `q4_0_quantization_error_is_bounded`(3100), `q4_k_w4a8_matches_f32_path`(3119), `q4_k_4rows_matches_single_row`(3170), `q4_k_r4_repack_roundtrips_through_dequant`(3219), `q4_k_r4_kernel_matches_single_row`(3246), `q6_k_r4_repack_roundtrips_through_dequant`(3286), `q6_k_r4_kernel_matches_single_row`(3311), `q6_k_w6a8_neon_matches_scalar`(3358), `q6_k_w6a8_r4_matches_single_row`(3379), `q6_k_w6a8_close_to_f32_path`(3401), `q4_k_smmla_x2_matches_single_row`(3419), `q6_k_smmla_x2_matches_single_row`(3473), `q6_k_neon_matches_scalar`(3510), `q5_k_neon_matches_scalar`(3542), `q4_k_q8k_scalar_matches_f32_path`(3572), `q4_k_r4_q8k_kernels_match_single_row`(3626), `q4_k_plain_4rows_q8k_matches_single_row`(3674), `q6_k_q8k_scalar_matches_f32_path`(3701), `q6_k_r4_q8k_kernels_match_single_row`(3742) | 11 are `#[cfg(target_arch="aarch64")]` (3169/3245/3310/3357/3378/3418/3472/3625/3673/3741 + helpers); several *additionally* early-`return` at runtime when `dotprod`/`i8mm` are absent (e.g. 2994, 3037, 3171, 3247, 3359, 3380, 3420, 3474, 3743). No env vars, no `#[ignore]`. Helpers: `seq`(2737), `lcg_bytes`(2763), `rand_x`(2775), `rand_q6_k_block`(2788), `rand_q5_k_block`(2794), `dot_q6_k_buggy`(2803), `dot_q5_k_buggy`(2840), `rel_err`(2878), `stats`(2882), `q8_0_weight_row`(2978), `q6_k_test_rows`(3339) |
| matmul.rs | **10** | `matmul_2x2`(1260), `matmul_nt_q4_0_matches_float`(1273), `q8_0_gemm_path_matches_per_row_path`(1312), `matmul_nt_q8_0_matches_float`(1355), `matmul_nt_q8_0_gguf_dimflip_matches_float`(1397), `matmul_nt_linear`(1440), `matmul_nt_linear_f16_weight`(1454), `matmul_rank_mismatch`(1470), `gemm_with_bias`(1477), `q4_k_r4_prefill_matches_per_row`(1495) | 1312 & 1495 are aarch64-only; 1312 also runtime-skips without dotprod. 1312 and 1495 assert **`to_bits()` equality** (true bit-identity gates). No env vars |
| attention.rs | **7** | `mha_output_shape`(312), `gqa_kv_repeat`(322), `causal_mask_shape`(332), `leading_masked_positions_do_not_nan`(346), `uniform_attention_recovers_v`(385), `flash_matches_naive`(419), `decode_mode_no_nan`(501) | none |
| elementwise.rs | **7** | `test_add`, `test_relu`, `test_sigmoid`, `test_gelu`, `test_erf`, `test_gelu_erf`, `test_scalar_broadcast` (185-226) | none |
| thermal.rs | **6** | `no_zones_is_inert`(272), `hot_steps_down_to_floor_and_cool_restores`(280), `hysteresis_holds_between_thresholds`(296), `hottest_zone_wins`(307), `external_cap_mapping`(317), `external_level_caps_effective_threads`(334) | The four governor tests build a **fake sysfs root directly** via `ThermalGovernor::new(&tmpdir, …)` — they do **NOT** set `SAPIENT_THERMAL_PATH` (that var only feeds the process-global singleton). `tmp()`(261) = `env::temp_dir()/sapient-thermal-test-{pid}-{name}`, removed+recreated per test. `external_level_caps_effective_threads` mutates process-global `EXTERNAL_LEVEL` and reads `rayon::current_num_threads()` — must stay the only test touching `effective_threads()` |
| spinpool.rs | **5** | `runs_every_chunk_exactly_once`(398), `concurrent_publishers_serialize`(413), `survives_park_and_wake`(431), `pool_speedup_probe`(461) **`#[ignore]`** (mod `perf_probe`, :451), `rapid_ops_with_constant_parking`(523, mod `stress`, :514 — builds its own `SpinPool::new(4, 50)`) | the 3 basic tests use the global `pool()`; `survives_park_and_wake` sleeps 300 ms to force parking |
| rope.rs | **4** | `rope_output_shape`(160), `rope_partial_leaves_tail_unchanged`(168), `rope_partial_full_matches_apply_rope`(184), `rope_position_zero_is_identity`(196) | none |
| softmax.rs | **3** | `softmax_sums_to_one`(92), `softmax_stable_large`(100), `log_softmax_finite`(113) | none |
| reduce.rs | **2** | `sum_all`(126), `mean_axis0`(133) | none |
| layernorm.rs | **2** | `layernorm_zero_mean_unit_var`(114), `rmsnorm_identity_weight`(126) | none |
| conv2d.rs | **1** | `conv2d_identity_kernel`(209) | none |
Total 71; exactly one `#[ignore]` (`spinpool::perf_probe::pool_speedup_probe`).

---

## 8. Remaining per-file inventories (small kernels) + dependencies

### `kernels/attention.rs`
* `:29 #[cfg(aarch64)] #[target_feature(enable="neon")] unsafe fn dot_f32_neon(a,b)->f32` (4-wide `vfmaq_f32`, `vaddvq_f32`, scalar tail) / `:50 #[cfg(not(aarch64))] fn dot_f32_neon` scalar.
* `:57 unsafe fn saxpby_neon(o:&mut[f32], v:&[f32], alpha, beta)` (`vdupq_n_f32, vld1q_f32, vmulq_f32, vfmaq_f32, vst1q_f32`) / `:78` scalar twin.
* `:98 #[inline(always)] fn flash_attn_row(q_row,k_head,v_head,o_row,scale,_seq_k,head_dim,attend_len,mask_row)`.
* `:181 pub fn scaled_dot_product_attention(q,k,v,mask:Option<&Tensor>,scale:Option<f32>,n_kv_heads) -> Result<Tensor>`. Decision tree: rank-4 guard (:192); `scale = scale.unwrap_or(1/sqrt(head_dim))` (:201); `kv_rep = n_heads/n_kv_heads` (:204); K/V forced contiguous via `to_contiguous_f32_vec()` (:210-211), Q via `to_f32_cow()` + strides (:214-216); `kv_offset = seq_k.saturating_sub(seq_q)` (:224); parallel over `(b,h)`; `kv_h = h/kv_rep`; per query row: q row zero-copy iff `q_strides[3]==1` else gathered (:253-260); **`attend_len = mask.is_some() ? seq_k : min(qi+kv_offset+1, seq_k)`** (:265-270) — i.e. `mask=None` means *causal*.
* `:292 pub fn causal_mask(seq_q, seq_k) -> Tensor` — `-inf` where `ki > qi + (seq_k-seq_q)`.
* No runtime feature detection; NEON is compile-time on aarch64.

### `kernels/rope.rs` (all scalar, no SIMD, no cfg)
`:18 pub fn apply_rope(x,&positions,base)`, `:71 pub fn apply_rope_partial(x,positions,base,rotary_dim)` → delegates to `:83 pub fn apply_rope_partial_scaled(x,positions,base,rotary_dim,pos_scale)`, `:140 pub fn rope_cos_sin_cache(max_seq_len,head_dim,base) -> (Vec<f32>,Vec<f32>)`.
Numerics: `freq = (pos as f32 / pos_scale) / base.powf(2.0*i as f32 / rotary_dim as f32)` (:124; `apply_rope` uses `head_dim` as the divisor, :49), `let (sin_f, cos_f) = freq.sin_cos()`, NEOX rotate-half over the first `rotary_dim` channels with `half = rotary_dim/2`: `out[i] = x0*cos - x1*sin; out[i+half] = x1*cos + x0*sin`. Guards: rank 4, even head_dim/rotary_dim, `positions.len() == seq_len`, `rotary_dim ∈ 1..=head_dim`. C++ must match `powf` and `sincosf` bit-for-bit — this is the biggest libm-portability risk in the crate.

### `kernels/elementwise.rs` (all scalar, F32 only)
Helpers `:15 fn unary_f32`, `:27 fn binary_f32` (same-len, else scalar broadcast if either side has numel 1, else `ShapeMismatch`). `unary_f32` **requires `dtype == F32`** and errors otherwise (:16); `binary_f32` does not. Public: `add, sub, mul, div, pow` (:61-75), `neg, abs, sqrt, exp, log, erf, floor, ceil, round` (:77-103), `relu`(107), `sigmoid`(111), `tanh_act`(115), `gelu`(120, tanh approx; `SQRT_2_OVER_PI = 0.797_884_56`, `COEF = 0.044_715`), `gelu_erf`(135, `0.5*x*(1+erf(x*FRAC_1_SQRT_2))`), `silu`(141), `hard_swish`(146), `leaky_relu`(150), `clip`(154). `:163 fn erf_approx(x)` — Abramowitz & Stegun 5-term rational in `t = 1/(1+0.327_591_1*|x|)` with coefficients `0.254_829_59, -0.284_496_74, 1.421_413_74, -1.453_152_03, 1.061_405_43`, times `exp(-x*x)`, sign-restored. **Port this exact polynomial — do NOT substitute `std::erff`.**

### `kernels/softmax.rs`
`:14 fn normalise_axis`, `:25 pub fn softmax(x,axis)`, `:30 pub fn log_softmax(x,axis)`, `:34 fn apply_softmax_impl(x,axis,log_mode)`. Gathers each lane into a `Vec` (outer/dim/inner decomposition, :50-52), `max` via `fold(NEG_INFINITY, f32::max)` with `-inf → 0.0` fixup (:62-65), `exps = (v-max).exp()`, `sum_e == 0.0 → f32::EPSILON` (:68-70), output `exps[d]/sum_e` or `(slice[d]-max) - sum_e.ln()`. Single-threaded.

### `kernels/reduce.rs`
`:9 fn normalise_axes` (empty axes ⇒ all), `:26 fn reduce<F>(x,axes,keep_dims,init,f)` — element-wise scatter-accumulate; note it rebuilds `Shape(out_dims).strides()` **inside the per-element loop** (:73). `:96 reduce_sum` (init 0, `+`), `:100 reduce_mean` (sum then `/count`), `:112 reduce_max` (init `-inf`, `f32::max`), `:116 reduce_min` (init `+inf`, `f32::min`). Single-threaded.

### `kernels/layernorm.rs`
`:17 pub fn layer_norm(x, weight:Option, bias:Option, axis:i64, epsilon:f32)` — `outer = Π dims[..ax]`, `norm_size = Π dims[ax..]`; two-pass mean then variance (`Σ(v-mean)²/n`), `inv_std = 1/sqrt(var+eps)`, then a 4-arm match on (w,b) presence (:59-64). `:75 pub fn rms_norm(x, weight:Option, epsilon)` — last-axis only; `rms_sq = Σv²/dim`, `inv_rms = 1/sqrt(rms_sq+eps)`, `out = slice[i]*inv_rms*w[i]`. Both single-threaded, sequential f32 sums.

### `kernels/conv2d.rs`
Public counters `:16 pub static IM2COL_NS: AtomicU64`, `:17 pub static GEMM_NS: AtomicU64` (Relaxed adds at :131 and :183). `:19 pub fn conv2d(x, weight, bias, _kernel_shape:[usize;2], pads:[t,l,b,r], strides:[2], dilations:[2], groups) -> Result<Tensor>`. `h_out = (h_in+pad_t+pad_b - dil0*(kh-1) - 1)/stride0 + 1` (:55), same for w (:56). Loops batch × group; im2col `col[col_rows=c_in_g*kh*kw][col_cols=h_out*w_out]` filled with `par_chunks_mut(col_cols)`, row decode `kj = row%kw, ki = (row/kw)%kh, ci = row/(kh*kw)` (:91-93), stride-1 fast path (:108). GEMM split over out-channel blocks, then bias added during the copy-out (:190-195).

### 8b. Dependencies to map in C++
* **sapient-core** (`matmul.rs:11-15`, `attention.rs:21-22`, others): `error::{Result, SapientError}` (variants used: `RankMismatch{expected,got}`, `ShapeMismatch{expected,got}`, `TypeMismatch{expected,got}`, `InvalidGraph(String)`, `internal(msg)`), `DType`, `Shape`, `Tensor`, and block-size constants `QUANT_BLOCK_SIZE, Q4_0_BLOCK_BYTES, Q8_0_BLOCK_BYTES, Q4_K_BLOCK_BYTES, Q5_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES` (these shadow quant.rs's own copies — keep both sets equal).
  Tensor API actually called: `shape()`, `Shape::{dims,ndim,new}`, `strides()`, `dtype()`, `to_f32_cow()`, `to_f32_vec()`, `to_contiguous_f32_vec()`, `as_f32_slice()`, `as_bytes()`, `as_quant_blocks()`, `t()`, `clone()`, `Tensor::from_f32(&[f32], shape)`, `Tensor::from_f32_vec(Vec<f32>, shape)`; tests additionally use `from_quant_bytes`, `from_f16_bytes`, `zeros`, `reshape`.
* **matrixmultiply::sgemm** — 4 call sites, all `unsafe`, signature `sgemm(m,k,n, alpha, a,*const f32, rsa:isize, csa:isize, b,*const f32, rsb, csb, beta, c:*mut f32, rsc, csc)`:
  * `matmul.rs:70` batched matmul — `1.0, a+off, k,1, b+off, n,1, 0.0, c+off, n,1`.
  * `matmul.rs:375` float prefill (per row-block) — `1.0, x+m0*k, k,1, w, 1,k (transposed via strides), 0.0, out_block, n,1`.
  * `matmul.rs:1210` `gemm()` — `alpha, a, a_strides[0], a_strides[1], b, b_strides[0], b_strides[1], 0.0, out, n, 1` (β applied only to bias, :1246).
  * `conv2d.rs:163` — `1.0, w+w_off+m0*k, k,1, col, n2,1, 0.0, out_block, n2,1`.
  C++ needs an SGEMM with the same blocking/accumulation to be bit-identical here (this is the largest non-kernel parity risk; `matmul_nt_float` prefill and conv2d both depend on it).
* **half** — `f16::{from_le_bytes, to_le_bytes, from_f32, from_bits, to_f32}`.
* **rayon** — `prelude::*` (`par_chunks_mut`, `enumerate`, `for_each`, `zip`) and `current_num_threads()` (which also reads `RAYON_NUM_THREADS`).
* **parking_lot** — NOT used in any file in scope (only `pool.rs:11`).
* **libc** — macOS only, `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)` at `spinpool.rs:178` (workers) and `:259` (publisher, once per thread via TLS flag).
* **tracing** — `warn!` (`thermal.rs:123`), `info!` (`thermal.rs:169`).

---

## 9. Porting hazards ranked (short)
1. `matrixmultiply::sgemm` replacement — bit-identity of the float prefill and conv2d paths hangs on it.
2. `rope.rs` `powf`/`sin_cos` and `elementwise.rs` `erf_approx` — libm/poly must match exactly.
3. Horizontal reductions: use `vaddvq_f32`/`vaddvq_s32` (and the exact AVX2 hsum sequence at `quant.rs:572-579`), never a serial lane sum.
4. Rounding: `roundf` (half away from zero) for activation/weight quant; **truncation** for the Q4_0 nibble.
5. Q6_K `+0/+2/+4/+6` scale indexing and Q5_K per-element `qh[l]` bit-plane — both are historical token-salad bugs with dedicated tests.
6. `for_each_out_chunk` must yield the identical `(chunk_index → [start,end))` partition on BOTH the pool and the fallback path, and `gemv_chunk`'s governed branch must compare against the **rayon** thread count (not the pool's), per `matmul.rs:418-426`.
7. spinpool seqlock: ODD bump strictly before the `active` drain; `SeqCst` on both the worker's `active.fetch_add` and its authoritative `generation` load; 128-byte padding on all four hot atomics.
