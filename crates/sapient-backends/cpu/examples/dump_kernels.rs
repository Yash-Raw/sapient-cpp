// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
//! Golden-dump generator for the Rust→C++ parity harness (TEST-ONLY; not part of the product).
//!
//! Writes one `.sapd` file per kernel case: seeded-random inputs plus the outputs the Rust
//! kernels produce ON THIS HOST. The public kernel entry points dispatch on the host ISA at
//! runtime (NEON/SDOT/SMMLA vs scalar/AVX2), so dumps are host-specific: regenerate them in the
//! same job that consumes them and never commit them (spec D1).
//!
//! Wire format v1 (little-endian):
//!   "SAPD" | u32 version=1 | u32 name_len | name | u32 n_arrays |
//!   per array: u32 name_len | name | u8 dtype | u32 ndim | u64 dims[ndim] | u64 byte_len | bytes
//!   dtype: 0=f32 1=u8 2=i8 3=i32 4=u32 5=u64.  Names are prefixed `in:` / `param:` / `out:`.
//!
//! Case families (`build_cases`): block quantizers (`quantize_q{4,8}_0_block`); row dot products
//! (`dot_q{4,8}_0_row_f32`, `dot_q{4,5,6}_k_row_f32`); `matmul_nt_{f32,q8_0,q4_k,q6_k}_m{1,3,4}`;
//! norm/activation/softmax (`rms_norm`, `softmax`, `silu`, `gelu_erf`); `apply_rope`;
//! `attention_{prefill,decode}`; sapient-core dequant: `dequant_*` (Q4_0/Q8_0/Q4_K/Q5_K/Q6_K,
//! the Q4_K_R4/Q6_K_R4 row-interleaved layouts, and F16/BF16 widening+narrowing).
//!
//! Plan C (dense kernels): `matmul_nt_{f16_m1,f32_m1_k512}` (the two GEMV paths; F16 weights as raw
//! bytes), `layer_norm`, `reduce` (four outputs), `apply_rope_partial(_scaled)`, `attention_masked`,
//! `gelu`, `softmax_axis0`, `log_softmax`, `conv2d_s{1,2}`.
//!
//! Plan D (quantized kernels): `quantize_row_to_i8_blocks`, `i8_block_sums`, `quantize_row_to_q8k`,
//! `repack_q{4,6}_k_rows4`, `matmul_nt_{q4_0_m{1,3},q5_k_m1,q8_0_m8,q4_k_r4_m{1,2,3,8},q6_k_r4_m{1,2,3,8}}`,
//! and the plan-C carry-overs `matmul_nt_f32_m1_k519`, `matmul_nt_f16_m1_k67`, `attention_decode_hd10`
//! (SIMD body + scalar tail). `--q8k-off` (run under `SAPIENT_Q8K_ACT=0`) writes ONLY the twelve
//! knob-sensitive `matmul_nt_{q4_k,q6_k,q4_k_r4,q6_k_r4}_m*` cases, renamed `*_q8k_off`; every RNG
//! draw happens exactly as in the default run so the inputs are byte-identical across the two.
//!
//! Usage:
//!   cargo run --release -p sapient-backends-cpu --example dump_kernels -- --out <dir> [--seed N] [--q8k-off]
//!   cargo run --release -p sapient-backends-cpu --example dump_kernels -- --format-sample <file>

use std::error::Error;
use std::fs;
use std::path::PathBuf;

use half::f16;
use sapient_backends_cpu::kernels::{
    attention, conv2d, elementwise, layernorm, matmul, quant, reduce, rope, softmax,
};
use sapient_core::{DType, Tensor};

const FORMAT_VERSION: u32 = 1;
const DT_F32: u8 = 0;
const DT_U8: u8 = 1;
const DT_I8: u8 = 2;
const DT_I32: u8 = 3;
const DT_U32: u8 = 4;
const DT_U64: u8 = 5;
const DEFAULT_SEED: u64 = 0x5A71_E47D_0000_0001;

const USAGE: &str =
    "usage: dump_kernels --out <dir> [--seed N] [--q8k-off] | --format-sample <file>";

/// xorshift64* — deterministic, dependency-free (same family as the sampler's RNG).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
    fn f32s(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| self.range(lo, hi)).collect()
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.next_u64() >> 56) as u8).collect()
    }
}

struct Array {
    name: String,
    dtype: u8,
    dims: Vec<u64>,
    bytes: Vec<u8>,
}

fn dtype_size(dtype: u8) -> usize {
    match dtype {
        DT_F32 | DT_I32 | DT_U32 => 4,
        DT_U8 | DT_I8 => 1,
        DT_U64 => 8,
        other => panic!("unknown dtype tag {other}"),
    }
}

impl Array {
    fn raw(name: &str, dtype: u8, dims: &[usize], bytes: Vec<u8>) -> Self {
        let numel: usize = dims.iter().product();
        assert_eq!(
            bytes.len(),
            numel * dtype_size(dtype),
            "{name}: byte length mismatch"
        );
        Array {
            name: name.to_string(),
            dtype,
            dims: dims.iter().map(|&d| d as u64).collect(),
            bytes,
        }
    }
    fn f32(name: &str, dims: &[usize], v: &[f32]) -> Self {
        Self::raw(
            name,
            DT_F32,
            dims,
            v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        )
    }
    fn u8(name: &str, dims: &[usize], v: &[u8]) -> Self {
        Self::raw(name, DT_U8, dims, v.to_vec())
    }
    fn i8(name: &str, dims: &[usize], v: &[i8]) -> Self {
        Self::raw(name, DT_I8, dims, v.iter().map(|&x| x as u8).collect())
    }
    fn i32(name: &str, dims: &[usize], v: &[i32]) -> Self {
        Self::raw(
            name,
            DT_I32,
            dims,
            v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        )
    }
    fn u32(name: &str, dims: &[usize], v: &[u32]) -> Self {
        Self::raw(
            name,
            DT_U32,
            dims,
            v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        )
    }
    fn u64(name: &str, dims: &[usize], v: &[u64]) -> Self {
        Self::raw(
            name,
            DT_U64,
            dims,
            v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        )
    }
    /// An F32 tensor (any layout) as `f32` with its shape.
    fn tensor(name: &str, t: &Tensor) -> Self {
        Self::f32(name, t.shape().dims(), &t.to_f32_vec())
    }
}

fn put_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn encode_case(name: &str, arrays: &[Array]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"SAPD");
    buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    put_str(&mut buf, name);
    buf.extend_from_slice(&(arrays.len() as u32).to_le_bytes());
    for a in arrays {
        put_str(&mut buf, &a.name);
        buf.push(a.dtype);
        buf.extend_from_slice(&(a.dims.len() as u32).to_le_bytes());
        for d in &a.dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf.extend_from_slice(&(a.bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(&a.bytes);
    }
    buf
}

// ── quantized block builders ─────────────────────────────────────────────────────────────────
// K-quant blocks are random bits with FINITE f16 scales (random f16 bits could be NaN/Inf).

fn f16_bytes(v: f32) -> [u8; 2] {
    f16::from_f32(v).to_le_bytes()
}

fn q4_k_block(rng: &mut Rng) -> Vec<u8> {
    let mut b = Vec::with_capacity(144);
    b.extend_from_slice(&f16_bytes(rng.range(0.002, 0.05))); // d
    b.extend_from_slice(&f16_bytes(rng.range(0.0, 0.02))); // dmin
    b.extend(rng.bytes(12)); // packed 6-bit scales/mins
    b.extend(rng.bytes(128)); // nibbles
    b
}

fn q5_k_block(rng: &mut Rng) -> Vec<u8> {
    let mut b = Vec::with_capacity(176);
    b.extend_from_slice(&f16_bytes(rng.range(0.002, 0.05)));
    b.extend_from_slice(&f16_bytes(rng.range(0.0, 0.02)));
    b.extend(rng.bytes(12)); // scales
    b.extend(rng.bytes(32)); // qh (5th bits)
    b.extend(rng.bytes(128)); // qs
    b
}

fn q6_k_block(rng: &mut Rng) -> Vec<u8> {
    let mut b = Vec::with_capacity(210);
    b.extend(rng.bytes(128)); // ql
    b.extend(rng.bytes(64)); // qh
    b.extend(rng.bytes(16)); // 16 signed int8 scales (any bits are valid)
    b.extend_from_slice(&f16_bytes(rng.range(0.002, 0.05))); // d
    b
}

fn q8_0_row(w: &[f32]) -> Vec<u8> {
    w.chunks(32).flat_map(quant::quantize_q8_0_block).collect()
}

fn q4_0_row(w: &[f32]) -> Vec<u8> {
    w.chunks(32).flat_map(quant::quantize_q4_0_block).collect()
}

type BlockFn = fn(&mut Rng) -> Vec<u8>;
type DotFn = fn(&[u8], &[f32]) -> f32;

fn kquant_rows(rng: &mut Rng, rows: usize, k: usize, block: BlockFn) -> Vec<u8> {
    (0..rows * k / 256).flat_map(|_| block(rng)).collect()
}

type Case = (String, Vec<Array>);

fn case(name: impl Into<String>, arrays: Vec<Array>) -> Case {
    (name.into(), arrays)
}

// ── the cases ────────────────────────────────────────────────────────────────────────────────

fn build_cases(seed: u64, q8k_off: bool) -> Result<Vec<Case>, Box<dyn Error>> {
    let mut rng = Rng::new(seed);
    let mut cases: Vec<Case> = Vec::new();
    // `--q8k-off`: the twelve SAPIENT_Q8K_ACT-sensitive matmul_nt cases get this suffix and are
    // the only ones kept (see the end of this function). Every other case is still computed so
    // the RNG sequence — and therefore every input — is identical to the default run.
    let q8k_suffix = if q8k_off { "_q8k_off" } else { "" };

    // Block quantizers.
    let x32 = rng.f32s(32, -1.0, 1.0);
    let q8 = quant::quantize_q8_0_block(&x32);
    cases.push(case(
        "quantize_q8_0_block",
        vec![
            Array::f32("in:x", &[32], &x32),
            Array::u8("out:block", &[34], &q8),
        ],
    ));
    let q4 = quant::quantize_q4_0_block(&x32);
    cases.push(case(
        "quantize_q4_0_block",
        vec![
            Array::f32("in:x", &[32], &x32),
            Array::u8("out:block", &[18], &q4),
        ],
    ));

    // Row dot products (k = 256 for the 32-wide formats, 512 = two super-blocks for K-quants).
    let w = rng.f32s(256, -1.0, 1.0);
    let x = rng.f32s(256, -1.0, 1.0);
    let row = q8_0_row(&w);
    let y = quant::dot_q8_0_row_f32(&row, &x);
    cases.push(case(
        "dot_q8_0_row_f32",
        vec![
            Array::u8("in:row_blocks", &[row.len()], &row),
            Array::f32("in:x", &[256], &x),
            Array::f32("out:y", &[1], &[y]),
        ],
    ));
    let row = q4_0_row(&w);
    let y = quant::dot_q4_0_row_f32(&row, &x);
    cases.push(case(
        "dot_q4_0_row_f32",
        vec![
            Array::u8("in:row_blocks", &[row.len()], &row),
            Array::f32("in:x", &[256], &x),
            Array::f32("out:y", &[1], &[y]),
        ],
    ));
    let xk = rng.f32s(512, -1.0, 1.0);
    let kq: [(&str, BlockFn, DotFn); 3] = [
        ("dot_q4_k_row_f32", q4_k_block, quant::dot_q4_k_row_f32),
        ("dot_q5_k_row_f32", q5_k_block, quant::dot_q5_k_row_f32),
        ("dot_q6_k_row_f32", q6_k_block, quant::dot_q6_k_row_f32),
    ];
    for (name, block, dot) in kq {
        let row = kquant_rows(&mut rng, 1, 512, block);
        let y = dot(&row, &xk);
        cases.push(case(
            name,
            vec![
                Array::u8("in:row_blocks", &[row.len()], &row),
                Array::f32("in:x", &[512], &xk),
                Array::f32("out:y", &[1], &[y]),
            ],
        ));
    }

    // matmul_nt: f32 weights (GEMV m=1 and GEMM m=4).
    let (k, n) = (64usize, 16usize);
    let wt = Tensor::from_f32(&rng.f32s(n * k, -1.0, 1.0), vec![n, k])?;
    for m in [1usize, 4] {
        let xt = Tensor::from_f32(&rng.f32s(m * k, -1.0, 1.0), vec![m, k])?;
        let y = matmul::matmul_nt(&xt, &wt)?;
        cases.push(case(
            format!("matmul_nt_f32_m{m}"),
            vec![
                Array::tensor("in:x", &xt),
                Array::tensor("in:w", &wt),
                Array::tensor("out:y", &y),
            ],
        ));
    }
    // matmul_nt: quantized weights [8, 512] (decode m=1 and prefill m=3 paths).
    let (rows, kq_len) = (8usize, 512usize);
    let wq8 = q8_0_row(&rng.f32s(rows * kq_len, -1.0, 1.0));
    let quants: [(&str, DType, Vec<u8>); 3] = [
        ("q8_0", DType::Q8_0, wq8),
        (
            "q4_k",
            DType::Q4_K,
            kquant_rows(&mut rng, rows, kq_len, q4_k_block),
        ),
        (
            "q6_k",
            DType::Q6_K,
            kquant_rows(&mut rng, rows, kq_len, q6_k_block),
        ),
    ];
    for (tag, dtype, wbytes) in quants {
        let wt = Tensor::from_quant_bytes(&wbytes, vec![rows, kq_len], dtype)?;
        for m in [1usize, 3] {
            let xt = Tensor::from_f32(&rng.f32s(m * kq_len, -1.0, 1.0), vec![m, kq_len])?;
            let y = matmul::matmul_nt(&xt, &wt)?;
            cases.push(case(
                format!(
                    "matmul_nt_{tag}_m{m}{}",
                    if tag == "q8_0" { "" } else { q8k_suffix }
                ),
                vec![
                    Array::tensor("in:x", &xt),
                    Array::u8("in:w_blocks", &[wbytes.len()], &wbytes),
                    Array::u32("param:w_shape", &[2], &[rows as u32, kq_len as u32]),
                    Array::tensor("out:y", &y),
                ],
            ));
        }
    }

    // Norm / activation / softmax.
    let xt = Tensor::from_f32(&rng.f32s(2 * 64, -2.0, 2.0), vec![2, 64])?;
    let wt = Tensor::from_f32(&rng.f32s(64, 0.5, 1.5), vec![64])?;
    let y = layernorm::rms_norm(&xt, Some(&wt), 1e-5)?;
    cases.push(case(
        "rms_norm",
        vec![
            Array::tensor("in:x", &xt),
            Array::tensor("in:weight", &wt),
            Array::f32("param:eps", &[1], &[1e-5]),
            Array::tensor("out:y", &y),
        ],
    ));
    let xt = Tensor::from_f32(&rng.f32s(2 * 3 * 8, -4.0, 4.0), vec![2, 3, 8])?;
    let y = softmax::softmax(&xt, -1)?;
    cases.push(case(
        "softmax",
        vec![
            Array::tensor("in:x", &xt),
            Array::i32("param:axis", &[1], &[-1]),
            Array::tensor("out:y", &y),
        ],
    ));
    let xt = Tensor::from_f32(&rng.f32s(64, -6.0, 6.0), vec![64])?;
    cases.push(case(
        "silu",
        vec![
            Array::tensor("in:x", &xt),
            Array::tensor("out:y", &elementwise::silu(&xt)?),
        ],
    ));
    cases.push(case(
        "gelu_erf",
        vec![
            Array::tensor("in:x", &xt),
            Array::tensor("out:y", &elementwise::gelu_erf(&xt)?),
        ],
    ));

    // RoPE on [batch=1, heads=2, seq=4, head_dim=16] at cache offset 5.
    let xt = Tensor::from_f32(&rng.f32s(2 * 4 * 16, -1.0, 1.0), vec![1, 2, 4, 16])?;
    let positions: Vec<usize> = vec![5, 6, 7, 8];
    let y = rope::apply_rope(&xt, &positions, 10_000.0)?;
    cases.push(case(
        "apply_rope",
        vec![
            Array::tensor("in:x", &xt),
            Array::u64(
                "param:positions",
                &[4],
                &positions.iter().map(|&p| p as u64).collect::<Vec<_>>(),
            ),
            Array::f32("param:base", &[1], &[10_000.0]),
            Array::tensor("out:y", &y),
        ],
    ));

    // GQA attention (4 heads over 2 kv heads), built-in causal mask (mask = None).
    let hd = 16usize;
    let attn = |rng: &mut Rng, seq_q: usize, seq_k: usize| -> Result<Vec<Array>, Box<dyn Error>> {
        let q = Tensor::from_f32(&rng.f32s(4 * seq_q * hd, -1.0, 1.0), vec![1, 4, seq_q, hd])?;
        let k = Tensor::from_f32(&rng.f32s(2 * seq_k * hd, -1.0, 1.0), vec![1, 2, seq_k, hd])?;
        let v = Tensor::from_f32(&rng.f32s(2 * seq_k * hd, -1.0, 1.0), vec![1, 2, seq_k, hd])?;
        let y = attention::scaled_dot_product_attention(&q, &k, &v, None, None, 2)?;
        Ok(vec![
            Array::tensor("in:q", &q),
            Array::tensor("in:k", &k),
            Array::tensor("in:v", &v),
            Array::u32("param:n_kv_heads", &[1], &[2]),
            Array::tensor("out:y", &y),
        ])
    };
    cases.push(case("attention_prefill", attn(&mut rng, 4, 4)?));
    cases.push(case("attention_decode", attn(&mut rng, 1, 5)?));

    // ── sapient-core dequantisation (Tensor::to_f32_vec) — sub-project 1a plan A gates ──────────
    let (dq_rows, dq_k) = (4usize, 512usize);
    let dq: [(&str, DType, Vec<u8>); 5] = [
        (
            "dequant_q4_0",
            DType::Q4_0,
            q4_0_row(&rng.f32s(dq_rows * dq_k, -1.0, 1.0)),
        ),
        (
            "dequant_q8_0",
            DType::Q8_0,
            q8_0_row(&rng.f32s(dq_rows * dq_k, -1.0, 1.0)),
        ),
        (
            "dequant_q4_k",
            DType::Q4_K,
            kquant_rows(&mut rng, dq_rows, dq_k, q4_k_block),
        ),
        (
            "dequant_q5_k",
            DType::Q5_K,
            kquant_rows(&mut rng, dq_rows, dq_k, q5_k_block),
        ),
        (
            "dequant_q6_k",
            DType::Q6_K,
            kquant_rows(&mut rng, dq_rows, dq_k, q6_k_block),
        ),
    ];
    for (name, dtype, bytes) in dq {
        let t = Tensor::from_quant_bytes(&bytes, vec![dq_rows, dq_k], dtype)?;
        cases.push(case(
            name,
            vec![
                Array::u8("in:bytes", &[bytes.len()], &bytes),
                Array::u32("param:shape", &[2], &[dq_rows as u32, dq_k as u32]),
                Array::f32("out:f32", &[dq_rows * dq_k], &t.to_f32_vec()),
            ],
        ));
    }
    // Row-interleaved layouts: the backends-cpu repack is the only producer of R4 bytes.
    let plain = kquant_rows(&mut rng, dq_rows, dq_k, q4_k_block);
    let packed = quant::repack_q4_k_rows4(&plain, dq_rows, dq_k);
    let t = Tensor::from_quant_bytes(&packed, vec![dq_rows, dq_k], DType::Q4_K_R4)?;
    cases.push(case(
        "dequant_q4_k_r4",
        vec![
            Array::u8("in:bytes", &[packed.len()], &packed),
            Array::u32("param:shape", &[2], &[dq_rows as u32, dq_k as u32]),
            Array::f32("out:f32", &[dq_rows * dq_k], &t.to_f32_vec()),
        ],
    ));
    let plain = kquant_rows(&mut rng, dq_rows, dq_k, q6_k_block);
    let packed = quant::repack_q6_k_rows4(&plain, dq_rows, dq_k);
    let t = Tensor::from_quant_bytes(&packed, vec![dq_rows, dq_k], DType::Q6_K_R4)?;
    cases.push(case(
        "dequant_q6_k_r4",
        vec![
            Array::u8("in:bytes", &[packed.len()], &packed),
            Array::u32("param:shape", &[2], &[dq_rows as u32, dq_k as u32]),
            Array::f32("out:f32", &[dq_rows * dq_k], &t.to_f32_vec()),
        ],
    ));
    // Half-precision storage: finite values only (NaN payload policy is not a parity target).
    // `in:src_f32` lets the C++ side also check its f32→f16/bf16 narrowing against `half`.
    let src = rng.f32s(64, -100.0, 100.0);
    let f16_bytes: Vec<u8> = src
        .iter()
        .flat_map(|v| f16::from_f32(*v).to_le_bytes())
        .collect();
    let t = Tensor::from_f16_bytes(&f16_bytes, vec![64])?;
    cases.push(case(
        "dequant_f16",
        vec![
            Array::f32("in:src_f32", &[64], &src),
            Array::u8("in:bytes", &[128], &f16_bytes),
            Array::f32("out:f32", &[64], &t.to_f32_vec()),
        ],
    ));
    let bf16_bytes: Vec<u8> = src
        .iter()
        .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
        .collect();
    let t = Tensor::from_bf16_bytes(&bf16_bytes, vec![64])?;
    cases.push(case(
        "dequant_bf16",
        vec![
            Array::f32("in:src_f32", &[64], &src),
            Array::u8("in:bytes", &[128], &bf16_bytes),
            Array::f32("out:f32", &[64], &t.to_f32_vec()),
        ],
    ));

    // ── plan C: dense kernels (sub-project 1a) ──────────────────────────────────────────────────
    // matmul_nt float GEMV paths. F16 weights are dumped as RAW bytes (`in:w_f16`): the aarch64
    // path widens f16 by NEON bit-surgery that differs from `half` for non-normal AND negative
    // values, and the C++ port must reproduce that, so the dump feeds real (signed) bits.
    // m=1, k>=64 (F16) → dot_f32_x_f16; m=1, k>=512 (F32) → dot_f32_fast.
    {
        let (k, n) = (64usize, 8usize);
        let w_src = rng.f32s(n * k, -1.0, 1.0);
        let w_f16: Vec<u8> = w_src
            .iter()
            .flat_map(|v| f16::from_f32(*v).to_le_bytes())
            .collect();
        let wt = Tensor::from_f16_bytes(&w_f16, vec![n, k])?;
        let xt = Tensor::from_f32(&rng.f32s(k, -1.0, 1.0), vec![1, k])?;
        let y = matmul::matmul_nt(&xt, &wt)?;
        cases.push(case(
            "matmul_nt_f16_m1",
            vec![
                Array::tensor("in:x", &xt),
                Array::u8("in:w_f16", &[w_f16.len()], &w_f16),
                Array::u32("param:w_shape", &[2], &[n as u32, k as u32]),
                Array::tensor("out:y", &y),
            ],
        ));
        let (k, n) = (512usize, 8usize);
        let wt = Tensor::from_f32(&rng.f32s(n * k, -1.0, 1.0), vec![n, k])?;
        let xt = Tensor::from_f32(&rng.f32s(k, -1.0, 1.0), vec![1, k])?;
        let y = matmul::matmul_nt(&xt, &wt)?;
        cases.push(case(
            "matmul_nt_f32_m1_k512",
            vec![
                Array::tensor("in:x", &xt),
                Array::tensor("in:w", &wt),
                Array::tensor("out:y", &y),
            ],
        ));
    }
    // layer_norm over the last axis with weight and bias.
    {
        let xt = Tensor::from_f32(&rng.f32s(3 * 32, -2.0, 2.0), vec![3, 32])?;
        let wt = Tensor::from_f32(&rng.f32s(32, 0.5, 1.5), vec![32])?;
        let bt = Tensor::from_f32(&rng.f32s(32, -0.5, 0.5), vec![32])?;
        let y = layernorm::layer_norm(&xt, Some(&wt), Some(&bt), -1, 1e-5)?;
        cases.push(case(
            "layer_norm",
            vec![
                Array::tensor("in:x", &xt),
                Array::tensor("in:weight", &wt),
                Array::tensor("in:bias", &bt),
                Array::f32("param:eps", &[1], &[1e-5]),
                Array::tensor("out:y", &y),
            ],
        ));
    }
    // reduce_{sum,mean,max,min} over a [2, 3, 4] tensor — one case, four outputs (mean over all
    // axes is a scalar: dims = []).
    {
        let xt = Tensor::from_f32(&rng.f32s(2 * 3 * 4, -3.0, 3.0), vec![2, 3, 4])?;
        cases.push(case(
            "reduce",
            vec![
                Array::tensor("in:x", &xt),
                Array::tensor("out:sum_axis1", &reduce::reduce_sum(&xt, &[1], false)?),
                Array::tensor("out:mean_all", &reduce::reduce_mean(&xt, &[], false)?),
                Array::tensor("out:max_axis0_keep", &reduce::reduce_max(&xt, &[0], true)?),
                Array::tensor("out:min_axis_neg1", &reduce::reduce_min(&xt, &[-1], false)?),
            ],
        ));
    }
    // Partial RoPE (Phi: rotary_dim < head_dim) and scaled partial RoPE (Gemma3: pos_scale 8).
    {
        let xt = Tensor::from_f32(&rng.f32s(2 * 3 * 16, -1.0, 1.0), vec![1, 2, 3, 16])?;
        let positions: Vec<usize> = vec![7, 8, 9];
        let pos_u64: Vec<u64> = positions.iter().map(|&p| p as u64).collect();
        let y = rope::apply_rope_partial(&xt, &positions, 10_000.0, 8)?;
        cases.push(case(
            "apply_rope_partial",
            vec![
                Array::tensor("in:x", &xt),
                Array::u64("param:positions", &[3], &pos_u64),
                Array::f32("param:base", &[1], &[10_000.0]),
                Array::u32("param:rotary_dim", &[1], &[8]),
                Array::tensor("out:y", &y),
            ],
        ));
        let y = rope::apply_rope_partial_scaled(&xt, &positions, 1_000_000.0, 16, 8.0)?;
        cases.push(case(
            "apply_rope_partial_scaled",
            vec![
                Array::tensor("in:x", &xt),
                Array::u64("param:positions", &[3], &pos_u64),
                Array::f32("param:base", &[1], &[1_000_000.0]),
                Array::u32("param:rotary_dim", &[1], &[16]),
                Array::f32("param:pos_scale", &[1], &[8.0]),
                Array::tensor("out:y", &y),
            ],
        ));
    }
    // Attention with an explicit additive mask (sliding window of 3, seq_q=2 over seq_k=6, GQA
    // 4/2, explicit scale) — exercises the mask branch, the -inf skip and the explicit scale.
    {
        let (seq_q, seq_k) = (2usize, 6usize);
        let q = Tensor::from_f32(&rng.f32s(4 * seq_q * hd, -1.0, 1.0), vec![1, 4, seq_q, hd])?;
        let k = Tensor::from_f32(&rng.f32s(2 * seq_k * hd, -1.0, 1.0), vec![1, 2, seq_k, hd])?;
        let v = Tensor::from_f32(&rng.f32s(2 * seq_k * hd, -1.0, 1.0), vec![1, 2, seq_k, hd])?;
        let mut m = vec![0.0f32; seq_q * seq_k];
        for qi in 0..seq_q {
            let pos = qi + (seq_k - seq_q);
            for ki in 0..seq_k {
                if ki > pos || ki + 3 <= pos {
                    m[qi * seq_k + ki] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::from_f32(&m, vec![seq_q, seq_k])?;
        let y = attention::scaled_dot_product_attention(&q, &k, &v, Some(&mask), Some(0.125), 2)?;
        cases.push(case(
            "attention_masked",
            vec![
                Array::tensor("in:q", &q),
                Array::tensor("in:k", &k),
                Array::tensor("in:v", &v),
                Array::tensor("in:mask", &mask),
                Array::f32("param:scale", &[1], &[0.125]),
                Array::u32("param:n_kv_heads", &[1], &[2]),
                Array::tensor("out:y", &y),
            ],
        ));
    }
    // gelu (tanh approximation), softmax over axis 0, log_softmax over the last axis.
    {
        let xt = Tensor::from_f32(&rng.f32s(64, -6.0, 6.0), vec![64])?;
        cases.push(case(
            "gelu",
            vec![
                Array::tensor("in:x", &xt),
                Array::tensor("out:y", &elementwise::gelu(&xt)?),
            ],
        ));
        let xt = Tensor::from_f32(&rng.f32s(3 * 5, -4.0, 4.0), vec![3, 5])?;
        cases.push(case(
            "softmax_axis0",
            vec![
                Array::tensor("in:x", &xt),
                Array::i32("param:axis", &[1], &[0]),
                Array::tensor("out:y", &softmax::softmax(&xt, 0)?),
            ],
        ));
        cases.push(case(
            "log_softmax",
            vec![
                Array::tensor("in:x", &xt),
                Array::i32("param:axis", &[1], &[-1]),
                Array::tensor("out:y", &softmax::log_softmax(&xt, -1)?),
            ],
        ));
    }
    // conv2d (im2col + sgemm → max-error gated): stride 1 with padding, stride 2 without;
    // groups 1, bias.
    {
        let xt = Tensor::from_f32(&rng.f32s(3 * 7 * 7, -1.0, 1.0), vec![1, 3, 7, 7])?;
        let wt = Tensor::from_f32(&rng.f32s(4 * 3 * 3 * 3, -1.0, 1.0), vec![4, 3, 3, 3])?;
        let bt = Tensor::from_f32(&rng.f32s(4, -0.5, 0.5), vec![4])?;
        let conv = |pads: [usize; 4], strides: [usize; 2]| -> Result<Vec<Array>, Box<dyn Error>> {
            let y = conv2d::conv2d(&xt, &wt, Some(&bt), [3, 3], pads, strides, [1, 1], 1)?;
            Ok(vec![
                Array::tensor("in:x", &xt),
                Array::tensor("in:w", &wt),
                Array::tensor("in:bias", &bt),
                Array::u32("param:pads", &[4], &pads.map(|p| p as u32)),
                Array::u32("param:strides", &[2], &strides.map(|s| s as u32)),
                Array::tensor("out:y", &y),
            ])
        };
        cases.push(case("conv2d_s1", conv([1, 1, 1, 1], [1, 1])?));
        cases.push(case("conv2d_s2", conv([0, 0, 0, 0], [2, 2])?));
    }

    // ── plan D: quantized kernels (sub-project 1a) ──────────────────────────────────────────────
    // Activation quantisers: per-32 int8 blocks (+ the precomputed sums) and the Q8_K per-256
    // format, with one outlier channel so the per-block scale isolation is exercised. Integer
    // outputs are compared exactly on the C++ side; scales bit-identically.
    {
        let mut xa = rng.f32s(512, -2.0, 2.0);
        xa[100] = 25.0;
        let (q, s) = quant::quantize_row_to_i8_blocks(&xa);
        cases.push(case(
            "quantize_row_to_i8_blocks",
            vec![
                Array::f32("in:x", &[512], &xa),
                Array::i8("out:q", &[512], &q),
                Array::f32("out:scales", &[16], &s),
            ],
        ));
        let sums = quant::i8_block_sums(&q);
        cases.push(case(
            "i8_block_sums",
            vec![
                Array::i8("in:q", &[512], &q),
                Array::i32("out:sums", &[16], &sums),
            ],
        ));
        let (q, s, sums) = quant::quantize_row_to_q8k(&xa);
        cases.push(case(
            "quantize_row_to_q8k",
            vec![
                Array::f32("in:x", &[512], &xa),
                Array::i8("out:q", &[512], &q),
                Array::f32("out:scales", &[2], &s),
                Array::i32("out:sums", &[16], &sums),
            ],
        ));
    }
    // Row-interleaved repacks (u8 exact) — the only producers of R4 bytes — and the matmul_nt
    // paths plan C stubbed: Q4_0 (m=1 GEMV, m=3), Q5_K (m=1, no SIMD/dotprod path exists), Q8_0
    // at m=8 (the blocked W8A8 GEMM), and the R4 layouts at m ∈ {1 (decode), 2 (SMMLA pairs),
    // 3 (pair + odd tail), 8}. The R4 cases feed the Rust-repacked bytes so the matmul gate is
    // independent of the repack gate.
    {
        let (r4_rows, r4_k) = (8usize, 512usize);
        let q4k_plain = kquant_rows(&mut rng, r4_rows, r4_k, q4_k_block);
        let q4k_packed = quant::repack_q4_k_rows4(&q4k_plain, r4_rows, r4_k);
        cases.push(case(
            "repack_q4_k_rows4",
            vec![
                Array::u8("in:blocks", &[q4k_plain.len()], &q4k_plain),
                Array::u32("param:shape", &[2], &[r4_rows as u32, r4_k as u32]),
                Array::u8("out:packed", &[q4k_packed.len()], &q4k_packed),
            ],
        ));
        let q6k_plain = kquant_rows(&mut rng, r4_rows, r4_k, q6_k_block);
        let q6k_packed = quant::repack_q6_k_rows4(&q6k_plain, r4_rows, r4_k);
        cases.push(case(
            "repack_q6_k_rows4",
            vec![
                Array::u8("in:blocks", &[q6k_plain.len()], &q6k_plain),
                Array::u32("param:shape", &[2], &[r4_rows as u32, r4_k as u32]),
                Array::u8("out:packed", &[q6k_packed.len()], &q6k_packed),
            ],
        ));

        let mut quant_matmul = |name: String,
                                wbytes: &[u8],
                                dtype: DType,
                                m: usize,
                                rng: &mut Rng|
         -> Result<(), Box<dyn Error>> {
            let wt = Tensor::from_quant_bytes(wbytes, vec![rows, kq_len], dtype)?;
            let xt = Tensor::from_f32(&rng.f32s(m * kq_len, -1.0, 1.0), vec![m, kq_len])?;
            let y = matmul::matmul_nt(&xt, &wt)?;
            cases.push(case(
                name,
                vec![
                    Array::tensor("in:x", &xt),
                    Array::u8("in:w_blocks", &[wbytes.len()], wbytes),
                    Array::u32("param:w_shape", &[2], &[rows as u32, kq_len as u32]),
                    Array::tensor("out:y", &y),
                ],
            ));
            Ok(())
        };
        let wq4 = q4_0_row(&rng.f32s(rows * kq_len, -1.0, 1.0));
        for m in [1usize, 3] {
            quant_matmul(
                format!("matmul_nt_q4_0_m{m}"),
                &wq4,
                DType::Q4_0,
                m,
                &mut rng,
            )?;
        }
        let wq5 = kquant_rows(&mut rng, rows, kq_len, q5_k_block);
        quant_matmul(
            "matmul_nt_q5_k_m1".to_string(),
            &wq5,
            DType::Q5_K,
            1,
            &mut rng,
        )?;
        let wq8 = q8_0_row(&rng.f32s(rows * kq_len, -1.0, 1.0));
        quant_matmul(
            "matmul_nt_q8_0_m8".to_string(),
            &wq8,
            DType::Q8_0,
            8,
            &mut rng,
        )?;
        for m in [1usize, 2, 3, 8] {
            quant_matmul(
                format!("matmul_nt_q4_k_r4_m{m}{q8k_suffix}"),
                &q4k_packed,
                DType::Q4_K_R4,
                m,
                &mut rng,
            )?;
        }
        for m in [1usize, 2, 3, 8] {
            quant_matmul(
                format!("matmul_nt_q6_k_r4_m{m}{q8k_suffix}"),
                &q6k_packed,
                DType::Q6_K_R4,
                m,
                &mut rng,
            )?;
        }
    }
    // Plan-C carry-over: the SIMD-body + scalar-tail mixes of the dense dots. k=519 → dot_f32_fast
    // runs its 16-wide body, one 4-wide step and a 3-element scalar tail; k=67 → dot_f32_x_f16
    // mixes the NEON bit-surgery body with the software-f16 tail in ONE dot; head_dim=10 →
    // attention's dot_f32_neon/saxpby_neon take their 4-wide body + 2-lane tails.
    {
        let (k, n) = (519usize, 8usize);
        let wt = Tensor::from_f32(&rng.f32s(n * k, -1.0, 1.0), vec![n, k])?;
        let xt = Tensor::from_f32(&rng.f32s(k, -1.0, 1.0), vec![1, k])?;
        let y = matmul::matmul_nt(&xt, &wt)?;
        cases.push(case(
            "matmul_nt_f32_m1_k519",
            vec![
                Array::tensor("in:x", &xt),
                Array::tensor("in:w", &wt),
                Array::tensor("out:y", &y),
            ],
        ));
        let (k, n) = (67usize, 8usize);
        let w_src = rng.f32s(n * k, -1.0, 1.0);
        let w_f16: Vec<u8> = w_src
            .iter()
            .flat_map(|v| f16::from_f32(*v).to_le_bytes())
            .collect();
        let wt = Tensor::from_f16_bytes(&w_f16, vec![n, k])?;
        let xt = Tensor::from_f32(&rng.f32s(k, -1.0, 1.0), vec![1, k])?;
        let y = matmul::matmul_nt(&xt, &wt)?;
        cases.push(case(
            "matmul_nt_f16_m1_k67",
            vec![
                Array::tensor("in:x", &xt),
                Array::u8("in:w_f16", &[w_f16.len()], &w_f16),
                Array::u32("param:w_shape", &[2], &[n as u32, k as u32]),
                Array::tensor("out:y", &y),
            ],
        ));
        let hd10 = 10usize;
        let q = Tensor::from_f32(&rng.f32s(4 * hd10, -1.0, 1.0), vec![1, 4, 1, hd10])?;
        let k = Tensor::from_f32(&rng.f32s(2 * 5 * hd10, -1.0, 1.0), vec![1, 2, 5, hd10])?;
        let v = Tensor::from_f32(&rng.f32s(2 * 5 * hd10, -1.0, 1.0), vec![1, 2, 5, hd10])?;
        let y = attention::scaled_dot_product_attention(&q, &k, &v, None, None, 2)?;
        cases.push(case(
            "attention_decode_hd10",
            vec![
                Array::tensor("in:q", &q),
                Array::tensor("in:k", &k),
                Array::tensor("in:v", &v),
                Array::u32("param:n_kv_heads", &[1], &[2]),
                Array::tensor("out:y", &y),
            ],
        ));
    }
    if q8k_off {
        cases.retain(|(name, _)| name.ends_with("_q8k_off"));
    }

    Ok(cases)
}

/// Fixed-content sample for the C++ reader's unit test (committed as a fixture; contains no
/// kernel output, so it is host-independent).
fn format_sample() -> Case {
    case(
        "format_sample",
        vec![
            Array::f32("in:f32", &[5], &[0.0, 1.0, -1.0, 0.5, 3.25]),
            Array::u8("in:u8", &[2, 2], &[0, 1, 2, 255]),
            Array::i8("in:i8", &[2], &[-128, 127]),
            Array::i32("param:i32", &[1], &[-42]),
            Array::u32("param:u32", &[1], &[4_000_000_000]),
            Array::u64("param:u64", &[1], &[1 << 40]),
            Array::f32("out:empty", &[0], &[]),
        ],
    )
}

fn take(args: &[String], i: usize, flag: &str) -> String {
    args.get(i + 1).cloned().unwrap_or_else(|| {
        eprintln!("{flag} needs a value\n{USAGE}");
        std::process::exit(2)
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut out: Option<PathBuf> = None;
    let mut sample: Option<PathBuf> = None;
    let mut seed = DEFAULT_SEED;
    let mut q8k_off = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                out = Some(PathBuf::from(take(&args, i, "--out")));
                i += 2;
            }
            "--format-sample" => {
                sample = Some(PathBuf::from(take(&args, i, "--format-sample")));
                i += 2;
            }
            "--seed" => {
                seed = take(&args, i, "--seed").parse()?;
                i += 2;
            }
            "--q8k-off" => {
                q8k_off = true;
                i += 1;
            }
            other => {
                eprintln!("unknown argument: {other}\n{USAGE}");
                std::process::exit(2);
            }
        }
    }

    if sample.is_none() && out.is_none() {
        eprintln!("{USAGE}");
        std::process::exit(2);
    }
    if let Some(path) = sample {
        let (name, arrays) = format_sample();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, encode_case(&name, &arrays))?;
        println!("wrote format sample to {}", path.display());
    }
    if let Some(dir) = out {
        fs::create_dir_all(&dir)?;
        let cases = build_cases(seed, q8k_off)?;
        for (name, arrays) in &cases {
            let path = dir.join(format!("{name}.sapd"));
            fs::write(&path, encode_case(name, arrays))?;
            println!("{}", path.display());
        }
        println!(
            "wrote {} cases to {} (seed {seed:#x})",
            cases.len(),
            dir.display()
        );
    }
    Ok(())
}
