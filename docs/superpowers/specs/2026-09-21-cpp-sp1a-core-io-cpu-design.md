# Sub-project 1a — core, IO and CPU kernels: design spec

> Status: **approved 2026-09-21** (brainstorming session; three design sections approved).
> Parent: `docs/superpowers/specs/2026-09-20-cpp-rewrite-design.md` (programme spec; D3
> conventions, D4 library map and D5 row 1a bind this document). Porting maps (line-cited
> inventories of the Rust code being ported): `docs/superpowers/notes/2026-09-21-sp1a-porting-map-core-io.md`
> and `docs/superpowers/notes/2026-09-21-sp1a-porting-map-cpu-kernels.md`.
> Implementation plans: one per plan A/C/D/E/B under `docs/superpowers/plans/`, written in that order.

## 1. Goal and scope

Port the three Rust crates that the live inference path's CPU tier is built on —
`sapient-core`, `sapient-io`, `sapient-backends-cpu` — to C++ under `cpp/libs/`, with the
quantized kernels **bit-identical** to the Rust oracle and every named Rust unit test ported.
This is the layer sub-project 1b's `LlamaForward` + `Pipeline` vertical slice stands on.

**In scope** (≈10.8k Rust LOC): `sapient-core` entirely (tensor, dtype, shape, buffer, error);
`sapient-io`'s GGUF parser + heap/mmap/metadata-only loaders + Q5_0→Q8_0 requantisation,
safetensors loader, mmap wrapper; `sapient-backends-cpu`'s nine kernel modules, `spinpool.rs`,
`thermal.rs`, and the parallel-for and SGEMM facilities the kernels rely on.

**Out of scope, moved to sub-project 8** (decision 2026-09-21, amends programme-spec row 1a):
the ONNX reader (`onnx.rs`), the `Graph`-producing loader entry points (`load_graph`,
`GgufLoader::load`, `SafetensorsLoader::load_as_graph`), and backends-cpu's graph executor
`backend.rs` with its `pool.rs` allocator — all used only by `sapient-runtime` and the Metal
graph backend. Also deferred to 8: the serde `Serialize/Deserialize` derives on
`Tensor`/`Shape`/`DType` (IR-path only). Dropped from 1a: Google Benchmark (the Rust criterion
bench is f32-only and gates nothing).

## 2. Structure

Three CMake targets under `cpp/libs/`, each with alias `sapient::<crate>`, namespace
`sapient::<crate>`, headers under `include/sapient/<crate>/`, one `.hpp/.cpp` pair per Rust
module with the same stem, GoogleTests under `tests/` with the Rust test names.

### 2.1 `sapient-core` → `sapient::core`
| File | Ports | Notes |
|---|---|---|
| `dtype.hpp/.cpp` | `dtype.rs` | 14 variants in Rust order; `element_size` (quant → 0), `alignment` (quant → 2), `block_bytes`/`block_numel` (panic on non-quant), `byte_count` (truncating division), `is_*`, `name`, `from_str` (same accepted spellings, R4 names not parseable), ONNX code mapping. |
| `shape.hpp/.cpp` | `shape.rs` | `dims` public, `strides()` row-major, `numel()` of scalar = 1, `broadcast_with` NumPy rules, `validate` rejects zero dims with the `InvalidGraph` error text, `flat_index` returns an element offset. |
| `buffer.hpp/.cpp` | `buffer.rs` | `Buffer` abstract class (`bytes`, `bytes_mut`, `len`, `is_mmap` default false, `alignment`, `device`); `CpuBuffer` zero-filled aligned allocation with the per-constructor alignments 64/64/16/4; `from_f32_vec` moves the vector into the buffer (no Rust-style layout trick); `BufferHandle = std::shared_ptr<Buffer>`. |
| `tensor.hpp/.cpp` | `tensor.rs` | Fields `{shape, dtype, strides (elements), buffer, offset (bytes)}`; constructors with the same validation and error variants; `bytes()` bounded by `byte_count` for quant dtypes and unbounded for float (load-bearing for zero-copy expert views); `bytes_mut()` requires exclusive ownership (`use_count()==1`; no `weak_ptr` is ever taken); `reshape`/`t()`/`slice_axis` are views with the same arithmetic (incl. the documented `element_size()==0` trap, plus a `debug_assert`); `to_f32_vec` dispatches to `dequant.hpp`; the integer-dtype fallback is an explicit unsupported-dtype panic. |
| `error.hpp/.cpp` | `error.rs` | `enum class ErrorCode` (25 codes) + `struct Error { code; fields…; std::string to_string() }` with byte-identical messages; `Result<T> = tl::expected<T, Error>`; `SAPIENT_TRY(expr)` ≈ `?`; helper constructors `backend()`, `unsupported_op()`, `internal()`. |
| `f16.hpp` | (from `half`) | `f16_to_f32` exact widening (subnormal/inf/NaN correct), `f32_to_f16` round-to-nearest-even, `bf16_to_f32`, `f32_to_bf16`. Software only, no F16C/`vcvt` dependence. Used by core, io and the kernels' scalar paths. |
| `dequant.hpp/.cpp` | `tensor.rs` dequant arms | **The single implementation** of `get_scale_min_k4` and the block dequantizers for Q4_0, Q8_0, Q4_K, Q5_K (per-element `qh[l]`), Q6_K (+0/+2/+4/+6 scale indexing), plus the R4 de-permutation. Rust keeps three copies; C++ keeps one (approved deviation). The stale, unreachable Q5_K copy in `sapient-io` is not ported. |
| `panic.hpp` | (Rust `panic!`) | `[[noreturn]] void panic(std::string_view)`: prints to stderr and `std::abort()` — mirrors the release profile's `panic = "abort"`. |

### 2.2 `sapient-io` → `sapient::io`
| File | Ports | Notes |
|---|---|---|
| `mmap.hpp/.cpp` | (from `memmap2`) | `MappedFile` RAII over POSIX `mmap(PROT_READ, MAP_PRIVATE)` / Windows `CreateFileMapping`+`MapViewOfFile`; whole-file, read-only, refcounted via `shared_ptr`. |
| `gguf.hpp/.cpp` | `gguf.rs` | `GgufValue` as `std::variant` (16 alternatives incl. `Other`), accessors with the same widening rules; `parse_header` (magic, versions 1–3 accepted identically, KV type codes 0–12, `skip_value` semantics, `general.alignment`, `data_start`); `GgmlType` table with `to_sapient_dtype`; `MmapBuffer` (`is_mmap`, `alignment()==32`, `bytes_mut` panics with the Rust message); `load_tensors_with_metadata` (heap, whole-file read), `load_tensors_mmap` (zero-copy for Q4_0/Q8_0/Q4_K/Q5_K/Q6_K), `parse_metadata_only`, `tensors_from_bytes`; `make_tensor`/`make_tensor_mmap` decision tree; `dequantize_to_f32` (F32/F16/BF16/Q4_0/Q5_0/Q8_0/Q4_K/Q6_K via `dequant.hpp`, Q5_0 local); `quantize_to_q8_0` (`roundf`, round-then-clamp, `f32_to_f16`). Dims stay in GGUF order. |
| `safetensors.hpp/.cpp` | `safetensors.rs` | Header via nlohmann/json (`__metadata__` skipped); dtype strings; the reachable set {F32, F16, BF16}; F32 copied once (improvement over Rust's double copy, behaviour-identical); F16/BF16 kept as raw bytes; same error texts. |
| `lib` (`io.hpp`) | `lib.rs` | `load_gguf`, `load_safetensors`. No `load_graph`. |

### 2.3 `sapient-backends-cpu` → `sapient::backends_cpu`
| File | Ports | Notes |
|---|---|---|
| `cpu_features.hpp/.cpp` | `is_*_feature_detected!` | Cached runtime detection: `has_dotprod()`, `has_i8mm()` (aarch64: `sysctlbyname` on macOS, `getauxval(AT_HWCAP)` on Linux, `IsProcessorFeaturePresent` on Windows), `has_avx2_fma()` (x86: `__builtin_cpu_supports`). NEON is compile-time on aarch64, as in Rust. |
| `parallel.hpp/.cpp` | rayon | Persistent parking pool sized like rayon (`RAYON_NUM_THREADS` honoured, else hardware concurrency); `num_threads()`; `par_chunks_mut(out, chunk, f)` reproducing rayon's `(ci, [ci*chunk, min((ci+1)*chunk, len)))` partition; `par_for(n, f)`. No work stealing needed — the chunk→range map is what parity depends on. |
| `sgemm.hpp/.cpp` | `matrixmultiply::sgemm` | Same signature `(m,k,n,alpha,a,rsa,csa,b,rsb,csb,beta,c,rsc,csc)`; own cache-blocked kernel with a 4-wide FMA microkernel (NEON / AVX2+FMA when available, scalar otherwise). **Max-error gated, not bit-identical** (approved deviation; `matrixmultiply`'s blocking is not reproducible without porting it). Performance is not a 1a goal. |
| `kernels/quant.hpp/.cpp` | `quant.rs` (3787 lines) | Standalone (no Tensor dependency, like Rust). Every function in the porting map's inventory, grouped by ISA: scalar; aarch64 NEON (`target("neon")`); NEON+dotprod (`target("neon,dotprod")`, `vdotq_s32` replaces the `sdot` asm); NEON+i8mm (`target("neon,i8mm")`, `vmmlaq_s32` replaces `smmla`); x86_64 AVX2+FMA (`target("avx2,fma")`, only `dot_q8_0_row_avx2`). Same accumulation orders and horizontal reductions (`vaddvq_f32`/`vaddvq_s32`; the exact AVX2 hsum). Quantizers: `roundf` then clamp; Q4_0 nibble by truncation; `fmaxf`-style NaN semantics of `f32::max`. `repack_q4_k_rows4`/`repack_q6_k_rows4` block-major interleave. |
| `kernels/matmul.hpp/.cpp` | `matmul.rs` | `matmul`, `matmul_nt` (thermal `tick()` first; dtype dispatch table verbatim), `matmul_nt_float` (m==1 F16 GEMV path incl. the normals-only bit-surgery of `dot_f32_x_f16_neon`; m==1 k≥512 fast GEMV; else sgemm row-blocks with the same `mblock` rule), all quant arms with their runtime dispatch (`m>=8` Q8_0 blocked GEMM, `q8k_activations()`, R4 prefill via SMMLA at `m>=2 && i8mm`, R4 decode via dotprod, portable fallbacks); `gemv_chunk` (governed branch compares against the rayon-domain thread count); `for_each_out_chunk` (pool branch in plan E, `par_chunks_mut` branch in plan C — identical partition); `gemm` with bias. |
| `kernels/attention.hpp/.cpp` | `attention.rs` | `dot_f32_neon`, `saxpby_neon`, `flash_attn_row` (online softmax, `continue` on fully-masked positions, `1/EPSILON` when `l==0`), `scaled_dot_product_attention` (rank-4, GQA `kv_rep`, `mask=None` ⇒ causal with `kv_offset`), `causal_mask`. Parallel over (batch, head). Bit-identical. |
| `kernels/rope.hpp/.cpp` | `rope.rs` | `apply_rope`, `apply_rope_partial`, `apply_rope_partial_scaled`, `rope_cos_sin_cache`; `powf` + `sinf`/`cosf` from the platform libm (same libm as Rust on each OS). |
| `kernels/elementwise.hpp/.cpp` | `elementwise.rs` | Unary/binary helpers with the scalar-broadcast rule; all activations; `erf_approx` = the Abramowitz–Stegun polynomial verbatim (never `std::erff`). |
| `kernels/softmax`, `reduce`, `layernorm`, `conv2d` | same | Sequential f32 sums in the Rust order; conv2d's im2col + out-channel-block GEMM (sgemm; max-error gated) + the two `std::atomic<uint64_t>` timing counters. |
| `spinpool.hpp/.cpp` | `spinpool.rs` | Direct port: `OpSlot`, `alignas(128) Pad<T>`, `std::atomic` with the same orderings, `std::mutex`/`std::condition_variable` (Rust used std, not parking_lot), worker threads named `sapient-spin-N`, macOS `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE)` for workers and (once per thread) publishers, the seqlock protocol (generation ODD **before** draining `active`), guided block claiming with the per-OS block-size rule, `run()` serial fast paths, process-lifetime singleton `pool()`, `parallelism()`, `enabled()` = env/platform default AND `thermal::effective_threads() >= parallel::num_threads()`. Env: `SAPIENT_SPINPOOL`, `_WORKERS`, `_SPINS` (default 4000), `_BLOCK`, `_DEBUG`. |
| `thermal.hpp/.cpp` | `thermal.rs` | `ThermalGovernor` (sorted `thermal_zone*/temp` files under a root, hysteresis 80/70 °C, floor `max/2`, one-shot warning), external level cap (0..3 → full/¾/½/¼, floor 1), `effective_threads()` = stricter of the two, `tick()` rate-limited to 500 ms with the compare-exchange winner sampling. Env: `SAPIENT_THERMAL`, `_PATH`, `_HOT`, `_COOL`. Uses `std::filesystem`. |

Kernel env knobs are read exactly where Rust reads them (per-call vs `OnceLock`-cached), with
the same defaults — see the kernels porting map §2.10.

## 3. Bit-identity rules (binding on every plan)

1. Build flags per programme spec D1 (Clang, `-ffp-contract=off`, no fast-math). Rust uses no
   `mul_add`; every fused multiply-add in C++ is an explicit intrinsic where Rust had one, plain
   `a*b + c` where Rust had none. **As built (plan C):** `-fno-math-errno` joined the parity flags
   — rustc lowers `sin`/`cos`/`exp`/`pow`/`sqrt` to LLVM intrinsics with no errno, and Clang
   matches that lowering only under the flag (the Darwin default, not Linux's); this amends
   programme spec D1's flag list by reference.
2. Mirror each function's ISA variant and its dispatch condition; never "upgrade" a scalar path
   to SIMD or vice versa. x86_64 K-quant kernels stay scalar; AVX2 exists only for
   `dot_q8_0_row_avx2` and `dot_f32_avx2`.
3. Same accumulation order, same horizontal reduction intrinsics, same per-block vs per-row
   reduction shape (documented per kernel in the porting map §4).
4. Rounding: `roundf` (half away from zero) for Q8_0/activation quantisation; truncation for the
   Q4_0 nibble; `f32_to_f16` round-to-nearest-even; `fmaxf` semantics for `f32::max`.
5. f16→f32 via the software exact conversion everywhere Rust used `half`; the single
   normals-only bit-surgery in `dot_f32_x_f16_neon` is reproduced verbatim, divergence included.
6. libm calls (`expf`, `powf`, `sinf`, `cosf`, `logf`, `sqrtf`, `tanhf`) go to the platform libm,
   as Rust's do; equality is verified per platform by the golden dumps, not assumed across platforms.
7. Chunk geometry: `gemv_chunk` and `for_each_out_chunk` produce the identical
   `(chunk_index → [start, end))` partition on the parking pool and on the spin pool.
8. Exceptions: none across library boundaries; Rust panics become `sapient::panic()` aborts;
   `Result` carries every recoverable error with the byte-identical message text.

## 4. Verification

**Tiers (programme spec D1) applied to 1a:**
- **Unit tests 1:1.** Every `#[test]` in the three crates is ported with its name and its
  assertions (core 22, io 2, backends-cpu 71 + 1 ignored probe). Tests that gate on
  `dotprod`/`i8mm` at runtime keep the same runtime skips, expressed as `GTEST_SKIP()` with a
  reason. The four thermal governor tests build a fake sysfs root in a temp dir exactly as Rust
  does; `external_level_caps_effective_threads` stays the only test touching the global level.
- **Golden dumps.** `crates/sapient-backends/cpu/examples/dump_kernels.rs` (test-only Rust,
  allowed) is extended per plan; the C++ side gains `golden_kernels_test.cpp` (backends-cpu)
  and `golden_dequant_test.cpp` (core) that load `SAPIENT_GOLDEN_DIR` and compare. Comparison
  helpers live in `sapient::testing` (`expect_bit_identical` over `std::bit_cast<uint32_t>` so
  NaN/−0 compare exactly; `max_abs_err`; `case_or_skip`). Bit-identical unless the case is
  sgemm-backed (`matmul_nt_f32_m4`, `conv2d`, float prefill), which use
  `max_abs_err ≤ 1e-5 · max(1, max|ref|)`. **As built (plan A):** these shipped as
  `sapient::testing::bit_identical` (not `expect_bit_identical`) and the `SAPIENT_GOLDEN_CASE(var,
  name)` macro (not `case_or_skip`) — `SAPIENT_GOLDEN_DIR` unset SKIPs, set-but-the-named-case-
  missing FAILs (a stale local dump directory must not silently downgrade a bit-identity gate to
  a skip).
- **Dump cases added by plan:** A — `to_f32_vec` of random Q4_0/Q8_0/Q4_K/Q5_K/Q6_K/Q4_K_R4/
  Q6_K_R4/F16/BF16 tensors; C — `layer_norm`, `conv2d` (stride 1 and 2, groups 1), `reduce_*`,
  `apply_rope_partial(_scaled)`, masked `attention`, `gelu`, `softmax` axis 0, `log_softmax`,
  `matmul_nt_f32` with F16 weights (m=1 GEMV path); D — `quantize_row_to_i8_blocks`,
  `quantize_row_to_q8k`, `i8_block_sums`, `repack_q4_k_rows4`/`q6_k`, `matmul_nt` over
  `Q4_K_R4`/`Q6_K_R4` at m ∈ {1, 2, 3, 8} and `Q8_0` at m=8 (the blocked GEMM path),
  `Q5_K` matmul, both `SAPIENT_Q8K_ACT` settings; E — the D suite re-run with
  `SAPIENT_SPINPOOL=1` and `=0` (bit-identical); B — none (io is gated by 1b's greedy parity;
  1a gives it synthetic-file tests and an env-gated real-file heap-vs-mmap byte check).
  **As built (plan D):** "both `SAPIENT_Q8K_ACT` settings" is a second `dump_kernels --q8k-off`
  pass under `SAPIENT_Q8K_ACT=0` (the knob is read once per process on both sides) whose twelve
  knob-sensitive cases carry a `_q8k_off` suffix, consumed by a second ctest entry with that
  environment; the D suite also carries plan C's odd-length carry-over cases
  (`matmul_nt_f32_m1_k519`, `matmul_nt_f16_m1_k67`, `attention_decode_hd10`).
- **Hosts.** macOS arm64 locally (NEON/dotprod/i8mm on Apple M-series); x86_64 scalar/AVX2 and
  Windows only via the CI `cpp-parity`/`cpp-build-windows` jobs.
- **Rust untouched** except `dump_kernels.rs` extensions (each plan lists its additions).

## 5. Plans

| Plan | Ports | Gate | Doc updates |
|---|---|---|---|
| **A: core** | §2.1 entirely; `dump_kernels` dequant cases | 22 core tests by name; dequant golden bit-identical | CLAUDE.md sp1a section, ROADMAP row 1a progress, PARITY.md rows |
| **C: dense kernels** | `cpu_features`, `parallel`, `sgemm`, `matmul` float paths + dispatcher skeleton, `attention`, `rope`, `elementwise`, `softmax`, `reduce`, `layernorm`, `conv2d` | 26 dense tests + 5 float matmul tests by name; golden dense cases bit-identical, sgemm cases max-error | same |
| **D: quant kernels** | `kernels/quant` (all ISA paths), quant arms of `matmul_nt`, R4 repack, Q8_K | 24 quant + 5 quant matmul tests (incl. the two `to_bits` gates); all quant golden cases bit-identical on arm64 (CI: x86_64) | same |
| **E: spinpool + thermal** | `spinpool`, `thermal`, `tick()` wiring, `enabled()`, `for_each_out_chunk` pool branch | 5+1 spinpool and 6 thermal tests; D golden suite pool-on == pool-off | same |
| **B: io** | §2.2 entirely | 2 io tests; synthetic GGUF/safetensors tests; `SAPIENT_TEST_GGUF` heap-vs-mmap byte check | same, plus PROJECT_GUIDE §6 note |

Branch: `feat/cpp-sp1a`, stacked on `feat/cpp-sp0-scaffold` (unpushed; the user pushes).
Execution: subagent-driven, per-task reviews, one final review per plan.

## 6. Third-party additions in 1a
`tl::expected` (CC0-1.0; core `Result`), nlohmann/json (MIT; safetensors header). Both pinned
in `cpp/cmake/deps.cmake` and listed in `cpp/third_party/LICENSES.md` when their plan lands
(A and B respectively). Nothing else.

## 7. Risks
- `sgemm` correctness vs `matrixmultiply` is max-error gated; its performance is a later concern.
- libm equality is per platform; a Windows UCRT difference would show only in CI.
- The spin pool's bugs are timing-dependent; the ported `rapid_ops_with_constant_parking` stress
  test and the pool-on/off golden run are the gates.
- The hardest single kernel family (Q4_K/Q6_K R4 + SMMLA, plan D) is the one whose x86 twin is
  scalar — x86 parity there is cheap; arm64 parity is the real work.

## 8. Open gaps carried in
From `docs/PARITY.md` (sub-project 0): the R4/SMMLA/partial-RoPE/activation-quantizer dump
cases (closed by plans C/D), the `sapient::testing` comparison helpers (plan A adds them), the
stale `(global)` comment in `cpp/CMakeLists.txt` and the two CLAUDE.md omissions (plan A's docs task).
