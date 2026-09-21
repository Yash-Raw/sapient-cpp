# Porting map: sapient-core + sapient-io internals (Rust → C++)

> Reference for sub-project 1a plans A (core) and B (io). Generated 2026-09-21 from a full read of both crates at commit 3cd6997; verify line numbers against the tree before relying on them.


---

## 0. BIT-IDENTITY HAZARDS (read first — these decide parity)

| Rust construct | C++ equivalent that is bit-identical | C++ that is WRONG |
|---|---|---|
| `f32::round()` in `quantize_to_q8_0` (`sapient-io/src/gguf.rs:297`) | `std::roundf` (half-away-from-zero; same as ggml) | `std::rint` / `nearbyint` (half-to-even) |
| `half::f16::from_f32(d)` (`gguf.rs:295`) | RNE f32→f16 incl. subnormals (F16C `_cvtss_sh(x,0)` or software RNE) | truncating f16 conversion |
| `half::f16::to_f32` / `bf16→f32` | exact widening — no rounding concerns | — |
| `d * sc[i] as i8 as f32 * q1` (`sapient-core/src/tensor.rs:555`) | `(d * (float)(int8_t)sc[i]) * q1` — **left-assoc, keep the order** | reassociating to `d*(sc*q)` |
| `(byte >> 4) as i32 - 8` etc. | cast to `int32_t` explicitly; `>>` on `uint8_t` is a zero-extending shift | signed shift of a promoted char |
| `as i8 as u8` (`gguf.rs:298`) | `(uint8_t)(int8_t)v` | direct `(uint8_t)` of a float |
| whole crate | build with `-ffp-contract=off` (already a repo rule) — every dequant is mul-then-add and must NOT become FMA | default `-ffp-contract=fast` |

Everything is **little-endian only**; there is no BE branch anywhere.

---

# PART A — `sapient-core`

## A1. `Tensor` internals — `sapient-core/src/tensor.rs`

### Fields (`tensor.rs:21-29`)
```rust
pub struct Tensor {            // #[derive(Debug, Clone)]
    shape: Shape,              // Vec<usize> newtype
    dtype: DType,
    strides: Vec<usize>,       // ELEMENT strides (not byte strides), row-major by default
    buffer: BufferHandle,      // = Arc<dyn Buffer>
    offset: usize,             // BYTE offset into the buffer of element [0,0,…]
}
```
C++ shape: `struct Tensor { Shape shape; DType dtype; std::vector<size_t> strides; BufferHandle buffer /*std::shared_ptr<Buffer>*/; size_t offset; };` — `Clone` = shallow (shared_ptr copy).

**Mixed-unit trap:** `strides` is in *elements*, `offset` is in *bytes*. `slice_axis` multiplies them together (see A1.8).

### A1.1 Strides — always recomputed, never inherited
Every constructor does `let strides = shape.strides();` (`:39, :62, :82, :105, :127, :160, :192`). `Shape::strides()` (`shape.rs:38-48`):
```
n = ndim; if n==0 -> {}
strides[n-1] = 1
for i in (0..n-1).rev(): strides[i] = strides[i+1] * dims[i+1]
```
Non-natural strides only ever arise from `t()` (swap) — `reshape` and `slice_axis` recompute/copy.

### A1.2 `is_contiguous()` (`:232-234`)
`self.strides == self.shape.strides() && self.offset == 0`. **Note the `offset == 0` clause** — any sliced view is "non-contiguous" even with natural strides.

### A1.3 `as_bytes()` — the quantized/float asymmetry (`:242-250`)
```
bytes = buffer.as_bytes()
if dtype.is_quantized():  return bytes[offset .. offset + dtype.byte_count(numel)]   // BOUNDED
else:                     return bytes[offset ..]                                    // UNBOUNDED to end of buffer
```
This bound is load-bearing: it is why zero-copy MoE expert views into a shared stacked buffer are safe for quant dtypes and **unsafe for F32/F16** (CLAUDE.md records this). Mirror it exactly.

Related: `as_quant_blocks()` (`:261-268`) = `assert(is_quantized())` then `as_bytes()`.
`as_f32_slice()` (`:271-281`) = `assert_eq!(dtype, F32)`, `assert(bytes.len()%4==0)`, reinterpret `bytes[offset..]` as `f32[len/4]` — **also unbounded by `numel`**.

### A1.4 Mutable access — exclusivity via `Arc::get_mut`, and the mmap panic
`as_bytes_mut()` (`:590-597`):
```
offset = self.offset; end = offset + dtype.byte_count(numel)      // BOUNDED for ALL dtypes
buf = Arc::get_mut(&mut self.buffer.0).ok_or(internal("Cannot mutate shared tensor buffer"))?
return &mut buf.as_bytes_mut()[offset..end]
```
`as_f32_slice_mut()` (`:599-615`):
```
if dtype != F32 -> Err(internal("Tensor dtype is not F32"))
buf = Arc::get_mut(...)  -> Err(internal("Cannot mutate shared tensor buffer"))
bytes = &mut buf.as_bytes_mut()[offset..]                          // UNBOUNDED
if bytes.len()%4 != 0 -> Err(internal("Buffer length not a multiple of 4"))
reinterpret as &mut [f32]
```
Key facts for the port:
- It is **`Arc::get_mut`, NOT `Arc::make_mut`** — no copy-on-write. Exclusive access means `strong_count == 1 && weak_count == 0`. C++: `shared_ptr::use_count() == 1` (plus no weak refs — use a `weak_count`-free design or `std::atomic` refcount you control).
- **There is no `is_mmap()` check in `Tensor`.** "mmap refusal" is a **panic** raised inside `MmapBuffer::as_bytes_mut` (`sapient-io/src/gguf.rs:63-65`: `panic!("MmapBuffer is read-only — model weights cannot be mutated in-place")`), not a `Result`. In C++ this should be a throw/abort from the mmap buffer's `bytes_mut()`.
- **Bounding asymmetry to mirror**: `as_bytes` (float) unbounded · `as_bytes` (quant) bounded · `as_bytes_mut` bounded for *all* dtypes · `as_f32_slice_mut` unbounded.

### A1.5 Constructors and their validation
| fn | line | validation | buffer |
|---|---|---|---|
| `zeros(shape, dtype)` | `:35-48` | `shape.validate()` | `CpuBuffer::zeros(numel, dtype)` (align 64) |
| `from_f32_vec(Vec<f32>, shape)` | `:53-71` | validate + `data.len() != numel → ShapeMismatch{expected: dims, got: [len]}` | `CpuBuffer::from_f32_vec` (zero-copy) |
| `from_f32(&[f32], shape)` | `:73-91` | same | `CpuBuffer::from_f32_slice` (copy, align 64) |
| `from_bf16_bytes(&[u8], shape)` | `:95-114` | validate + `len != numel*2 → ShapeMismatch{expected: dims, got: [len/2]}` | `from_bytes_slice` (copy, align 16), dtype BF16 |
| `from_f16_bytes` | `:117-136` | same | same, dtype F16 |
| `from_quant_bytes(&[u8], shape, dtype)` | `:143-169` | `!dtype.is_quantized() → TypeMismatch{expected:"a quantized dtype (Q4_0, Q8_0, Q4_K, Q5_K, Q6_K)", got: dtype.to_string()}`; validate; `len != dtype.byte_count(numel) → ShapeMismatch{expected:[expected_bytes], got:[len]}` | `from_bytes_slice` |
| `scalar_f32(v)` | `:172-174` | `from_f32(&[v], Shape::scalar())` | — |
| **`from_buffer(shape, dtype, buffer, offset)`** | `:177-200` | `shape.validate()`; `required = dtype.byte_count(shape.numel())`; **`if buffer.len() < offset + required → BufferSizeMismatch{expected: offset+required, got: buffer.len()}`**; strides = `shape.strides()` | caller-supplied handle, offset stored verbatim |

### A1.6 `reshape` — pure view, never a copy (`:621-631`)
`Shape::reshape` validates `new.numel() == old.numel()` (else `ShapeMismatch{expected: old dims, got: new dims}`), then a new `Tensor` with **new natural strides**, `buffer.clone()` (Arc bump), **same `offset`**. No data touched, no contiguity check (a transposed tensor can be "reshaped" producing a wrong view — mirror as-is).

### A1.7 `t()` — view with swapped dims + swapped strides (`:634-649`)
```
if ndim != 2 -> Err(internal("t() requires a 2-D tensor"))
dims.swap(0,1); strides.swap(0,1)
Tensor{ shape: Shape(dims), dtype, strides, buffer: clone, offset }   // NO data movement
```

### A1.8 `slice_axis(axis, start, end)` — view, and the quant trap (`:652-669`)
```
if axis >= dims.len()             -> Err(internal("slice axis out of bounds"))
if start > end || end > dims[axis]-> Err(internal("slice range out of bounds"))
dims[axis] = end - start
offset = self.offset + start * self.strides[axis] * self.dtype.element_size()   // <-- byte offset
new Tensor { Shape(dims), dtype, strides: self.strides.clone(), buffer.clone(), offset }
```
- **No dtype validation.** For any quantized dtype `element_size() == 0`, so the offset is unchanged → silently wrong view. This is the documented "slice_axis/element_size are invalid for quant dtypes" trap (CLAUDE.md); callers use byte-count slicing instead. The C++ port should reproduce the arithmetic but is a good place to add a debug assert.
- Strides are **not** adjusted (slicing does not change strides), so the result is generally non-contiguous (`offset != 0`).

### A1.9 Other internals worth mirroring
- `is_scalar()` (`:227-229`) = `shape.is_scalar() || numel()==1`.
- `byte_size()` (`:674-676`) = `dtype.byte_count(numel())`.
- `is_mmap()` (`:254-256`) delegates to `buffer.is_mmap()`.
- `to_f32_tensor()` (`:580-585`): F32 → `self.clone()` (shares buffer!); else `Tensor::from_f32(&to_f32_vec(), shape.clone())`.
- `to_f32_cow()` (`:331-337`): F32 → **borrow** `as_f32_slice()` (unbounded); else owned `to_f32_vec()`. C++: return a `std::variant<span<const float>, vector<float>>` or a small `Cow` type.
- `to_contiguous_f32_vec()` (`:292-328`): contiguous fast path truncates to `numel`; non-contiguous slow path does flat→multi-index→`Σ idx_d·strides[d]` gather from the *f32-interpreted* buffer, with `raw.get(src).unwrap_or(&0.0)` → **out-of-range reads silently yield 0.0**.
- `Display` (`:679-689`): `"Tensor(shape={}, dtype={}, device={})"`.
- Serde (`:694-727`): proxy `TensorProxy{shape, dtype, data: Vec<f32>}`; `Serialize` emits f32 data only when dtype==F32, else an **empty vec**; `Deserialize` maps empty→`Tensor::zeros(shape,dtype)`, non-empty→`from_f32`. `TensorMeta{shape,dtype}` (`:731-743`) + `From<&Tensor>`.

---

## A2. Dequantization — `to_f32_vec()` (`tensor.rs:341-576`)

Dispatch on `self.dtype`; all arms allocate `out = vec![0.0; numel]` and read `self.as_bytes()` (which for quant dtypes is already bounded to `byte_count(numel)`).

**f16 conversion used everywhere: `half::f16::from_le_bytes([lo,hi]).to_f32()`** — an exact widening. Must be bit-identical to the CPU kernels (`sapient-backends-cpu/src/kernels/quant.rs`) and to `sapient-io`'s `f16_to_f32`.

### F32 (`:347`)
`as_f32_slice().to_vec()` — copies the *whole remaining buffer*, not `numel`.

### BF16 (`:348-354`)
`bytes.chunks_exact(2) → f32::from(half::bf16::from_le_bytes(c))`. Equivalent to `bits<<16` reinterpreted as f32.

### F16 (`:355-361`)
`bytes.chunks_exact(2) → half::f16::from_le_bytes(c).to_f32()`.

### Q4_0 — 32 weights / 18 bytes (`:362-377`)
```
per block b of 18 bytes:
  d = f16(block[0..2])
  for j in 0..16:                       # QUANT_BLOCK_SIZE/2
    byte = block[2+j]
    lo = (byte & 0x0f) as i32 - 8
    hi = (byte >> 4)   as i32 - 8
    out[b*32 + j]      = (f32)lo * d
    out[b*32 + j + 16] = (f32)hi * d    # ggml SPLIT order, not interleaved
```

### Q8_0 — 32 weights / 34 bytes (`:378-389`)
```
d = f16(block[0..2]); for j in 0..32: out[b*32+j] = (f32)(int8_t)block[2+j] * d
```

### `get_scale_min_k4(j, scales[12]) -> (u8 sc, u8 m)` (`:567-576`)
```
if j < 4: (scales[j] & 63, scales[j+4] & 63)
else:     ( (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4),
            (scales[j+4] >> 4)   | ((scales[j]   >> 6) << 4) )
```
Byte-identical to `sapient-io/src/gguf.rs:347-355` and to the wgpu shader version. **Do not "simplify".**

### Q4_K — 256 weights / 144 bytes — `dequant_q4_k_block` (`:504-527`)
```
d    = f16(block[0..2]); dmin = f16(block[2..4])
scales = block[4..16] (12 B); qs = block[16..144] (128 B)
out_idx = 0; q_off = 0; is = 0
repeat 4 times (256/64):
   (sc1,m1) = get_scale_min_k4(is,   scales); d1 = d*sc1; m1v = dmin*m1
   (sc2,m2) = get_scale_min_k4(is+1, scales); d2 = d*sc2; m2v = dmin*m2
   for l in 0..32:
       out[out_idx+l]    = d1 * (f32)(qs[q_off+l] & 0x0F) - m1v
       out[out_idx+l+32] = d2 * (f32)(qs[q_off+l] >> 4)   - m2v
   out_idx += 64; q_off += 32; is += 2
```
Called from the `Q4_K` arm (`:390-398`) as `dequant_q4_k_block(block, &mut out[b*256..])`.

### Q4_K_R4 — row-interleaved de-permutation (`:399-420`)
Requires 2-D shape. `k = dims.last()`, `nb = k/256` (super-blocks per row). For packed block index `p`:
```
g = p / (4*nb);  rem = p % (4*nb);  b = rem / 4;  r = rem % 4
row = g*4 + r
off = (row*nb + b) * 256
dequant_q4_k_block(block, &mut out[off..])
```
Panics `"Q4_K_R4 tensor must be 2-D"` on empty dims.

### Q5_K — 256 weights / 176 bytes (`:421-467`) — **PER-ELEMENT high bit**
```
d = f16(block[0..2]); dmin = f16(block[2..4])
scales = block[4..16]; qh = block[16..48] (32 B); ql = block[48..176] (128 B)
out_idx=0; ql_off=0; is=0; u1=1u8; u2=2u8
repeat 4 times (256/64):
   (sc1,m1)=get_scale_min_k4(is,  scales); d1=d*sc1; m1v=dmin*m1
   (sc2,m2)=get_scale_min_k4(is+1,scales); d2=d*sc2; m2v=dmin*m2
   for l in 0..32:
       hi  = (qh[l] & u1) ? 16.0 : 0.0            # qh indexed by l — PER ELEMENT
       out[out_idx+l]    = d1 * ((f32)(ql[ql_off+l] & 0x0F) + hi ) - m1v
       hi2 = (qh[l] & u2) ? 16.0 : 0.0
       out[out_idx+l+32] = d2 * ((f32)(ql[ql_off+l] >> 4)   + hi2) - m2v
   out_idx += 64; ql_off += 32; is += 2
   if is % 8 == 0 { u1 = 1; u2 = 2 } else { u1 <<= 2; u2 <<= 2 }
```
This is the **fixed** form. The historic bug (one `qh[is/8]` byte per 32-element sub-block) is gated by test `q5_k_dequant_high_bits_per_element`. Note: because `is` runs 0,2,4,6 within a super-block, `is % 8 == 0` is only true at `is==0` (start) — in practice `u1/u2` shift `<<2` three times per super-block and reset at each new block's first iteration. Port the code literally.

### Q6_K — 256 weights / 210 bytes — `dequant_q6_k_block` (`:534-565`)
```
ql = block[0..128]; qh = block[128..192]; sc = block[192..208] (16 × i8); d = f16(block[208..210])
out_idx=0; ql_off=0; qh_off=0; sc_base=0
repeat 2 times (256/128):
  for l in 0..32:
     is = l / 16                                       # 0 for l<16, 1 for l>=16
     q1 = (i32)(( ql[ql_off+l]      & 0x0F) | ((qh[qh_off+l]      & 3) << 4)) - 32
     q2 = (i32)(( ql[ql_off+l+32]   & 0x0F) | (((qh[qh_off+l]>>2) & 3) << 4)) - 32
     q3 = (i32)(( ql[ql_off+l]      >> 4  ) | (((qh[qh_off+l]>>4) & 3) << 4)) - 32
     q4 = (i32)(( ql[ql_off+l+32]   >> 4  ) | (((qh[qh_off+l]>>6) & 3) << 4)) - 32
     out[out_idx+l]    = d * (f32)(i8)sc[sc_base+is]   * (f32)q1
     out[out_idx+l+32] = d * (f32)(i8)sc[sc_base+is+2] * (f32)q2
     out[out_idx+l+64] = d * (f32)(i8)sc[sc_base+is+4] * (f32)q3
     out[out_idx+l+96] = d * (f32)(i8)sc[sc_base+is+6] * (f32)q4
  out_idx += 128; ql_off += 64; qh_off += 32; sc_base += 8
```
The `+0/+2/+4/+6` scale offsets with the `is = l/16` split and `sc_base += 8` is the historically-broken indexing — **copy verbatim** (must stay in sync with `sapient-backends-cpu` `dot_q6_k_row_f32` and `sapient-io` `dequantize_q6_k`).

### Q6_K_R4 (`:477-495`)
Same permutation as Q4_K_R4 over 210-byte blocks: `off = ((g*4+r)*nb + b) * 256`, then `dequant_q6_k_block`.

### Fallback arm (`:496`) — comment is WRONG
`_ => self.as_f32_slice().to_vec()` with the comment `// fallback for integer dtypes`. `as_f32_slice()` asserts `dtype == F32` → for I32/I64/U8/Bool this **panics**, it does not convert. Port as an explicit unsupported-dtype error or replicate the panic; do not implement integer conversion.

---

## A3. `DType` — `sapient-core/src/dtype.rs`

`#[allow(non_camel_case_types)] #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)] #[non_exhaustive]` (`:13-16`).
**`#[non_exhaustive]`** = external crates may not exhaustively match nor construct-by-literal-pattern; internally it has no effect. In C++ this is a no-op — just document that new variants may be appended (do **not** rely on a `Count` sentinel across an ABI).

Variants in declaration order (`:16-54`): `F32, F16, BF16, I32, I64, U8, Bool, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, Q4_K_R4, Q6_K_R4`.
(Note the declaration order places `Q4_K_R4`/`Q6_K_R4` **after** `Q6_K` — if you give the C++ enum explicit values for serialization, serde uses the *name*, not the index, so any stable numbering is fine.)

Constants (`:56-69`, re-exported from `lib.rs:13-16`):
```
QUANT_BLOCK_SIZE   = 32     Q4_0_BLOCK_BYTES = 18    Q8_0_BLOCK_BYTES = 34
K_QUANT_BLOCK_SIZE = 256    Q4_K_BLOCK_BYTES = 144   Q5_K_BLOCK_BYTES = 176   Q6_K_BLOCK_BYTES = 210
```

`element_size()` (`:75-93`) — `const`: F32 4, F16 2, BF16 2, I32 4, I64 8, U8 1, Bool 1, **all quant → 0**.
`alignment()` (`:97-107`): F32 4, F16/BF16 2, I32 4, I64 8, U8/Bool 1, **all quant → 2**.
`block_bytes()` (`:111-120`): Q4_0 18, Q8_0 34, Q4_K|Q4_K_R4 144, Q5_K 176, Q6_K|Q6_K_R4 210, else **panic** `"block_bytes() called on non-quantized dtype"`.
`block_numel()` (`:124-132`): Q4_0|Q8_0 → 32, K-quants + R4 → 256, else **panic** `"block_numel() called on non-quantized dtype"`.
`byte_count(numel)` (`:136-145`) — **truncating integer division**:
```
Q4_0        -> numel/32  * 18
Q8_0        -> numel/32  * 34
Q4_K|Q4_K_R4-> numel/256 * 144
Q5_K        -> numel/256 * 176
Q6_K|Q6_K_R4-> numel/256 * 210
_           -> numel * element_size()
```
A `numel` that is not a block multiple silently drops the tail — and `from_quant_bytes`' length check truncates identically, so it still passes.
`is_quantized()` (`:149-160`), `is_float()` (F32|F16|BF16), `is_integer()` (I32|I64|U8|Bool).
`name()` (`:175-192`): `"f32","f16","bf16","i32","i64","u8","bool","q4_0","q8_0","q4_k","q4_k_r4","q5_k","q6_k","q6_k_r4"`. `Display` writes `name()`.
`FromStr` (`:207-229`, lowercases input first):
```
"f32"|"float32"->F32   "f16"|"float16"->F16   "bf16"|"bfloat16"->BF16
"i32"|"int32"  ->I32   "i64"|"int64"  ->I64   "u8"|"uint8"->U8   "bool"->Bool
"q4_0"->Q4_0   "q8_0"->Q8_0
"q4_k"|"q4_k_m"|"q4_k_s"->Q4_K     "q5_k"|"q5_k_m"|"q5_k_s"->Q5_K     "q6_k"->Q6_K
_ -> TypeMismatch{expected:"a valid dtype", got: other}
```
**Asymmetry: `"q4_k_r4"`/`"q6_k_r4"` are emitted by `name()` but NOT parseable** — not round-trippable. `DType::from_str(s)` (`:196-198`) is just `s.parse()`.

ONNX mapping (`:235-270`): `from_onnx_dtype`: 1→F32, 2→U8, 5→I32, 7→I64, 9→Bool, 10→F16, 16→BF16, else `TypeMismatch{expected:"a supported ONNX dtype", got:"ONNX code {n}"}`. `to_onnx_dtype`: inverse; **all quant dtypes → 0**.

---

## A4. `Buffer` / `CpuBuffer` / `BufferHandle` — `sapient-core/src/buffer.rs`

### `trait Buffer: Send + Sync + Debug` (`:20-48`)
```
fn as_bytes(&self) -> &[u8];
fn as_bytes_mut(&mut self) -> &mut [u8];
fn len(&self) -> usize;
fn is_mmap(&self) -> bool { false }        // default impl
fn is_empty(&self) -> bool { self.len()==0 }  // default impl
fn alignment(&self) -> usize;
fn device(&self) -> &str;                  // "cpu" | "cpu-mmap" | …
```
C++: abstract base with `virtual bool is_mmap() const { return false; }` and `virtual std::string_view device() const`.

### `BufferHandle(pub Arc<dyn Buffer>)` (`:53-76`)
Tuple struct, **field `.0` is public** — `Tensor::as_bytes_mut` reaches into `self.buffer.0` for `Arc::get_mut`. Methods: `new(impl Buffer+'static)`, `as_bytes`, `len`, `is_mmap`, `is_empty`. C++: `using BufferHandle = std::shared_ptr<Buffer>` (or a thin wrapper exposing `.get()`).

### `CpuBuffer` (`:84-251`)
```rust
pub struct CpuBuffer { ptr: NonNull<u8>, len: usize, align: usize, layout: Layout }
unsafe impl Send for CpuBuffer {}   // :93
unsafe impl Sync for CpuBuffer {}   // :94
```
Custom `Debug` prints only `{len, align}` (`:96-103`).

**Allocation — `with_capacity(bytes, align)` (`:114-142`)**: `Layout::from_size_align` (err → `AllocationFailed{bytes, align}`), then **`std::alloc::alloc_zeroed(layout)`**, null → `AllocationFailed`. **`bytes == 0` still allocates 1 byte** with the requested alignment but stores `len: 0` (`:116-127`). C++: `std::aligned_alloc(align, round_up(size, align))` + `memset 0`, or `operator new(size, std::align_val_t{align})` + zero-fill. Free must use the matching aligned deallocation.

**Alignment differs per constructor — mirror exactly:**
| ctor | line | alignment |
|---|---|---|
| `zeros(numel, dtype)` | `:107-111` | `dtype.alignment().max(64)` → **always 64** (all dtype alignments ≤ 8) |
| `from_f32_slice(&[f32])` | `:145-153` | **64**, `copy_nonoverlapping` |
| `from_bytes_slice(&[u8])` | `:185-196` | **16** (empty → `with_capacity(0,16)`), `copy_nonoverlapping` |
| `from_f32_vec(Vec<f32>)` | `:158-182` | **4** (`align_of::<f32>()`) — zero-copy |

**`from_f32_vec` zero-copy trick (`:158-182`)**:
```
if data.is_empty() -> with_capacity(0, 4)
len    = data.len()*4
layout = Layout::array::<f32>(data.len())       // NOTE: uses len, not capacity
ptr    = data.as_ptr() as *mut u8
std::mem::forget(data)                          // suppress Vec's Drop
Self { ptr, len, align: 4, layout }
```
→ `Drop` (`:246-251`) calls `alloc::dealloc(ptr, layout)`. **Latent issue:** if the source `Vec` had `capacity > len`, dealloc uses a smaller layout than was allocated (UB per the `GlobalAlloc` contract; benign on malloc). **Do NOT mirror this in C++** — take `std::vector<float>` by move and keep the vector alive inside the buffer (or `release()` via a custom allocator).

Other methods: `as_f32_slice`/`as_f32_slice_mut` (`:199-210`, assert `len%4==0`), `as_ptr`/`as_mut_ptr` (`:213-219`).
`impl Buffer for CpuBuffer` (`:222-244`): `as_bytes`/`as_bytes_mut` = `from_raw_parts(ptr, len)`, `len()`, `alignment()` = stored `align`, `device()` = `"cpu"`. `is_mmap()` inherits the `false` default.

---

## A5. `Shape` — `sapient-core/src/shape.rs`

`pub struct Shape(pub Vec<usize>)` — `Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default` (`:10-11`). Public field `.0`, used directly by `Tensor::t()` / `slice_axis()`.

| fn | line | semantics |
|---|---|---|
| `new(iter)` | `:15-17` | collect |
| `ndim()` | `:21-23` | `len()` |
| `numel()` | `:27-29` | `iter().product()` → **scalar (empty dims) gives 1** |
| `dims()` | `:33-35` | slice |
| `strides()` | `:38-48` | row-major; `ndim==0` → `[]` |
| `scalar()` | `:51-53` | `Shape(vec![])` |
| `is_scalar()` | `:56-58` | `dims.is_empty()` |
| `reshape(new_dims)` | `:61-70` | numel must match, else `ShapeMismatch{expected: self.0, got: new.0}` |
| `broadcast_with(other)` | `:73-102` | NumPy rules — see below |
| `expand_dims(axis)` | `:105-115` | `axis > ndim` → `internal("expand_dims: axis {axis} out of range for rank {n}")`; inserts 1 |
| `squeeze()` | `:118-120` | filters out **all** dims equal to 1 |
| `validate()` | `:123-132` | rejects **only zero dims** → `InvalidGraph("Shape has zero dimension at axis {i}")` (note: `InvalidGraph`, not a shape variant — mirror it) |
| `flat_index(idx)` | `:135-154` | see below |

**`broadcast_with` (`:73-102`)** — right-aligned NumPy:
```
len = max(a.len(), b.len())
for i in 0..len:
   ai = (i < len - a.len()) ? 1 : a[i - (len - a.len())]
   bi = (i < len - b.len()) ? 1 : b[i - (len - b.len())]
   if ai==bi -> ai; elif ai==1 -> bi; elif bi==1 -> ai
   else -> Err(BroadcastError{ lhs: self.0, rhs: other.0 })
```

**`flat_index(idx)` (`:135-154`)** — despite the doc comment saying "byte offset", it returns an **element** offset:
```
if idx.len() != ndim -> Err(RankMismatch{expected: ndim, got: idx.len()})
strides = self.strides()
for i,(ix,st): if ix >= dims[i] -> Err(internal("Index {ix} out of bounds for dim {i} (size {d})"))
               offset += ix * st
```

`Display` (`:157-168`): `"[d0, d1, …]"` (comma+space separated, `"[]"` for scalar).
`From<Vec<usize>>` (`:170-174`), `From<&[usize]>` (`:176-180`). These are what make `impl Into<Shape>` accept `vec![2,3]` at every `Tensor` constructor.

---

## A6. `SapientError` — `sapient-core/src/error.rs`

`#[derive(Debug, Error)] pub enum SapientError` (thiserror). All variants with their exact `#[error]` format strings (`:12-96`):

```
ShapeMismatch{expected:Vec<usize>, got:Vec<usize>}   "Shape mismatch: expected {expected:?}, got {got:?}"
RankMismatch{expected:usize, got:usize}              "Rank mismatch: expected {expected}, got {got}"
TypeMismatch{expected:String, got:String}            "Type mismatch: expected {expected}, got {got}"
BroadcastError{lhs:Vec<usize>, rhs:Vec<usize>}       "Incompatible shapes for broadcasting: {lhs:?} and {rhs:?}"
CyclicGraph                                          "Graph contains a cycle — execution is impossible"
NodeNotFound(String)                                 "Node {0:?} not found in graph"        (Debug-quoted)
InvalidGraph(String)                                 "Graph validation failed: {0}"
ShapeInferenceFailed{op:String, reason:String}       "Shape inference failed for op '{op}': {reason}"
UnsupportedOp{backend:String, op:String}             "Backend '{backend}' does not support op '{op}'"
BackendError{backend:String, message:String}         "Backend error from '{backend}': {message}"
NoBackendAvailable                                   "No suitable backend found for execution"
AllocationFailed{bytes:usize, align:usize}           "Allocation failed: requested {bytes} bytes (alignment {align})"
BufferSizeMismatch{expected:usize, got:usize}        "Buffer size mismatch: expected {expected} bytes, got {got}"
PoolExhausted                                        "Memory pool exhausted — consider increasing pool capacity"
OnnxParseError(String)                               "ONNX parse error: {0}"
GgufParseError(String)                               "GGUF parse error: {0}"
SafetensorsParseError(String)                        "Safetensors parse error: {0}"
UnsupportedFormat(String)                            "Unsupported model format: {0}"
ModelNotFound(String)                                "Model not found at path '{0}'"
Io(#[from] std::io::Error)                           "IO error: {0}"
DeadlineExceeded                                     "Request timed out (deadline exceeded)"
SchedulerShutdown                                    "Batch scheduler is shut down"
UninitializedRuntime                                 "Runtime is not initialized — call Session::new() first"
TelemetryError(String)                               "Telemetry export failed: {0}"
Internal(String)                                     "Internal error: {0}"
```
Note the em-dashes (U+2014) in `CyclicGraph`, `PoolExhausted`, `UninitializedRuntime` — keep them if messages are compared.

- **`From` impls: exactly one** — `#[from] std::io::Error` on `Io` (`:77`). No other `From`.
- `pub type Result<T> = std::result::Result<T, SapientError>;` (`:99`).
- Helper constructors (`:103-124`): `backend(backend, message) -> BackendError`, `unsupported_op(backend, op) -> UnsupportedOp`, `internal(msg) -> Internal`. (All `impl Into<String>`.)
- Note for the FFI layer (CLAUDE.md invariant): the field name is `message` here in `BackendError`; the *FFI* error type must use `reason`. Not relevant to `sapient-core` itself but don't rename either.

C++ shape: a `std::variant`-tagged struct or an exception hierarchy; keep the `to_string()` formats byte-identical if any test compares messages (`error_display` does a `contains` check).

---

## A7. The 22 `sapient-core` unit tests (verified counts: tensor 7, shape 6, dtype 4, buffer 3, error 2)

**`tensor.rs`** (`:745-825`)
1. `zeros_dtype_shape` (`:750`) — `zeros([2,3], F32)`: dims `[2,3]`, dtype F32, numel 6.
2. `from_f32_roundtrip` (`:758`) — `from_f32(6 floats, [2,3])`; `as_f32_slice()` equals the input slice.
3. `reshape_preserves_data` (`:765`) — `[2,3]→[3,2]`: new dims `[3,2]`, data unchanged (shared buffer).
4. `reshape_wrong_numel` (`:774`) — `[2,3].reshape([5])` is `Err`.
5. `transpose_2d` (`:780`) — `zeros([3,4]).t()` has dims `[4,3]`.
6. `byte_size` (`:787`) — `zeros([4,4], F32).byte_size() == 64`.
7. `q5_k_dequant_high_bits_per_element` (`:797`) — builds a 176-B Q5_K block (d=1, dmin=0, all sc=1/m=0 via `scales[4+i]=1, scales[12+i]=1`), sets `qh[5]=0b0000_0001` and `qh[0]=0b0000_0100`, `ql[3]=0x02`; asserts `out.len()==256`, `out[3]==2.0`, `out[5]==16.0`, `out[64]==16.0`, and **every other element == 0.0** — i.e. the 5th bit is per-element, not per-sub-block.

**`shape.rs`** (`:182-224`)
8. `numel` (`:187`) — `[2,3,4]`→24; `Shape::scalar()`→**1**.
9. `strides_row_major` (`:193`) — `[2,3,4]`→`[12,4,1]`.
10. `broadcast` (`:199`) — `[1,3] ⊕ [2,3] = [2,3]`.
11. `broadcast_fail` (`:206`) — `[2,3] ⊕ [2,4]` is `Err`.
12. `reshape` (`:213`) — `[2,3]→[6]` ok.
13. `flat_index` (`:220`) — `[2,3,4].flat_index([1,2,3]) == 12+8+3 == 23`.

**`dtype.rs`** (`:273-319`)
14. `element_sizes` (`:278`) — F32 4, I64 8, Bool 1.
15. `byte_count` (`:285`) — `F32.byte_count(10) == 40`.
16. `from_str_roundtrip` (`:290`) — parses `f32,f16,bf16,i32,i64,u8,bool` to their variants (quant names **not** covered).
17. `onnx_roundtrip` (`:305`) — `from_onnx_dtype(to_onnx_dtype(dt)) == dt` for F32,F16,BF16,I32,I64,U8,Bool.

**`buffer.rs`** (`:253-276`)
18. `zeros_and_read` (`:258`) — `CpuBuffer::zeros(4, F32)`: `len()==16`, all bytes zero.
19. `from_f32_roundtrip` (`:265`) — `from_f32_slice([1..4])`, `as_f32_slice()` equals input.
20. `alignment_guarantee` (`:272`) — `with_capacity(32, 64)`: `as_ptr() as usize % 64 == 0`.

**`error.rs`** (`:126-147`)
21. `error_display` (`:131`) — `ShapeMismatch{[2,3],[2,4]}.to_string()` contains `"Shape mismatch"` and `"[2, 3]"`.
22. `from_io_error` (`:142`) — `io::Error(NotFound,"file missing").into()` matches `SapientError::Io(_)`.

---

# PART B — `sapient-io`

## B1. GGUF parser — `sapient-io/src/gguf.rs`

### B1.1 Header (`parse_header`, `:517-578`)
Reads from a `Cursor<&[u8]>`, **all little-endian**:
```
magic   = u32;  != 0x46554747 ("GGUF" LE) -> GgufParseError("bad GGUF magic")
version = u32;  !(1..=3).contains(v) -> GgufParseError("unsupported GGUF version {v} (expected 1–3)")
tensor_count = u64
kv_count     = u64
for kv_count:      (key, value) = read_kv()        -> HashMap<String, GgufValue>
for tensor_count:  name=string, n_dims=u32, dims=[u64; n_dims], kind=u32, offset=u64
                   GgmlType::from_u32(kind) else GgufParseError("unknown ggml type {kind_raw}")
alignment  = metadata["general.alignment"].as_u64() or DEFAULT_ALIGNMENT (32)
data_start = ceil_div(cursor.position(), alignment) * alignment
```
**Port notes / hazards:**
- v2 and v3 are handled identically; **v1 is accepted but would mis-parse** (real GGUF v1 used u32 counts/lengths, this reader always uses u64). Harmless in practice (no v1 files) but do not "fix" without a parity decision.
- `general.alignment == 0` → division by zero panic at `:575`. Add a guard in C++ or reproduce.
- There is no explicit handling of the GGUF v3-vs-v2 string/array width difference — both are u64 here.
- `GGUF_MAGIC = 0x46554747` (`:82`), `DEFAULT_ALIGNMENT = 32` (`:84`).

### B1.2 Metadata KV type codes (`read_value`, `:789-831`)
| code | read | `GgufValue` |
|---|---|---|
| 0 | u8 | `U8` |
| 1 | u8 as i8 | `I8` |
| 2 | u16 | `U16` |
| 3 | u16 as i16 | `I16` |
| 4 | u32 | `U32` |
| 5 | i32 | `I32` |
| 6 | f32 | `F32` |
| 7 | u8 != 0 | `Bool` |
| 8 | string | `Str` |
| 9 | array: `item_type=u32`, `count=u64`, then per item | item 4 → `ArrayU32`, item 8 → `ArrayStr`, item 6 → `ArrayF32`, **anything else → loop `skip_value` → `Other`** |
| 10 | u64 | `U64` |
| 11 | i64 | `I64` |
| 12 | f64 | `F64` |
| other | **nothing consumed** | `Other` → cursor desync |

`skip_value` (`:833-853`): `0|1|7`→1 B, `2|3`→2 B, `4..=6`→4 B, `8`→string, `10..=12`→8 B, **`_ => {}` consumes 0 bytes** → a nested array (item_type 9) desyncs the cursor. Reproduce or harden explicitly (a deliberate choice for the port; note it in PARITY.md).

`GgufValue` enum (`:184-201`): `U8,I8,U16,I16,U32,I32,F32,Bool,Str,U64,I64,F64,ArrayU32,ArrayStr,ArrayF32,Other`.
Accessors (`:203-246`) with their widening rules:
```
as_u32():  U32->v | U64->v as u32 (truncating) | I32 if >=0 -> v | else None
as_u64():  U64->v | U32->v as u64 | else None
as_f32():  F32->v | F64->v as f32 | else None
as_f64():  F64->v | F32->v as f64 | else None
as_bool(): Bool->v | U8-> v!=0 | else None
as_str():  Str->&s | else None
```
(No array accessors — consumers match on the variant.)

Low-level readers (`:855-909`): `read_u8/u16/u32/u64/i32/i64/f32/f64` = fixed-size LE `read_exact`, any io error → `GgufParseError(e.to_string())`. `read_gguf_string` (`:903-909`) = `u64 len` + `len` bytes + `String::from_utf8` (invalid UTF-8 → `GgufParseError`). C++: `std::string` from bytes (validate UTF-8 if you want parity on the error path).

### B1.3 Tensor info + ggml type table
```rust
struct GgufTensorInfo { name: String, dims: Vec<usize>, kind: GgmlType, offset: u64 }   // :250-255
```
`dims` are stored in **GGUF/ggml order (ne0 fastest-varying)** — i.e. a linear weight is `[in, out]`. **No flip happens in sapient-io.**

`enum GgmlType` (`:88-106`) with `from_u32` (`:111-129`), `block_size` (`:131-137`), `type_size` (`:139-155`), `to_sapient_dtype` (`:160-169`):

| code | GgmlType | block_size | type_size | `to_sapient_dtype` | fate in `make_tensor(_mmap)` |
|---|---|---|---|---|---|
| 0 | F32 | 1 | 4 | `None` | dequant → **F32 heap** |
| 1 | F16 | 1 | 2 | `None` | dequant → **F32 heap (2× expansion)** |
| 2 | Q4_0 | 32 | 18 | `Some(Q4_0)` | **kept as blocks** |
| 3 | Q4_1 | 32 | 20 | `None` | `dequantize_to_f32` **errors** "unsupported GGUF quantization type Q4_1" |
| 6 | Q5_0 | 32 | 22 | `None` | dequant → **re-quantize to Q8_0 heap** |
| 7 | Q5_1 | 32 | 24 | `None` | errors |
| 8 | Q8_0 | 32 | 34 | `Some(Q8_0)` | **kept as blocks** |
| 9 | Q8_1 | 32 | 36 | `None` | errors |
| 10 | Q2_K | 256 | 84 | `None` | errors |
| 11 | Q3_K | 256 | 110 | `None` | errors |
| 12 | Q4_K | 256 | 144 | `Some(Q4_K)` | **kept as blocks** |
| 13 | Q5_K | 256 | 176 | `Some(Q5_K)` | **kept as blocks** |
| 14 | Q6_K | 256 | 210 | `Some(Q6_K)` | **kept as blocks** |
| 30 | BF16 | 1 | 2 | `None` | dequant → **F32 heap (2× expansion)** |
| any other | — | — | — | — | `from_u32` → `None` → header parse error "unknown ggml type {n}" |

`QK_K = 256` (`:108`).
`tensor_byte_len(kind, numel)` (`:172-178`): `F32|F16|BF16 → numel * type_size`, else `(numel / block_size) * type_size` (truncating).

**Stale doc comments — trust the code, not the prose:** `gguf.rs:582`, `:634-635` and `:717-719` (and CLAUDE.md) claim K-quants / F16 / BF16 are dequantized to F32 on the mmap path. The `to_sapient_dtype` table (`:160-169`) keeps **all five** of Q4_0/Q8_0/Q4_K/Q5_K/Q6_K zero-copy in both paths.

### B1.4 Offsets
`start = data_start + info.offset` (`:586`, `:639`). Bounds check `start + byte_len > file_len` → `GgufParseError("tensor '{name}': data range [{start}..{end}] exceeds file size {len}")`.
Shape: `info.dims.is_empty() → Shape::new([1])`, else `Shape::new(info.dims.clone())` (`:598-602`, `:652-656`). `numel = dims.iter().product().max(1)`.

## B2. `load_tensors_with_metadata` — heap path (`:752-766`)
```
bytes = std::fs::read(path)            -> ModelNotFound("{path}: {e}") on failure   // WHOLE FILE into heap
(metadata, infos, data_start) = parse_header(&bytes)
for info: tensors[info.name] = make_tensor(info, &bytes, data_start)
```
This is the ~2× peak-RSS path (whole file buffer + per-tensor copies) — the reason MoE/large models default to mmap.
`load_tensors(path)` (`:743-746`) = this, discarding metadata. `tensors_from_bytes(bytes)` (`:769-777`) = same over a caller-supplied slice (test helper).

### `make_tensor(info, bytes, data_start)` (`:636-676`)
```
raw = bytes[start .. start+byte_len]
if let Some(dtype) = kind.to_sapient_dtype():
     Tensor::from_quant_bytes(raw, shape, dtype)          // COPIES raw into a CpuBuffer (align 16)
else:
     f32_data = dequantize_to_f32(kind, raw, numel)?      // may Err for Q4_1/Q5_1/Q8_1/Q2_K/Q3_K
     if kind.block_size() > 1 && numel % 32 == 0:
         Tensor::from_quant_bytes(&quantize_to_q8_0(&f32_data), shape, DType::Q8_0)
     else:
         Tensor::from_f32(&f32_data, shape)               // F32/F16/BF16 land here
```
All errors wrapped as `GgufParseError(e.to_string())`.
**`block_size() > 1` is true only for quantized ggml types**, so F32/F16/BF16 always take the F32 branch; in practice **Q5_0 is the only type that reaches the re-quantize branch** (all other `block_size>1 && to_sapient_dtype()==None` types error out in `dequantize_to_f32`).

### `quantize_to_q8_0(&[f32]) -> Vec<u8>` (`:289-301`) — the exact algorithm
```
for each chunk of exactly 32 f32 (chunks_exact(32) — a non-multiple-of-32 tail is DROPPED):
    amax = fold(0.0, |a,v| a.max(v.abs()))        # f32::max — NaN-propagating per Rust's max semantics
    d    = amax / 127.0
    id   = (d > 0.0) ? 1.0/d : 0.0
    emit  half::f16::from_f32(d).to_le_bytes()    # 2 bytes, RNE
    for v in chunk: emit (int8_t)((v*id).round().clamp(-127.0, 127.0))   # roundf, then clamp, then cast
Output size = data.len()/32 * 34
```
C++: `float amax = 0; for (v) amax = std::max(amax, std::fabs(v));` then `roundf`, `std::clamp(-127.f,127.f)`, `static_cast<int8_t>`. Note the order is **round-then-clamp**.

### `dequantize_to_f32(kind, bytes, numel)` (`:478-505`)
```
F32  -> bytes[..numel*4].chunks_exact(4) |> f32::from_le_bytes
F16  -> bytes[..numel*2].chunks_exact(2) |> f16_to_f32(u16 LE)
BF16 -> bytes[..numel*2].chunks_exact(2) |> f32::from(half::bf16::from_le_bytes)
Q4_0 -> dequantize_q4_0   Q5_0 -> dequantize_q5_0   Q8_0 -> dequantize_q8_0
Q4_K -> dequantize_q4_k   Q5_K -> dequantize_q5_k   Q6_K -> dequantize_q6_k
other -> Err(GgufParseError("unsupported GGUF quantization type {other:?}"))
```
`f16_to_f32(bits: u16)` (`:507-509`) = `half::f16::from_bits(bits).to_f32()`.

Per-format dequant functions (all mirror the `sapient-core` arms except where noted):
- `dequantize_q4_0` (`:259-283`) — 18 B blocks, split nibble order, `blocks = data.len()/18`, bounds-guarded writes.
- `dequantize_q8_0` (`:303-318`) — 34 B blocks, `(i8)byte * scale`.
- **`dequantize_q5_0` (`:320-345`)** — the only place Q5_0 is decoded; 22 B blocks:
  ```
  scale = f16(data[base..base+2]);  qh = u32 LE(data[base+2..base+6])
  for j in 0..16:
     byte = data[base+6+j]
     xh_0 = ((qh >> j) << 4) & 0x10
     xh_1 =  (qh >> (j+12))  & 0x10
     x0 = (i32)((byte & 0x0F) as u32 | xh_0) - 16
     x1 = (i32)((byte >> 4)   as u32 | xh_1) - 16
     out[b*32 + j]      = x0 * scale
     out[b*32 + j + 16] = x1 * scale
  ```
- `get_scale_min_k4` (`:347-355`) — identical to `sapient-core`'s.
- `dequantize_q4_k` (`:357-389`) — same arithmetic as `dequant_q4_k_block`, written with a running `out_idx++` (low nibbles 0..32 then high nibbles 0..32 per 64-group) — **equivalent output**.
- `dequantize_q6_k` (`:436-476`) — byte-identical to `dequant_q6_k_block` incl. the `+0/+2/+4/+6` / `sc_base += 8` indexing.
- **`dequantize_q5_k` (`:391-434`) — DIVERGENT / DEAD.** It reads `let qh_byte = qh[is / 8];` at `:414` (one byte per 32-element sub-block) — the **pre-fix** formula that `sapient-core` `tensor.rs:448-452` replaced with per-element `qh[l]`. It is **unreachable**: `GgmlType::Q5_K.to_sapient_dtype()` returns `Some(DType::Q5_K)`, so neither `make_tensor` nor `make_tensor_mmap` ever calls it. **Recommendation for the port: drop it, or port the `sapient-core` per-element form.** Do not port the `qh[is/8]` version.

### B2.1 The `[in,out] → [out,in]` flip is NOT in sapient-io
It lives in `sapient-models/src/gguf_weights.rs:61-96` (`map_gguf_tensors_to_hf`): for keys ending `.weight` with `ndim()==2`, `tensor.reshape(Shape::new([dims[1], dims[0]]))`. It is a **`reshape` (pure metadata), not a `t()`** — no data movement, because ggml's ne0 is fastest-varying so the row-major `[ne1, ne0]` interpretation is already the HF layout. The C++ `sapient::io` target must leave dims in GGUF order; the flip belongs to the models layer.

## B3. `load_tensors_mmap` — mmap path (`:720-737`)
```
file = File::open(path)                -> ModelNotFound("{path}: {e}")
mmap = unsafe { Mmap::map(&file) }     -> GgufParseError("mmap failed: {e}")
mmap = Arc::new(mmap)
(metadata, infos, data_start) = parse_header(mmap.as_ref())
for info: tensors[name] = make_tensor_mmap(info, &mmap, data_start)
```
memmap2 API used: **`memmap2::Mmap` + `unsafe Mmap::map(&File)`** — a whole-file read-only private mapping, no offset/length variant, no `MmapOptions`, no `advise()`/`madvise` calls anywhere. C++ equivalent: `mmap(NULL, st_size, PROT_READ, MAP_PRIVATE, fd, 0)` (POSIX) / `CreateFileMapping`+`MapViewOfFile` (Windows), refcounted by a `shared_ptr` holding the mapping + fd.

### `MmapBuffer` (`:38-78`)
```rust
struct MmapBuffer { mmap: Arc<Mmap>, offset: usize, len: usize }   // :38-42
unsafe impl Send/Sync for MmapBuffer {}                            // :51-52
impl fmt::Debug -> "MmapBuffer(offset={}, len={})"                 // :44-48
impl Buffer:
   as_bytes()      = &self.mmap[offset .. offset+len]              // :55-57
   is_mmap()       = true                                          // :59-61
   as_bytes_mut()  = panic!("MmapBuffer is read-only — model weights cannot be mutated in-place")   // :63-65
   len()           = self.len
   alignment()     = 32                                            // HARDCODED, regardless of actual ptr alignment
   device()        = "cpu-mmap"
```
**Two-level offset to mirror exactly:** `MmapBuffer.offset = data_start + info.offset` and the `Tensor.offset` passed to `from_buffer` is **`0`** (`:606-612`). So a quant tensor's `as_bytes()` = `mmap[buf.offset .. buf.offset + byte_count(numel)]` (the `Tensor` bound applies on top of the buffer's own `len`, and here they coincide).

`Buffer::is_mmap()` is the gate used elsewhere (Q4_K_R4 repack skips mmap tensors; `as_bytes_mut` would panic on them).

### `make_tensor_mmap(info, mmap, data_start)` (`:583-632`)
```
if let Some(dtype) = kind.to_sapient_dtype():
     buf = MmapBuffer{ mmap: Arc::clone(mmap), offset: start, len: byte_len }   // ZERO-COPY
     Tensor::from_buffer(shape, dtype, BufferHandle::new(buf), 0)
else:
     raw = &mmap[start .. start+byte_len]                 // borrow from the mapping, no heap copy of the file
     f32_data = dequantize_to_f32(kind, raw, numel)?
     if kind.block_size() > 1 && numel % 32 == 0: from_quant_bytes(quantize_to_q8_0(&f32_data), shape, Q8_0)
     else:                                        from_f32(&f32_data, shape)
```
Identical decision tree to `make_tensor`. Net effect (both paths):
- **Zero-copy views (mmap path only):** Q4_0, Q8_0, Q4_K, Q5_K, Q6_K.
- **Heap CpuBuffer copies:** the same five on the heap path; Q5_0 → Q8_0 blocks on both; F32/F16/BF16 → F32 on both.

## B4. `parse_metadata_only` (`:701-709`)
```
file = File::open(path)               -> ModelNotFound("{path}: {e}")
mmap = unsafe{Mmap::map(&file)}       -> GgufParseError("mmap failed for header read: {e}")
(metadata, _, _) = parse_header(mmap.as_ref())
return metadata                        // HashMap<String, GgufValue>; the mmap is DROPPED here
```
Zero tensor allocation. Note it still parses (and discards) the full tensor-info list.

`GgufLoader::load(path) -> Graph` (`:684-694`) is the **dead IR path**: `fs::read` + `make_tensor` per tensor into `Graph::add_constant`.

## B5. safetensors — `sapient-io/src/safetensors.rs`

### Header struct (`:21-26`)
```rust
#[derive(Debug, Deserialize)]
struct StMeta { dtype: String, shape: Vec<usize>, data_offsets: [usize; 2] }
```

### `from_bytes(bytes) -> HashMap<String, Tensor>` (`:43-133`)
```
if bytes.len() < 8 -> SafetensorsParseError("file too short")
header_len = u64 LE(bytes[..8]);  header_end = 8 + header_len
if header_end > bytes.len() -> SafetensorsParseError("header overflows file")
header_json = str::from_utf8(bytes[8..header_end])           -> SafetensorsParseError(e)
header_raw: HashMap<String, serde_json::Value> = serde_json::from_str(header_json)   -> SafetensorsParseError(e)
data_section = &bytes[header_end..]
for (name, value) in &header_raw:
    if name == "__metadata__": continue
    meta: StMeta = serde_json::from_value(value.clone())      -> SafetensorsParseError("tensor '{name}': {e}")
    dtype = match meta.dtype.as_str() { "F32"->F32, "F16"->F16, "BF16"->BF16, "I32"->I32,
                                        "I64"->I64, "U8"->U8, "BOOL"->Bool,
                                        other -> Err(SafetensorsParseError("unknown dtype '{other}'")) }
    [start, end] = meta.data_offsets
    if end > data_section.len() -> SafetensorsParseError("tensor '{name}' data out of bounds")
    raw   = &data_section[start..end]
    shape = Shape::new(meta.shape.clone())
    tensor = match dtype {
        F32  -> Tensor::from_f32(&raw.chunks_exact(4).map(f32::from_le_bytes).collect::<Vec<f32>>(), shape)
        BF16 -> Tensor::from_bf16_bytes(raw, shape)
        F16  -> Tensor::from_f16_bytes(raw, shape)
        other-> Err(SafetensorsParseError("unsupported safetensors dtype '{other}' for tensor '{name}'"))
    }
    tensors.insert(name.clone(), tensor)
```

**serde_json types used:** exactly two — `HashMap<String, serde_json::Value>` for the outer map (so unknown keys / `__metadata__` don't break parsing), then `serde_json::from_value::<StMeta>(value.clone())` per entry. C++ mapping: `nlohmann::json` object iteration + a per-entry struct decode; note the **`.clone()`** means the port can just read fields in place.

**Findings for the port:**
- **Nothing is zero-copy, despite the module doc.** `load()` (`:33-41`) mmaps the file, calls `from_bytes(&mmap)`, and the `Mmap` is **dropped at the end of `load()`** — so every tensor must be (and is) heap-copied. `load()` and `from_bytes()` are byte-equivalent in result.
- **The F32 path copies twice**: `chunks_exact(4)` builds a `Vec<f32>`, then `Tensor::from_f32` copies it into a `CpuBuffer` (align 64). The C++ port should copy once (or use the `from_f32_vec` move path) — this is a safe, behaviour-preserving improvement.
- **BF16/F16 keep RAW BYTES** (`from_bf16_bytes` / `from_f16_bytes` → `CpuBuffer::from_bytes_slice`, align 16), dtype preserved as `BF16`/`F16`. Conversion to f32 is deferred to `Tensor::to_f32_vec()`/`to_f32_cow()` at compute time. **They are not converted at load.**
- **I32/I64/U8/BOOL** are accepted by the dtype-string match but then hit the `other =>` arm and **error out**. So the reachable dtype set is {F32, F16, BF16}.
- **Bounds checking is incomplete**: only `end > data_section.len()` is checked. `start > end`, or `start > len`, panics on the slice. There is no check that `end - start == shape.numel() * element_size` — but `Tensor::from_f32` (numel check) and `from_bf16_bytes`/`from_f16_bytes` (`len == numel*2` check) catch the mismatch as a `ShapeMismatch` wrapped into `SafetensorsParseError`.
- Iteration order is `HashMap` order (non-deterministic) — irrelevant since the result is also a map.
- `load_as_graph` (`:136-143`) is the dead IR path.  `load_tensors` (`:146-148`) is a plain alias for `load`.

## B6. `sapient-io/src/lib.rs`
```
pub mod gguf; pub mod onnx; pub mod safetensors;
pub use gguf::{GgufLoader, GgufValue}; pub use onnx::OnnxLoader; pub use safetensors::SafetensorsLoader;
load_graph(path)      -> by lowercased extension: "onnx"->OnnxLoader::load, "gguf"->GgufLoader::load,
                         _ -> Err(UnsupportedFormat(ext))            // :21-33  (dead IR path)
load_gguf(path)       -> GgufLoader::load_tensors(path)              // :36-38
load_safetensors(path)-> SafetensorsLoader::load_tensors(path)       // :41-43
```

## B7. The 2 `sapient-io` unit tests — `gguf.rs:911-935`, module `q8_quant_tests`
1. **`q8_0_quantize_roundtrips_and_sizes`** (`:916`) — 64 f32 values `(i-32)*0.1`; asserts `quantize_to_q8_0(...).len() == 64/32*34` ("Q8_0 = 34 bytes / 32 weights") and that `dequantize_q8_0(q, 64)` returns 64 values each within `0.03` of the original (bound ≈ amax/254).
2. **`q8_0_quantize_handles_all_zeros`** (`:930`) — 32 zeros → 34 bytes, and dequantizing gives all exact `0.0` (guards the `d == 0 → id = 0` branch against NaN).

**No fixtures are used anywhere in `sapient-io`'s unit tests** — both are fully synthetic, and there are no `#[test]`s in `safetensors.rs`, `onnx.rs`, or `lib.rs`. (`tempfile` is a dev-dependency in `sapient-io/Cargo.toml` but no test in these files uses it.)

## B8. External crates — actual usage vs declared

**`sapient-core/Cargo.toml`** declares `thiserror, serde, bytemuck, half, num-traits, tracing` (dev: `approx, proptest`).
| crate | actually used in src? | where / for what |
|---|---|---|
| `thiserror` | **yes** | `error.rs:6` — `#[derive(Error)]` + `#[error("…")]` + `#[from]`. C++: hand-write the message formatting. |
| `serde` | **yes** | `dtype.rs:7`, `shape.rs:7`, `tensor.rs:9-10` — `Serialize/Deserialize` derives + the manual `Tensor` impls |
| `half` | **yes** | `tensor.rs` only — `half::f16::from_le_bytes/.to_f32()`, `half::bf16::from_le_bytes`, `half::f16::from_f32` (test). **The entire f16/bf16 surface the port must reproduce.** C++: F16C intrinsics or a software RNE converter. |
| `bytemuck` | **no** — 0 hits in src | unused; the reinterpret casts are raw `slice::from_raw_parts`. Omit in C++ (use `std::bit_cast`/`memcpy`). |
| `num-traits` | **no** — 0 hits | unused |
| `tracing` | **no** — 0 hits in these 6 files | unused in sapient-core |

**`sapient-io/Cargo.toml`** declares `sapient-core, sapient-ir, thiserror, tracing, serde, serde_json, memmap2, bytemuck, half, ordered-float` (dev: `tempfile`).
| crate | used where (gguf/safetensors/lib only) |
|---|---|
| `memmap2` | `gguf.rs:24` + `safetensors.rs:12` — **only `Mmap` + `unsafe Mmap::map(&File)`**. No `MmapOptions`, no offsets, no advise. |
| `serde_json` | `safetensors.rs:61, 71` — `serde_json::Value`, `from_str`, `from_value`. Not used in gguf.rs (GGUF KV is binary). |
| `half` | `gguf.rs:295` (`f16::from_f32(d).to_le_bytes()`), `:490` (`bf16::from_le_bytes`), `:508` (`f16::from_bits(bits).to_f32()`) |
| `serde` | `safetensors.rs:13` — `#[derive(Deserialize)]` on `StMeta` only |
| `sapient-ir` | `gguf.rs:28/684-694`, `safetensors.rs:17/136-143`, `lib.rs:16/21-33` — **only the dead IR `Graph` path** (sub-project 8). The C++ `sapient::io` target can omit `Graph` entirely and drop `GgufLoader::load`, `SafetensorsLoader::load_as_graph`, `load_graph`. |
| `bytemuck` | **unused** in gguf/safetensors/lib (0 hits) |
| `ordered-float` | **only `onnx.rs:428,429,453`** — not needed once ONNX is deferred |
| `tracing` | **only `onnx.rs:574`** — not needed |
| `thiserror` | **unused** in src (0 hits) — errors come from `sapient_core::SapientError` |

→ For the deferred-ONNX C++ port, the third-party surface of `sapient::io` collapses to: an mmap wrapper, `nlohmann/json`, and an f16/bf16 conversion helper (shared with `sapient::core`).

---

## C. Cross-cutting checklist for the C++ implementer

1. **Three copies of Q5_K/Q6_K dequant must stay in sync** (CLAUDE.md invariant): `sapient-core/src/tensor.rs` (§A2), `sapient-backends-cpu/src/kernels/quant.rs`, and `sapient-io/src/gguf.rs`. Today the `sapient-io` Q5_K copy is stale-but-unreachable (§B2 last bullet) — decide once and record in `docs/PARITY.md`.
2. **`element_size() == 0` for quant dtypes** is load-bearing: any C++ code doing `offset * element_size()` inherits the `slice_axis` trap. Consider a `debug_assert(!dtype.is_quantized())` at that site while keeping release behaviour identical.
3. **`as_bytes()` boundedness** (quant bounded / float unbounded) is what makes zero-copy expert views safe — do not "unify" the two branches.
4. `Arc::get_mut` semantics ≠ `shared_ptr` alone: you need an exclusivity check that also accounts for weak refs, or a design with no weak refs.
5. Reproduce `CpuBuffer`'s per-constructor alignments (64 / 64 / 16 / 4) — a different alignment changes nothing numerically but will change any pointer-alignment assertions and possibly SIMD load selection downstream.
6. Everything is **little-endian, `-ffp-contract=off`, `std::roundf` (not rint), RNE f16** — see §0.
