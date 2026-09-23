// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-core/src/tensor.rs. Invariants that later layers rely on:
//   * `bytes()` is BOUNDED by byte_count(numel) for quantized dtypes and UNBOUNDED (to the end of the
//     buffer) for float dtypes — zero-copy MoE expert views depend on the quant bound (CLAUDE.md).
//   * `bytes_mut()` is bounded for all dtypes and needs exclusive ownership of the buffer.
//   * `reshape`/`t()`/`slice_axis` are views (shared buffer); `slice_axis` multiplies element
//     strides by element_size(), which is 0 for quantized dtypes (documented Rust trap, kept).
//   * `strides` are in elements, `offset` is in bytes.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "sapient/core/buffer.hpp"
#include "sapient/core/dtype.hpp"
#include "sapient/core/error.hpp"
#include "sapient/core/shape.hpp"

namespace sapient::core {

/// Rust `Cow<'_, [f32]>`: a borrowed view when the tensor is already F32, else an owned copy.
///
/// The Tensor that produced this Cow must outlive it (the borrowed `view` arm points into the
/// Tensor's buffer), and this Cow must outlive any span returned by `get()` (the owned arm's span
/// points into `owned`). `get()` is therefore only callable on an lvalue Cow — calling it on a
/// temporary (e.g. `t.to_f32_cow().get()`) would return a span into memory that is destroyed at
/// the end of that full expression, which is undefined behaviour on the owned arm.
struct F32Cow {
    std::span<const float> view;
    std::vector<float> owned;
    bool is_owned{false};
    std::span<const float> get() const& { return is_owned ? std::span<const float>(owned) : view; }
    std::span<const float> get() && = delete;
};

class Tensor {
public:
    static Result<Tensor> zeros(Shape shape, DType dtype);
    static Result<Tensor> from_f32_vec(std::vector<float>&& data, Shape shape); // zero-copy (moved)
    static Result<Tensor> from_f32(std::span<const float> data, Shape shape);   // copy, align 64
    static Result<Tensor> from_bf16_bytes(std::span<const uint8_t> data, Shape shape);
    static Result<Tensor> from_f16_bytes(std::span<const uint8_t> data, Shape shape);
    static Result<Tensor> from_quant_bytes(std::span<const uint8_t> data, Shape shape, DType dtype);
    static Result<Tensor> scalar_f32(float v);
    static Result<Tensor> from_buffer(Shape shape, DType dtype, BufferHandle buffer, size_t offset);

    const Shape& shape() const { return shape_; }
    DType dtype() const { return dtype_; }
    size_t ndim() const { return shape_.ndim(); }
    size_t numel() const { return shape_.numel(); }
    std::span<const size_t> strides() const { return strides_; }
    /// Read-only access (the twin of Rust's `&Arc<dyn Buffer>` deref): `len()`, `is_mmap()`,
    /// `alignment()`, `device()`, `bytes()`. Deliberately NOT `bytes_mut()` — that needs a
    /// `BufferHandle` with `use_count()==1`, which a const reference cannot prove; use
    /// `share_buffer()` to build a second handle if you need to check/act on that.
    const Buffer& buffer() const { return *buffer_; }
    /// The twin of Rust's `Arc::clone`: a new handle sharing this tensor's buffer, for building
    /// views via `from_buffer`. Bumps `use_count`, so `bytes_mut()` on either tensor errors until
    /// the other handle is dropped — exactly `Arc::get_mut`'s exclusivity rule.
    BufferHandle share_buffer() const { return buffer_; }
    size_t offset() const { return offset_; }
    bool is_scalar() const { return shape_.is_scalar() || numel() == 1; }
    bool is_contiguous() const { return strides_ == shape_.strides() && offset_ == 0; }
    bool is_mmap() const { return buffer_->is_mmap(); }

    std::span<const uint8_t> bytes() const;
    std::span<const uint8_t> quant_blocks() const; // panics unless is_quantized(dtype)
    std::span<const float> f32_slice() const;      // panics unless F32 and bytes().size() % 4 == 0;
                                                   // unbounded (mirrors Rust's `as_f32_slice`,
                                                   // tensor.rs:271-278)
    std::vector<float> to_contiguous_f32_vec() const;
    F32Cow to_f32_cow() const;
    std::vector<float> to_f32_vec() const;
    Result<Tensor> to_f32_tensor() const;

    Result<std::span<uint8_t>> bytes_mut();
    Result<std::span<float>> f32_slice_mut();

    Result<Tensor> reshape(Shape new_shape) const;
    Result<Tensor> t() const;
    Result<Tensor> slice_axis(size_t axis, size_t start, size_t end) const;

    size_t byte_size() const { return byte_count(dtype_, numel()); }
    std::string to_string() const; // "Tensor(shape=[2, 3], dtype=f32, device=cpu)"

private:
    Tensor(
        Shape shape, DType dtype, std::vector<size_t> strides, BufferHandle buffer, size_t offset)
        : shape_(std::move(shape)), dtype_(dtype), strides_(std::move(strides)),
          buffer_(std::move(buffer)), offset_(offset) {}
    Shape shape_;
    DType dtype_{DType::F32};
    std::vector<size_t> strides_; // elements
    BufferHandle buffer_;
    size_t offset_{0}; // bytes
};

struct TensorMeta {
    Shape shape;
    DType dtype;
    static TensorMeta of(const Tensor& t) { return {t.shape(), t.dtype()}; }
};

} // namespace sapient::core
