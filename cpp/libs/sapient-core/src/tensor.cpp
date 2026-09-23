// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/core/tensor.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>

#include "sapient/core/dequant.hpp"
#include "sapient/core/f16.hpp"
#include "sapient/core/panic.hpp"

namespace sapient::core {

namespace {
// Bounds-checked span slicing: mirrors Rust's slice-index panic (`buf[offset..offset+len]`
// panics out of bounds) where plain `std::span::subspan` would be UB instead. Unreachable today
// (every caller already keeps `offset_`/`byte_count` in bounds) — this is a defensive parity
// guard, not a currently-exercised path.
template <class Span> Span checked_subspan(Span s, size_t offset, size_t len) {
    if (offset > s.size() || len > s.size() - offset) panic("Tensor: byte range out of buffer");
    return s.subspan(offset, len);
}
template <class Span> Span checked_subspan(Span s, size_t offset) {
    if (offset > s.size()) panic("Tensor: byte range out of buffer");
    return s.subspan(offset);
}
} // namespace

Result<Tensor> Tensor::zeros(Shape shape, DType dtype) {
    SAPIENT_TRY(shape.validate());
    SAPIENT_TRY_ASSIGN(auto buf, CpuBuffer::zeros(shape.numel(), dtype));
    auto strides = shape.strides();
    return Tensor(std::move(shape), dtype, std::move(strides), std::move(buf), 0);
}

Result<Tensor> Tensor::from_f32_vec(std::vector<float>&& data, Shape shape) {
    SAPIENT_TRY(shape.validate());
    if (data.size() != shape.numel())
        return tl::unexpected(Error::shape_mismatch(shape.dims, {data.size()}));
    auto strides = shape.strides();
    return Tensor(std::move(shape),
                  DType::F32,
                  std::move(strides),
                  CpuBuffer::from_f32_vec(std::move(data)),
                  0);
}

Result<Tensor> Tensor::from_f32(std::span<const float> data, Shape shape) {
    SAPIENT_TRY(shape.validate());
    if (data.size() != shape.numel())
        return tl::unexpected(Error::shape_mismatch(shape.dims, {data.size()}));
    SAPIENT_TRY_ASSIGN(auto buf, CpuBuffer::from_f32_slice(data));
    auto strides = shape.strides();
    return Tensor(std::move(shape), DType::F32, std::move(strides), std::move(buf), 0);
}

namespace {
Result<Tensor> from_half_bytes(std::span<const uint8_t> data, Shape shape, DType dtype) {
    SAPIENT_TRY(shape.validate());
    if (data.size() != shape.numel() * 2)
        return tl::unexpected(Error::shape_mismatch(shape.dims, {data.size() / 2}));
    SAPIENT_TRY_ASSIGN(auto buf, CpuBuffer::from_bytes_slice(data));
    return Tensor::from_buffer(std::move(shape), dtype, std::move(buf), 0);
}
} // namespace

Result<Tensor> Tensor::from_bf16_bytes(std::span<const uint8_t> data, Shape shape) {
    return from_half_bytes(data, std::move(shape), DType::BF16);
}
Result<Tensor> Tensor::from_f16_bytes(std::span<const uint8_t> data, Shape shape) {
    return from_half_bytes(data, std::move(shape), DType::F16);
}

Result<Tensor> Tensor::from_quant_bytes(std::span<const uint8_t> data, Shape shape, DType dtype) {
    if (!is_quantized(dtype))
        // Qualified: unqualified `to_string(dtype)` inside a Tensor member function resolves to the
        // member `Tensor::to_string()` (name lookup stops at class scope), not the free
        // `sapient::core::to_string(DType)` — the same name-hiding trap as buffer.cpp's `alignment()`.
        return tl::unexpected(Error::type_mismatch(
            "a quantized dtype (Q4_0, Q8_0, Q4_K, Q5_K, Q6_K)", sapient::core::to_string(dtype)));
    SAPIENT_TRY(shape.validate());
    const size_t expected = byte_count(dtype, shape.numel());
    if (data.size() != expected)
        return tl::unexpected(Error::shape_mismatch({expected}, {data.size()}));
    SAPIENT_TRY_ASSIGN(auto buf, CpuBuffer::from_bytes_slice(data));
    return from_buffer(std::move(shape), dtype, std::move(buf), 0);
}

Result<Tensor> Tensor::scalar_f32(float v) {
    return from_f32(std::span<const float>(&v, 1), Shape::scalar());
}

Result<Tensor> Tensor::from_buffer(Shape shape, DType dtype, BufferHandle buffer, size_t offset) {
    SAPIENT_TRY(shape.validate());
    const size_t required = byte_count(dtype, shape.numel());
    if (buffer->len() < offset + required)
        return tl::unexpected(Error::buffer_size_mismatch(offset + required, buffer->len()));
    auto strides = shape.strides();
    return Tensor(std::move(shape), dtype, std::move(strides), std::move(buffer), offset);
}

std::span<const uint8_t> Tensor::bytes() const {
    const auto all = buffer_->bytes();
    if (is_quantized(dtype_)) return checked_subspan(all, offset_, byte_count(dtype_, numel()));
    return checked_subspan(all, offset_);
}

std::span<const uint8_t> Tensor::quant_blocks() const {
    if (!is_quantized(dtype_)) panic("as_quant_blocks() called on a non-quantized tensor");
    return bytes();
}

std::span<const float> Tensor::f32_slice() const {
    if (dtype_ != DType::F32) panic("as_f32_slice() called on a non-F32 tensor");
    const auto b = bytes();
    if (b.size() % 4 != 0) panic("Buffer length not a multiple of 4");
    return {reinterpret_cast<const float*>(b.data()), b.size() / 4};
}

std::vector<float> Tensor::to_f32_vec() const {
    const size_t n = numel();
    switch (dtype_) {
    case DType::F32: {
        const auto s = f32_slice();
        return {s.begin(), s.end()};
    } // whole remaining buffer (Rust parity)
    case DType::BF16: {
        const auto b = bytes();
        std::vector<float> out;
        out.reserve(b.size() / 2);
        for (size_t i = 0; i + 1 < b.size(); i += 2)
            out.push_back(bf16_le_to_f32(b.data() + i));
        return out;
    }
    case DType::F16: {
        const auto b = bytes();
        std::vector<float> out;
        out.reserve(b.size() / 2);
        for (size_t i = 0; i + 1 < b.size(); i += 2)
            out.push_back(f16_le_to_f32(b.data() + i));
        return out;
    }
    case DType::Q4_0: {
        std::vector<float> out(n, 0.0f);
        dequant::q4_0(bytes(), n, out.data());
        return out;
    }
    case DType::Q8_0: {
        std::vector<float> out(n, 0.0f);
        dequant::q8_0(bytes(), n, out.data());
        return out;
    }
    case DType::Q4_K: {
        std::vector<float> out(n, 0.0f);
        dequant::q4_k(bytes(), n, out.data());
        return out;
    }
    case DType::Q5_K: {
        std::vector<float> out(n, 0.0f);
        dequant::q5_k(bytes(), n, out.data());
        return out;
    }
    case DType::Q6_K: {
        std::vector<float> out(n, 0.0f);
        dequant::q6_k(bytes(), n, out.data());
        return out;
    }
    case DType::Q4_K_R4: {
        if (shape_.dims.empty()) panic("Q4_K_R4 tensor must be 2-D");
        if (n == 0)
            return {}; // a zero-width view (e.g. slice_axis(last, i, i)) would divide by zero below
        std::vector<float> out(n, 0.0f);
        dequant::q4_k_r4(bytes(), n / shape_.dims.back(), shape_.dims.back(), out.data());
        return out;
    }
    case DType::Q6_K_R4: {
        if (shape_.dims.empty()) panic("Q6_K_R4 tensor must be 2-D");
        if (n == 0)
            return {}; // a zero-width view (e.g. slice_axis(last, i, i)) would divide by zero below
        std::vector<float> out(n, 0.0f);
        dequant::q6_k_r4(bytes(), n / shape_.dims.back(), shape_.dims.back(), out.data());
        return out;
    }
    default:
        panic("to_f32_vec: integer dtypes are not convertible (Rust panics here too)");
    }
}

F32Cow Tensor::to_f32_cow() const {
    F32Cow c;
    if (dtype_ == DType::F32) {
        c.view = f32_slice();
        c.is_owned = false;
    } else {
        c.owned = to_f32_vec();
        c.is_owned = true;
    }
    return c;
}

std::vector<float> Tensor::to_contiguous_f32_vec() const {
    const size_t n = numel();
    if (is_contiguous()) {
        if (dtype_ == DType::F32) {
            const auto s = checked_subspan(f32_slice(), 0, n);
            return {s.begin(), s.end()};
        }
        auto v = to_f32_vec();
        v.resize(std::min(n, v.size()));
        return v;
    }
    const std::vector<float> raw = to_f32_vec(); // F32: whole slice; others: dequantised
    const auto& dims = shape_.dims;
    std::vector<float> out(n, 0.0f);
    for (size_t flat = 0; flat < n; ++flat) {
        size_t rem = flat, src = 0;
        for (size_t d = dims.size(); d-- > 0;) {
            const size_t idx = rem % dims[d];
            rem /= dims[d];
            src += idx * strides_[d];
        }
        out[flat] = src < raw.size() ? raw[src] : 0.0f; // Rust: raw.get(src).unwrap_or(&0.0)
    }
    return out;
}

Result<Tensor> Tensor::to_f32_tensor() const {
    if (dtype_ == DType::F32) return *this;
    return from_f32(to_f32_vec(), shape_);
}

Result<std::span<uint8_t>> Tensor::bytes_mut() {
    if (buffer_.use_count() != 1)
        return tl::unexpected(Error::internal("Cannot mutate shared tensor buffer"));
    const size_t end = offset_ + byte_count(dtype_, numel());
    return checked_subspan(buffer_->bytes_mut(), offset_, end - offset_);
}

Result<std::span<float>> Tensor::f32_slice_mut() {
    if (dtype_ != DType::F32) return tl::unexpected(Error::internal("Tensor dtype is not F32"));
    if (buffer_.use_count() != 1)
        return tl::unexpected(Error::internal("Cannot mutate shared tensor buffer"));
    auto b = buffer_->bytes_mut().subspan(offset_);
    if (b.size() % 4 != 0)
        return tl::unexpected(Error::internal("Buffer length not a multiple of 4"));
    return std::span<float>(reinterpret_cast<float*>(b.data()), b.size() / 4);
}

Result<Tensor> Tensor::reshape(Shape new_shape) const {
    SAPIENT_TRY_ASSIGN(Shape ns, shape_.reshape(std::move(new_shape.dims)));
    auto strides = ns.strides();
    return Tensor(std::move(ns), dtype_, std::move(strides), buffer_, offset_);
}

Result<Tensor> Tensor::t() const {
    if (ndim() != 2) return tl::unexpected(Error::internal("t() requires a 2-D tensor"));
    Shape s({shape_.dims[1], shape_.dims[0]});
    std::vector<size_t> st = {strides_[1], strides_[0]};
    return Tensor(std::move(s), dtype_, std::move(st), buffer_, offset_);
}

Result<Tensor> Tensor::slice_axis(size_t axis, size_t start, size_t end) const {
    if (axis >= ndim()) return tl::unexpected(Error::internal("slice axis out of bounds"));
    if (start > end || end > shape_.dims[axis])
        return tl::unexpected(Error::internal("slice range out of bounds"));
    Shape s = shape_;
    s.dims[axis] = end - start;
    // Rust parity: element_size() is 0 for quantized dtypes, so the offset does not move (documented trap).
    assert(!is_quantized(dtype_) || start == 0);
    const size_t off = offset_ + start * strides_[axis] * element_size(dtype_);
    return Tensor(std::move(s), dtype_, strides_, buffer_, off);
}

std::string Tensor::to_string() const {
    return "Tensor(shape=" + shape_.to_string() + ", dtype=" + std::string(name(dtype_)) +
           ", device=" + std::string(buffer_->device()) + ")";
}

} // namespace sapient::core
