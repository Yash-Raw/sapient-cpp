// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/core/buffer.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "sapient/core/panic.hpp"

#ifdef _WIN32
#include <malloc.h>
#endif

namespace sapient::core {

namespace {
bool is_pow2(size_t a) {
    return a != 0 && (a & (a - 1)) == 0;
}

void* aligned_zeroed(size_t size, size_t align) {
    const size_t alloc_align = std::max(align, alignof(std::max_align_t)); // posix_memalign floor
#ifdef _WIN32
    void* p = _aligned_malloc(size, alloc_align);
    if (p == nullptr) return nullptr;
#else
    void* p = nullptr;
    if (posix_memalign(&p, alloc_align, size) != 0) return nullptr;
#endif
    std::memset(p, 0, size);
    return p;
}

void aligned_free(void* p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
}
} // namespace

CpuBuffer::~CpuBuffer() {
    if (raw_ != nullptr) aligned_free(raw_);
}

Result<std::shared_ptr<CpuBuffer>> CpuBuffer::with_capacity(size_t bytes, size_t align) {
    if (!is_pow2(align)) return tl::unexpected(Error::allocation_failed(bytes, align));
    std::shared_ptr<CpuBuffer> b(new CpuBuffer());
    const size_t alloc_bytes = bytes == 0 ? 1 : bytes; // Rust allocates 1 byte for an empty buffer
    b->raw_ = aligned_zeroed(alloc_bytes, align);
    if (b->raw_ == nullptr) return tl::unexpected(Error::allocation_failed(bytes, align));
    b->ptr_ = static_cast<uint8_t*>(b->raw_);
    b->len_ = bytes;
    b->align_ = align;
    return b;
}

Result<std::shared_ptr<CpuBuffer>> CpuBuffer::zeros(size_t numel, DType dtype) {
    // Qualified: unqualified `alignment(dtype)` here would resolve to the inherited
    // `Buffer::alignment()` member (name lookup stops at class scope before the
    // free function in the enclosing namespace is considered).
    return with_capacity(byte_count(dtype, numel),
                         std::max(sapient::core::alignment(dtype), size_t{64}));
}

Result<std::shared_ptr<CpuBuffer>> CpuBuffer::from_f32_slice(std::span<const float> v) {
    SAPIENT_TRY_ASSIGN(auto b, with_capacity(v.size() * sizeof(float), 64));
    if (!v.empty()) std::memcpy(b->ptr_, v.data(), v.size() * sizeof(float));
    return b;
}

Result<std::shared_ptr<CpuBuffer>> CpuBuffer::from_bytes_slice(std::span<const uint8_t> v) {
    SAPIENT_TRY_ASSIGN(auto b, with_capacity(v.size(), 16));
    if (!v.empty()) std::memcpy(b->ptr_, v.data(), v.size());
    return b;
}

std::shared_ptr<CpuBuffer> CpuBuffer::from_f32_vec(std::vector<float>&& v) {
    std::shared_ptr<CpuBuffer> b(new CpuBuffer());
    b->vec_ = std::move(v);
    b->len_ = b->vec_.size() * sizeof(float);
    b->align_ = alignof(float);
    b->ptr_ = reinterpret_cast<uint8_t*>(b->vec_.data());
    if (b->ptr_ ==
        nullptr) { // an empty vector may have no storage; Rust falls back to a 1-byte allocation
        b->raw_ = aligned_zeroed(1, alignof(float));
        b->ptr_ = static_cast<uint8_t*>(b->raw_);
    }
    return b;
}

std::span<const float> CpuBuffer::f32s() const {
    if (len_ % sizeof(float) != 0) panic("CpuBuffer::f32s: length is not a multiple of 4");
    return {reinterpret_cast<const float*>(ptr_), len_ / sizeof(float)};
}

std::span<float> CpuBuffer::f32s_mut() {
    if (len_ % sizeof(float) != 0) panic("CpuBuffer::f32s_mut: length is not a multiple of 4");
    return {reinterpret_cast<float*>(ptr_), len_ / sizeof(float)};
}

} // namespace sapient::core
