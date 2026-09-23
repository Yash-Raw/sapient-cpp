// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-core/src/buffer.rs. `Buffer` is the `dyn Buffer` trait; `CpuBuffer`
// owns a zero-filled aligned allocation (or a moved std::vector<float> — the safe twin of
// Rust's `from_f32_vec` layout trick). The mmap buffer lives in sapient::io, as in Rust.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "sapient/core/dtype.hpp"
#include "sapient/core/error.hpp"

namespace sapient::core {

class Buffer {
public:
    virtual ~Buffer() = default;
    virtual std::span<const uint8_t> bytes() const = 0;
    virtual std::span<uint8_t> bytes_mut() = 0;
    virtual size_t len() const = 0;
    virtual bool is_mmap() const { return false; }
    bool is_empty() const { return len() == 0; }
    virtual size_t alignment() const = 0;
    virtual std::string_view device() const = 0; // "cpu" | "cpu-mmap"
};

/// Rust `BufferHandle(Arc<dyn Buffer>)`. Exclusive-mutation checks use `use_count() == 1`;
/// no code in the C++ tree ever takes a `weak_ptr` to a Buffer (keep it that way).
using BufferHandle = std::shared_ptr<Buffer>;

class CpuBuffer final : public Buffer {
public:
    /// Zero-filled, `align`-aligned allocation. `bytes == 0` still allocates one byte (Rust parity)
    /// but reports len 0. Non-power-of-two `align` → AllocationFailed (Rust: Layout error).
    static Result<std::shared_ptr<CpuBuffer>> with_capacity(size_t bytes, size_t align);
    static Result<std::shared_ptr<CpuBuffer>> zeros(size_t numel,
                                                    DType dtype); // align max(alignment(dtype), 64)
    static Result<std::shared_ptr<CpuBuffer>>
    from_f32_slice(std::span<const float> v); // copy, align 64
    static Result<std::shared_ptr<CpuBuffer>>
    from_bytes_slice(std::span<const uint8_t> v);                           // copy, align 16
    static std::shared_ptr<CpuBuffer> from_f32_vec(std::vector<float>&& v); // move, align 4

    std::span<const float> f32s() const; // panics unless len % 4 == 0 (Rust assert)
    std::span<float> f32s_mut();
    const uint8_t* data() const { return ptr_; }
    uint8_t* data() { return ptr_; }

    std::span<const uint8_t> bytes() const override { return {ptr_, len_}; }
    std::span<uint8_t> bytes_mut() override { return {ptr_, len_}; }
    size_t len() const override { return len_; }
    size_t alignment() const override { return align_; }
    std::string_view device() const override { return "cpu"; }

    CpuBuffer(const CpuBuffer&) = delete;
    CpuBuffer& operator=(const CpuBuffer&) = delete;
    ~CpuBuffer() override;

private:
    CpuBuffer() = default;
    uint8_t* ptr_{nullptr};  // points into raw_ or vec_
    void* raw_{nullptr};     // aligned allocation (freed with the matching aligned free)
    std::vector<float> vec_; // storage for the moved-vector constructor
    size_t len_{0};
    size_t align_{0};
};

} // namespace sapient::core
