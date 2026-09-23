// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "sapient/core/f16.hpp"
#include "sapient/core/tensor.hpp"

using namespace sapient::core;

// Rust: zeros_dtype_shape
TEST(Tensor, zeros_dtype_shape) {
    auto t = Tensor::zeros({2, 3}, DType::F32);
    ASSERT_TRUE(t.has_value());
    EXPECT_EQ(t->shape().dims, (std::vector<size_t>{2, 3}));
    EXPECT_EQ(t->dtype(), DType::F32);
    EXPECT_EQ(t->numel(), 6u);
}
// Rust: from_f32_roundtrip
TEST(Tensor, from_f32_roundtrip) {
    const float data[] = {1, 2, 3, 4, 5, 6};
    auto t = Tensor::from_f32(data, {2, 3});
    ASSERT_TRUE(t.has_value());
    const auto s = t->f32_slice();
    ASSERT_EQ(s.size(), 6u);
    for (size_t i = 0; i < 6; ++i)
        EXPECT_EQ(s[i], data[i]);
}
// Rust: reshape_preserves_data
TEST(Tensor, reshape_preserves_data) {
    const float data[] = {1, 2, 3, 4, 5, 6};
    auto t = Tensor::from_f32(data, {2, 3});
    ASSERT_TRUE(t.has_value());
    auto r = t->reshape({3, 2});
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->shape().dims, (std::vector<size_t>{3, 2}));
    EXPECT_EQ(r->f32_slice()[5], 6.0f);
    EXPECT_EQ(&r->buffer(), &t->buffer()); // shared buffer, no copy
}
// Rust: reshape_wrong_numel
TEST(Tensor, reshape_wrong_numel) {
    auto t = Tensor::zeros({2, 3}, DType::F32);
    ASSERT_TRUE(t.has_value());
    EXPECT_FALSE(t->reshape({5}).has_value());
}
// Rust: transpose_2d
TEST(Tensor, transpose_2d) {
    auto t = Tensor::zeros({3, 4}, DType::F32);
    auto tt = t->t();
    ASSERT_TRUE(tt.has_value());
    EXPECT_EQ(tt->shape().dims, (std::vector<size_t>{4, 3}));
    EXPECT_EQ(tt->strides()[0], 1u);
    EXPECT_EQ(tt->strides()[1], 4u);
    EXPECT_FALSE(tt->is_contiguous());
    auto z = Tensor::zeros({3}, DType::F32);
    ASSERT_TRUE(z.has_value());
    EXPECT_EQ(z->t().error().to_string(), "Internal error: t() requires a 2-D tensor");
}
// Rust: byte_size
TEST(Tensor, byte_size) {
    EXPECT_EQ(Tensor::zeros({4, 4}, DType::F32)->byte_size(), 64u);
}
// Rust: q5_k_dequant_high_bits_per_element
TEST(Tensor, q5_k_dequant_high_bits_per_element) {
    std::vector<uint8_t> block(176, 0);
    f16_to_le(f32_to_f16_bits(1.0f), block.data());     // d = 1
    f16_to_le(f32_to_f16_bits(0.0f), block.data() + 2); // dmin = 0
    // Rust (tensor.rs:797-825) makes every (sc, m) pair unpack to sc = 1, m = 0:
    // scales[0..4] = 1 (sc for j < 4), scales[8..12] = 1 (low nibble → sc for j >= 4), scales[4..8] = 0 (mins).
    for (int j = 0; j < 4; ++j) {
        block[4 + j] = 1;
        block[12 + j] = 1;
    }
    block[16 + 5] = 0b0000'0001; // qh[5] bit 0 → element 5 gets +16
    block[16 + 0] = 0b0000'0100; // qh[0] bit 2 → element 64 (group 1, u1 = 4) gets +16
    block[48 + 3] = 0x02;        // ql[3] low nibble 2 → element 3 = 2
    auto t = Tensor::from_quant_bytes(block, {256}, DType::Q5_K);
    ASSERT_TRUE(t.has_value());
    const auto out = t->to_f32_vec();
    ASSERT_EQ(out.size(), 256u);
    EXPECT_EQ(out[3], 2.0f);
    EXPECT_EQ(out[5], 16.0f);
    EXPECT_EQ(out[64], 16.0f);
    for (size_t i = 0; i < 256; ++i) {
        if (i == 3 || i == 5 || i == 64) continue;
        EXPECT_EQ(out[i], 0.0f) << "element " << i;
    }
}
TEST(Tensor, bytes_bounded_for_quant_unbounded_for_float) {
    auto buf = *CpuBuffer::with_capacity(34 * 2 + 10, 16); // 10 extra bytes after two Q8_0 blocks
    auto q = Tensor::from_buffer({64}, DType::Q8_0, buf, 0);
    ASSERT_TRUE(q.has_value());
    EXPECT_EQ(q->bytes().size(), 68u); // bounded by byte_count(64)
    // The float view is unbounded to the end of ITS OWN buffer, not to shape().numel() — a
    // separate 80-byte (4-aligned) buffer keeps bytes()/f32_slice() beyond numel()==4 without
    // hitting the len%4!=0 panic that f32_slice_panics_on_unaligned_length exercises below.
    auto buf80 = *CpuBuffer::with_capacity(80, 16);
    auto f = Tensor::from_buffer({4}, DType::F32, buf80, 0);
    ASSERT_TRUE(f.has_value());
    EXPECT_EQ(f->bytes().size(), 80u);     // unbounded: to the end of the buffer (Rust parity)
    EXPECT_EQ(f->f32_slice().size(), 20u); // 80 / 4, whole buffer as f32 elements
    EXPECT_EQ(Tensor::from_buffer({64}, DType::Q8_0, buf, 20).error().to_string(),
              "Buffer size mismatch: expected 88 bytes, got 78");
}
TEST(Tensor, f32_slice_panics_on_unaligned_length) {
    auto buf = *CpuBuffer::with_capacity(34 * 2 + 10, 16); // 78 bytes — not a multiple of 4
    auto t = *Tensor::from_buffer({4}, DType::F32, buf, 0);
    EXPECT_DEATH(t.f32_slice(), "sapient panic");
}
TEST(Tensor, bytes_mut_requires_exclusive_ownership) {
    auto t = *Tensor::zeros({4}, DType::F32);
    Tensor alias = t; // shares the buffer
    EXPECT_EQ(t.bytes_mut().error().to_string(),
              "Internal error: Cannot mutate shared tensor buffer");
    alias = *Tensor::zeros({1}, DType::F32);
    auto m = t.bytes_mut();
    ASSERT_TRUE(m.has_value());
    EXPECT_EQ(m->size(), 16u);
    (*m)[0] = 1;
    EXPECT_EQ(t.bytes()[0], 1);
}
TEST(Tensor, slice_axis_and_contiguous_gather) {
    const float data[] = {1, 2, 3, 4, 5, 6};
    auto t = *Tensor::from_f32(data, {2, 3});
    auto s = t.slice_axis(0, 1, 2);
    ASSERT_TRUE(s.has_value());
    EXPECT_EQ(s->shape().dims, (std::vector<size_t>{1, 3}));
    EXPECT_EQ(s->offset(), 12u); // 1 * stride(3) * 4 bytes
    EXPECT_FALSE(s->is_contiguous());
    EXPECT_EQ(s->to_contiguous_f32_vec(), (std::vector<float>{4, 5, 6}));
    auto tt = *t.t();
    EXPECT_EQ(tt.to_contiguous_f32_vec(), (std::vector<float>{1, 4, 2, 5, 3, 6}));
    EXPECT_EQ(t.slice_axis(2, 0, 1).error().to_string(),
              "Internal error: slice axis out of bounds");
}
TEST(Tensor, f16_bf16_and_cow) {
    const uint8_t h[] = {0x00, 0x3C, 0x00, 0xC0}; // 1.0, -2.0 as f16 LE
    auto t = *Tensor::from_f16_bytes(h, {2});
    EXPECT_EQ(t.to_f32_vec(), (std::vector<float>{1.0f, -2.0f}));
    EXPECT_EQ(t.dtype(), DType::F16);
    auto cow = t.to_f32_cow();
    EXPECT_TRUE(cow.is_owned);
    auto f = *Tensor::from_f32(std::vector<float>{3.0f}, {1});
    auto cow2 = f.to_f32_cow();
    EXPECT_FALSE(cow2.is_owned);
    EXPECT_EQ(cow2.get()[0], 3.0f);
    EXPECT_EQ(Tensor::from_f16_bytes(h, {3}).error().to_string(),
              "Shape mismatch: expected [3], got [2]");
    EXPECT_EQ(t.to_string(), "Tensor(shape=[2], dtype=f16, device=cpu)");
}
TEST(Tensor, from_f32_vec_is_zero_copy) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    const float* captured = data.data();
    auto t = Tensor::from_f32_vec(std::move(data), {4});
    ASSERT_TRUE(t.has_value());
    EXPECT_EQ(t->dtype(), DType::F32);
    EXPECT_EQ(t->numel(), 4u);
    EXPECT_EQ(t->f32_slice().data(), captured);
}
