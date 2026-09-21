// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Software IEEE binary16 / bfloat16 conversions — the C++ twin of the `half` crate as SAPIENT
// uses it: exact widening, round-to-nearest-even narrowing (subnormals included). Software only:
// no F16C / NEON vcvt, so every host and every ISA path produces the same bits (spec §3 rule 5).

#include <bit>
#include <cstdint>
#include <cstring>

namespace sapient::core {

float f16_bits_to_f32(uint16_t h);
uint16_t f32_to_f16_bits(float f);

inline float bf16_bits_to_f32(uint16_t b) {
    return std::bit_cast<float>(static_cast<uint32_t>(b) << 16);
}
uint16_t f32_to_bf16_bits(float f);

inline uint16_t load_le16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0] | (static_cast<uint16_t>(p[1]) << 8));
}
inline float f16_le_to_f32(const uint8_t* p) {
    return f16_bits_to_f32(load_le16(p));
}
inline float bf16_le_to_f32(const uint8_t* p) {
    return bf16_bits_to_f32(load_le16(p));
}
inline void f16_to_le(uint16_t bits, uint8_t* p) {
    p[0] = static_cast<uint8_t>(bits & 0xFF);
    p[1] = static_cast<uint8_t>(bits >> 8);
}

} // namespace sapient::core
