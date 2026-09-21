// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/core/f16.hpp"

static_assert(std::endian::native == std::endian::little, "SAPIENT assumes a little-endian host");

namespace sapient::core {

float f16_bits_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign; // ±0
        } else {         // subnormal: value = mant · 2^-24 → normalise
            uint32_t m = mant;
            uint32_t s = 0;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                ++s;
            }
            bits = sign | ((113u - s) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13); // inf, or NaN with the payload shifted up
        if (mant != 0) bits |= 0x00400000u; // quiet a signalling NaN on widening, matching `half`
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    return std::bit_cast<float>(bits);
}

uint16_t f32_to_f16_bits(float f) {
    const uint32_t x = std::bit_cast<uint32_t>(f);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t exp = (x >> 23) & 0xFFu;
    const uint32_t mant = x & 0x7FFFFFu;
    if (exp == 0xFF) { // inf or NaN
        if (mant == 0) return static_cast<uint16_t>(sign | 0x7C00u);
        return static_cast<uint16_t>(sign | 0x7E00u |
                                     ((mant >> 13) & 0x1FFu)); // quiet NaN, high payload kept
    }
    const int32_t e = static_cast<int32_t>(exp) - 127 + 15;    // rebias to binary16
    if (e >= 31) return static_cast<uint16_t>(sign | 0x7C00u); // overflow → inf
    if (e >=
        1) { // normal result; round-to-nearest-even on the 13 dropped bits; carry may overflow into inf
        uint32_t out = sign | (static_cast<uint32_t>(e) << 10) | (mant >> 13);
        const uint32_t round = mant & 0x1FFFu;
        if (round > 0x1000u || (round == 0x1000u && (out & 1u))) out += 1;
        return static_cast<uint16_t>(out);
    }
    if (e < -10) return static_cast<uint16_t>(sign); // below half the smallest subnormal → ±0
    // subnormal result: shift the 24-bit significand (implicit 1) right by 14 - e, round to nearest even
    const uint32_t m = mant | 0x800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - e);
    uint32_t out = m >> shift;
    const uint32_t rem = m & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1);
    if (rem > halfway || (rem == halfway && (out & 1u)))
        out += 1; // may carry into the smallest normal
    return static_cast<uint16_t>(sign | out);
}

uint16_t f32_to_bf16_bits(float f) {
    uint32_t x = std::bit_cast<uint32_t>(f);
    if ((x & 0x7F800000u) == 0x7F800000u && (x & 0x7FFFFFu) != 0)
        return static_cast<uint16_t>((x >> 16) | 0x40u); // quiet NaN
    x += 0x7FFFu + ((x >> 16) & 1u);                     // round to nearest even
    return static_cast<uint16_t>(x >> 16);
}

} // namespace sapient::core
