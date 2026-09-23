// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/io/rust_std.hpp"

#include <system_error>

namespace sapient::io::rust_std {

namespace {
// Rust: `next!() as i8 >= -64` rejects — i.e. only 0x80..=0xBF is a continuation byte.
bool is_cont(uint8_t b) {
    return (b & 0xC0u) == 0x80u;
}
// core::str::utf8_char_width.
int char_width(uint8_t first) {
    if (first < 0x80u) return 1;
    if (first >= 0xC2u && first <= 0xDFu) return 2;
    if (first >= 0xE0u && first <= 0xEFu) return 3;
    if (first >= 0xF0u && first <= 0xF4u) return 4;
    return 0; // 0x80..=0xC1 and 0xF5..=0xFF are never valid lead bytes
}
std::string invalid(size_t n, size_t from) {
    return "invalid utf-8 sequence of " + std::to_string(n) + " bytes from index " +
           std::to_string(from);
}
std::string incomplete(size_t from) {
    return "incomplete utf-8 byte sequence from index " + std::to_string(from);
}
} // namespace

std::optional<std::string> utf8_error(std::span<const uint8_t> v) {
    const size_t len = v.size();
    size_t i = 0;
    while (i < len) {
        const size_t start = i; // Rust `old_offset` = Utf8Error::valid_up_to
        const uint8_t first = v[i];
        const int w = char_width(first);
        if (w == 1) {
            ++i;
            continue;
        }
        if (w == 0) return invalid(1, start);
        // Rust `next!()`: advance; running off the end is "incomplete" (error_len None).
        if (++i >= len) return incomplete(start);
        const uint8_t second = v[i];
        if (w == 2) {
            if (!is_cont(second)) return invalid(1, start);
        } else if (w == 3) {
            const bool ok = (first == 0xE0u && second >= 0xA0u && second <= 0xBFu) ||
                            (first >= 0xE1u && first <= 0xECu && is_cont(second)) ||
                            (first == 0xEDu && second >= 0x80u && second <= 0x9Fu) ||
                            (first >= 0xEEu && first <= 0xEFu && is_cont(second));
            if (!ok) return invalid(1, start);
            if (++i >= len) return incomplete(start);
            if (!is_cont(v[i])) return invalid(2, start);
        } else { // w == 4
            const bool ok = (first == 0xF0u && second >= 0x90u && second <= 0xBFu) ||
                            (first >= 0xF1u && first <= 0xF3u && is_cont(second)) ||
                            (first == 0xF4u && second >= 0x80u && second <= 0x8Fu);
            if (!ok) return invalid(1, start);
            if (++i >= len) return incomplete(start);
            if (!is_cont(v[i])) return invalid(2, start);
            if (++i >= len) return incomplete(start);
            if (!is_cont(v[i])) return invalid(3, start);
        }
        ++i;
    }
    return std::nullopt;
}

std::string os_error_message(int code) {
    return std::system_category().message(code) + " (os error " + std::to_string(code) + ")";
}

} // namespace sapient::io::rust_std
