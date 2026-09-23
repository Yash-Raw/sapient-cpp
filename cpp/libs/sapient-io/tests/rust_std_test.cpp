// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
// The Rust-std text twins. Expected strings are verbatim Rust output (probe crate, 2026-09-23).
#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "sapient/io/rust_std.hpp"

namespace rust_std = sapient::io::rust_std;

namespace {
std::optional<std::string> utf8(std::vector<uint8_t> v) {
    return rust_std::utf8_error(v);
}
} // namespace

TEST(RustStd, read_exact_eof_text) {
    EXPECT_EQ(rust_std::READ_EXACT_EOF, "failed to fill whole buffer");
}

TEST(RustStd, utf8_error_matches_rust_display) {
    EXPECT_EQ(utf8({0xFF}), "invalid utf-8 sequence of 1 bytes from index 0");
    EXPECT_EQ(utf8({'a', 0xE2, 0x82}), "incomplete utf-8 byte sequence from index 1");
    EXPECT_EQ(utf8({0xE0, 0x80}), "invalid utf-8 sequence of 1 bytes from index 0");
    EXPECT_EQ(utf8({0xF0, 0x90, 0x41}), "invalid utf-8 sequence of 2 bytes from index 0");
    EXPECT_EQ(utf8({0xF0, 0x90, 0x80, 0x41}), "invalid utf-8 sequence of 3 bytes from index 0");
    EXPECT_EQ(utf8({0xED, 0xA0, 0x80}),
              "invalid utf-8 sequence of 1 bytes from index 0");                     // surrogate
    EXPECT_EQ(utf8({0xC0, 0x80}), "invalid utf-8 sequence of 1 bytes from index 0"); // overlong
    EXPECT_EQ(utf8({'o', 'k', 0xF4, 0x90, 0x80, 0x80}),
              "invalid utf-8 sequence of 1 bytes from index 2"); // > U+10FFFF
    EXPECT_EQ(utf8({0xF0, 0x9F, 0x98}), "incomplete utf-8 byte sequence from index 0");
}

TEST(RustStd, utf8_error_accepts_valid_text) {
    EXPECT_EQ(utf8({}), std::nullopt);
    EXPECT_EQ(utf8({'a', 'b', 'c'}), std::nullopt);
    EXPECT_EQ(utf8({0xC3, 0xA9}), std::nullopt);             // é
    EXPECT_EQ(utf8({0xE2, 0x82, 0xAC}), std::nullopt);       // €
    EXPECT_EQ(utf8({0xF0, 0x90, 0x8D, 0x88}), std::nullopt); // 𐍈
    EXPECT_EQ(utf8({0xEF, 0xBF, 0xBF}), std::nullopt);       // U+FFFF
    EXPECT_EQ(utf8({0xF4, 0x8F, 0xBF, 0xBF}), std::nullopt); // U+10FFFF
}

TEST(RustStd, os_error_message_format) {
    // Portable: the format is "<system_category text> (os error <code>)" on every OS.
    const std::string m = rust_std::os_error_message(2);
    EXPECT_TRUE(m.ends_with(" (os error 2)")) << m;
#if !defined(_WIN32)
    EXPECT_EQ(rust_std::os_error_message(2), "No such file or directory (os error 2)");
    EXPECT_EQ(rust_std::os_error_message(21), "Is a directory (os error 21)");
    EXPECT_EQ(rust_std::os_error_message(22), "Invalid argument (os error 22)");
#endif
}
