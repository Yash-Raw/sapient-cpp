// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Twins of the Rust standard-library error texts that sapient-io embeds in its Result-path
// errors (and that are therefore parity-bound): `read_exact`'s EOF error, `Utf8Error`'s Display,
// and `io::Error`'s Display for an OS error.

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace sapient::io::rust_std {

/// `std::io::Read::read_exact` on too few bytes (`io::Error::READ_EXACT_EOF`).
inline constexpr std::string_view READ_EXACT_EOF = "failed to fill whole buffer";

/// `std::str::from_utf8(bytes)`: nullopt when valid, else `Utf8Error`'s Display text —
/// "invalid utf-8 sequence of {n} bytes from index {i}" / "incomplete utf-8 byte sequence from
/// index {i}". A transcription of core::str::validations::run_utf8_validation.
std::optional<std::string> utf8_error(std::span<const uint8_t> bytes);

/// `io::Error::from_raw_os_error(code).to_string()`: "{description} (os error {code})". The
/// description is `std::system_category().message(code)` — strerror text on POSIX (identical to
/// Rust's strerror_r), FormatMessage text on Windows (not parity-bound).
std::string os_error_message(int code);

} // namespace sapient::io::rust_std
