// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once

#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace sapient::core {

/// The C++ twin of a Rust `panic!` under `[profile.release] panic = "abort"`: print and abort.
/// Used only where the Rust code panics (e.g. `DType::block_bytes()` on a float dtype,
/// mutating an mmap buffer). Recoverable failures use `Result<T>` instead.
[[noreturn]] inline void panic(std::string_view msg) {
    std::fprintf(stderr, "sapient panic: %.*s\n", static_cast<int>(msg.size()), msg.data());
    std::fflush(stderr);
    std::abort();
}

}  // namespace sapient::core
