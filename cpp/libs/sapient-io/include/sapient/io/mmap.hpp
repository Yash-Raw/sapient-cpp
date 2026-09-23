// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// The two file facilities sapient-io uses: `memmap2::Mmap::map(&File)` (a whole-file read-only
// mapping; memmap2 0.9.11 semantics) and `std::fs::read`. Failures carry Rust's io::Error Display
// text; MappedFile also reports WHICH step failed, because the Rust callers wrap an open failure
// and a map failure differently (plan B Global Constraints table).

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <tl/expected.hpp>

namespace sapient::io {

/// A failed OS call. `code` is errno (POSIX) or GetLastError() (Windows); `message` is exactly
/// what Rust's `io::Error` Display prints for it (`rust_std::os_error_message(code)`).
struct OsError {
    int code{0};
    std::string message;
};

/// `File::open` (Open) vs `Mmap::map` (Map — includes the fstat for the length).
enum class MapStage : uint8_t { Open, Map };
struct MapError {
    MapStage stage{MapStage::Open};
    OsError os;
};

/// A whole-file, read-only mapping, shared by every MmapBuffer that points into it (the twin of
/// Rust's `Arc<Mmap>`). An empty file is a valid, empty mapping (memmap2 maps max(len, 1) bytes
/// on POSIX and reports len).
class MappedFile {
    struct Private {};

public:
    static tl::expected<std::shared_ptr<const MappedFile>, MapError>
    open(const std::filesystem::path& path);

    std::span<const uint8_t> bytes() const { return {static_cast<const uint8_t*>(ptr_), len_}; }
    size_t size() const { return len_; }

    explicit MappedFile(Private) {}
    ~MappedFile();
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

private:
    const void* ptr_{nullptr};
    size_t len_{0};     // Rust `Mmap::len()`
    size_t map_len_{0}; // bytes actually mapped (POSIX: max(len_, 1)); 0 = nothing to unmap
};

/// `std::fs::read(path)`: open + read to EOF. Every failure (open or read) is one OsError.
tl::expected<std::vector<uint8_t>, OsError> read_file(const std::filesystem::path& path);

/// `Path::display()` for the UTF-8 paths SAPIENT uses. Uses `u8string()` because
/// `path::string()` can throw on Windows.
std::string display_path(const std::filesystem::path& path);

} // namespace sapient::io
