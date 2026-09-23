// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/io/mmap.hpp"

#include <new>
#include <optional>
#include <stdexcept>
#include <utility>

#include "sapient/io/rust_std.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace sapient::io {

namespace {

OsError last_os_error() {
#if defined(_WIN32)
    const int code = static_cast<int>(::GetLastError());
#else
    const int code = errno;
#endif
    return OsError{code, rust_std::os_error_message(code)};
}

tl::unexpected<MapError> map_failure(MapStage stage, OsError os) {
    return tl::unexpected(MapError{stage, std::move(os)});
}

/// Rust's `File::open`/`fs::read` convert the path to the OS call's native string type before
/// issuing any syscall — on POSIX via `CString::new`, on Windows via a UTF-16 conversion — and
/// both reject an embedded NUL byte up front with a fixed `io::ErrorKind::InvalidInput` error
/// (not a raw OS errno, so its Display has no "(os error N)" suffix). `path.c_str()` has no such
/// check: for a `path` whose internal representation contains a NUL (constructible in C++ from a
/// `std::string`/`std::wstring` with an embedded NUL, unlike a real OS path), it silently returns
/// a pointer that a C API reads only up to the first NUL — opening a different, shorter path
/// instead of failing. Check for the NUL ourselves before any syscall so this never happens.
std::optional<OsError> nul_in_path_error(const std::filesystem::path& path) {
    const auto& native = path.native();
#if defined(_WIN32)
    // Rust std's Windows text for this case (io/error/repr_bitpacked.rs's INVALID_INPUT path via
    // sys::args::to_u16s); not parity-bound (Windows OS texts are exempt).
    if (native.find(L'\0') != std::filesystem::path::string_type::npos)
        return OsError{0, "strings passed to WinAPI cannot contain NULs"};
#else
    if (native.find('\0') != std::filesystem::path::string_type::npos)
        return OsError{0, "file name contained an unexpected NUL byte"};
#endif
    return std::nullopt;
}

#if !defined(_WIN32)
// Rust's File::open retries open(2) on EINTR (cvt_r). O_CLOEXEC as Rust sets it.
int open_readonly(const std::filesystem::path& path) {
    int fd = -1;
    do {
        fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    } while (fd < 0 && errno == EINTR);
    return fd;
}
#endif

} // namespace

std::string display_path(const std::filesystem::path& path) {
    const std::u8string s = path.u8string();
    return {s.begin(), s.end()};
}

#if defined(_WIN32)

tl::expected<std::shared_ptr<const MappedFile>, MapError>
MappedFile::open(const std::filesystem::path& path) {
    if (const auto err = nul_in_path_error(path)) return map_failure(MapStage::Open, *err);
    // Rust File::open on Windows: GENERIC_READ, share read|write|delete, OPEN_EXISTING.
    HANDLE file = ::CreateFileW(path.c_str(),
                                GENERIC_READ,
                                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                nullptr,
                                OPEN_EXISTING,
                                FILE_ATTRIBUTE_NORMAL,
                                nullptr);
    if (file == INVALID_HANDLE_VALUE) return map_failure(MapStage::Open, last_os_error());
    LARGE_INTEGER size{};
    if (::GetFileSizeEx(file, &size) == 0) {
        OsError err = last_os_error();
        ::CloseHandle(file);
        return map_failure(MapStage::Map, std::move(err));
    }
    auto mf = std::make_shared<MappedFile>(Private{});
    mf->len_ = static_cast<size_t>(size.QuadPart);
    if (mf->len_ == 0) { // CreateFileMappingW rejects empty files; memmap2 returns an empty map
        ::CloseHandle(file);
        return mf;
    }
    HANDLE mapping = ::CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping == nullptr) {
        OsError err = last_os_error();
        ::CloseHandle(file);
        return map_failure(MapStage::Map, std::move(err));
    }
    void* view = ::MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    std::optional<OsError> err;
    if (view == nullptr) err = last_os_error(); // capture before CloseHandle clobbers it
    ::CloseHandle(mapping);                     // the view keeps the section alive
    ::CloseHandle(file);
    if (err) return map_failure(MapStage::Map, std::move(*err));
    mf->ptr_ = view;
    mf->map_len_ = mf->len_;
    return mf;
}

MappedFile::~MappedFile() {
    if (ptr_ != nullptr) ::UnmapViewOfFile(ptr_);
}

tl::expected<std::vector<uint8_t>, OsError> read_file(const std::filesystem::path& path) {
    if (const auto err = nul_in_path_error(path)) return tl::unexpected(*err);
    HANDLE file = ::CreateFileW(path.c_str(),
                                GENERIC_READ,
                                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                nullptr,
                                OPEN_EXISTING,
                                FILE_ATTRIBUTE_NORMAL,
                                nullptr);
    if (file == INVALID_HANDLE_VALUE) return tl::unexpected(last_os_error());
    std::vector<uint8_t> out;
    std::vector<uint8_t> chunk(size_t{1} << 16);
    for (;;) {
        DWORD got = 0;
        if (::ReadFile(file, chunk.data(), static_cast<DWORD>(chunk.size()), &got, nullptr) == 0) {
            OsError err = last_os_error();
            ::CloseHandle(file);
            return tl::unexpected(std::move(err));
        }
        if (got == 0) break;
        out.insert(out.end(), chunk.begin(), chunk.begin() + got);
    }
    ::CloseHandle(file);
    return out;
}

#else // POSIX

tl::expected<std::shared_ptr<const MappedFile>, MapError>
MappedFile::open(const std::filesystem::path& path) {
    if (const auto err = nul_in_path_error(path)) return map_failure(MapStage::Open, *err);
    const int fd = open_readonly(path);
    if (fd < 0) return map_failure(MapStage::Open, last_os_error());
    struct stat st {};
    if (::fstat(fd, &st) != 0) { // memmap2 `file_len` — part of Mmap::map
        OsError err = last_os_error();
        ::close(fd);
        return map_failure(MapStage::Map, std::move(err));
    }
    const auto len = static_cast<size_t>(st.st_size);
    // memmap2 0.9.11 adjust_mmap_params: POSIX rejects a zero-length mmap, so map one byte and
    // keep reporting len 0 (the byte is never exposed). A directory reaches here and fails.
    const size_t map_len = len == 0 ? size_t{1} : len;
    void* p = ::mmap(nullptr, map_len, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
        OsError err = last_os_error();
        ::close(fd);
        return map_failure(MapStage::Map, std::move(err));
    }
    ::close(fd); // the mapping stays valid; Rust drops the File at the end of the loader too
    auto mf = std::make_shared<MappedFile>(Private{});
    mf->ptr_ = p;
    mf->len_ = len;
    mf->map_len_ = map_len;
    return mf;
}

MappedFile::~MappedFile() {
    if (map_len_ != 0) ::munmap(const_cast<void*>(ptr_), map_len_);
}

tl::expected<std::vector<uint8_t>, OsError> read_file(const std::filesystem::path& path) {
    if (const auto err = nul_in_path_error(path)) return tl::unexpected(*err);
    const int fd = open_readonly(path);
    if (fd < 0) return tl::unexpected(last_os_error());
    std::vector<uint8_t> out;
    struct stat st {};
    if (::fstat(fd, &st) == 0 && st.st_size > 0) {
        // Rust fs::read pre-sizes from metadata. A real file's size is not an untrusted header
        // field, but a reservation that cannot be satisfied must still not throw across the
        // library boundary: report it as ENOMEM (not parity-bound; Rust reports its own OOM).
        try {
            out.reserve(static_cast<size_t>(st.st_size));
        } catch (const std::bad_alloc&) {
            ::close(fd);
            return tl::unexpected(OsError{ENOMEM, rust_std::os_error_message(ENOMEM)});
        } catch (const std::length_error&) {
            ::close(fd);
            return tl::unexpected(OsError{ENOMEM, rust_std::os_error_message(ENOMEM)});
        }
    }
    std::vector<uint8_t> chunk(size_t{1} << 16);
    for (;;) {
        const ssize_t n = ::read(fd, chunk.data(), chunk.size());
        if (n < 0) {
            if (errno == EINTR) continue; // Rust read_to_end retries Interrupted
            OsError err = last_os_error();
            ::close(fd);
            return tl::unexpected(std::move(err));
        }
        if (n == 0) break;
        out.insert(out.end(), chunk.begin(), chunk.begin() + n);
    }
    ::close(fd);
    return out;
}

#endif

} // namespace sapient::io
