// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Test helpers for sapient_io_tests. std::filesystem calls use the std::error_code overloads only
// (a throwing overload is one missing directory away from std::terminate — plan E lesson).
// Windows cannot delete a file that is still mapped: declare the TempDir BEFORE any Tensor or
// MappedFile that points into it, so it is destroyed last.

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <system_error>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace sapient::io::test {

inline int current_pid() {
#if defined(_WIN32)
    return _getpid();
#else
    return static_cast<int>(::getpid());
#endif
}

/// A fresh directory unique to this process + name + instance (ctest runs each gtest in its own
/// process, possibly in parallel). Removed, best-effort, on destruction.
class TempDir {
public:
    explicit TempDir(std::string_view name) {
        static std::atomic<unsigned> counter{0};
        std::error_code ec;
        dir_ = std::filesystem::temp_directory_path(ec) /
               ("sapient-io-test-" + std::to_string(current_pid()) + "-" + std::string(name) + "-" +
                std::to_string(counter.fetch_add(1)));
        std::filesystem::remove_all(dir_, ec);
        ec.clear();
        std::filesystem::create_directories(dir_, ec);
        EXPECT_FALSE(ec) << "create_directories " << dir_.string() << ": " << ec.message();
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    const std::filesystem::path& path() const { return dir_; }

    std::filesystem::path write(std::string_view file, std::span<const uint8_t> bytes) const {
        const std::filesystem::path p = dir_ / std::string(file);
        std::ofstream f(p, std::ios::binary | std::ios::trunc);
        EXPECT_TRUE(f.is_open()) << "open " << p.string();
        f.write(reinterpret_cast<const char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
        EXPECT_TRUE(f.good()) << "write " << p.string();
        return p;
    }

private:
    std::filesystem::path dir_;
};

} // namespace sapient::io::test
