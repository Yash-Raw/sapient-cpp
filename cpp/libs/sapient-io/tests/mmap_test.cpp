// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
// MappedFile / read_file / display_path — the memmap2 + std::fs::read twins.
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "io_test_util.hpp"

#include "sapient/io/mmap.hpp"
#include "sapient/io/rust_std.hpp"

using sapient::io::MappedFile;
using sapient::io::MapStage;
using sapient::io::test::TempDir;

namespace {
std::vector<uint8_t> pattern(size_t n) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i)
        v[i] = static_cast<uint8_t>((i * 7 + 3) & 0xFF);
    return v;
}
} // namespace

TEST(Mmap, maps_whole_file_read_only) {
    TempDir dir("maps_whole_file");
    const auto data = pattern(1000);
    const auto p = dir.write("f.bin", data);
    auto m = MappedFile::open(p);
    ASSERT_TRUE(m.has_value()) << m.error().os.message;
    ASSERT_EQ((*m)->size(), data.size());
    EXPECT_TRUE(std::equal(data.begin(), data.end(), (*m)->bytes().begin()));
}

TEST(Mmap, empty_file_maps_to_empty_span) {
    // memmap2 0.9.11 maps max(len,1) bytes and reports len: an empty file is NOT an error here —
    // it fails later, in the parser, with the parser's own text (Review Focus 1).
    TempDir dir("empty_file");
    const auto p = dir.write("empty.bin", {});
    auto m = MappedFile::open(p);
    ASSERT_TRUE(m.has_value()) << m.error().os.message;
    EXPECT_EQ((*m)->size(), 0u);
    EXPECT_TRUE((*m)->bytes().empty());
}

TEST(Mmap, missing_file_fails_at_open_stage) {
    TempDir dir("missing_file");
    auto m = MappedFile::open(dir.path() / "missing.gguf");
    ASSERT_FALSE(m.has_value());
    EXPECT_EQ(m.error().stage, MapStage::Open);
    EXPECT_EQ(m.error().os.code, 2); // ENOENT == ERROR_FILE_NOT_FOUND == 2
    EXPECT_EQ(m.error().os.message, sapient::io::rust_std::os_error_message(2));
}

TEST(Mmap, directory_fails_at_map_stage) {
    TempDir dir("directory");
    auto m = MappedFile::open(dir.path());
    ASSERT_FALSE(m.has_value());
#if defined(_WIN32)
    EXPECT_EQ(m.error().stage, MapStage::Open); // CreateFileW refuses a directory handle
#else
    // POSIX open(2) accepts a directory; mmap(2) then fails (macOS EINVAL, Linux ENODEV).
    EXPECT_EQ(m.error().stage, MapStage::Map);
#endif
    EXPECT_EQ(m.error().os.message, sapient::io::rust_std::os_error_message(m.error().os.code));
}

TEST(Mmap, mapping_outlives_every_other_handle) {
    TempDir dir("outlives");
    const auto data = pattern(4096 + 17);
    std::shared_ptr<const MappedFile> keep;
    {
        auto m = MappedFile::open(dir.write("f.bin", data));
        ASSERT_TRUE(m.has_value());
        keep = *m;
    }
    EXPECT_TRUE(std::equal(data.begin(), data.end(), keep->bytes().begin()));
    keep.reset(); // unmapped before ~TempDir (Windows cannot delete a mapped file)
}

TEST(ReadFile, reads_whole_file) {
    TempDir dir("read_whole");
    const auto data = pattern(200000); // larger than one 64 KiB read chunk
    auto r = sapient::io::read_file(dir.write("f.bin", data));
    ASSERT_TRUE(r.has_value()) << r.error().message;
    EXPECT_EQ(*r, data);
}

TEST(ReadFile, empty_file_reads_empty) {
    TempDir dir("read_empty");
    auto r = sapient::io::read_file(dir.write("e.bin", {}));
    ASSERT_TRUE(r.has_value()) << r.error().message;
    EXPECT_TRUE(r->empty());
}

TEST(ReadFile, missing_file_is_code_2) {
    TempDir dir("read_missing");
    auto r = sapient::io::read_file(dir.path() / "nope.bin");
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, 2);
    EXPECT_EQ(r.error().message, sapient::io::rust_std::os_error_message(2));
}

#if !defined(_WIN32)
TEST(ReadFile, directory_fails_with_eisdir) {
    // Rust std::fs::read: open(2) succeeds on a directory, read(2) fails with EISDIR (21).
    TempDir dir("read_dir");
    auto r = sapient::io::read_file(dir.path());
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, 21);
    EXPECT_EQ(r.error().message, "Is a directory (os error 21)");
}
#endif

TEST(DisplayPath, is_utf8) {
    const std::filesystem::path p =
        std::filesystem::path(u8"models") / std::filesystem::path(u8"héllo.gguf");
    const std::string d = sapient::io::display_path(p);
    EXPECT_TRUE(d.ends_with("h\xC3\xA9llo.gguf")) << d;
}
