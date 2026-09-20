// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "sapient/testing/golden.hpp"

namespace fs = std::filesystem;
using sapient::testing::GoldenDType;
using sapient::testing::list_golden;
using sapient::testing::read_golden;

namespace {

fs::path fixtures() { return fs::path(SAPIENT_TESTING_FIXTURES_DIR); }

fs::path temp_file(const char* stem) {
    return fs::temp_directory_path() / (std::string("sapient_golden_") + stem + ".sapd");
}

void write_bytes(const fs::path& p, const std::vector<uint8_t>& bytes) {
    std::ofstream out(p, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

std::vector<uint8_t> read_bytes(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

}  // namespace

TEST(Golden, ReadsFormatSampleWrittenByRust) {
    std::string err;
    auto c = read_golden(fixtures() / "format_sample.sapd", &err);
    ASSERT_TRUE(c.has_value()) << err;
    EXPECT_EQ(c->name, "format_sample");
    ASSERT_EQ(c->arrays.size(), 7u);

    const auto& f = c->get("in:f32");
    EXPECT_EQ(f.dtype, GoldenDType::F32);
    EXPECT_EQ(f.dims, (std::vector<uint64_t>{5}));
    EXPECT_EQ(f.numel(), 5u);
    EXPECT_EQ(f.as<float>(), (std::vector<float>{0.0f, 1.0f, -1.0f, 0.5f, 3.25f}));

    EXPECT_EQ(c->get("in:u8").dims, (std::vector<uint64_t>{2, 2}));
    EXPECT_EQ(c->get("in:u8").as<uint8_t>(), (std::vector<uint8_t>{0, 1, 2, 255}));
    EXPECT_EQ(c->get("in:i8").as<int8_t>(), (std::vector<int8_t>{-128, 127}));
    EXPECT_EQ(c->get("param:i32").as<int32_t>(), (std::vector<int32_t>{-42}));
    EXPECT_EQ(c->get("param:u32").as<uint32_t>(), (std::vector<uint32_t>{4000000000u}));
    EXPECT_EQ(c->get("param:u64").as<uint64_t>(), (std::vector<uint64_t>{uint64_t{1} << 40}));
    EXPECT_EQ(c->get("out:empty").numel(), 0u);
    EXPECT_EQ(c->find("does-not-exist"), nullptr);
    EXPECT_THROW(c->get("does-not-exist"), std::out_of_range);
    EXPECT_THROW(c->get("in:f32").as<uint8_t>(), std::logic_error);  // element size mismatch
}

TEST(Golden, RejectsBadMagic) {
    const auto p = temp_file("badmagic");
    write_bytes(p, {'X', 'X', 'X', 'X', 1, 0, 0, 0});
    std::string err;
    EXPECT_FALSE(read_golden(p, &err).has_value());
    EXPECT_NE(err.find("magic"), std::string::npos) << err;
    fs::remove(p);
}

TEST(Golden, RejectsTruncatedFile) {
    auto bytes = read_bytes(fixtures() / "format_sample.sapd");
    ASSERT_GT(bytes.size(), 40u);
    bytes.resize(bytes.size() / 2);
    const auto p = temp_file("truncated");
    write_bytes(p, bytes);
    std::string err;
    EXPECT_FALSE(read_golden(p, &err).has_value());
    EXPECT_NE(err.find("truncated"), std::string::npos) << err;
    fs::remove(p);
}

TEST(Golden, RejectsMissingFile) {
    std::string err;
    EXPECT_FALSE(read_golden(temp_file("nope-missing"), &err).has_value());
    EXPECT_FALSE(err.empty());
}

// Real gate when SAPIENT_GOLDEN_DIR points at dumps made on this host by
// cpp/tests/parity/golden_dump.sh; an explicit, visible skip otherwise (never a silent pass).
TEST(Golden, InventoryFromEnv) {
    const char* dir = std::getenv("SAPIENT_GOLDEN_DIR");
    if (dir == nullptr) {
        GTEST_SKIP() << "SAPIENT_GOLDEN_DIR not set — run cpp/tests/parity/golden_dump.sh <dir> and export it";
    }
    const auto files = list_golden(dir);
    ASSERT_FALSE(files.empty()) << "no *.sapd files in " << dir;
    for (const auto& f : files) {
        std::string err;
        auto c = read_golden(f, &err);
        ASSERT_TRUE(c.has_value()) << f << ": " << err;
        EXPECT_EQ(c->name, f.stem().string()) << f;
        bool has_out = false;
        for (const auto& a : c->arrays) has_out = has_out || a.name.starts_with("out:");
        EXPECT_TRUE(has_out) << f << " has no out: array";
    }
    std::printf("golden inventory: %zu cases in %s\n", files.size(), dir);
}
