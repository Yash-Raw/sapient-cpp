// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#include "sapient/testing/compare.hpp"
#include "sapient/testing/golden.hpp"

namespace fs = std::filesystem;
using sapient::testing::GoldenDType;
using sapient::testing::list_golden;
using sapient::testing::read_golden;

namespace {

fs::path fixtures() {
    return fs::path(SAPIENT_TESTING_FIXTURES_DIR);
}

// Suffixed with the process id so concurrent test binaries (e.g. two ctest jobs, or a stray
// re-run) never collide on the same filename in the shared temp directory.
int current_pid() {
#ifdef _WIN32
    return _getpid();
#else
    return static_cast<int>(getpid());
#endif
}

fs::path temp_file(const char* stem) {
    return fs::temp_directory_path() /
           (std::string("sapient_golden_") + stem + "_" + std::to_string(current_pid()) + ".sapd");
}

void write_bytes(const fs::path& p, const std::vector<uint8_t>& bytes) {
    std::ofstream out(p, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
}

std::vector<uint8_t> read_bytes(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// Tiny little-endian appenders for hand-building a .sapd byte buffer — no new dependencies.
void put_u32(std::vector<uint8_t>& buf, uint32_t v) {
    for (int i = 0; i < 4; ++i)
        buf.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}

void put_u64(std::vector<uint8_t>& buf, uint64_t v) {
    for (int i = 0; i < 8; ++i)
        buf.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}

void put_str(std::vector<uint8_t>& buf, const std::string& s) {
    put_u32(buf, static_cast<uint32_t>(s.size()));
    buf.insert(buf.end(), s.begin(), s.end());
}

} // namespace

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
    EXPECT_THROW(c->get("in:f32").as<uint8_t>(), std::logic_error); // element size mismatch
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

// A corrupted/adversarial .sapd must fail cleanly (nullopt + reason), never wrap into UB or an
// uncaught exception, when its length/count fields are implausible.
TEST(Golden, RejectsImplausibleLengthFields) {
    // Variant 1: byte_len is near SIZE_MAX with no payload bytes following it. Cursor::has()
    // must reject this via subtraction, not `pos + len` (which would wrap and read out of bounds).
    {
        std::vector<uint8_t> buf;
        buf.insert(buf.end(), {'S', 'A', 'P', 'D'});
        put_u32(buf, 1);                     // version
        put_str(buf, "x");                   // case name
        put_u32(buf, 1);                     // n_arrays
        put_str(buf, "in:a");                // array name
        buf.push_back(0);                    // dtype = F32
        put_u32(buf, 1);                     // ndim
        put_u64(buf, 1);                     // dims[0]
        put_u64(buf, 0xFFFFFFFFFFFFFFF0ULL); // byte_len — implausible, no payload follows
        const auto p = temp_file("hugelen");
        write_bytes(p, buf);
        std::string err;
        EXPECT_FALSE(read_golden(p, &err).has_value());
        EXPECT_FALSE(err.empty()) << err;
        fs::remove(p);
    }
    // Variant 2: ndim is near UINT32_MAX. Must be rejected before it drives an unbounded
    // std::vector::resize (std::bad_alloc/std::length_error instead of a clean nullopt).
    {
        std::vector<uint8_t> buf;
        buf.insert(buf.end(), {'S', 'A', 'P', 'D'});
        put_u32(buf, 1);           // version
        put_str(buf, "x");         // case name
        put_u32(buf, 1);           // n_arrays
        put_str(buf, "in:a");      // array name
        buf.push_back(0);          // dtype = F32
        put_u32(buf, 0xFFFFFFFFu); // ndim — implausible
        const auto p = temp_file("hugendim");
        write_bytes(p, buf);
        std::string err;
        EXPECT_FALSE(read_golden(p, &err).has_value());
        EXPECT_FALSE(err.empty()) << err;
        fs::remove(p);
    }
    // Variant 3: a single huge dim (2^62+1) passes checked_numel() (it fits in one size_t) but
    // must not be allowed to wrap once multiplied by dtype_size against a small, matching-looking
    // byte_len/payload — the numel×element-size guard must catch this before the byte-length
    // comparison would spuriously accept it.
    {
        std::vector<uint8_t> buf;
        buf.insert(buf.end(), {'S', 'A', 'P', 'D'});
        put_u32(buf, 1);                     // version
        put_str(buf, "x");                   // case name
        put_u32(buf, 1);                     // n_arrays
        put_str(buf, "in:a");                // array name
        buf.push_back(0);                    // dtype = F32
        put_u32(buf, 1);                     // ndim
        put_u64(buf, 0x4000000000000001ULL); // dims[0] = 2^62 + 1
        put_u64(buf, 4);                     // byte_len — small, matches a 4-byte payload
        buf.insert(buf.end(), {0, 0, 0, 0}); // 4 payload bytes
        const auto p = temp_file("hugenumelbytesize");
        write_bytes(p, buf);
        std::string err;
        EXPECT_FALSE(read_golden(p, &err).has_value());
        EXPECT_FALSE(err.empty()) << err;
        fs::remove(p);
    }
}

// Real gate when SAPIENT_GOLDEN_DIR points at dumps made on this host by
// cpp/tests/parity/golden_dump.sh; an explicit, visible skip otherwise (never a silent pass).
TEST(Golden, InventoryFromEnv) {
    const char* dir = std::getenv("SAPIENT_GOLDEN_DIR");
    if (dir == nullptr) {
        GTEST_SKIP() << "SAPIENT_GOLDEN_DIR not set — run cpp/tests/parity/golden_dump.sh <dir> "
                        "and export it";
    }
    const auto files = list_golden(dir);
    ASSERT_FALSE(files.empty()) << "no *.sapd files in " << dir;
    for (const auto& f : files) {
        std::string err;
        auto c = read_golden(f, &err);
        ASSERT_TRUE(c.has_value()) << f << ": " << err;
        EXPECT_EQ(c->name, f.stem().string()) << f;
        bool has_out = false;
        for (const auto& a : c->arrays)
            has_out = has_out || a.name.starts_with("out:");
        EXPECT_TRUE(has_out) << f << " has no out: array";
    }
    std::printf("golden inventory: %zu cases in %s\n", files.size(), dir);
}

TEST(Compare, bit_identical_reports_first_mismatch_with_bits) {
    const float a[] = {1.0f, 2.0f, 0.0f};
    const float b[] = {1.0f, 2.0f, -0.0f}; // -0 differs bitwise
    EXPECT_TRUE(sapient::testing::bit_identical(a, a));
    const auto r = sapient::testing::bit_identical(a, b);
    EXPECT_FALSE(r);
    EXPECT_NE(std::string(r.message()).find("[2]"), std::string::npos) << r.message();
    EXPECT_NE(std::string(r.message()).find("0x80000000"), std::string::npos) << r.message();
    const float c[] = {1.0f};
    EXPECT_FALSE(sapient::testing::bit_identical(a, c)); // length mismatch
}

TEST(Compare, within_abs_and_max_abs_err) {
    const float a[] = {1.0f, 2.0f};
    const float b[] = {1.0f, 2.5f};
    EXPECT_EQ(sapient::testing::max_abs_err(a, b), 0.5f);
    EXPECT_TRUE(sapient::testing::within_abs(a, b, 0.5f));
    EXPECT_FALSE(sapient::testing::within_abs(a, b, 0.4f));
}

TEST(Compare, within_abs_rejects_nan_where_reference_is_finite) {
    const float got[] = {1.0f, NAN, 3.0f};
    const float ref[] = {1.0f, 2.0f, 3.0f};
    EXPECT_TRUE(std::isinf(sapient::testing::max_abs_err(got, ref)));
    EXPECT_FALSE(sapient::testing::within_abs(got, ref, 1e30f));

    // Both-NaN at the same index is treated as equal, not a mismatch.
    const float both_nan_got[] = {NAN};
    const float both_nan_ref[] = {NAN};
    EXPECT_EQ(sapient::testing::max_abs_err(both_nan_got, both_nan_ref), 0.0f);
    EXPECT_TRUE(sapient::testing::within_abs(both_nan_got, both_nan_ref, 0.0f));
}

TEST(Compare, golden_case_macro_skips_without_env) {
    if (std::getenv("SAPIENT_GOLDEN_DIR") != nullptr)
        GTEST_SKIP() << "env set; the skip path is exercised without it";
    std::string why;
    EXPECT_FALSE(sapient::testing::load_golden_case("does_not_matter", &why).has_value());
    EXPECT_NE(why.find("SAPIENT_GOLDEN_DIR"), std::string::npos);
}
