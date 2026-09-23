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
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

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

// ── Little-endian byte helpers ──────────────────────────────────────────────────────────────
template <class T> void put_le(std::vector<uint8_t>& out, T v) {
    uint8_t b[sizeof(T)];
    std::memcpy(b, &v, sizeof(T)); // hosts are little-endian (sapient::io static_asserts it)
    out.insert(out.end(), b, b + sizeof(T));
}
inline void put_str(std::vector<uint8_t>& out, std::string_view s) {
    put_le<uint64_t>(out, s.size());
    out.insert(out.end(), s.begin(), s.end());
}
inline std::vector<uint8_t> le_u16(std::initializer_list<uint16_t> v) {
    std::vector<uint8_t> out;
    for (const uint16_t x : v)
        put_le(out, x);
    return out;
}
inline std::vector<uint8_t> le_f32(std::initializer_list<float> v) {
    std::vector<uint8_t> out;
    for (const float x : v)
        put_le(out, x);
    return out;
}

/// Writes GGUF bytes the way llama.cpp does: header, KVs, tensor infos, zero padding to
/// `alignment`, then each tensor's data at an `alignment`-aligned offset from data_start.
/// `alignment` here is only the WRITER's padding — tests that set `general.alignment` must set
/// this to the same value.
struct GgufBuilder {
    uint32_t magic = 0x46554747;
    uint32_t version = 3;
    size_t alignment = 32;
    std::vector<uint8_t> kv;
    uint64_t kv_count = 0;
    std::optional<uint64_t> tensor_count_override; // for the absurd-count tests
    std::optional<uint64_t> kv_count_override;

    struct T {
        std::string name;
        std::vector<uint64_t> dims;
        uint32_t kind;
        std::vector<uint8_t> data;
        std::optional<uint64_t> offset_override;
    };
    std::vector<T> tensors;

    GgufBuilder& kv_raw(std::string_view key, uint32_t vtype, std::span<const uint8_t> payload) {
        put_str(kv, key);
        put_le<uint32_t>(kv, vtype);
        kv.insert(kv.end(), payload.begin(), payload.end());
        ++kv_count;
        return *this;
    }
    template <class V> GgufBuilder& kv_scalar(std::string_view key, uint32_t vtype, V v) {
        std::vector<uint8_t> p;
        put_le(p, v);
        return kv_raw(key, vtype, p);
    }
    GgufBuilder& kv_u8(std::string_view k, uint8_t v) { return kv_scalar(k, 0, v); }
    GgufBuilder& kv_i8(std::string_view k, int8_t v) { return kv_scalar(k, 1, v); }
    GgufBuilder& kv_u16(std::string_view k, uint16_t v) { return kv_scalar(k, 2, v); }
    GgufBuilder& kv_i16(std::string_view k, int16_t v) { return kv_scalar(k, 3, v); }
    GgufBuilder& kv_u32(std::string_view k, uint32_t v) { return kv_scalar(k, 4, v); }
    GgufBuilder& kv_i32(std::string_view k, int32_t v) { return kv_scalar(k, 5, v); }
    GgufBuilder& kv_f32(std::string_view k, float v) { return kv_scalar(k, 6, v); }
    GgufBuilder& kv_bool_byte(std::string_view k, uint8_t v) { return kv_scalar(k, 7, v); }
    GgufBuilder& kv_u64(std::string_view k, uint64_t v) { return kv_scalar(k, 10, v); }
    GgufBuilder& kv_i64(std::string_view k, int64_t v) { return kv_scalar(k, 11, v); }
    GgufBuilder& kv_f64(std::string_view k, double v) { return kv_scalar(k, 12, v); }
    GgufBuilder& kv_str(std::string_view k, std::string_view v) {
        std::vector<uint8_t> p;
        put_str(p, v);
        return kv_raw(k, 8, p);
    }
    /// An array KV: item type, element count, then the raw item payload.
    GgufBuilder& kv_array(std::string_view k,
                          uint32_t item_type,
                          uint64_t count,
                          std::span<const uint8_t> items) {
        std::vector<uint8_t> p;
        put_le<uint32_t>(p, item_type);
        put_le<uint64_t>(p, count);
        p.insert(p.end(), items.begin(), items.end());
        return kv_raw(k, 9, p);
    }
    GgufBuilder& kv_arr_str(std::string_view k, std::initializer_list<std::string_view> v) {
        std::vector<uint8_t> items;
        for (const auto s : v)
            put_str(items, s);
        return kv_array(k, 8, v.size(), items);
    }

    GgufBuilder&
    tensor(std::string name, std::vector<uint64_t> dims, uint32_t kind, std::vector<uint8_t> data) {
        tensors.push_back({std::move(name), std::move(dims), kind, std::move(data), std::nullopt});
        return *this;
    }

    static size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

    /// Relative data offsets: each tensor aligned, unless overridden.
    std::vector<uint64_t> offsets() const {
        std::vector<uint64_t> out;
        size_t off = 0;
        for (const auto& t : tensors) {
            off = align_up(off, alignment);
            out.push_back(t.offset_override.value_or(off));
            off += t.data.size();
        }
        return out;
    }

    /// Everything parse_header reads — no padding, no data.
    std::vector<uint8_t> header_bytes() const {
        std::vector<uint8_t> out;
        put_le<uint32_t>(out, magic);
        put_le<uint32_t>(out, version);
        put_le<uint64_t>(out, tensor_count_override.value_or(tensors.size()));
        put_le<uint64_t>(out, kv_count_override.value_or(kv_count));
        out.insert(out.end(), kv.begin(), kv.end());
        const auto offs = offsets();
        for (size_t i = 0; i < tensors.size(); ++i) {
            put_str(out, tensors[i].name);
            put_le<uint32_t>(out, static_cast<uint32_t>(tensors[i].dims.size()));
            for (const uint64_t d : tensors[i].dims)
                put_le<uint64_t>(out, d);
            put_le<uint32_t>(out, tensors[i].kind);
            put_le<uint64_t>(out, offs[i]);
        }
        return out;
    }

    std::vector<uint8_t> build() const {
        std::vector<uint8_t> out = header_bytes();
        const size_t data_start = align_up(out.size(), alignment);
        out.resize(data_start, 0);
        const auto offs = offsets();
        for (size_t i = 0; i < tensors.size(); ++i) {
            if (tensors[i].offset_override) continue; // deliberately out-of-place: data not written
            out.resize(data_start + offs[i], 0);
            out.insert(out.end(), tensors[i].data.begin(), tensors[i].data.end());
        }
        return out;
    }
};

} // namespace sapient::io::test
