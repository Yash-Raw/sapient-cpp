// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/testing/golden.hpp"

#include <algorithm>
#include <bit>
#include <fstream>
#include <iterator>

static_assert(std::endian::native == std::endian::little, "the .sapd reader assumes a little-endian host");

namespace sapient::testing {

namespace {

constexpr uint32_t kFormatVersion = 1;
// Sanity caps on file-provided counts, checked before they drive allocations (reserve/resize) —
// a corrupted or adversarial .sapd must fail with a clean error, never std::bad_alloc/length_error.
constexpr uint32_t kMaxArrays = 4096;
constexpr uint32_t kMaxDims = 64;

struct Cursor {
    const std::vector<uint8_t>& buf;
    size_t pos = 0;

    // Subtraction form: `n` comes straight from the file (e.g. an 8-byte byte_len) and can be
    // near SIZE_MAX, so `pos + n` would silently wrap and this would wrongly report `true`.
    bool has(size_t n) const { return pos <= buf.size() && n <= buf.size() - pos; }

    template <class T>
    bool read(T& out) {
        if (!has(sizeof(T))) return false;
        std::memcpy(&out, buf.data() + pos, sizeof(T));
        pos += sizeof(T);
        return true;
    }

    bool read_string(std::string& out) {
        uint32_t len = 0;
        if (!read(len) || !has(len)) return false;
        out.assign(reinterpret_cast<const char*>(buf.data() + pos), len);
        pos += len;
        return true;
    }

    bool read_bytes(std::vector<uint8_t>& out, uint64_t len) {
        if (!has(static_cast<size_t>(len))) return false;
        // Never form `pos + len` first — `has()` already proved `len` fits, so build the end
        // iterator from `first`, not from a second wrap-prone sum.
        const auto first = buf.begin() + static_cast<std::ptrdiff_t>(pos);
        out.assign(first, first + static_cast<std::ptrdiff_t>(len));
        pos += static_cast<size_t>(len);
        return true;
    }
};

void set_error(std::string* error, std::string message) {
    if (error != nullptr) *error = std::move(message);
}

// Overflow-checked product of `dims`, matching GoldenArray::numel()'s convention (empty -> 1).
// Used to validate file-provided dims BEFORE they're multiplied against dtype_size for the
// byte-length check — a crafted dims array must not be able to wrap `numel()` into a small
// value that spuriously satisfies `bytes.size() == numel()*dtype_size`.
bool checked_numel(const std::vector<uint64_t>& dims, size_t& out) {
    size_t n = 1;
    for (const auto d : dims) {
        const auto dv = static_cast<size_t>(d);
        if (dv != 0 && n > SIZE_MAX / dv) return false;
        n *= dv;
    }
    out = dims.empty() ? 1 : n;
    return true;
}

}  // namespace

size_t dtype_size(GoldenDType dtype) {
    switch (dtype) {
    case GoldenDType::F32:
    case GoldenDType::I32:
    case GoldenDType::U32:
        return 4;
    case GoldenDType::U8:
    case GoldenDType::I8:
        return 1;
    case GoldenDType::U64:
        return 8;
    }
    throw std::logic_error("unknown GoldenDType tag");
}

size_t GoldenArray::numel() const {
    size_t n = 1;
    for (const auto d : dims) n *= static_cast<size_t>(d);
    return dims.empty() ? 1 : n;
}

const GoldenArray* GoldenCase::find(std::string_view array_name) const {
    for (const auto& a : arrays) {
        if (a.name == array_name) return &a;
    }
    return nullptr;
}

const GoldenArray& GoldenCase::get(std::string_view array_name) const {
    const auto* a = find(array_name);
    if (a == nullptr) throw std::out_of_range("golden case '" + name + "' has no array '" + std::string(array_name) + "'");
    return *a;
}

std::optional<GoldenCase> read_golden(const std::filesystem::path& file, std::string* error) {
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        set_error(error, "cannot open " + file.string());
        return std::nullopt;
    }
    const std::vector<uint8_t> buf{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    Cursor cur{buf};

    if (buf.size() < 4 || std::memcmp(buf.data(), "SAPD", 4) != 0) {
        set_error(error, file.string() + ": bad magic (expected \"SAPD\")");
        return std::nullopt;
    }
    cur.pos = 4;
    uint32_t version = 0;
    if (!cur.read(version)) {
        set_error(error, file.string() + ": truncated before version");
        return std::nullopt;
    }
    if (version != kFormatVersion) {
        set_error(error, file.string() + ": unsupported .sapd version " + std::to_string(version));
        return std::nullopt;
    }

    GoldenCase c;
    uint32_t n_arrays = 0;
    if (!cur.read_string(c.name) || !cur.read(n_arrays)) {
        set_error(error, file.string() + ": truncated header");
        return std::nullopt;
    }
    if (n_arrays > kMaxArrays) {
        set_error(error, file.string() + ": implausible n_arrays " + std::to_string(n_arrays));
        return std::nullopt;
    }
    c.arrays.reserve(n_arrays);
    for (uint32_t i = 0; i < n_arrays; ++i) {
        GoldenArray a;
        uint8_t tag = 0;
        uint32_t ndim = 0;
        uint64_t byte_len = 0;
        if (!cur.read_string(a.name) || !cur.read(tag) || !cur.read(ndim)) {
            set_error(error, file.string() + ": truncated array header #" + std::to_string(i));
            return std::nullopt;
        }
        if (tag > static_cast<uint8_t>(GoldenDType::U64)) {
            set_error(error, file.string() + ": unknown dtype tag " + std::to_string(tag) + " in " + a.name);
            return std::nullopt;
        }
        a.dtype = static_cast<GoldenDType>(tag);
        if (ndim > kMaxDims) {
            set_error(error, file.string() + ": implausible ndim " + std::to_string(ndim) + " in " + a.name);
            return std::nullopt;
        }
        a.dims.resize(ndim);
        for (auto& d : a.dims) {
            if (!cur.read(d)) {
                set_error(error, file.string() + ": truncated dims in " + a.name);
                return std::nullopt;
            }
        }
        size_t expected_numel = 0;
        if (!checked_numel(a.dims, expected_numel)) {
            set_error(error, file.string() + ": dims overflow in " + a.name);
            return std::nullopt;
        }
        if (!cur.read(byte_len) || !cur.read_bytes(a.bytes, byte_len)) {
            set_error(error, file.string() + ": truncated payload in " + a.name);
            return std::nullopt;
        }
        if (a.bytes.size() != expected_numel * dtype_size(a.dtype)) {
            set_error(error, file.string() + ": byte length does not match dims×dtype in " + a.name);
            return std::nullopt;
        }
        c.arrays.push_back(std::move(a));
    }
    if (cur.pos != buf.size()) {
        set_error(error, file.string() + ": trailing bytes after last array");
        return std::nullopt;
    }
    return c;
}

std::vector<std::filesystem::path> list_golden(const std::filesystem::path& dir) {
    std::vector<std::filesystem::path> out;
    if (!std::filesystem::is_directory(dir)) return out;
    for (const auto& e : std::filesystem::directory_iterator(dir)) {
        if (e.is_regular_file() && e.path().extension() == ".sapd") out.push_back(e.path());
    }
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace sapient::testing
