// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include <gtest/gtest.h>

#include <string>
#include <system_error>
#include <vector>

#include "sapient/core/error.hpp"

using sapient::core::Error;
using sapient::core::ErrorCode;
using sapient::core::Result;

// Rust: error_display
TEST(Error, error_display) {
    const auto e = Error::shape_mismatch({2, 3}, {2, 4});
    const std::string s = e.to_string();
    EXPECT_NE(s.find("Shape mismatch"), std::string::npos) << s;
    EXPECT_NE(s.find("[2, 3]"), std::string::npos) << s;
    EXPECT_EQ(s, "Shape mismatch: expected [2, 3], got [2, 4]");
}

// Rust: from_io_error
TEST(Error, from_io_error) {
    const auto e =
        Error::io(std::make_error_code(std::errc::no_such_file_or_directory), "file missing");
    EXPECT_EQ(e.code, ErrorCode::Io);
    EXPECT_EQ(e.to_string().rfind("IO error: ", 0), 0u) << e.to_string();
}

TEST(Error, messages_match_rust_format_strings) {
    EXPECT_EQ(Error::rank_mismatch(4, 3).to_string(), "Rank mismatch: expected 4, got 3");
    EXPECT_EQ(Error::type_mismatch("a quantized dtype", "f32").to_string(),
              "Type mismatch: expected a quantized dtype, got f32");
    EXPECT_EQ(Error::broadcast({2, 3}, {2, 4}).to_string(),
              "Incompatible shapes for broadcasting: [2, 3] and [2, 4]");
    EXPECT_EQ(Error::cyclic_graph().to_string(),
              "Graph contains a cycle — execution is impossible");
    EXPECT_EQ(Error::node_not_found("x").to_string(), "Node \"x\" not found in graph");
    EXPECT_EQ(Error::invalid_graph("Shape has zero dimension at axis 1").to_string(),
              "Graph validation failed: Shape has zero dimension at axis 1");
    EXPECT_EQ(Error::allocation_failed(32, 64).to_string(),
              "Allocation failed: requested 32 bytes (alignment 64)");
    EXPECT_EQ(Error::buffer_size_mismatch(10, 4).to_string(),
              "Buffer size mismatch: expected 10 bytes, got 4");
    EXPECT_EQ(Error::internal("t() requires a 2-D tensor").to_string(),
              "Internal error: t() requires a 2-D tensor");
    EXPECT_EQ(Error::gguf_parse("bad GGUF magic").to_string(), "GGUF parse error: bad GGUF magic");
    EXPECT_EQ(Error::model_not_found("/x").to_string(), "Model not found at path '/x'");
}

TEST(Error, try_macro_propagates) {
    auto inner = [](bool ok) -> Result<int> {
        if (!ok) return tl::unexpected(Error::internal("boom"));
        return 7;
    };
    auto outer = [&](bool ok) -> Result<int> {
        SAPIENT_TRY_ASSIGN(int v, inner(ok));
        return v + 1;
    };
    ASSERT_TRUE(outer(true).has_value());
    EXPECT_EQ(*outer(true), 8);
    ASSERT_FALSE(outer(false).has_value());
    EXPECT_EQ(outer(false).error().to_string(), "Internal error: boom");
}

TEST(Error, debug_str_escapes_control_characters_like_rust) {
    EXPECT_EQ(Error::node_not_found(std::string("a\0b", 3)).to_string(),
              "Node \"a\\0b\" not found in graph");
    EXPECT_EQ(Error::node_not_found("x\x1by\x7f").to_string(),
              "Node \"x\\u{1b}y\\u{7f}\" not found in graph");
    EXPECT_EQ(Error::node_not_found("q\"\\\n\t\r").to_string(),
              "Node \"q\\\"\\\\\\n\\t\\r\" not found in graph");
}

TEST(Error, try_macro_discard_form_and_result_void) {
    auto step = [](bool ok) -> Result<void> {
        if (!ok) return tl::unexpected(Error::internal("step failed"));
        return {};
    };
    auto run = [&](bool ok) -> Result<int> {
        SAPIENT_TRY(step(ok));
        // clang-format off
        SAPIENT_TRY(step(true)); SAPIENT_TRY(step(true));  // two expansions on one line must compile
        // clang-format on
        return 1;
    };
    ASSERT_TRUE(run(true).has_value());
    EXPECT_EQ(run(false).error().to_string(), "Internal error: step failed");
}
