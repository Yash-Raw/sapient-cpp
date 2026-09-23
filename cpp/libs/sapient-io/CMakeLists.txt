# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
# Port of crates/sapient-io (sub-project 1a plan B). `mmap` stands in for memmap2 + std::fs::read,
# `rust_std` for the Rust-std error texts the loaders embed. ONNX and the Graph entry points are
# sub-project 8.
add_library(sapient_io STATIC src/rust_std.cpp src/mmap.cpp src/gguf.cpp src/safetensors.cpp
                               src/io.cpp)
add_library(sapient::io ALIAS sapient_io)
target_include_directories(sapient_io PUBLIC include)
target_link_libraries(sapient_io PUBLIC sapient::core)
# PRIVATE: no public header of sapient_io includes <nlohmann/json.hpp>.
target_link_libraries(sapient_io PRIVATE nlohmann_json::nlohmann_json)
sapient_apply_warnings(sapient_io)

if(SAPIENT_BUILD_TESTS)
  add_executable(sapient_io_tests tests/rust_std_test.cpp tests/mmap_test.cpp tests/gguf_test.cpp
                                  tests/safetensors_test.cpp tests/real_file_test.cpp)
  target_include_directories(sapient_io_tests PRIVATE tests)
  # backends_cpu is TEST-ONLY: plan D's golden-gated quantize_q8_0_block is the differential
  # oracle for quantize_to_q8_0. The io library itself must never link it.
  target_link_libraries(sapient_io_tests PRIVATE sapient::io sapient::backends_cpu GTest::gtest_main)
  sapient_apply_warnings(sapient_io_tests)
  gtest_discover_tests(sapient_io_tests)
endif()
