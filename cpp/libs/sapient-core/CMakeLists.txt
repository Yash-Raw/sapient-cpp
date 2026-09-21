# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
add_library(sapient_core STATIC src/version.cpp src/error.cpp src/dtype.cpp src/shape.cpp src/f16.cpp src/buffer.cpp src/dequant.cpp)
add_library(sapient::core ALIAS sapient_core)
target_include_directories(sapient_core PUBLIC include)
target_link_libraries(sapient_core PUBLIC tl::expected)
target_compile_definitions(sapient_core PRIVATE SAPIENT_VERSION_STRING="${PROJECT_VERSION}")
sapient_apply_warnings(sapient_core)

if(SAPIENT_BUILD_TESTS)
  add_executable(sapient_core_tests
    tests/version_test.cpp tests/build_flags_test.cpp tests/error_test.cpp tests/dtype_test.cpp tests/shape_test.cpp tests/f16_test.cpp tests/buffer_test.cpp tests/dequant_test.cpp)
  target_link_libraries(sapient_core_tests PRIVATE sapient::core GTest::gtest_main)
  target_compile_definitions(sapient_core_tests PRIVATE
    SAPIENT_VERSION_STRING="${PROJECT_VERSION}"
    SAPIENT_CARGO_TOML_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../../../Cargo.toml")
  sapient_apply_warnings(sapient_core_tests)
  gtest_discover_tests(sapient_core_tests)
endif()
