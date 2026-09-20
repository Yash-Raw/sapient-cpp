# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
# Test-support library: reader for the Rust oracle's .sapd golden dumps. Not shipped.
add_library(sapient_testing STATIC src/golden.cpp)
add_library(sapient::testing ALIAS sapient_testing)
target_include_directories(sapient_testing PUBLIC include)
sapient_apply_warnings(sapient_testing)

if(SAPIENT_BUILD_TESTS)
  add_executable(sapient_testing_tests tests/golden_test.cpp)
  target_link_libraries(sapient_testing_tests PRIVATE sapient::testing GTest::gtest_main)
  target_compile_definitions(sapient_testing_tests PRIVATE
    SAPIENT_TESTING_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")
  sapient_apply_warnings(sapient_testing_tests)
  gtest_discover_tests(sapient_testing_tests)
endif()
