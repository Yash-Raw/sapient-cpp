# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
cmake_minimum_required(VERSION 3.24)

# Version: single source of truth is the Rust workspace manifest while crates/ exists.
# No hardcoded fallback here on purpose: a broken extraction (or a missing Cargo.toml once
# the Rust tree is removed) must fail configure loudly, not silently pin a stale version.
set(_cargo_toml "${CMAKE_CURRENT_SOURCE_DIR}/../Cargo.toml")
if(EXISTS "${_cargo_toml}")
  file(READ "${_cargo_toml}" _cargo)
  # `version` is the first key after the [workspace.package] header in ../Cargo.toml.
  string(REGEX MATCH "\\[workspace\\.package\\][ \t\r\n]*version[ \t]*=[ \t]*\"([0-9]+\\.[0-9]+\\.[0-9]+)\"" _m "${_cargo}")
  if(NOT CMAKE_MATCH_1)
    message(FATAL_ERROR
      "SAPIENT: could not extract [workspace.package] version from ../Cargo.toml — "
      "fix the regex in cpp/CMakeLists.txt or pass -DSAPIENT_VERSION=X.Y.Z")
  endif()
  set(_extracted_version "${CMAKE_MATCH_1}")
  if(DEFINED SAPIENT_VERSION AND NOT "${SAPIENT_VERSION}" STREQUAL "")
    if(NOT "${SAPIENT_VERSION}" STREQUAL "${_extracted_version}")
      message(FATAL_ERROR
        "SAPIENT: -DSAPIENT_VERSION=${SAPIENT_VERSION} does not match the version extracted "
        "from ../Cargo.toml (${_extracted_version})")
    endif()
  else()
    set(SAPIENT_VERSION "${_extracted_version}")
  endif()
else()
  if(NOT DEFINED SAPIENT_VERSION OR "${SAPIENT_VERSION}" STREQUAL "")
    message(FATAL_ERROR
      "SAPIENT: ../Cargo.toml not found (the Rust tree has been removed) and no "
      "-DSAPIENT_VERSION=X.Y.Z was given — pass the version explicitly")
  endif()
endif()

project(sapient VERSION ${SAPIENT_VERSION} LANGUAGES C CXX)

option(SAPIENT_BUILD_TESTS "Build the GoogleTest suites and lint ctests" ON)
option(SAPIENT_MLX "Metal engine via the MLX C++ API (macOS, sub-project 4)" OFF)
option(SAPIENT_WGPU "Cross-platform GPU engine via wgpu-native (sub-project 4)" OFF)
option(SAPIENT_AUDIO_IO "Live mic/speaker via miniaudio (sub-project 5c)" ON)
option(SAPIENT_ACCELERATE "Apple Accelerate cblas for f32 GEMM (sub-project 1a)" OFF)
option(SAPIENT_ALLOW_NON_CLANG "Allow GCC/MSVC — NOT a parity build" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(flags)     # Clang gate + -ffp-contract=off (global)
include(warnings)  # sapient_apply_warnings(<target>)
include(deps)      # FetchContent pins

if(SAPIENT_BUILD_TESTS)
  enable_testing()
  include(GoogleTest)
endif()

# apps/ and ffi/ (later sub-projects) must call sapient_apply_parity_flags_here() the same way
# libs/CMakeLists.txt does, at their own directory scope.
add_subdirectory(libs)

if(SAPIENT_BUILD_TESTS)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_test(NAME lint.spdx_headers
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_spdx.py ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME lint.shader_sync
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/shader_sync.py ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()
