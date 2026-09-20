# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
cmake_minimum_required(VERSION 3.24)

# Version: single source of truth is the Rust workspace manifest while crates/ exists.
set(_cargo_toml "${CMAKE_CURRENT_SOURCE_DIR}/../Cargo.toml")
if(EXISTS "${_cargo_toml}")
  file(READ "${_cargo_toml}" _cargo)
  # `version` is the first key after the [workspace.package] header in ../Cargo.toml.
  string(REGEX MATCH "\\[workspace\\.package\\][ \t\r\n]*version[ \t]*=[ \t]*\"([0-9]+\\.[0-9]+\\.[0-9]+)\"" _m "${_cargo}")
  set(SAPIENT_VERSION "${CMAKE_MATCH_1}")
endif()
if(NOT SAPIENT_VERSION)
  set(SAPIENT_VERSION "0.6.0")
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

add_subdirectory(libs/sapient-core)
