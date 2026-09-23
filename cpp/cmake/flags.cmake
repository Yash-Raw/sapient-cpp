# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
# Build-flag discipline for bit-identity with the Rust oracle (spec D1).

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  if(SAPIENT_ALLOW_NON_CLANG)
    message(WARNING "SAPIENT: ${CMAKE_CXX_COMPILER_ID} is not a parity compiler — kernel golden gates may fail")
  else()
    message(FATAL_ERROR
      "SAPIENT requires Clang (clang-cl on Windows) for parity builds; found ${CMAKE_CXX_COMPILER_ID}. "
      "Pass -DSAPIENT_ALLOW_NON_CLANG=ON for a non-parity build.")
  endif()
endif()

# Rust never contracts a*b+c into an FMA implicitly; Clang defaults to -ffp-contract=on. Scoped
# to OUR code only (final review #7): this is a function, not applied here, so FetchContent deps
# (e.g. googletest, populated at the top-level directory scope by cmake/deps.cmake) never see it
# — only libs/apps/ffi CMakeLists.txt files that call sapient_apply_parity_flags_here() do, at
# their own directory scope (inherited by their add_subdirectory() children).
# -fno-math-errno: rustc lowers sin/cos/exp/pow/sqrt to LLVM intrinsics (no errno); Clang matches
# that lowering only with this flag (Darwin default, NOT the Linux default) — plan C ruling.
function(sapient_apply_parity_flags_here)
  if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    # UCRT marks getenv deprecated; clang-cl reports it as -Wdeprecated-declarations, and our
    # per-target /WX (sapient_apply_warnings) makes that fatal. The port reads SAPIENT_* knobs
    # via getenv, so silence the deprecation instead of avoiding the standard C API.
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    add_compile_options(/clang:-ffp-contract=off /clang:-fno-math-errno)
  else()
    add_compile_options(-ffp-contract=off -fno-fast-math -fno-math-errno)
  endif()
endfunction()

foreach(_bad IN ITEMS "-march=native" "-ffast-math" "-Ofast")
  if(CMAKE_CXX_FLAGS MATCHES "${_bad}" OR CMAKE_C_FLAGS MATCHES "${_bad}")
    message(FATAL_ERROR "SAPIENT: '${_bad}' breaks kernel bit-identity with the Rust build (spec D1)")
  endif()
endforeach()
