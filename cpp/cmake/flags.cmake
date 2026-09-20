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

# Rust never contracts a*b+c into an FMA implicitly; Clang defaults to -ffp-contract=on.
if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  add_compile_options(/clang:-ffp-contract=off)
else()
  add_compile_options(-ffp-contract=off -fno-fast-math)
endif()

foreach(_bad IN ITEMS "-march=native" "-ffast-math" "-Ofast")
  if(CMAKE_CXX_FLAGS MATCHES "${_bad}" OR CMAKE_C_FLAGS MATCHES "${_bad}")
    message(FATAL_ERROR "SAPIENT: '${_bad}' breaks kernel bit-identity with the Rust build (spec D1)")
  endif()
endforeach()
