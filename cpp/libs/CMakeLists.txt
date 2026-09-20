# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
# Applies the parity FP-contract flags at THIS directory's scope (final review #7) — every
# add_subdirectory() below inherits them, but FetchContent deps (e.g. googletest, populated at
# the top-level directory scope by cmake/deps.cmake, before this file runs) never see them.
sapient_apply_parity_flags_here()

add_subdirectory(sapient-core)
if(SAPIENT_BUILD_TESTS)
  add_subdirectory(sapient-testing)
endif()
