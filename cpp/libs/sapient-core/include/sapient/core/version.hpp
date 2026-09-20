// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once

#include <string_view>

namespace sapient::core {

/// Engine version — mirrors the Rust workspace `[workspace.package] version`.
std::string_view version() noexcept;

}  // namespace sapient::core
