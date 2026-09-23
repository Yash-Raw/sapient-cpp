// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/core/version.hpp"

namespace sapient::core {

std::string_view version() noexcept {
    return SAPIENT_VERSION_STRING;
}

} // namespace sapient::core
