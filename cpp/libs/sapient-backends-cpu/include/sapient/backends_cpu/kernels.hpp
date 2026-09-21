// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// = crates/sapient-backends/cpu/src/kernels/mod.rs: one include per kernel family.
// kernels/quant.hpp joins in plan D.

#include "sapient/backends_cpu/kernels/attention.hpp"
#include "sapient/backends_cpu/kernels/conv2d.hpp"
#include "sapient/backends_cpu/kernels/elementwise.hpp"
#include "sapient/backends_cpu/kernels/layernorm.hpp"
#include "sapient/backends_cpu/kernels/matmul.hpp"
#include "sapient/backends_cpu/kernels/reduce.hpp"
#include "sapient/backends_cpu/kernels/rope.hpp"
#include "sapient/backends_cpu/kernels/softmax.hpp"
