#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#
# Regenerate the kernel golden dumps ON THIS HOST with the Rust oracle, then point the C++
# tests at them:   cpp/tests/parity/golden_dump.sh <out-dir> [--seed N]
#                  SAPIENT_GOLDEN_DIR=<out-dir> ctest --preset dev
# Dumps are host-specific (ISA dispatch) — never commit them.
set -euo pipefail
# Resolve to an absolute path before handing it to `cargo run` — a relative out-dir would be
# interpreted relative to the Rust workspace root (we `cd "$repo"` below), not the caller's cwd.
out=$(mkdir -p "${1:?usage: golden_dump.sh <out-dir> [--seed N]}" && cd "$1" && pwd)
shift
repo=$(cd "$(dirname "$0")/../../.." && pwd)
(cd "$repo" && cargo run --release -q -p sapient-backends-cpu --example dump_kernels -- --out "$out" "$@")
count=$(find "$out" -name '*.sapd' | wc -l | tr -d ' ')
[ "$count" -gt 0 ] || { echo "golden_dump: no .sapd files were written to $out" >&2; exit 1; }
echo "golden_dump: $count cases in $out — export SAPIENT_GOLDEN_DIR=$out before running ctest"
