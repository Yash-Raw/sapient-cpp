#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
"""Byte-compare the WGSL shader copies under cpp/ with the Rust originals under crates/.

The C++ wgpu engine (sub-project 4) runs the same 20 shaders through wgpu-native; until the
Rust tree is removed (sub-project 9) the two copies must stay identical.
Usage: shader_sync.py <repo-root>
"""
import filecmp
import pathlib
import sys

RUST = "crates/sapient-backends/wgpu/src/shaders"
CPP = "cpp/libs/sapient-backends-wgpu/shaders"


def main(repo: str) -> int:
    root = pathlib.Path(repo)
    rust, cpp = root / RUST, root / CPP
    if not cpp.is_dir():
        print(f"shader_sync: not a repository root (no {CPP}): {repo}", file=sys.stderr)
        return 2
    if not rust.is_dir():
        print("shader_sync: Rust shader directory absent (post-removal) — nothing to compare")
        return 0
    r = {p.name for p in rust.glob("*.wgsl")}
    c = {p.name for p in cpp.glob("*.wgsl")}
    problems = [f"missing in cpp/: {n}" for n in sorted(r - c)]
    problems += [f"extra in cpp/ (no Rust original): {n}" for n in sorted(c - r)]
    problems += [f"differs: {n}" for n in sorted(r & c) if not filecmp.cmp(rust / n, cpp / n, shallow=False)]
    for p in problems:
        print(f"shader_sync: {p}", file=sys.stderr)
    print(f"shader_sync: {'FAIL' if problems else 'OK'} ({len(r & c)} shader(s) identical, {len(problems)} problem(s))")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
