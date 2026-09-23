#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
"""Fail if any source file under <root> lacks the two-line SAPIENT SPDX header.

`//` header: .hpp .cpp .h .c .cc .cxx .hh .hxx .mm .m .wgsl .metal .rs .swift .kt .ts .tsx
`#`  header: .cmake .py .sh .clang-format .clang-tidy CMakeLists.txt   (a shebang may precede it)
Skipped: build/, third_party/, _deps/, fixtures/, node_modules/, .git/, *.json, *.md
Usage: check_spdx.py <root> [<root> ...]
"""
import pathlib
import sys

SLASH = (
    "// SPDX-License-Identifier: AGPL-3.0-only",
    "// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)",
)
HASH = tuple(line.replace("//", "#", 1) for line in SLASH)
SLASH_EXT = {
    ".hpp", ".cpp", ".h", ".c", ".cc", ".cxx", ".hh", ".hxx",
    ".mm", ".m", ".wgsl", ".metal", ".rs", ".swift", ".kt", ".ts", ".tsx",
}
HASH_EXT = {".cmake", ".py", ".sh"}
HASH_NAMES = {"CMakeLists.txt", ".clang-format", ".clang-tidy"}
SKIP_DIRS = {"build", "third_party", "_deps", "fixtures", "node_modules", ".git", "target"}


def expected_header(path: pathlib.Path, base: pathlib.Path):
    rel_parts = path.relative_to(base).parts
    if any(part in SKIP_DIRS for part in rel_parts):
        return None
    if path.name in HASH_NAMES or path.suffix in HASH_EXT:
        return HASH
    if path.suffix in SLASH_EXT:
        return SLASH
    return None


def check_root(root: str):
    base = pathlib.Path(root)
    if not base.is_dir():
        print(f"check_spdx: root does not exist: {root}", file=sys.stderr)
        sys.exit(2)
    scanned = 0
    bad = []
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        expected = expected_header(path, base)
        if expected is None:
            continue
        scanned += 1
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if lines and lines[0].startswith("#!"):
            lines = lines[1:]
        if tuple(lines[:2]) != expected:
            bad.append(path)
    if scanned == 0:
        print(f"check_spdx: no source files found under {root} (nothing was checked)", file=sys.stderr)
        sys.exit(2)
    return scanned, bad


def main(roots):
    scanned_total = 0
    bad = []
    for root in roots:
        scanned, root_bad = check_root(root)
        scanned_total += scanned
        bad.extend(root_bad)
    for p in bad:
        print(f"check_spdx: missing/incorrect SPDX header: {p}", file=sys.stderr)
    print(
        f"check_spdx: {'FAIL' if bad else 'OK'} "
        f"({scanned_total} file(s) checked, {len(bad)} without the header; roots: {', '.join(roots)})"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["."]))
