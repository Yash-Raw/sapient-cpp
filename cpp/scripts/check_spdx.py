#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
"""Fail if any source file under <root> lacks the two-line SAPIENT SPDX header.

`//` header: .hpp .cpp .h .c .mm .m .wgsl .metal .rs .swift .kt .ts .tsx
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
SLASH_EXT = {".hpp", ".cpp", ".h", ".c", ".mm", ".m", ".wgsl", ".metal", ".rs", ".swift", ".kt", ".ts", ".tsx"}
HASH_EXT = {".cmake", ".py", ".sh"}
HASH_NAMES = {"CMakeLists.txt", ".clang-format", ".clang-tidy"}
SKIP_DIRS = {"build", "third_party", "_deps", "fixtures", "node_modules", ".git", "target"}


def expected_header(path: pathlib.Path):
    if any(part in SKIP_DIRS for part in path.parts):
        return None
    if path.name in HASH_NAMES or path.suffix in HASH_EXT:
        return HASH
    if path.suffix in SLASH_EXT:
        return SLASH
    return None


def check_root(root: str):
    bad = []
    for path in sorted(pathlib.Path(root).rglob("*")):
        if not path.is_file():
            continue
        expected = expected_header(path)
        if expected is None:
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if lines and lines[0].startswith("#!"):
            lines = lines[1:]
        if tuple(lines[:2]) != expected:
            bad.append(path)
    return bad


def main(roots):
    bad = [p for root in roots for p in check_root(root)]
    for p in bad:
        print(f"check_spdx: missing/incorrect SPDX header: {p}", file=sys.stderr)
    print(f"check_spdx: {'FAIL' if bad else 'OK'} ({len(bad)} file(s) without the header; roots: {', '.join(roots)})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["."]))
