#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#
# Greedy token-id parity between two `greedy_ids` tools (Rust oracle vs C++ port).
#   greedy_parity.sh --rust <bin> --cpp <bin> --model <alias> [--prompt "<text>"] [--max-new N]
#   greedy_parity.sh --rust <bin> --self-check --model <alias>      # Rust vs Rust (determinism only)
# Both tools take `<model> <prompt> <max_new>` and print `prompt_ids\t…` / `output_ids\t…` lines.
# Exit 0 only when both lines are byte-identical. A missing binary is an ERROR, never a skip.
set -euo pipefail

RUST="" CPP="" MODEL="" PROMPT="Name three planets." MAX_NEW=64 SELF=0
need_value() { [ "$#" -ge 2 ] || { echo "greedy_parity: missing value for '$1'" >&2; exit 2; }; }
while [ $# -gt 0 ]; do
  case "$1" in
    --rust) need_value "$@"; RUST=$2; shift 2 ;;
    --cpp) need_value "$@"; CPP=$2; shift 2 ;;
    --model) need_value "$@"; MODEL=$2; shift 2 ;;
    --prompt) need_value "$@"; PROMPT=$2; shift 2 ;;
    --max-new) need_value "$@"; MAX_NEW=$2; shift 2 ;;
    --self-check) SELF=1; shift ;;
    *) echo "greedy_parity: unknown argument '$1'" >&2; exit 2 ;;
  esac
done
[ -n "$RUST" ] && [ -x "$RUST" ] || { echo "greedy_parity: --rust binary missing or not executable: '$RUST'" >&2; exit 2; }
[ -n "$MODEL" ] || { echo "greedy_parity: --model is required" >&2; exit 2; }
if [ "$SELF" = 1 ]; then
  CPP="$RUST"
  echo "greedy_parity: SELF-CHECK — comparing the Rust oracle against itself (determinism only, NOT C++ parity)"
fi
[ -n "$CPP" ] && [ -x "$CPP" ] || { echo "greedy_parity: --cpp binary missing or not executable: '$CPP'" >&2; exit 2; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
"$RUST" "$MODEL" "$PROMPT" "$MAX_NEW" > "$tmp/rust.txt"
"$CPP" "$MODEL" "$PROMPT" "$MAX_NEW" > "$tmp/cpp.txt"
grep -E $'^(prompt_ids|output_ids)\t' "$tmp/rust.txt" > "$tmp/rust.ids" || true   # no match → the count check below reports it
grep -E $'^(prompt_ids|output_ids)\t' "$tmp/cpp.txt" > "$tmp/cpp.ids" || true
[ "$(wc -l < "$tmp/rust.ids")" -eq 2 ] || { echo "greedy_parity: Rust output lacks prompt_ids/output_ids lines" >&2; cat "$tmp/rust.txt" >&2; exit 2; }
[ "$(wc -l < "$tmp/cpp.ids")" -eq 2 ] || { echo "greedy_parity: C++ output lacks prompt_ids/output_ids lines" >&2; cat "$tmp/cpp.txt" >&2; exit 2; }

n=$(awk -F'\t' '/^output_ids/ { print split($2, a, " ") }' "$tmp/rust.ids")
if cmp -s "$tmp/rust.ids" "$tmp/cpp.ids"; then
  echo "greedy_parity: PARITY OK  model=$MODEL output_tokens=$n prompt=\"$PROMPT\""
  exit 0
fi

echo "greedy_parity: PARITY FAIL  model=$MODEL" >&2
diff "$tmp/rust.ids" "$tmp/cpp.ids" >&2 || true
awk -F'\t' '/^output_ids/ { print $2 }' "$tmp/rust.ids" | tr ' ' '\n' > "$tmp/r"
awk -F'\t' '/^output_ids/ { print $2 }' "$tmp/cpp.ids" | tr ' ' '\n' > "$tmp/c"
idx=$(paste "$tmp/r" "$tmp/c" | awk -F'\t' '$1 != $2 { print NR - 1; exit }')
[ -n "$idx" ] && echo "greedy_parity: first divergent output token index: $idx" >&2
exit 1
