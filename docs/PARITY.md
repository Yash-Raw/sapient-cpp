# Parity ledger — Rust oracle vs C++ port

Every sub-project of the C++ rewrite (spec: `docs/superpowers/specs/2026-09-20-cpp-rewrite-design.md`,
§D5) records its gate results here. A row is only added after the gate ran on real hardware; a gate
that could not run is recorded as such, never as a pass. Commit hashes refer to this repository.

Conventions: **bit-identical** = byte-equal output; **token-identical** = equal `prompt_ids` and
`output_ids` from `cpp/tests/parity/greedy_parity.sh`; **max_err** = largest absolute difference.

## Harness self-validation (sub-project 0)

| Date | Gate | Host / ISA path | Rust commit | C++ commit | Result |
|---|---|---|---|---|---|
| 2026-09-20 | `.sapd` format round-trip: Rust `dump_kernels --format-sample` → C++ `sapient::testing::read_golden` (`Golden.ReadsFormatSampleWrittenByRust`) | macOS arm64 (Apple M5; NEON/SDOT) | 34d8449 | 2110b99 | pass — 7 arrays decoded exactly (fixture 306 B, byte-identical to a fresh tool run) |
| 2026-09-20 | Kernel dump determinism: two `dump_kernels --out` runs, same seed | macOS arm64 (Apple M5; NEON/SDOT) | 34d8449 | — | bit-identical across 3 runs, 22 cases |
| 2026-09-20 | `Golden.InventoryFromEnv` over the 22 host dumps | macOS arm64 (Apple M5; NEON/SDOT) | 34d8449 | 2110b99 | pass |
| 2026-09-20 | `greedy_parity.sh --self-check` smollm2-135m-q4, 32 tokens | macOS arm64 (Apple M5; NEON/SDOT) | 43c61da | — (Rust vs Rust) | token-identical |

## Sub-project 1a — core + IO + CPU kernels

### Open gaps carried from sub-project 0

- (a) The golden dumps cover the 22 public entry points only — `Q4_K_R4`/`Q6_K_R4` (SDOT/SMMLA/Q8_K
  decode paths), `apply_rope_partial`, and the `quantize_row_to_i8_blocks` /
  `quantize_row_to_q8k` activation formats are not dumped yet and must be added before 1a claims
  bit-identity on them.
- (b) The FMA-contraction detector's x86 twin (`BuildFlags.FpContractIsOffUnderFmaTarget`) is
  exercised only on the Linux/Windows CI hosts — this repo's dev/test host is arm64, where the
  `#if defined(__x86_64__) || defined(_M_X64)` block compiles out entirely.
- (c) The `sapient::testing` golden reader has no `expect_bit_identical`/`max_abs_err`/f16-tag
  helpers yet.

### Results

_(no rows yet)_

## Sub-project 1b — CPU chat vertical slice

_(no rows yet)_
