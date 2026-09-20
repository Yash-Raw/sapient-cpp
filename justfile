# SAPIENT workspace Justfile
# Run `just --list` to see all available tasks.

default:
    @just --list

# ── Build ─────────────────────────────────────────────────────────────────────

build:
    cargo build --workspace

release:
    cargo build --workspace --release

# ── Test ─────────────────────────────────────────────────────────────────────

test:
    cargo test --workspace

test-verbose:
    cargo test --workspace -- --nocapture

# ── Lint ─────────────────────────────────────────────────────────────────────

clippy:
    cargo clippy --workspace --all-targets -- -D warnings

fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all -- --check

# ── Benchmark ─────────────────────────────────────────────────────────────────

bench:
    cargo bench --workspace

# ── Docs ─────────────────────────────────────────────────────────────────────

doc:
    cargo doc --workspace --no-deps --open

# ── CLI shortcuts ─────────────────────────────────────────────────────────────

# Example: just run model.onnx
run model:
    cargo run -p sapient-cli -- run {{model}}

# Example: just bench-model model.onnx
bench-model model:
    cargo run -p sapient-cli --release -- bench {{model}} --batch-sizes 1,4,8,16

# Example: just serve model.onnx
serve model port="8080":
    cargo run -p sapient-cli --release -- serve {{model}} --port {{port}}

# ── Cross-compilation for Raspberry Pi ───────────────────────────────────────

cross-rpi:
    cross build --target aarch64-unknown-linux-gnu --release

# ── Publish to crates.io (in dependency order) ───────────────────────────────

publish: fmt-check test
    @echo "Publishing SAPIENT crates to crates.io..."
    cargo publish -p sapient-core      && sleep 10
    cargo publish -p sapient-ir        && sleep 10
    cargo publish -p sapient-backends-cpu && sleep 10
    cargo publish -p sapient-scheduler && sleep 10
    cargo publish -p sapient-io        && sleep 10
    cargo publish -p sapient-telemetry && sleep 10
    cargo publish -p sapient-runtime   && sleep 10
    cargo publish -p sapient-cli
    @echo "✅ All crates published!"

# Dry-run publish (check metadata without uploading)
publish-dry:
    cargo publish -p sapient-core         --dry-run
    cargo publish -p sapient-ir           --dry-run
    cargo publish -p sapient-backends-cpu --dry-run
    cargo publish -p sapient-scheduler    --dry-run
    cargo publish -p sapient-io           --dry-run
    cargo publish -p sapient-telemetry    --dry-run
    cargo publish -p sapient-runtime      --dry-run
    cargo publish -p sapient-cli          --dry-run

# ── C++ tree (cpp/) — Rust→C++ parity port ─────────────────────────────────────
# Configure the C++ tree (presets: dev, release, ci-macos, ci-linux, ci-windows)
cpp-configure preset="dev":
    cd cpp && cmake --preset {{preset}}

# Build the C++ tree
cpp-build preset="dev":
    cd cpp && cmake --build --preset {{preset}}

# Run the C++ tests (set SAPIENT_GOLDEN_DIR to include the kernel golden gate)
cpp-test preset="dev":
    cd cpp && ctest --preset {{preset}}

# Format the C++ sources in place
cpp-fmt:
    git ls-files -- 'cpp/*.hpp' 'cpp/*.cpp' 'cpp/*.h' 'cpp/*.mm' | xargs clang-format -i

# Lint gates that need no build: SPDX headers (both trees) + WGSL shader sync
cpp-lint:
    python3 cpp/scripts/check_spdx.py cpp crates
    python3 cpp/scripts/shader_sync.py .

# Regenerate the kernel golden dumps on this host with the Rust oracle
cpp-golden out="/tmp/sapient-golden":
    cpp/tests/parity/golden_dump.sh {{out}}
