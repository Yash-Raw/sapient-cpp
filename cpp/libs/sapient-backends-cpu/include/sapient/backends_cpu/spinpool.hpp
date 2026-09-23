// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#pragma once
// Port of crates/sapient-backends/cpu/src/spinpool.rs (plan E). Persistent spin-wait worker pool
// for the decode hot path (llama.cpp-style).
//
// One decoded token runs ~200+ GEMV parallel regions — one per matmul call — and every fork/join
// region pays worker wake + park latency (µs-scale futex round-trips), ~230 barriers per token.
// Here the workers stay HOT for the duration of a generation: between ops they spin (bounded
// iterations, then park on a condvar), so dispatching an op during decode costs a few atomic
// operations instead of thread wakeups.
//
// ## Contract
// - `run(n_chunks, f)` executes `f(0..n_chunks)` exactly once each, in parallel (the caller
//   participates), returning only after ALL chunks complete. Chunks must touch disjoint data —
//   the same contract as `parallel::par_chunks_mut`.
// - Concurrent publishers serialize on a lock: two overlapping matmuls degrade to two
//   back-to-back fully-parallel ops (never a deadlock, and each still uses the whole pool).
//   `f` must therefore NOT call `run` again — the publish mutex is not recursive, exactly as in
//   Rust. No kernel nests parallel regions.
// - `enabled()` is false while the thermal governor is shedding cores (spinning workers would
//   defeat the backoff) and when `SAPIENT_SPINPOOL=0`. Callers fall back to `parallel`.
//
// ## Op-handoff protocol (seqlock-style)
// The generation counter is EVEN when an op is published and ODD while the slot is being
// rewritten. A publisher (holding `publish_`):
//   1. bumps `generation_` to ODD — closes the door: no worker can newly join,
//   2. waits for `active_ == 0` (workers still inside the previous op leave),
//   3. rewrites the op slot + chunk counters,
//   4. bumps `generation_` to EVEN and wakes parked workers.
// A worker joins by registering in `active_` FIRST and then making its authoritative
// `generation_` read (both SeqCst): if that read saw the old even generation it happened before
// the odd bump, so the publisher's drain in step 2 observes the registration and waits the worker
// out; if it happened after, the worker sees ODD and backs out. Either way a worker can never
// copy the slot while it is being rewritten. (The original order — drain BEFORE the odd bump —
// left exactly that window, and it segfaulted in practice under rapid park/wake cycling.
// `Spinpool.rapid_ops_with_constant_parking` pins the fixed behaviour.)
//
// The plain (non-atomic) `op_` member is not a data race: the publisher's writes are ordered
// before the SeqCst store that makes `generation_` even, and a worker only reads the slot after
// an acquiring/SeqCst load that observed that value — a textbook release/acquire pair.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace sapient::backends_cpu::spinpool {

/// Spin iterations before a worker parks on the condvar. MEASURED optimum on M4 (llama-1B Q4_K_M
/// decode sweep: 0 → 54.2 tok/s, 4k → 63.6, 16k → 59.1, 50k → 52.3, 200k hot-spin → 38.6): ~4k
/// iterations (~10–20 µs) catches the back-to-back GEMVs inside a layer, while parking through the
/// longer serial phases (attention, sampling). Long spins actively HURT — workers burning cores
/// during serial phases steal the package power budget / scheduler slots from the critical-path
/// thread. Overridable via `SAPIENT_SPINPOOL_SPINS`.
inline constexpr uint64_t kDefaultSpinIters = 4000;

class SpinPool {
public:
    /// Rust `SpinPool::new` — allocates a pool, spawns `workers` detached threads and returns a
    /// reference that lives for the rest of the process (Rust's `Box::leak`). Deliberately leaked:
    /// the detached workers must never outlive the mutexes and condition variables they wait on,
    /// which a static destructor running at exit would destroy underneath them. Reached by `pool()`
    /// and, as in Rust, directly by the `rapid_ops_with_constant_parking` stress test.
    static SpinPool& create(size_t workers, uint64_t spin_iters);

    SpinPool(const SpinPool&) = delete;
    SpinPool& operator=(const SpinPool&) = delete;

    /// Execute `f(0..n_chunks)` in parallel across the pool + this thread. Returns after every
    /// chunk has run. Chunks must write disjoint data. `run` itself installs no exception handler:
    /// an escaping exception is `std::terminate` on a worker, and on the publisher it would unwind
    /// out of `run_erased` while workers may still be inside `execute_blocks` holding `op.ctx` — a
    /// pointer into the frame being destroyed. `f` must therefore not throw; the in-tree caller
    /// (`for_each_out_chunk`) is responsible for wrapping it. `f` must also not itself call `run`.
    template <class F> void run(size_t n_chunks, const F& f) {
        run_erased(n_chunks, &thunk<F>, static_cast<const void*>(&f));
    }

    /// Worker threads (excludes the participating publisher).
    size_t workers() const { return workers_; }

private:
    using CallFn = void (*)(const void*, size_t);

    /// Rust's `thunk::<F>` trampoline: `ctx` is the `&F` handed to `run`, alive until `run` returns
    /// (the publisher blocks until every block completes).
    template <class F> static void thunk(const void* ctx, size_t c) {
        (*static_cast<const F*>(ctx))(c);
    }

    struct OpSlot {
        CallFn call;
        const void* ctx;
        size_t n_chunks;
        /// Chunks per claimed block (guided scheduling granularity).
        size_t block;
    };

    /// Pad each hot atomic to its own cache line (128 B covers Apple Silicon's line pairs). Without
    /// this, `generation_` — which every idle worker spins on — shares a line with
    /// `completed_`/`next_block_`, so every completion invalidates the spinners' line and every
    /// spin-load contends the completer's store: measured ~2× decode REGRESSION on M4 before
    /// padding. This is why llama.cpp's threadpool pads its counters.
    template <class T> struct alignas(128) Pad {
        T v;
    };

    SpinPool(size_t workers, uint64_t spin_iters);

    void run_erased(size_t n_chunks, CallFn call, const void* ctx);
    void execute_blocks(const OpSlot& op);
    void worker_loop();

    /// Seqlock generation: even = published, odd = slot being rewritten.
    Pad<std::atomic<uint64_t>> generation_{};
    OpSlot op_;
    /// GUIDED claiming: participants grab contiguous BLOCKS of chunks off this counter (~3 blocks
    /// per participant off macOS). v1 per-chunk claiming load-balanced the M4's P/E cores (+5% vs
    /// the fork/join path) but its ~112 RMWs/op on two hot lines was a 2× regression on 14-core
    /// Thor; v2 static shares fixed Thor but made every op wait for the slowest E-core on M4.
    /// Guided blocks keep v2's contiguity at ~4× less claim traffic than v1 while letting fast
    /// cores take more blocks.
    Pad<std::atomic<size_t>> next_block_{};
    /// Completed BLOCKS (one increment per block, not per chunk).
    Pad<std::atomic<size_t>> completed_{};
    /// Workers currently inside an op (validated slot copy → last share).
    Pad<std::atomic<size_t>> active_{};
    std::mutex publish_;
    std::mutex sleep_;
    std::condition_variable wake_;
    Pad<std::atomic<size_t>> parked_{};
    size_t workers_;
    uint64_t spin_iters_;
};

/// The process-global pool: `parallel::num_threads() - 1` workers (the publishing thread
/// participates), matching the task budget `gemv_chunk` computes from the same figure. Lazily
/// created on first use. Env: `SAPIENT_SPINPOOL_WORKERS`, `SAPIENT_SPINPOOL_SPINS`.
SpinPool& pool();

/// Worker threads + the participating publisher — the parallelism the task count should be sized
/// for when the pool is active (`gemv_chunk` uses it). Instantiates the pool.
size_t parallelism();

/// Whether the spin pool should be used for this dispatch. Off when `SAPIENT_SPINPOOL=0` (A/B
/// lever / escape hatch) and while the thermal governor or an external level is shedding cores —
/// parked workers shed heat, spinning workers do not, so governed decode must stay on the
/// `parallel` path. Does NOT instantiate the pool.
bool enabled();

} // namespace sapient::backends_cpu::spinpool
