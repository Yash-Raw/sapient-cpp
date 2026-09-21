// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/backends_cpu/parallel.hpp"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <thread>

#include "sapient/backends_cpu/env.hpp"
#include "sapient/core/panic.hpp"

namespace sapient::backends_cpu::parallel {
namespace {

size_t default_threads() {
    const unsigned hc = std::thread::hardware_concurrency();
    return hc == 0 ? 1 : static_cast<size_t>(hc);
}

// rayon-core 1.13 ThreadPoolBuilder::get_num_threads: Some(x ≥ 1) → x, Some(0) → default,
// None → fall through to the deprecated RAYON_RS_NUM_CPUS, then the default.
size_t compute_num_threads() {
    for (const char* name : {"RAYON_NUM_THREADS", "RAYON_RS_NUM_CPUS"}) {
        const auto v = env_usize(name);
        if (!v.has_value()) continue;
        return *v >= 1 ? *v : default_threads();
    }
    return default_threads();
}

struct Job {
    const std::function<void(size_t)>* f;
    size_t n;
    size_t next{0}; // guarded by Pool::mu_
    size_t done{0}; // guarded by Pool::mu_
};

// A job queue with the caller always participating in its own job: progress never depends on a
// worker being free, so nested par_for (a chunk that itself calls par_for) cannot deadlock —
// every waiting thread only waits on chunks that some running thread has already claimed.
class Pool {
public:
    // Leaked on purpose, like rayon's global registry: worker threads are detached and must never
    // outlive their mutex/condvars, which a static destructor at exit would destroy.
    static Pool& instance() {
        static Pool* p = new Pool();
        return *p;
    }

    void run(Job& job) {
        if (workers_ == 0) {
            for (size_t i = 0; i < job.n; ++i)
                (*job.f)(i);
            return;
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            queue_.push_back(&job);
        }
        cv_.notify_all();
        while (true) {
            size_t i = 0;
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (job.next >= job.n) {
                    erase_locked(&job);
                    break;
                }
                i = job.next++;
            }
            (*job.f)(i);
            finish(job);
        }
        std::unique_lock<std::mutex> lk(mu_);
        done_cv_.wait(lk, [&] { return job.done == job.n; });
    }

private:
    Pool() : workers_(num_threads() - 1) {
        for (size_t w = 0; w < workers_; ++w)
            std::thread([this] { worker(); }).detach();
    }

    void worker() {
        while (true) {
            Job* job = nullptr;
            size_t i = 0;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return !queue_.empty(); });
                job = queue_.front();
                if (job->next >= job->n) {
                    queue_.pop_front(); // fully claimed; its owner is waiting on `done`
                    continue;
                }
                i = job->next++;
            }
            (*job->f)(i);
            finish(*job);
        }
    }

    void finish(Job& job) {
        std::lock_guard<std::mutex> lk(mu_);
        if (++job.done == job.n) done_cv_.notify_all();
    }

    void erase_locked(Job* job) {
        const auto it = std::find(queue_.begin(), queue_.end(), job);
        if (it != queue_.end()) queue_.erase(it);
    }

    std::mutex mu_;
    std::condition_variable cv_;
    std::condition_variable done_cv_;
    std::deque<Job*> queue_;
    size_t workers_;
};

} // namespace

size_t num_threads() {
    static const size_t n = compute_num_threads();
    return n;
}

void par_for(size_t n, const std::function<void(size_t)>& f) {
    if (n == 0) return;
    Job job{&f, n};
    Pool::instance().run(job);
}

void par_chunks_mut(std::span<float> out,
                    size_t chunk,
                    const std::function<void(size_t, std::span<float>)>& f) {
    if (chunk == 0) sapient::core::panic("par_chunks_mut: chunk size must not be zero");
    if (out.empty()) return;
    const size_t len = out.size();
    const size_t n_chunks = (len + chunk - 1) / chunk;
    par_for(n_chunks, [&](size_t ci) {
        const size_t start = ci * chunk;
        const size_t end = std::min(start + chunk, len);
        f(ci, out.subspan(start, end - start));
    });
}

} // namespace sapient::backends_cpu::parallel
