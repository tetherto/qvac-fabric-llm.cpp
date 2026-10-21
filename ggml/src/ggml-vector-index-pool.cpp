// ggml-vector-index-pool.cpp - the process-wide worker pool behind
// parallel_for.
//
// One pool serves every index and every calling thread. It starts no
// threads until a call first asks for more than one, and grows to the
// largest count asked for (at most kMaxThreads - 1 helpers, since the
// caller always works too). A call posts a job, runs tasks on its own
// thread as worker 0, and waits only for the helpers that joined its job,
// so concurrent callers share the helpers and never wait on each other's
// tasks. A parallel_for made from inside a task runs inline.

#include "ggml-vector-index-impl.h"

#include <condition_variable>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#endif
#if defined(_MSC_VER) && !defined(__clang__) && defined(_M_ARM64)
#include <intrin.h>
#endif

namespace {

// Set while this thread runs a parallel_for task, as a helper or as the
// caller; a parallel_for made then runs inline.
thread_local bool t_in_task = false;

struct PoolJob {
    size_t n_tasks = 0;
    size_t max_helpers = 0;
    ParallelTaskFn fn = nullptr;
    void * ctx = nullptr;
    // The caller's floating-point environment, which every helper runs
    // under, so a task computes the same on any thread.
    std::fenv_t env;
    std::atomic<size_t> next{ 0 };
    // Helpers still running tasks; changed under the pool mutex, read
    // without it while the caller spins.
    std::atomic<size_t> active{ 0 };
    // Under the pool mutex: helpers that took a worker id, whether the
    // caller sleeps on `done`, and the job's place in the pool's queue.
    size_t joined = 0;
    bool caller_waiting = false;
    bool queued = false;
    PoolJob * queue_next = nullptr;
    std::condition_variable done;
    std::mutex error_mutex;
    std::exception_ptr error;
};

// Hands out the job's tasks to this worker until none are left. The first
// exception a task throws is kept and stops the hand-out.
void run_tasks(PoolJob & job, size_t worker) {
    try {
        for (size_t task = job.next.fetch_add(1); task < job.n_tasks; task = job.next.fetch_add(1)) {
            job.fn(job.ctx, task, worker);
        }
    } catch (...) {
        std::lock_guard<std::mutex> lock(job.error_mutex);
        if (!job.error) {
            job.error = std::current_exception();
        }
        job.next.store(job.n_tasks);
    }
}

inline void spin_pause() {
#if defined(__aarch64__) || defined(_M_ARM64)
#if defined(_MSC_VER) && !defined(__clang__)
    __yield();
#else
    __asm__ __volatile__("yield");
#endif
#elif defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}

// Spins until done() holds or about kSpinMicros pass; true if it held. A
// search runs a few short jobs back to back, and a helper that would sleep
// between them pays a wake-up each time.
constexpr int64_t kSpinMicros = 50;

template <typename Done>
bool spin_until(Done && done) {
    const auto start = std::chrono::steady_clock::now();
    for (;;) {
        for (int i = 0; i < 64; ++i) {
            if (done()) {
                return true;
            }
            spin_pause();
        }
        if (std::chrono::steady_clock::now() - start > std::chrono::microseconds(kSpinMicros)) {
            return done();
        }
    }
}

class Pool {
public:
    // Runs the job with up to job.max_helpers helpers; false when the pool
    // is shut down and the caller must run it alone.
    bool run(PoolJob & job) {
        size_t to_wake = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stop_) {
                return false;
            }
            grow(job.max_helpers);
            enqueue(job);
            to_wake = std::min(job.max_helpers, sleeping_);
        }
        for (size_t i = 0; i < to_wake; ++i) {
            wake_.notify_one();
        }
        t_in_task = true;
        run_tasks(job, 0);
        t_in_task = false;
        // No helper joins once the tasks are all handed out, so the caller
        // may wait for the ones running before it takes the job off the
        // queue.
        spin_until([&] { return job.active.load(std::memory_order_acquire) == 0; });
        // Taking the lock also waits out a helper still inside its
        // critical section after its last decrement.
        std::unique_lock<std::mutex> lock(mutex_);
        dequeue(job);
        if (job.active.load(std::memory_order_relaxed) != 0) {
            job.caller_waiting = true;
            job.done.wait(lock, [&] { return job.active.load(std::memory_order_relaxed) == 0; });
        }
        return true;
    }

    // Stops the helpers once they finish what they are running. Jobs still
    // in flight finish on their callers' threads.
    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            stop_flag_.store(true, std::memory_order_relaxed);
        }
        wake_.notify_all();
        // No thread is started once stop_ is set, so threads_ is final.
        for (std::thread & thread : threads_) {
#ifdef _WIN32
            // A DLL's static destructors run under the loader lock (and at
            // process exit after its threads are gone), where a join can
            // hang; the stopped helpers exit on their own.
            thread.detach();
#else
            thread.join();
#endif
        }
    }

    size_t size() {
        std::lock_guard<std::mutex> lock(mutex_);
        return threads_.size();
    }

#ifndef _WIN32
    pid_t owner_pid = ::getpid();
#endif

private:
    // Starts helpers up to `want`; a thread that fails to start leaves the
    // pool smaller, as a failed spawn left parallel_for with fewer threads.
    void grow(size_t want) {
        want = std::min(want, static_cast<size_t>(kMaxThreads - 1));
        while (threads_.size() < want) {
            try {
                threads_.emplace_back([this] { helper_main(); });
            } catch (...) {
                return;
            }
        }
    }

    void enqueue(PoolJob & job) {
        job.queued = true;
        job.queue_next = nullptr;
        if (tail_ == nullptr) {
            head_ = &job;
        } else {
            tail_->queue_next = &job;
        }
        tail_ = &job;
        n_queued_.fetch_add(1, std::memory_order_release);
    }

    void dequeue(PoolJob & job) {
        if (!job.queued) {
            return;
        }
        PoolJob * prev = nullptr;
        for (PoolJob * it = head_; it != nullptr; prev = it, it = it->queue_next) {
            if (it == &job) {
                (prev == nullptr ? head_ : prev->queue_next) = job.queue_next;
                if (tail_ == &job) {
                    tail_ = prev;
                }
                break;
            }
        }
        job.queued = false;
        job.queue_next = nullptr;
        n_queued_.fetch_sub(1, std::memory_order_relaxed);
    }

    void helper_main() {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            if (!stop_ && head_ == nullptr) {
                // Just finished a job, or just started: spin a little for the
                // next before sleeping.
                lock.unlock();
                spin_until([&] {
                    return n_queued_.load(std::memory_order_acquire) != 0 || stop_flag_.load(std::memory_order_relaxed);
                });
                lock.lock();
                ++sleeping_;
                wake_.wait(lock, [&] { return stop_ || head_ != nullptr; });
                --sleeping_;
            }
            if (stop_) {
                return;
            }
            PoolJob & job = *head_;
            if (job.next.load() >= job.n_tasks) {
                // Nothing left to hand out; the caller finishes it.
                dequeue(job);
                continue;
            }
            const size_t worker = ++job.joined;
            job.active.fetch_add(1, std::memory_order_relaxed);
            if (job.joined == job.max_helpers) {
                dequeue(job);
            }
            lock.unlock();
            std::fesetenv(&job.env);
            t_in_task = true;
            run_tasks(job, worker);
            t_in_task = false;
            lock.lock();
            // The job lives on its caller's stack: once active reaches zero
            // the caller may return as soon as it gets the lock, so the job
            // is not touched after this unlocks.
            if (job.active.fetch_sub(1, std::memory_order_acq_rel) == 1 && job.caller_waiting) {
                job.done.notify_all();
            }
        }
    }

    std::mutex mutex_;
    std::condition_variable wake_;
    std::vector<std::thread> threads_;
    PoolJob * head_ = nullptr;
    PoolJob * tail_ = nullptr;
    bool stop_ = false;
    // Helpers asleep on wake_ (under the mutex), and lock-free mirrors of
    // the queue length and stop_ for spinning helpers.
    size_t sleeping_ = 0;
    std::atomic<size_t> n_queued_{ 0 };
    std::atomic<bool> stop_flag_{ false };
};

// The pool is never freed: a helper may still be leaving it while static
// destructors run, and a forked child abandons its parent's pool, whose
// threads and lock state did not survive the fork.
std::atomic<Pool *> g_pool{ nullptr };
std::atomic<bool> g_pool_shut_down{ false };

#ifndef _WIN32
// A forked child's abandoned copy of its parent's pool, kept referenced so a
// leak checker still reaches it.
Pool * g_abandoned_pool = nullptr;
#endif

Pool * acquire_pool() {
    if (g_pool_shut_down.load(std::memory_order_acquire)) {
        return nullptr;
    }
    Pool * pool = g_pool.load(std::memory_order_acquire);
#ifndef _WIN32
    if (pool != nullptr && pool->owner_pid != ::getpid()) {
        // A forked child: start a pool of its own.
        Pool * stale = pool;
        Pool * fresh = new (std::nothrow) Pool();
        if (fresh == nullptr) {
            return nullptr;
        }
        if (g_pool.compare_exchange_strong(stale, fresh, std::memory_order_acq_rel)) {
            g_abandoned_pool = stale;
            return fresh;
        }
        delete fresh;
        return stale != nullptr && stale->owner_pid == ::getpid() ? stale : nullptr;
    }
#endif
    if (pool != nullptr) {
        return pool;
    }
    Pool * fresh = new (std::nothrow) Pool();
    if (fresh == nullptr) {
        return nullptr;
    }
    if (g_pool.compare_exchange_strong(pool, fresh, std::memory_order_acq_rel)) {
        return fresh;
    }
    delete fresh;
    return pool;
}

// Stops the helpers when the library unloads or the process exits; a
// parallel_for after that runs on its caller's thread.
struct PoolShutdown {
    PoolShutdown() = default;
    PoolShutdown(const PoolShutdown &) = delete;
    PoolShutdown & operator=(const PoolShutdown &) = delete;
    ~PoolShutdown() {
        g_pool_shut_down.store(true, std::memory_order_release);
        // Stopped but kept referenced: never freed, as above, and still
        // reachable through g_pool for a leak checker.
        Pool * pool = g_pool.load(std::memory_order_acquire);
#ifndef _WIN32
        if (pool != nullptr && pool->owner_pid != ::getpid()) {
            return;
        }
#endif
        if (pool != nullptr) {
            pool->shutdown();
        }
    }
};
PoolShutdown g_pool_shutdown;

void run_inline(size_t n_tasks, ParallelTaskFn fn, void * ctx) {
    for (size_t task = 0; task < n_tasks; ++task) {
        fn(ctx, task, 0);
    }
}

}  // namespace

void parallel_run(size_t n_tasks, size_t workers, ParallelTaskFn fn, void * ctx) {
    if (workers <= 1 || t_in_task) {
        run_inline(n_tasks, fn, ctx);
        return;
    }
    Pool * pool = acquire_pool();
    if (pool == nullptr) {
        run_inline(n_tasks, fn, ctx);
        return;
    }
    PoolJob job;
    job.n_tasks = n_tasks;
    job.max_helpers = workers - 1;
    job.fn = fn;
    job.ctx = ctx;
    std::fegetenv(&job.env);
    if (!pool->run(job)) {
        run_inline(n_tasks, fn, ctx);
        return;
    }
    if (job.error) {
        std::rethrow_exception(job.error);
    }
}

#ifdef GGML_VEC_INDEX_TEST_HOOKS
size_t parallel_pool_threads_for_test() {
    Pool * pool = acquire_pool();
    return pool == nullptr ? 0 : pool->size();
}

// Exercises parallel_for at n_threads: every task runs once with a worker
// id below the worker count, a nested call runs inline on its task's
// thread, a task's exception reaches the caller and stops the hand-out,
// and the caller's rounding mode reaches the helpers. Returns 0 or the
// number of the first failed check.
int parallel_pool_selftest_for_test(int n_threads) {
    constexpr size_t kTasks = 257;
    const size_t workers = std::min(kTasks, static_cast<size_t>(std::max(1, n_threads)));
    std::vector<std::atomic<int>> runs(kTasks);
    std::atomic<bool> bad_worker{ false };
    std::atomic<bool> nested_moved{ false };
    std::atomic<bool> nested_short{ false };
    parallel_for(kTasks, n_threads, [&](size_t task, size_t worker) {
        if (worker >= workers) {
            bad_worker = true;
        }
        runs[task].fetch_add(1);
        const std::thread::id self = std::this_thread::get_id();
        size_t inner = 0;
        parallel_for(8, n_threads, [&](size_t, size_t inner_worker) {
            if (std::this_thread::get_id() != self || inner_worker != 0) {
                nested_moved = true;
            }
            ++inner;
        });
        if (inner != 8) {
            nested_short = true;
        }
    });
    for (const std::atomic<int> & r : runs) {
        if (r.load() != 1) {
            return 1;
        }
    }
    if (bad_worker) {
        return 2;
    }
    if (nested_moved || nested_short) {
        return 3;
    }
    std::atomic<size_t> started{ 0 };
    bool caught = false;
    try {
        parallel_for(100000, n_threads, [&](size_t task, size_t) {
            started.fetch_add(1);
            if (task == 3) {
                throw std::runtime_error("task failed");
            }
        });
    } catch (const std::runtime_error &) {
        caught = true;
    }
    if (!caught) {
        return 4;
    }
    if (n_threads <= 1 && started.load() != 4) {
        return 5;
    }
    const int saved = std::fegetround();
    std::fesetround(FE_UPWARD);
    std::atomic<bool> wrong_rounding{ false };
    parallel_for(64, n_threads, [&](size_t, size_t) {
        if (std::fegetround() != FE_UPWARD) {
            wrong_rounding = true;
        }
    });
    std::fesetround(saved);
    if (wrong_rounding) {
        return 6;
    }
    return 0;
}
#endif
