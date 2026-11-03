#include "ggml.h"
#include "ggml-cpu.h"

#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sched.h>

static cpu_set_t affinity() {
    cpu_set_t mask;
    GGML_ASSERT(sched_getaffinity(0, sizeof(mask), &mask) == 0);
    return mask;
}

static void expect_affinity(const cpu_set_t & expected) {
    const cpu_set_t actual = affinity();
    GGML_ASSERT(CPU_EQUAL(&actual, &expected));
}

struct probe {
    cpu_set_t expected;
    bool abort;
    int calls;
    std::atomic<int> workers{0};
};

static bool check_compute(void * data) {
    auto * p = static_cast<probe *>(data);
    expect_affinity(p->expected);
    p->calls++;
    return p->abort;
}

static void check_workers(ggml_tensor * dst, const ggml_tensor * src, int ith, int nth, void * data) {
    auto * p = static_cast<probe *>(data);
    expect_affinity(p->expected);
    p->workers.fetch_add(1);
    for (int64_t i = ith; i < ggml_nelements(src); i += nth) {
        ggml_set_f32_1d(dst, i, ggml_get_f32_1d(src, i));
    }
}

static void run_case(const cpu_set_t & inherited, int cpu, bool paused, int threads) {
    GGML_ASSERT(sched_setaffinity(0, sizeof(inherited), &inherited) == 0);
    auto params = ggml_threadpool_params_default(threads);
    params.cpumask[cpu] = true;
    params.paused = paused;
    auto * pool = ggml_threadpool_new(&params);
    GGML_ASSERT(pool);
    expect_affinity(inherited);

    auto * ctx = ggml_init({1024 * 1024, nullptr, false});
    GGML_ASSERT(ctx);
    auto * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
    ggml_set_f32(input, 2.0f);
    probe p = {};
    CPU_ZERO(&p.expected);
    CPU_SET(cpu, &p.expected);
    auto * checked = ggml_map_custom1(ctx, input, check_workers, GGML_N_TASKS_MAX, &p);
    auto * output = ggml_sqr(ctx, ggml_sqr(ctx, checked));
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    auto plan = ggml_graph_plan(graph, threads, pool);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.data();
    plan.abort_callback = check_compute;
    plan.abort_callback_data = &p;

    for (int round = 0; round < 4; round++) {
        if (round == 1) ggml_threadpool_pause(pool);
        if (round == 2) ggml_threadpool_resume(pool);
        p.abort = round == 2;
        p.calls = 0;
        p.workers = 0;
        const auto status = ggml_graph_compute(graph, &plan);
        GGML_ASSERT(status == (p.abort ? GGML_STATUS_ABORTED : GGML_STATUS_SUCCESS));
        GGML_ASSERT(p.calls > 0);
        GGML_ASSERT(p.workers == threads);
        expect_affinity(inherited);
        std::thread child([&] { expect_affinity(inherited); });
        child.join();
        if (!p.abort) GGML_ASSERT(ggml_get_f32_1d(output, 0) == 16.0f);
    }
    ggml_threadpool_free(pool);
    expect_affinity(inherited);
    ggml_free(ctx);
}
#endif

int main() {
#if defined(__linux__)
    const cpu_set_t original = affinity();
    std::vector<int> cpus;
    for (int i = 0; i < GGML_MAX_N_THREADS && i < CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, &original)) cpus.push_back(i);
    }
    if (cpus.size() < 2) {
        std::puts("SKIP: CPU affinity test needs two allowed CPUs");
        return 0;
    }
    for (bool paused : {false, true}) {
        for (int threads : {1, 2}) {
            run_case(original, cpus[0], paused, threads);
            cpu_set_t restricted = original;
            CPU_CLR(cpus.back(), &restricted);
            run_case(restricted, cpus[0], paused, threads);
        }
    }
    GGML_ASSERT(sched_setaffinity(0, sizeof(original), &original) == 0);
    std::puts("CPU affinity preserved across creation, compute, abort, resume and destruction");
#else
    std::puts("SKIP: Linux/Android CPU affinity regression");
#endif
}
