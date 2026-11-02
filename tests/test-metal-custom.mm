#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"

#import <Metal/Metal.h>

#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <thread>

static void start_tag(ggml_tensor *, int, int, void *) { GGML_ABORT("Metal required"); }
static void finish_tag(ggml_tensor *, int, int, void *) { GGML_ABORT("Metal required"); }

struct worker_state {
    id<MTLSharedEvent> ready;
    id<MTLSharedEvent> done;
    uint64_t sequence = 0;
    std::thread worker;
    std::atomic<bool> failed{false};
};

static bool execute_start(ggml_backend_t backend, ggml_tensor * dst, void * userdata) {
    auto * state = static_cast<worker_state *>(userdata);
    if (state->worker.joinable()) { state->worker.join(); }
    const uint64_t value = ++state->sequence;
    const float * input = static_cast<const float *>(dst->src[0]->data);
    float * output = static_cast<float *>(dst->data);
    ggml_backend_metal_shared_event(backend, (__bridge void *) state->ready, value, false);
    state->worker = std::thread([=] {
        @autoreleasepool {
            if (![state->ready waitUntilSignaledValue:value timeoutMS:10000]) { state->failed = true; }
            for (int i = 0; i < 32; ++i) { output[i] = 2 * input[i]; output[32 + i] = 3 * input[i]; }
            state->done.signaledValue = value;
        }
    });
    return true;
}

static bool execute_finish(ggml_backend_t backend, ggml_tensor *, void * userdata) {
    auto * state = static_cast<worker_state *>(userdata);
    ggml_backend_metal_shared_event(backend, (__bridge void *) state->done, state->sequence, true);
    return true;
}

static bool run(ggml_backend_dev_t dev) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    worker_state state;
    state.ready = [device newSharedEvent];
    state.done = [device newSharedEvent];
    ggml_context * ctx = ggml_init({ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(32, false), nullptr, true});
    auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 8);
    ggml_set_input(input);
    auto * x = ggml_scale(ctx, input, 2);
    auto * start = ggml_custom_4d(ctx, GGML_TYPE_F32, 4, 8, 2, 1, &x, 1, start_tag, 1, &state);
    auto * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, start);
    auto * gpu = ggml_scale(ctx, x, 5);
    ggml_build_forward_expand(graph, gpu);
    ggml_tensor * dependencies[] = {x, gpu};
    auto * finish = ggml_custom_inplace(ctx, start, dependencies, 2, finish_tag, 1, &state);
    auto * a = ggml_view_2d(ctx, finish, 4, 8, finish->nb[1], 0);
    auto * b = ggml_view_2d(ctx, finish, 4, 8, finish->nb[1], finish->nb[2]);
    auto * out = ggml_add(ctx, gpu, ggml_add(ctx, a, b));
    ggml_set_output(out);
    ggml_build_forward_expand(graph, out);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    bool ok = ggml_backend_dev_supports_op(dev, start) && ggml_backend_dev_supports_op(dev, finish) && ggml_gallocr_alloc_graph(alloc, graph);
    for (int iteration = 0; ok && iteration < 3; ++iteration) {
        float values[32], result[32];
        for (int i = 0; i < 32; ++i) { values[i] = float(i + iteration); }
        ggml_backend_tensor_set(input, values, 0, sizeof(values));
        ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        ggml_backend_tensor_get(out, result, 0, sizeof(result));
        for (int i = 0; i < 32; ++i) { ok = ok && result[i] == 20 * values[i]; }
    }
    if (state.worker.joinable()) { state.worker.join(); }
    ok = ok && !state.failed;
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return ok;
}

int main() {
    @autoreleasepool {
        ggml_backend_load_all();
        auto * dev = ggml_backend_dev_by_name("MTL0");
        if (!dev) { fprintf(stderr, "Metal unavailable\n"); return 77; }
        ggml_backend_metal_set_custom_serial(start_tag, execute_start);
        ggml_backend_metal_set_custom_serial(finish_tag, execute_finish);
        // Cleaning up the legacy experiment must not remove library registrations.
        ggml_backend_metal_set_custom(nullptr, nullptr);
        bool first = false, second = false;
        std::thread a([&] { @autoreleasepool { first = run(dev); } });
        std::thread b([&] { @autoreleasepool { second = run(dev); } });
        a.join();
        b.join();
        ggml_backend_metal_set_custom_serial(start_tag, nullptr);
        ggml_backend_metal_set_custom_serial(finish_tag, nullptr);
        printf("Metal custom-op registration, identity waits and independent workers: %s\n", first && second ? "passed" : "FAILED");
        return first && second ? 0 : 1;
    }
}
