// fit_price_graph on a metadata-only graph: the weights are marked allocated
// (dummy non-null data) with no backend buffer, the way the loaders' measure
// mode marks them. The forced scheduler shape must price such a graph instead
// of aborting in ggml-alloc, and must agree with the direct pricer.
#include "fit_price.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace tts_cpp::detail;

namespace {

int g_failures = 0;

void fail(const std::string & what) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
}

void expect(bool cond, const std::string & what) {
    if (!cond) fail(what);
}

constexpr int64_t kDim        = 64;
constexpr int64_t kColumns    = 8;
constexpr size_t  kGraphSize  = 64;
constexpr size_t  kMaxTensors = 8;

void set_force_sched(bool forced) {
#ifdef _WIN32
    _putenv_s("TTS_CPP_FORCE_SCHED", forced ? "1" : "0");
#else
    setenv("TTS_CPP_FORCE_SCHED", forced ? "1" : "0", 1);
#endif
}

void mark_externally_allocated(ggml_context * ctx) {
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (!t->data && !t->view_src) {
            t->data = reinterpret_cast<void *>(static_cast<uintptr_t>(1));
        }
    }
}

struct metadata_only_graph {
    ggml_context * weights = nullptr;
    ggml_context * compute = nullptr;
    ggml_cgraph *  gf      = nullptr;
};

metadata_only_graph build_metadata_only_graph() {
    metadata_only_graph g;

    ggml_init_params weight_params = { ggml_tensor_overhead() * kMaxTensors, nullptr, true };
    g.weights = ggml_init(weight_params);
    ggml_tensor * w = ggml_new_tensor_2d(g.weights, GGML_TYPE_F32, kDim, kDim);
    ggml_tensor * b = ggml_new_tensor_1d(g.weights, GGML_TYPE_F32, kDim);
    mark_externally_allocated(g.weights);

    ggml_init_params compute_params = {
        ggml_tensor_overhead() * kMaxTensors + ggml_graph_overhead_custom(kGraphSize, false),
        nullptr, true };
    g.compute = ggml_init(compute_params);
    ggml_tensor * x = ggml_new_tensor_2d(g.compute, GGML_TYPE_F32, kDim, kColumns);
    ggml_set_input(x);
    ggml_tensor * w_view = ggml_reshape_2d(g.compute, w, kDim, kDim);
    ggml_tensor * b_view = ggml_reshape_1d(g.compute, b, kDim);
    ggml_tensor * y = ggml_add(g.compute, ggml_mul_mat(g.compute, w_view, x), b_view);
    ggml_set_output(y);
    g.gf = ggml_new_graph_custom(g.compute, kGraphSize, false);
    ggml_build_forward_expand(g.gf, y);
    return g;
}

void free_graph(metadata_only_graph & g) {
    ggml_free(g.compute);
    ggml_free(g.weights);
}

fit_graph_price price(ggml_backend_t backend, ggml_cgraph * gf, bool forced_sched) {
    set_force_sched(forced_sched);
    fit_graph_price out;
    expect(fit_price_graph(backend, gf, kGraphSize, out),
           forced_sched ? "forced scheduler pricing failed" : "direct pricing failed");
    return out;
}

// A GPU primary gives the forced scheduler its real two-backend shape,
// [primary, CPU-last], which is where an unplaceable leaf aborts the reserve;
// a CPU-only host still exercises the direct fallback.
ggml_backend_t init_primary_backend() {
    ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu) {
        ggml_backend_t backend = ggml_backend_dev_init(gpu, nullptr);
        if (backend) return backend;
    }
    return init_cpu_backend();
}

}  // namespace

int main() {
    ggml_backend_t primary = init_primary_backend();
    if (!primary) {
        fail("no backend");
        return g_failures;
    }
    std::printf("primary backend: %s\n", ggml_backend_name(primary));

    metadata_only_graph g = build_metadata_only_graph();
    expect(graph_has_unbuffered_weights(g.gf), "metadata-only weights were not detected");

    const fit_graph_price direct = price(primary, g.gf, false);
    expect(direct.device_bytes > 0, "direct pricing measured no compute scratch");
    expect(!direct.used_sched, "direct pricing went through the scheduler");

    const fit_graph_price forced = price(primary, g.gf, true);
    expect(!forced.used_sched, "a metadata-only graph was priced through the scheduler");
    expect(forced.device_bytes == direct.device_bytes,
           "forced pricing of a metadata-only graph disagrees with the direct pricer");

    free_graph(g);
    ggml_backend_free(primary);
    if (g_failures == 0) {
        std::printf("test-fit-price-unbuffered: all checks passed\n");
    }
    return g_failures;
}
