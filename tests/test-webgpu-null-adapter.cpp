#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <vector>

// Dawn's Null adapter accepts every feature and executes no shader, so a
// WebGPU device built on it returns stale data instead of results. The
// backend must refuse it: every WebGPU device that registers has to compute,
// and with no usable GPU at all none may register and the CPU must remain the
// best backend (the --expect-no-device form, run with the Vulkan drivers
// hidden from the process).

static const int64_t elements        = 64;
static const float   addend_a        = 1.0f;
static const float   addend_b        = 2.0f;
static const char *  webgpu_reg_name = "WebGPU";
static const char *  expect_no_device_flag = "--expect-no-device";

struct add_graph {
    ggml_context * ctx;
    ggml_tensor *  a;
    ggml_tensor *  b;
    ggml_tensor *  sum;
    ggml_cgraph *  graph;
};

static add_graph make_add_graph() {
    ggml_init_params params = { ggml_tensor_overhead() * 3 + ggml_graph_overhead(), nullptr, true };
    ggml_context *   ctx    = ggml_init(params);
    ggml_tensor *    a      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, elements);
    ggml_tensor *    b      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, elements);
    ggml_tensor *    sum    = ggml_add(ctx, a, b);
    ggml_cgraph *    graph  = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, sum);
    return { ctx, a, b, sum, graph };
}

static bool all_equal(const std::vector<float> & values, float expected) {
    for (float value : values) {
        if (value != expected) {
            return false;
        }
    }
    return true;
}

static bool backend_computes(ggml_backend_t backend) {
    add_graph             g      = make_add_graph();
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(g.ctx, backend);
    bool                  ok     = buffer != nullptr;
    if (ok) {
        const std::vector<float> a(elements, addend_a);
        const std::vector<float> b(elements, addend_b);
        std::vector<float>       sum(elements, 0.0f);
        ggml_backend_tensor_set(g.a, a.data(), 0, ggml_nbytes(g.a));
        ggml_backend_tensor_set(g.b, b.data(), 0, ggml_nbytes(g.b));
        ok = ggml_backend_graph_compute(backend, g.graph) == GGML_STATUS_SUCCESS;
        ggml_backend_tensor_get(g.sum, sum.data(), 0, ggml_nbytes(g.sum));
        ok = ok && all_equal(sum, addend_a + addend_b);
        ggml_backend_buffer_free(buffer);
    }
    ggml_free(g.ctx);
    return ok;
}

static int count_devices_that_do_not_compute(ggml_backend_reg_t reg) {
    int failures = 0;
    for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); i++) {
        ggml_backend_dev_t dev     = ggml_backend_reg_dev_get(reg, i);
        ggml_backend_t     backend = ggml_backend_dev_init(dev, nullptr);
        const bool         ok      = backend && backend_computes(backend);
        fprintf(stderr, "  %s: %s\n", ggml_backend_dev_name(dev), ok ? "computes" : "DOES NOT COMPUTE");
        if (!ok) {
            failures++;
        }
        ggml_backend_free(backend);
    }
    return failures;
}

static bool best_backend_is_cpu() {
    ggml_backend_t     best = ggml_backend_init_best();
    ggml_backend_dev_t dev  = best ? ggml_backend_get_device(best) : nullptr;
    const bool         cpu  = dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    fprintf(stderr, "  best backend: %s\n", dev ? ggml_backend_dev_name(dev) : "none");
    ggml_backend_free(best);
    return cpu;
}

static int check_no_device(ggml_backend_reg_t reg) {
    int failures = 0;
    if (ggml_backend_reg_dev_count(reg) != 0) {
        fprintf(stderr, "  a WebGPU device registered although no GPU driver is reachable\n");
        failures++;
    }
    if (!best_backend_is_cpu()) {
        fprintf(stderr, "  the CPU backend did not remain the best backend\n");
        failures++;
    }
    return failures;
}

int main(int argc, char ** argv) {
    const bool expect_no_device = argc > 1 && strcmp(argv[1], expect_no_device_flag) == 0;
    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name(webgpu_reg_name);
    if (!reg) {
        fprintf(stderr, "no WebGPU backend registered\n");
        return 1;
    }
    int failures = count_devices_that_do_not_compute(reg);
    if (expect_no_device) {
        failures += check_no_device(reg);
    }
    return failures == 0 ? 0 : 1;
}
