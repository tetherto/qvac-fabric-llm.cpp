#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <initializer_list>

int main() {
    ggml_backend_load_all();
    auto * dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(dev);
    auto reg = ggml_backend_dev_backend_reg(dev);
    auto plan_graph = reinterpret_cast<decltype(&ggml_graph_plan)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_graph_plan"));
    GGML_ASSERT(plan_graph);
    auto * ctx = ggml_init({1024 * 1024, nullptr, true});
    GGML_ASSERT(ctx);
    auto * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 32);
    auto * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 16);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_mul_mat(ctx, a, b));
    for (int threads : {1, 4}) {
        const auto plan = plan_graph(graph, threads, nullptr);
        GGML_ASSERT(plan.work_size > 0);
        GGML_ASSERT(plan.work_data == nullptr);
        GGML_ASSERT(a->data == nullptr && b->data == nullptr);
#ifndef GGML_BACKEND_DL
        const auto direct = ggml_graph_plan(graph, threads, nullptr);
        GGML_ASSERT(plan.work_size == direct.work_size);
        GGML_ASSERT(plan.n_threads == direct.n_threads);
#endif
    }
    ggml_free(ctx);
    return 0;
}
