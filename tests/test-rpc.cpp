#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "../ggml/src/ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static bool check_values(const std::vector<float> & values, float expected) {
    for (float value : values) {
        if (std::fabs(value - expected) > 1e-5f) {
            fprintf(stderr, "unexpected value: %.6f, expected %.6f\n", value, expected);
            return false;
        }
    }
    return true;
}

static ggml_backend_ptr init_backend(const std::string & endpoint, ggml_backend_dev_t & device) {
    ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
    if (rpc_reg == nullptr) {
        return ggml_backend_ptr(nullptr);
    }
    using add_rpc_server_fn = ggml_backend_reg_t (*)(const char *);
    auto add_server = (add_rpc_server_fn) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_add_server");
    if (add_server == nullptr) {
        return ggml_backend_ptr(nullptr);
    }
    ggml_backend_reg_t reg = add_server(endpoint.c_str());
    if (reg == nullptr || ggml_backend_reg_dev_count(reg) == 0) {
        return ggml_backend_ptr(nullptr);
    }
    device = ggml_backend_reg_dev_get(reg, 0);
    return ggml_backend_ptr(ggml_backend_dev_init(device, nullptr));
}

int main() {
    const char * endpoints_env = std::getenv("GGML_RPC_TEST_ENDPOINTS");
    if (endpoints_env == nullptr) {
        printf("GGML_RPC_TEST_ENDPOINTS is not set, skipping RPC integration test\n");
        return 77;
    }

    std::string endpoints(endpoints_env);
    size_t      separator = endpoints.find(',');
    if (separator == std::string::npos) {
        fprintf(stderr, "GGML_RPC_TEST_ENDPOINTS must contain two comma-separated endpoints\n");
        return 1;
    }
    std::string endpoint_a = endpoints.substr(0, separator);
    std::string endpoint_b = endpoints.substr(separator + 1);

    ggml_backend_load_all();

    ggml_backend_dev_t device_a  = nullptr;
    ggml_backend_dev_t device_b  = nullptr;
    ggml_backend_ptr   backend_a = init_backend(endpoint_a, device_a);
    ggml_backend_ptr   backend_b = init_backend(endpoint_b, device_b);
    if (backend_a == nullptr || backend_b == nullptr) {
        fprintf(stderr, "failed to initialize RPC test backends\n");
        return 1;
    }

    ggml_init_params params = {
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx_a{ ggml_init(params) };
    ggml_context_ptr ctx_b{ ggml_init(params) };
    if (ctx_a == nullptr || ctx_b == nullptr) {
        return 1;
    }

    ggml_tensor * tensor_a = ggml_new_tensor_2d(ctx_a.get(), GGML_TYPE_F32, 8, 4);
    ggml_tensor * result_a = ggml_scale(ctx_a.get(), tensor_a, 2.0f);
    ggml_cgraph * graph    = ggml_new_graph_custom(ctx_a.get(), 16, false);
    ggml_build_forward_expand(graph, result_a);
    graph->uid = ggml_graph_next_uid();
    ggml_cgraph * foreign_graph = ggml_new_graph_custom(ctx_a.get(), 1, false);
    ggml_graph_add_node(foreign_graph, tensor_a);

    ggml_tensor *           tensor_b = ggml_new_tensor_2d(ctx_b.get(), GGML_TYPE_F32, 8, 4);
    ggml_backend_buffer_ptr buffer_a(ggml_backend_alloc_ctx_tensors(ctx_a.get(), backend_a.get()));
    ggml_backend_buffer_ptr buffer_b(ggml_backend_alloc_ctx_tensors(ctx_b.get(), backend_b.get()));
    if (buffer_a == nullptr || buffer_b == nullptr) {
        fprintf(stderr, "failed to allocate RPC test buffers\n");
        return 1;
    }

    if (ggml_backend_graph_compute(backend_b.get(), foreign_graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "cross-server graph serialization failed\n");
        return 1;
    }

    const size_t         row_size      = tensor_a->nb[1];
    const size_t         rows          = tensor_a->ne[1];
    const size_t         input_stride  = row_size + 2 * sizeof(float);
    const size_t         output_stride = row_size + 3 * sizeof(float);
    std::vector<uint8_t> input_2d(input_stride * rows, 0);
    std::vector<uint8_t> output_2d(output_stride * rows, 0);
    for (size_t row = 0; row < rows; row++) {
        for (size_t col = 0; col < 8; col++) {
            float value = (float) (row * 8 + col);
            memcpy(input_2d.data() + row * input_stride + col * sizeof(value), &value, sizeof(value));
        }
    }
    ggml_backend_tensor_set_2d_async(backend_a.get(), tensor_a, input_2d.data(), 0, row_size, rows, tensor_a->nb[1],
                                     input_stride);
    ggml_backend_tensor_get_2d_async(backend_a.get(), tensor_a, output_2d.data(), 0, row_size, rows, tensor_a->nb[1],
                                     output_stride);
    ggml_backend_synchronize(backend_a.get());
    for (size_t row = 0; row < rows; row++) {
        if (memcmp(input_2d.data() + row * input_stride, output_2d.data() + row * output_stride, row_size) != 0) {
            fprintf(stderr, "2D RPC transfer mismatch\n");
            return 1;
        }
    }

    ggml_backend_event_t events[4] = {
        ggml_backend_event_new(device_a),
        ggml_backend_event_new(device_a),
        ggml_backend_event_new(device_a),
        ggml_backend_event_new(device_a),
    };
    for (ggml_backend_event_t event : events) {
        if (event == nullptr) {
            fprintf(stderr, "failed to create RPC event\n");
            return 1;
        }
    }

    std::vector<float> input(ggml_nelements(tensor_a));
    for (int iteration = 0; iteration < 8; iteration++) {
        if (iteration == 4) {
            ggml_backend_buffer_ptr unrelated(ggml_backend_alloc_buffer(backend_a.get(), 4096));
            if (unrelated == nullptr) {
                fprintf(stderr, "failed to allocate unrelated RPC buffer\n");
                return 1;
            }
        }
        ggml_backend_event_t event = events[iteration % 4];
        ggml_backend_event_synchronize(event);
        std::fill(input.begin(), input.end(), (float) iteration);
        ggml_backend_tensor_set_async(backend_a.get(), tensor_a, input.data(), 0, ggml_nbytes(tensor_a));
        if (ggml_backend_graph_compute_async(backend_a.get(), graph) != GGML_STATUS_SUCCESS) {
            return 1;
        }
        ggml_backend_event_record(event, backend_a.get());
    }
    ggml_backend_synchronize(backend_a.get());

    std::vector<float> result(ggml_nelements(result_a));
    ggml_backend_tensor_get(result_a, result.data(), 0, ggml_nbytes(result_a));
    if (!check_values(result, 14.0f)) {
        return 1;
    }

    std::fill(input.begin(), input.end(), 9.0f);
    ggml_backend_tensor_set_async(backend_a.get(), tensor_a, input.data(), 0, ggml_nbytes(tensor_a));
    if (ggml_backend_graph_compute_async(backend_a.get(), graph) != GGML_STATUS_SUCCESS) {
        return 1;
    }
    ggml_backend_tensor_copy_async(backend_a.get(), backend_b.get(), result_a, tensor_b);
    ggml_backend_synchronize(backend_b.get());
    std::vector<float> copied(ggml_nelements(tensor_b));
    ggml_backend_tensor_get(tensor_b, copied.data(), 0, ggml_nbytes(tensor_b));
    if (!check_values(copied, 18.0f)) {
        return 1;
    }

    ggml_backend_tensor_memset(tensor_a, 0, 0, ggml_nbytes(tensor_a));
    std::vector<float> cleared(ggml_nelements(tensor_a), 1.0f);
    ggml_backend_tensor_get(tensor_a, cleared.data(), 0, ggml_nbytes(tensor_a));
    if (!check_values(cleared, 0.0f)) {
        return 1;
    }

    ggml_backend_reg_t rpc_reg = ggml_backend_dev_backend_reg(device_a);
    auto comm_init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_comm_init");
    auto comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_comm_free");
    auto comm_allreduce = (ggml_backend_comm_allreduce_tensor_t) ggml_backend_reg_get_proc_address(
        rpc_reg, "ggml_backend_comm_allreduce_tensor");
    if (comm_init == nullptr || comm_free == nullptr || comm_allreduce == nullptr) {
        fprintf(stderr, "RPC communicator functions are unavailable\n");
        return 1;
    }

    std::vector<float> ones(ggml_nelements(tensor_a), 1.0f);
    std::vector<float> twos(ggml_nelements(tensor_b), 2.0f);
    ggml_backend_tensor_set(tensor_a, ones.data(), 0, ggml_nbytes(tensor_a));
    ggml_backend_tensor_set(tensor_b, twos.data(), 0, ggml_nbytes(tensor_b));
    tensor_a->flags |= GGML_TENSOR_FLAG_COMPUTE;
    tensor_b->flags |= GGML_TENSOR_FLAG_COMPUTE;
    ggml_backend_t comm_backends[2] = { backend_a.get(), backend_b.get() };
    void *         comm             = comm_init(comm_backends, 2);
    if (comm == nullptr) {
        fprintf(stderr, "failed to initialize RPC communicator\n");
        return 1;
    }
    ggml_tensor * comm_tensors[2] = { tensor_a, tensor_b };
    if (!comm_allreduce(comm, comm_tensors)) {
        return 1;
    }
    ggml_backend_synchronize(backend_a.get());
    ggml_backend_synchronize(backend_b.get());
    ggml_backend_tensor_get(tensor_a, ones.data(), 0, ggml_nbytes(tensor_a));
    ggml_backend_tensor_get(tensor_b, twos.data(), 0, ggml_nbytes(tensor_b));
    if (!check_values(ones, 3.0f) || !check_values(twos, 3.0f)) {
        return 1;
    }

    {
        ggml_context_ptr        ctx_large_a{ ggml_init(params) };
        ggml_context_ptr        ctx_large_b{ ggml_init(params) };
        ggml_tensor *           large_a = ggml_new_tensor_1d(ctx_large_a.get(), GGML_TYPE_F32, 32768);
        ggml_tensor *           large_b = ggml_new_tensor_1d(ctx_large_b.get(), GGML_TYPE_F32, 32768);
        ggml_backend_buffer_ptr buffer_large_a(ggml_backend_alloc_ctx_tensors(ctx_large_a.get(), backend_a.get()));
        ggml_backend_buffer_ptr buffer_large_b(ggml_backend_alloc_ctx_tensors(ctx_large_b.get(), backend_b.get()));
        if (buffer_large_a == nullptr || buffer_large_b == nullptr) {
            return 1;
        }
        std::vector<float> large_ones(32768, 1.0f);
        std::vector<float> large_twos(32768, 2.0f);
        ggml_backend_tensor_set(large_a, large_ones.data(), 0, ggml_nbytes(large_a));
        ggml_backend_tensor_set(large_b, large_twos.data(), 0, ggml_nbytes(large_b));
        large_a->flags |= GGML_TENSOR_FLAG_COMPUTE;
        large_b->flags |= GGML_TENSOR_FLAG_COMPUTE;
        ggml_tensor * large_tensors[2] = { large_a, large_b };
        if (!comm_allreduce(comm, large_tensors)) {
            return 1;
        }
        ggml_backend_synchronize(backend_a.get());
        ggml_backend_synchronize(backend_b.get());
        ggml_backend_tensor_get(large_a, large_ones.data(), 0, ggml_nbytes(large_a));
        ggml_backend_tensor_get(large_b, large_twos.data(), 0, ggml_nbytes(large_b));
        if (!check_values(large_ones, 3.0f) || !check_values(large_twos, 3.0f)) {
            return 1;
        }
    }

    comm_free(comm);
    ggml_backend_synchronize(backend_a.get());
    ggml_backend_synchronize(backend_b.get());

    for (ggml_backend_event_t event : events) {
        ggml_backend_event_free(event);
    }

    printf("RPC integration test passed\n");
    return 0;
}
