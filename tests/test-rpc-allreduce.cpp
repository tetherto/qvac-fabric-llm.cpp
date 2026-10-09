#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// Invoke the communicator directly: these tests must never pass through fallback.
static bool reduce_and_check(ggml_backend_comm_allreduce_tensor_t allreduce, void * comm,
                             const std::vector<ggml_backend_t> & backends, size_t count, bool random_input) {
    const size_t world = backends.size();
    std::vector<ggml_context_ptr> contexts;
    std::vector<ggml_backend_buffer_ptr> buffers;
    std::vector<ggml_tensor *> tensors;
    std::vector<double> expected(count, 0.0), magnitude(count, 0.0);
    std::vector<float> data(count);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (size_t rank = 0; rank < world; rank++) {
        ggml_init_params params = {ggml_tensor_overhead(), nullptr, true};
        contexts.emplace_back(ggml_init(params));
        if (!contexts.back()) {
            return false;
        }
        tensors.push_back(ggml_new_tensor_1d(contexts.back().get(), GGML_TYPE_F32, count));
        buffers.emplace_back(ggml_backend_alloc_ctx_tensors(contexts.back().get(), backends[rank]));
        if (!buffers.back()) {
            return false;
        }
        tensors.back()->flags |= GGML_TENSOR_FLAG_COMPUTE;
        for (size_t j = 0; j < count; j++) {
            data[j] = random_input ? distribution(rng) : float(rank + 1);
            expected[j] += data[j];
            magnitude[j] += std::fabs(data[j]);
        }
        ggml_backend_tensor_set(tensors.back(), data.data(), 0, count * sizeof(float));
    }

    // Rejection must happen before dispatch and must not advance the operation ID.
    if (allreduce(comm, nullptr)) {
        return false;
    }
    ggml_tensor * first = tensors[0];
    tensors[0] = nullptr;
    if (allreduce(comm, tensors.data())) {
        return false;
    }
    tensors[0] = first;
    tensors.back()->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
    if (allreduce(comm, tensors.data())) {
        return false;
    }
    tensors.back()->flags |= GGML_TENSOR_FLAG_COMPUTE;
    tensors.back()->type = GGML_TYPE_F16;
    if (allreduce(comm, tensors.data())) {
        return false;
    }
    tensors.back()->type = GGML_TYPE_F32;
    tensors.back()->ne[0]++;
    if (allreduce(comm, tensors.data())) {
        return false;
    }
    tensors.back()->ne[0]--;
    tensors.back()->nb[0] *= 2;
    if (allreduce(comm, tensors.data())) {
        return false;
    }
    tensors.back()->nb[0] /= 2;
    for (auto * tensor : tensors) {
        tensor->ne[0] = 0;
    }
    if (!allreduce(comm, tensors.data())) {
        return false;
    }
    for (auto * tensor : tensors) {
        tensor->ne[0] = count;
    }

    if (!allreduce(comm, tensors.data())) {
        fprintf(stderr, "direct all-reduce rejected %zu ranks / %zu elements\n", world, count);
        return false;
    }
    // Queue a second reduction without a client-side barrier to check ordering.
    const int repeats = random_input ? 1 : 2;
    if (repeats == 2 && !allreduce(comm, tensors.data())) {
        return false;
    }
    const bool bf16 = std::getenv("GGML_RPC_NO_WIRE_BF16") == nullptr && count >= 32768;
    const double tolerance = bf16 ? 0.004 * std::log2(double(world)) : 1e-6;
    std::vector<float> rank_zero;
    for (size_t rank = 0; rank < world; rank++) {
        ggml_backend_synchronize(backends[rank]);
        ggml_backend_tensor_get(tensors[rank], data.data(), 0, count * sizeof(float));
        for (size_t j = 0; j < count; j++) {
            const double target = expected[j] * (repeats == 2 ? world : 1);
            const double bound = random_input ? tolerance * magnitude[j] + 1e-6 : 0.0;
            if (!std::isfinite(data[j]) || std::fabs(data[j] - target) > bound ||
                    (rank > 0 && data[j] != rank_zero[j])) {
                fprintf(stderr, "rank %zu element %zu/%zu: got %.9g, expected %.9g (bound %.9g)\n",
                        rank, j, count, data[j], target, bound);
                return false;
            }
        }
        if (rank == 0) {
            rank_zero = data;
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s HOST:PORT HOST:PORT [HOST:PORT ...]\n", argv[0]);
        return 1;
    }
    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("RPC");
    if (!reg) {
        return 1;
    }
    auto add_server = (ggml_backend_reg_t (*)(const char *)) ggml_backend_reg_get_proc_address(
        reg, "ggml_backend_rpc_add_server");
    auto init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init");
    auto free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_free");
    auto allreduce = (ggml_backend_comm_allreduce_tensor_t) ggml_backend_reg_get_proc_address(
        reg, "ggml_backend_comm_allreduce_tensor");
    if (!add_server || !init || !free || !allreduce) {
        return 1;
    }
    std::vector<ggml_backend_ptr> owners;
    std::vector<ggml_backend_t> backends;
    for (int i = 1; i < argc; i++) {
        ggml_backend_reg_t server = add_server(argv[i]);
        if (!server || ggml_backend_reg_dev_count(server) == 0) {
            return 1;
        }
        owners.emplace_back(ggml_backend_dev_init(ggml_backend_reg_dev_get(server, 0), nullptr));
        if (!owners.back()) {
            return 1;
        }
        backends.push_back(owners.back().get());
    }
    const size_t world = backends.size();
    if (std::getenv("GGML_RPC_NO_COMM")) {
        if (init(backends.data(), world) != nullptr) {
            return 1;
        }
        printf("RPC communicator disabled as requested\n");
        return 0;
    }
    if (std::getenv("GGML_RPC_TEST_INIT_FAILURE")) {
        // The runner blocks rank 1's listener, which is first used in round 1.
        // Repeat on the same client connections to check partial-init cleanup.
        for (int attempt = 0; attempt < 2; attempt++) {
            if (init(backends.data(), world) != nullptr) {
                fprintf(stderr, "expected communicator initialization failure\n");
                return 1;
            }
        }
        backends.resize(2);
        void * comm = init(backends.data(), backends.size());
        if (!comm || !reduce_and_check(allreduce, comm, backends, 7, false)) {
            fprintf(stderr, "communicator did not recover after partial initialization\n");
            return 1;
        }
        free(comm);
        printf("RPC partial initialization cleanup passed\n");
        return 0;
    }
    if (init(backends.data(), 0) || init(backends.data(), 1)) {
        return 1;
    }
    if (world >= 4 && init(backends.data(), 3)) {
        return 1;
    }
    ggml_backend_t duplicate[2] = {backends[0], backends[0]};
    if (init(duplicate, 2)) {
        return 1;
    }

    void * comm = init(backends.data(), world);
    void * reused = init(backends.data(), world);
    if (!comm || !reused) {
        fprintf(stderr, "failed to initialize/reuse %zu-rank communicator\n", world);
        return 1;
    }
    for (size_t count : {size_t(1), size_t(7), size_t(16383), size_t(16384), size_t(32767),
                         size_t(32768), size_t(32769), size_t(262145)}) {
        if (!reduce_and_check(allreduce, comm, backends, count, false) ||
                !reduce_and_check(allreduce, reused, backends, count, true)) {
            return 1;
        }
    }
    free(comm);
    if (!reduce_and_check(allreduce, reused, backends, 32769, true)) {
        return 1;
    }
    free(reused);
    comm = init(backends.data(), world);
    if (!comm || !reduce_and_check(allreduce, comm, backends, 7, false)) {
        return 1;
    }
    free(comm);
    // Cached device and buffer-type pointers outlive individual backends and
    // communicators. Reconnect using those borrowed pointers after releasing
    // every backend, then let LeakSanitizer check cache destruction at exit.
    std::vector<ggml_backend_dev_t> devices;
    std::vector<ggml_backend_buffer_type_t> buffer_types;
    for (auto backend : backends) {
        ggml_backend_synchronize(backend);
        devices.push_back(ggml_backend_get_device(backend));
        buffer_types.push_back(ggml_backend_get_default_buffer_type(backend));
    }
    backends.clear();
    owners.clear();
    for (size_t rank = 0; rank < world; rank++) {
        ggml_backend_reg_t server = add_server(argv[rank + 1]);
        if (!server || ggml_backend_reg_dev_get(server, 0) != devices[rank] ||
                ggml_backend_dev_buffer_type(devices[rank]) != buffer_types[rank]) {
            fprintf(stderr, "cached RPC metadata changed after releasing backends\n");
            return 1;
        }
        owners.emplace_back(ggml_backend_dev_init(devices[rank], nullptr));
        if (!owners.back()) {
            return 1;
        }
        backends.push_back(owners.back().get());
    }
    comm = init(backends.data(), world);
    if (!comm || !reduce_and_check(allreduce, comm, backends, 7, false)) {
        return 1;
    }
    free(comm);
    for (auto backend : backends) {
        ggml_backend_synchronize(backend);
    }
    printf("RPC butterfly all-reduce passed (%zu ranks)\n", world);
    return 0;
}
