#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>

int main() {
    ggml_backend_load_all();

    ggml_backend_dev_t opencl = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "OpenCL") == 0) {
            opencl = dev;
            break;
        }
    }
    if (!opencl) {
        std::fprintf(stderr, "No OpenCL device available\n");
        return 77;
    }

    // Some model loaders allocate weights from a device buffer type before
    // creating the compute backend. This must initialize the device context.
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(opencl);
    ggml_context * ctx = ggml_init({ ggml_tensor_overhead() * 4, nullptr, true });
    if (!ctx || !ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 32)) {
        std::fprintf(stderr, "Failed to create the test tensor\n");
        if (ctx) ggml_free(ctx);
        return 1;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    const bool passed = buffer && ggml_backend_buft_get_alignment(buft) > 0 &&
                        ggml_backend_buft_get_max_size(buft) >= ggml_backend_buffer_get_size(buffer);
    if (buffer) ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    if (!passed) {
        std::fprintf(stderr, "OpenCL buffer-type allocation before backend initialization failed\n");
        return 1;
    }
    return 0;
}
