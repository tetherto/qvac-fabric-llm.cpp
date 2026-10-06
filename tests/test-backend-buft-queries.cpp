#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>

static const int64_t probe_elements = 64;

struct probe_tensors {
    ggml_context * ctx;
    ggml_tensor * dense;
    ggml_tensor * quantized;
};

static probe_tensors make_probe_tensors() {
    ggml_init_params params = { ggml_tensor_overhead() * 2, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    return {
        ctx,
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, probe_elements),
        ggml_new_tensor_1d(ctx, GGML_TYPE_Q4_0, probe_elements),
    };
}

static bool is_power_of_two(size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

static bool check(bool ok, const char * device, const char * what) {
    if (!ok) {
        fprintf(stderr, "  %s: %s\n", device, what);
    }
    return ok;
}

static bool probe_cold_queries(ggml_backend_dev_t dev, const probe_tensors & probes, size_t & alignment_out) {
    const char * name = ggml_backend_dev_name(dev);
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    if (!check(buft != nullptr, name, "device has no buffer type")) {
        return false;
    }
    alignment_out = ggml_backend_buft_get_alignment(buft);
    bool ok = check(is_power_of_two(alignment_out), name, "alignment is not a power of two");
    ok = check(ggml_backend_buft_get_max_size(buft) > 0, name, "max size is zero") && ok;
    ok = check(ggml_backend_buft_get_alloc_size(buft, probes.dense) >= ggml_nbytes(probes.dense),
               name, "dense alloc size is below nbytes") && ok;
    ok = check(ggml_backend_buft_get_alloc_size(buft, probes.quantized) >= ggml_nbytes(probes.quantized),
               name, "quantized alloc size is below nbytes") && ok;
    return ok;
}

static bool probe_warm_alignment_matches(ggml_backend_dev_t dev, size_t cold_alignment) {
    const char * name = ggml_backend_dev_name(dev);
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!check(backend != nullptr, name, "backend init failed")) {
        return false;
    }
    size_t warm_alignment = ggml_backend_buft_get_alignment(ggml_backend_dev_buffer_type(dev));
    ggml_backend_free(backend);
    return check(warm_alignment == cold_alignment, name, "alignment differs before and after backend init");
}

static bool probe_device(size_t index, const probe_tensors & probes) {
    ggml_backend_dev_t dev = ggml_backend_dev_get(index);
    size_t cold_alignment = 0;
    bool ok = probe_cold_queries(dev, probes, cold_alignment);
    ok = probe_warm_alignment_matches(dev, cold_alignment) && ok;
    printf("  %-40s %s\n", ggml_backend_dev_name(dev), ok ? "OK" : "FAIL");
    return ok;
}

static bool probe_all_devices(const probe_tensors & probes) {
    bool ok = true;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ok = probe_device(i, probes) && ok;
    }
    return ok;
}

int main() {
    ggml_backend_load_all();
    probe_tensors probes = make_probe_tensors();
    printf("buffer type queries before backend init on %zu device(s)\n", ggml_backend_dev_count());
    bool ok = probe_all_devices(probes);
    ggml_free(probes.ctx);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
