#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>

static const int64_t CHANNELS  = 32;
static const int64_t TIMESTEPS = 16;
static const float   EPS       = 1e-5f;
static const char *  METAL_REG = "MTL";

static bool check(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
    }
    return ok;
}

static ggml_backend_dev_t find_metal_device() {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name(METAL_REG);
    if (reg == nullptr || ggml_backend_reg_dev_count(reg) == 0) {
        return nullptr;
    }
    return ggml_backend_reg_dev_get(reg, 0);
}

static bool layer_norm_channel_rejected(ggml_backend_dev_t dev, ggml_context * ctx, ggml_tensor * x) {
    ggml_tensor * g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, CHANNELS);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, CHANNELS);
    ggml_tensor * op = ggml_supertonic_layer_norm_channel(ctx, x, g, b, EPS);
    return check(!ggml_backend_dev_supports_op(dev, op), "SUPERTONIC_LAYER_NORM_CHANNEL needs simdgroup reductions");
}

static bool bias_gelu_still_supported(ggml_backend_dev_t dev, ggml_context * ctx, ggml_tensor * x) {
    ggml_tensor * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, CHANNELS);
    ggml_tensor * op = ggml_supertonic_bias_gelu(ctx, x, bias);
    return check(ggml_backend_dev_supports_op(dev, op), "SUPERTONIC_BIAS_GELU does not need simdgroup reductions");
}

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = find_metal_device();
    if (!check(dev != nullptr, "a Metal device is available")) {
        return 1;
    }

    ggml_init_params params = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, TIMESTEPS, CHANNELS);

    const bool rejected  = layer_norm_channel_rejected(dev, ctx, x);
    const bool supported = bias_gelu_still_supported(dev, ctx, x);
    const bool ok = rejected && supported;
    ggml_free(ctx);
    return ok ? 0 : 1;
}
