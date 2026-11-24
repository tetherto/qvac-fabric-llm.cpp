// Model-free on every host: the weight-width floor that keeps the Core ML
// vocoder sidecar off tiers it would substitute rather than accelerate, and
// the lookup that applies it to the bound vocoder weights
// (supertonic_internal.h).

#include "supertonic_internal.h"

#include <cstdio>
#include <string>
#include <unordered_set>
#include <vector>

using tts_cpp::supertonic::detail::kSupertonicCoremlVocoderMinWeightBits;
using tts_cpp::supertonic::detail::supertonic_first_low_bit_vocoder_weight;
using tts_cpp::supertonic::detail::supertonic_tensor_weight_bits;
using tts_cpp::supertonic::detail::supertonic_vocoder_weights;

namespace {

int g_failures = 0;

void fail(const std::string & what) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
}

void expect(bool ok, const std::string & what) {
    if (!ok) fail(what);
}

struct scoped_ctx {
    ggml_context * ctx = nullptr;
    scoped_ctx() {
        ggml_init_params params = {};
        params.mem_size = 16u * 1024u * 1024u;
        params.no_alloc = true;
        ctx = ggml_init(params);
    }
    ~scoped_ctx() { ggml_free(ctx); }
};

ggml_tensor * weight(ggml_context * ctx, ggml_type type, const char * name) {
    ggml_tensor * t = ggml_new_tensor_2d(ctx, type, 512, 512);
    ggml_set_name(t, name);
    return t;
}

void expect_bits(ggml_context * ctx, ggml_type type, int want) {
    const int got = supertonic_tensor_weight_bits(weight(ctx, type, "probe"));
    if (got != want) {
        std::fprintf(stderr, "FAIL: %s -> %d bits (want %d)\n", ggml_type_name(type), got, want);
        ++g_failures;
    }
}

void expect_meets_floor(ggml_context * ctx, ggml_type type, bool want) {
    const bool got =
        supertonic_tensor_weight_bits(weight(ctx, type, "probe")) >=
        kSupertonicCoremlVocoderMinWeightBits;
    if (got != want) {
        std::fprintf(stderr, "FAIL: %s meets floor=%d (want %d)\n", ggml_type_name(type),
                     (int) got, (int) want);
        ++g_failures;
    }
}

// Every weight the sidecar stands in for, each under its own name.
supertonic_vocoder_weights bound_vocoder(ggml_context * ctx) {
    supertonic_vocoder_weights v;
    v.normalizer_scale = weight(ctx, GGML_TYPE_F32, "normalizer_scale");
    v.latent_mean = weight(ctx, GGML_TYPE_F32, "latent_mean");
    v.latent_std = weight(ctx, GGML_TYPE_F32, "latent_std");
    v.embed_w = weight(ctx, GGML_TYPE_F32, "embed_w");
    v.embed_b = weight(ctx, GGML_TYPE_F32, "embed_b");
    v.final_norm_g = weight(ctx, GGML_TYPE_F32, "final_norm_g");
    v.final_norm_b = weight(ctx, GGML_TYPE_F32, "final_norm_b");
    v.final_norm_running_mean = weight(ctx, GGML_TYPE_F32, "final_norm_running_mean");
    v.final_norm_running_var = weight(ctx, GGML_TYPE_F32, "final_norm_running_var");
    v.head1_w = weight(ctx, GGML_TYPE_F32, "head1_w");
    v.head1_b = weight(ctx, GGML_TYPE_F32, "head1_b");
    v.head_prelu = weight(ctx, GGML_TYPE_F32, "head_prelu");
    v.head2_w = weight(ctx, GGML_TYPE_F32, "head2_w");
    for (size_t i = 0; i < v.convnext.size(); ++i) {
        const std::string p = "convnext" + std::to_string(i) + "_";
        auto & c = v.convnext[i];
        c.dw_w = weight(ctx, GGML_TYPE_F32, (p + "dw_w").c_str());
        c.dw_b = weight(ctx, GGML_TYPE_F32, (p + "dw_b").c_str());
        c.norm_g = weight(ctx, GGML_TYPE_F32, (p + "norm_g").c_str());
        c.norm_b = weight(ctx, GGML_TYPE_F32, (p + "norm_b").c_str());
        c.pw1_w = weight(ctx, GGML_TYPE_F32, (p + "pw1_w").c_str());
        c.pw1_b = weight(ctx, GGML_TYPE_F32, (p + "pw1_b").c_str());
        c.pw2_w = weight(ctx, GGML_TYPE_F32, (p + "pw2_w").c_str());
        c.pw2_b = weight(ctx, GGML_TYPE_F32, (p + "pw2_b").c_str());
        c.gamma = weight(ctx, GGML_TYPE_F32, (p + "gamma").c_str());
    }
    return v;
}

void expect_reports(const supertonic_vocoder_weights & v, const std::string & name) {
    const ggml_tensor * got = supertonic_first_low_bit_vocoder_weight(v, {name});
    if (!got) {
        fail(name + " is not reported as low-bit");
        return;
    }
    if (std::string(ggml_get_name(got)) != name) {
        fail(name + " reported as " + ggml_get_name(got));
    }
}

}  // namespace

int main() {
    scoped_ctx ctx;
    if (!ctx.ctx) {
        std::fprintf(stderr, "FAIL: ggml_init\n");
        return 1;
    }

    expect(kSupertonicCoremlVocoderMinWeightBits == 8, "the floor is 8 bits per weight");

    expect_bits(ctx.ctx, GGML_TYPE_F32, 32);
    expect_bits(ctx.ctx, GGML_TYPE_F16, 16);
    expect_bits(ctx.ctx, GGML_TYPE_BF16, 16);
    expect_bits(ctx.ctx, GGML_TYPE_Q8_0, 8);
    expect_bits(ctx.ctx, GGML_TYPE_Q5_0, 5);
    expect_bits(ctx.ctx, GGML_TYPE_Q4_0, 4);

    // Tiers the sidecar may stand in for.
    expect_meets_floor(ctx.ctx, GGML_TYPE_F32, true);
    expect_meets_floor(ctx.ctx, GGML_TYPE_F16, true);
    expect_meets_floor(ctx.ctx, GGML_TYPE_BF16, true);
    expect_meets_floor(ctx.ctx, GGML_TYPE_Q8_0, true);
    // Tiers it would replace rather than accelerate.
    expect_meets_floor(ctx.ctx, GGML_TYPE_Q5_0, false);
    expect_meets_floor(ctx.ctx, GGML_TYPE_Q4_0, false);
    expect_meets_floor(ctx.ctx, GGML_TYPE_Q4_K, false);

    const supertonic_vocoder_weights v = bound_vocoder(ctx.ctx);

    expect(supertonic_first_low_bit_vocoder_weight(v, {}) == nullptr,
           "an empty low-bit set leaves every weight above the floor");
    expect(supertonic_first_low_bit_vocoder_weight(v, {"vector_estimator_w"}) == nullptr,
           "a low-bit weight outside the vocoder does not gate the sidecar");

    // One low-bit weight anywhere in the stack is enough, wherever it sits.
    expect_reports(v, "embed_w");
    expect_reports(v, "convnext0_dw_w");
    expect_reports(v, "convnext9_gamma");
    expect_reports(v, "head2_w");
    expect_reports(v, "latent_std");

    expect(supertonic_first_low_bit_vocoder_weight(supertonic_vocoder_weights{}, {"embed_w"}) ==
               nullptr,
           "unbound weights do not trip the floor");

    if (g_failures == 0) {
        std::printf("test_supertonic_coreml_tier: OK\n");
        return 0;
    }
    std::fprintf(stderr, "test_supertonic_coreml_tier: %d failure(s)\n", g_failures);
    return 1;
}
