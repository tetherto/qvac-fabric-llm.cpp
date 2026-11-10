#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>

namespace {

constexpr int64_t kDHead       = 8;
constexpr int64_t kNHead       = 3;
constexpr int64_t kLength      = 5;
constexpr int64_t kBatch       = 2;
constexpr int     kThreads     = 2;
constexpr float   kAngleStep   = 0.1f;
constexpr double  kTolerance   = 1e-4;
constexpr size_t  kContextSize = 4 * 1024 * 1024;

float input_value(int64_t d, int64_t h, int64_t l, int64_t n) {
    return (float) (((n*kLength + l)*kNHead + h)*kDHead + d);
}

float angle(int64_t l, int64_t pair) {
    return kAngleStep * (float) ((l + 1)*(pair + 1));
}

float rotation_entry(int64_t in, int64_t out, float theta) {
    if (in == out) {
        return cosf(theta);
    }
    return in == 0 ? sinf(theta) : -sinf(theta);
}

void fill_input(ggml_tensor * x) {
    int64_t d, h, l, n;
    for (int64_t i = 0; i < ggml_nelements(x); ++i) {
        ggml_unravel_index(x, i, &d, &h, &l, &n);
        ggml_set_f32_nd(x, d, h, l, n, input_value(d, h, l, n));
    }
}

void fill_pe(ggml_tensor * pe) {
    int64_t in, out, pair, l;
    for (int64_t i = 0; i < ggml_nelements(pe); ++i) {
        ggml_unravel_index(pe, i, &in, &out, &pair, &l);
        ggml_set_f32_nd(pe, in, out, pair, l, rotation_entry(in, out, angle(l, pair)));
    }
}

float expected_value(int64_t d, int64_t l, int64_t bh, bool with_pe) {
    const int64_t h = bh % kNHead;
    const int64_t n = bh / kNHead;
    if (!with_pe) {
        return input_value(d, h, l, n);
    }
    const int64_t pair  = d / 2;
    const float   theta = angle(l, pair);
    const float   x0    = input_value(2*pair, h, l, n);
    const float   x1    = input_value(2*pair + 1, h, l, n);
    return d % 2 == 0 ? cosf(theta)*x0 - sinf(theta)*x1 : sinf(theta)*x0 + cosf(theta)*x1;
}

double max_error(const ggml_tensor * out, bool with_pe) {
    double err = 0.0;
    int64_t d, l, bh, unused;
    for (int64_t i = 0; i < ggml_nelements(out); ++i) {
        ggml_unravel_index(out, i, &d, &l, &bh, &unused);
        err = fmax(err, fabs(ggml_get_f32_nd(out, d, l, bh, 0) - expected_value(d, l, bh, with_pe)));
    }
    return err;
}

ggml_tensor * new_input(ggml_context * ctx, bool permuted) {
    if (!permuted) {
        return ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kDHead, kNHead, kLength, kBatch);
    }
    ggml_tensor * storage = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kDHead, kLength, kNHead, kBatch);
    return ggml_permute(ctx, storage, 0, 2, 1, 3);
}

bool shape_matches(const ggml_tensor * out) {
    return out->ne[0] == kDHead && out->ne[1] == kLength && out->ne[2] == kBatch*kNHead && out->ne[3] == 1 &&
           ggml_is_contiguous(out);
}

bool run_case(const char * name, bool with_pe, bool permuted) {
    ggml_init_params params = { kContextSize, nullptr, false };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * x = new_input(ctx, permuted);
    fill_input(x);

    ggml_tensor * pe = nullptr;
    if (with_pe) {
        pe = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, kDHead/2, kLength);
        fill_pe(pe);
    }

    ggml_tensor * out = ggml_rope_flux(ctx, x, pe);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, kThreads);

    const bool   shape_ok = shape_matches(out);
    const double err      = max_error(out, with_pe);
    ggml_free(ctx);

    const bool ok = shape_ok && err <= kTolerance;
    printf("%s: %s (max error %g)\n", name, ok ? "ok" : "FAILED", err);
    return ok;
}

} // namespace

int main() {
    bool ok = true;
    ok &= run_case("rotate", true, false);
    ok &= run_case("permute only", false, false);
    ok &= run_case("rotate strided input", true, true);
    ok &= run_case("permute only strided input", false, true);
    return ok ? 0 : 1;
}
