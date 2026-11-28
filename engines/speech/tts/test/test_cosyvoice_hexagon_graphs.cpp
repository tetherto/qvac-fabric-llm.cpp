#include "backend_selection.h"
#include "backend_util.h"
#include "cosyvoice_pipeline.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace {

constexpr int   kFft         = 16;
constexpr int   kSpecCh      = 18;
constexpr int   kSpecFrames  = 257;
constexpr int   kHop         = 4;
constexpr int   kDim         = 256;
constexpr int   kDimHead     = 64;
constexpr int   kTokens      = 37;
constexpr int   kBatch       = 2;
constexpr int   kFusedCopies = 3;
constexpr float kCpuTolerance    = 1e-5f;
constexpr float kDeviceTolerance = 2e-3f;
constexpr size_t kArenaBytes = 64 * 1024 * 1024;

struct graph_input {
    std::string        name;
    std::vector<float> f32;
    std::vector<int>   i32;
};

using graph_builder = std::function<ggml_tensor *(ggml_context *)>;

std::vector<float> make_values(size_t n, float phase, float scale) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = scale * std::sin(phase * (float) i + 0.3f) + 0.01f * (float) (i % 7);
    return v;
}

void set_input(ggml_cgraph * gf, const graph_input & in) {
    ggml_tensor * t = ggml_graph_get_tensor(gf, in.name.c_str());
    if (!in.i32.empty()) {
        ggml_backend_tensor_set(t, in.i32.data(), 0, in.i32.size() * sizeof(int));
    } else {
        ggml_backend_tensor_set(t, in.f32.data(), 0, in.f32.size() * sizeof(float));
    }
}

std::vector<float> run(ggml_backend_t backend, const graph_builder & build, const std::vector<graph_input> & inputs) {
    ggml_init_params p = { kArenaBytes, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph *  gf  = ggml_new_graph(ctx);
    ggml_tensor *  out = build(ctx);
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, gf);
    for (const graph_input & in : inputs) set_input(gf, in);
    ggml_backend_graph_compute(backend, gf);
    std::vector<float> res(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    return res;
}

bool matches(const std::vector<float> & got, const std::vector<float> & ref, float tol, const char * what) {
    if (got.size() != ref.size()) {
        fprintf(stderr, "FAIL: %s size %zu vs %zu\n", what, got.size(), ref.size());
        return false;
    }
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!(std::fabs(got[i] - ref[i]) <= tol + tol * std::fabs(ref[i]))) {
            fprintf(stderr, "FAIL: %s mismatch @ %zu: got=%.7g ref=%.7g\n", what, i, got[i], ref[i]);
            return false;
        }
    }
    return true;
}

ggml_tensor * named_input(ggml_context * ctx, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, const char * name) {
    ggml_tensor * t = ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
    ggml_set_name(t, name);
    ggml_set_input(t);
    return t;
}

ggml_tensor * build_istft_transposed_conv(ggml_context * ctx) {
    ggml_tensor * k    = named_input(ctx, GGML_TYPE_F32, kFft, 1, kSpecCh, "k");
    ggml_tensor * spec = ggml_reshape_2d(ctx, named_input(ctx, GGML_TYPE_F32, kSpecFrames, kSpecCh, 1, "spec"),
                                         kSpecFrames, kSpecCh);
    return ggml_conv_transpose_1d(ctx, k, spec, kHop, 0, 1);
}

ggml_tensor * build_istft_columns(ggml_context * ctx) {
    ggml_tensor * cols = ggml_reshape_2d(ctx, named_input(ctx, GGML_TYPE_F32, kSpecCh, kFft, 1, "k"), kSpecCh, kFft);
    ggml_tensor * spec = ggml_reshape_2d(ctx, named_input(ctx, GGML_TYPE_F32, kSpecFrames, kSpecCh, 1, "spec"),
                                         kSpecFrames, kSpecCh);
    return cosyvoice_istft_columns(ctx, cols, spec, kHop);
}

std::vector<graph_input> istft_inputs(bool columns) {
    const std::vector<float> kernel = make_values((size_t) kFft * kSpecCh, 0.17f, 0.5f);
    return {
        { "k", columns ? cosyvoice_istft_kernel_columns(kernel, kFft, kSpecCh) : kernel, {} },
        { "spec", make_values((size_t) kSpecFrames * kSpecCh, 0.07f, 2.0f), {} },
    };
}

ggml_tensor * build_rope(ggml_context * ctx, bool rows_in_place) {
    ggml_tensor * fused = named_input(ctx, GGML_TYPE_F32, (int64_t) kFusedCopies * kDim, kTokens, kBatch, "fused");
    ggml_tensor * pos   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kTokens);
    ggml_set_name(pos, "pos");
    ggml_set_input(pos);
    ggml_tensor * z = ggml_view_3d(ctx, fused, kDim, kTokens, kBatch, fused->nb[1], fused->nb[2], 0);
    return cosyvoice_dit_rope_first_head(ctx, z, pos, kDimHead, kTokens, kBatch, rows_in_place);
}

std::vector<graph_input> rope_inputs() {
    std::vector<int> pos(kTokens);
    for (int i = 0; i < kTokens; ++i) pos[i] = i;
    return {
        { "fused", make_values((size_t) kFusedCopies * kDim * kTokens * kBatch, 0.013f, 3.0f), {} },
        { "pos", {}, pos },
    };
}

bool check_weight_types() {
    const ggml_type runs[]    = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_I32 };
    const ggml_type refused[] = { GGML_TYPE_BF16, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K };
    for (ggml_type t : runs) {
        if (!cosyvoice_hexagon_runs_weight_type(t)) {
            fprintf(stderr, "FAIL: Hexagon should run %s weights\n", ggml_type_name(t));
            return false;
        }
    }
    for (ggml_type t : refused) {
        if (cosyvoice_hexagon_runs_weight_type(t)) {
            fprintf(stderr, "FAIL: Hexagon should refuse %s weights\n", ggml_type_name(t));
            return false;
        }
    }
    return true;
}

bool check_frontend_backend(ggml_backend_t backend) {
    ggml_backend_t expected = ::tts_cpp::detail::backend_is_hexagon(backend) ? nullptr : backend;
    if (cosyvoice_frontend_backend(backend) != expected) {
        fprintf(stderr, "FAIL: cloning frontend backend on %s\n", ggml_backend_name(backend));
        return false;
    }
    return true;
}

bool check_device(ggml_backend_t device, const std::vector<float> & istft_ref, const std::vector<float> & rope_ref) {
    const auto istft = run(device, build_istft_columns, istft_inputs(true));
    const auto rope  = run(device, [](ggml_context * c) { return build_rope(c, true); }, rope_inputs());
    return check_frontend_backend(device) && matches(istft, istft_ref, kDeviceTolerance, "device iSTFT columns") &&
           matches(rope, rope_ref, kDeviceTolerance, "device in-place rope");
}

} // namespace

int main() {
    ggml_backend_t cpu = ::tts_cpp::detail::init_cpu_backend();
    if (!cpu) { fprintf(stderr, "FAIL: no CPU backend\n"); return 1; }

    const auto istft_ref  = run(cpu, build_istft_transposed_conv, istft_inputs(false));
    const auto istft_cols = run(cpu, build_istft_columns, istft_inputs(true));
    const auto rope_ref   = run(cpu, [](ggml_context * c) { return build_rope(c, false); }, rope_inputs());
    const auto rope_rows  = run(cpu, [](ggml_context * c) { return build_rope(c, true); }, rope_inputs());

    bool ok = check_weight_types() && check_frontend_backend(cpu) &&
              matches(istft_cols, istft_ref, kCpuTolerance, "CPU iSTFT columns") &&
              matches(rope_rows, rope_ref, kCpuTolerance, "CPU in-place rope");

    const char * requested = std::getenv("COSYVOICE_TEST_BACKEND");
    if (ok && requested && *requested) {
        ggml_backend_t device = ::tts_cpp::detail::init_requested_backend(requested, /*verbose=*/true, "test");
        if (!device) {
            fprintf(stderr, "FAIL: COSYVOICE_TEST_BACKEND=%s is not available\n", requested);
            ok = false;
        } else {
            ok = check_device(device, istft_ref, rope_ref);
            ggml_backend_free(device);
        }
    }

    ggml_backend_free(cpu);
    if (ok) fprintf(stderr, "PASS\n");
    return ok ? 0 : 1;
}
