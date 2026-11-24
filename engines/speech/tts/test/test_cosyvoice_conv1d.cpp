// Pins cosyvoice_conv1d_f32's non-contiguous-input guard. The builder must
// ggml_cont a strided view before im2col: ggml-vulkan's IM2COL supports_op
// requires a contiguous signal, so without the guard a view input demotes
// the whole stage to the sched-fallback path. Deviceless on CPU (a permuted
// view must produce bit-identical output to a pre-contiguous input); with
// COSYVOICE_TEST_GPU=1 it additionally requires the built graph to be fully
// supported by the selected GPU backend (no scheduler fallback) and the GPU
// output to match the CPU reference.

#include "backend_selection.h"
#include "cosyvoice_pipeline.h"
#include "sched_dispatch.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kKernel = 3, kCin = 4, kCout = 2, kLen = 17;

// Deterministic fills so the two legs and both backends see identical bytes.
std::vector<float> make_signal() {
    std::vector<float> v((size_t)kCin * kLen);
    for (size_t i = 0; i < v.size(); ++i) v[i] = std::sin(0.37f * (float)i) + 0.11f * (float)(i % 5);
    return v;
}
std::vector<float> make_weights() {
    std::vector<float> v((size_t)kKernel * kCin * kCout);
    for (size_t i = 0; i < v.size(); ++i) v[i] = std::cos(0.23f * (float)i) - 0.07f * (float)(i % 3);
    return v;
}

bool close_enough(float a, float b) {
    return std::fabs(a - b) <= 1e-4f + 1e-4f * std::fabs(b);
}

// The f16 leg loses kernel precision to the half round-trip, so it gets a
// correspondingly looser bound.
bool close_enough_f16(float a, float b) {
    return std::fabs(a - b) <= 5e-3f + 5e-3f * std::fabs(b);
}

// Builds conv over either a contiguous copy of the signal or a strided
// permute view of it, runs on `backend`, and reports whether the graph was
// fully supported there (i.e. needed no sched fallback).
std::vector<float> run_conv(ggml_backend_t backend, bool use_view, bool & fully_supported,
                            ggml_type wtype = GGML_TYPE_F32) {
    static const size_t buf_size = 4 * 1024 * 1024;
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    // Signal staged transposed [Cin, Len]; the permute view flips it to the
    // conv layout [Len, Cin, 1] WITHOUT making it contiguous.
    ggml_tensor * xt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kCin, kLen);
    ggml_set_name(xt, "xt"); ggml_set_input(xt);
    ggml_tensor * w = ggml_new_tensor_3d(ctx, wtype, kKernel, kCin, kCout);
    ggml_set_name(w, "w"); ggml_set_input(w);

    ggml_tensor * x = ggml_permute(ctx, xt, 1, 0, 2, 3);          // [Len, Cin, 1] strided view
    if (!use_view) x = ggml_cont(ctx, x);                          // pre-contiguous reference leg
    ggml_tensor * y = cosyvoice_conv1d_f32(ctx, w, x, 1, 0, 1);
    ggml_set_name(y, "y"); ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    fully_supported = ::tts_cpp::detail::graph_fully_supported(backend, gf);

    auto * allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);
    const auto sig = make_signal();
    const auto wts = make_weights();
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "xt"), sig.data(), 0, sig.size() * sizeof(float));
    if (wtype == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> wts16(wts.size());
        ggml_fp32_to_fp16_row(wts.data(), wts16.data(), (int64_t)wts.size());
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "w"), wts16.data(), 0, wts16.size() * sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "w"), wts.data(), 0, wts.size() * sizeof(float));
    }
    ggml_backend_graph_compute(backend, gf);
    ggml_tensor * out = ggml_graph_get_tensor(gf, "y");
    std::vector<float> res(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    return res;
}

// Grouped conv legs: the per-(group, batch) form and the batched form must
// produce the same output — each output element is the same dot product over
// the same group slice, only the dispatch shape differs (test pins the
// equivalence the DiT's conv_pos_embed relies on when it picks the batched
// form on Metal).
constexpr int kGK = 5, kGCin = 32, kGCout = 16, kGGroups = 4, kGLen = 33, kGB = 2;

std::vector<float> make_grouped_signal() {
    std::vector<float> v((size_t)kGLen * kGCin * kGB);
    for (size_t i = 0; i < v.size(); ++i) v[i] = std::sin(0.19f * (float)i) - 0.05f * (float)(i % 7);
    return v;
}
std::vector<float> make_grouped_weights() {
    std::vector<float> v((size_t)kGK * (kGCin / kGGroups) * kGCout);
    for (size_t i = 0; i < v.size(); ++i) v[i] = std::cos(0.31f * (float)i) + 0.03f * (float)(i % 4);
    return v;
}

std::vector<float> run_grouped_conv(ggml_backend_t backend, bool batched, bool & fully_supported) {
    static const size_t buf_size = 8 * 1024 * 1024;
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kGLen, kGCin, kGB);
    ggml_set_name(x, "x"); ggml_set_input(x);
    ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kGK, kGCin / kGGroups, kGCout);
    ggml_set_name(w, "w"); ggml_set_input(w);

    ggml_tensor * y = batched
        ? cosyvoice_conv1d_grouped_batched(ctx, w, x, kGGroups)
        : cosyvoice_conv1d_grouped(ctx, w, x, kGGroups);
    ggml_set_name(y, "y"); ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    fully_supported = ::tts_cpp::detail::graph_fully_supported(backend, gf);

    auto * allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);
    const auto sig = make_grouped_signal();
    const auto wts = make_grouped_weights();
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x"), sig.data(), 0, sig.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "w"), wts.data(), 0, wts.size() * sizeof(float));
    ggml_backend_graph_compute(backend, gf);
    ggml_tensor * out = ggml_graph_get_tensor(gf, "y");
    std::vector<float> res(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    return res;
}

int compare(const std::vector<float> & got, const std::vector<float> & ref, const char * what) {
    if (got.size() != ref.size()) {
        fprintf(stderr, "FAIL: %s size mismatch (%zu vs %zu)\n", what, got.size(), ref.size());
        return 1;
    }
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!close_enough(got[i], ref[i])) {
            fprintf(stderr, "FAIL: %s mismatch @ %zu: got=%.6g ref=%.6g\n", what, i, got[i], ref[i]);
            return 1;
        }
    }
    return 0;
}

} // namespace

int main() {
    ggml_backend_t cpu = ::tts_cpp::detail::init_cpu_backend();
    if (!cpu) { fprintf(stderr, "FAIL: no CPU backend\n"); return 1; }

    bool supported = false;
    const auto ref_cont = run_conv(cpu, /*use_view=*/false, supported);
    const auto cpu_view = run_conv(cpu, /*use_view=*/true, supported);
    // Same dot products in the same order once the guard conts the view, so
    // the CPU comparison is exact.
    if (cpu_view != ref_cont) {
        fprintf(stderr, "FAIL: CPU view leg differs from pre-contiguous leg\n");
        return 1;
    }

    // f16 kernel leg: the graph must stay fully CPU-supported (the im2col dst
    // follows the kernel type, so the matmul sees an f16/f16 pair the CPU
    // accepts) and the result must track the f32 reference.
    bool f16_supported = false;
    const auto cpu_f16 = run_conv(cpu, /*use_view=*/false, f16_supported, GGML_TYPE_F16);
    if (!f16_supported) {
        fprintf(stderr, "FAIL: f16-kernel conv graph not fully supported on the CPU backend\n");
        ggml_backend_free(cpu);
        return 1;
    }
    if (cpu_f16.size() != ref_cont.size()) {
        fprintf(stderr, "FAIL: f16 leg size mismatch\n");
        ggml_backend_free(cpu);
        return 1;
    }
    for (size_t i = 0; i < ref_cont.size(); ++i) {
        if (!close_enough_f16(cpu_f16[i], ref_cont[i])) {
            fprintf(stderr, "FAIL: f16 leg mismatch @ %zu: got=%.6g ref=%.6g\n",
                    i, cpu_f16[i], ref_cont[i]);
            ggml_backend_free(cpu);
            return 1;
        }
    }

    bool grp_supported = false, grp_batched_supported = false;
    const auto grp_ref     = run_grouped_conv(cpu, /*batched=*/false, grp_supported);
    const auto grp_batched = run_grouped_conv(cpu, /*batched=*/true, grp_batched_supported);
    if (!grp_batched_supported) {
        fprintf(stderr, "FAIL: batched grouped conv graph not fully supported on the CPU backend\n");
        ggml_backend_free(cpu);
        return 1;
    }
    if (compare(grp_batched, grp_ref, "CPU batched grouped conv vs per-group")) {
        ggml_backend_free(cpu);
        return 1;
    }

    if (std::getenv("COSYVOICE_TEST_GPU")) {
        ggml_backend_t gpu = ::tts_cpp::detail::init_gpu_backend(
            99, /*verbose=*/false, "test-cosyvoice", /*vulkan_device=*/0,
            /*allow_arm_mali=*/false, /*out_gpu_present_but_unused=*/nullptr,
            cosyvoice_gpu_requirement());
        if (!gpu) { fprintf(stderr, "FAIL: COSYVOICE_TEST_GPU set but no GPU backend\n"); return 1; }
        bool gpu_supported = false;
        const auto gpu_view = run_conv(gpu, /*use_view=*/true, gpu_supported);
        if (!gpu_supported) {
            fprintf(stderr, "FAIL: view-input conv graph not fully supported on %s "
                            "(guard regressed; stage would demote to sched fallback)\n",
                    ggml_backend_name(gpu));
            ggml_backend_free(gpu);
            ggml_backend_free(cpu);
            return 1;
        }
        const int rc = compare(gpu_view, ref_cont, "GPU view leg vs CPU");
        if (rc) { ggml_backend_free(gpu); ggml_backend_free(cpu); return 1; }

        bool gpu_grp_supported = false;
        const auto gpu_grp_batched = run_grouped_conv(gpu, /*batched=*/true, gpu_grp_supported);
        if (!gpu_grp_supported) {
            fprintf(stderr, "FAIL: batched grouped conv graph not fully supported on %s "
                            "(stage would demote to sched fallback)\n",
                    ggml_backend_name(gpu));
            ggml_backend_free(gpu);
            ggml_backend_free(cpu);
            return 1;
        }
        const int grc = compare(gpu_grp_batched, grp_ref, "GPU batched grouped conv vs CPU per-group");
        ggml_backend_free(gpu);
        if (grc) { ggml_backend_free(cpu); return 1; }
    }

    ggml_backend_free(cpu);
    fprintf(stderr, "PASS\n");
    return 0;
}
