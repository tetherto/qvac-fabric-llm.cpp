// Which precision the block-quantised matmuls of a built graph ask for.
//
// The f32 fixtures cannot reach this: GGML_PREC_F32 only changes what a
// backend does with a quantised weight. CUDA's default route is the integer
// dot product, which accumulates in f32 already, so the marker there buys
// nothing and costs a dequantise-to-f32 round trip through cuBLAS. Every other
// backend needs the marker -- ggml-vulkan reduces quantised matmuls in f16
// under PREC_DEFAULT -- so the relaxation has to stay scoped to CUDA, and a
// graph is the only place that scoping is observable.
//
// Two readings, because the language model and the codec fail differently. The
// LM arm walks its built graphs, which also says every matmul in them reaches
// the shared helper rather than calling ggml_mul_mat itself. The codec arm
// runs the helper over the decoder's own resident weights: its graph builders
// are file-local, and what is at risk there is the rule meeting q8_0 weights
// the LM fixtures do not carry.

#include "audio8/graph.h"
#include "audio8/internal.h"
#include "backend_util.h"
#include "gpu_arm.h"

#include "ggml.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace tts_cpp::audio8::detail;

namespace {

constexpr int GPU_LAYERS = 99;
constexpr int DECODE_WIDTH = 1;
constexpr int DECODE_N_PAST = 0;

int failures = 0;

void fail(const std::string & what) {
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
}

int requested_layers() {
    return audio8_test::is_gpu_test() ? GPU_LAYERS : 0;
}

// CUDA is the one backend whose default quantised route already accumulates in
// f32; everywhere else the marker is what keeps the reduction out of f16.
bool backend_drops_the_marker(ggml_backend_t backend) {
    return ::tts_cpp::detail::backend_is_cuda(backend);
}

bool is_quantised_weight_matmul(const ggml_tensor * node) {
    return node->op == GGML_OP_MUL_MAT && node->src[0] &&
           ggml_is_quantized(node->src[0]->type);
}

ggml_prec prec_of(const ggml_tensor * node) {
    return static_cast<ggml_prec>(node->op_params[0]);
}

struct prec_tally {
    int quantised = 0;
    int wrong = 0;
};

const char * prec_name(ggml_prec prec) {
    return prec == GGML_PREC_F32 ? "GGML_PREC_F32" : "GGML_PREC_DEFAULT";
}

// Graph matmuls carry no name, so the node index and the weight it multiplies
// are what locates a failure.
void report_wrong_node(int index, const ggml_tensor * node, ggml_prec want) {
    std::fprintf(stderr, "  node %d: %s weight [%lld x %lld] at %s, want %s\n", index,
                 ggml_type_name(node->src[0]->type),
                 static_cast<long long>(node->src[0]->ne[0]),
                 static_cast<long long>(node->src[0]->ne[1]), prec_name(prec_of(node)),
                 prec_name(want));
}

void tally_graph(ggml_cgraph * graph, ggml_prec want, prec_tally & tally) {
    const int n_nodes = ggml_graph_n_nodes(graph);
    for (int index = 0; index < n_nodes; ++index) {
        const ggml_tensor * node = ggml_graph_node(graph, index);
        if (!is_quantised_weight_matmul(node)) continue;
        ++tally.quantised;
        if (prec_of(node) == want) continue;
        ++tally.wrong;
        report_wrong_node(index, node, want);
    }
}

void tally_slow_graph(lm_model & model, ggml_prec want, prec_tally & tally) {
    scratch build(AUDIO8_MAX_NODES);
    if (!build.ok()) {
        fail("could not create the slow graph context");
        return;
    }
    slow_graph_outputs outs;
    build_slow_graph(model, build, DECODE_WIDTH, DECODE_N_PAST, outs);
    tally_graph(build.graph, want, tally);
}

void tally_fast_graph(lm_model & model, ggml_prec want, prec_tally & tally) {
    scratch build(AUDIO8_MAX_NODES);
    if (!build.ok()) {
        fail("could not create the fast graph context");
        return;
    }
    build_fast_fit_graph(model, build, /*position=*/0, /*prime=*/true);
    tally_graph(build.graph, want, tally);
}

void report_tally(const char * arm, const prec_tally & tally, ggml_prec want) {
    std::printf("  %s: %d quantised matmuls, %d not at %s\n", arm, tally.quantised,
                tally.wrong, prec_name(want));
    if (tally.quantised == 0) {
        fail(std::string(arm) + ": no quantised matmul -- the model is not a block format");
    }
    if (tally.wrong != 0) {
        fail(std::to_string(tally.wrong) + " quantised " + arm +
             " matmuls are not at " + prec_name(want));
    }
}

void check_language_model(lm_model & model, ggml_prec want) {
    prec_tally tally;
    tally_slow_graph(model, want, tally);
    tally_fast_graph(model, want, tally);
    report_tally("language model", tally, want);
}

// The codec's graph builders are file-local, so the rule is read off its
// resident weights instead: every quantised one, through the helper the codec's
// linear and swiglu reach.
void tally_codec_weights(codec_model & model, ggml_prec want, prec_tally & tally) {
    scratch build(AUDIO8_MAX_NODES);
    if (!build.ok()) {
        fail("could not create the codec probe context");
        return;
    }
    for (ggml_tensor * weight = ggml_get_first_tensor(model.ctx_w); weight;
         weight = ggml_get_next_tensor(model.ctx_w, weight)) {
        if (!ggml_is_quantized(weight->type)) continue;
        ggml_tensor * column = ggml_new_tensor_2d(build.ctx, GGML_TYPE_F32, weight->ne[0], 1);
        ggml_tensor * product = precise_mul_mat(build.ctx, weight, column);
        ++tally.quantised;
        if (prec_of(product) == want) continue;
        ++tally.wrong;
        report_wrong_node(tally.quantised, product, want);
    }
}

void check_codec(const std::string & path, int n_gpu_layers, ggml_prec want) {
    codec_model model;
    std::string error;
    if (!load_codec(path, n_gpu_layers, /*backend=*/"", model, &error)) {
        fail("loading the codec: " + error);
        return;
    }
    prec_tally tally;
    tally_codec_weights(model, want, tally);
    report_tally("codec decoder", tally, want);
    free_codec(model);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <lm-quantised.gguf> [codec-decoder-quantised.gguf]\n",
                     argv[0]);
        return 1;
    }

    lm_model model;
    std::string error;
    if (!load_lm(argv[1], requested_layers(), /*backend=*/"", model, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    if (!audio8_test::check_requested_gpu("quantised precision", model.backend)) {
        free_lm(model);
        return 1;
    }

    const ggml_prec want =
        backend_drops_the_marker(model.backend) ? GGML_PREC_DEFAULT : GGML_PREC_F32;
    std::printf("backend: %s, expecting %s on quantised weights\n",
                ggml_backend_name(model.backend), prec_name(want));
    check_language_model(model, want);
    free_lm(model);
    if (argc > 2) check_codec(argv[2], requested_layers(), want);

    if (failures == 0) {
        std::fprintf(stderr, "audio8 quantised precision: PASS\n");
        return 0;
    }
    std::fprintf(stderr, "audio8 quantised precision: %d failure(s)\n", failures);
    return 1;
}
