#include "parakeet_ctc.h"
#include "parakeet_tdt.h"

#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

bool test_model_type() {
    return std::strcmp(
        parakeet::model_type_name(parakeet::ParakeetModelType::RNNT),
        "rnnt") == 0;
}

bool test_graph_outputs() {
    constexpr size_t context_size = 64 * 1024;
    constexpr size_t graph_size = 32;
    constexpr int token_count = 3;
    constexpr int duration_count = 2;

    ggml_init_params params{};
    params.mem_size = context_size;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * rnnt_logits =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, token_count, 1);
    const parakeet::TransducerGraphOutputs rnnt =
        parakeet::build_transducer_argmax_outputs(
            ctx, rnnt_logits, token_count, 0);

    ggml_tensor * tdt_logits =
        ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, token_count + duration_count, 1);
    const parakeet::TransducerGraphOutputs tdt =
        parakeet::build_transducer_argmax_outputs(
            ctx, tdt_logits, token_count, duration_count);

    ggml_cgraph * graph =
        ggml_new_graph_custom(ctx, graph_size, false);
    ggml_build_forward_expand(graph, rnnt.token);

    const bool valid =
        rnnt.token != nullptr &&
        rnnt.duration == nullptr &&
        tdt.token != nullptr &&
        tdt.duration != nullptr &&
        ggml_nelements(rnnt.token) == 1 &&
        ggml_graph_n_nodes(graph) > 0;
    ggml_free(ctx);
    return valid;
}

parakeet::TdtRuntimeWeights make_cpu_runtime() {
    parakeet::TdtRuntimeWeights weights;
    weights.H_pred = 1;
    weights.H_joint = 1;
    weights.D_enc = 1;
    weights.V_plus_1 = 3;
    weights.V_out = 3;
    weights.L = 1;
    weights.num_durations = 0;
    weights.use_graphs = false;
    weights.embed = {1.0f, -1.0f, 0.0f};

    parakeet::TdtHostLstmLayer layer;
    layer.w_ih = {10.0f, -10.0f, 1.0f, 10.0f};
    layer.w_hh = {0.0f, 0.0f, 0.0f, 0.0f};
    layer.b_ih = {0.0f, 0.0f, 0.0f, 0.0f};
    layer.b_hh = {0.0f, 0.0f, 0.0f, 0.0f};
    weights.host_lstm = {layer};

    weights.host_joint_enc_w = {1.0f};
    weights.host_joint_enc_b = {0.0f};
    weights.host_joint_pred_w = {1.0f};
    weights.host_joint_pred_b = {0.0f};
    weights.host_joint_out_w = {-1.0f, 0.0f, 1.0f};
    weights.host_joint_out_b = {2.0f, -10.0f, -0.5f};
    return weights;
}

bool test_cpu_predictor_state() {
    parakeet::ParakeetCtcModel model;
    model.model_type = parakeet::ParakeetModelType::RNNT;
    model.blank_id = 2;
    model.vocab_size = 2;
    model.encoder_cfg.d_model = 1;

    parakeet::TdtRuntimeWeights weights = make_cpu_runtime();
    parakeet::RnntDecodeOptions options;
    options.max_symbols_per_step = 2;
    parakeet::RnntDecodeState state;
    const float encoder[] = {1.0f};
    std::vector<int32_t> tokens;
    int first_steps = 0;

    const int first_rc = parakeet::rnnt_decode_window(
        model, weights, encoder, 1, 1, options, state, tokens, first_steps);
    if (first_rc != 0 ||
        tokens != std::vector<int32_t>({0}) ||
        first_steps != 2 ||
        state.pred_out.size() != 1 ||
        state.pred_out[0] <= 0.5f) {
        return false;
    }

    int second_steps = 0;
    const int second_rc = parakeet::rnnt_decode_window(
        model, weights, encoder, 1, 1, options, state, tokens, second_steps);
    return second_rc == 0 &&
           tokens == std::vector<int32_t>({0}) &&
           second_steps == 1 &&
           state.carry_frames == 0;
}

}

int main() {
    if (!test_model_type()) {
        std::fprintf(stderr, "RNNT model type reporting failed\n");
        return 1;
    }
    if (!test_graph_outputs()) {
        std::fprintf(stderr, "RNNT graph output construction failed\n");
        return 2;
    }
    if (!test_cpu_predictor_state()) {
        std::fprintf(stderr, "RNNT CPU predictor state semantics failed\n");
        return 3;
    }
    std::fprintf(stderr, "RNNT tests passed\n");
    return 0;
}
