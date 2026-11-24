#include "cosyvoice_pipeline.h"
#include "cosyvoice_fit_internal.h"
#include "backend_util.h"
#include "sched_dispatch.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

constexpr int kHeadDim = 64;
constexpr int kHeads = 14;
constexpr int kKvHeads = 2;
constexpr int kCacheSlack = 17;
constexpr int kProbeLength = 64;
constexpr int kUnsupportedHeadDim = 16;
constexpr float kTolerance = 0.005f;
constexpr float kQueryFrequency = 0.031f;
constexpr float kKeyFrequency = 0.017f;
constexpr float kValueFrequency = 0.023f;
constexpr size_t kContextBytes = 1024 * 1024;

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

bool supports_lm_attention(ggml_backend_t backend, const qwen_hp & hp) {
    if (!tts_cpp::detail::backend_is_cuda(backend) &&
        !tts_cpp::detail::backend_is_metal(backend)) return false;
    ggml_context * ctx = ggml_init({kContextBytes, nullptr, true});
    require(ctx != nullptr, "cannot create attention probe context");
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hp.head_dim, 1, hp.n_head, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hp.head_dim, kProbeLength, hp.n_kv, 1);
    ggml_tensor * v = ggml_dup_tensor(ctx, k);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f, 0, 0);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    const bool supported = ggml_backend_supports_op(backend, out);
    ggml_free(ctx);
    return supported;
}

std::vector<float> make_values(size_t count, float frequency) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) values[i] = std::sin(float(i) * frequency);
    return values;
}

double dot(const float * q, const float * k) {
    double value = 0;
    for (int d = 0; d < kHeadDim; ++d) value += double(q[d]) * k[d];
    return value / std::sqrt(double(kHeadDim));
}

std::vector<double> scores(const float * q, const std::vector<float> & k, int head, int length) {
    std::vector<double> values(length);
    for (int t = 0; t < length; ++t) {
        values[t] = dot(q, k.data() + (t * kKvHeads + head) * kHeadDim);
    }
    return values;
}

double normalize(std::vector<double> & values) {
    const double maximum = *std::max_element(values.begin(), values.end());
    double total = 0;
    for (double & value : values) {
        value = std::exp(value - maximum);
        total += value;
    }
    return total;
}

double weighted_value(const std::vector<double> & weights, double total,
                      const std::vector<float> & v, int head, int dimension) {
    double value = 0;
    for (size_t t = 0; t < weights.size(); ++t) {
        value += weights[t] * v[(t * kKvHeads + head) * kHeadDim + dimension];
    }
    return value / total;
}

void check_head(const float * got, const float * q, const std::vector<float> & k,
                const std::vector<float> & v, int head, int length) {
    std::vector<double> weights = scores(q, k, head, length);
    const double total = normalize(weights);
    for (int d = 0; d < kHeadDim; ++d) {
        const double expected = weighted_value(weights, total, v, head, d);
        require(std::isfinite(got[d]) && std::abs(got[d] - expected) < kTolerance,
                "strided cache attention differs from the scalar reference");
    }
}

void check_output(const std::vector<float> & got, const std::vector<float> & q,
                  const std::vector<float> & k, const std::vector<float> & v, int length) {
    for (int h = 0; h < kHeads; ++h) {
        check_head(got.data() + h * kHeadDim, q.data() + h * kHeadDim,
                   k, v, h / (kHeads / kKvHeads), length);
    }
}

void check_attention(ggml_backend_t backend, int length) {
    ggml_context * ctx = ggml_init({kContextBytes, nullptr, true});
    require(ctx != nullptr, "cannot create attention context");
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kHeadDim, 1, kHeads, 1);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, length + kCacheSlack);
    ggml_tensor * v = ggml_dup_tensor(ctx, k);
    ggml_tensor * kh = ggml_view_3d(ctx, k, kHeadDim, length, kKvHeads, k->nb[2], k->nb[1], 0);
    ggml_tensor * vh = ggml_view_3d(ctx, v, kHeadDim, length, kKvHeads, v->nb[2], v->nb[1], 0);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, kh, vh, nullptr, 1.0f / std::sqrt(float(kHeadDim)), 0, 0);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "cannot allocate attention tensors");
    const auto q_data = make_values(ggml_nelements(q), kQueryFrequency);
    const auto k_data = make_values(ggml_nelements(k), kKeyFrequency);
    const auto v_data = make_values(ggml_nelements(v), kValueFrequency);
    ggml_backend_tensor_set(q, q_data.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, k_data.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, v_data.data(), 0, ggml_nbytes(v));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "attention dispatch failed");
    std::vector<float> got(ggml_nelements(out));
    ggml_backend_tensor_get(out, got.data(), 0, ggml_nbytes(out));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    check_output(got, q_data, k_data, v_data, length);
}

void check_lengths(ggml_backend_t backend) {
    for (int length : {1, 63, 64, 65, 324, 1025}) check_attention(backend, length);
}

constexpr int kReplayVocabulary = 32;
constexpr int kReplaySteps = 260;
constexpr int kReplaySeed = 42;
constexpr int kReplayDepth = 2;
constexpr int kReplayHidden = 128;
constexpr int kReplayHeads = 2;
constexpr int kReplayKvHeads = 1;
constexpr int kReplayTextToken = 2;
constexpr int kReplaySos = 0;
constexpr int kReplayTask = 1;
constexpr int kReplayBoundaryTextLengths[] = {1, 125, 126, 127};
constexpr float kReplayWeightScale = 0.05f;

void add_weight(model_ctx & model, const std::string & name, int rows, int columns) {
    model.tensors[name] = ggml_new_tensor_2d(model.ctx_w, GGML_TYPE_F32, rows, columns);
}

void add_projections(model_ctx & model, const qwen_hp & hp, const std::string & prefix) {
    for (const std::string projection : {"q_proj", "k_proj", "v_proj"}) {
        const int width = projection == "q_proj" ? hp.hidden : hp.head_dim * hp.n_kv;
        add_weight(model, prefix + projection + "/weight", hp.hidden, width);
        add_weight(model, prefix + projection + "/bias", width, 1);
    }
}

void add_layer(model_ctx & model, const qwen_hp & hp, int layer) {
    const std::string prefix = "lm/blk/" + std::to_string(layer) + "/";
    add_weight(model, prefix + "in_ln/weight", hp.hidden, 1);
    add_weight(model, prefix + "post_ln/weight", hp.hidden, 1);
    add_projections(model, hp, prefix);
    add_weight(model, prefix + "o_proj/weight", hp.hidden, hp.hidden);
    add_weight(model, prefix + "gate/weight", hp.hidden, hp.inter);
    add_weight(model, prefix + "up/weight", hp.hidden, hp.inter);
    add_weight(model, prefix + "down/weight", hp.inter, hp.hidden);
}

void add_layers(model_ctx & model, const qwen_hp & hp) {
    for (int layer = 0; layer < hp.depth; ++layer) add_layer(model, hp, layer);
}

std::vector<float> random_values(size_t count, std::mt19937 & random) {
    std::uniform_real_distribution<float> distribution(-kReplayWeightScale, kReplayWeightScale);
    std::vector<float> values(count);
    for (float & value : values) value = distribution(random);
    return values;
}

void fill_model(model_ctx & model) {
    std::mt19937 random(kReplaySeed);
    for (const auto & entry : model.tensors) {
        auto values = random_values(ggml_nelements(entry.second), random);
        if (entry.first.find("ln/weight") != std::string::npos || entry.first == "lm/norm/weight") {
            std::fill(values.begin(), values.end(), 1.0f);
        }
        ggml_backend_tensor_set(entry.second, values.data(), 0, ggml_nbytes(entry.second));
    }
}

void make_model(model_ctx & model, const qwen_hp & hp) {
    model.ctx_w = ggml_init({kContextBytes, nullptr, true});
    model.ctx_h = ggml_init({kContextBytes, nullptr, true});
    require(model.ctx_w && model.ctx_h, "cannot create replay model contexts");
    model.tensors["lm/embed_tokens/weight"] = ggml_new_tensor_2d(model.ctx_h, GGML_TYPE_F32, hp.hidden, kReplayVocabulary);
    model.tensors["lm/speech_embedding/weight"] = ggml_new_tensor_2d(model.ctx_h, GGML_TYPE_F32, hp.hidden, kReplayVocabulary);
    add_layers(model, hp);
    add_weight(model, "lm/norm/weight", hp.hidden, 1);
    add_weight(model, "lm/llm_decoder/weight", hp.hidden, kReplayVocabulary);
    model.buffer_w = ggml_backend_alloc_ctx_tensors(model.ctx_w, model.backend);
    require(model.buffer_w != nullptr, "cannot allocate replay model weights");
    model.kv_i["cosyvoice3.llm.sos"] = kReplaySos;
    model.kv_i["cosyvoice3.llm.task_id"] = kReplayTask;
    model.kv_i["cosyvoice3.llm.speech_token_size"] = kReplayVocabulary;
    model.buffer_h = ggml_backend_alloc_ctx_tensors_from_buft(model.ctx_h, ggml_backend_cpu_buffer_type());
    require(model.buffer_h != nullptr, "cannot allocate replay embeddings");
    fill_model(model);
}

void check_replay_lengths(model_ctx & model, const qwen_hp & hp) {
    for (int text_length : kReplayBoundaryTextLengths) {
        const std::vector<int> text(text_length, kReplayTextToken);
        const auto expected = cosyvoice_llm_generate(model, hp, text, {}, kReplaySteps,
            false, kReplaySeed, kReplaySteps, nullptr, false);
        const auto actual = cosyvoice_llm_generate(model, hp, text, {}, kReplaySteps,
            false, kReplaySeed, kReplaySteps, nullptr, true);
        require(actual.size() == kReplaySteps, "replay stopped before crossing window boundaries");
        require(std::set<int>(expected.begin(), expected.end()).size() > 1, "degenerate replay fixture");
        require(actual == expected, "replayed LM trajectory differs across a cache window boundary");
    }
}

void check_replay_memory(model_ctx & model, const qwen_hp & hp) {
    constexpr int kPrefill = 3;
    uint64_t measured_kv = 0, actual_kv = 0, actual_arena = 0;
    cosyvoice_fit_price measured_arena;
    std::string error;
    require(cosyvoice_fit_measure_llm(model, hp, kPrefill, kReplaySteps,
        measured_kv, measured_arena, &error), "cannot measure replay memory");
    require(cosyvoice_fit_llm_parity_probe(model, hp, kPrefill, kReplaySteps,
        actual_kv, actual_arena, &error), "cannot allocate replay memory probe");
    require(measured_kv == actual_kv, "replay KV memory estimate differs from allocation");
    require(measured_arena.device_bytes >= actual_arena, "replay arena memory estimate is too small");
}

void check_replay(ggml_backend_t backend) {
    qwen_hp hp;
    hp.depth = kReplayDepth; hp.hidden = kReplayHidden;
    hp.n_head = kReplayHeads; hp.n_kv = kReplayKvHeads; hp.inter = kReplayHidden;
    const bool enabled = cosyvoice_lm_replay_enabled(backend, hp);
    require(!tts_cpp::detail::sched_force_enabled() || !enabled, "replay enabled with forced scheduler");
    if (!enabled) return;
    model_ctx model;
    model.backend = backend;
    try {
        make_model(model, hp);
        check_replay_lengths(model, hp);
        check_replay_memory(model, hp);
    } catch (...) {
        cosyvoice_free(model);
        throw;
    }
    cosyvoice_free(model);
}

}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_name(argc > 1 ? argv[1] : "CPU", nullptr);
    if (!backend) return 1;
    int result = 0;
    try {
        qwen_hp hp;
        const bool enabled = cosyvoice_lm_fa_enabled(backend, hp);
        require(enabled == supports_lm_attention(backend, hp), "unexpected LM flash-attention selection");
        if (!enabled || !tts_cpp::detail::backend_is_cuda(backend)) {
            require(!cosyvoice_lm_replay_enabled(backend, hp), "replay enabled without CUDA flash attention");
        }
        if (enabled) {
            hp.head_dim = kUnsupportedHeadDim;
            require(!cosyvoice_lm_fa_enabled(backend, hp), "unsupported geometry must use regular attention");
            check_lengths(backend);
            check_replay(backend);
        }
        fprintf(stderr, "PASS: CosyVoice LM attention (%s)\n",
                enabled ? "flash attention" : "regular attention");
    } catch (const std::exception & error) {
        fprintf(stderr, "FAIL: %s\n", error.what());
        result = 1;
    }
    ggml_backend_free(backend);
    return result;
}
