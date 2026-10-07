#include "common.h"
#include "speculative.h"
#include "../src/llama-context.h"
#include "../ggml/src/ggml-backend-impl.h"
#include <set>
#include "log.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"
#include "uint8-buff-stream.h"

// TODO: replace with #include "llama-ext.h" in the future
#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"
#include "../src/llama-model.h"

#include <algorithm>
#include <bitset>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    size_t seed = *(const size_t *) userdata;
    std::hash<std::string> hasher;
    seed ^= hasher(tensor->name);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);
    // Keep MPT's activation divisors away from zero to avoid FP16 overflow.
    std::uniform_real_distribution<float> dis_scale(0.5f, 1.5f);
    const bool is_act_scale = string_ends_with(tensor->name, ".ffn.act.scales");

    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = is_act_scale ? dis_scale(gen) : dis(gen);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = ggml_fp32_to_fp16(is_act_scale ? dis_scale(gen) : dis(gen));
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

static void usage(char ** argv) {
    printf("Usage: %s [-a/--arch arch] [-s/--seed seed] [-o/--out dir] [-v N] [-h/--help] [--mtp-shared|--mtp-shared-cpu|--hadamard-contracts|--qsa-unified-multiseq|--glm5-kpool-sequences|--glm5-multiseq|--glm5-multiseq-gpu|--glm5-tensor-oom|--glm5-reserve-covers-context|--glm5-reserve-scrambled-states|--glm5-invalid-metadata|--layer-inp-pos-min|--layer-split-cuda-graph]\n", argv[0]);
}

static std::vector<llama_token> get_tokens(const uint32_t n_tokens, const uint32_t n_vocab, const size_t seed){
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    std::vector<llama_token> ret;
    ret.reserve(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        ret.push_back(dis(gen));
    }
    return ret;
}

static gguf_context_ptr get_gguf_ctx(const llm_arch arch, const bool moe) {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());
    const uint32_t n_ctx = 256;

    uint32_t n_vocab = 128;
    uint32_t n_embd  = 256;
    uint32_t n_head  = 2;
    uint32_t n_ff    = 384;
    uint32_t n_layer = 2;
    if (arch == LLM_ARCH_LLAMA4) {
        n_layer = 4; // hparams.n_no_rope_layer_step is hard-coded to 4
    } else if (arch == LLM_ARCH_GEMMA4) {
        n_embd = 128;
        n_head = 2;
        n_ff   = 192;
        n_layer = 5; // need at least 5 for swa_pattern (every 5th is full_attention)
    } else if (arch == LLM_ARCH_GEMMA3N) {
        n_embd = 64;
        n_head = 1;
        n_ff   = 96;
        n_layer = 22; // hparams.n_layer_kv_from_start = 20 is hardcoded
    } else if (arch == LLM_ARCH_DEEPSEEK4) {
        // head size 64 so that GPU flash attention kernels support the model
        n_embd  = 512;
        n_head  = 8;
        n_ff    = 1024;
        n_layer = 4;
    } else if (arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_LAGUNA) {
        n_embd = 160; // exercise per-head tensor split granularity with head size 80
    } else if (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_AFMOE) {
        n_head = 4;
    } else if (arch == LLM_ARCH_GLM5_NEXT) {
        n_embd = 128;
        n_head = 8;
        n_ff   = 192;
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_DOTS3NOTE
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3
            || arch == LLM_ARCH_MISTRAL4
            || arch == LLM_ARCH_HY_V4) {
        n_embd = 128;
        n_head = 1;
        n_ff   = 192;
    } else if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        n_layer = 3;
    } else if (arch == LLM_ARCH_CHAMELEON) {
        n_vocab = 10240;
    } else if (arch == LLM_ARCH_QWEN3TTS) {
        //n_vocab = 4096; // must be >= the hard-coded codec head size (3072)
        n_vocab = 3072; // TODO: should be 4096, but user code cannot get `n_vocab_out` yet [TAG_LLAMA_N_VOCAB_OUT]
    } else if (arch == LLM_ARCH_HRM_TEXT) {
        n_layer = 8; // 1 layer per stack x 2 h-cycles x (3 l-cycles + 1) cache slots
    }

    uint32_t n_head_kv = n_head;
    if (arch == LLM_ARCH_QWEN3) {
        n_head_kv = 1; // MQA coverage
    } else if (arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_AFMOE) {
        n_head_kv = 2; // GQA coverage
    }
    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));

    if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        std::vector<uint32_t> n_ff_per_layer;
        n_ff_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_ff_per_layer.push_back(il <= 1 ? 0 : n_ff);
        }
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff_per_layer);
    } else {
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff);
    }

    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,   false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,             1.0f);
    ms.add_kv(LLM_KV_TIME_MIX_EXTRA_DIM,      uint32_t(64));
    ms.add_kv(LLM_KV_TIME_DECAY_EXTRA_DIM,    uint32_t(128));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));

    if (arch == LLM_ARCH_PLAMO2 || arch == LLM_ARCH_JAMBA || arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE ||
            arch == LLM_ARCH_GRANITE_HYBRID || arch == LLM_ARCH_LFM2 || arch == LLM_ARCH_LFM2MOE || arch == LLM_ARCH_KIMI_LINEAR ||
            arch == LLM_ARCH_BAILINGMOE3 || arch == LLM_ARCH_KIMI_K3 || arch == LLM_ARCH_GLM5_NEXT) {
        GGML_ASSERT(n_layer >= 2);
        std::vector<uint32_t> n_head_per_layer;
        n_head_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            // GLM5 stores one compressed MLA latent on DSA layers; its query
            // heads are still described by the uniform attention head count.
            n_head_per_layer.push_back(il == 1 ? 0 : (arch == LLM_ARCH_GLM5_NEXT ? 1 : n_head));
        }
        // GLM5 next KDA heads come from the uniform head count, only head_count_kv is per layer.
        if (arch == LLM_ARCH_GLM5_NEXT) {
            ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        } else {
            ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head_per_layer);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_per_layer);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(1) : n_head_kv);
    }

    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, 8.0f);
    if (arch == LLM_ARCH_DEEPSEEK4) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,   n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH, n_embd_head);
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,   n_embd_head/2);
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_DOTS3NOTE
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3
            || arch == LLM_ARCH_GLM5_NEXT
            || arch == LLM_ARCH_MISTRAL4
            || arch == LLM_ARCH_HY_V4) {
        // GLM5 next MLA is nope only, the cache row is the compressed latent alone.
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       arch == LLM_ARCH_GLM5_NEXT ? uint32_t(512) : uint32_t(576));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     uint32_t(512));
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       arch == LLM_ARCH_GLM5_NEXT ? uint32_t(0) : uint32_t(64));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   uint32_t(192));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, uint32_t(128));
        if (arch == LLM_ARCH_DOTS3NOTE) {
            // SWA layers reuse the same MLA geometry as the full layers in this fixture
            ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,       uint32_t(576));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA_SWA,   uint32_t(192));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA_SWA, uint32_t(128));
            ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,             10000.0f);
            // indexer on the full-attention layers (inverse of the swa pattern)
            std::vector<uint32_t> indexer_types;
            indexer_types.reserve(n_layer);
            for (uint32_t il = 0; il < n_layer; il++) {
                indexer_types.push_back(il % 2 ? 0 : 1);
            }
            ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, indexer_types);
        }
    } else if (arch == LLM_ARCH_MINIMAX_M3) {
        // partial rotary: n_rot must not exceed the indexer key length (64)
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
    }
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,            arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(64) : uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         n_ctx/8);

    if (arch == LLM_ARCH_GEMMA4) {
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,      n_embd/2);
        ms.add_kv(LLM_KV_ATTENTION_SHARED_KV_LAYERS,      uint32_t(0));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,        n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,      n_embd_head);
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,              10000.0f);
        // SWA pattern: every 5th layer is full attention (matches E2B layer_types)
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(5));
    } else if (arch == LLM_ARCH_COHERE2MOE || arch == LLM_ARCH_MIMO2 || arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_SPARK2_5 ||
            arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_GRANITE_SWA || arch == LLM_ARCH_DOTS3NOTE ||
            arch == LLM_ARCH_MAPLE) {
        std::vector<uint32_t> pattern;
        pattern.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern.push_back(il % 2);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    }

    // MSA requires one indexer head per GQA (KV) head, unlike the DSA archs where the
    // indexer head count is independent of the main attention head count.
    if (arch == LLM_ARCH_QWEN4EXP || arch == LLM_ARCH_GLM5_NEXT) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,    uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,  1.0e-6f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_LOW_RANK, uint32_t(8));
        // without this the QSA layers fall back to dense and go uncovered
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS, std::vector<uint32_t>(n_layer, 4));

        // has_cell_ext() needs ple_n_heads here: the indexer cache serializes no ext without it
        const uint32_t ple_ngram_size      = 3;
        const uint32_t ple_heads_per_ngram = 2;
        const uint32_t ple_n_heads         = (ple_ngram_size - 1)*ple_heads_per_ngram;
        GGML_ASSERT(n_embd % ple_n_heads == 0);
        const uint32_t ple_head_dim = n_embd/ple_n_heads;

        std::vector<uint64_t> ple_head_offsets(ple_n_heads);
        std::vector<uint64_t> ple_head_vocab_sizes(ple_n_heads, n_vocab);
        for (uint32_t h = 0; h < ple_n_heads; h++) {
            ple_head_offsets[h] = uint64_t(h)*n_vocab;
        }

        // the PLE history lives in the recurrent cache, so it must sit on a linear attention layer
        ms.add_kv(LLM_KV_PLE_LAYERS,                  std::vector<uint32_t>({ 0 }));
        ms.add_kv(LLM_KV_PLE_NGRAM_SIZE,              ple_ngram_size);
        ms.add_kv(LLM_KV_PLE_HEADS_PER_NGRAM,         ple_heads_per_ngram);
        ms.add_kv(LLM_KV_PLE_CONV_KERNEL,             uint32_t(4));
        ms.add_kv(LLM_KV_PLE_EOS_TOKEN_ID,            uint32_t(0));
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  ple_head_dim);
        ms.add_kv(LLM_KV_PLE_LAYER_MULTIPLIERS,       std::vector<uint64_t>({ 1, 3, 5 }));
        ms.add_kv(LLM_KV_PLE_HEAD_OFFSETS,            ple_head_offsets);
        ms.add_kv(LLM_KV_PLE_HEAD_VOCAB_SIZES,        ple_head_vocab_sizes);
    }

    // minimax-m3 and deepseek4 keep one indexer head per GQA head; the rest use a fixed 64 to match the fused
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,
              (arch == LLM_ARCH_MINIMAX_M3 || arch == LLM_ARCH_DEEPSEEK4) ? n_head : uint32_t(64));
    // qwen4exp ropes indexer keys with the main rotary width, so its head can't be < n_rot
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,
              arch == LLM_ARCH_QWEN4EXP ? n_embd_head : uint32_t(128));

    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,   uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL,        uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL_SELECT_TAIL, true);
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS, uint32_t(1));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));

    if (arch == LLM_ARCH_HY_V4) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,     uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,   1.0e-6f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_MAGNITUDE, 2.0f);
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,           10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,       1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,        true);
        // layer 0 must own an indexer, the odd layers share it
        std::vector<uint32_t> indexer_types;
        indexer_types.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            indexer_types.push_back(il % 2 ? 0 : 1);
        }
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, indexer_types);
    }

    if (arch == LLM_ARCH_DEEPSEEK4) {
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,         uint32_t(8));
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,           uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,            std::vector<uint32_t>({0, 0, 4, 128}));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,    160000.0f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,               uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,             1.0e-6f);
        ms.add_kv(LLM_KV_HASH_LAYER_COUNT,                      uint32_t(0));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,                      10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                  1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                   true);
    }

    if (arch == LLM_ARCH_HRM_TEXT) {
        // 8 cache slots alias 2 physical blocks: 1 low-stack layer + 1 high-stack layer
        ms.add_kv(LLM_KV_HRM_LAYERS_PER_STACK, uint32_t(1));
        ms.add_kv(LLM_KV_HRM_H_CYCLES,         uint32_t(2));
        ms.add_kv(LLM_KV_HRM_L_CYCLES,         uint32_t(3));
    }

    if (arch == LLM_ARCH_MAPLE) {
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP, 7.0f);
    }

    ms.add_kv(LLM_KV_TOKENIZER_MODEL,         "no_vocab");
    // ms.add_kv(LLM_KV_DENSE_2_FEAT_OUT,     n_embd);
    // ms.add_kv(LLM_KV_DENSE_3_FEAT_IN,      n_embd);

    if (moe) {
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
        ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, n_ff / 2);  // distinct from n_ff so a saver key-clobber surfaces on reload
        ms.add_kv(LLM_KV_EXPERT_LATENT_LENGTH,       n_ff);
        ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,  uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_COUNT,               uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,        uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,         arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(4) : uint32_t(2)); // sqrtsoftplus : sigmoid
        ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,         1.0f);
        ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,          uint32_t(1));
    }

    ms.add_kv(LLM_KV_POSNET_EMBEDDING_LENGTH,   n_embd);
    ms.add_kv(LLM_KV_POSNET_BLOCK_COUNT,        n_layer);
    ms.add_kv(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_CONVNEXT_BLOCK_COUNT,      n_layer);
    ms.add_kv(LLM_KV_XIELU_ALPHA_N,             1.0f);
    ms.add_kv(LLM_KV_XIELU_ALPHA_P,             1.0f);
    ms.add_kv(LLM_KV_XIELU_BETA,                1.0f);
    ms.add_kv(LLM_KV_XIELU_EPS,                 1.0e-7f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,            arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE || arch == LLM_ARCH_QWEN4EXP ? 256 : 2*n_embd);
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,           uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,            uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,        n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,           arch == LLM_ARCH_PLAMO2 ? 0 : uint32_t(2));
    ms.add_kv(LLM_KV_KDA_HEAD_DIM,              uint32_t(128));
    ms.add_kv(LLM_KV_KDA_SAFE_GATE,              true);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,       -5.0f);
    if (arch == LLM_ARCH_BAILINGMOE3) {
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,   std::vector<float>({0.0f, 4.0f}));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_SHEXP, std::vector<float>({0.0f, 5.0f}));
    }
    ms.add_kv(LLM_KV_WKV_HEAD_SIZE,             n_embd/n_head);
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,         uint32_t(3));
    ms.add_kv(LLM_KV_RESIDUAL_SCALE,            3.5565588200778455f);
    ms.add_kv(LLM_KV_ATTN_RES_BLOCK_SIZE,       uint32_t(12));
    ms.add_kv(LLM_KV_ACTIVATION_SITU_BETA,      4.0f);
    ms.add_kv(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, 25.0f);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,      -5.0f);

    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_tensor t;
        memset(&t, 0, sizeof(ggml_tensor));
        t.type = GGML_TYPE_F16;
        ggml_format_name(&t, "conv%" PRIu32 "d.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv1.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv2.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "convnext.%" PRIu32 ".dw.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
    }
    return ret;
}

static bool silent_model_load_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

static std::pair<llama_model_ptr, llama_context_ptr> get_model_and_ctx(
        struct gguf_context * gguf_ctx, FILE * file, const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        const llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER, bool encode = false,
        uint32_t n_seq_max = 1, bool kv_unified = false,
        ggml_backend_sched_eval_callback cb_eval = nullptr, void * cb_eval_user_data = nullptr,
        void (*init_tensor)(struct ggml_tensor *, void *) = set_tensor_data) {
    GGML_ASSERT((gguf_ctx == nullptr) != (file == nullptr));
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    std::vector<ggml_backend_dev_t> devs_copy = devs;
    devs_copy.push_back(nullptr);
    model_params.devices = devs_copy.data();
    model_params.split_mode = split_mode;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 0;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    ctx_params.n_seq_max = n_seq_max;
    ctx_params.kv_unified = kv_unified;
    ctx_params.cb_eval = cb_eval;
    ctx_params.cb_eval_user_data = cb_eval_user_data;
    if (!encode) {
        ctx_params.n_ubatch = 64;
    }

    size_t tmp = seed;
    llama_model_ptr model(gguf_ctx != nullptr ?
        llama_model_init_from_user(gguf_ctx, init_tensor, &tmp, model_params) :
        llama_model_load_from_file_ptr(file, model_params));
    if (!model) {
        throw std::runtime_error("failed to create llama model");
    }
    llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
    if (!lctx) {
        throw std::runtime_error("failed to create llama context");
    }
    return std::make_pair(std::move(model), std::move(lctx));
}

static std::vector<float> get_logits(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, bool encode = false) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_ctx    = llama_n_ctx(lctx);
    const uint32_t n_tokens = tokens.size();
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    GGML_ASSERT(n_tokens <= n_ctx);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        common_batch_add(batch, tokens[pos], pos, {0}, true);
    }
    batch.n_tokens = n_tokens;
    if (encode) {
        if (llama_encode(lctx, batch)) {
            llama_batch_free(batch);
            throw std::runtime_error("failed to encode batch");
        }
    }
    if (llama_decode(lctx, batch)) {
        llama_batch_free(batch);
        throw std::runtime_error("failed to decode batch");
    }

    std::vector<float> ret;
    ret.reserve(n_tokens*n_vocab);
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        for (uint32_t j = 0; j < n_vocab; j++) {
            ret.push_back(logits_ith[j]);
        }
    }
    llama_batch_free(batch);
    return ret;
}

static bool moe_mandatory(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_COHERE2MOE:
        case LLM_ARCH_GROK:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_QWEN3NEXT:
        case LLM_ARCH_QWEN3VLMOE:
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_QWEN4EXP:
        case LLM_ARCH_PHIMOE:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_DEEPSEEK32:
        case LLM_ARCH_DOTS3NOTE:
        case LLM_ARCH_DEEPSEEK4:
        case LLM_ARCH_GLM4_MOE:
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_EXAONE_MOE:
        case LLM_ARCH_BAILINGMOE:
        case LLM_ARCH_BAILINGMOE2:
        case LLM_ARCH_BAILINGMOE3:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_AFMOE:
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_HY_V3:
        case LLM_ARCH_HY_V4:
        case LLM_ARCH_OPENAI_MOE:
        case LLM_ARCH_LFM2MOE:
        case LLM_ARCH_SMALLTHINKER:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_GROVEMOE:
        case LLM_ARCH_MINIMAX_01:
        case LLM_ARCH_MINIMAX_M2:
        case LLM_ARCH_MINIMAX_M3:
        case LLM_ARCH_RND1:
        case LLM_ARCH_PADDLEOCR:
        case LLM_ARCH_MIMO2:
        case LLM_ARCH_KIMI_LINEAR:
        case LLM_ARCH_KIMI_K3:
        case LLM_ARCH_GLM5_NEXT:
        case LLM_ARCH_STEP35:
        case LLM_ARCH_MISTRAL4:
        case LLM_ARCH_MELLUM:
        case LLM_ARCH_LAGUNA:
        case LLM_ARCH_MAPLE:
            return true;
        default:
            return false;
    }
}

static bool moe_implemented(const llm_arch arch) {
    if (moe_mandatory(arch)) {
        return true;
    }
    switch (arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_MISTRAL3:
        case LLM_ARCH_LLAMA_EMBED:
            return true;
        default:
            return false;
    }
}

static bool arch_supported(const llm_arch arch) {
    if (arch == LLM_ARCH_CLIP || arch == LLM_ARCH_GPTJ || arch == LLM_ARCH_UNKNOWN) {
        return false; // These models don't have usable implementations.
    }
    if (arch == LLM_ARCH_CHAMELEON) {
        return false; // Only half-implemented and to be removed in the future.
    }
    if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {
        return false; // FIXME CUDA backend crashes.
    }
    if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        return false; // FIXME @ngxson
    }
    if (arch == LLM_ARCH_GRANITE_SWITCH) {
        return false; // FIXME adapter fixture
    }
    if (arch == LLM_ARCH_LLAMA_EMBED || arch == LLM_ARCH_GEMMA_EMBEDDING || arch == LLM_ARCH_T5ENCODER) {
        return false; // FIXME Embedding (?) models produce inconsistent results.
    }
    if (arch == LLM_ARCH_RWKV6 || arch == LLM_ARCH_RWKV6QWEN2 || arch == LLM_ARCH_RWKV7 || arch == LLM_ARCH_ARWKV7) {
        return false; // FIXME RWKV models hang indefinitely.
    }
    if (arch == LLM_ARCH_BERT || arch == LLM_ARCH_MODERN_BERT || arch == LLM_ARCH_NOMIC_BERT || arch == LLM_ARCH_NOMIC_BERT_MOE ||
            arch == LLM_ARCH_NEO_BERT || arch == LLM_ARCH_JINA_BERT_V2 || arch == LLM_ARCH_JINA_BERT_V3 || arch == LLM_ARCH_EUROBERT ||
            arch == LLM_ARCH_LAYA) {
        return false; // TODO vocab
    }
    if (arch == LLM_ARCH_PLM) {
        return false; // TODO tensor shapes
    }
    if (arch == LLM_ARCH_DEEPSEEK2OCR) {
        return false;
    }
    // FIXME: these hit scheduler/view-backed-output issues with WebGPU on CI.
#ifdef GGML_USE_WEBGPU
    if (arch == LLM_ARCH_DEEPSEEK32 || arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_GLM_DSA || arch == LLM_ARCH_DOTS3NOTE || arch == LLM_ARCH_QWEN4EXP ||
            arch == LLM_ARCH_HY_V4) {
        return false;
    }
#endif // GGML_USE_WEBGPU

    // FIXME: jamba produces incorrect output (~0.55 NMSE vs CPU) on the HIP
    // backend on RDNA3.5 (gfx1151); the SSM kernels need investigation.
#ifdef GGML_USE_HIP
    if (arch == LLM_ARCH_JAMBA) {
        return false;
    }
#endif // GGML_USE_HIP

    return true;
}

static int save_models(const llm_arch target_arch, const size_t seed, const int verbosity, const std::string & dir) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (target_arch != LLM_ARCH_UNKNOWN && arch != target_arch) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue; // FIXME: ISWA KV cache initialization needs more fixture params
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH) {
            continue;
        }
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            if (!llama_model_saver_supports_arch(arch) || !arch_supported(arch)) {
                LOG_INF("%s: %s model (%s) is unsupported, skipping\n", __func__, llm_arch_name(arch), moe ? "MoE" : "dense");
                continue;
            }
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            auto model_and_ctx = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {});
            const std::string path = dir + "/" + llm_arch_name(arch) + (moe ? "-moe.gguf" : "-dense.gguf");
            LOG_INF("%s: Saving %s model (%s) to %s...\n", __func__, llm_arch_name(arch), moe ? "MoE" : "dense", path.c_str());
            llama_model_save_to_file(model_and_ctx.first.get(), path.c_str());
        }
    }
    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return 0;
}

// A unified cache has one cell-to-block map for every sequence. Image patches can share a
// temporal position, so until QSA has sequence-local ranks this configuration must stay dense.
static int test_qsa_unified_multiseq(size_t seed) {
    struct observed_nodes {
        bool top_k = false;
    } observed;

    auto observe = [](ggml_tensor * tensor, bool ask, void * data) {
        if (ask) {
            auto & nodes = *static_cast<observed_nodes *>(data);
            nodes.top_k |= std::strncmp(tensor->name, "indexer_top_k-", 14) == 0;
        }
        return false;
    };

    auto gguf_sparse = get_gguf_ctx(LLM_ARCH_QWEN4EXP, true);
    auto gguf_dense  = get_gguf_ctx(LLM_ARCH_QWEN4EXP, true);

    const uint32_t dense_ratios[] = { 0, 0 };
    gguf_set_arr_data(gguf_dense.get(), "qwen4exp.attention.compress_ratios",
            GGUF_TYPE_UINT32, dense_ratios, 2);

    auto sparse = get_model_and_ctx(gguf_sparse.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER,
            false, 2, true, observe, &observed);
    auto dense = get_model_and_ctx(gguf_dense.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER,
            false, 2, true);

    constexpr uint32_t n_seq     = 2;
    constexpr uint32_t n_patches = 8;
    constexpr uint32_t n_tokens  = n_seq*n_patches;

    const uint32_t n_embd  = llama_model_n_embd(sparse.first.get());
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(sparse.first.get()));

    std::vector<float> embeddings(n_tokens*n_embd);
    std::mt19937 rng(seed);
    std::normal_distribution<float> distribution(0.0f, 0.1f);
    for (float & value : embeddings) {
        value = distribution(rng);
    }

    std::vector<llama_pos> positions(4*n_tokens, 0);
    std::vector<int32_t> n_seq_ids(n_tokens, 1);
    std::vector<llama_seq_id> seq_ids(n_tokens);
    std::vector<llama_seq_id *> seq_id_ptrs(n_tokens);
    std::vector<int8_t> logits(n_tokens, 1);

    for (uint32_t s = 0; s < n_seq; ++s) {
        for (uint32_t p = 0; p < n_patches; ++p) {
            const uint32_t i = s*n_patches + p;
            positions[i] = 0;
            positions[i + n_tokens] = p/4;
            positions[i + 2*n_tokens] = p%4;
            seq_ids[i] = (llama_seq_id) s;
            seq_id_ptrs[i] = &seq_ids[i];
        }
    }

    llama_batch image = {};
    image.n_tokens = n_tokens;
    image.embd = embeddings.data();
    image.pos = positions.data();
    image.n_seq_id = n_seq_ids.data();
    image.seq_id = seq_id_ptrs.data();
    image.logits = logits.data();

    auto decode = [n_vocab](llama_context * ctx, llama_batch batch) {
        GGML_ASSERT(llama_decode(ctx, batch) == 0);
        llama_synchronize(ctx);

        std::vector<float> result;
        result.reserve((size_t) batch.n_tokens*n_vocab);
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            const float * row = llama_get_logits_ith(ctx, i);
            GGML_ASSERT(row != nullptr);
            result.insert(result.end(), row, row + n_vocab);
        }
        return result;
    };

    const auto actual   = decode(sparse.second.get(), image);
    const auto expected = decode(dense.second.get(), image);

    GGML_ASSERT(!observed.top_k);
    GGML_ASSERT(nmse(actual, expected) < 1e-8);

    printf("QSA unified multi-sequence fallback test passed\n");
    return 0;
}

static int test_backends(const llm_arch target_arch, const size_t seed, const int verbosity) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    const std::vector<llama_token> tokens = get_tokens(128, 128, seed);

    struct device_config {
        std::vector<ggml_backend_dev_t> devs;
        std::string                     label;
        llama_split_mode                split_mode;

        device_config(std::vector<ggml_backend_dev_t> devs, std::string name, llama_split_mode split_mode)
            : devs(std::move(devs)), label(std::move(name)), split_mode(split_mode) {}
    };

    std::vector<device_config> dev_configs;
    size_t max_device_label_length = 4;
    {
        std::vector<ggml_backend_dev_t> devices_meta;
        {
            const size_t device_count = ggml_backend_dev_count();
            for (size_t i = 0; i < device_count; i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{dev}, ggml_backend_dev_description(dev), LLAMA_SPLIT_MODE_LAYER);
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());

                // cpu-based devices cannot be used in tensor split mode
                if (ggml_backend_dev_buffer_type(dev) != ggml_backend_cpu_buffer_type()) {
                    devices_meta.push_back(dev);
                }
            }
        }

        dev_configs.emplace_back(devices_meta, "Meta", LLAMA_SPLIT_MODE_TENSOR);
    }

    size_t max_arch_name_length = 0;
    for (const llm_arch & arch : llm_arch_all()) {
        max_arch_name_length = std::max(max_arch_name_length, strlen(llm_arch_name(arch)));
    }

    const std::string template_header  = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|%15s|%9s|\n";
    const std::string template_row_cfg = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|";
    const std::string template_row_res = "%15s %10s|%20s|\n";

    bool all_ok = true;
    common_log_flush(common_log_main());
    printf(template_header.c_str(), "Model arch.", "Device", "Config", "NMSE vs. CPU", "Roundtrip");
    printf("|");
    for (size_t i = 0; i < max_arch_name_length; i++) {
        printf("-");
    }
    printf("|");
    for (size_t i = 0; i < max_device_label_length; i++) {
        printf("-");
    }
    printf("|------|---------------|---------|\n");
    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (target_arch != LLM_ARCH_UNKNOWN && arch != target_arch) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue; // FIXME: ISWA KV cache initialization needs more fixture params
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH) {
            continue;
        }

        const bool encode = arch == LLM_ARCH_T5 || arch == LLM_ARCH_DREAM || arch == LLM_ARCH_LLADA || arch == LLM_ARCH_LLADA_MOE || arch == LLM_ARCH_RND1;
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            const std::string config_name = moe ? "MoE" : "Dense";
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            if (arch == LLM_ARCH_BAILINGMOE3) {
                GGML_ASSERT(gguf_remove_key(gguf_ctx.get(), "bailingmoe3.kda.safe_gate") >= 0);
            }
            std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_cpu;
            std::vector<float> logits_cpu;
            for (device_config & dc : dev_configs) {
                // print test config first; should anything fail during model loading or inference, at least we know which test case caused it
                printf(template_row_cfg.c_str(),
                    llm_arch_name(arch), dc.label.c_str(), config_name.c_str());
                fflush(stdout);

                std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_dev;
                std::vector<float> logits_dev;
                std::string status_nmse      = "\033[1;33mSKIP\033[0m";
                std::string status_roundtrip = "\033[1;33mSKIP\033[0m";
                char nmse_str[12] = {0};

                // GLM5 query and indexer head counts are not divisible by three.
                const bool unsupported_glm5_split =
                    arch == LLM_ARCH_GLM5_NEXT && dc.split_mode == LLAMA_SPLIT_MODE_TENSOR && dc.devs.size() == 3;
                bool skip = !arch_supported(arch) || (dc.split_mode == LLAMA_SPLIT_MODE_TENSOR && dc.devs.empty()) || unsupported_glm5_split;
                if (!skip) {
                    if (logits_cpu.empty()) {
                        model_and_ctx_cpu = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER, encode);
                        logits_cpu = get_logits(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens, encode);
                        if (arch == LLM_ARCH_DEEPSEEK4) {
                            GGML_ASSERT(llama_memory_seq_rm(
                                    llama_get_memory(model_and_ctx_cpu.second.get()), 0, -1, -1));
                        }
                    }
                    if (dc.split_mode != LLAMA_SPLIT_MODE_TENSOR || llm_arch_supports_sm_tensor(arch)) {
                        model_and_ctx_dev = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, dc.devs, dc.split_mode, encode);
                        logits_dev = get_logits(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens, encode);
                        if (arch == LLM_ARCH_DEEPSEEK4) {
                            GGML_ASSERT(llama_memory_seq_rm(
                                    llama_get_memory(model_and_ctx_dev.second.get()), 0, -1, -1));
                        }
                        const double nmse_val = nmse(logits_cpu, logits_dev);
                        snprintf(nmse_str, sizeof(nmse_str), "(%.2e)", nmse_val);
                        status_nmse = "\033[1;32mOK\033[0m";
                        if (nmse_val > 1e-4) {
                            all_ok = false;
                            status_nmse = "\033[1;31mFAIL\033[0m";
                        }
                    }

                    FILE * file = tmpfile(); // Can be null on Windows without administrator privileges.
                    // FIXME: when adding a tensor to a gguf_context a copy is made, this changes the pointer which the meta backend
                    //     in turn uses to map the tensors to their simple equivalents - this is fundamentally incompatible
                    // FIXME: DSV4 metadata is not implemented by llama_model_saver.
                    const bool can_roundtrip = llama_model_saver_supports_arch(arch) && arch != LLM_ARCH_DEEPSEEK4;
                    if (file != nullptr && can_roundtrip && dc.split_mode != LLAMA_SPLIT_MODE_TENSOR) {
                        GGML_ASSERT(model_and_ctx_dev.first && model_and_ctx_dev.second);
                        llama_model_saver ms = llama_model_saver(model_and_ctx_dev.first.get());
                        ms.add_kv_from_model();
                        ms.add_tensors_from_model();
                        ms.save(file);
                        rewind(file);

                        auto model_and_ctx_roundtrip = get_model_and_ctx(nullptr, file, seed, dc.devs, dc.split_mode, encode);
                        const std::vector<float> logits_roundtrip = get_logits(
                            model_and_ctx_roundtrip.first.get(), model_and_ctx_roundtrip.second.get(), tokens, encode);
                        status_roundtrip = "\033[1;32mOK\033[0m";
                        GGML_ASSERT(logits_roundtrip.size() == logits_dev.size());
                        for (size_t i = 0; i < logits_roundtrip.size(); i++) {
                            if (logits_roundtrip[i] != logits_dev[i]) {
                                all_ok = false;
                                status_roundtrip = "\033[1;31mFAIL\033[0m";
                                break;
                            }
                        }
                    }
                }

                // log the results for this test case
                printf(template_row_res.c_str(),
                    status_nmse.c_str(), nmse_str, status_roundtrip.c_str());
            }
        }
    }
    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return all_ok ? 0 : 1;
}

struct mtp_backend_probe {
    ggml_backend_i iface;
    bool pending = false;
    bool fail = false;
    std::set<ggml_backend_buffer_t> buffers;
};

static std::map<ggml_backend_t, mtp_backend_probe> mtp_probes;

static ggml_status mtp_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto & probe = mtp_probes.at(backend);
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto * tensor = ggml_graph_node(graph, i);
        if (tensor->buffer && ggml_backend_buffer_get_usage(tensor->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
            probe.buffers.insert(tensor->buffer);
        }
    }
    auto status = probe.iface.graph_compute(backend, graph);
    probe.pending = true;
    if (probe.fail) {
        probe.fail = false;
        return GGML_STATUS_FAILED; // Fail after submitting work to exercise error-path synchronization.
    }
    return status;
}

static void mtp_synchronize(ggml_backend_t backend) {
    auto & probe = mtp_probes.at(backend);
    if (probe.iface.synchronize) {
        probe.iface.synchronize(backend);
    }
    probe.pending = false;
}

static int test_mtp_shared(bool cpu) {
    // A CPU helper with ACCEL classification exercises the same device gate as BLAS/Accelerate.
    static ggml_backend_device helper = *ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    helper.iface.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_ACCEL; };
    helper.iface.init_backend = [](ggml_backend_dev_t dev, const char * params) {
        auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        auto backend = cpu->iface.init_backend(cpu, params);
        backend->device = dev;
        return backend;
    };
    helper.iface.supports_op = [](ggml_backend_dev_t, const ggml_tensor *) { return false; };
    ggml_backend_device_register(&helper);
    auto metadata = get_gguf_ctx(LLM_ARCH_QWEN35, false);
    gguf_set_val_u32(metadata.get(), "qwen35.block_count", 3);
    gguf_set_val_u32(metadata.get(), "qwen35.nextn_predict_layers", 1);
    auto mp = llama_model_default_params();
    static ggml_backend_device compute = *ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    compute.iface.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; };
    compute.iface.init_backend = helper.iface.init_backend;
    ggml_backend_dev_t devices[] = {&compute, nullptr};
    if (cpu) {
        mp.devices = devices; // Exercise sharing and source destruction under CPU sanitizers too.
    }
    mp.n_gpu_layers = 999;
    mp.load_mtp = true;
    size_t seed = 1234;
    llama_model_ptr model(llama_model_init_from_user(metadata.get(), set_tensor_data, &seed, mp));
    GGML_ASSERT(model);
    std::vector<float> reference;
    for (bool share : {false, true}) {
        auto cp = llama_context_default_params();
        GGML_ASSERT(!cp.ctx_other_share_compute);
        cp.n_ctx = 512;
        cp.n_batch = cp.n_ubatch = 32;
        cp.n_outputs_max = cp.n_outputs_max_per_seq = 32;
        cp.n_seq_max = 1;
        cp.n_threads = cp.n_threads_batch = 2;
        cp.no_perf = false;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        llama_context_ptr target(llama_init_from_model(model.get(), cp));
        GGML_ASSERT(target);
        cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        cp.ctx_other = target.get();
        cp.ctx_other_share_compute = share;
        llama_context_ptr draft(llama_init_from_model(model.get(), cp));
        GGML_ASSERT(draft);
        common_params_speculative params;
        params.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
        params.draft.ctx_tgt = target.get();
        params.draft.ctx_dft = draft.get();
        params.draft.backend_sampling = false;
        common_speculative_ptr spec(common_speculative_init(params, 1));
        GGML_ASSERT(spec);
        auto install = [](llama_context * ctx) {
            auto sched = ctx->get_sched();
            for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
                auto backend = ggml_backend_sched_get_backend(sched, i);
                mtp_probes[backend].iface = backend->iface;
                backend->iface.graph_compute = mtp_graph_compute;
                backend->iface.synchronize = mtp_synchronize;
            }
        };
        install(target.get());
        install(draft.get());
        auto finished = [](llama_context * ctx) {
            auto sched = ctx->get_sched();
            for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
                GGML_ASSERT(!mtp_probes.at(ggml_backend_sched_get_backend(sched, i)).pending);
            }
        };
        auto buffers = [](llama_context * ctx) {
            std::set<ggml_backend_buffer_t> result;
            auto sched = ctx->get_sched();
            for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
                const auto & probe = mtp_probes.at(ggml_backend_sched_get_backend(sched, i));
                result.insert(probe.buffers.begin(), probe.buffers.end());
            }
            return result;
        };
        llama_batch batch = llama_batch_init(32, 0, 1);
        std::vector<float> logits;
        llama_sampler_ptr sampler(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
        common_speculative_begin(spec.get(), 0, {});
        for (int pass = 0; pass < 8; ++pass) {
            if (pass == 3 || pass == 5) {
                GGML_ASSERT(llama_set_sampler(target.get(), 0, pass == 3 ? sampler.get() : nullptr));
                for (auto & entry : mtp_probes) {
                    entry.second.buffers.clear();
                }
            }
            common_batch_clear(batch);
            for (int i = 0; i < 32; ++i) {
                common_batch_add(batch, (i + pass) % 128, pass * 32 + i, {0}, true);
            }
            GGML_ASSERT(llama_decode(target.get(), batch) == 0);
            const auto * output = llama_get_logits(target.get());
            logits.insert(logits.end(), output, output + 128 * 32);
            if (pass == 6) {
                auto backend = ggml_backend_sched_get_backend(draft->get_sched(), 0);
                mtp_probes.at(backend).fail = true;
            }
            GGML_ASSERT(common_speculative_process(spec.get(), batch) == (pass != 6));
            finished(draft.get());
            if (pass == 2 || pass == 4 || pass == 5) {
                auto tgt = buffers(target.get());
                auto dft = buffers(draft.get());
                bool aliases = false;
                for (auto buffer : tgt) {
                    aliases |= dft.count(buffer) != 0;
                }
                auto backend = ggml_backend_sched_get_backend(target->get_sched(), 0);
                const auto type = ggml_backend_dev_type(ggml_backend_get_device(backend));
                if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                    GGML_ASSERT(aliases == share);
                }
            }
        }
        std::vector<uint8_t> state;
        GGML_ASSERT(common_speculative_get_state(spec.get(), 0, state));
        GGML_ASSERT(state.size() == sizeof(llama_pos) + (size_t) llama_model_n_embd_out(model.get()) * sizeof(float));
        llama_pos state_pos = -1;
        std::memcpy(&state_pos, state.data(), sizeof(state_pos));
        GGML_ASSERT(state_pos == 255);

        const auto saved_state = state;
        common_speculative_set_state(spec.get(), 0, { 0 });
        GGML_ASSERT(common_speculative_get_state(spec.get(), 0, state));
        GGML_ASSERT(state == saved_state);

        auto            invalid_pos_state = saved_state;
        const llama_pos invalid_pos       = -1;
        std::memcpy(invalid_pos_state.data(), &invalid_pos, sizeof(invalid_pos));
        common_speculative_set_state(spec.get(), 0, invalid_pos_state);
        GGML_ASSERT(common_speculative_get_state(spec.get(), 0, state));
        GGML_ASSERT(state == saved_state);

        common_speculative_set_state(spec.get(), 0, {});
        GGML_ASSERT(!common_speculative_get_state(spec.get(), 0, state));
        common_speculative_set_state(spec.get(), 0, saved_state);
        GGML_ASSERT(common_speculative_get_state(spec.get(), 0, state));
        GGML_ASSERT(state == saved_state);

        GGML_ASSERT(llama_perf_context(target.get()).n_reused > 0);
        GGML_ASSERT(llama_perf_context(draft.get()).n_reused > 0);
        if (share) {
            GGML_ASSERT(nmse(reference, logits) < 1e-8);
        } else {
            reference = logits;
        }
        llama_batch_free(batch);
        spec.reset();
        target.reset();
        llama_set_embeddings(draft.get(), true);
        draft->sched_reserve(); // The source is gone; the draft must reserve independently.
        draft.reset();
        mtp_probes.clear();
    }
    printf("MTP shared compute: passed\n");
    return 0;
}

static void add_hadamard_metadata(gguf_context * ctx, uint32_t block_size = 256, bool include_attn_q = false) {
    const char * weight_names[] = { "output.weight", "blk.0.attn_q.weight" };
    const char * inverse_names[] = { "token_embd.weight" };
    gguf_set_val_u32(ctx, "prism.hadamard.version", 1);
    gguf_set_val_u32(ctx, "prism.hadamard.block_size", block_size);
    gguf_set_val_str(ctx, "prism.hadamard.transform", "normalized-sylvester-walsh-hadamard");
    gguf_set_val_str(ctx, "prism.hadamard.axis", "input-last-dimension");
    gguf_set_val_str(ctx, "prism.hadamard.sign_mode", "identity");
    gguf_set_arr_str(ctx, "prism.hadamard.weight_names", weight_names, include_attn_q ? 2 : 1);
    gguf_set_arr_str(ctx, "prism.hadamard.inverse_weight_names", inverse_names, 1);
}

static std::string save_model_to_string(llama_model_saver & saver) {
    FILE * file = tmpfile();
    GGML_ASSERT(file);
    saver.save(file);
    GGML_ASSERT(fseek(file, 0, SEEK_END) == 0);
    const long size = ftell(file);
    GGML_ASSERT(size >= 0);
    rewind(file);
    std::string data((size_t) size, '\0');
    GGML_ASSERT(fread(data.data(), 1, data.size(), file) == data.size());
    fclose(file);
    return data;
}

static void test_hadamard_split_futures(bool with_hadamard) {
    const size_t seed = 1234;
    auto metadata = get_gguf_ctx(LLM_ARCH_LLAMA, false);
    auto source = get_model_and_ctx(metadata.get(), nullptr, seed, {});

    llama_model_saver source_saver(source.first.get());
    source_saver.add_kv_from_model();
    source_saver.add_tensors_from_model();

    const int64_t n_tensors = gguf_get_n_tensors(source_saver.gguf_ctx);
    std::vector<std::string> split_data;
    std::string tensor_list;
    bool found_token_embd = false;
    bool found_output = false;

    for (int split = 0; split < 2; ++split) {
        gguf_context_ptr split_ctx(gguf_init_empty());
        gguf_set_kv(split_ctx.get(), source_saver.gguf_ctx);
        gguf_set_val_u16(split_ctx.get(), "split.no", split);
        gguf_set_val_u16(split_ctx.get(), "split.count", 2);
        gguf_set_val_i32(split_ctx.get(), "split.tensors.count", (int32_t) n_tensors);
        if (with_hadamard) {
            add_hadamard_metadata(split_ctx.get());
        }
        llama_model_saver split_saver(LLM_ARCH_LLAMA, split_ctx.get());

        for (int64_t i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(source_saver.gguf_ctx, i);
            if (split == 0) {
                tensor_list += name;
                tensor_list += '\n';
            }
            const bool is_token_embd = strcmp(name, "token_embd.weight") == 0;
            const bool is_output = strcmp(name, "output.weight") == 0;
            found_token_embd |= is_token_embd;
            found_output |= is_output;
            if ((split == 0) != is_token_embd) {
                continue;
            }
            const ggml_tensor * tensor = source.first->get_tensor(name);
            GGML_ASSERT(tensor);
            split_saver.add_tensor(tensor);
        }
        split_data.push_back(save_model_to_string(split_saver));
    }
    GGML_ASSERT(found_token_embd && found_output);

    const char * paths[] = {
        "hadamard-split-00001-of-00002.gguf",
        "hadamard-split-00002-of-00002.gguf",
    };
    const char * tensor_list_path = "hadamard-split.tensors.txt";
    const char * context = with_hadamard ? "hadamard-split-test" : "plain-split-test";
    std::thread fulfill_thread([&]() {
        std::vector<uint8_t> list_data(tensor_list.begin(), tensor_list.end());
        auto list_buf = std::make_unique<Uint8BufferStreamBuf>(std::move(list_data));
        GGML_ASSERT(llama_model_load_fulfill_split_future(tensor_list_path, context, std::move(list_buf)));
        for (size_t i = 0; i < 2; ++i) {
            std::vector<uint8_t> data(split_data[i].begin(), split_data[i].end());
            auto split_buf = std::make_unique<Uint8BufferStreamBuf>(std::move(data));
            GGML_ASSERT(llama_model_load_fulfill_split_future(paths[i], context, std::move(split_buf)));
        }
    });

    auto params = llama_model_default_params();
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    llama_model * loaded = llama_model_load_from_split_futures(paths, 2, context, tensor_list_path, params);
    fulfill_thread.join();
    GGML_ASSERT(loaded);
    GGML_ASSERT(loaded->hadamard_rotations.size() == (with_hadamard ? 1 : 0));
    llama_model_free(loaded);
}

static void test_hadamard_tied_output() {
    const size_t seed = 1234;
    auto metadata = get_gguf_ctx(LLM_ARCH_QWEN35, false);
    auto source = get_model_and_ctx(metadata.get(), nullptr, seed, {});

    llama_model_saver source_saver(source.first.get());
    source_saver.add_kv_from_model();
    source_saver.add_tensors_from_model();

    gguf_context_ptr fixture_ctx(gguf_init_empty());
    gguf_set_kv(fixture_ctx.get(), source_saver.gguf_ctx);
    add_hadamard_metadata(fixture_ctx.get());
    llama_model_saver fixture(LLM_ARCH_QWEN35, fixture_ctx.get());

    bool skipped_output = false;
    for (int64_t i = 0; i < gguf_get_n_tensors(source_saver.gguf_ctx); ++i) {
        const char * name = gguf_get_tensor_name(source_saver.gguf_ctx, i);
        if (strcmp(name, "output.weight") == 0) {
            skipped_output = true;
            continue;
        }
        const ggml_tensor * tensor = source.first->get_tensor(name);
        GGML_ASSERT(tensor);
        fixture.add_tensor(tensor);
    }
    GGML_ASSERT(skipped_output);

    FILE * file = tmpfile();
    GGML_ASSERT(file);
    fixture.save(file);
    rewind(file);

    auto loaded = get_model_and_ctx(nullptr, file, seed, {});
    const auto tokens = get_tokens(4, llama_vocab_n_tokens(llama_model_get_vocab(loaded.first.get())), seed);
    GGML_ASSERT(!get_logits(loaded.first.get(), loaded.second.get(), tokens).empty());
    fclose(file);
}

static void test_hadamard_repack() {
    const size_t seed = 1234;
    auto metadata = get_gguf_ctx(LLM_ARCH_LLAMA, false);
    auto source = get_model_and_ctx(metadata.get(), nullptr, seed, {});

    llama_model_saver source_saver(source.first.get());
    source_saver.add_kv_from_model();
    source_saver.add_tensors_from_model();

    gguf_context_ptr fixture_ctx(gguf_init_empty());
    gguf_set_kv(fixture_ctx.get(), source_saver.gguf_ctx);
    add_hadamard_metadata(fixture_ctx.get(), 256, true);
    llama_model_saver fixture(LLM_ARCH_LLAMA, fixture_ctx.get());
    std::vector<ggml_context_ptr> quantized_contexts;

    for (int64_t i = 0; i < gguf_get_n_tensors(source_saver.gguf_ctx); ++i) {
        const char * name = gguf_get_tensor_name(source_saver.gguf_ctx, i);
        const ggml_tensor * tensor = source.first->get_tensor(name);
        GGML_ASSERT(tensor);
        if (strcmp(name, "output.weight") != 0 && strcmp(name, "blk.0.attn_q.weight") != 0) {
            fixture.add_tensor(tensor);
            continue;
        }

        ggml_init_params params = { 16 * 1024 * 1024, nullptr, false };
        ggml_context_ptr ctx(ggml_init(params));
        GGML_ASSERT(ctx);
        ggml_tensor * quantized = ggml_new_tensor_4d(
                ctx.get(), GGML_TYPE_Q8_0, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
        ggml_set_name(quantized, name);

        std::vector<float> data(ggml_nelements(tensor));
        if (tensor->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(tensor, data.data(), 0, ggml_nbytes(tensor));
        } else {
            GGML_ASSERT(tensor->type == GGML_TYPE_F16);
            std::vector<ggml_fp16_t> data_f16(ggml_nelements(tensor));
            ggml_backend_tensor_get(tensor, data_f16.data(), 0, ggml_nbytes(tensor));
            ggml_fp16_to_fp32_row(data_f16.data(), data.data(), data.size());
        }
        ggml_quantize_chunk(GGML_TYPE_Q8_0, data.data(), quantized->data, 0, ggml_nrows(tensor), tensor->ne[0], nullptr);
        fixture.add_tensor(quantized);
        quantized_contexts.emplace_back(std::move(ctx));
    }

    FILE * file = tmpfile();
    GGML_ASSERT(file);
    fixture.save(file);
    rewind(file);

    auto loaded = get_model_and_ctx(nullptr, file, seed, {});
    const auto cpu_buft = ggml_backend_dev_buffer_type(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
    bool tested_extra_buft = false;
    for (const char * name : { "output.weight", "blk.0.attn_q.weight" }) {
        const ggml_tensor * weight = loaded.first->get_tensor(name);
        GGML_ASSERT(weight);
        const auto it = loaded.first->hadamard_rotations.find(weight);
        GGML_ASSERT(it != loaded.first->hadamard_rotations.end());
        if (ggml_backend_buffer_get_type(weight->buffer) != cpu_buft) {
            GGML_ASSERT(ggml_backend_buffer_get_type(it->second.rot->buffer) == cpu_buft);
            tested_extra_buft = true;
        }
    }
    GGML_ASSERT(tested_extra_buft);
    const auto tokens = get_tokens(4, llama_vocab_n_tokens(llama_model_get_vocab(loaded.first.get())), seed);
    GGML_ASSERT(!get_logits(loaded.first.get(), loaded.second.get(), tokens).empty());
    fclose(file);
}

static void test_hadamard_invalid_block() {
    auto metadata = get_gguf_ctx(LLM_ARCH_QWEN35, false);
    add_hadamard_metadata(metadata.get(), 16384);
    auto params = llama_model_default_params();
    size_t seed = 1234;
    llama_model_ptr model(llama_model_init_from_user(metadata.get(), set_tensor_data, &seed, params));
    GGML_ASSERT(!model);
}

static void test_hadamard_mtp_arch(llm_arch arch, bool moe) {
    auto metadata = get_gguf_ctx(arch, moe);
    const std::string prefix = llm_arch_name(arch);
    gguf_set_val_u32(metadata.get(), (prefix + ".block_count").c_str(), 3);
    gguf_set_val_u32(metadata.get(), (prefix + ".nextn_predict_layers").c_str(), 1);
    add_hadamard_metadata(metadata.get());

    auto mp = llama_model_default_params();
    mp.load_mtp = true;
    size_t seed = 1234;
    llama_model_ptr model(llama_model_init_from_user(metadata.get(), set_tensor_data, &seed, mp));
    GGML_ASSERT(model);

    auto cp = llama_context_default_params();
    cp.n_ctx = 128;
    cp.n_batch = cp.n_ubatch = 8;
    cp.n_outputs_max = cp.n_outputs_max_per_seq = 8;
    cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 2;
    llama_context_ptr target(llama_init_from_model(model.get(), cp));
    GGML_ASSERT(target);

    cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cp.ctx_other = target.get();
    llama_context_ptr draft(llama_init_from_model(model.get(), cp));
    GGML_ASSERT(draft);

    common_params_speculative params;
    params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    params.draft.ctx_tgt = target.get();
    params.draft.ctx_dft = draft.get();
    params.draft.backend_sampling = false;
    common_speculative_ptr spec(common_speculative_init(params, 1));
    GGML_ASSERT(spec);

    llama_batch batch = llama_batch_init(8, 0, 1);
    for (int i = 0; i < 8; ++i) {
        common_batch_add(batch, i, i, { 0 }, true);
    }
    common_speculative_begin(spec.get(), 0, {});
    GGML_ASSERT(llama_decode(target.get(), batch) == 0);
    GGML_ASSERT(common_speculative_process(spec.get(), batch));
    llama_batch_free(batch);
}

// normalized Sylvester Walsh-Hadamard entry, the transform prism.hadamard folds with
static float hadamard_entry(int64_t row, int64_t col, int64_t n) {
    const float scale = 1.0f / std::sqrt(float(n));
    return std::bitset<64>(uint64_t(row & col)).count() % 2 ? -scale : scale;
}

// H is symmetric and H * H = I: one H folds a head row, and one H stores a latent embedding row
static void fold_row(float * row, int64_t n) {
    std::vector<float> folded(n, 0.0f);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            folded[i] += hadamard_entry(i, j, n) * row[j];
        }
    }
    std::copy(folded.begin(), folded.end(), row);
}

static void fold_rows(std::vector<float> & rows, int64_t n) {
    for (size_t r = 0; r < rows.size(); r += n) {
        fold_row(rows.data() + r, n);
    }
}

static std::vector<float> get_tensor_f32(const ggml_tensor * tensor) {
    std::vector<float> data(ggml_nelements(tensor));
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, data.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ASSERT(tensor->type == GGML_TYPE_F16);
        std::vector<ggml_fp16_t> data_f16(data.size());
        ggml_backend_tensor_get(tensor, data_f16.data(), 0, ggml_nbytes(tensor));
        ggml_fp16_to_fp32_row(data_f16.data(), data.data(), data.size());
    }
    return data;
}

// a copy of the target with prism.hadamard metadata and its token_embd, output and extra_folded rows folded, which leaves
// its logits unchanged; the extra weights must read n_embd-wide inputs
static FILE * save_folded_target(llm_arch arch, const llama_model * model, const std::vector<std::string> & extra_folded = {}) {
    llama_model_saver source_saver(model);
    source_saver.add_kv_from_model();
    source_saver.add_tensors_from_model();

    gguf_context_ptr fixture_ctx(gguf_init_empty());
    gguf_set_kv(fixture_ctx.get(), source_saver.gguf_ctx);
    add_hadamard_metadata(fixture_ctx.get(), model->hparams.n_embd);
    std::vector<const char *> weight_names = { "output.weight" };
    for (const std::string & name : extra_folded) {
        weight_names.push_back(name.c_str());
    }
    gguf_set_arr_str(fixture_ctx.get(), "prism.hadamard.weight_names", weight_names.data(), weight_names.size());
    llama_model_saver fixture(arch, fixture_ctx.get());

    ggml_init_params params = { 16 * 1024 * 1024, nullptr, false };
    ggml_context_ptr folded_ctx(ggml_init(params));
    GGML_ASSERT(folded_ctx);
    for (int64_t i = 0; i < gguf_get_n_tensors(source_saver.gguf_ctx); ++i) {
        const char * name = gguf_get_tensor_name(source_saver.gguf_ctx, i);
        const ggml_tensor * tensor = model->get_tensor(name);
        GGML_ASSERT(tensor);
        const bool folded_weight = strcmp(name, "token_embd.weight") == 0 || strcmp(name, "output.weight") == 0 ||
            std::find(extra_folded.begin(), extra_folded.end(), name) != extra_folded.end();
        if (!folded_weight) {
            fixture.add_tensor(tensor);
            continue;
        }
        std::vector<float> rows = get_tensor_f32(tensor);
        fold_rows(rows, tensor->ne[0]);
        ggml_tensor * folded = ggml_new_tensor_2d(folded_ctx.get(), GGML_TYPE_F32, tensor->ne[0], tensor->ne[1]);
        ggml_set_name(folded, name);
        memcpy(folded->data, rows.data(), ggml_nbytes(folded));
        fixture.add_tensor(folded);
    }

    FILE * file = tmpfile();
    GGML_ASSERT(file);
    fixture.save(file);
    rewind(file);
    return file;
}

// a DFlash drafter as loaded from a file without token_embd and output, so it reads the target's through ctx_other
static llama_model_ptr get_borrowing_dflash(size_t seed) {
    auto metadata = get_gguf_ctx(LLM_ARCH_DFLASH, false);
    // DFlash reads the window pattern as a per-layer array; keep the drafter full-attention
    gguf_remove_key(metadata.get(), "dflash.attention.sliding_window");
    const int32_t target_layers[] = { 1 };
    gguf_set_arr_data(metadata.get(), "dflash.target_layers", GGUF_TYPE_INT32, target_layers, 1);
    llama_model_ptr draft(llama_model_init_from_user(metadata.get(), set_tensor_data, &seed, llama_model_default_params()));
    GGML_ASSERT(draft);
    // user-initialized models create optional tensors too
    draft->tok_embd = nullptr;
    draft->output = nullptr;
    return draft;
}

static std::vector<float> get_borrowed_draft_logits(llama_model * draft, llama_context * target, const std::vector<llama_token> & tokens) {
    auto cp = llama_context_default_params();
    cp.n_ctx = 128;
    cp.n_batch = cp.n_ubatch = 8;
    cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 2;
    cp.ctx_other = target;
    llama_context_ptr ctx(llama_init_from_model(draft, cp));
    GGML_ASSERT(ctx);
    llama_set_causal_attn(ctx.get(), false);
    return get_logits(draft, ctx.get(), tokens);
}

// a drafter that borrows a Hadamard-folded target's embeddings and head drafts the same logits as with the plain target
static void test_hadamard_dflash_borrowed_io() {
    const size_t seed = 1234;
    auto metadata = get_gguf_ctx(LLM_ARCH_QWEN35, false);
    auto plain = get_model_and_ctx(metadata.get(), nullptr, seed, {});
    FILE * target_file = save_folded_target(LLM_ARCH_QWEN35, plain.first.get());
    auto folded = get_model_and_ctx(nullptr, target_file, seed, {});
    GGML_ASSERT(!folded.first->hadamard_rotations.empty() && !folded.first->hadamard_inverses.empty());

    const auto tokens = get_tokens(4, llama_vocab_n_tokens(llama_model_get_vocab(plain.first.get())), seed);
    GGML_ASSERT(nmse(get_logits(plain.first.get(), plain.second.get(), tokens),
                     get_logits(folded.first.get(), folded.second.get(), tokens)) < 1e-6);

    llama_model_ptr draft = get_borrowing_dflash(seed);
    const auto from_plain  = get_borrowed_draft_logits(draft.get(), plain.second.get(), tokens);
    const auto from_folded = get_borrowed_draft_logits(draft.get(), folded.second.get(), tokens);
    GGML_ASSERT(nmse(from_plain, from_folded) < 1e-6);
    fclose(target_file);
}

static bool count_hadamard_transforms(ggml_tensor * t, bool ask, void * user_data) {
    if (ask && t->op == GGML_OP_MUL_MAT && ((const int32_t *) t->op_params)[1] == GGML_HINT_SRC0_IS_HADAMARD) {
        ++*(int *) user_data;
    }
    return false;
}

// folded weights that read the same activation share one activation transform: ffn_gate and ffn_up of every layer,
// plus the output head and the token_embd lookup
static void test_hadamard_shared_input() {
    const size_t seed = 1234;
    auto metadata = get_gguf_ctx(LLM_ARCH_QWEN35, false);
    auto plain = get_model_and_ctx(metadata.get(), nullptr, seed, {});
    const int n_layer = plain.first->hparams.n_layer();

    std::vector<std::string> extra_folded;
    for (int il = 0; il < n_layer; ++il) {
        extra_folded.push_back("blk." + std::to_string(il) + ".ffn_gate.weight");
        extra_folded.push_back("blk." + std::to_string(il) + ".ffn_up.weight");
    }
    FILE * target_file = save_folded_target(LLM_ARCH_QWEN35, plain.first.get(), extra_folded);

    int n_transforms = 0;
    auto folded = get_model_and_ctx(nullptr, target_file, seed, {}, LLAMA_SPLIT_MODE_LAYER, false, 1, false,
                                    count_hadamard_transforms, &n_transforms);
    GGML_ASSERT(folded.first->hadamard_rotations.size() == extra_folded.size() + 1);

    const auto tokens = get_tokens(4, llama_vocab_n_tokens(llama_model_get_vocab(plain.first.get())), seed);
    GGML_ASSERT(nmse(get_logits(plain.first.get(), plain.second.get(), tokens),
                     get_logits(folded.first.get(), folded.second.get(), tokens)) < 1e-6);
    GGML_ASSERT(n_transforms == n_layer + 2);
    fclose(target_file);
}

static int test_hadamard_contracts() {
    test_hadamard_invalid_block();
    test_hadamard_tied_output();
    test_hadamard_split_futures(false);
    test_hadamard_split_futures(true);
    test_hadamard_mtp_arch(LLM_ARCH_QWEN35, false);
    test_hadamard_mtp_arch(LLM_ARCH_QWEN35MOE, true);
    test_hadamard_mtp_arch(LLM_ARCH_QWEN3NEXT, true);
    test_hadamard_repack();
    test_hadamard_dflash_borrowed_io();
    test_hadamard_shared_input();
    printf("Hadamard contracts: passed\n");
    return 0;
}

static int test_glm5_kpool_sequences() {
    struct selected_cells {
        std::vector<int32_t> indices;
    } selected;

    auto observe = [](ggml_tensor * tensor, bool ask, void * data) {
        if (std::strncmp(tensor->name, "indexer_sel_idx-0", 17) != 0) {
            return false;
        }
        if (!ask) {
            auto & indices = static_cast<selected_cells *>(data)->indices;
            indices.resize(ggml_nelements(tensor));
            ggml_backend_tensor_get(tensor, indices.data(), 0, ggml_nbytes(tensor));
        }
        return true;
    };
    auto metadata = get_gguf_ctx(LLM_ARCH_GLM5_NEXT, true);
    gguf_set_val_u32(metadata.get(), "glm5-next.attention.indexer.top_k", 12);
    auto loaded = get_model_and_ctx(metadata.get(), nullptr, 1234, {}, LLAMA_SPLIT_MODE_LAYER,
            false, 2, true, observe, &selected);
    auto * ctx = loaded.second.get();

    struct test_token {
        llama_token token;
        llama_pos pos;
        std::vector<llama_seq_id> seq_ids;
    };
    auto decode = [ctx](const std::vector<test_token> & entries) {
        llama_batch batch = llama_batch_init(entries.size(), 0, 2);
        for (size_t i = 0; i < entries.size(); ++i) {
            common_batch_add(batch, entries[i].token, entries[i].pos, entries[i].seq_ids, i + 1 == entries.size());
        }
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        GGML_ASSERT(rc == 0);
    };
    auto unique_count = [&]() {
        GGML_ASSERT(!selected.indices.empty());
        return std::set<int32_t>(selected.indices.begin(), selected.indices.end()).size();
    };

    decode({{2, 0, {0}}, {3, 1, {0}}, {4, 2, {0}}, {5, 3, {0}}});
    llama_memory_seq_add(llama_get_memory(ctx), 0, 2, 4, 1); // live positions 0, 1, 3, 4
    selected.indices.clear();
    decode({{6, 5, {0}}});
    GGML_ASSERT(llama_memory_seq_token_count(llama_get_memory(ctx), 0) == 5);
    GGML_ASSERT(unique_count() >= 5); // positions 0, 1, 3, 4, 5 must all be selected

    llama_memory_clear(llama_get_memory(ctx), true);
    decode({{10, 0, {0, 1}}, {11, 1, {0, 1}}, {12, 2, {0, 1}}, {13, 3, {0, 1}}});
    decode({{20, 4, {0}}, {21, 5, {0}}, {22, 6, {0}}, {23, 7, {0}}});
    decode({{30, 4, {1}}, {31, 5, {1}}, {32, 6, {1}}, {33, 7, {1}}});
    selected.indices.clear();
    decode({{40, 8, {0, 1}}});
    GGML_ASSERT(unique_count() >= 13); // shared prefix once, both unique branches, current token

    printf("GLM5 k-pool sequence edit and shared-token tests passed\n");
    return 0;
}

struct glm5_multiseq_config {
    std::vector<ggml_backend_dev_t> devs;
    llama_split_mode split_mode;
    std::string label;
};

static constexpr uint32_t GLM5_MULTISEQ_N_SEQ   = 3;
static constexpr uint32_t GLM5_MULTISEQ_N_STEPS = 4;
static constexpr uint32_t GLM5_MULTISEQ_TOP_K   = 12; // three pools, so the prompts outgrow the selection
static constexpr double   GLM5_MULTISEQ_MAX_NMSE = 1e-5;
static const uint32_t     GLM5_MULTISEQ_PROMPT_LEN[GLM5_MULTISEQ_N_SEQ] = {24, 20, 28};

// Decodes the prompts of seqs in one batch, then single-token steps for all of them together.
// Returns the logits of every output, grouped per sequence.
static std::map<llama_seq_id, std::vector<float>> glm5_multiseq_decode(
        llama_context * ctx, const std::vector<std::vector<llama_token>> & prompts, const std::vector<llama_seq_id> & seqs) {
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    llama_memory_clear(llama_get_memory(ctx), true);

    std::map<llama_seq_id, std::vector<float>> logits;
    auto decode = [&](const std::vector<std::pair<llama_seq_id, llama_pos>> & entries, bool prompt) {
        llama_batch batch = llama_batch_init(entries.size(), 0, 1);
        std::vector<std::pair<llama_seq_id, int32_t>> outputs;
        for (size_t i = 0; i < entries.size(); ++i) {
            const auto [s, pos] = entries[i];
            const auto & toks = prompts[s];
            const bool output = !prompt || (size_t) pos + 1 == toks.size();
            common_batch_add(batch, toks[pos % toks.size()], pos, {s}, output);
            if (output) {
                outputs.emplace_back(s, (int32_t) i);
            }
        }
        GGML_ASSERT(llama_decode(ctx, batch) == 0);
        for (const auto & [s, i] : outputs) {
            const float * row = llama_get_logits_ith(ctx, i);
            logits[s].insert(logits[s].end(), row, row + n_vocab);
        }
        llama_batch_free(batch);
    };

    std::vector<std::pair<llama_seq_id, llama_pos>> entries;
    for (const llama_seq_id s : seqs) {
        for (llama_pos p = 0; p < (llama_pos) prompts[s].size(); ++p) {
            entries.emplace_back(s, p);
        }
    }
    decode(entries, true);

    for (uint32_t step = 0; step < GLM5_MULTISEQ_N_STEPS; ++step) {
        entries.clear();
        for (const llama_seq_id s : seqs) {
            entries.emplace_back(s, (llama_pos) (prompts[s].size() + step));
        }
        decode(entries, false);
    }
    return logits;
}

struct glm5_multiseq_probe {
    int64_t max_scatter_streams = 0; // widest stream count seen on the dense DSA mask
};

static bool glm5_multiseq_observe(ggml_tensor * tensor, bool ask, void * data) {
    if (ask && std::strncmp(tensor->name, "kq_mask_dsa-", 12) == 0) {
        auto * probe = static_cast<glm5_multiseq_probe *>(data);
        probe->max_scatter_streams = std::max(probe->max_scatter_streams, tensor->ne[3]);
    }
    return false;
}

// The default 1e-2 weights make attention uniform, hiding which query rows it reads.
// Give the DSA attention and indexer unit norms and fan-in scaled projections instead.
static void set_tensor_data_glm5_attn(struct ggml_tensor * tensor, void * userdata) {
    const std::string name = tensor->name;
    const bool attn = name.find(".attn_") != std::string::npos || name.find(".indexer") != std::string::npos;
    if (!attn || tensor->type != GGML_TYPE_F32) {
        set_tensor_data(tensor, userdata);
        return;
    }
    const bool norm = string_ends_with(name, "norm.weight");
    std::mt19937 gen(*(const size_t *) userdata ^ std::hash<std::string>()(name));
    std::normal_distribution<float> dis(norm ? 1.0f : 0.0f, norm ? 0.1f : 1.0f/std::sqrt((float) tensor->ne[0]));
    std::vector<float> tmp(ggml_nelements(tensor));
    for (float & v : tmp) {
        v = dis(gen);
    }
    ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
}

static bool glm5_multiseq_matches(const glm5_multiseq_config & cfg, bool unified) {
    auto metadata = get_gguf_ctx(LLM_ARCH_GLM5_NEXT, true);
    gguf_set_val_u32(metadata.get(), "glm5-next.attention.indexer.top_k", GLM5_MULTISEQ_TOP_K);
    glm5_multiseq_probe probe;
    auto loaded = get_model_and_ctx(metadata.get(), nullptr, 1234, cfg.devs, cfg.split_mode,
            false, GLM5_MULTISEQ_N_SEQ, unified, glm5_multiseq_observe, &probe, set_tensor_data_glm5_attn);
    llama_context * ctx = loaded.second.get();
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(loaded.first.get()));

    std::vector<std::vector<llama_token>> prompts;
    std::vector<llama_seq_id> all_seqs;
    for (uint32_t s = 0; s < GLM5_MULTISEQ_N_SEQ; ++s) {
        prompts.push_back(get_tokens(GLM5_MULTISEQ_PROMPT_LEN[s], n_vocab, 100 + s));
        all_seqs.push_back((llama_seq_id) s);
    }

    const auto together = glm5_multiseq_decode(ctx, prompts, all_seqs);
    const int64_t expected_streams = unified ? 1 : GLM5_MULTISEQ_N_SEQ;
    bool ok = true;
    if (probe.max_scatter_streams != expected_streams) {
        printf("FAIL: GLM5 %s %s batch did not reach the dense DSA mask with %" PRId64 " streams (saw %" PRId64 ")\n",
                cfg.label.c_str(), unified ? "unified" : "non-unified", expected_streams, probe.max_scatter_streams);
        ok = false;
    }
    for (const llama_seq_id s : all_seqs) {
        const auto alone = glm5_multiseq_decode(ctx, prompts, {s});
        const double err = nmse(alone.at(s), together.at(s));
        printf("GLM5 %s %s seq %d nmse %.3e\n", cfg.label.c_str(), unified ? "unified" : "non-unified", s, err);
        if (!(err < GLM5_MULTISEQ_MAX_NMSE)) {
            printf("FAIL: GLM5 %s %s seq %d differs when decoded with other sequences (nmse %.3e)\n",
                    cfg.label.c_str(), unified ? "unified" : "non-unified", s, err);
            ok = false;
        }
    }
    return ok;
}

static std::vector<ggml_backend_dev_t> get_gpu_devices() {
    std::vector<ggml_backend_dev_t> gpus;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpus.push_back(dev);
        }
    }
    return gpus;
}

static std::vector<glm5_multiseq_config> glm5_multiseq_configs(bool gpu) {
    if (!gpu) {
        return {{{}, LLAMA_SPLIT_MODE_LAYER, "CPU"}};
    }
    const std::vector<ggml_backend_dev_t> gpus = get_gpu_devices();
    std::vector<glm5_multiseq_config> configs;
    if (!gpus.empty()) {
        configs.push_back({{gpus[0]}, LLAMA_SPLIT_MODE_LAYER, ggml_backend_dev_name(gpus[0])});
    }
    // GLM5 query and indexer head counts are not divisible by three.
    if (gpus.size() >= 2 && gpus.size() != 3) {
        configs.push_back({gpus, LLAMA_SPLIT_MODE_TENSOR, "Meta"});
    }
    return configs;
}

static int test_glm5_multiseq(bool gpu) {
    const auto configs = glm5_multiseq_configs(gpu);
    if (configs.empty()) {
        printf("GLM5 multi-sequence test skipped: no GPU\n");
        return 0;
    }
    bool ok = true;
    for (const auto & cfg : configs) {
        for (bool unified : {true, false}) {
            ok = glm5_multiseq_matches(cfg, unified) && ok;
        }
    }
    if (ok) {
        printf("GLM5 concurrent sequences match their solo decodes\n");
    }
    return ok ? 0 : 1;
}

// Mirrored f16 KV bytes per cell of the test model: 512 latent + 3*128 indexer values on its one DSA layer.
static constexpr uint64_t GLM5_TENSOR_OOM_BYTES_PER_CELL = 2*(512 + 3*128);

static int test_glm5_tensor_oom() {
    const auto configs = glm5_multiseq_configs(true);
    const auto meta = std::find_if(configs.begin(), configs.end(),
            [](const glm5_multiseq_config & cfg) { return cfg.split_mode == LLAMA_SPLIT_MODE_TENSOR; });
    if (meta == configs.end()) {
        printf("GLM5 tensor split OOM test skipped: needs 2 or 4+ GPUs\n");
        return 0;
    }
    size_t free_mem = 0;
    size_t total_mem = 0;
    ggml_backend_dev_memory(meta->devs[0], &free_mem, &total_mem);
    const uint64_t n_ctx = std::min<uint64_t>(INT32_MAX, 2*total_mem/GLM5_TENSOR_OOM_BYTES_PER_CELL);

    auto metadata = get_gguf_ctx(LLM_ARCH_GLM5_NEXT, true);
    std::vector<ggml_backend_dev_t> devs = meta->devs;
    devs.push_back(nullptr);
    llama_model_params mparams = llama_model_default_params();
    mparams.progress_callback = silent_model_load_progress;
    mparams.devices = devs.data();
    mparams.split_mode = LLAMA_SPLIT_MODE_TENSOR;
    size_t seed = 1234;
    llama_model_ptr model(llama_model_init_from_user(metadata.get(), set_tensor_data, &seed, mparams));
    GGML_ASSERT(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t) n_ctx;
    llama_context_ptr ctx(llama_init_from_model(model.get(), cparams));
    if (ctx) {
        printf("FAIL: GLM5 tensor split created a context of %" PRIu64 " cells\n", n_ctx);
        return 1;
    }
    printf("GLM5 tensor split reports an oversized KV cache without aborting\n");
    return 0;
}

static constexpr uint32_t GLM5_RESERVE_N_SEQ = 2;
// Sequence 0 starts this fraction of the fill ahead, so the sequence groups differ in pool count.
static constexpr uint32_t GLM5_RESERVE_LEAD_PARTS = 4;
// Tokens per sequence of the last span, a gather ubatch with one group per sequence.
static constexpr uint32_t GLM5_RESERVE_TAIL = 4;

// Decodes n tokens of every prompt s from position p0[s] + off in one batch, so its ubatches hold several sequence groups.
static void glm5_decode_from(llama_context * ctx, const std::vector<std::vector<llama_token>> & prompts,
        const std::vector<uint32_t> & p0, uint32_t off, uint32_t n) {
    llama_batch batch = llama_batch_init(prompts.size()*n, 0, 1);
    for (size_t s = 0; s < prompts.size(); ++s) {
        for (uint32_t pos = p0[s] + off; pos < p0[s] + off + n; ++pos) {
            common_batch_add(batch, prompts[s][pos], pos, {(llama_seq_id) s}, pos + 1 == prompts[s].size());
        }
    }
    GGML_ASSERT(llama_decode(ctx, batch) == 0);
    llama_batch_free(batch);
}

// Decodes positions [p0, p1) of every prompt in one batch.
static void glm5_decode_span(llama_context * ctx, const std::vector<std::vector<llama_token>> & prompts, uint32_t p0, uint32_t p1) {
    glm5_decode_from(ctx, prompts, std::vector<uint32_t>(prompts.size(), p0), 0, p1 - p0);
}

// Decodes n tokens of every sequence s from position p0[s], span tokens per sequence and batch.
static void glm5_decode_spans(llama_context * ctx, const std::vector<std::vector<llama_token>> & prompts,
        const std::vector<uint32_t> & p0, uint32_t n, uint32_t span) {
    for (uint32_t done = 0; done < n; done += span) {
        glm5_decode_from(ctx, prompts, p0, done, std::min(span, n - done));
    }
}

// Fills sequence 0 to its share of the cache and the others to a quarter less, then decodes one gather ubatch.
static void glm5_fill_sequences(llama_context * ctx, uint32_t n_seq, uint32_t n_vocab) {
    const uint32_t n_share = llama_n_ctx(ctx)/n_seq;
    const uint32_t n_fill  = n_share - GLM5_RESERVE_TAIL;
    const uint32_t n_lead  = n_fill/GLM5_RESERVE_LEAD_PARTS;
    const uint32_t span    = llama_n_batch(ctx)/n_seq;
    std::vector<std::vector<llama_token>> prompts;
    for (uint32_t s = 0; s < n_seq; ++s) {
        prompts.push_back(get_tokens(n_share, n_vocab, 200 + s));
    }
    glm5_decode_spans(ctx, {prompts[0]}, {0}, n_lead, span);

    std::vector<uint32_t> p0(n_seq, 0);
    p0[0] = n_lead;
    glm5_decode_spans(ctx, prompts, p0, n_fill - n_lead, span);

    std::vector<uint32_t> tail(n_seq, n_fill - n_lead);
    tail[0] = n_fill;
    glm5_decode_spans(ctx, prompts, tail, GLM5_RESERVE_TAIL, GLM5_RESERVE_TAIL);
}

// Fails when a graph decoded since n_planned_reserve needed a new allocation plan.
static bool glm5_check_no_plans(llama_context * ctx, size_t n_planned_reserve, const std::string & label, const char * what) {
    const size_t n_planned = ggml_backend_sched_get_n_planned(ctx->get_sched()) - n_planned_reserve;
    if (n_planned != 0) {
        printf("FAIL: GLM5 %s %s planned %zu new graphs\n", label.c_str(), what, n_planned);
        return false;
    }
    printf("GLM5 %s %s without re-planning\n", label.c_str(), what);
    return true;
}

static bool glm5_decode_after_reserve(const glm5_multiseq_config & cfg, bool unified) {
    auto metadata = get_gguf_ctx(LLM_ARCH_GLM5_NEXT, true);
    gguf_set_val_u32(metadata.get(), "glm5-next.attention.indexer.top_k", GLM5_MULTISEQ_TOP_K);
    auto loaded = get_model_and_ctx(metadata.get(), nullptr, 1234, cfg.devs, cfg.split_mode,
            false, GLM5_RESERVE_N_SEQ, unified);
    llama_context * ctx = loaded.second.get();
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(loaded.first.get()));
    const size_t n_planned = ggml_backend_sched_get_n_planned(ctx->get_sched());

    // a single-token graph replaces the reserved plan, as the server warmup does
    glm5_decode_span(ctx, {get_tokens(1, n_vocab, 199)}, 0, 1);
    llama_memory_clear(llama_get_memory(ctx), true);
    glm5_fill_sequences(ctx, GLM5_RESERVE_N_SEQ, n_vocab);
    return glm5_check_no_plans(ctx, n_planned, cfg.label, unified ? "unified filled its context" : "non-unified filled its context");
}

// CUDA decodes GLM5 with sparse FA from this many cells per stream, so a reserve of full streams takes the gather path only on request.
static constexpr uint32_t GLM5_SPARSE_FA_MIN_CELLS = 4096;
// A stream count that does not divide the gather token count of the reserve.
static constexpr uint32_t GLM5_LONG_STREAM_N_SEQ = 3;
static constexpr uint32_t GLM5_LONG_STREAM_PROMPT = 16;
static constexpr uint32_t GLM5_LONG_STREAM_STEPS  = 4;

// Decodes one token for every sequence at position pos.
static void glm5_decode_step(llama_context * ctx, uint32_t n_seq, llama_pos pos, uint32_t n_vocab) {
    llama_batch batch = llama_batch_init(n_seq, 0, 1);
    for (uint32_t s = 0; s < n_seq; ++s) {
        common_batch_add(batch, get_tokens(1, n_vocab, 300 + pos*n_seq + s)[0], pos, {(llama_seq_id) s}, true);
    }
    GGML_ASSERT(llama_decode(ctx, batch) == 0);
    llama_batch_free(batch);
}

// The reserve holds every stream of a non-unified cache, so its small ubatches must split into all of them.
// Decode on the still short cache takes the gather path, which full streams reserve only on request.
static bool glm5_reserve_long_streams(const glm5_multiseq_config & cfg) {
    auto metadata = get_gguf_ctx(LLM_ARCH_GLM5_NEXT, true);
    gguf_set_val_u32(metadata.get(), "glm5-next.attention.indexer.top_k", GLM5_MULTISEQ_TOP_K);
    gguf_set_val_u32(metadata.get(), "glm5-next.context_length", GLM5_SPARSE_FA_MIN_CELLS*GLM5_LONG_STREAM_N_SEQ);
    auto loaded = get_model_and_ctx(metadata.get(), nullptr, 1234, cfg.devs, cfg.split_mode,
            false, GLM5_LONG_STREAM_N_SEQ, false);
    llama_context * ctx = loaded.second.get();
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(loaded.first.get()));
    const size_t n_planned = ggml_backend_sched_get_n_planned(ctx->get_sched());

    std::vector<std::vector<llama_token>> prompts;
    for (uint32_t s = 0; s < GLM5_LONG_STREAM_N_SEQ; ++s) {
        prompts.push_back(get_tokens(GLM5_LONG_STREAM_PROMPT, n_vocab, 400 + s));
    }
    glm5_decode_span(ctx, prompts, 0, GLM5_LONG_STREAM_PROMPT);
    for (uint32_t step = 0; step < GLM5_LONG_STREAM_STEPS; ++step) {
        glm5_decode_step(ctx, GLM5_LONG_STREAM_N_SEQ, GLM5_LONG_STREAM_PROMPT + step, n_vocab);
    }
    return glm5_check_no_plans(ctx, n_planned, cfg.label, "non-unified with long streams decoded");
}

static constexpr uint32_t GLM5_SPLIT_N_SEQ      = 5;
static constexpr uint32_t GLM5_SPLIT_CELLS     = 256;
static constexpr uint32_t GLM5_SPLIT_PROMPT    = 3;
static constexpr uint32_t GLM5_SPLIT_N_SEQ_GROW = 3;
static constexpr uint32_t GLM5_SPLIT_GROW      = 5;

// The gather reserve spans all 5 streams with 4 tokens each, but 3 sequences one token short of a pool
// that take 5 tokens each complete 6 pools of 4, in a graph of the same size.
static bool glm5_reserve_covers_uneven_split(const glm5_multiseq_config & cfg) {
    auto metadata = get_gguf_ctx(LLM_ARCH_GLM5_NEXT, true);
    gguf_set_val_u32(metadata.get(), "glm5-next.attention.indexer.top_k", GLM5_MULTISEQ_TOP_K);
    gguf_set_val_u32(metadata.get(), "glm5-next.context_length", GLM5_SPLIT_CELLS*GLM5_SPLIT_N_SEQ);
    auto loaded = get_model_and_ctx(metadata.get(), nullptr, 1234, cfg.devs, cfg.split_mode,
            false, GLM5_SPLIT_N_SEQ, false);
    llama_context * ctx = loaded.second.get();
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(loaded.first.get()));
    const size_t n_planned = ggml_backend_sched_get_n_planned(ctx->get_sched());

    std::vector<std::vector<llama_token>> prompts;
    for (uint32_t s = 0; s < GLM5_SPLIT_N_SEQ; ++s) {
        prompts.push_back(get_tokens(GLM5_SPLIT_PROMPT, n_vocab, 500 + s));
    }
    glm5_decode_span(ctx, prompts, 0, GLM5_SPLIT_PROMPT);

    std::vector<std::vector<llama_token>> grown;
    for (uint32_t s = 0; s < GLM5_SPLIT_N_SEQ_GROW; ++s) {
        grown.push_back(get_tokens(GLM5_SPLIT_PROMPT + GLM5_SPLIT_GROW, n_vocab, 600 + s));
    }
    glm5_decode_span(ctx, grown, GLM5_SPLIT_PROMPT, GLM5_SPLIT_PROMPT + GLM5_SPLIT_GROW);
    return glm5_check_no_plans(ctx, n_planned, cfg.label, "non-unified gather ubatch over fewer streams decoded");
}

static constexpr uint32_t GLM5_KPOOL_TOP_K       = 12;
static constexpr double   GLM5_KPOOL_MAX_NMSE    = 1e-5;
static constexpr uint32_t GLM5_SHARED_PREFIX     = 24; // six pools, shared by sequences 0 and 1
static constexpr uint32_t GLM5_SHARED_OTHER      = 9;  // two pools and a tail, sequence 2 alone
static constexpr uint32_t GLM5_SHARED_N_SEQ      = 3;
static constexpr uint32_t GLM5_RESERVE_SIZE_CELLS = 16384;
static constexpr uint32_t GLM5_RESERVE_SIZE_N_SEQ = 8;
static constexpr double   GLM5_RESERVE_SIZE_MAX_RATIO = 1.5;
static constexpr uint32_t GLM5_SKEW_N_SEQ        = 4;
static constexpr uint32_t GLM5_SKEW_LONG         = 240; // 60 of the 64 pools of the default 256-cell cache
static constexpr uint32_t GLM5_SKEW_SHORT        = 3;

struct glm5_pool_probe {
    int64_t n_new = -1; // pool slots of the last new-pool block of the first DSA layer
};

static bool glm5_pool_observe(ggml_tensor * tensor, bool ask, void * data) {
    if (ask && std::strcmp(tensor->name, "indexer_pool_k_new-0") == 0) {
        static_cast<glm5_pool_probe *>(data)->n_new = tensor->ne[1];
    }
    return false;
}

struct glm5_token {
    llama_token token;
    llama_pos pos;
    std::vector<llama_seq_id> seq_ids;
};

// Decodes the tokens in one batch, every token an output, and returns the batch result.
static int glm5_decode_tokens(llama_context * ctx, const std::vector<glm5_token> & tokens) {
    llama_batch batch = llama_batch_init(tokens.size(), 0, GLM5_SHARED_N_SEQ);
    for (const auto & t : tokens) {
        common_batch_add(batch, t.token, t.pos, t.seq_ids, true);
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return rc;
}

// Decodes positions [p0, p0 + n) of sequence seq.
static void glm5_decode_seq(llama_context * ctx, llama_seq_id seq, llama_pos p0, uint32_t n, uint32_t n_vocab) {
    std::vector<glm5_token> tokens;
    const auto ids = get_tokens(n, n_vocab, 700 + seq);
    for (uint32_t i = 0; i < n; ++i) {
        tokens.push_back({ids[i], p0 + (llama_pos) i, {seq}});
    }
    GGML_ASSERT(glm5_decode_tokens(ctx, tokens) == 0);
}

// Sequences 0 and 1 share a prefix and sequence 2 is independent.
static void glm5_shared_history(llama_context * ctx, uint32_t n_vocab) {
    llama_memory_clear(llama_get_memory(ctx), true);
    glm5_decode_seq(ctx, 0, 0, GLM5_SHARED_PREFIX, n_vocab);
    llama_memory_seq_cp(llama_get_memory(ctx), 0, 1, -1, -1);
    glm5_decode_seq(ctx, 2, 0, GLM5_SHARED_OTHER, n_vocab);
}

// Decodes the next token of each sequence in seqs in one batch and returns the logits per sequence.
static std::map<llama_seq_id, std::vector<float>> glm5_shared_step(llama_context * ctx, const std::vector<llama_seq_id> & seqs, uint32_t n_vocab) {
    std::vector<glm5_token> tokens;
    for (const llama_seq_id s : seqs) {
        tokens.push_back({(llama_token) (800 + s) % (llama_token) n_vocab, (llama_pos) (s == 2 ? GLM5_SHARED_OTHER : GLM5_SHARED_PREFIX), {s}});
    }
    GGML_ASSERT(glm5_decode_tokens(ctx, tokens) == 0);
    std::map<llama_seq_id, std::vector<float>> logits;
    for (size_t i = 0; i < seqs.size(); ++i) {
        const float * row = llama_get_logits_ith(ctx, (int32_t) i);
        logits[seqs[i]].assign(row, row + n_vocab);
    }
    return logits;
}

// Compares the logits of every sequence of a joint step with a step of that sequence alone.
static bool glm5_shared_step_matches_solo(llama_context * ctx, const std::map<llama_seq_id, std::vector<float>> & joint, uint32_t n_vocab) {
    bool ok = true;
    for (const auto & [s, logits] : joint) {
        glm5_shared_history(ctx, n_vocab);
        const double err = nmse(glm5_shared_step(ctx, {s}, n_vocab).at(s), logits);
        if (!(err < GLM5_KPOOL_MAX_NMSE)) {
            printf("FAIL: GLM5 shared-prefix step of seq %d differs from its solo step (nmse %.3e)\n", s, err);
            ok = false;
        }
    }
    return ok;
}

// An uncached ubatch pools only the real pools of its layouts, and a sequence that shares no cell stays cached.
static bool glm5_uncached_pools_real_only() {
    auto metadata = get_gguf_ctx(LLM_ARCH_GLM5_NEXT, true);
    gguf_set_val_u32(metadata.get(), "glm5-next.attention.indexer.top_k", GLM5_KPOOL_TOP_K);
    glm5_pool_probe probe;
    auto loaded = get_model_and_ctx(metadata.get(), nullptr, 1234, {}, LLAMA_SPLIT_MODE_LAYER,
            false, GLM5_SHARED_N_SEQ, true, glm5_pool_observe, &probe, set_tensor_data_glm5_attn);
    llama_context * ctx = loaded.second.get();
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(loaded.first.get()));
    bool ok = true;

    // Sequence 2 completes no pool here, so the cached path pools one dummy slot.
    glm5_shared_history(ctx, n_vocab);
    glm5_shared_step(ctx, {2}, n_vocab);
    if (probe.n_new != 1) {
        printf("FAIL: GLM5 seq 2 next to a shared prefix pooled %" PRId64 " slots instead of the cached dummy slot\n", probe.n_new);
        ok = false;
    }

    // 6 + 6 + 2 real pools fit one padded block of 64 slots.
    glm5_shared_history(ctx, n_vocab);
    const auto joint = glm5_shared_step(ctx, {0, 1, 2}, n_vocab);
    if (probe.n_new != 64) {
        printf("FAIL: GLM5 uncached step pooled %" PRId64 " slots instead of one block of 64\n", probe.n_new);
        ok = false;
    }
    ok = glm5_shared_step_matches_solo(ctx, joint, n_vocab) && ok;
    if (ok) {
        printf("GLM5 uncached ubatch pools only its real pools\n");
    }
    return ok;
}

// Compute buffer bytes over all backends of the context.
static size_t glm5_compute_buffer_size(llama_context * ctx) {
    ggml_backend_sched_t sched = ctx->get_sched();
    size_t size = 0;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        size += ggml_backend_sched_get_buffer_size(sched, ggml_backend_sched_get_backend(sched, i));
    }
    return size;
}

static size_t glm5_unified_reserve_bytes(uint32_t n_seq) {
    auto metadata = get_gguf_ctx(LLM_ARCH_GLM5_NEXT, true);
    gguf_set_val_u32(metadata.get(), "glm5-next.attention.indexer.top_k", GLM5_KPOOL_TOP_K);
    gguf_set_val_u32(metadata.get(), "glm5-next.context_length", GLM5_RESERVE_SIZE_CELLS);
    auto loaded = get_model_and_ctx(metadata.get(), nullptr, 1234, {}, LLAMA_SPLIT_MODE_LAYER, false, n_seq, true);
    return glm5_compute_buffer_size(loaded.second.get());
}

// All sequences of a unified cache hold at most the pools of one stream, so the reserve must not scale with them.
static bool glm5_unified_reserve_size() {
    const size_t size_1 = glm5_unified_reserve_bytes(1);
    const size_t size_n = glm5_unified_reserve_bytes(GLM5_RESERVE_SIZE_N_SEQ);
    const double ratio  = (double) size_n/size_1;
    printf("GLM5 unified compute buffer: %zu bytes for 1 sequence, %zu for %u (ratio %.2f)\n", size_1, size_n, GLM5_RESERVE_SIZE_N_SEQ, ratio);
    if (!(ratio <= GLM5_RESERVE_SIZE_MAX_RATIO)) {
        printf("FAIL: GLM5 unified reserve grows with the sequence count\n");
        return false;
    }
    return true;
}

// Positions [p0, p0 + n) of every short sequence, from s0 to the last skew sequence.
static std::vector<glm5_token> glm5_skew_tokens(llama_seq_id s0, llama_pos p0, uint32_t n, uint32_t n_vocab) {
    std::vector<glm5_token> tokens;
    for (llama_seq_id s = s0; s < (llama_seq_id) GLM5_SKEW_N_SEQ; ++s) {
        for (uint32_t i = 0; i < n; ++i) {
            tokens.push_back({(llama_token) (s + i) % (llama_token) n_vocab, p0 + (llama_pos) i, {s}});
        }
    }
    return tokens;
}

// Short sequences next to a long one need more padded per-sequence pools than the reserve holds,
// so the ubatch scores as one group and fits the reserved plan.
static bool glm5_unified_skew_without_plans() {
    auto metadata = get_gguf_ctx(LLM_ARCH_GLM5_NEXT, true);
    gguf_set_val_u32(metadata.get(), "glm5-next.attention.indexer.top_k", GLM5_KPOOL_TOP_K);
    auto loaded = get_model_and_ctx(metadata.get(), nullptr, 1234, {}, LLAMA_SPLIT_MODE_LAYER, false, GLM5_SKEW_N_SEQ, true);
    llama_context * ctx = loaded.second.get();
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(loaded.first.get()));
    const size_t n_planned = ggml_backend_sched_get_n_planned(ctx->get_sched());

    glm5_decode_seq(ctx, 0, 0, GLM5_SKEW_LONG, n_vocab);
    GGML_ASSERT(glm5_decode_tokens(ctx, glm5_skew_tokens(1, 0, GLM5_SKEW_SHORT, n_vocab)) == 0);
    std::vector<glm5_token> step = glm5_skew_tokens(1, GLM5_SKEW_SHORT, 1, n_vocab);
    step.push_back({1, (llama_pos) GLM5_SKEW_LONG, {0}});
    GGML_ASSERT(glm5_decode_tokens(ctx, step) == 0);
    return glm5_check_no_plans(ctx, n_planned, "CPU", "unified with one long and several short sequences decoded");
}

// split_seq (embeddings) puts a token of sequences {0, 1} and tokens of sequence 0 in one ubatch.
static bool glm5_split_seq_mixed_layouts() {
    auto metadata = get_gguf_ctx(LLM_ARCH_GLM5_NEXT, true);
    gguf_set_val_u32(metadata.get(), "glm5-next.attention.indexer.top_k", GLM5_KPOOL_TOP_K);
    auto loaded = get_model_and_ctx(metadata.get(), nullptr, 1234, {}, LLAMA_SPLIT_MODE_LAYER, false, 2, true);
    llama_context * ctx = loaded.second.get();
    const uint32_t n_embd = llama_model_n_embd(loaded.first.get());
    llama_set_embeddings(ctx, true);

    const std::vector<glm5_token> shared = {{10, 0, {0, 1}}, {11, 1, {0, 1}}, {12, 2, {0, 1}}, {13, 3, {0, 1}}};
    const std::vector<glm5_token> own    = {{20, 4, {0}},    {21, 5, {0}},    {22, 6, {0}},    {23, 7, {0}}};
    auto embeddings = [&](int32_t i0) {
        std::vector<float> res;
        for (int32_t i = i0; i < i0 + (int32_t) own.size(); ++i) {
            const float * row = llama_get_embeddings_ith(ctx, i);
            res.insert(res.end(), row, row + n_embd);
        }
        return res;
    };

    GGML_ASSERT(glm5_decode_tokens(ctx, shared) == 0);
    GGML_ASSERT(glm5_decode_tokens(ctx, own) == 0);
    const auto ref = embeddings(0);

    llama_memory_clear(llama_get_memory(ctx), true);
    std::vector<glm5_token> mixed = shared;
    mixed.insert(mixed.end(), own.begin(), own.end());
    if (glm5_decode_tokens(ctx, mixed) != 0) {
        printf("FAIL: GLM5 embeddings batch with mixed sequence sets did not decode\n");
        return false;
    }
    const double err = nmse(ref, embeddings((int32_t) shared.size()));
    if (!(err < GLM5_KPOOL_MAX_NMSE)) {
        printf("FAIL: GLM5 embeddings of a mixed split_seq ubatch differ from two batches (nmse %.3e)\n", err);
        return false;
    }
    printf("GLM5 split_seq ubatch with mixed sequence sets decoded (nmse %.3e)\n", err);
    return true;
}

// Sequence groups of the k-pool indexer: uncached and oversized ubatches, unified reserve size, split_seq.
static int test_glm5_kpool_groups() {
    bool ok = glm5_uncached_pools_real_only();
    ok = glm5_unified_reserve_size() && ok;
    ok = glm5_unified_skew_without_plans() && ok;
    ok = glm5_split_seq_mixed_layouts() && ok;
    return ok ? 0 : 1;
}

static constexpr uint32_t SCRAMBLE_N_SEQ    = 3;
static constexpr uint32_t SCRAMBLE_N_TOKENS = 4;
// Tokens per sequence of a batch of two sequences: a decode step, a gather ubatch, and a prompt ubatch.
static const uint32_t SCRAMBLE_BATCH_TOKENS[] = {1, 4, 12};
// Solo decodes in this order give the sequences recurrent state cells out of sequence order.
static const llama_seq_id SCRAMBLE_ORDER[SCRAMBLE_N_SEQ] = {2, 0, 1};

// Tokens [p0, p0 + n) of every sequence of seqs.
static std::vector<glm5_token> glm5_seqs_tokens(const std::vector<llama_seq_id> & seqs, llama_pos p0, uint32_t n, uint32_t n_vocab) {
    std::vector<glm5_token> tokens;
    for (const llama_seq_id s : seqs) {
        const auto ids = get_tokens(n, n_vocab, 800 + s);
        for (uint32_t i = 0; i < n; ++i) {
            tokens.push_back({ids[i], p0 + (llama_pos) i, {s}});
        }
    }
    return tokens;
}

// Clears the memory, decodes the sequences alone in SCRAMBLE_ORDER, then n tokens of sequences 1 and 2 in one batch.
static void glm5_scrambled_batch(llama_context * ctx, uint32_t n, uint32_t n_vocab) {
    llama_memory_clear(llama_get_memory(ctx), true);
    for (const llama_seq_id s : SCRAMBLE_ORDER) {
        glm5_decode_seq(ctx, s, 0, SCRAMBLE_N_TOKENS, n_vocab);
    }
    GGML_ASSERT(glm5_decode_tokens(ctx, glm5_seqs_tokens({1, 2}, SCRAMBLE_N_TOKENS, n, n_vocab)) == 0);
}

// The scrambled batches, then a full prompt ubatch of one sequence, which outgrows a plan sized for them.
static void glm5_scrambled_batches(llama_context * ctx, uint32_t n_vocab) {
    for (const uint32_t n : SCRAMBLE_BATCH_TOKENS) {
        glm5_scrambled_batch(ctx, n, n_vocab);
    }
    llama_memory_clear(llama_get_memory(ctx), true);
    glm5_decode_seq(ctx, 0, 0, llama_n_ubatch(ctx), n_vocab);
}

// Recurrent state cells follow arrival order, so a ubatch of sequences 1 and 2 also copies the state of sequence 0.
// No reserve of n_seq_max sequences holds these extra state rows, so the plans grow once and are then reused.
static bool glm5_scrambled_states_reuse_plans(const glm5_multiseq_config & cfg, bool unified) {
    auto metadata = get_gguf_ctx(LLM_ARCH_GLM5_NEXT, true);
    gguf_set_val_u32(metadata.get(), "glm5-next.attention.indexer.top_k", GLM5_MULTISEQ_TOP_K);
    auto loaded = get_model_and_ctx(metadata.get(), nullptr, 1234, cfg.devs, cfg.split_mode, false, SCRAMBLE_N_SEQ, unified);
    llama_context * ctx = loaded.second.get();
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(loaded.first.get()));

    glm5_scrambled_batches(ctx, n_vocab);
    const size_t n_planned = ggml_backend_sched_get_n_planned(ctx->get_sched());
    glm5_scrambled_batches(ctx, n_vocab);
    return glm5_check_no_plans(ctx, n_planned, cfg.label,
            unified ? "unified with scrambled state cells repeated its ubatches" : "non-unified with scrambled state cells repeated its ubatches");
}

// The first scrambled ubatches re-plan with the same graph size, so ctest runs this without the Debug re-plan abort.
static int test_glm5_reserve_scrambled_states() {
    const auto configs = glm5_multiseq_configs(true);
    if (configs.empty()) {
        printf("GLM5 scrambled state test skipped: no GPU\n");
        return 0;
    }
    bool ok = true;
    for (bool unified : {true, false}) {
        ok = glm5_scrambled_states_reuse_plans(configs.front(), unified) && ok;
    }
    return ok ? 0 : 1;
}

static int test_glm5_reserve_covers_context() {
    const auto configs = glm5_multiseq_configs(true);
    if (configs.empty()) {
        printf("GLM5 reserve test skipped: no GPU\n");
        return 0;
    }
    bool ok = true;
    for (bool unified : {true, false}) {
        ok = glm5_decode_after_reserve(configs.front(), unified) && ok;
    }
    ok = glm5_reserve_long_streams(configs.front()) && ok;
    ok = glm5_reserve_covers_uneven_split(configs.front()) && ok;
    return ok ? 0 : 1;
}

static int test_glm5_invalid_metadata() {
    struct invalid_case {
        const char * key;
        uint32_t value;
    };
    const invalid_case cases[] = {
        {"glm5-next.nextn_predict_layers", 2},
        {"glm5-next.attention.indexer.kpool", 0},
        {"glm5-next.attention.indexer.top_k", 0},
        {"glm5-next.attention.indexer.top_k", 5},
        {"glm5-next.hyper_connection.count", 3},
    };

    for (const auto & test : cases) {
        auto metadata = get_gguf_ctx(LLM_ARCH_GLM5_NEXT, true);
        gguf_set_val_u32(metadata.get(), test.key, test.value);
        auto params = llama_model_default_params();
        size_t seed = 1234;
        llama_model_ptr model(llama_model_init_from_user(metadata.get(), set_tensor_data, &seed, params));
        if (model) {
            printf("FAIL: GLM5 accepted %s=%u\n", test.key, test.value);
            return 1;
        }
    }

    printf("GLM5 invalid metadata rejected without aborting\n");
    return 0;
}

static constexpr uint32_t  LAYER_INP_LID       = 1;
static constexpr uint32_t  LAYER_INP_N_TOKENS  = 128; // two ubatches of 64
static constexpr llama_pos LAYER_INP_POS_MIN   = 80;  // skips the first ubatch and part of the second
static constexpr float     LAYER_INP_STALE     = 1.0e30f;
static constexpr float     LAYER_INP_TOLERANCE = 1.0e-4f;

static void fill_layer_inp_stale(llama_context * ctx, size_t n_floats) {
    std::fill_n(llama_get_embeddings_layer_inp(ctx, LAYER_INP_LID), n_floats, LAYER_INP_STALE);
}

static std::vector<float> decode_layer_inp(llama_context * ctx, const std::vector<llama_token> & tokens, size_t n_floats) {
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        common_batch_add(batch, tokens[i], i, {0}, i + 1 == tokens.size());
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    GGML_ASSERT(rc == 0);

    const float * rows = llama_get_embeddings_layer_inp(ctx, LAYER_INP_LID);
    return std::vector<float>(rows, rows + n_floats);
}

// rows below first_row keep the stale data, the other rows match the reference
static bool layer_inp_rows_match(const std::vector<float> & rows, const std::vector<float> & ref, size_t n_embd, size_t first_row) {
    for (size_t i = 0; i < rows.size(); ++i) {
        const bool ok = i / n_embd < first_row ? rows[i] == LAYER_INP_STALE :
            std::fabs(rows[i] - ref[i]) <= LAYER_INP_TOLERANCE * std::max(1.0f, std::fabs(ref[i]));
        if (!ok) {
            return false;
        }
    }
    return true;
}

// layer input rows are copied from the first row at or above the sequence's pos_min, -1 copies all rows again
static bool test_layer_inp_pos_min_arch(llm_arch arch) {
    auto metadata = get_gguf_ctx(arch, false);
    auto loaded = get_model_and_ctx(metadata.get(), nullptr, 1234, {});
    llama_model * model = loaded.first.get();
    llama_context * ctx = loaded.second.get();
    llama_set_embeddings_layer_inp(ctx, LAYER_INP_LID, true);

    const size_t n_embd   = llama_model_n_embd(model);
    const size_t n_floats = n_embd * LAYER_INP_N_TOKENS;
    const auto   tokens   = get_tokens(LAYER_INP_N_TOKENS, llama_vocab_n_tokens(llama_model_get_vocab(model)), 1234);
    const std::vector<float> ref = decode_layer_inp(ctx, tokens, n_floats);

    fill_layer_inp_stale(ctx, n_floats);
    llama_set_embeddings_layer_inp_pos_min(ctx, 0, LAYER_INP_POS_MIN);
    const bool skipped = layer_inp_rows_match(decode_layer_inp(ctx, tokens, n_floats), ref, n_embd, LAYER_INP_POS_MIN);

    fill_layer_inp_stale(ctx, n_floats);
    llama_set_embeddings_layer_inp_pos_min(ctx, 0, -1);
    const bool all = layer_inp_rows_match(decode_layer_inp(ctx, tokens, n_floats), ref, n_embd, 0);

    if (!skipped || !all) {
        printf("FAIL: %s layer input rows (pos_min %d: %s, pos_min -1: %s)\n", llm_arch_name(arch), LAYER_INP_POS_MIN,
                skipped ? "ok" : "wrong", all ? "ok" : "wrong");
    }
    return skipped && all;
}

static int test_layer_inp_pos_min() {
    const bool ok = test_layer_inp_pos_min_arch(LLM_ARCH_LLAMA) && test_layer_inp_pos_min_arch(LLM_ARCH_QWEN35);
    if (ok) {
        printf("layer input pos_min tests passed\n");
    }
    return ok ? 0 : 1;
}

// Enough layers that the second GPU's split of a two-GPU layer split is launched as one CUDA graph.
static constexpr uint32_t LAYER_SPLIT_GRAPH_N_LAYER  = 64;
static constexpr uint32_t LAYER_SPLIT_GRAPH_N_TOKENS = 192;
static constexpr double   LAYER_SPLIT_GRAPH_MAX_NMSE = 1e-6;

static std::vector<float> get_layer_split_logits(const std::vector<ggml_backend_dev_t> & devs) {
    auto metadata = get_gguf_ctx(LLM_ARCH_LLAMA, false);
    gguf_set_val_u32(metadata.get(), "llama.block_count", LAYER_SPLIT_GRAPH_N_LAYER);
    auto loaded = get_model_and_ctx(metadata.get(), nullptr, 1234, devs);
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(loaded.first.get()));
    const auto tokens = get_tokens(LAYER_SPLIT_GRAPH_N_TOKENS, n_vocab, 300);
    return get_logits(loaded.first.get(), loaded.second.get(), tokens);
}

// The CUDA debug message of a split launched as one graph.
static constexpr const char * LAYER_SPLIT_GATED_LOG = "waits on another GPU as one graph";
// A chain longer than the node count from which CUDA launches a split that waits on another GPU as one graph.
static constexpr int LAYER_SPLIT_CHAIN_NODES = 1024;
static constexpr int LAYER_SPLIT_CHAIN_ELEMS = 256;

struct gated_log_counter {
    ggml_log_callback callback  = nullptr;
    void *            user_data = nullptr;
    int               n_gated   = 0;
};

static void count_gated_launch(ggml_log_level level, const char * text, void * user_data) {
    auto * counter = (gated_log_counter *) user_data;
    if (strstr(text, LAYER_SPLIT_GATED_LOG) != nullptr) {
        counter->n_gated++;
    }
    counter->callback(level, text, counter->user_data);
}

// Runs f and returns how many splits CUDA launched as one graph meanwhile.
template <typename F>
static int count_gated_launches(F && f) {
    gated_log_counter counter;
    ggml_log_get(&counter.callback, &counter.user_data);
    ggml_log_set(count_gated_launch, &counter);
    f();
    ggml_log_set(counter.callback, counter.user_data);
    return counter.n_gated;
}

// Copies a tensor from one GPU to another outside the scheduler, then computes a long chain on the destination.
static void compute_after_bare_copy(ggml_backend_dev_t dev_src, ggml_backend_dev_t dev_dst) {
    ggml_backend_ptr backend_src(ggml_backend_dev_init(dev_src, nullptr));
    ggml_backend_ptr backend_dst(ggml_backend_dev_init(dev_dst, nullptr));
    const size_t mem_size = ggml_tensor_overhead()*(LAYER_SPLIT_CHAIN_NODES + 2) +
        ggml_graph_overhead_custom(LAYER_SPLIT_CHAIN_NODES + 1, false);
    ggml_context_ptr ctx_src(ggml_init({ ggml_tensor_overhead(), nullptr, true }));
    ggml_context_ptr ctx_dst(ggml_init({ mem_size, nullptr, true }));

    ggml_tensor * src = ggml_new_tensor_1d(ctx_src.get(), GGML_TYPE_F32, LAYER_SPLIT_CHAIN_ELEMS);
    ggml_tensor * dst = ggml_new_tensor_1d(ctx_dst.get(), GGML_TYPE_F32, LAYER_SPLIT_CHAIN_ELEMS);
    ggml_tensor * out = dst;
    for (int i = 0; i < LAYER_SPLIT_CHAIN_NODES; i++) {
        out = ggml_scale(ctx_dst.get(), out, 1.0f);
    }
    ggml_cgraph * gf = ggml_new_graph_custom(ctx_dst.get(), LAYER_SPLIT_CHAIN_NODES + 1, false);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_ptr buf_src(ggml_backend_alloc_ctx_tensors(ctx_src.get(), backend_src.get()));
    ggml_backend_buffer_ptr buf_dst(ggml_backend_alloc_ctx_tensors(ctx_dst.get(), backend_dst.get()));
    GGML_ASSERT(buf_src && buf_dst);
    ggml_backend_tensor_copy_async(backend_src.get(), backend_dst.get(), src, dst);
    GGML_ASSERT(ggml_backend_graph_compute(backend_dst.get(), gf) == GGML_STATUS_SUCCESS);
}

static int test_layer_split_cuda_graph() {
    const std::vector<ggml_backend_dev_t> gpus = get_gpu_devices();
    if (gpus.size() < 2) {
        printf("layer split CUDA graph test skipped: needs 2 GPUs\n");
        return 0;
    }
    const auto single = get_layer_split_logits({gpus[0]});
    std::vector<float> split;
    const int n_gated_split = count_gated_launches([&]() { split = get_layer_split_logits({gpus[0], gpus[1]}); });
    const int n_gated_copy  = count_gated_launches([&]() { compute_after_bare_copy(gpus[0], gpus[1]); });
    const double err = nmse(single, split);
    printf("layer split over 2 GPUs vs 1 GPU: nmse %.3e, %d splits launched as one graph, %d after a bare copy\n",
            err, n_gated_split, n_gated_copy);
    if (!(err < LAYER_SPLIT_GRAPH_MAX_NMSE)) {
        printf("FAIL: layer split over 2 GPUs differs from 1 GPU\n");
        return 1;
    }
    if (n_gated_split == 0 || n_gated_copy != 0) {
        printf("FAIL: only a scheduler split that waits on another GPU must launch as one graph\n");
        return 1;
    }
    return 0;
}

int main(int argc, char ** argv) {
    // init the logger at max verbosity. filter with a custom callback respecting the user-configure verbosity
    common_log_set_verbosity_thold(LOG_LEVEL_DEBUG);
    common_init();

    if (argc == 2 && (strcmp(argv[1], "--mtp-shared") == 0 || strcmp(argv[1], "--mtp-shared-cpu") == 0)) {
        return test_mtp_shared(strcmp(argv[1], "--mtp-shared-cpu") == 0);
    }
    if (argc == 2 && strcmp(argv[1], "--hadamard-contracts") == 0) {
        return test_hadamard_contracts();
    }
    if (argc == 2 && strcmp(argv[1], "--glm5-kpool-sequences") == 0) {
        const int rc = test_glm5_kpool_sequences();
        return test_glm5_kpool_groups() != 0 ? 1 : rc;
    }
    if (argc == 2 && (strcmp(argv[1], "--glm5-multiseq") == 0 || strcmp(argv[1], "--glm5-multiseq-gpu") == 0)) {
        return test_glm5_multiseq(strcmp(argv[1], "--glm5-multiseq-gpu") == 0);
    }
    if (argc == 2 && strcmp(argv[1], "--glm5-tensor-oom") == 0) {
        return test_glm5_tensor_oom();
    }
    if (argc == 2 && strcmp(argv[1], "--glm5-reserve-covers-context") == 0) {
        return test_glm5_reserve_covers_context();
    }
    if (argc == 2 && strcmp(argv[1], "--glm5-reserve-scrambled-states") == 0) {
        return test_glm5_reserve_scrambled_states();
    }
    if (argc == 2 && strcmp(argv[1], "--layer-split-cuda-graph") == 0) {
        return test_layer_split_cuda_graph();
    }
    if (argc == 2 && strcmp(argv[1], "--glm5-invalid-metadata") == 0) {
        return test_glm5_invalid_metadata();
    }
    if (argc == 2 && strcmp(argv[1], "--layer-inp-pos-min") == 0) {
        return test_layer_inp_pos_min();
    }
    std::random_device rd;

    llm_arch arch = LLM_ARCH_UNKNOWN;
    size_t seed = rd();
    std::string out;
    bool qsa_unified_multiseq = false;

    int verbosity = LOG_LEVEL_ERROR;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv);
            return 0;
        }
        if (strcmp(argv[i], "--qsa-unified-multiseq") == 0) {
            qsa_unified_multiseq = true;
            continue;
        }

        if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--arch") == 0) {
            if (i + 1 < argc) {
                const std::string arch_name = argv[++i];
                arch = llm_arch_from_string(arch_name);
                if (arch == LLM_ARCH_UNKNOWN) {
                    LOG_ERR("%s: unkown LLM architecture: %s\n", __func__, arch_name.c_str());
                    return 1;
                }
            } else {
                usage(argv);
                return 1;
            }
        }
        if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seed") == 0) {
            if (i + 1 < argc) {
                seed = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        }
        if (strcmp(argv[i], "-v") == 0) {
            if (i + 1 < argc) {
                verbosity = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        }
        if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--out") == 0) {
            if (i + 1 < argc) {
                out = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        }
    }
    printf("%s: using seed %zu\n", __func__, seed);

    try {
        if (qsa_unified_multiseq) {
            return test_qsa_unified_multiseq(seed);
        }
        if (!out.empty()) {
            return save_models(arch, seed, verbosity, out);
        }
        return test_backends(arch, seed, verbosity);
    } catch (const std::exception & err) {
        fprintf(stderr, "encountered runtime error: %s\n", err.what());
        return -1;
    }
}
