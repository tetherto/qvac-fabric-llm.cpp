#pragma once

#include "llama-coreml.h"
#include "llama-batch.h"
#include "llama-hparams.h"

#include <cmath>
#include <map>
#include <vector>

inline constexpr int llama_coreml_lengths[] = {32, 48, 64, 128, 256, LLAMA_COREML_MAX_TOKENS};
inline constexpr int llama_coreml_n_lengths = sizeof(llama_coreml_lengths) / sizeof(llama_coreml_lengths[0]);
inline constexpr double llama_coreml_min_cosine = 0.998;
inline constexpr int llama_coreml_probe_tokens = 32;
inline constexpr char llama_coreml_probe_text[] = "title: none | text: The quick brown fox jumps over the lazy dog.";

inline bool llama_coreml_valid_inflight(int value) {
    return value >= 1 && value <= LLAMA_COREML_MAX_INFLIGHT;
}

inline bool llama_coreml_context_compatible(const llama_cparams & params, const llama_hparams & hparams) {
    return params.embeddings && params.pooling_type == LLAMA_POOLING_TYPE_MEAN &&
            params.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT && !params.training && !params.training_ctx && !params.embeddings_nextn &&
            params.causal_attn == hparams.causal_attn &&
            params.rope_freq_base == hparams.rope_freq_base_train &&
            params.rope_freq_scale == hparams.rope_freq_scale_train && params.yarn_ext_factor == 0.0f &&
            params.yarn_attn_factor == hparams.yarn_attn_factor * hparams.rope_attn_factor;
}

inline double llama_coreml_cosine(const float * a, const float * b, size_t dimension) {
    double dot = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < dimension; ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            return 0;
        }
        dot += double(a[i]) * b[i];
        aa += double(a[i]) * a[i];
        bb += double(b[i]) * b[i];
    }
    return aa > 0 && bb > 0 ? dot / sqrt(aa * bb) : 0;
}

struct llama_coreml_job {
    int count;
    int function_index;
};

// Inputs are validated before planning. The last function is pack_256.
inline llama_coreml_job llama_coreml_plan_job(const llama_coreml_embedding_sequence * sequences, int remaining, bool pack, int slots) {
    int count = 1;
    int used = sequences[0].n_tokens;
    while (pack && count < remaining && count < slots && used + sequences[count].n_tokens <= LLAMA_COREML_PACK_TOKENS) {
        used += sequences[count++].n_tokens;
    }
    if (count > 1) {
        return {count, llama_coreml_n_lengths};
    }
    int bucket = 0;
    while (llama_coreml_lengths[bucket] < used) {
        ++bucket;
    }
    return {1, bucket};
}

using llama_coreml_tokens = std::map<llama_seq_id, std::vector<int32_t>>;

inline bool llama_coreml_group_tokens(const llama_ubatch & batch, llama_coreml_tokens & tokens) {
    if (!batch.token || batch.embd || batch.is_mixed() || batch.n_pos != 1 ||
            !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.n_tokens) {
        return false;
    }
    llama_coreml_tokens grouped;
    for (uint32_t i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] != 1 || !batch.seq_id[i]) {
            return false;
        }
        auto & sequence = grouped[batch.seq_id[i][0]];
        if (batch.pos[i] != (llama_pos) sequence.size() || sequence.size() >= LLAMA_COREML_MAX_TOKENS) {
            return false;
        }
        sequence.push_back(batch.token[i]);
    }
    tokens.swap(grouped);
    return true;
}
