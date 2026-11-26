#pragma once

#include "llama.h"

#include <vector>

struct llama_vocab;
struct llama_grammar;

// sampler chain

struct llama_sampler_chain {
    llama_sampler_chain_params params;

    // has .backend_init() been called?
    bool is_init = false;

    // the arguments and the result of .backend_init(), a later call with the same arguments on the same samplers
    // returns the result; n_edits counts the samplers added and removed
    ggml_backend_buffer_type_t init_buft = nullptr;
    uint32_t init_n_outputs_max_per_seq  = 0;
    uint32_t init_n_edits                = 0;
    bool     init_res                    = false;

    uint32_t n_edits = 0;

    uint32_t n_nodes = 0;

    struct info {
        bool is_backend;

        llama_sampler * ptr;
    };

    std::vector<info> samplers;

    // pre-allocated buffer for llama_sampler_sample to avoid repeated allocations
    std::vector<llama_token_data> cur;

    // timing

    mutable int64_t t_sample_us;

    mutable int32_t n_sample;
};

uint32_t llama_sampler_backend_n_nodes(const llama_sampler * sampler);
void llama_sampler_backend_begin(llama_sampler * sampler);

struct llama_sampler * llama_sampler_init_dry_testing(
        float   dry_multiplier,
        float   dry_base,
        int32_t dry_allowed_length,
        int32_t dry_penalty_last_n,
        const std::vector<std::vector<llama_token>> & seq_breakers);
