#pragma once

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

struct llama_model;

struct llama_moe_cache_lru_fill {
    int32_t layer;
    int32_t expert;
    int32_t slot;
};

struct llama_moe_cache_lru_stats {
    size_t hits = 0;
    size_t misses = 0;
    size_t evictions = 0;
};

class llama_moe_cache_lru {
public:
    llama_moe_cache_lru(int32_t n_layers, int32_t n_experts, int32_t n_slots);

    bool plan(
            int32_t layer,
            const int32_t * ids,
            size_t n_ids,
            int32_t * remapped_ids,
            std::vector<llama_moe_cache_lru_fill> & fills,
            llama_moe_cache_lru_stats & stats);

    int32_t capacity() const;

private:
    struct slot {
        int32_t layer = -1;
        int32_t expert = -1;
        uint64_t last_used = 0;
    };

    int32_t n_layers;
    int32_t n_experts;
    uint64_t clock = 0;
    std::vector<int32_t> slot_for_expert;
    std::vector<slot> slots;
};

class llama_moe_cache {
public:
    llama_moe_cache(
            const llama_model & model,
            ggml_backend_t backend,
            ggml_backend_buffer_type_t buft,
            size_t size);
    ~llama_moe_cache();

    ggml_backend_t backend() const;
    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;
    void log_stats() const;

    static bool sched_resolve(
            void * user_data,
            const ggml_tensor * weight,
            ggml_backend_t backend,
            ggml_tensor ** cached_weight,
            void ** cache_entry);
    static void sched_begin(void * user_data);
    static bool sched_prepare(
            void * user_data,
            void * cache_entry,
            const int32_t * ids,
            size_t n_ids,
            int32_t * remapped_ids);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
