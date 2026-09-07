#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <unordered_map>

llama_moe_cache_lru::llama_moe_cache_lru(int32_t n_layers, int32_t n_experts, int32_t n_slots) :
        n_layers(n_layers),
        n_experts(n_experts),
        slot_for_expert(size_t(n_layers) * n_experts, -1),
        slots(n_slots) {
    if (n_layers <= 0 || n_experts <= 0 || n_slots <= 0) {
        throw std::invalid_argument("invalid MoE cache LRU dimensions");
    }
}

bool llama_moe_cache_lru::plan(
        int32_t layer,
        const int32_t * ids,
        size_t n_ids,
        int32_t * remapped_ids,
        std::vector<llama_moe_cache_lru_fill> & fills,
        llama_moe_cache_lru_stats & stats) {
    if (layer < 0 || layer >= n_layers || ids == nullptr || remapped_ids == nullptr) {
        return false;
    }

    fills.clear();
    stats = {};

    std::vector<bool> selected(n_experts, false);
    size_t n_selected = 0;
    for (size_t i = 0; i < n_ids; ++i) {
        GGML_ASSERT(ids[i] >= 0 && ids[i] < n_experts);
        if (!selected[ids[i]]) {
            selected[ids[i]] = true;
            n_selected++;
        }
    }
    if (n_selected > slots.size()) {
        return false;
    }

    std::vector<bool> pinned(slots.size(), false);
    for (int32_t expert = 0; expert < n_experts; ++expert) {
        if (!selected[expert]) {
            continue;
        }
        const size_t key = size_t(layer) * n_experts + expert;
        const int32_t slot_id = slot_for_expert[key];
        if (slot_id < 0) {
            continue;
        }
        pinned[slot_id] = true;
        slots[slot_id].last_used = ++clock;
        stats.hits++;
    }

    for (int32_t expert = 0; expert < n_experts; ++expert) {
        if (!selected[expert]) {
            continue;
        }
        const size_t key = size_t(layer) * n_experts + expert;
        if (slot_for_expert[key] >= 0) {
            continue;
        }

        int32_t victim = -1;
        uint64_t oldest = UINT64_MAX;
        for (size_t slot_id = 0; slot_id < slots.size(); ++slot_id) {
            if (pinned[slot_id]) {
                continue;
            }
            if (slots[slot_id].layer < 0) {
                victim = slot_id;
                break;
            }
            if (slots[slot_id].last_used < oldest) {
                oldest = slots[slot_id].last_used;
                victim = slot_id;
            }
        }
        if (victim < 0) {
            return false;
        }

        slot & dst = slots[victim];
        if (dst.layer >= 0) {
            const size_t old_key = size_t(dst.layer) * n_experts + dst.expert;
            slot_for_expert[old_key] = -1;
            stats.evictions++;
        }

        dst.layer = layer;
        dst.expert = expert;
        dst.last_used = ++clock;
        slot_for_expert[key] = victim;
        pinned[victim] = true;
        fills.push_back({ layer, expert, victim });
        stats.misses++;
    }

    for (size_t i = 0; i < n_ids; ++i) {
        const size_t key = size_t(layer) * n_experts + ids[i];
        remapped_ids[i] = slot_for_expert[key];
    }
    return true;
}

int32_t llama_moe_cache_lru::capacity() const {
    return slots.size();
}

namespace {

enum class projection_kind {
    GATE,
    UP,
    DOWN,
    GATE_UP,
};

using projection_list = std::vector<std::pair<projection_kind, ggml_tensor *>>;

struct layer_source {
    int32_t layer;
    projection_list projections;
};

static projection_list get_layer_projections(const llama_layer & layer) {
    projection_list result;
    if (layer.ffn_gate_up_exps != nullptr) {
        result.push_back({ projection_kind::GATE_UP, layer.ffn_gate_up_exps });
    } else {
        if (layer.ffn_gate_exps != nullptr) {
            result.push_back({ projection_kind::GATE, layer.ffn_gate_exps });
        }
        if (layer.ffn_up_exps != nullptr) {
            result.push_back({ projection_kind::UP, layer.ffn_up_exps });
        }
    }
    if (layer.ffn_down_exps != nullptr) {
        result.push_back({ projection_kind::DOWN, layer.ffn_down_exps });
    }
    return result;
}

static bool same_projection_layout(
        const projection_list & a,
        const projection_list & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].first != b[i].first ||
            a[i].second->type != b[i].second->type ||
            !ggml_are_same_shape(a[i].second, b[i].second)) {
            return false;
        }
    }
    return true;
}

}

struct llama_moe_cache::impl {
    struct cache_stats {
        size_t hits = 0;
        size_t misses = 0;
        size_t evictions = 0;
        size_t fill_bytes = 0;
    };

    struct projection { ggml_tensor * bank = nullptr; };

    struct binding {
        int32_t layer;
        size_t projection;
        ggml_tensor * source;
        ggml_tensor * cached;
    };

    struct layer {
        std::vector<size_t> bindings;
        uint64_t planned_epoch = 0;
        std::vector<int32_t> planned_ids;
        std::vector<int32_t> remapped_ids;
    };

    ggml_backend_t backend;
    bool no_alloc;
    uint64_t epoch = 0;
    cache_stats stats;
    llama_moe_cache_lru lru;
    std::vector<projection> projections;
    std::vector<binding> bindings;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, size_t> binding_for_source;
    ggml_context_ptr context;
    ggml_backend_buffer_ptr buffer;

    impl(
            const llama_model & model,
            ggml_backend_t backend,
            ggml_backend_buffer_type_t buft,
            size_t requested_size) :
        backend(backend),
        no_alloc(model.hparams.no_alloc),
        lru(model.layers.size(), model.hparams.n_expert, 1),
        layers(model.layers.size()) {
        if (backend == nullptr || buft == nullptr) {
            throw std::runtime_error("MoE cache requires a backend");
        }
        const ggml_backend_dev_t backend_device = ggml_backend_get_device(backend);
        const enum ggml_backend_dev_type device_type = ggml_backend_dev_type(backend_device);
        if (device_type != GGML_BACKEND_DEVICE_TYPE_GPU && device_type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        const ggml_backend_reg_t backend_reg = ggml_backend_dev_backend_reg(backend_device);
        if (backend_reg != nullptr && std::strcmp(ggml_backend_reg_name(backend_reg), "OpenCL") == 0) {
            throw std::runtime_error("MoE cache does not support OpenCL tensor uploads");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || model.hparams.n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires routed experts");
        }

        const ggml_backend_dev_t target_device = ggml_backend_get_device(backend);
        std::vector<layer_source> complete_source_layers;
        std::vector<layer_source> source_layers;
        projection_list reference;

        size_t n_partial_host_layers = 0;
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto layer_projections = get_layer_projections(model.layers[il]);
            if (layer_projections.empty() || model.dev_layer(il) != target_device) {
                continue;
            }

            bool any_host = false;
            bool all_host = true;
            for (const auto & item : layer_projections) {
                const ggml_tensor * tensor = item.second;
                const bool is_host =
                    tensor->buffer != nullptr &&
                    ggml_backend_buffer_get_usage(tensor->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    ggml_backend_buffer_is_host(tensor->buffer);
                any_host = any_host || is_host;
                all_host = all_host && is_host;
            }
            if (!any_host) {
                continue;
            }
            if (!all_host) {
                // --fit can leave one fractional MoE layer at the placement boundary.
                // Its host projections retain the scheduler's ordinary offload path;
                // cache only layers whose complete expert bank has one source backend.
                n_partial_host_layers++;
                continue;
            }

            complete_source_layers.push_back({ int32_t(il), std::move(layer_projections) });
        }

        if (complete_source_layers.empty()) {
            LLAMA_LOG_DEBUG(
                "llama_moe_cache: no complete host-resident expert layers; cache inactive\n");
            return;
        }
        if (n_partial_host_layers > 0) {
            LLAMA_LOG_DEBUG(
                "llama_moe_cache: %zu fractional host expert layers use ordinary offload\n",
                n_partial_host_layers);
        }

        // One cache bank can only hold tensors with one physical layout. Some
        // quantized models mix layouts between layers (for example Q4_1 down
        // projections followed by Q4_0 projections). Cache the largest uniform
        // group and leave the other complete layers on ordinary offload.
        size_t reference_index = 0;
        size_t reference_count = 0;
        for (size_t candidate = 0; candidate < complete_source_layers.size(); ++candidate) {
            size_t count = 0;
            for (const layer_source & source_layer : complete_source_layers) {
                count += same_projection_layout(
                    complete_source_layers[candidate].projections,
                    source_layer.projections);
            }
            if (count > reference_count) {
                reference_index = candidate;
                reference_count = count;
            }
        }
        reference = complete_source_layers[reference_index].projections;
        for (layer_source & source_layer : complete_source_layers) {
            if (same_projection_layout(reference, source_layer.projections)) {
                source_layers.push_back(std::move(source_layer));
            }
        }
        const size_t n_layout_mismatch_layers = complete_source_layers.size() - source_layers.size();
        if (n_layout_mismatch_layers > 0) {
            LLAMA_LOG_WARN(
                "llama_moe_cache: %zu expert layers with non-dominant tensor layouts use ordinary offload\n",
                n_layout_mismatch_layers);
        }

        if (reference[0].second->ne[2] != int64_t(model.hparams.n_expert)) {
            throw std::runtime_error("MoE cache expert tensor count does not match model metadata");
        }

        const size_t alignment = ggml_backend_buft_get_alignment(buft);
        const int32_t max_slots = source_layers.size() * model.hparams.n_expert;
        auto allocation_size = [&](int32_t n_slots) {
            size_t result = 0;
            for (const auto & item : reference) {
                // CUDA MMQ can read padding beyond the last expert slice. The
                // cached tensor views expose n_slots and leave this guard hidden.
                result += GGML_PAD(item.second->nb[2] * size_t(n_slots + 1), alignment);
            }
            return result;
        };

        int32_t n_slots = 0;
        for (int32_t candidate = 1; candidate <= max_slots; ++candidate) {
            if (allocation_size(candidate) > requested_size) {
                break;
            }
            n_slots = candidate;
        }
        if (n_slots < int32_t(model.hparams.n_expert_used)) {
            throw std::runtime_error("MoE cache budget is smaller than one routed layer working set");
        }
        lru = llama_moe_cache_lru(model.layers.size(), model.hparams.n_expert, n_slots);

        const size_t n_tensors = reference.size() + source_layers.size() * reference.size();
        ggml_init_params params = {
            /*.mem_size   =*/ n_tensors * ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context_ptr ctx_ptr { ggml_init(params) };
        if (!ctx_ptr) {
            throw std::runtime_error("failed to create MoE cache tensor context");
        }
        ggml_context * ctx = ctx_ptr.get();

        projections.reserve(reference.size());
        for (const auto & item : reference) {
            ggml_tensor * bank = ggml_dup_tensor_layout(ctx, item.second);
            bank->ne[2] = n_slots + 1;
            bank->nb[3] = bank->nb[2] * bank->ne[2];
            ggml_format_name(bank, "moe_cache.%s", item.second->name);
            projections.push_back({ bank });
        }

        bindings.reserve(source_layers.size() * reference.size());
        for (const layer_source & source_layer : source_layers) {
            layer & cache_layer = layers[source_layer.layer];
            for (size_t ip = 0; ip < source_layer.projections.size(); ++ip) {
                ggml_tensor * source = source_layer.projections[ip].second;
                ggml_tensor * bank = projections[ip].bank;
                ggml_tensor * cached = ggml_view_3d(
                    ctx, bank, bank->ne[0], bank->ne[1], n_slots, bank->nb[1], bank->nb[2], 0);
                ggml_format_name(cached, "moe_cache.%s", source->name);
                cache_layer.bindings.push_back(bindings.size());
                bindings.push_back({ source_layer.layer, ip, source, cached });
            }
        }
        for (size_t ib = 0; ib < bindings.size(); ++ib) {
            binding_for_source.emplace(bindings[ib].source, ib);
        }

        ggml_backend_buffer_ptr buffer_ptr;
        if (no_alloc) {
            buffer_ptr.reset(ggml_backend_buft_alloc_buffer(buft, 0));
            if (!buffer_ptr) {
                throw std::runtime_error("failed to create dummy MoE cache buffer");
            }
            for (ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor != nullptr; tensor = ggml_get_next_tensor(ctx, tensor)) {
                tensor->buffer = buffer_ptr.get();
            }
        } else {
            buffer_ptr.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft));
            if (!buffer_ptr) {
                throw std::runtime_error("failed to allocate MoE cache buffer");
            }
            if (ggml_backend_buffer_get_size(buffer_ptr.get()) > requested_size) {
                throw std::runtime_error("allocated MoE cache exceeds its budget");
            }
            ggml_backend_buffer_clear(buffer_ptr.get(), 0);
        }

        LLAMA_LOG_INFO("llama_moe_cache: %10s MoE cache size = %8.2f MiB, slots = %d\n",
            ggml_backend_buffer_name(buffer_ptr.get()),
            (no_alloc ? allocation_size(n_slots) : ggml_backend_buffer_get_size(buffer_ptr.get())) / (1024.0 * 1024.0),
            n_slots);
        context = std::move(ctx_ptr);
        buffer = std::move(buffer_ptr);
    }

    bool resolve(
            const ggml_tensor * weight,
            ggml_backend_t target,
            ggml_tensor ** cached_weight,
            void ** cache_entry) {
        if (target != backend) {
            return false;
        }
        const auto it = binding_for_source.find(weight);
        if (it == binding_for_source.end()) {
            return false;
        }
        binding & entry = bindings[it->second];
        *cached_weight = entry.cached;
        *cache_entry = &entry;
        return true;
    }

    void begin() {
        stats = {};
        if (++epoch == 0) {
            // Only possible after uint64_t rollover. Invalidate the initial zero
            // epochs before reserving a non-zero value for the next plan.
            epoch = 1;
            for (layer & cache_layer : layers) {
                cache_layer.planned_epoch = 0;
            }
        }
    }

    bool prepare(
            binding * entry,
            const int32_t * ids,
            size_t n_ids,
            int32_t * remapped_ids) {
        if (entry == nullptr || ids == nullptr || remapped_ids == nullptr) {
            return false;
        }
        layer & cache_layer = layers[entry->layer];
        if (cache_layer.planned_epoch == epoch) {
            if (cache_layer.planned_ids.size() != n_ids ||
                !std::equal(cache_layer.planned_ids.begin(), cache_layer.planned_ids.end(), ids)) {
                return false;
            }
            std::copy(cache_layer.remapped_ids.begin(), cache_layer.remapped_ids.end(), remapped_ids);
            return true;
        }

        cache_layer.planned_ids.assign(ids, ids + n_ids);
        cache_layer.remapped_ids.resize(n_ids);
        std::vector<llama_moe_cache_lru_fill> fills;
        llama_moe_cache_lru_stats lru_stats;
        if (!lru.plan(
                entry->layer, ids, n_ids, cache_layer.remapped_ids.data(), fills, lru_stats)) {
            return false;
        }

        size_t fill_bytes = 0;
        for (size_t binding_id : cache_layer.bindings) {
            const binding & source_binding = bindings[binding_id];
            const projection & cache_projection = projections[source_binding.projection];
            for (const llama_moe_cache_lru_fill & fill : fills) {
                const size_t src_offset = size_t(fill.expert) * source_binding.source->nb[2];
                const size_t dst_offset = size_t(fill.slot) * cache_projection.bank->nb[2];
                const size_t src_size = ggml_nbytes(source_binding.source);
                if (source_binding.source->data == nullptr || src_offset >= src_size) {
                    return false;
                }
                const size_t src_available = src_size - src_offset;
                const size_t copy_size = std::min(cache_projection.bank->nb[2], src_available);
                if (copy_size == 0) {
                    return false;
                }
                ggml_backend_tensor_set_async(
                    backend,
                    cache_projection.bank,
                    static_cast<const uint8_t *>(source_binding.source->data) + src_offset,
                    dst_offset,
                    copy_size);
                fill_bytes += copy_size;
            }
        }

        stats.hits += lru_stats.hits;
        stats.misses += lru_stats.misses;
        stats.evictions += lru_stats.evictions;
        stats.fill_bytes += fill_bytes;
        cache_layer.planned_epoch = epoch;
        std::copy(cache_layer.remapped_ids.begin(), cache_layer.remapped_ids.end(), remapped_ids);
        return true;
    }

    void log_stats() const {
        const size_t accesses = stats.hits + stats.misses;
        if (accesses == 0) {
            return;
        }
        LLAMA_LOG_INFO(
            "llama_moe_cache: hits = %zu, misses = %zu, hit rate = %.2f%%, evictions = %zu, transferred = %.2f MiB\n",
            stats.hits, stats.misses, 100.0 * stats.hits / accesses, stats.evictions,
            stats.fill_bytes / (1024.0 * 1024.0));
    }

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const {
        std::map<ggml_backend_buffer_type_t, size_t> result;
        if (!buffer) {
            return result;
        }

        const ggml_backend_buffer_type_t buffer_type = ggml_backend_buffer_get_type(buffer.get());
        result[buffer_type] = no_alloc
            ? ggml_backend_alloc_ctx_tensors_from_buft_size(context.get(), buffer_type)
            : ggml_backend_buffer_get_size(buffer.get());
        return result;
    }
};

llama_moe_cache::llama_moe_cache(
        const llama_model & model,
        ggml_backend_t backend,
        ggml_backend_buffer_type_t buft,
        size_t size) :
    pimpl(new impl(model, backend, buft, size)) {
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_backend_t llama_moe_cache::backend() const {
    return pimpl->backend;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    return pimpl->memory_breakdown();
}

void llama_moe_cache::log_stats() const {
    pimpl->log_stats();
}

bool llama_moe_cache::sched_resolve(
        void * user_data,
        const ggml_tensor * weight,
        ggml_backend_t backend,
        ggml_tensor ** cached_weight,
        void ** cache_entry) {
    return static_cast<llama_moe_cache *>(user_data)->pimpl->resolve(
        weight, backend, cached_weight, cache_entry);
}

void llama_moe_cache::sched_begin(void * user_data) {
    static_cast<llama_moe_cache *>(user_data)->pimpl->begin();
}

bool llama_moe_cache::sched_prepare(
        void * user_data,
        void * cache_entry,
        const int32_t * ids,
        size_t n_ids,
        int32_t * remapped_ids) {
    return static_cast<llama_moe_cache *>(user_data)->pimpl->prepare(
        static_cast<impl::binding *>(cache_entry), ids, n_ids, remapped_ids);
}
