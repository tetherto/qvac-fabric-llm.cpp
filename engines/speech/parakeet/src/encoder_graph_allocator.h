#pragma once

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <array>
#include <limits>
#include <vector>

namespace parakeet {

class EncoderGraphAllocator {
public:
    EncoderGraphAllocator() = default;
    EncoderGraphAllocator(const EncoderGraphAllocator &) = delete;
    EncoderGraphAllocator & operator=(const EncoderGraphAllocator &) = delete;
    ~EncoderGraphAllocator() { release(); }

    bool initialize(ggml_backend_t active, ggml_backend_t cpu, size_t graph_slots, bool fallback) {
        release();
        active_ = active;
        cpu_ = cpu;
        if (fallback) {
            std::array<ggml_backend_t, 2> backends{active, cpu};
            scheduler_ = ggml_backend_sched_new(backends.data(), nullptr, backends.size(),
                                                 graph_slots, false, false);
            return scheduler_ != nullptr;
        }
        direct_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(active));
        return direct_ != nullptr;
    }

    void release() {
        if (scheduler_) ggml_backend_sched_free(scheduler_);
        if (direct_) ggml_gallocr_free(direct_);
        scheduler_ = nullptr;
        direct_ = nullptr;
        active_ = cpu_ = nullptr;
    }

    bool reserve(ggml_cgraph * graph, size_t * measurement) {
        if (scheduler_) {
            if (!measurement) return ggml_backend_sched_alloc_graph(scheduler_, graph);
            auto * marker = ggml_backend_alloc_buffer(cpu_, 0);
            if (!marker) return false;
            ggml_backend_buffer_set_usage(marker, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            const auto external_inputs = mark_bufferless_inputs(graph, marker);
            std::array<size_t, 2> sizes{};
            const bool measured = ggml_backend_sched_reserve_size(scheduler_, graph, sizes.data());
            restore_input_buffers(external_inputs);
            ggml_backend_buffer_free(marker);
            if (!measured) return false;
            if (ggml_backend_sched_get_buffer_type(scheduler_, active_) ==
                ggml_backend_sched_get_buffer_type(scheduler_, cpu_)) sizes[1] = 0;
            if (sizes[0] > std::numeric_limits<size_t>::max() - sizes[1]) return false;
            *measurement = sizes[0] + sizes[1];
            return true;
        }
        if (measurement) {
            *measurement = 0;
            ggml_gallocr_reserve_n_size(direct_, graph, nullptr, nullptr, measurement);
            return true;
        }
        return ggml_gallocr_reserve(direct_, graph);
    }

    bool allocate(ggml_cgraph * graph) {
        return scheduler_ || ggml_gallocr_alloc_graph(direct_, graph);
    }

    ggml_status compute(ggml_cgraph * graph) {
        return scheduler_ ? ggml_backend_sched_graph_compute(scheduler_, graph)
                          : ggml_backend_graph_compute(active_, graph);
    }

    size_t bytes() const {
        if (scheduler_) {
            return ggml_backend_sched_get_buffer_size(scheduler_, active_) +
                   ggml_backend_sched_get_buffer_size(scheduler_, cpu_);
        }
        return direct_ ? ggml_gallocr_get_buffer_size(direct_, 0) : 0;
    }

    ggml_backend_sched_t scheduler() const { return scheduler_; }

private:
    using InputBuffers = std::vector<ggml_tensor *>;

    static void mark_node_inputs(ggml_tensor * node, ggml_backend_buffer_t marker, InputBuffers & inputs) {
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            auto * source = node->src[i];
            if (source && source->op == GGML_OP_NONE && !source->buffer &&
                !(source->flags & GGML_TENSOR_FLAG_INPUT)) {
                inputs.push_back(source);
                source->buffer = marker;
            }
        }
    }

    static InputBuffers mark_bufferless_inputs(ggml_cgraph * graph, ggml_backend_buffer_t marker) {
        InputBuffers inputs;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            mark_node_inputs(ggml_graph_node(graph, i), marker, inputs);
        }
        return inputs;
    }

    static void restore_input_buffers(const InputBuffers & inputs) {
        for (auto * input : inputs) input->buffer = nullptr;
    }

    ggml_gallocr_t direct_ = nullptr;
    ggml_backend_sched_t scheduler_ = nullptr;
    ggml_backend_t active_ = nullptr;
    ggml_backend_t cpu_ = nullptr;
};

}
