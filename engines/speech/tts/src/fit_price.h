#pragma once

// Size-only graph pricing that mirrors the engines' dual-path dispatch
// (src/sched_dispatch.h) by construction: a graph the primary backend fully
// supports is priced with the gallocr the direct path would reserve
// (ggml_gallocr_reserve_n_size); a graph it does not is priced through the
// same [primary, CPU-last] scheduler shape the fallback path allocates with
// (ggml_backend_sched_reserve_size). Both APIs are the size-only twins of the
// real allocators in the fabric ggml pin -- nothing is allocated on any
// device and nothing runs.
//
// Shared by the audio8 and chatterbox fit projectors.

#include "backend_selection.h"
#include "backend_util.h"
#include "sched_dispatch.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cstdint>

namespace tts_cpp::detail {

struct fit_graph_price {
    uint64_t device_bytes = 0;  // the primary backend's portion
    uint64_t host_bytes   = 0;  // the CPU-fallback portion on sched-path graphs
    bool     used_sched   = false;
};

// Running aggregate over graphs that are each replayed from their own resident
// arena until the first of them prices scheduler-backed -- at which point the
// runtime drops every kept graph and all of them share arenas per call: the
// direct ones one growing allocator, the scheduler-backed ones the scheduler's.
// total() follows that fork: the sum of the arenas while all are replayed, the
// widest direct plus the widest scheduler-backed once any is not.
struct fit_price_aggregate {
    fit_graph_price direct_sum;
    fit_graph_price direct_peak;
    fit_graph_price sched_peak;
    bool any_sched = false;

    static uint64_t sat_add_u64(uint64_t a, uint64_t b) {
        const uint64_t sum = a + b;
        return sum < a ? UINT64_MAX : sum;
    }

    void add(const fit_graph_price & one) {
        if (one.used_sched) {
            any_sched = true;
            sched_peak.device_bytes = std::max(sched_peak.device_bytes, one.device_bytes);
            sched_peak.host_bytes = std::max(sched_peak.host_bytes, one.host_bytes);
            return;
        }
        direct_sum.device_bytes = sat_add_u64(direct_sum.device_bytes, one.device_bytes);
        direct_sum.host_bytes = sat_add_u64(direct_sum.host_bytes, one.host_bytes);
        direct_peak.device_bytes = std::max(direct_peak.device_bytes, one.device_bytes);
        direct_peak.host_bytes = std::max(direct_peak.host_bytes, one.host_bytes);
    }

    bool all_replayed() const { return !any_sched; }

    fit_graph_price total() const {
        if (all_replayed()) return direct_sum;
        fit_graph_price mixed;
        mixed.device_bytes = sat_add_u64(direct_peak.device_bytes, sched_peak.device_bytes);
        mixed.host_bytes = sat_add_u64(direct_peak.host_bytes, sched_peak.host_bytes);
        mixed.used_sched = true;
        return mixed;
    }
};

// A metadata-only model marks its weights allocated (non-null data) without a
// backend buffer. The scheduler places leafs from that buffer and skips view
// ops when it assigns backends, so a reshape or transpose over such a weight
// reaches the size-only reserve unassigned and aborts it
// (GGML_ASSERT(buffer_id >= 0) in ggml-alloc). Detected through the operands
// of the graph's nodes: ggml exposes nodes, not leafs.
inline bool tensor_is_unbuffered_weight(const ggml_tensor * t) {
    return t && t->data && !t->buffer && !t->view_src;
}

inline bool graph_has_unbuffered_weights(ggml_cgraph * graph) {
    const int n_nodes = ggml_graph_n_nodes(graph);
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (tensor_is_unbuffered_weight(node->src[s])) return true;
        }
    }
    return false;
}

// Price one freshly built graph. The graph's weight/state leafs must already
// be marked externally allocated (non-null data) so only true compute scratch
// is counted. Returns false when no pricer could be constructed.
//
// The scheduler shape is used where the backend cannot run every op, except
// on a metadata-only model, which the scheduler cannot place; the direct
// pricer then sizes every op on the primary backend, an upper bound on the
// device figure with nothing charged to the host slot.
inline bool fit_price_graph(ggml_backend_t backend, ggml_cgraph * graph,
                            size_t sched_graph_size, fit_graph_price & out) {
    out = fit_graph_price{};
    const bool use_sched =
        (sched_force_enabled() || !graph_fully_supported(backend, graph)) &&
        !graph_has_unbuffered_weights(graph);
    if (!use_sched) {
        ggml_gallocr_t pricer =
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!pricer) return false;
        size_t size = 0;
        ggml_gallocr_reserve_n_size(pricer, graph, nullptr, nullptr, &size);
        ggml_gallocr_free(pricer);
        out.device_bytes = size;
        return true;
    }
    // Same shape as sched_fallback_ensure: [primary, CPU-last], CPU skipped
    // when the primary IS the CPU backend.
    ggml_backend_t cpu = backend_is_cpu(backend) ? nullptr : init_cpu_backend();
    ggml_backend_t backends[2];
    int n = 0;
    backends[n++] = backend;
    if (cpu) backends[n++] = cpu;
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, /*bufts=*/nullptr, n, sched_graph_size,
        /*parallel=*/false, /*op_offload=*/false);
    if (!sched) {
        if (cpu) ggml_backend_free(cpu);
        return false;
    }
    size_t sizes[2] = {0, 0};
    ggml_backend_sched_reserve_size(sched, graph, sizes);
    ggml_backend_sched_free(sched);
    if (cpu) ggml_backend_free(cpu);
    out.device_bytes = sizes[0];
    out.host_bytes   = cpu ? sizes[1] : 0;
    out.used_sched   = true;
    return true;
}

}  // namespace tts_cpp::detail
