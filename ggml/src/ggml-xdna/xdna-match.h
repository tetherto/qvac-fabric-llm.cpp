#pragma once

// How the graph planners recognise the nodes llama builds around a layer;
// test-xdna-match checks them against the shapes llama's graph code makes.

#include "ggml.h"

#include <cstring>
#include <string>

// The GET_ROWS that reads a recurrent state: the node itself, or the gather of
// all n_rs states whose first rows build_rs hands out as a view at offset 0.
static inline const ggml_tensor * xdna_rec_state_gather(const ggml_tensor * n) {
    if (n->op == GGML_OP_GET_ROWS) {
        return n;
    }
    const ggml_tensor * g = n->src[0];
    return n->op == GGML_OP_VIEW && n->view_offs == 0 && g && g->op == GGML_OP_GET_ROWS ? g : nullptr;
}

// A GET_ROWS of the one row of a one-row tensor is that tensor: llama crops the
// last layer to its output rows, which in a single-token decode is all of it.
static inline ggml_tensor * xdna_skip_row_crop(ggml_tensor * t) {
    const bool crop = t && t->op == GGML_OP_GET_ROWS && t->src[0] && t->src[1] && ggml_nelements(t->src[1]) == 1 &&
                      ggml_nrows(t->src[0]) == 1;
    return crop ? t->src[0] : t;
}

// A FLASH_ATTN_EXT the prefill attention kernels can take: q, k and v, no sinks
// (src[4]) and no K/V row lists (src[5], ggml_flash_attn_ext_rows). The kernels
// read each head's keys and values as the leading cells of the K/V view; with
// row lists, mask column i stands for cache row kv_rows[i] instead. The decode
// routes take an identity list (xdna_att_eligible).
static inline bool xdna_fa_plain(const ggml_tensor * n) {
    return n && n->op == GGML_OP_FLASH_ATTN_EXT && n->src[0] && n->src[1] && n->src[2] && !n->src[4] &&
           !n->src[5];
}

// A K/V row list (src[5], ggml_flash_attn_ext_rows) of one slice, one cache
// cell a mask column; xdna_att_rows_identity checks its values at run time.
// The decode routes scan k's ne[1] mask columns, so a list must be exactly as
// wide: a visible column past those could name any cell.
static inline bool xdna_att_rows_shape_ok(const ggml_tensor * rows, const ggml_tensor * m, const ggml_tensor * k) {
    return !rows || (rows->type == GGML_TYPE_I32 && ggml_is_contiguous(rows) && rows->ne[0] == m->ne[0] &&
                     rows->ne[1] == 1 && m->ne[0] == k->ne[1]);
}

// The pool reads the first n_valid cache cells: a row list has to name
// exactly those, as one sequence's list over the cells from 0 does.
static inline bool xdna_att_rows_identity(const ggml_tensor * fa, int n_valid) {
    const ggml_tensor * r = fa->src[5];
    if (!r) {
        return true;
    }
    const int32_t * rows = (const int32_t *) r->data;
    if (!rows) {
        return false;
    }
    for (int j = 0; j < n_valid; j++) {
        if (rows[j] != j) {
            return false;
        }
    }
    return true;
}

// Whether `name` ends in `suffix`.
static inline bool xdna_name_ends(const char * name, const char * suffix) {
    const size_t n = strlen(name), s = strlen(suffix);
    return n >= s && strcmp(name + n - s, suffix) == 0;
}

// Whether the fused decode reads weight `w` in its own format: ssm_out in the gated layout the artifacts were built
// with (`gated_fmt` 0: Q4_K; 1: Q5_K or Q6_K), the FFN's gate and up as Q4_K, its down as Q4_K or Q6_K. Every other
// tensor is not one of these and passes.
static inline bool xdna_fused_weight_ok(const ggml_tensor * w, int gated_fmt) {
    const char *    name = ggml_get_name(w);
    const ggml_type t    = w->type;
    if (xdna_name_ends(name, ".ssm_out.weight")) {
        return gated_fmt == 0 ? t == GGML_TYPE_Q4_K : t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q6_K;
    }
    if (xdna_name_ends(name, ".ffn_gate.weight") || xdna_name_ends(name, ".ffn_up.weight")) {
        return t == GGML_TYPE_Q4_K;
    }
    if (xdna_name_ends(name, ".ffn_down.weight")) {
        return t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q6_K;
    }
    return true;
}

// The first weight node `n` reads that the fused decode does not (xdna_fused_weight_ok), or null.
static inline const ggml_tensor * xdna_fused_uncovered_src(const ggml_tensor * n, int gated_fmt) {
    for (int s = 0; s < GGML_MAX_SRC && n->src[s]; s++) {
        const ggml_tensor * w = n->src[s];
        if (w->op == GGML_OP_NONE && !xdna_fused_weight_ok(w, gated_fmt)) {
            return w;
        }
    }
    return nullptr;
}

// The first weight of these nodes the fused decode does not read, or null when it reads them all: a chunk with one
// is not handed to the fused layers.
static inline const ggml_tensor * xdna_fused_uncovered(const ggml_tensor * const * nodes, int n_nodes, int gated_fmt) {
    for (int i = 0; i < n_nodes; i++) {
        if (const ggml_tensor * w = xdna_fused_uncovered_src(nodes[i], gated_fmt)) {
            return w;
        }
    }
    return nullptr;
}

// Widths at which the fused decode runs a weight: `min`, then every `step` up to `max`.
struct xdna_width_range {
    int64_t min, max, step;
};

static inline bool xdna_width_in(const xdna_width_range & r, int64_t v) {
    return v >= r.min && v <= r.max && (v - r.min) % r.step == 0;
}

// The layer shapes the fused decode runs: a recurrent layer's convolution channels (ssm_conv1d ne[1]), an attention
// layer's head size (attn_k_norm ne[0]) and query width (attn_output ne[0]).
struct xdna_fused_shapes {
    xdna_width_range conv;
    int64_t          head;
    xdna_width_range query;
};

// What a model's decode chunks showed of its layers against the ones the fused decode runs (xdna_layer_census_add).
struct xdna_layer_census {
    bool        recurrent = false;  // a recurrent layer of a shape it runs
    bool        whole     = false;  // a whole layer went by: its output, l_out-N
    std::string foreign;            // the name of the first weight of a layer it does not run, kept past its graph
};

static constexpr char XDNA_LAYER_OUT[] = "l_out-";

// Whether weight `w` is of a layer the fused decode does not run: an FFN weight other than the dense gate, up and
// down (a mixture of experts' router, experts and shared expert), or a shape outside `s`.
static inline bool xdna_layer_foreign(const ggml_tensor * w, const xdna_fused_shapes & s) {
    const char * name = ggml_get_name(w);
    if (strncmp(name, "blk.", 4) != 0) {
        return false;
    }
    if (strstr(name, ".ffn_")) {
        return !xdna_name_ends(name, ".ffn_gate.weight") && !xdna_name_ends(name, ".ffn_up.weight") &&
               !xdna_name_ends(name, ".ffn_down.weight");
    }
    if (xdna_name_ends(name, ".ssm_conv1d.weight")) {
        return !xdna_width_in(s.conv, w->ne[1]);
    }
    if (xdna_name_ends(name, ".attn_k_norm.weight")) {
        return w->ne[0] != s.head;
    }
    if (xdna_name_ends(name, ".attn_output.weight")) {
        return !xdna_width_in(s.query, w->ne[0]);
    }
    return false;
}

static inline void xdna_layer_census_weight(xdna_layer_census & c, const ggml_tensor * w, const xdna_fused_shapes & s) {
    if (w->op != GGML_OP_NONE) {
        return;
    }
    if (xdna_layer_foreign(w, s)) {
        if (c.foreign.empty()) {
            c.foreign = ggml_get_name(w);
        }
        return;
    }
    c.recurrent = c.recurrent || xdna_name_ends(ggml_get_name(w), ".ssm_conv1d.weight");
}

static inline void xdna_layer_census_node(xdna_layer_census & c, const ggml_tensor * n, const xdna_fused_shapes & s) {
    c.whole = c.whole || strncmp(ggml_get_name(n), XDNA_LAYER_OUT, sizeof(XDNA_LAYER_OUT) - 1) == 0;
    for (int k = 0; k < GGML_MAX_SRC && n->src[k]; k++) {
        xdna_layer_census_weight(c, n->src[k], s);
    }
}

// Adds a decode chunk's nodes to `c`. A model's chunks add up, so a graph an eval callback splits tells the same.
static inline void xdna_layer_census_add(xdna_layer_census &         c,
                                         const ggml_tensor * const * nodes,
                                         int                         n_nodes,
                                         const xdna_fused_shapes &   s) {
    for (int i = 0; i < n_nodes; i++) {
        xdna_layer_census_node(c, nodes[i], s);
    }
}

// A model's decode route (xdna_wants_host_decode): what its decode chunks showed of its layers, and, once a chunk
// with a layer norm could tell, whether the fused designs leave its decode to the host.
struct xdna_model_route {
    xdna_layer_census layers;
    bool              known  = false;
    bool              misfit = false;
};

// Whether the fused layers may run a chunk whose own width and weights they take: not once its model's layers told
// that the designs do not run them.
static inline bool xdna_route_fused(const xdna_model_route & r, bool chunk_fits) {
    return chunk_fits && !(r.known && r.misfit);
}
