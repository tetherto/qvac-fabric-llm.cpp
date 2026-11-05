#pragma once

// How the graph planners recognise the nodes llama builds around a layer;
// test-xdna-match checks them against the shapes llama's graph code makes.

#include "ggml.h"

#include <cstring>

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

// A FLASH_ATTN_EXT the attention kernels can take: q, k and v, no sinks
// (src[4]) and no K/V row lists (src[5], ggml_flash_attn_ext_rows). The kernels
// read each head's keys and values as the leading cells of the K/V view; with
// row lists, mask column i stands for cache row kv_rows[i] instead.
static inline bool xdna_fa_plain(const ggml_tensor * n) {
    return n && n->op == GGML_OP_FLASH_ATTN_EXT && n->src[0] && n->src[1] && n->src[2] && !n->src[4] &&
           !n->src[5];
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
