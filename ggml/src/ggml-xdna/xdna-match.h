#pragma once

// How the graph planners recognise the nodes llama builds around a layer;
// test-xdna-match checks them against the shapes llama's graph code makes.

#include "ggml.h"

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
