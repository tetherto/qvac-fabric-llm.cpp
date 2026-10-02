#include "ggml-metal-fusion.h"

#include "ggml-backend-impl.h"
#include "ggml-metal-common.h"
#include "ggml-metal-device.h"
#include "ggml-metal-impl.h"

#include <algorithm>
#include <string>
#include <vector>

// ---- helpers -------------------------------------------------------------

// true if two tensors live in the same Metal buffer
static bool ggml_metal_fusion_same_buffer(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a || !b) {
        return false;
    }

    ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    ggml_metal_buffer_t ca = (ggml_metal_buffer_t) ba->context;
    ggml_metal_buffer_t cb = (ggml_metal_buffer_t) bb->context;

    return ggml_metal_buffer_get_id(ca, a).metal == ggml_metal_buffer_get_id(cb, b).metal;
}

// ---- pattern checks ------------------------------------------------------

// NORM/RMS_NORM + MUL + ADD: the weight/bias of each fused step must match the norm input
// width, be contiguous rows, and the fused outputs must stay F32
static bool ggml_metal_fusion_check_norm(
        const ggml_metal_fusion      * fusion,
        const ggml_tensor * const    * nodes,
        const ggml_metal_device_props * props,
              ggml_metal_fusion_mode   mode) {
    GGML_UNUSED(props);
    GGML_UNUSED(mode);

    GGML_ASSERT(fusion->n_ops >= 2);

    for (int j = 1; j < fusion->n_ops; j++) {
        // the fused MUL/ADD must read the previous node as src0
        if (nodes[j]->src[0] != nodes[j - 1]) {
            return false;
        }

        // the weight/bias must have the same row width as the norm input
        if (nodes[j]->src[1]->ne[0] != nodes[0]->ne[0]) {
            return false;
        }

        if (!ggml_is_contiguous_rows(nodes[j]->src[1])) {
            return false;
        }

        if (nodes[j]->type != GGML_TYPE_F32) {
            return false;
        }
    }

    return true;
}

// ADD x N: each ADD reads the previous ADD as src0, and all addends must share layout
// (and, in FULL mode, live in the same Metal buffer)
static bool ggml_metal_fusion_check_add_chain(
        const ggml_metal_fusion      * fusion,
        const ggml_tensor * const    * nodes,
        const ggml_metal_device_props * props,
              ggml_metal_fusion_mode   mode) {
    GGML_UNUSED(props);

    GGML_ASSERT(fusion->n_ops >= 2);

    for (int j = 1; j < fusion->n_ops; j++) {
        if (nodes[j]->src[0] != nodes[j - 1]) {
            return false;
        }

        if (!ggml_are_same_layout(nodes[j]->src[1], nodes[j - 1]->src[1])) {
            return false;
        }

        if (mode == GGML_METAL_FUSION_FULL) {
            if (!ggml_metal_fusion_same_buffer(nodes[j]->src[1], nodes[0]->src[1])) {
                return false;
            }
        }
    }

    return true;
}

// GATED_DELTA_NET + CPY: the trailing cpy scatters the gdn state snapshots into the recurrent
// cache, so the gdn kernel writes them straight to the cache and the cpy is elided.
// mirrors ggml_metal_op_can_fuse_gdn_cache (PR #25788). the gdn output has other consumers (the
// attn scores view), so unlike the other patterns this is not an elision chain: the structural
// checks live entirely in this callback (unsafe = true).
static bool ggml_metal_fusion_check_gdn_cache(
        const ggml_metal_fusion      * fusion,
        const ggml_tensor * const    * nodes,
        const ggml_metal_device_props * props,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(props);

    const ggml_tensor * gdn = nodes[0];
    const ggml_tensor * cpy = nodes[1];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }

    if (cpy->op != GGML_OP_CPY || (cpy->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }

    const int64_t S_v      = gdn->src[2]->ne[0];
    const int64_t H        = gdn->src[2]->ne[1];
    const int64_t n_tokens = gdn->src[2]->ne[2];
    const int64_t n_seqs   = gdn->src[2]->ne[3];
    const int64_t K        = ggml_get_op_params_i32(gdn, 0);
    const size_t  tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    const int64_t D         = S_v * S_v * H;
    const int64_t n_written = std::min<int64_t>(n_tokens, K);

    const ggml_tensor * src = cpy->src[0]; // gdn snapshot tail view
    const ggml_tensor * dst = cpy->src[1]; // cache view

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn ||
        src->view_offs != tail_off || !ggml_is_contiguous(src)) {
        return false;
    }

    const int64_t expected_ne[GGML_MAX_DIMS] = { D, n_seqs, n_written, 1 };
    if (dst->type != GGML_TYPE_F32 ||
        !std::equal(expected_ne, expected_ne + GGML_MAX_DIMS, dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) ||
        dst->nb[1] != ggml_row_size(GGML_TYPE_F32, D)) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        // the cache must be allocated so the kernel can write straight to its buffer
        if (dst->data == nullptr) {
            return false;
        }
    }

    return true;
}

// MUL + SIN + SQR + MUL + ADD (snake activation)
static bool ggml_metal_fusion_check_snake(
        const ggml_metal_fusion      * fusion,
        const ggml_tensor * const    * nodes,
        const ggml_metal_device_props * props,
              ggml_metal_fusion_mode   mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(props);
    GGML_UNUSED(mode);

    const ggml_tensor * mul0     = nodes[0];
    const ggml_tensor * sin_node = nodes[1];
    const ggml_tensor * sqr      = nodes[2];
    const ggml_tensor * mul1     = nodes[3];
    const ggml_tensor * add      = nodes[4];

    // x carries the full activation shape, a is the broadcast operand
    const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    // closure check: the trailing add reads the same x as the leading mul
    const ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

    // x is in the supported whitelist and every chain intermediate shares x's type.
    // a and inv_b bind as device const float * in the kernel, so they stay F32.
    const bool types_ok =
        (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
        (a->type    == GGML_TYPE_F32) && (inv_b->type    == GGML_TYPE_F32) &&
        (mul0->type == x->type)       && (sin_node->type == x->type) &&
        (sqr->type  == x->type)       && (mul1->type     == x->type) &&
        (add->type  == x->type);

    // a / inv_b collapse to [1, C, 1, 1], x and add stay 2D
    const bool shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];
    const bool dim_ok =
        (x->ne[2]     == 1) && (x->ne[3]     == 1) &&
        (add->ne[2]   == 1) && (add->ne[3]   == 1) &&
        (a->ne[2]     == 1) && (a->ne[3]     == 1) &&
        (inv_b->ne[2] == 1) && (inv_b->ne[3] == 1);

    // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
    const bool contig_ok =
        ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
        ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

    return types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x;
}

// true if the byte ranges of two tensors overlap in the same Metal buffer
static bool ggml_metal_fusion_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    const ggml_metal_buffer_id bid_a = ggml_metal_buffer_get_id((ggml_metal_buffer_t) ba->context, a);
    const ggml_metal_buffer_id bid_b = ggml_metal_buffer_get_id((ggml_metal_buffer_t) bb->context, b);

    if (bid_a.metal == nullptr || bid_a.metal != bid_b.metal) {
        return false;
    }

    return bid_a.offs <= bid_b.offs
        ? bid_b.offs - bid_a.offs < ggml_nbytes(a)
        : bid_a.offs - bid_b.offs < ggml_nbytes(b);
}

// MUL + MUL_MAT(hadamard): the sign vector folds into the FWHT kernel, so the MUL is elided.
// the MUL_MAT reads the MUL through a RESHAPE, which is not a chain link, so the checks live
// here (unsafe = true); the encoder verifies MUL/RESHAPE have no other consumers.
static bool ggml_metal_fusion_check_fwht_signed(
        const ggml_metal_fusion      * fusion,
        const ggml_tensor * const    * nodes,
        const ggml_metal_device_props * props,
              ggml_metal_fusion_mode   mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(props);

    const ggml_tensor * mul     = nodes[0];
    const ggml_tensor * mm      = nodes[1];
    const ggml_tensor * reshape = mm->src[1];

    if (reshape == nullptr || reshape->op != GGML_OP_RESHAPE || reshape->src[0] != mul ||
        !ggml_metal_op_mul_mat_use_fwht(mm)) {
        return false;
    }

    const ggml_tensor * x     = ggml_are_same_shape(mul, mul->src[0]) ? mul->src[0] : mul->src[1];
    const ggml_tensor * signs = x == mul->src[0] ? mul->src[1] : mul->src[0];
    const int64_t n = mm->src[0]->ne[0];

    const bool ok =
        signs->type == GGML_TYPE_F32 &&
        signs->ne[1] == 1 && signs->ne[2] == 1 && signs->ne[3] == 1 &&
        x->type == GGML_TYPE_F32 && mul->type == GGML_TYPE_F32 && mm->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(x) && ggml_is_contiguous(signs) && ggml_is_contiguous(mm) &&
        signs->ne[0] == x->ne[0] && signs->ne[0] % n == 0;

    if (!ok) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        // the kernel reads x and writes mm in one pass
        if (ggml_metal_fusion_overlap(x, mm)) {
            return false;
        }
    }

    return true;
}

// MUL_MAT + ADD of an f32 non-weight: the reorder packs it without reading row counts, so ubatch sizes share one order;
// the encoder fuses only a same-shape residual in the few-row MMA store, which the sum may overlap only in place
static bool ggml_metal_fusion_check_mul_mat_add(
        const ggml_metal_fusion      * fusion,
        const ggml_tensor * const    * nodes,
        const ggml_metal_device_props * props,
              ggml_metal_fusion_mode   mode) {
    GGML_UNUSED(fusion);

    const ggml_tensor * mm  = nodes[0];
    const ggml_tensor * add = nodes[1];

    if (ggml_metal_mul_mat_add_operand(mm, add) == nullptr ||
        !ggml_metal_mul_mat_may_use_mma(mm, props->supports_gpu_family_apple7, props->has_tensor)) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_STRUCTURAL) {
        return true;
    }

    const ggml_tensor * res = ggml_metal_mul_mat_add_residual(mm, add);

    if (res == nullptr || ggml_metal_mul_mat_use_nc(mm) ||
        !ggml_metal_mul_mat_use_mma(mm, props->supports_gpu_family_apple7, props->has_tensor)) {
        return false;
    }

    return !ggml_metal_fusion_overlap(add, mm->src[0]) && !ggml_metal_fusion_overlap(add, mm->src[1]) &&
        (add->data == res->data || !ggml_metal_fusion_overlap(add, res));
}

// true if CPY a copies an f32 view of one tensor into an f32 view of another and can start a batch
static bool ggml_metal_fusion_cpy_batch_start(const ggml_tensor * a) {
    return (a->flags & GGML_TENSOR_FLAG_COMPUTE) && a->src[0]->type == GGML_TYPE_F32 && a->type == GGML_TYPE_F32 &&
        a->src[0]->view_src && a->view_src && a->src[0]->view_src != a->view_src;
}

// true if CPY b moves the same view layout between the same two tensors as CPY a
static bool ggml_metal_fusion_cpy_same_layout(const ggml_tensor * a, const ggml_tensor * b) {
    return (b->flags & GGML_TENSOR_FLAG_COMPUTE) &&
        b->src[0]->type == a->src[0]->type && b->type == a->type &&
        b->src[0]->view_src == a->src[0]->view_src && b->view_src == a->view_src &&
        ggml_are_same_shape(a->src[0], b->src[0]) && ggml_are_same_stride(a->src[0], b->src[0]) &&
        ggml_are_same_shape(a, b) && ggml_are_same_stride(a, b);
}

// true if the first n copies all have the layout of the first one
static bool ggml_metal_fusion_cpy_batch_same_layout(const ggml_tensor * const * nodes, int n) {
    for (int j = 1; j < n; j++) {
        if (!ggml_metal_fusion_cpy_same_layout(nodes[0], nodes[j])) {
            return false;
        }
    }

    return true;
}

// true if copy next writes where one of the n copies before it reads or writes, reads where one of them writes, or uses other buffers than the first copy
static bool ggml_metal_fusion_cpy_batch_conflicts(const ggml_tensor * const * nodes, int n, const ggml_tensor * next) {
    if (!ggml_metal_fusion_same_buffer(next->src[0], nodes[0]->src[0]) || !ggml_metal_fusion_same_buffer(next, nodes[0])) {
        return true;
    }

    for (int i = 0; i < n; i++) {
        const ggml_tensor * prev = nodes[i];
        if (ggml_metal_fusion_overlap(next, prev) || ggml_metal_fusion_overlap(next->src[0], prev) ||
            ggml_metal_fusion_overlap(next, prev->src[0])) {
            return true;
        }
    }

    return false;
}

// true if no copy of the batch conflicts with the copies before it
static bool ggml_metal_fusion_cpy_batch_disjoint(const ggml_tensor * const * nodes, int n) {
    for (int j = 1; j < n; j++) {
        if (ggml_metal_fusion_cpy_batch_conflicts(nodes, j, nodes[j])) {
            return false;
        }
    }

    return true;
}

// CPY x N: same-layout f32 copies between two tensors run as one dispatch; they do not chain, so the checks live here (unsafe = true)
// the batch runs the copies concurrently, so in FULL mode no copy may write where another one reads or writes
static bool ggml_metal_fusion_check_cpy_batch(
        const ggml_metal_fusion      * fusion,
        const ggml_tensor * const    * nodes,
        const ggml_metal_device_props * props,
              ggml_metal_fusion_mode   mode) {
    GGML_UNUSED(props);

    if (!ggml_metal_fusion_cpy_batch_start(nodes[0]) || !ggml_metal_fusion_cpy_batch_same_layout(nodes, fusion->n_ops)) {
        return false;
    }

    return mode != GGML_METAL_FUSION_FULL || ggml_metal_fusion_cpy_batch_disjoint(nodes, fusion->n_ops);
}

// ---- patterns ------------------------------------------------------------

static const ggml_op ops_norm_mul[]         = { GGML_OP_NORM, GGML_OP_MUL };
static const ggml_op ops_norm_mul_add[]     = { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ADD };
static const ggml_op ops_rms_norm_mul[]     = { GGML_OP_RMS_NORM, GGML_OP_MUL };
static const ggml_op ops_rms_norm_mul_add[] = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD };

static const ggml_op ops_add_2[] = { GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_add_3[] = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_add_4[] = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_add_5[] = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_add_6[] = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_add_7[] = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_snake[] = { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD };

static const ggml_op ops_gdn_cache[] = { GGML_OP_GATED_DELTA_NET, GGML_OP_CPY };

static const ggml_op ops_fwht_signed[] = { GGML_OP_MUL, GGML_OP_MUL_MAT };

static const ggml_op ops_mul_mat_add[] = { GGML_OP_MUL_MAT, GGML_OP_ADD };

// a batch of n copies reads the first n ops
static const ggml_op ops_cpy_batch[GGML_METAL_CPY_BATCH_MAX] = {
    GGML_OP_CPY, GGML_OP_CPY, GGML_OP_CPY, GGML_OP_CPY, GGML_OP_CPY, GGML_OP_CPY, GGML_OP_CPY, GGML_OP_CPY,
    GGML_OP_CPY, GGML_OP_CPY, GGML_OP_CPY, GGML_OP_CPY, GGML_OP_CPY, GGML_OP_CPY, GGML_OP_CPY, GGML_OP_CPY,
};

static const ggml_metal_fusion ggml_metal_fusions[] = {
    { GGML_METAL_FUSION_NORM_MUL,     ops_norm_mul,         2, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD, ops_norm_mul_add,     3, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL,     ops_rms_norm_mul,     2, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD, ops_rms_norm_mul_add, 3, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_ADD_CHAIN,    ops_add_2,            2, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,    ops_add_3,            3, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,    ops_add_4,            4, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,    ops_add_5,            5, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,    ops_add_6,            6, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,    ops_add_7,            7, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_SNAKE,        ops_snake,            5, false, ggml_metal_fusion_check_snake },
    { GGML_METAL_FUSION_GDN_CACHE,    ops_gdn_cache,        2, true,  ggml_metal_fusion_check_gdn_cache },
    { GGML_METAL_FUSION_FWHT_SIGNED,  ops_fwht_signed,      2, true,  ggml_metal_fusion_check_fwht_signed },
    { GGML_METAL_FUSION_MUL_MAT_ADD,  ops_mul_mat_add,      2, false, ggml_metal_fusion_check_mul_mat_add },
    // longest batch first, so ggml_metal_fusion_next checks no shorter batch once one matches
    { GGML_METAL_FUSION_CPY_BATCH,    ops_cpy_batch,       16, true,  ggml_metal_fusion_check_cpy_batch },
    { GGML_METAL_FUSION_CPY_BATCH,    ops_cpy_batch,       15, true,  ggml_metal_fusion_check_cpy_batch },
    { GGML_METAL_FUSION_CPY_BATCH,    ops_cpy_batch,       14, true,  ggml_metal_fusion_check_cpy_batch },
    { GGML_METAL_FUSION_CPY_BATCH,    ops_cpy_batch,       13, true,  ggml_metal_fusion_check_cpy_batch },
    { GGML_METAL_FUSION_CPY_BATCH,    ops_cpy_batch,       12, true,  ggml_metal_fusion_check_cpy_batch },
    { GGML_METAL_FUSION_CPY_BATCH,    ops_cpy_batch,       11, true,  ggml_metal_fusion_check_cpy_batch },
    { GGML_METAL_FUSION_CPY_BATCH,    ops_cpy_batch,       10, true,  ggml_metal_fusion_check_cpy_batch },
    { GGML_METAL_FUSION_CPY_BATCH,    ops_cpy_batch,        9, true,  ggml_metal_fusion_check_cpy_batch },
    { GGML_METAL_FUSION_CPY_BATCH,    ops_cpy_batch,        8, true,  ggml_metal_fusion_check_cpy_batch },
    { GGML_METAL_FUSION_CPY_BATCH,    ops_cpy_batch,        7, true,  ggml_metal_fusion_check_cpy_batch },
    { GGML_METAL_FUSION_CPY_BATCH,    ops_cpy_batch,        6, true,  ggml_metal_fusion_check_cpy_batch },
    { GGML_METAL_FUSION_CPY_BATCH,    ops_cpy_batch,        5, true,  ggml_metal_fusion_check_cpy_batch },
    { GGML_METAL_FUSION_CPY_BATCH,    ops_cpy_batch,        4, true,  ggml_metal_fusion_check_cpy_batch },
    { GGML_METAL_FUSION_CPY_BATCH,    ops_cpy_batch,        3, true,  ggml_metal_fusion_check_cpy_batch },
    { GGML_METAL_FUSION_CPY_BATCH,    ops_cpy_batch,        2, true,  ggml_metal_fusion_check_cpy_batch },
};

static_assert(GGML_METAL_CPY_BATCH_MAX <= GGML_METAL_FUSION_MAX, "a copy batch must fit in one fusion");

const ggml_metal_fusion * ggml_metal_fusion_all(int * n) {
    *n = (int) sizeof(ggml_metal_fusions) / sizeof(ggml_metal_fusions[0]);

    return ggml_metal_fusions;
}

// ---- shared fusion info ---------------------------------------------------

static std::string ggml_metal_fusion_label(const ggml_metal_fusion * fusion) {
    GGML_ASSERT(fusion != nullptr);

    std::string label;
    for (int j = 0; j < fusion->n_ops; j++) {
        if (j > 0) {
            label += '+';
        }
        label += ggml_op_name(fusion->ops[j]);
    }
    return label;
}

struct ggml_metal_fusion_info {
    std::vector<std::string> labels;
    std::vector<uint64_t>    counts;
    bool enabled;
    bool stats;
    bool labels_set;
    int  debug;
};

struct ggml_metal_fusion_info * ggml_metal_fusion_info_init(bool enabled, int debug) {
    ggml_metal_fusion_info * finfo = new ggml_metal_fusion_info;
    finfo->enabled    = enabled;
    finfo->stats      = debug > 0;
    finfo->labels_set = false;
    finfo->debug      = debug;

    if (finfo->stats) {
        ggml_metal_fusion_info_labels_init(finfo);
    }

    return finfo;
}

void ggml_metal_fusion_info_free(struct ggml_metal_fusion_info * finfo) {
    delete finfo;
}

bool ggml_metal_fusion_info_enabled(const struct ggml_metal_fusion_info * finfo) {
    return finfo->enabled;
}

bool ggml_metal_fusion_info_stats(const struct ggml_metal_fusion_info * finfo) {
    return finfo->stats;
}

int ggml_metal_fusion_info_debug(const struct ggml_metal_fusion_info * finfo) {
    return finfo->debug;
}

int ggml_metal_fusion_info_n_fusions(const struct ggml_metal_fusion_info * finfo) {
    return (int) finfo->labels.size();
}

const char * ggml_metal_fusion_info_label(const struct ggml_metal_fusion_info * finfo, int idx) {
    GGML_ASSERT(idx >= 0 && idx < (int) finfo->labels.size());
    return finfo->labels[idx].c_str();
}

uint64_t ggml_metal_fusion_info_count(const struct ggml_metal_fusion_info * finfo, int idx) {
    GGML_ASSERT(idx >= 0 && idx < (int) finfo->counts.size());
    return finfo->counts[idx];
}

void ggml_metal_fusion_info_count_fusion(struct ggml_metal_fusion_info * finfo, const struct ggml_metal_fusion * fusion) {
    if (!finfo->stats || fusion == nullptr) {
        return;
    }

    int n = 0;
    const ggml_metal_fusion * all = ggml_metal_fusion_all(&n);

    int idx = -1;
    for (int i = 0; i < n; i++) {
        if (&all[i] == fusion) {
            idx = i;
            break;
        }
    }

    if (idx >= 0 && idx < (int) finfo->counts.size()) {
        finfo->counts[idx]++;
    }
}

void ggml_metal_fusion_info_set_enabled(struct ggml_metal_fusion_info * finfo, bool enabled) {
    finfo->enabled = enabled;
}

void ggml_metal_fusion_info_labels_init(struct ggml_metal_fusion_info * finfo) {
    if (finfo->labels_set) {
        return;
    }

    int n = 0;
    const ggml_metal_fusion * all = ggml_metal_fusion_all(&n);

    finfo->labels.clear();
    finfo->counts.assign(n, 0);
    finfo->labels.reserve(n);

    for (int i = 0; i < n; i++) {
        finfo->labels.emplace_back(ggml_metal_fusion_label(&all[i]));
    }

    finfo->labels_set = true;
}

void ggml_metal_fusion_info_stats_init(struct ggml_metal_fusion_info * finfo) {
    finfo->stats = true;
    ggml_metal_fusion_info_labels_init(finfo);
}

void ggml_metal_fusion_info_stats_reset(struct ggml_metal_fusion_info * finfo) {
    std::fill(finfo->counts.begin(), finfo->counts.end(), 0);
}

int ggml_metal_fusion_info_stats_get(const struct ggml_metal_fusion_info * finfo, const char ** labels, uint64_t * counts, int n) {
    const int n_fusions = (int) finfo->labels.size();

    if (labels == nullptr) {
        return n_fusions;
    }

    const int n_fill = std::min(n, n_fusions);
    for (int i = 0; i < n_fill; i++) {
        labels[i] = finfo->labels[i].c_str();
        if (counts != nullptr) {
            counts[i] = finfo->counts[i];
        }
    }

    return n_fill;
}

// ---- queries -------------------------------------------------------------

// find the longest pattern matching the node sequence starting at idx
// (idx is a position in node_idxs, which maps to graph node indices)
const ggml_metal_fusion * ggml_metal_fusion_next(
        const ggml_cgraph * gf,
        const int * node_idxs,
        int n_idxs,
        int idx,
        const ggml_metal_device_props * props,
        ggml_metal_fusion_mode mode,
        int * n_out) {
    int n = 0;
    const ggml_metal_fusion * all = ggml_metal_fusion_all(&n);

    const ggml_metal_fusion * res = nullptr;
    int best = 1;

    for (int i = 0; i < n; i++) {
        const ggml_metal_fusion * fusion = &all[i];

        // only look for a longer match than the current best
        if (fusion->n_ops <= best) {
            continue;
        }
        if (idx + fusion->n_ops > n_idxs) {
            continue;
        }

        const ggml_tensor * nodes[GGML_METAL_FUSION_MAX];

        // the op sequence must match exactly
        bool ok = true;
        for (int j = 0; j < fusion->n_ops; j++) {
            nodes[j] = gf->nodes[node_idxs[idx + j]];
            if (nodes[j]->op != fusion->ops[j]) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            continue;
        }

        if (!fusion->unsafe) {
            // common element-wise chain constraints: each node reads the previous one,
            // and all nodes have the same shape
            for (int j = 1; j < fusion->n_ops && ok; j++) {
                if (nodes[j]->src[0] != nodes[j - 1] && nodes[j]->src[1] != nodes[j - 1]) {
                    ok = false;
                    break;
                }
                if (!ggml_are_same_shape(nodes[j], nodes[j - 1])) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                continue;
            }

            // all current fusions are single-output elision chains, so the last node is the only output
            // TODO: multi-output fusions: store pattern-relative offsets in the table and translate them here
            int outputs_buf[1];
            outputs_buf[0] = node_idxs[idx + fusion->n_ops - 1];

            // structural subgraph checks (op sequence, elidable uses, view containment)
            if (!ggml_can_fuse_subgraph_ext(gf, node_idxs + idx, fusion->n_ops, fusion->ops, outputs_buf, 1)) {
                continue;
            }
        }

        // pattern-specific checks (the sole validator for unsafe patterns)
        if (fusion->check && !fusion->check(fusion, nodes, props, mode)) {
            continue;
        }

        best = fusion->n_ops;
        res = fusion;
    }

    *n_out = best;

    return res;
}

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused, chaining patterns back-to-back. matching runs on the same filtered (view
// transparent) node sequence that the compute phase uses, so the returned count is the raw index
// span from idx to the last matched node (intermediate views are packed along).
int ggml_metal_fusion_max(const ggml_cgraph * gf, int idx, const ggml_metal_device_props * props) {
    // an empty/view node cannot start a pattern - pack it alone
    if (ggml_op_is_empty(gf->nodes[idx]->op) || ggml_is_empty(gf->nodes[idx])) {
        return 1;
    }

    // collect the non-empty node indices starting at idx
    int idxs[GGML_METAL_FUSION_MAX];
    int n_idxs = 0;
    for (int i = idx; i < gf->n_nodes && n_idxs < GGML_METAL_FUSION_MAX; i++) {
        if (!ggml_op_is_empty(gf->nodes[i]->op) && !ggml_is_empty(gf->nodes[i])) {
            idxs[n_idxs++] = i;
        }
    }
    if (n_idxs == 0) {
        return 1;
    }

    int total = 0;
    int i_f = 0;

    while (i_f < n_idxs && total < GGML_METAL_FUSION_MAX) {
        int len = 1;
        const ggml_metal_fusion * fusion = ggml_metal_fusion_next(gf, idxs, n_idxs, i_f, props, GGML_METAL_FUSION_STRUCTURAL, &len);
        if (!fusion || total + len > GGML_METAL_FUSION_MAX) {
            break;
        }

        total += len;
        i_f += len;
    }

    if (i_f == 0) {
        return 1;
    }

    // map the matched non-empty nodes back to the raw index span; its views do not count toward GGML_METAL_FUSION_MAX
    return idxs[i_f - 1] - idx + 1;
}
