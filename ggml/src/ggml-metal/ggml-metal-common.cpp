#include "ggml-metal-common.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <vector>

// represents a memory range (i.e. an interval from a starting address p0 to an ending address p1 in a given buffer pb)
// the type indicates whether it is a source range (i.e. ops read data from it) or a destination range (i.e. ops write data to it)
struct ggml_mem_range {
    uint64_t pb; // buffer id

    uint64_t p0; // begin
    uint64_t p1; // end

    ggml_mem_range_type pt;
};

struct ggml_mem_ranges {
    std::vector<ggml_mem_range> ranges;

    int debug = 0;
};

ggml_mem_ranges_t ggml_mem_ranges_init(int debug) {
    auto * res = new ggml_mem_ranges;

    res->ranges.reserve(256);
    res->debug = debug;

    return res;
}

void ggml_mem_ranges_free(ggml_mem_ranges_t mrs) {
    delete mrs;
}

void ggml_mem_ranges_reset(ggml_mem_ranges_t mrs) {
    mrs->ranges.clear();
}

static bool ggml_mem_ranges_add(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    mrs->ranges.push_back(mr);

    return true;
}

static ggml_mem_range ggml_mem_range_from_tensor(const ggml_tensor * tensor, ggml_mem_range_type pt) {
    // always use the base tensor
    tensor = tensor->view_src ? tensor->view_src : tensor;

    GGML_ASSERT(!tensor->view_src);

    ggml_mem_range mr;

    if (tensor->buffer) {
        // when the tensor is allocated, use the actual memory address range in the buffer
        //
        // take the actual allocated size with ggml_backend_buft_get_alloc_size()
        // this can be larger than the tensor size if the buffer type allocates extra memory
        // ref: https://github.com/ggml-org/llama.cpp/pull/15966
        mr = {
            /*.pb =*/ (uint64_t) tensor->buffer,
            /*.p0 =*/ (uint64_t) tensor->data,
            /*.p1 =*/ (uint64_t) tensor->data + ggml_backend_buft_get_alloc_size(tensor->buffer->buft, tensor),
            /*.pt =*/ pt,
        };
    } else {
        // otherwise, the pointer address is used as an unique id of the memory ranges
        //   that the tensor will be using when it is allocated
        mr = {
            /*.pb =*/ (uint64_t) tensor,
            /*.p0 =*/ 0,    //
            /*.p1 =*/ 1024, // [0, 1024) is a dummy range, not used
            /*.pt =*/ pt,
        };
    };

    return mr;
}

static ggml_mem_range ggml_mem_range_from_tensor_src(const ggml_tensor * tensor) {
    return ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_SRC);
}

static ggml_mem_range ggml_mem_range_from_tensor_dst(const ggml_tensor * tensor) {
    return ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_DST);
}

static bool ggml_mem_ranges_add_src(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    if (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add src range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    return ggml_mem_ranges_add(mrs, mr);
}

static bool ggml_mem_ranges_add_dst(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    if (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add dst range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    return ggml_mem_ranges_add(mrs, mr);
}

bool ggml_mem_ranges_add(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i]) {
            ggml_mem_ranges_add_src(mrs, tensor->src[i]);
        }
    }

    return ggml_mem_ranges_add_dst(mrs, tensor);
}

static bool ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    for (size_t i = 0; i < mrs->ranges.size(); i++) {
        const auto & cmp = mrs->ranges[i];

        // two memory ranges cannot intersect if they are in different buffers
        if (mr.pb != cmp.pb) {
            continue;
        }

        // intersecting source ranges are allowed
        if (mr.pt == MEM_RANGE_TYPE_SRC && cmp.pt == MEM_RANGE_TYPE_SRC) {
            continue;
        }

        if (mr.p0 < cmp.p1 && mr.p1 >= cmp.p0) {
            if (mrs->debug > 2) {
                GGML_LOG_DEBUG("%s: the %s range buf=%lld, [%lld, %lld) overlaps with a previous %s range buf=%lld, [%lld, %lld)\n",
                        __func__,
                        mr.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        mr.pb, mr.p0, mr.p1,
                        cmp.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        cmp.pb, cmp.p0, cmp.p1);
            }

            return false;
        }
    }

    return true;
}

static bool ggml_mem_ranges_check_src(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    const bool res = ggml_mem_ranges_check(mrs, mr);

    return res;
}

static bool ggml_mem_ranges_check_dst(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    const bool res = ggml_mem_ranges_check(mrs, mr);

    return res;
}

bool ggml_mem_ranges_check(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i]) {
            if (!ggml_mem_ranges_check_src(mrs, tensor->src[i])) {
                return false;
            }
        }
    }

    return ggml_mem_ranges_check_dst(mrs, tensor);
}

struct node_info {
    ggml_tensor * node;

    std::vector<ggml_tensor *> fused;

    // outputs of the fused group other than dst() (e.g. the ids of a fused topk-moe)
    std::vector<ggml_tensor *> extra_dsts;

    ggml_op op() const {
        return node->op;
    }

    const ggml_tensor * dst() const {
        return fused.empty() ? node : fused.back();
    }

    bool is_empty() const {
        return ggml_op_is_empty(node->op);
    }

    void add_fused(ggml_tensor * t) {
        fused.push_back(t);
    }
};

static std::vector<int> ggml_metal_graph_optimize_reorder(const std::vector<node_info> & nodes) {
    // helper to add node src and dst ranges
    const auto & h_add = [](ggml_mem_ranges_t mrs, const node_info & node) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (node.node->src[i]) {
                if (!ggml_mem_ranges_add_src(mrs, node.node->src[i])) {
                    return false;
                }
            }
        }

        // keep track of the sources of the fused nodes as well
        for (const auto * fused : node.fused) {
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                if (fused->src[i]) {
                    if (!ggml_mem_ranges_add_src(mrs, fused->src[i])) {
                        return false;
                    }
                }
            }
        }

        for (const auto * d : node.extra_dsts) {
            if (!ggml_mem_ranges_add_dst(mrs, d)) {
                return false;
            }
        }

        return ggml_mem_ranges_add_dst(mrs, node.dst());
    };

    // helper to check if a node can run concurrently with the existing set of nodes
    const auto & h_check = [](ggml_mem_ranges_t mrs, const node_info & node) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (node.node->src[i]) {
                if (!ggml_mem_ranges_check_src(mrs, node.node->src[i])) {
                    return false;
                }
            }
        }

        for (const auto * fused : node.fused) {
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                if (fused->src[i]) {
                    if (!ggml_mem_ranges_check_src(mrs, fused->src[i])) {
                        return false;
                    }
                }
            }
        }

        for (const auto * d : node.extra_dsts) {
            if (!ggml_mem_ranges_check_dst(mrs, d)) {
                return false;
            }
        }

        return ggml_mem_ranges_check_dst(mrs, node.dst());
    };

    // perform reorders only across these types of ops
    // can be expanded when needed
    const auto & h_safe = [](ggml_op op) {
        switch (op) {
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID:
            case GGML_OP_ROPE:
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
            case GGML_OP_GROUP_NORM:
            case GGML_OP_L2_NORM:
            case GGML_OP_SUM_ROWS:
            case GGML_OP_SSM_CONV:
            case GGML_OP_SSM_SCAN:
            case GGML_OP_CLAMP:
            case GGML_OP_TRI:
            case GGML_OP_DIAG:
            case GGML_OP_MUL:
            case GGML_OP_ADD:
            case GGML_OP_SUB:
            case GGML_OP_DIV:
            case GGML_OP_GLU:
            case GGML_OP_SCALE:
            case GGML_OP_UNARY:
            case GGML_OP_GET_ROWS:
            case GGML_OP_SET_ROWS:
            case GGML_OP_SET:
            case GGML_OP_CPY:
            case GGML_OP_CONT:
            case GGML_OP_REPEAT:
                return true;
            default:
                return ggml_op_is_empty(op);
        }
    };

    const int n = nodes.size();

    std::vector<int> res;
    res.reserve(n);

    std::vector<bool> used(n, false);

    // the memory ranges for the set of currently concurrent nodes
    ggml_mem_ranges_t mrs0 = ggml_mem_ranges_init(0);

    // the memory ranges for the set of nodes that haven't been processed yet, when looking forward for a node to reorder
    ggml_mem_ranges_t mrs1 = ggml_mem_ranges_init(0);

    for (int i0 = 0; i0 < n; i0++) {
        if (used[i0]) {
            continue;
        }

        const auto & node0 = nodes[i0];

        // the node is not concurrent with the existing concurrent set, so we have to "put a barrier" (i.e reset mrs0)
        // but before we do that, look forward for some other nodes that can be added to the concurrent set mrs0
        //
        // note: we can always add empty nodes to the concurrent set as they don't read nor write anything
        if (!node0.is_empty() && !h_check(mrs0, node0)) {
            // this will hold the set of memory ranges from the nodes that haven't been processed yet
            // if a node is not concurrent with this set, we cannot reorder it
            ggml_mem_ranges_reset(mrs1);

            // initialize it with the current node
            h_add(mrs1, node0);

            // that many nodes forward to search for a concurrent node
            constexpr int N_FORWARD = 64;

            for (int i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
                if (used[i1]) {
                    continue;
                }

                const auto & node1 = nodes[i1];

                // disallow reordering of certain ops
                if (!h_safe(node1.op())) {
                    break;
                }

                const bool is_empty = node1.is_empty();

                // to reorder a node and add it to the concurrent set, it has to be:
                //   + empty or concurrent with all nodes in the existing concurrent set (mrs0)
                //   + concurrent with all nodes prior to it that haven't been processed yet (mrs1)
                if ((is_empty || h_check(mrs0, node1)) && h_check(mrs1, node1)) {
                    // add the node to the existing concurrent set (i.e. reorder it for early execution)
                    h_add(mrs0, node1);
                    res.push_back(i1);

                    // mark as used, so we skip re-processing it later
                    used[i1] = true;
                } else {
                    // expand the set of nodes that haven't been processed yet
                    h_add(mrs1, node1);
                }
            }

            // finalize the concurrent set and begin a new one
            ggml_mem_ranges_reset(mrs0);
        }

        // expand the concurrent set with the current node
        {
            h_add(mrs0, node0);
            res.push_back(i0);
        }
    }

    ggml_mem_ranges_free(mrs0);
    ggml_mem_ranges_free(mrs1);

    return res;
}

static bool ggml_metal_is_view_or_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
int ggml_metal_try_gdn_cache_fusion(
        const ggml_cgraph * gf, int node_idx, const ggml_tensor ** cache, int64_t * slot_stride) {
    const ggml_tensor * gdn = gf->nodes[node_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const ggml_tensor * src_v     = gdn->src[2];
    const int64_t       S_v       = src_v->ne[0];
    const int64_t       H         = src_v->ne[1];
    const int64_t       n_tokens  = src_v->ne[2];
    const int64_t       n_seqs    = src_v->ne[3];
    const int64_t       D         = S_v * S_v * H;
    const int64_t       K         = ggml_get_op_params_i32(gdn, 0);
    const int64_t       n_written = n_tokens < K ? n_tokens : K;

    const size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < gf->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = gf->nodes[j];
        if (ggml_metal_is_view_or_noop(n)) {
            // views of the snapshot tail other than the matched cpy src are checked below
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0];
    const ggml_tensor * dst = cpy->src[1];

    if (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel assumes)
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 ||
        dst->ne[0] != D || dst->ne[1] != n_seqs || dst->ne[2] != n_written || dst->ne[3] != 1 ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || dst->nb[1] != ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    // fusion writes snapshots to the cache and leaves the gdn tail unwritten, so the only
    // consumer of that tail must be this cpy. attn-prefix views (view_offs < tail_off) are fine.
    int src_idx = -1;
    int tail_views = 0;
    for (int j = 0; j < gf->n_nodes; ++j) {
        const ggml_tensor * n = gf->nodes[j];
        if (n == src) {
            src_idx = j;
        }
        if (n->op == GGML_OP_VIEW && n->src[0] == gdn && n->view_offs >= tail_off) {
            ++tail_views;
            if (n != src) {
                return 0;
            }
        }
        if (n == gdn || n == cpy || ggml_metal_is_view_or_noop(n)) {
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (n->src[s] == gdn) {
                return 0;
            }
        }
    }
    if (tail_views != 1 || src_idx < 0 || ggml_node_get_use_count(gf, src_idx) != 1) {
        return 0;
    }

    if (cache) {
        *cache = dst;
    }
    if (slot_stride) {
        *slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;
    }
    return skip;
}

// extra nodes to keep with gf->nodes[i] so later reorder cannot split a metal fusion
// extra_dsts receives the outputs of the pack other than the last node (see node_info::extra_dsts)
static int ggml_metal_graph_optimize_pack(const ggml_cgraph * gf, int i, std::vector<ggml_tensor *> & extra_dsts) {
    const int n = gf->n_nodes;
    ggml_tensor ** nodes = gf->nodes;

    if (nodes[i]->op == GGML_OP_GATED_DELTA_NET) {
        const int skip = ggml_metal_try_gdn_cache_fusion(gf, i, nullptr, nullptr);
        if (skip > 0) {
            // gdn attn output is consumed outside the pack; cpy (fused.back()) covers the cache
            extra_dsts.push_back(nodes[i]);
        }
        return skip;
    }

    if (nodes[i]->op == GGML_OP_SOFT_MAX) {
        // keep in sync with ggml_metal_op_try_topk_moe
        static const ggml_op topk_ops[] = {
            GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
            GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE,
            GGML_OP_SCALE,
        };
        const int lens[] = { 11, 10, 6, 5 };
        for (int n_ops : lens) {
            if (i + n_ops > n) {
                continue;
            }
            const int outs[] = { i + 3, i + n_ops - 1 };
            if (ggml_can_fuse_subgraph(gf, i, n_ops, topk_ops, outs, 2)) {
                extra_dsts.push_back(nodes[outs[0]]);
                return n_ops - 1;
            }
        }
    }

    const ggml_op op0 = nodes[i]->op;
    if (op0 == GGML_OP_MUL_MAT || op0 == GGML_OP_MUL_MAT_ID) {
        if (i + 3 <= n) {
            const ggml_op ops[] = { op0, op0, GGML_OP_GLU };
            const int out[] = { i + 2 };
            if (ggml_can_fuse_subgraph(gf, i, 3, ops, out, 1)) {
                return 2;
            }
        }
        if (op0 == GGML_OP_MUL_MAT_ID && i + 4 <= n) {
            const ggml_op ops[] = { GGML_OP_MUL_MAT_ID, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_GLU };
            const int out[] = { i + 3 };
            if (ggml_can_fuse_subgraph(gf, i, 4, ops, out, 1)) {
                return 3;
            }
        }
        if (op0 == GGML_OP_MUL_MAT_ID && i + 2 <= n) {
            const ggml_op ops[] = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL };
            const int out[] = { i + 1 };
            if (ggml_can_fuse_subgraph(gf, i, 2, ops, out, 1) &&
                    (nodes[i + 1]->src[0] == nodes[i] || nodes[i + 1]->src[1] == nodes[i])) {
                return 1;
            }
        }
    }

    if (op0 == GGML_OP_SSM_CONV && i + 2 <= n) {
        const ggml_op ops[] = { GGML_OP_SSM_CONV, GGML_OP_UNARY };
        const int out[] = { i + 1 };
        if (ggml_can_fuse_subgraph(gf, i, 2, ops, out, 1) &&
                ggml_get_unary_op(nodes[i + 1]) == GGML_UNARY_OP_SILU) {
            return 1;
        }
    }

    if (op0 == GGML_OP_UNARY && i + 2 <= n) {
        const ggml_unary_op uop = ggml_get_unary_op(nodes[i]);
        if (uop == GGML_UNARY_OP_SILU ||
                uop == GGML_UNARY_OP_SIGMOID ||
                uop == GGML_UNARY_OP_SOFTPLUS) {
            const ggml_op ops[] = { GGML_OP_UNARY, GGML_OP_MUL };
            const int out[] = { i + 1 };
            if (ggml_can_fuse_subgraph(gf, i, 2, ops, out, 1) &&
                    (nodes[i + 1]->src[0] == nodes[i] || nodes[i + 1]->src[1] == nodes[i]) &&
                    ggml_are_same_shape(nodes[i]->src[0], nodes[i + 1])) {
                return 1;
            }
        }
    }

    return 0;
}

void ggml_graph_optimize(ggml_cgraph * gf) {
    constexpr int MAX_FUSE = 16;

    const int n = gf->n_nodes;

    enum ggml_op ops[MAX_FUSE];

    std::vector<node_info> nodes;
    nodes.reserve(gf->n_nodes);

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we first pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    for (int i = 0; i < n; i++) {
        node_info node = {
            /*.node =*/ gf->nodes[i],
            /*.fused =*/ {},
            /*.extra_dsts =*/ {},
        };

        int n_extra = 0;

        // fuse only ops that start with these operations
        // can be expanded when needed
        if (node.op() == GGML_OP_ADD ||
            node.op() == GGML_OP_NORM ||
            node.op() == GGML_OP_RMS_NORM) {
            ops[0] = node.op();

            int f = i + 1;
            while (f < n && f < i + MAX_FUSE) {
                // conservatively allow fusing only these ops
                // can be expanded when needed
                if (gf->nodes[f]->op != GGML_OP_ADD &&
                    gf->nodes[f]->op != GGML_OP_MUL &&
                    gf->nodes[f]->op != GGML_OP_NORM &&
                    gf->nodes[f]->op != GGML_OP_RMS_NORM) {
                    break;
                }
                ops[f - i] = gf->nodes[f]->op;
                f++;
            }

            f -= i;
            for (; f > 1; f--) {
                if (ggml_can_fuse(gf, i, ops, f)) {
                    break;
                }
            }

            n_extra = f > 1 ? f - 1 : 0;
        } else {
            n_extra = ggml_metal_graph_optimize_pack(gf, i, node.extra_dsts);
        }

        for (int k = 0; k < n_extra; k++) {
            ++i;
            node.add_fused(gf->nodes[i]);
        }

        nodes.push_back(std::move(node));
    }

#if 1
    // reorder to improve concurrency
    const auto order = ggml_metal_graph_optimize_reorder(nodes);
#else
    std::vector<int> order(nodes.size());
    for (size_t i = 0; i < nodes.size(); i++) {
        order[i] = i;
    }
#endif

    // unfuse
    {
        int j = 0;
        for (const auto i : order) {
            const auto & node = nodes[i];

            gf->nodes[j++] = node.node;

            for (auto * fused : node.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }
}
