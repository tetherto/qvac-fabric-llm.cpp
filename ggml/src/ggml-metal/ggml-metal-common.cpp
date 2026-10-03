#include "ggml-metal-common.h"
#include "ggml-metal-fusion.h"
#include "ggml-metal-impl.h"

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <algorithm>
#include <vector>

// must stay in sync with the kernel_fwht_<type>_<N> templates in kernels/misc.metal
static bool ggml_metal_fwht_supported_size(int64_t n) {
    return n == 64 || n == 128 || n == 256 || n == 512 || n == 1024 ||
           n == 2048 || n == 4096 || n == 8192;
}

// supports_op and dispatch must use the same FWHT conditions.
bool ggml_metal_op_mul_mat_use_fwht(const struct ggml_tensor * op) {
    return ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
           op->type == GGML_TYPE_F32 &&
           (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
           ggml_is_contiguous(op->src[1]) &&
           ggml_is_contiguous(op) &&
           ggml_are_same_shape(op->src[1], op) &&
           ggml_metal_fwht_supported_size(op->src[1]->ne[0]);
}

bool ggml_metal_op_mul_mat_use_mm(const struct ggml_tensor * op, bool has_simdgroup_mm) {
    const int64_t ne00 = op->src[0]->ne[0];
    const int64_t ne11 = op->src[1]->ne[1];

    return !ggml_is_transposed(op->src[0]) &&
           !ggml_is_transposed(op->src[1]) &&
           has_simdgroup_mm && ne00 >= 64 && ne11 > 8;
}

bool ggml_metal_op_mul_mat_id_use_mm(const struct ggml_tensor * op, bool has_simdgroup_mm) {
    const int64_t ne00 = op->src[0]->ne[0];
    const int64_t ne21 = op->src[2]->ne[1];

    return has_simdgroup_mm && ne00 >= 64 && ne21 >= 32;
}

// the most src1 rows of the few-row MMA kernels
static constexpr int64_t GGML_METAL_MMA_ROWS_MAX = 16;

// src1 rows per 8x8 simdgroup matrix tile of the few-row MMA kernels
static constexpr int64_t GGML_METAL_MMA_TILE_ROWS = 8;

// weights per K step of the q5_K and generic few-row MMA kernels
static constexpr int64_t GGML_METAL_MMA_K_CHUNK = 64;

enum ggml_metal_mma_kind ggml_metal_mul_mv_mma_kind(enum ggml_type type, int rt) {
    if (type == GGML_TYPE_Q4_0 || (type == GGML_TYPE_Q8_0 && rt == 1)) {
        return GGML_METAL_MMA_KIND_BLK;
    }
    return type == GGML_TYPE_Q5_K ? GGML_METAL_MMA_KIND_Q5_K : GGML_METAL_MMA_KIND_GEN;
}

int ggml_metal_mul_mv_mma_rt(const struct ggml_tensor * op) {
    return op->src[1]->ne[1] > GGML_METAL_MMA_TILE_ROWS ? 2 : 1;
}

static bool ggml_metal_mul_mv_mma_type_supported(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

int64_t ggml_metal_mul_mv_mma_k_step(enum ggml_type type, int rt) {
    if (!ggml_metal_mul_mv_mma_type_supported(type)) {
        return 0;
    }
    return ggml_metal_mul_mv_mma_kind(type, rt) == GGML_METAL_MMA_KIND_BLK ? ggml_blck_size(type) : GGML_METAL_MMA_K_CHUNK;
}

static bool ggml_metal_mul_mat_mma_type_ok(const struct ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const int64_t step = ggml_metal_mul_mv_mma_k_step(src0->type, ggml_metal_mul_mv_mma_rt(op));

    return step > 0 && src0->ne[0] % step == 0 && src0->nb[0] == ggml_type_size(src0->type);
}

static bool ggml_metal_mul_mat_mma_shape_ok(const struct ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    // the batch shape goes into int16 function constants
    const bool batch_ok = src1->ne[2] <= INT16_MAX && src1->ne[2]/src0->ne[2] <= INT16_MAX && src1->ne[3]/src0->ne[3] <= INT16_MAX;

    return ggml_metal_mul_mat_mma_type_ok(op) && batch_ok &&
        src1->type == GGML_TYPE_F32 && src1->ne[1] >= 2 && src1->ne[1] <= GGML_METAL_MMA_ROWS_MAX &&
        !ggml_is_transposed(src0) && !ggml_is_transposed(src1) &&
        src1->nb[0] == sizeof(float) && src1->nb[1] % 16 == 0 && src1->nb[2] % 16 == 0 && src1->nb[3] % 16 == 0;
}

static bool ggml_metal_mma_device_ok(bool has_native_simdgroup_mm, bool has_tensor) {
    return has_native_simdgroup_mm && !has_tensor;
}

bool ggml_metal_mul_mat_use_mma(const struct ggml_tensor * op, bool has_native_simdgroup_mm, bool has_tensor) {
    // the FWHT kernel takes the hadamard mat-muls first
    return ggml_metal_mma_device_ok(has_native_simdgroup_mm, has_tensor) && !ggml_metal_op_mul_mat_use_fwht(op) &&
        ggml_metal_mul_mat_mma_shape_ok(op);
}

bool ggml_metal_mul_mat_may_use_mma(const struct ggml_tensor * op, bool has_native_simdgroup_mm, bool has_tensor) {
    return ggml_metal_mma_device_ok(has_native_simdgroup_mm, has_tensor) &&
        ggml_get_op_params_i32(op, 1) != GGML_HINT_SRC0_IS_HADAMARD &&
        ggml_metal_mul_mv_mma_type_supported(op->src[0]->type) && op->src[1]->type == GGML_TYPE_F32;
}

bool ggml_metal_mul_mat_use_nc(const struct ggml_tensor * op) {
    return op->src[0]->type == GGML_TYPE_Q4_0 && op->src[1]->ne[1] == N_NC_Q4_0;
}

// true if t is or views a tensor in a buffer marked as weights, such as a bias; the model loader marks its buffers before
// any graph is optimized, and tensors in unmarked or not yet allocated buffers count as non-weights in both phases
static bool ggml_metal_tensor_is_weight(const struct ggml_tensor * t) {
    const ggml_tensor * base = t->view_src != NULL ? t->view_src : t;

    return base->buffer != NULL && ggml_backend_buffer_get_usage(base->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

const struct ggml_tensor * ggml_metal_mul_mat_add_operand(const struct ggml_tensor * mm, const struct ggml_tensor * add) {
    if (add->op != GGML_OP_ADD || (add->src[0] == mm) == (add->src[1] == mm)) {
        return NULL;
    }

    const ggml_tensor * other = add->src[0] == mm ? add->src[1] : add->src[0];

    const bool ok = other->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 && !ggml_metal_tensor_is_weight(other);

    return ok ? other : NULL;
}

const struct ggml_tensor * ggml_metal_mul_mat_add_residual(const struct ggml_tensor * mm, const struct ggml_tensor * add) {
    const ggml_tensor * res = ggml_metal_mul_mat_add_operand(mm, add);

    const bool ok = res != NULL && ggml_are_same_shape(res, mm) &&
        ggml_is_contiguous(res) && ggml_is_contiguous(mm) && ggml_is_contiguous(add);

    return ok ? res : NULL;
}

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

    // narrowed ranges depend on view offsets and sizes, which change between graphs with the same nodes
    bool narrow_dst_views = true;
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

static ggml_mem_range ggml_mem_range_from_tensor(const ggml_tensor * tensor, ggml_mem_range_type pt, bool narrow_dst_views) {
    // use the base tensor, except that a written view is narrowed to its own extent if narrow_dst_views
    const ggml_tensor * base = tensor->view_src ? tensor->view_src : tensor;

    GGML_ASSERT(!base->view_src);

    ggml_mem_range mr;

    if (base->buffer) {
        // when the tensor is allocated, use the actual memory address range in the buffer
        //
        // take the actual allocated size with ggml_backend_buft_get_alloc_size()
        // this can be larger than the tensor size if the buffer type allocates extra memory
        // ref: https://github.com/ggml-org/llama.cpp/pull/15966
        mr = {
            /*.pb =*/ (uint64_t) base->buffer,
            /*.p0 =*/ (uint64_t) base->data,
            /*.p1 =*/ (uint64_t) base->data + ggml_backend_buft_get_alloc_size(base->buffer->buft, base),
            /*.pt =*/ pt,
        };

        // ops write only inside their destination view, so writes to disjoint views of one tensor do not conflict
        if (narrow_dst_views && pt == MEM_RANGE_TYPE_DST && tensor != base && tensor->data) {
            mr.p0 = (uint64_t) tensor->data;
            mr.p1 = mr.p0 + ggml_nbytes(tensor);
        }
    } else {
        // otherwise, the pointer address is used as an unique id of the memory ranges
        //   that the tensor will be using when it is allocated
        mr = {
            /*.pb =*/ (uint64_t) base,
            /*.p0 =*/ 0,    //
            /*.p1 =*/ 1024, // [0, 1024) is a dummy range, not used
            /*.pt =*/ pt,
        };
    };

    return mr;
}

static ggml_mem_range ggml_mem_range_from_tensor_src(const ggml_tensor * tensor) {
    return ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_SRC, false);
}

static ggml_mem_range ggml_mem_range_from_tensor_dst(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    return ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_DST, mrs->narrow_dst_views);
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

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(mrs, tensor);

    if (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add dst range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    return ggml_mem_ranges_add(mrs, mr);
}

// whether node reads its source i; the destination operand of a copy is written through the node itself, never read
static bool ggml_mem_range_reads_src(const ggml_tensor * node, int i) {
    return node->src[i] && !(node->op == GGML_OP_CPY && i == 1);
}

static bool ggml_mem_ranges_add_srcs(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (ggml_mem_range_reads_src(tensor, i) && !ggml_mem_ranges_add_src(mrs, tensor->src[i])) {
            return false;
        }
    }

    return true;
}

bool ggml_mem_ranges_add(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    return ggml_mem_ranges_add_srcs(mrs, tensor) && ggml_mem_ranges_add_dst(mrs, tensor);
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

        if (mr.p0 < cmp.p1 && mr.p1 > cmp.p0) {
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

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(mrs, tensor);

    const bool res = ggml_mem_ranges_check(mrs, mr);

    return res;
}

static bool ggml_mem_ranges_check_srcs(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (ggml_mem_range_reads_src(tensor, i) && !ggml_mem_ranges_check_src(mrs, tensor->src[i])) {
            return false;
        }
    }

    return true;
}

bool ggml_mem_ranges_check(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    return ggml_mem_ranges_check_srcs(mrs, tensor) && ggml_mem_ranges_check_dst(mrs, tensor);
}

struct node_info {
    ggml_tensor * node;

    std::vector<ggml_tensor *> fused;

    ggml_op op() const {
        return node->op;
    }

    const ggml_tensor * dst() const {
        return fused.empty() ? node : fused.back();
    }

    // true if the reorder tracks what t, a node of the group, writes: every node the encoder runs (not only the last
    // one, as readers of inner outputs follow the group), and dst(), which an unfused view keeps
    bool writes(const ggml_tensor * t) const {
        return t == dst() || !ggml_op_is_empty(t->op);
    }

    template <typename F>
    bool all_nodes(F && f) const {
        return f(node) && std::all_of(fused.begin(), fused.end(), f);
    }

    bool is_empty() const {
        return ggml_op_is_empty(node->op);
    }

    void add_fused(ggml_tensor * t) {
        fused.push_back(t);
    }
};

static std::vector<int> ggml_metal_graph_optimize_reorder(const std::vector<node_info> & nodes) {
    // helper to add the src and dst ranges of every node of a group
    const auto & h_add = [](ggml_mem_ranges_t mrs, const node_info & node) {
        return node.all_nodes([&](const ggml_tensor * t) {
            return ggml_mem_ranges_add_srcs(mrs, t) && (!node.writes(t) || ggml_mem_ranges_add_dst(mrs, t));
        });
    };

    // helper to check if a group can run concurrently with the existing set of nodes
    const auto & h_check = [](ggml_mem_ranges_t mrs, const node_info & node) {
        return node.all_nodes([&](const ggml_tensor * t) {
            return ggml_mem_ranges_check_srcs(mrs, t) && (!node.writes(t) || ggml_mem_ranges_check_dst(mrs, t));
        });
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

    // an order that depends on view extents changes the graph allocation between ubatches with the same nodes
    mrs0->narrow_dst_views = false;
    mrs1->narrow_dst_views = false;

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

// extra nodes to keep with gf->nodes[i] so later reorder cannot split a metal fusion
static int ggml_metal_graph_optimize_pack(const ggml_cgraph * gf, int i) {
    const int n = gf->n_nodes;
    ggml_tensor ** nodes = gf->nodes;

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

void ggml_graph_optimize(ggml_cgraph * gf, const ggml_metal_device_props * props) {
    const int n = gf->n_nodes;

    std::vector<node_info> nodes;
    nodes.reserve(gf->n_nodes);

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we first pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    //
    // the fusable sequences are declared in the fusion table (ggml-metal-fuse.cpp), so the
    // packing here is driven by the same patterns that the op encoders will later use
    for (int i = 0; i < n; i++) {
        node_info node = {
            /*.node =*/ gf->nodes[i],
            /*.fused =*/ {},
        };

        int n_extra = ggml_metal_fusion_max(gf, i, props) - 1;

        if (n_extra == 0) {
            // no table fusion starts here - try the downstream packing patterns
            n_extra = ggml_metal_graph_optimize_pack(gf, i);
        }

        // add the fused tensors into the node info so we can unfuse them later
        for (int k = 0; k < n_extra; k++) {
            ++i;

            // the .dst() becomes the last fused tensor
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
