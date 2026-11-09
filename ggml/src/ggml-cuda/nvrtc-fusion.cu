#include "nvrtc-fusion.cuh"

#ifdef GGML_CUDA_NVRTC_FUSION

#include "ggml-impl.h"

#include <cuda.h>
#include <nvrtc.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>


static int nf_dump() {
    static const int v = getenv("GGML_CUDA_NVRTC_DUMP") ? atoi(getenv("GGML_CUDA_NVRTC_DUMP")) : 0;
    return v;
}

// run the normal path and compare every generated kernel against it on the real data
static bool nf_verify() {
    static const bool v = getenv("GGML_CUDA_NVRTC_VERIFY") && atoi(getenv("GGML_CUDA_NVRTC_VERIFY")) != 0;
    return v;
}

// ---------------------------------------------------------------------------------------------
// node classification
// ---------------------------------------------------------------------------------------------

enum nf_kind : uint8_t {
    NF_NONE, // not fusible: computed by the normal path
    NF_VIEW, // view, reshape, permute, transpose: index arithmetic only
    NF_PW,   // pointwise in the index space of its output (incl. concat, repeat, cont, cpy)
    NF_ROW,  // reduction over dim 0 with a pointwise result (rms_norm, norm, l2_norm)
    NF_CONV, // ssm_conv: short stencil along dim 0 of its input
};

static bool nf_float(ggml_type t) {
    return t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16;
}

static bool nf_unary_ok(ggml_unary_op op) {
    switch (op) {
        case GGML_UNARY_OP_ABS:
        case GGML_UNARY_OP_SGN:
        case GGML_UNARY_OP_NEG:
        case GGML_UNARY_OP_STEP:
        case GGML_UNARY_OP_TANH:
        case GGML_UNARY_OP_ELU:
        case GGML_UNARY_OP_RELU:
        case GGML_UNARY_OP_SIGMOID:
        case GGML_UNARY_OP_GELU:
        case GGML_UNARY_OP_GELU_QUICK:
        case GGML_UNARY_OP_SILU:
        case GGML_UNARY_OP_HARDSWISH:
        case GGML_UNARY_OP_HARDSIGMOID:
        case GGML_UNARY_OP_EXP:
        case GGML_UNARY_OP_EXPM1:
        case GGML_UNARY_OP_SOFTPLUS:
        case GGML_UNARY_OP_GELU_ERF:
            return true;
        default:
            return false;
    }
}

static bool nf_is_view_op(ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

static nf_kind nf_classify(const ggml_tensor * t) {
    if (nf_is_view_op(t->op)) {
        return NF_VIEW;
    }
    if ((t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || ggml_is_empty(t) || !nf_float(t->type) ||
        ggml_nelements(t) >= INT32_MAX) {
        return NF_NONE;
    }
    // in-place ops write into their source; only cpy is handled
    if (t->view_src != nullptr && t->op != GGML_OP_CPY) {
        return NF_NONE;
    }
    const ggml_tensor * a = t->src[0];
    const ggml_tensor * b = t->src[1];
    if (a == nullptr || !nf_float(a->type) || ggml_nelements(a) >= INT32_MAX) {
        return NF_NONE;
    }
    switch (t->op) {
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return b && nf_float(b->type) && t->src[2] == nullptr && ggml_are_same_shape(t, a) && ggml_can_repeat(b, a)
                ? NF_PW : NF_NONE;
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_LOG:
        case GGML_OP_CLAMP:
            return ggml_are_same_shape(t, a) ? NF_PW : NF_NONE;
        case GGML_OP_UNARY:
            return nf_unary_ok(ggml_get_unary_op(t)) && ggml_are_same_shape(t, a) ? NF_PW : NF_NONE;
        case GGML_OP_CONT:
        case GGML_OP_DUP:
            return ggml_nelements(t) == ggml_nelements(a) ? NF_PW : NF_NONE;
        case GGML_OP_CPY:
            return b && nf_float(b->type) && ggml_nelements(a) == ggml_nelements(t) ? NF_PW : NF_NONE;
        case GGML_OP_CONCAT:
            return b && a->type == t->type && b->type == t->type ? NF_PW : NF_NONE;
        case GGML_OP_REPEAT:
            return ggml_can_repeat(a, t) ? NF_PW : NF_NONE;
        case GGML_OP_RMS_NORM:
        case GGML_OP_NORM:
        case GGML_OP_L2_NORM:
            return a->type == GGML_TYPE_F32 && t->type == GGML_TYPE_F32 && ggml_are_same_shape(t, a) && t->ne[0] <= 32768
                ? NF_ROW : NF_NONE;
        case GGML_OP_SSM_CONV:
            return b && a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && t->type == GGML_TYPE_F32 &&
                b->ne[0] <= 8 && b->ne[2] == 1 && b->ne[3] == 1 && a->ne[3] == 1 ? NF_CONV : NF_NONE;
        default:
            return NF_NONE;
    }
}

// ops that only move data: recomputing them for every consumer costs no extra arithmetic
static bool nf_cheap(const ggml_tensor * t) {
    return t->op == GGML_OP_CONCAT || t->op == GGML_OP_CONT || t->op == GGML_OP_DUP || t->op == GGML_OP_REPEAT;
}

// ops that read their sources at an index other than their own
static bool nf_index_changing(const ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_CONCAT:
        case GGML_OP_REPEAT:
        case GGML_OP_SSM_CONV:
            return true;
        case GGML_OP_CONT:
        case GGML_OP_DUP:
        case GGML_OP_CPY:
            return !ggml_are_same_shape(t, t->src[0]);
        default:
            return false;
    }
}

// ---------------------------------------------------------------------------------------------
// planning: which nodes are kept in registers, which are stored, and how stores group into kernels
// ---------------------------------------------------------------------------------------------

struct nf_kernel {
    std::vector<int>                 stores; // nodes written to memory by the kernel
    std::vector<int>                 cone;   // nodes computed in registers only
    std::vector<const ggml_tensor *> ext;    // tensors read from memory
    int     first  = INT_MAX;
    int     anchor = -1; // the kernel runs in place of this node
    bool    row    = false;
    bool    alive  = true;
    int64_t dom[4] = { 1, 1, 1, 1 };

    // filled at compute time
    std::vector<uint64_t>            params;
    std::vector<const ggml_tensor *> ins;
    std::vector<uint8_t>             ins_identity;
    std::vector<int>                 store_slots;
    std::string                      name;
    std::vector<int>                 sinks;    // FP8 sinks written by the kernel
    std::vector<int>                 no_f32;   // stores whose F32 copy nobody reads
    unsigned   grid[3]  = { 1, 1, 1 };
    unsigned   block[3] = { 1, 1, 1 };
    CUfunction fn       = nullptr;
};

struct nf_graph {
    const ggml_cgraph *                                g = nullptr;
    int                                                n = 0;
    std::unordered_map<const ggml_tensor *, int>       idx;
    std::vector<nf_kind>                               kind;
    std::vector<std::vector<std::pair<int, int>>>      cons; // (consumer, src slot)
    std::vector<uint8_t>                               mat;
    std::vector<uint8_t>                               taint; // depends on a row reduction in registers
    std::vector<int>                                   owner; // kernel storing the node
    std::vector<nf_kernel>                             ks;

    // per-node execution after planning
    std::vector<uint8_t> member;
    std::vector<uint8_t> is_store;
    std::vector<uint8_t> exec;

    ggml_tensor * node(int i) const { return g->nodes[i]; }

    int find(const ggml_tensor * t) const {
        if (t == nullptr) {
            return -1;
        }
        auto it = idx.find(t);
        return it == idx.end() ? -1 : it->second;
    }

    bool fusible(int i) const { return i >= 0 && (kind[i] == NF_PW || kind[i] == NF_ROW || kind[i] == NF_CONV); }
    bool virt(int i) const { return fusible(i) && !mat[i]; }

    // node that owns the memory a tensor reads from (the view source for views)
    int base_of(const ggml_tensor * t) const {
        const int i = find(t);
        if (i >= 0 && kind[i] == NF_VIEW) {
            return find(t->view_src);
        }
        return i;
    }

    int n_src(const ggml_tensor * t) const {
        return t->op == GGML_OP_CPY ? 1 : GGML_MAX_SRC; // src[1] of a cpy is its destination
    }

    // consumers of node i, looking through views; -1 means a use the plan cannot see or inline
    void eff_cons(int i, std::vector<std::pair<int, bool>> & out, bool via_view) const {
        for (const auto & [c, slot] : cons[i]) {
            const ggml_tensor * ct = node(c);
            if (kind[c] == NF_VIEW) {
                if ((ct->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(g, c) > (int) cons[c].size()) {
                    out.push_back({ -1, true });
                }
                eff_cons(c, out, true);
            } else if (ct->op == GGML_OP_CPY && slot == 1) {
                out.push_back({ -1, via_view });
            } else {
                out.push_back({ c, via_view });
            }
        }
    }

    void build_consumers() {
        n = g->n_nodes;
        idx.clear();
        idx.reserve(2 * n);
        kind.assign(n, NF_NONE);
        cons.assign(n, {});
        for (int i = 0; i < n; ++i) {
            idx[node(i)] = i;
            kind[i] = nf_classify(node(i));
        }
        for (int i = 0; i < n; ++i) {
            const ggml_tensor * t = node(i);
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                const int j = find(t->src[s]);
                if (j >= 0) {
                    cons[j].push_back({ i, s });
                }
            }
        }
    }

    void compute_taint() {
        taint.assign(n, 0);
        for (int i = 0; i < n; ++i) {
            if (!fusible(i)) {
                continue;
            }
            if (kind[i] == NF_ROW) {
                taint[i] = 1;
                continue;
            }
            const ggml_tensor * t = node(i);
            for (int s = 0; s < n_src(t); ++s) {
                const int b = base_of(t->src[s]);
                if (b >= 0 && virt(b) && taint[b]) {
                    taint[i] = 1;
                }
            }
        }
    }

    bool must_store(int i) const {
        const ggml_tensor * t = node(i);
        if ((t->flags & GGML_TENSOR_FLAG_OUTPUT) || t->op == GGML_OP_CPY) {
            return true;
        }
        if (ggml_node_get_use_count(g, i) > (int) cons[i].size()) {
            return true; // used by another split
        }
        std::vector<std::pair<int, bool>> ec;
        eff_cons(i, ec, false);
        if (ec.empty()) {
            return true;
        }
        for (const auto & [c, via] : ec) {
            if (c < 0 || !fusible(c)) {
                return true;
            }
            const ggml_tensor * ct = node(c);
            if (via && (!ggml_is_contiguous(t) || taint[i] || kind[i] == NF_CONV)) {
                return true;
            }
            if (taint[i] && (kind[c] != NF_PW || nf_index_changing(ct) || !ggml_are_same_shape(ct, t))) {
                return true; // a row result is only available at its own index
            }
            if (kind[c] == NF_CONV && kind[i] == NF_CONV) {
                return true;
            }
        }
        return ec.size() > 1 && !nf_cheap(t);
    }

    void plan_materialize() {
        mat.assign(n, 0);
        for (int iter = 0; iter < 16; ++iter) {
            compute_taint();
            bool changed = false;
            for (int i = 0; i < n; ++i) {
                if (virt(i) && must_store(i)) {
                    mat[i] = 1;
                    changed = true;
                }
            }
            if (!changed) {
                break;
            }
        }
        compute_taint();
    }

    void add_cone(nf_kernel & k, std::unordered_set<int> & seen, std::unordered_set<const ggml_tensor *> & ext_seen, int i) {
        const ggml_tensor * t = node(i);
        for (int s = 0; s < n_src(t); ++s) {
            const ggml_tensor * src = t->src[s];
            if (src == nullptr) {
                continue;
            }
            const int j = find(src);
            int b = -1;
            if (j >= 0 && kind[j] == NF_VIEW) {
                const int vb = find(src->view_src);
                if (vb >= 0 && virt(vb)) {
                    b = vb;
                }
            } else if (j >= 0 && virt(j)) {
                b = j;
            }
            if (b >= 0) {
                if (seen.insert(b).second) {
                    k.cone.push_back(b);
                    add_cone(k, seen, ext_seen, b);
                }
            } else if (ext_seen.insert(src).second) {
                k.ext.push_back(src);
            }
        }
    }

    void kernel_dom(int s, int64_t * dom) const {
        const ggml_tensor * t = node(s);
        const ggml_tensor * d = t->op == GGML_OP_CPY ? t->src[0] : t;
        for (int k = 0; k < 4; ++k) {
            dom[k] = d->ne[k];
        }
    }

    void rebuild_kernel(nf_kernel & k) {
        std::sort(k.stores.begin(), k.stores.end());
        std::unordered_set<int> seen;
        std::unordered_set<const ggml_tensor *> ext_seen;
        std::unordered_set<int> stores(k.stores.begin(), k.stores.end());
        k.cone.clear();
        k.ext.clear();
        for (int s : k.stores) {
            add_cone(k, seen, ext_seen, s);
        }
        // stores of the same kernel are computed in registers, not read back
        std::vector<const ggml_tensor *> ext;
        for (const ggml_tensor * x : k.ext) {
            const int j = find(x);
            if (j >= 0 && stores.count(j)) {
                continue;
            }
            ext.push_back(x);
        }
        k.ext = ext;
        std::sort(k.cone.begin(), k.cone.end());
        k.first  = INT_MAX;
        k.anchor = -1;
        k.row    = false;
        for (int s : k.stores) {
            k.first  = std::min(k.first, s);
            k.anchor = std::max(k.anchor, s);
            k.row    = k.row || taint[s];
        }
        for (int c : k.cone) {
            k.first  = std::min(k.first, c);
            k.anchor = std::max(k.anchor, c);
        }
    }

    bool is_member(const nf_kernel & k, int i) const {
        return std::binary_search(k.stores.begin(), k.stores.end(), i) || std::binary_search(k.cone.begin(), k.cone.end(), i);
    }

    // merge the kernel storing p into kc when both cover the same index space
    bool try_merge(int ic, int ip) {
        nf_kernel & kc = ks[ic];
        nf_kernel & kp = ks[ip];
        if (memcmp(kc.dom, kp.dom, sizeof(kc.dom)) != 0) {
            return false;
        }
        const int anchor = std::max(kc.anchor, kp.anchor);
        for (int s : kp.stores) {
            std::vector<std::pair<int, bool>> ec;
            eff_cons(s, ec, false);
            for (const auto & [c, via] : ec) {
                if (c >= 0 && is_member(kp, c)) {
                    continue;
                }
                if (c >= 0 && is_member(kc, c) && !via && ggml_are_same_shape(node(c), node(s))) {
                    continue;
                }
                if (c < 0 || c <= anchor) {
                    return false;
                }
            }
        }
        for (const ggml_tensor * x : kp.ext) {
            const int b = base_of(x);
            if (b >= 0 && b < (int) owner.size() && owner[b] == ic) {
                return false;
            }
        }
        for (int s : kp.stores) {
            kc.stores.push_back(s);
            owner[s] = ic;
        }
        kp.alive = false;
        kp.stores.clear();
        rebuild_kernel(kc);
        return true;
    }

    void plan_kernels() {
        ks.clear();
        owner.assign(n, -1);
        for (int i = 0; i < n; ++i) {
            if (!fusible(i) || !mat[i]) {
                continue;
            }
            nf_kernel k;
            k.stores.push_back(i);
            kernel_dom(i, k.dom);
            owner[i] = (int) ks.size();
            ks.push_back(std::move(k));
            rebuild_kernel(ks.back());
        }
        for (int ic = 0; ic < (int) ks.size(); ++ic) {
            bool merged = true;
            while (merged && ks[ic].alive) {
                merged = false;
                for (const ggml_tensor * x : ks[ic].ext) {
                    const int j = find(x);
                    if (j < 0 || !fusible(j) || !mat[j] || owner[j] < 0 || owner[j] == ic || !ks[owner[j]].alive) {
                        continue;
                    }
                    if (try_merge(ic, owner[j])) {
                        merged = true;
                        break;
                    }
                }
            }
        }
        // a kernel with one op does not save anything
        for (nf_kernel & k : ks) {
            if (k.alive && k.stores.size() + k.cone.size() < 2) {
                k.alive = false;
            }
        }
    }

    // nodes that still run on the normal path, and the kernels whose ordering would break
    void compute_exec() {
        member.assign(n, 0);
        is_store.assign(n, 0);
        exec.assign(n, 0);
        for (const nf_kernel & k : ks) {
            if (!k.alive) {
                continue;
            }
            for (int s : k.stores) {
                member[s]   = 1;
                is_store[s] = 1;
            }
            for (int c : k.cone) {
                member[c] = 1;
            }
        }
        for (int i = 0; i < n; ++i) {
            const ggml_tensor * t = node(i);
            exec[i] = !member[i] && kind[i] != NF_VIEW && t->op != GGML_OP_NONE && !ggml_is_empty(t) &&
                (t->flags & GGML_TENSOR_FLAG_COMPUTE);
        }
        for (int j = n - 1; j >= 0; --j) {
            if (!exec[j]) {
                continue;
            }
            const ggml_tensor * t = node(j);
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                const int b = base_of(t->src[s]);
                if (b >= 0 && member[b] && !is_store[b]) {
                    exec[b] = 1;
                }
            }
        }
    }

    bool check_order() {
        bool killed = false;
        for (int j = 0; j < n; ++j) {
            if (!exec[j]) {
                continue;
            }
            const ggml_tensor * t = node(j);
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                const int b = base_of(t->src[s]);
                if (b >= 0 && is_store[b] && ks[owner[b]].alive && ks[owner[b]].anchor > j) {
                    ks[owner[b]].alive = false;
                    killed = true;
                }
            }
        }
        for (int ik = 0; ik < (int) ks.size(); ++ik) {
            nf_kernel & k = ks[ik];
            if (!k.alive) {
                continue;
            }
            for (const ggml_tensor * x : k.ext) {
                const int b = base_of(x);
                if (b >= 0 && is_store[b] && owner[b] != ik && ks[owner[b]].alive && ks[owner[b]].anchor >= k.anchor) {
                    k.alive = false;
                    killed = true;
                    break;
                }
            }
        }
        return killed;
    }

    void settle() {
        for (int iter = 0; iter < 64; ++iter) {
            compute_exec();
            if (!check_order()) {
                break;
            }
        }
        compute_exec();
    }

    int n_alive() const {
        int c = 0;
        for (const nf_kernel & k : ks) {
            c += k.alive;
        }
        return c;
    }

    void build(const ggml_cgraph * cgraph) {
        g = cgraph;
        build_consumers();
        plan_materialize();
        plan_kernels();
        settle();
    }
};

// ---------------------------------------------------------------------------------------------
// code generation
// ---------------------------------------------------------------------------------------------

static const char * nf_preamble = R"(typedef unsigned int u32;
typedef unsigned long long u64;
static __device__ __forceinline__ float ld_f32(const char * p) { return *(const float *) p; }
static __device__ __forceinline__ float ld_f16(const char * p) { const unsigned short h = *(const unsigned short *) p; float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }
static __device__ __forceinline__ float ld_bf16(const char * p) { return __uint_as_float(((u32) *(const unsigned short *) p) << 16); }
static __device__ __forceinline__ unsigned short f2h(float v) { unsigned short h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v)); return h; }
static __device__ __forceinline__ unsigned short f2bf(float v) { const u32 u = __float_as_uint(v); if ((u & 0x7fffffffu) > 0x7f800000u) { return (unsigned short) ((u >> 16) | 64u); } return (unsigned short) ((u + (0x7fffu + ((u >> 16) & 1u))) >> 16); }
static __device__ __forceinline__ void st_f32(char * p, float v) { *(float *) p = v; }
static __device__ __forceinline__ void st_f16(char * p, float v) { *(unsigned short *) p = f2h(v); }
static __device__ __forceinline__ void st_bf16(char * p, float v) { *(unsigned short *) p = f2bf(v); }
static __device__ __forceinline__ float rnd_f16(float v) { const unsigned short h = f2h(v); float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }
static __device__ __forceinline__ float rnd_bf16(float v) { return __uint_as_float(((u32) f2bf(v)) << 16); }
static __device__ __forceinline__ u32 fdv(u32 n, u32 mp, u32 L) { return (__umulhi(n, mp) + n) >> L; }
static __device__ __forceinline__ void st_e4m3(unsigned char * p, float v) { unsigned short r; asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(0.0f), "f"(v)); *p = (unsigned char) r; }
static __device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, o, 32); }
    return v;
}
static __device__ __forceinline__ float bsum(float v, float * sm) {
    v = wsum(v);
    const u32 w = threadIdx.x >> 5, l = threadIdx.x & 31u;
    __syncthreads();
    if (l == 0) { sm[w] = v; }
    __syncthreads();
    v = l < (blockDim.x >> 5) ? sm[l] : 0.0f;
    return wsum(v);
}
)";

static const char * nf_type_sfx(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F16:  return "f16";
        case GGML_TYPE_BF16: return "bf16";
        default:             return "f32";
    }
}

struct nf_gen {
    using idx4 = std::array<std::string, 4>;

    const nf_graph & P;
    nf_kernel &      K;

    std::unordered_set<const ggml_tensor *>                   inkern;
    std::unordered_map<std::string, int>                      slots;
    std::vector<std::unordered_map<std::string, std::string>> memo;
    std::unordered_map<const ggml_tensor *, int>              in_index;

    std::string body;
    std::vector<std::pair<std::string, std::string>> fp8_scales;
    const std::vector<ggml_cuda_nvrtc_fp8_sink> * sinks = nullptr;
    int         nv     = 0;
    int         indent = 1;
    bool        ok     = true;
    std::string err;
    idx4        I;

    nf_gen(const nf_graph & P, nf_kernel & K) : P(P), K(K) {
        for (int s : K.stores) {
            inkern.insert(P.node(s));
        }
        for (int c : K.cone) {
            inkern.insert(P.node(c));
        }
        memo.emplace_back();
    }

    void fail(const std::string & why) {
        if (ok) {
            ok  = false;
            err = why;
        }
    }

    std::string var() { return "v" + std::to_string(nv++); }

    void line(const std::string & s) {
        body.append(4 * indent, ' ');
        body += s;
        body += '\n';
    }

    static std::string tkey(const ggml_tensor * t) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%p", (const void *) t);
        return buf;
    }

    int slot_index(const std::string & key, uint64_t v) {
        auto it = slots.find(key);
        if (it != slots.end()) {
            return it->second;
        }
        const int s = (int) K.params.size();
        K.params.push_back(v);
        slots[key] = s;
        return s;
    }

    std::string slot(const std::string & key, uint64_t v) {
        return "p.v[" + std::to_string(slot_index(key, v)) + "]";
    }

    // slots that hold data pointers or per-call values, the rest only depend on shapes and op params
    std::vector<std::pair<int, const ggml_tensor *>> ptr_slots;
    std::vector<std::array<int, 3>>                  sink_slots; // slot, sink, 0 = values / 1 = scale ptr / 2 = input scale

    std::string sink_slot(const std::string & key, int si, int field, uint64_t v) {
        const size_t n = K.params.size();
        const int    s = slot_index(key, v);
        if (K.params.size() != n) {
            sink_slots.push_back({ s, si, field });
        }
        return "p.v[" + std::to_string(s) + "]";
    }

    std::string su32(const std::string & key, uint64_t v) { return "((u32) " + slot(key, v) + ")"; }

    std::string sf32(const std::string & key, float f) {
        uint32_t u;
        memcpy(&u, &f, sizeof(u));
        return "__uint_as_float((u32) " + slot(key, u) + ")";
    }

    // runtime division by d: returns q = n / d, r = n % d
    void divmod(const std::string & n, uint64_t d, const std::string & key, std::string & q, std::string & r) {
        if (d == 1) {
            q = n;
            r = "0";
            return;
        }
        const uint3 fd = init_fastdiv_values(d);
        q = var();
        r = var();
        line("const u32 " + q + " = fdv(" + n + ", " + su32("fdm" + key, fd.x) + ", " + su32("fdl" + key, fd.y) + ");");
        line("const u32 " + r + " = " + n + " - " + q + " * " + su32("fdd" + key, fd.z) + ";");
    }

    std::string mod(const std::string & n, uint64_t d, const std::string & key) {
        std::string q, r;
        divmod(n, d, key, q, r);
        return r;
    }

    // linear element index of ix in a tensor of shape ne, in row-major order
    std::string linear(const idx4 & ix, const int64_t * ne, const std::string & key) {
        std::string L = ix[3];
        for (int k = 2; k >= 0; --k) {
            if (L == "0") {
                L = ix[k];
            } else if (ne[k] == 1) {
                // ix[k] is 0
            } else {
                L = "(" + ix[k] + " + " + su32("lin" + std::to_string(k) + key, ne[k]) + " * " + L + ")";
            }
        }
        const std::string v = var();
        line("const u32 " + v + " = " + L + ";");
        return v;
    }

    idx4 unravel(const std::string & L, const int64_t * ne, const std::string & key) {
        idx4 r = { "0", "0", "0", "0" };
        std::string q = L;
        for (int k = 0; k < 4; ++k) {
            if (ne[k] == 1) {
                continue;
            }
            bool last = true;
            for (int m = k + 1; m < 4; ++m) {
                last = last && ne[m] == 1;
            }
            if (last) {
                r[k] = q;
                break;
            }
            std::string qq, rr;
            divmod(q, ne[k], "unr" + std::to_string(k) + key, qq, rr);
            r[k] = rr;
            q    = qq;
        }
        return r;
    }

    std::string offset(const ggml_tensor * x, const idx4 & ix) {
        std::string o;
        for (int k = 0; k < 4; ++k) {
            if (ix[k] == "0") {
                continue;
            }
            std::string term;
            if (k == 0 && x->nb[0] == ggml_type_size(x->type)) {
                term = "(long long) " + ix[0] + " * " + std::to_string(x->nb[0]);
            } else {
                term = "(long long) " + ix[k] + " * (long long) " + slot("nb" + std::to_string(k) + tkey(x), x->nb[k]);
            }
            o += " + " + term;
        }
        return o;
    }

    std::string ptr(const ggml_tensor * x) {
        const size_t n = K.params.size();
        const int    s = slot_index("ptr" + tkey(x), (uint64_t) (uintptr_t) x->data);
        if (K.params.size() != n) {
            ptr_slots.push_back({ s, x });
        }
        return "((char *) p.v[" + std::to_string(s) + "])";
    }

    std::string load(const ggml_tensor * x, const idx4 & ix) {
        auto it = in_index.find(x);
        int  ii;
        if (it == in_index.end()) {
            ii = (int) K.ins.size();
            in_index[x] = ii;
            K.ins.push_back(x);
            K.ins_identity.push_back(1);
        } else {
            ii = it->second;
        }
        bool identity = ix == I;
        for (int k = 0; k < 4; ++k) {
            identity = identity && x->ne[k] == K.dom[k];
        }
        if (!identity) {
            K.ins_identity[ii] = 0;
        }
        const std::string v = var();
        line("const float " + v + " = ld_" + nf_type_sfx(x->type) + "(" + ptr(x) + offset(x, ix) + ");");
        return v;
    }

    idx4 bcast(const ggml_tensor * b, const ggml_tensor * t, const idx4 & ix) {
        idx4 r;
        for (int k = 0; k < 4; ++k) {
            if (b->ne[k] == t->ne[k]) {
                r[k] = ix[k];
            } else if (b->ne[k] == 1) {
                r[k] = "0";
            } else {
                r[k] = mod(ix[k], b->ne[k], "bc" + std::to_string(k) + tkey(b));
            }
        }
        return r;
    }

    std::string emit(const ggml_tensor * t, const idx4 & ix) {
        if (!ok) {
            return "0.0f";
        }
        const std::string key = tkey(t) + "|" + ix[0] + "|" + ix[1] + "|" + ix[2] + "|" + ix[3];
        for (auto it = memo.rbegin(); it != memo.rend(); ++it) {
            auto f = it->find(key);
            if (f != it->end()) {
                return f->second;
            }
        }
        const std::string v = emit_new(t, ix);
        memo.back()[key] = v;
        return v;
    }

    std::string emit_view(const ggml_tensor * t, const ggml_tensor * b, const idx4 & ix) {
        if (ggml_are_same_shape(t, b) && memcmp(t->nb, b->nb, sizeof(t->nb)) == 0 && t->view_offs == 0) {
            return emit(b, ix);
        }
        // a permutation of the source dims (transpose, permute, reshape that only adds or drops 1s)
        if (t->view_offs == 0 && ggml_nelements(t) == ggml_nelements(b)) {
            idx4 jb   = { "0", "0", "0", "0" };
            bool used[4] = { false, false, false, false };
            bool perm = true;
            for (int k = 0; perm && k < 4; ++k) {
                if (t->ne[k] == 1) {
                    continue;
                }
                int m = 0;
                while (m < 4 && (used[m] || b->ne[m] != t->ne[k] || b->nb[m] != t->nb[k])) {
                    ++m;
                }
                if (m == 4) {
                    perm = false;
                    break;
                }
                used[m] = true;
                jb[m]   = ix[k];
            }
            for (int m = 0; perm && m < 4; ++m) {
                perm = used[m] || b->ne[m] == 1;
            }
            if (perm) {
                return emit(b, jb);
            }
        }
        if (t->view_offs == 0 && ggml_is_contiguous(t) && ggml_is_contiguous(b) && ggml_nelements(t) == ggml_nelements(b)) {
            int dt[4], db[4], nt = 0, nb = 0;
            for (int k = 0; k < 4; ++k) {
                if (t->ne[k] != 1) {
                    dt[nt++] = k;
                }
                if (b->ne[k] != 1) {
                    db[nb++] = k;
                }
            }
            bool same = nt == nb;
            for (int m = 0; same && m < nt; ++m) {
                same = t->ne[dt[m]] == b->ne[db[m]];
            }
            if (same) {
                idx4 jb = { "0", "0", "0", "0" };
                for (int m = 0; m < nt; ++m) {
                    jb[db[m]] = ix[dt[m]];
                }
                return emit(b, jb);
            }
        }
        if (!ggml_is_contiguous(b)) {
            fail("view of a non-contiguous register tensor");
            return "0.0f";
        }
        const size_t es = ggml_type_size(b->type);
        std::string L = su32("vo" + tkey(t), t->view_offs / es);
        for (int k = 0; k < 4; ++k) {
            if (ix[k] != "0") {
                L += " + " + ix[k] + " * " + su32("vn" + std::to_string(k) + tkey(t), t->nb[k] / es);
            }
        }
        const std::string lv = var();
        line("const u32 " + lv + " = " + L + ";");
        return emit(b, unravel(lv, b->ne, tkey(b)));
    }

    std::string unary(ggml_unary_op op, const std::string & x) {
        switch (op) {
            case GGML_UNARY_OP_ABS:         return "fabsf(" + x + ")";
            case GGML_UNARY_OP_SGN:         return "(" + x + " > 0.0f ? 1.0f : (" + x + " < 0.0f ? -1.0f : 0.0f))";
            case GGML_UNARY_OP_NEG:         return "(-" + x + ")";
            case GGML_UNARY_OP_STEP:        return "(" + x + " > 0.0f ? 1.0f : 0.0f)";
            case GGML_UNARY_OP_TANH:        return "tanhf(" + x + ")";
            case GGML_UNARY_OP_ELU:         return "(" + x + " > 0.0f ? " + x + " : expm1f(" + x + "))";
            case GGML_UNARY_OP_RELU:        return "fmaxf(" + x + ", 0.0f)";
            case GGML_UNARY_OP_SIGMOID:     return "(1.0f / (1.0f + expf(-" + x + ")))";
            case GGML_UNARY_OP_GELU:        return "(0.5f * " + x + " * (1.0f + tanhf(0.79788456080286535587989211986876f * " + x + " * (1.0f + 0.044715f * " + x + " * " + x + "))))";
            case GGML_UNARY_OP_GELU_QUICK:  return "(" + x + " * (1.0f / (1.0f + expf(-1.702f * " + x + "))))";
            case GGML_UNARY_OP_SILU:        return "(" + x + " / (1.0f + expf(-" + x + ")))";
            case GGML_UNARY_OP_HARDSWISH:   return "(" + x + " * fminf(1.0f, fmaxf(0.0f, (" + x + " + 3.0f) / 6.0f)))";
            case GGML_UNARY_OP_HARDSIGMOID: return "fminf(1.0f, fmaxf(0.0f, (" + x + " + 3.0f) / 6.0f))";
            case GGML_UNARY_OP_EXP:         return "expf(" + x + ")";
            case GGML_UNARY_OP_EXPM1:       return "expm1f(" + x + ")";
            case GGML_UNARY_OP_SOFTPLUS:    return "(" + x + " > 20.0f ? " + x + " : logf(1.0f + expf(" + x + ")))";
            case GGML_UNARY_OP_GELU_ERF:    return "(0.5f * " + x + " * (1.0f + erff(" + x + " * 0.70710678118654752440084436210484f)))";
            default:
                fail("unary op");
                return x;
        }
    }

    std::string emit_new(const ggml_tensor * t, const idx4 & ix) {
        if (nf_is_view_op(t->op)) {
            const ggml_tensor * b = t->view_src;
            if (b == nullptr || !inkern.count(b)) {
                return load(t, ix);
            }
            return emit_view(t, b, ix);
        }
        if (!inkern.count(t)) {
            return load(t, ix);
        }
        const int ti = P.find(t);
        if (P.kind[ti] == NF_ROW) {
            fail("row reduction read outside its row");
            return "0.0f";
        }
        if (P.taint[ti] && ix != I) {
            fail("row-dependent value read at another index");
            return "0.0f";
        }
        const ggml_tensor * a = t->src[0];
        const ggml_tensor * b = t->src[1];
        std::string r;
        switch (t->op) {
            case GGML_OP_ADD:
            case GGML_OP_SUB:
            case GGML_OP_MUL:
            case GGML_OP_DIV: {
                const std::string va = emit(a, ix);
                const std::string vb = emit(b, bcast(b, t, ix));
                const char * op = t->op == GGML_OP_ADD ? " + " : t->op == GGML_OP_SUB ? " - " : t->op == GGML_OP_MUL ? " * " : " / ";
                r = va + op + vb;
            } break;
            case GGML_OP_SCALE: {
                const std::string va = emit(a, ix);
                r = va + " * " + sf32("s" + tkey(t), ggml_get_op_params_f32(t, 0)) + " + " + sf32("b" + tkey(t), ggml_get_op_params_f32(t, 1));
            } break;
            case GGML_OP_SQR: {
                const std::string va = emit(a, ix);
                r = va + " * " + va;
            } break;
            case GGML_OP_SQRT: r = "sqrtf(" + emit(a, ix) + ")"; break;
            case GGML_OP_SIN:  r = "sinf(" + emit(a, ix) + ")"; break;
            case GGML_OP_COS:  r = "cosf(" + emit(a, ix) + ")"; break;
            case GGML_OP_LOG:  r = "logf(" + emit(a, ix) + ")"; break;
            case GGML_OP_CLAMP:
                r = "fminf(fmaxf(" + emit(a, ix) + ", " + sf32("lo" + tkey(t), ggml_get_op_params_f32(t, 0)) + "), " +
                    sf32("hi" + tkey(t), ggml_get_op_params_f32(t, 1)) + ")";
                break;
            case GGML_OP_UNARY: r = unary(ggml_get_unary_op(t), emit(a, ix)); break;
            case GGML_OP_CONT:
            case GGML_OP_DUP:
            case GGML_OP_CPY: {
                if (ggml_are_same_shape(t, a)) {
                    r = emit(a, ix);
                } else {
                    r = emit(a, unravel(linear(ix, t->ne, tkey(t)), a->ne, "c" + tkey(a)));
                }
            } break;
            case GGML_OP_CONCAT: {
                const int dim = ggml_get_op_params_i32(t, 0);
                const std::string na = su32("cn" + tkey(t), a->ne[dim]);
                const std::string v = var();
                line("float " + v + ";");
                line("if (" + ix[dim] + " < " + na + ") {");
                indent++;
                memo.emplace_back();
                line(v + " = " + emit(a, ix) + ";");
                memo.pop_back();
                indent--;
                line("} else {");
                indent++;
                memo.emplace_back();
                idx4 ib = ix;
                ib[dim] = "(" + ix[dim] + " - " + na + ")";
                line(v + " = " + emit(b, ib) + ";");
                memo.pop_back();
                indent--;
                line("}");
                return v;
            }
            case GGML_OP_REPEAT: {
                idx4 ia;
                for (int k = 0; k < 4; ++k) {
                    ia[k] = a->ne[k] == t->ne[k] ? ix[k] : a->ne[k] == 1 ? "0" : mod(ix[k], a->ne[k], "rp" + std::to_string(k) + tkey(a));
                }
                r = emit(a, ia);
            } break;
            case GGML_OP_SSM_CONV: {
                const std::string acc = var();
                line("float " + acc + " = 0.0f;");
                for (int64_t j = 0; j < b->ne[0]; ++j) {
                    const std::string it = j == 0 ? ix[1] : "(" + ix[1] + " + " + std::to_string(j) + "u)";
                    const std::string xv = emit(a, { it, ix[0], ix[2], "0" });
                    const std::string wv = emit(b, { std::to_string(j) + "u", ix[0], "0", "0" });
                    line(acc + " += " + xv + " * " + wv + ";");
                }
                r = acc;
            } break;
            default:
                fail(std::string("op ") + ggml_op_desc(t));
                return "0.0f";
        }
        const std::string v = var();
        // F16/BF16 results, copies included, are rounded: a reader in the same kernel sees what it would load
        if (t->type == GGML_TYPE_F16) {
            line("const float " + v + " = rnd_f16(" + r + ");");
        } else if (t->type == GGML_TYPE_BF16) {
            line("const float " + v + " = rnd_bf16(" + r + ");");
        } else {
            line("const float " + v + " = " + r + ";");
        }
        return v;
    }

    void emit_stores() {
        for (int s : K.stores) {
            const ggml_tensor * st = P.node(s);
            std::string val;
            idx4 jd = I;
            if (st->op == GGML_OP_CPY) {
                val = emit(st->src[0], I);
                if (!ggml_are_same_shape(st, st->src[0])) {
                    jd = unravel(linear(I, K.dom, "dom"), st->ne, "st" + tkey(st));
                }
            } else {
                val = emit(st, I);
            }
            if (!std::binary_search(K.no_f32.begin(), K.no_f32.end(), s)) {
                line(std::string("st_") + nf_type_sfx(st->type) + "(" + ptr(st) + offset(st, jd) + ", " + val + ");");
            }
            for (int si : K.sinks) {
                const ggml_cuda_nvrtc_fp8_sink & sk = (*sinks)[si];
                if (P.base_of(sk.src1) != s) {
                    continue;
                }
                const std::string key = "fp8" + std::to_string(si);
                uint32_t is_bits;
                memcpy(&is_bits, &sk.input_scale, sizeof(is_bits));
                const std::string vp  = "((unsigned char *) " + sink_slot("v" + key, si, 0, (uint64_t) (uintptr_t) sk.values) + ")";
                const std::string is  = "__uint_as_float((u32) " + sink_slot("x" + key, si, 2, is_bits) + ")";
                line("st_e4m3(" + vp + " + ((0" + offset(st, I) + ") >> 2), " + val + " / " + is + ");");
                fp8_scales.push_back({ sink_slot("p" + key, si, 1, (uint64_t) (uintptr_t) sk.scale), is });
            }
        }
    }

    void row_decomp() {
        const int64_t * d = K.dom;
        std::string q = "row";
        const char * names[4] = { "", "r1", "r2", "r3" };
        for (int k = 1; k < 4; ++k) {
            if (d[k] == 1) {
                continue;
            }
            bool last = true;
            for (int m = k + 1; m < 4; ++m) {
                last = last && d[m] == 1;
            }
            if (last) {
                line(std::string("const u32 ") + names[k] + " = " + q + ";");
                break;
            }
            std::string qq, rr;
            divmod(q, d[k], "dom" + std::to_string(k), qq, rr);
            line(std::string("const u32 ") + names[k] + " = " + rr + ";");
            q = qq;
        }
    }

    // returns the kernel source, or "" on failure
    std::string build() {
        const int64_t ne0   = K.dom[0];
        const int64_t nrows = K.dom[1] * K.dom[2] * K.dom[3];
        if (ne0 * nrows >= INT32_MAX) {
            fail("too large");
            return "";
        }
        I = { ne0 > 1 ? "c0" : "0", K.dom[1] > 1 ? "r1" : "0", K.dom[2] > 1 ? "r2" : "0", K.dom[3] > 1 ? "r3" : "0" };

        int nthreads = 0;
        if (!K.row) {
            if (ne0 >= 128) {
                const int64_t bx = ne0 >= 256 ? 256 : 128;
                const int64_t u  = std::min<int64_t>(4, (ne0 + bx - 1) / bx);
                const int64_t ch = bx * u;
                K.grid[0]  = (unsigned) nrows;
                K.grid[1]  = (unsigned) ((ne0 + ch - 1) / ch);
                K.block[0] = (unsigned) bx;
                nthreads   = (int) bx;
                line("const u32 row = blockIdx.x;");
                row_decomp();
                line("#pragma unroll");
                line("for (u32 u = 0; u < " + std::to_string(u) + "u; ++u) {");
                indent++;
                line("const u32 c0 = blockIdx.y * " + std::to_string(ch) + "u + u * " + std::to_string(bx) + "u + threadIdx.x;");
                const bool partial = ne0 % ch != 0;
                if (partial) {
                    line("if (c0 < " + std::to_string(ne0) + "u) {");
                    indent++;
                }
                memo.emplace_back();
                emit_stores();
                memo.pop_back();
                if (partial) {
                    indent--;
                    line("}");
                }
                indent--;
                line("}");
            } else {
                const int64_t total = ne0 * nrows;
                K.grid[0]  = (unsigned) ((total + 255) / 256);
                K.block[0] = 256;
                nthreads   = 256;
                line("const u32 gid = blockIdx.x * 256u + threadIdx.x;");
                line("if (gid >= " + su32("total", total) + ") { return; }");
                if (ne0 > 1) {
                    line("const u32 c0 = gid % " + std::to_string(ne0) + "u;");
                    line("const u32 row = gid / " + std::to_string(ne0) + "u;");
                } else {
                    line("const u32 row = gid;");
                }
                row_decomp();
                emit_stores();
            }
        } else {
            int bt, wpb;
            if (ne0 <= 1024) {
                bt  = 32;
                wpb = 4;
                K.grid[0]  = (unsigned) ((nrows + wpb - 1) / wpb);
                K.block[0] = 32;
                K.block[1] = (unsigned) wpb;
                line("const u32 row = blockIdx.x * " + std::to_string(wpb) + "u + threadIdx.y;");
                line("if (row >= " + su32("nrows", nrows) + ") { return; }");
                line("const u32 tid = threadIdx.x;");
            } else {
                bt  = ne0 <= 8192 ? 256 : ne0 <= 16384 ? 512 : 1024;
                wpb = 1;
                K.grid[0]  = (unsigned) nrows;
                K.block[0] = (unsigned) bt;
                line("__shared__ float sm[32];");
                line("const u32 row = blockIdx.x;");
                line("const u32 tid = threadIdx.x;");
            }
            nthreads = bt * wpb;
            const int  per     = (int) ((ne0 + bt - 1) / bt);
            const bool partial = ne0 % bt != 0;
            const std::string red_open = bt == 32 ? "wsum(" : "bsum(";
            const std::string red_close = bt == 32 ? ")" : ", sm)";
            row_decomp();

            std::vector<int> rows;
            for (int s : K.stores) {
                if (P.kind[s] == NF_ROW) {
                    rows.push_back(s);
                }
            }
            for (int c : K.cone) {
                if (P.kind[c] == NF_ROW) {
                    rows.push_back(c);
                }
            }
            std::sort(rows.begin(), rows.end());

            auto open_loop = [&]() {
                line("#pragma unroll");
                line("for (int k = 0; k < " + std::to_string(per) + "; ++k) {");
                indent++;
                line("const u32 c0 = tid + (u32) k * " + std::to_string(bt) + "u;");
                if (partial) {
                    line("if (c0 < " + std::to_string(ne0) + "u) {");
                    indent++;
                }
                memo.emplace_back();
            };
            auto close_loop = [&]() {
                memo.pop_back();
                if (partial) {
                    indent--;
                    line("}");
                }
                indent--;
                line("}");
            };

            for (size_t rn = 0; rn < rows.size(); ++rn) {
                const ggml_tensor * R = P.node(rows[rn]);
                const std::string id = std::to_string(rn);
                const std::string arr = "a" + id, sum = "s" + id, st = "m" + id, mean = "mu" + id;
                line("float " + arr + "[" + std::to_string(per) + "];");
                line("float " + sum + " = 0.0f;");
                open_loop();
                const std::string x = emit(R->src[0], I);
                line(arr + "[k] = " + x + ";");
                line(sum + " += " + (R->op == GGML_OP_NORM ? x : x + " * " + x) + ";");
                close_loop();
                line(sum + " = " + red_open + sum + red_close + ";");
                const std::string eps = sf32("eps" + tkey(R), ggml_get_op_params_f32(R, 0));
                const std::string n0  = std::to_string(ne0) + ".0f";
                std::string val;
                if (R->op == GGML_OP_RMS_NORM) {
                    line("const float " + st + " = rsqrtf(" + sum + " / " + n0 + " + " + eps + ");");
                    val = "(" + arr + "[k] * " + st + ")";
                } else if (R->op == GGML_OP_L2_NORM) {
                    line("const float " + st + " = rsqrtf(fmaxf(" + sum + ", " + eps + " * " + eps + "));");
                    val = "(" + arr + "[k] * " + st + ")";
                } else {
                    line("const float " + mean + " = " + sum + " / " + n0 + ";");
                    line("float " + sum + "v = 0.0f;");
                    line("#pragma unroll");
                    line("for (int k = 0; k < " + std::to_string(per) + "; ++k) {");
                    line("    const float d = " + arr + "[k] - " + mean + ";");
                    line(std::string("    ") + (partial ? "if (tid + (u32) k * " + std::to_string(bt) + "u < " + std::to_string(ne0) + "u) { " : "{ ") + sum + "v += d * d; }");
                    line("}");
                    line(sum + "v = " + red_open + sum + "v" + red_close + ";");
                    line("const float " + st + " = rsqrtf(" + sum + "v / " + n0 + " + " + eps + ");");
                    val = "((" + arr + "[k] - " + mean + ") * " + st + ")";
                }
                memo[0][tkey(R) + "|" + I[0] + "|" + I[1] + "|" + I[2] + "|" + I[3]] = val;
                memo[0][tkey(R->src[0]) + "|" + I[0] + "|" + I[1] + "|" + I[2] + "|" + I[3]] = arr + "[k]";
            }
            open_loop();
            emit_stores();
            close_loop();
        }
        if (!ok) {
            return "";
        }
        if (!fp8_scales.empty()) {
            std::string first = "blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0";
            body = "    if (" + first + ") {\n" + [&] {
                std::string w;
                for (const auto & [p, v] : fp8_scales) {
                    w += "        *(float *) " + p + " = " + v + ";\n";
                }
                return w;
            }() + "    }\n" + body;
        }
        std::string src = nf_preamble;
        src += "struct P { u64 v[" + std::to_string(std::max<size_t>(1, K.params.size())) + "]; };\n";
        src += "extern \"C\" __global__ void __launch_bounds__(" + std::to_string(nthreads) + ") nf_k(const P p) {\n";
        src += body;
        src += "}\n";
        // a readable kernel name for profilers: the ops in node order plus a hash of the source
        std::vector<int> members = K.cone;
        members.insert(members.end(), K.stores.begin(), K.stores.end());
        std::sort(members.begin(), members.end());
        std::string name = "nf";
        for (int m : members) {
            name += std::string("_") + ggml_op_desc(P.node(m));
        }
        if (name.size() > 96) {
            name.resize(96);
        }
        uint64_t h = 0xcbf29ce484222325ull;
        for (unsigned char ch : src) {
            h = (h ^ ch) * 0x100000001b3ull;
        }
        char hs[16];
        snprintf(hs, sizeof(hs), "_%06llx", (unsigned long long) (h & 0xffffff));
        name += hs;
        for (char & ch : name) {
            if (!isalnum((unsigned char) ch) && ch != '_') {
                ch = '_';
            }
        }
        const size_t pos = src.find(" nf_k(");
        src.replace(pos + 1, 4, name);
        K.name = name;
        if (K.params.empty()) {
            K.params.push_back(0);
        }
        for (int s : K.stores) {
            auto it = slots.find("ptr" + tkey(P.node(s)));
            K.store_slots.push_back(it == slots.end() ? -1 : it->second);
        }
        return src;
    }
};

// ---------------------------------------------------------------------------------------------
// NVRTC compilation with an in-memory and an on-disk cache
// ---------------------------------------------------------------------------------------------

struct nf_compiler {
    std::mutex                                  mtx;
    std::unordered_map<std::string, CUfunction> fns;
    std::string                                 dir;
    bool                                        dir_ok   = false;
    bool                                        inited   = false;
    int                                         n_nvrtc  = 0;
    int                                         n_disk   = 0;
    double                                      ms_nvrtc = 0.0;
    double                                      ms_disk  = 0.0;

    ~nf_compiler() {
        if (nf_dump() > 0 && (n_nvrtc > 0 || n_disk > 0)) {
            fprintf(stderr, "nvrtc-fusion: %d kernels compiled in %.1f ms, %d loaded from disk in %.1f ms, %zu in memory\n",
                    n_nvrtc, ms_nvrtc, n_disk, ms_disk, fns.size());
        }
    }
};

static nf_compiler & nf_comp() {
    static nf_compiler c;
    return c;
}

static uint64_t nf_hash(const std::string & s, uint64_t h) {
    for (unsigned char c : s) {
        h ^= c;
        h *= 0x100000001b3ull;
    }
    return h;
}

static void nf_mkdirs(const std::string & path) {
    std::string cur;
    for (size_t i = 0; i < path.size(); ++i) {
        cur += path[i];
        if ((path[i] == '/' && i > 0) || i + 1 == path.size()) {
            mkdir(cur.c_str(), 0755);
        }
    }
}

static void nf_init_dir(nf_compiler & c) {
    if (c.inited) {
        return;
    }
    c.inited = true;
    const char * env = getenv("GGML_CUDA_NVRTC_CACHE_DIR");
    if (env != nullptr) {
        c.dir = env;
    } else if (getenv("XDG_CACHE_HOME") != nullptr) {
        c.dir = std::string(getenv("XDG_CACHE_HOME")) + "/ggml-nvrtc";
    } else if (getenv("HOME") != nullptr) {
        c.dir = std::string(getenv("HOME")) + "/.cache/ggml-nvrtc";
    }
    if (!c.dir.empty() && c.dir != "0") {
        nf_mkdirs(c.dir);
        struct stat st;
        c.dir_ok = stat(c.dir.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
    }
}

static CUfunction nf_load(const std::vector<char> & image, const std::string & name) {
    CUmodule   mod = nullptr;
    CUfunction fn  = nullptr;
    if (cuModuleLoadData(&mod, image.data()) != CUDA_SUCCESS || cuModuleGetFunction(&fn, mod, name.c_str()) != CUDA_SUCCESS) {
        return nullptr;
    }
    return fn;
}

static CUfunction nf_get_function(int device, const std::string & src, const std::string & name) {
    nf_compiler & c = nf_comp();
    std::lock_guard<std::mutex> lock(c.mtx);

    const std::string key = std::to_string(device) + "\n" + src;
    auto it = c.fns.find(key);
    if (it != c.fns.end()) {
        return it->second;
    }
    nf_init_dir(c);

    int major = 0, minor = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    int nv_major = 0, nv_minor = 0;
    nvrtcVersion(&nv_major, &nv_minor);
    const std::string arch = "--gpu-architecture=sm_" + std::to_string(major) + std::to_string(minor);
    const char * opts[] = { arch.c_str(), "--use_fast_math", "--std=c++17" };

    const std::string id = arch + " nvrtc " + std::to_string(nv_major) + "." + std::to_string(nv_minor) + "\n" + src;
    const uint64_t h1 = nf_hash(id, 0xcbf29ce484222325ull);
    const uint64_t h2 = nf_hash(id, 0x84222325cbf29ce4ull);
    char file[64];
    snprintf(file, sizeof(file), "%016llx.cubin", (unsigned long long) h1);
    const std::string path = c.dir_ok ? c.dir + "/" + file : "";

    const auto t0 = std::chrono::steady_clock::now();
    if (!path.empty()) {
        FILE * f = fopen(path.c_str(), "rb");
        if (f != nullptr) {
            uint64_t hdr[2] = { 0, 0 };
            std::vector<char> image;
            if (fread(hdr, sizeof(hdr), 1, f) == 1 && hdr[0] == 0x31434e46ull && hdr[1] == h2) {
                fseek(f, 0, SEEK_END);
                const long size = ftell(f) - (long) sizeof(hdr);
                fseek(f, sizeof(hdr), SEEK_SET);
                if (size > 0) {
                    image.resize(size);
                    if (fread(image.data(), 1, size, f) != (size_t) size) {
                        image.clear();
                    }
                }
            }
            fclose(f);
            if (!image.empty()) {
                CUfunction fn = nf_load(image, name);
                if (fn != nullptr) {
                    c.n_disk++;
                    c.ms_disk += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                    c.fns[key] = fn;
                    return fn;
                }
            }
        }
    }

    nvrtcProgram prog;
    if (nvrtcCreateProgram(&prog, src.c_str(), "nf.cu", 0, nullptr, nullptr) != NVRTC_SUCCESS) {
        c.fns[key] = nullptr;
        return nullptr;
    }
    const nvrtcResult res = nvrtcCompileProgram(prog, sizeof(opts) / sizeof(opts[0]), opts);
    if (res != NVRTC_SUCCESS) {
        size_t log_size = 0;
        nvrtcGetProgramLogSize(prog, &log_size);
        std::string log(log_size, '\0');
        nvrtcGetProgramLog(prog, log.data());
        GGML_LOG_ERROR("nvrtc-fusion: compile failed (%s):\n%s\n%s\n", nvrtcGetErrorString(res), log.c_str(), src.c_str());
        nvrtcDestroyProgram(&prog);
        c.fns[key] = nullptr;
        return nullptr;
    }
    size_t size = 0;
    nvrtcGetCUBINSize(prog, &size);
    std::vector<char> image(size);
    nvrtcGetCUBIN(prog, image.data());
    nvrtcDestroyProgram(&prog);

    CUfunction fn = nf_load(image, name);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    c.n_nvrtc++;
    c.ms_nvrtc += ms;
    if (nf_dump() > 0) {
        fprintf(stderr, "nvrtc-fusion: compiled kernel %d in %.1f ms\n", c.n_nvrtc, ms);
    }
    if (fn != nullptr && !path.empty()) {
        const std::string tmp = path + ".tmp" + std::to_string(getpid());
        FILE * f = fopen(tmp.c_str(), "wb");
        if (f != nullptr) {
            const uint64_t hdr[2] = { 0x31434e46ull, h2 };
            const bool ok = fwrite(hdr, sizeof(hdr), 1, f) == 1 && fwrite(image.data(), 1, image.size(), f) == image.size();
            fclose(f);
            if (!ok || rename(tmp.c_str(), path.c_str()) != 0) {
                unlink(tmp.c_str());
            }
        }
    }
    c.fns[key] = fn;
    return fn;
}

// ---------------------------------------------------------------------------------------------
// memory checks at compute time
// ---------------------------------------------------------------------------------------------

struct nf_range {
    uintptr_t a = 0;
    uintptr_t b = 0;
};

static nf_range nf_range_of(const ggml_tensor * t) {
    if (t == nullptr || t->data == nullptr) {
        return {};
    }
    return { (uintptr_t) t->data, (uintptr_t) t->data + ggml_nbytes(t) };
}

static bool nf_overlap(const nf_range & x, const nf_range & y) {
    return x.a < x.b && y.a < y.b && x.a < y.b && y.a < x.b;
}

static bool nf_same_layout(const ggml_tensor * x, const ggml_tensor * y) {
    return x->data == y->data && x->type == y->type && ggml_are_same_shape(x, y) && memcmp(x->nb, y->nb, sizeof(x->nb)) == 0;
}

// the kernel reads its inputs and writes its stores at the anchor instead of at their own positions.
// an input may not be written after the first member that reads it, and a store may not be read or
// written by another node between its own position and the anchor
static bool nf_check_memory(const nf_graph & P, const nf_kernel & K, const std::vector<int> & anchor_of, std::string & why) {
    std::vector<nf_range> in, st;
    std::vector<int>      in_first(K.ins.size(), INT_MAX);
    for (const ggml_tensor * x : K.ins) {
        in.push_back(nf_range_of(x));
    }
    for (int s : K.stores) {
        st.push_back(nf_range_of(P.node(s)));
    }
    auto note_reads = [&](int m) {
        const ggml_tensor * t = P.node(m);
        for (int s = 0; s < P.n_src(t); ++s) {
            for (size_t i = 0; i < K.ins.size(); ++i) {
                if (t->src[s] == K.ins[i]) {
                    in_first[i] = std::min(in_first[i], m);
                }
            }
        }
    };
    for (int m : K.stores) {
        note_reads(m);
    }
    for (int m : K.cone) {
        note_reads(m);
    }
    for (size_t i = 0; i < K.ins.size(); ++i) {
        for (size_t j = 0; j < K.stores.size(); ++j) {
            if (nf_overlap(in[i], st[j]) && !(K.ins_identity[i] && nf_same_layout(K.ins[i], P.node(K.stores[j])))) {
                why = std::string("input ") + K.ins[i]->name + " overlaps store " + P.node(K.stores[j])->name;
                return false;
            }
        }
    }
    auto bad_write = [&](int j, const nf_range & w) {
        for (size_t i = 0; i < in.size(); ++i) {
            if (j > in_first[i] && nf_overlap(w, in[i])) {
                return true;
            }
        }
        for (size_t i = 0; i < st.size(); ++i) {
            if (j > K.stores[i] && nf_overlap(w, st[i])) {
                return true;
            }
        }
        return false;
    };
    auto bad_read = [&](int j, const nf_range & r) {
        for (size_t i = 0; i < st.size(); ++i) {
            if (j > K.stores[i] && nf_overlap(r, st[i])) {
                return true;
            }
        }
        return false;
    };
    for (int j = K.first + 1; j < K.anchor; ++j) {
        const ggml_tensor * t = P.node(j);
        if (P.is_member(K, j)) {
            continue;
        }
        if (P.exec[j]) {
            if (bad_write(j, nf_range_of(t))) {
                why = "node " + std::to_string(j) + " " + t->name + " writes an input or store";
                return false;
            }
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (t->src[s] != nullptr && bad_read(j, nf_range_of(t->src[s]))) {
                    why = "node " + std::to_string(j) + " " + t->name + " reads a store";
                    return false;
                }
            }
        }
        const int ko = anchor_of[j];
        if (ko >= 0 && &P.ks[ko] != &K) {
            for (const ggml_tensor * x : P.ks[ko].ins) {
                if (bad_read(j, nf_range_of(x))) {
                    why = "kernel at node " + std::to_string(j) + " reads a store";
                    return false;
                }
            }
            for (int s : P.ks[ko].stores) {
                if (bad_write(j, nf_range_of(P.node(s)))) {
                    why = "kernel at node " + std::to_string(j) + " writes an input or store";
                    return false;
                }
            }
        }
    }
    return true;
}

static void nf_dump_graph(const nf_graph & P) {
    static const char * kn[] = { "-", "view", "pw", "row", "conv" };
    for (int i = 0; i < P.n; ++i) {
        const ggml_tensor * t = P.node(i);
        int kid = -1;
        for (int k = 0; k < (int) P.ks.size(); ++k) {
            if (P.ks[k].alive && P.is_member(P.ks[k], i)) {
                kid = k;
            }
        }
        std::string srcs;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (t->src[s] != nullptr) {
                const int j = P.find(t->src[s]);
                srcs += " " + (j >= 0 ? std::to_string(j) : std::string("L:") + t->src[s]->name);
            }
        }
        fprintf(stderr, "nf %5d %-12s %-4s %s%s k=%-4d %-5s [%lld,%lld,%lld,%lld] %-28s <-%s\n", i, ggml_op_desc(t), kn[P.kind[i]],
                      P.mat.size() && P.mat[i] ? "M" : " ", P.taint.size() && P.taint[i] ? "T" : " ", kid, ggml_type_name(t->type),
                      (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], t->name, srcs.c_str());
    }
}

struct nf_graph_lite {
    std::vector<const ggml_tensor *> stores;
};

struct nf_verify_stats {
    std::mutex mtx;
    std::unordered_map<std::string, std::array<double, 3>> by_name; // count, max abs err, max rel err

    ~nf_verify_stats() {
        if (by_name.empty()) {
            return;
        }
        fprintf(stderr, "nvrtc-fusion verify: %zu kernel kinds\n", by_name.size());
        for (const auto & [name, v] : by_name) {
            fprintf(stderr, "  %-40s n=%6.0f max_abs=%.3e max_rel=%.3e\n", name.c_str(), v[0], v[1], v[2]);
        }
    }
};

static nf_verify_stats & nf_vstats() {
    static nf_verify_stats s;
    return s;
}

static float nf_host_value(const char * p, ggml_type t) {
    switch (t) {
        case GGML_TYPE_F16:  return ggml_fp16_to_fp32(*(const ggml_fp16_t *) p);
        case GGML_TYPE_BF16: return ggml_bf16_to_fp32(*(const ggml_bf16_t *) p);
        default:             return *(const float *) p;
    }
}

static void nf_verify_kernel(ggml_backend_cuda_context & ctx, const nf_graph_lite & G, nf_kernel & k) {
    cudaStream_t stream = ctx.stream();
    cudaStreamCaptureStatus cs;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &cs));
    if (cs != cudaStreamCaptureStatusNone) {
        return;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<uint64_t> params = k.params;
    std::vector<void *>   scratch;
    for (size_t i = 0; i < k.stores.size(); ++i) {
        const ggml_tensor * st = G.stores[i];
        void * d = nullptr;
        CUDA_CHECK(cudaMalloc(&d, ggml_nbytes(st) + 256));
        scratch.push_back(d);
        if (k.store_slots[i] >= 0) {
            params[k.store_slots[i]] = (uint64_t) (uintptr_t) d;
        }
    }
    void * args[] = { params.data() };
    if (cuLaunchKernel(k.fn, k.grid[0], k.grid[1], k.grid[2], k.block[0], k.block[1], k.block[2], 0, (CUstream) stream, args, nullptr) != CUDA_SUCCESS) {
        GGML_ABORT("nvrtc-fusion: verify launch failed");
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < k.stores.size(); ++i) {
        const ggml_tensor * st = G.stores[i];
        const size_t nbytes = ggml_nbytes(st);
        std::vector<char> ref(nbytes), got(nbytes);
        CUDA_CHECK(cudaMemcpy(ref.data(), st->data, nbytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(got.data(), scratch[i], nbytes, cudaMemcpyDeviceToHost));
        double max_ref = 0.0, max_err = 0.0;
        for (int64_t i3 = 0; i3 < st->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < st->ne[2]; ++i2) {
                for (int64_t i1 = 0; i1 < st->ne[1]; ++i1) {
                    for (int64_t i0 = 0; i0 < st->ne[0]; ++i0) {
                        const size_t off = i0 * st->nb[0] + i1 * st->nb[1] + i2 * st->nb[2] + i3 * st->nb[3];
                        const double a = nf_host_value(ref.data() + off, st->type);
                        const double b = nf_host_value(got.data() + off, st->type);
                        max_ref = std::max(max_ref, std::fabs(a));
                        max_err = std::max(max_err, std::isfinite(a - b) ? std::fabs(a - b) : 1e30);
                    }
                }
            }
        }
        std::string name = st->name;
        const size_t dash = name.rfind('-');
        if (dash != std::string::npos) {
            name = name.substr(0, dash);
        }
        name += std::string(" (") + ggml_op_desc(st) + ")";
        {
            nf_verify_stats & vs = nf_vstats();
            std::lock_guard<std::mutex> lock(vs.mtx);
            auto & v = vs.by_name[name];
            v[0] += 1;
            v[1] = std::max(v[1], max_err);
            v[2] = std::max(v[2], max_err / std::max(max_ref, 1e-30));
        }
        CUDA_CHECK(cudaFree(scratch[i]));
    }
}


struct ggml_cuda_nvrtc_plan {
    int                  n_nodes = 0;
    int                  verify_next = 0;
    std::vector<uint8_t> claimed;
    std::vector<int>     anchor_kernel;
    std::vector<int>     next_claimed;
    std::vector<nf_kernel> ks;
    std::vector<nf_graph_lite> stores;
};

// single-token and small-batch graphs keep the hand-written mat-vec fusions (scale, bias, GLU, norm)
static bool nf_small_batch(const ggml_cgraph * cgraph) {
    static const int64_t min_rows = getenv("GGML_CUDA_NVRTC_MIN_ROWS") ? atoll(getenv("GGML_CUDA_NVRTC_MIN_ROWS")) : 8;
    int64_t rows = -1;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * t = cgraph->nodes[i];
        if ((t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_MUL_MAT_ID) && t->src[1] != nullptr) {
            rows = std::max(rows, ggml_nrows(t->src[1]));
        }
    }
    return rows >= 0 && rows <= min_rows;
}

bool ggml_cuda_nvrtc_fusion_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_NVRTC_FUSION");
        return env == nullptr || atoi(env) != 0;
    }();
    return enabled;
}

void ggml_cuda_nvrtc_fusion_add_alloc_deps(
        const ggml_cgraph * cgraph, void (*add_alloc_dep)(void *, ggml_tensor *, ggml_tensor *), void * user_data) {
    if (!ggml_cuda_nvrtc_fusion_enabled() || add_alloc_dep == nullptr || cgraph->n_nodes == 0 || nf_small_batch(cgraph)) {
        return;
    }
    nf_graph P;
    P.build(cgraph);
    for (const nf_kernel & k : P.ks) {
        if (!k.alive) {
            continue;
        }
        for (const ggml_tensor * x : k.ext) {
            const ggml_tensor * base = x->view_src ? x->view_src : x;
            if (base->data == nullptr && base->buffer == nullptr) {
                add_alloc_dep(user_data, const_cast<ggml_tensor *>(x), P.node(k.anchor));
            }
        }
    }
}

// kernels with the same structure share their source, function and shape-dependent parameters.
// tensors are numbered in a canonical order: members by node index, then their sources
struct nf_cached_kernel {
    bool                             ok = false;
    std::string                      name;
    CUfunction                       fn = nullptr;
    std::vector<uint64_t>            params;
    std::vector<std::pair<int, int>> ptr_slots;  // slot, canonical tensor
    std::vector<std::array<int, 3>>  sink_slots; // slot, position in the kernel's sink list, field
    std::vector<int>                 ins;
    std::vector<uint8_t>             ins_identity;
    std::vector<int>                 store_slots;
    unsigned                         grid[3]  = { 1, 1, 1 };
    unsigned                         block[3] = { 1, 1, 1 };
};

static std::vector<const ggml_tensor *> nf_canonical(const nf_graph & P, const nf_kernel & k) {
    std::vector<int> members = k.cone;
    members.insert(members.end(), k.stores.begin(), k.stores.end());
    std::sort(members.begin(), members.end());
    std::vector<const ggml_tensor *> list;
    auto add = [&](const ggml_tensor * t) {
        if (t != nullptr && std::find(list.begin(), list.end(), t) == list.end()) {
            list.push_back(t);
        }
    };
    for (int m : members) {
        add(P.node(m));
    }
    for (int m : members) {
        const ggml_tensor * t = P.node(m);
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            add(t->src[s]);
            if (t->src[s] != nullptr) {
                add(t->src[s]->view_src);
            }
        }
    }
    return list;
}

static std::string nf_signature(const nf_graph & P, const nf_kernel & k, const std::vector<const ggml_tensor *> & canon,
                                const std::vector<ggml_cuda_nvrtc_fp8_sink> * sinks, int device) {
    struct desc {
        int32_t role, op, type, kind, taint, no_f32, view_src;
        int32_t src[GGML_MAX_SRC];
        int64_t ne[4];
        size_t  nb[4];
        size_t  view_offs;
        int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    };
    auto index_of = [&](const ggml_tensor * t) {
        const auto it = std::find(canon.begin(), canon.end(), t);
        return it == canon.end() ? -1 : (int32_t) (it - canon.begin());
    };
    std::string sig;
    sig.reserve(64 + canon.size() * sizeof(desc));
    const int64_t head[7] = { device, k.row, k.dom[0], k.dom[1], k.dom[2], k.dom[3], (int64_t) k.sinks.size() };
    sig.append((const char *) head, sizeof(head));
    for (const ggml_tensor * t : canon) {
        desc d;
        memset(&d, 0, sizeof(d));
        const int i = P.find(t);
        const bool member = i >= 0 && P.is_member(k, i);
        d.role = member ? (std::binary_search(k.stores.begin(), k.stores.end(), i) ? 2 : 1) : 0;
        d.type = t->type;
        d.view_src = index_of(t->view_src);
        memcpy(d.ne, t->ne, sizeof(d.ne));
        memcpy(d.nb, t->nb, sizeof(d.nb));
        d.view_offs = t->view_offs;
        if (member) {
            d.op     = t->op;
            d.kind   = P.kind[i];
            d.taint  = P.taint[i];
            d.no_f32 = std::binary_search(k.no_f32.begin(), k.no_f32.end(), i);
            memcpy(d.op_params, t->op_params, sizeof(d.op_params));
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                d.src[s] = index_of(t->src[s]);
            }
        } else {
            d.op = nf_is_view_op(t->op) ? t->op : -1;
        }
        sig.append((const char *) &d, sizeof(d));
    }
    for (int si : k.sinks) {
        const int32_t b = index_of(P.node(P.base_of((*sinks)[si].src1)));
        sig.append((const char *) &b, sizeof(b));
    }
    return sig;
}

static std::mutex                                        nf_cache_mutex;
static std::unordered_map<std::string, nf_cached_kernel> nf_cache;

// attach FP8 activation groups to the kernels that store their src1
static void nf_assign_sinks(nf_graph & P, std::vector<ggml_cuda_nvrtc_fp8_sink> & sinks) {
    static const bool keep_f32 = getenv("GGML_CUDA_NVRTC_FP8_KEEP_F32") != nullptr;
    for (int si = 0; si < (int) sinks.size(); ++si) {
        const ggml_cuda_nvrtc_fp8_sink & sk = sinks[si];
        const int j = P.base_of(sk.src1);
        if (j < 0 || !P.is_store[j] || P.owner[j] < 0 || sk.input_scale <= 0.0f) {
            continue;
        }
        // src1 may be a reshape of the stored tensor: the element order must be the same
        if (sk.src1 != P.node(j) && (!ggml_is_contiguous(sk.src1) || sk.src1->view_offs != 0 ||
                                     ggml_nelements(sk.src1) != ggml_nelements(P.node(j)))) {
            continue;
        }
        nf_kernel & k = P.ks[P.owner[j]];
        const ggml_tensor * st = P.node(j);
        if (!k.alive || st->op == GGML_OP_CPY || st->type != GGML_TYPE_F32 || !ggml_is_contiguous(st) || k.anchor < sk.min_anchor) {
            continue;
        }
        k.sinks.push_back(si);
    }
    if (keep_f32 || nf_verify()) {
        return;
    }
    for (nf_kernel & k : P.ks) {
        if (!k.alive || k.sinks.empty()) {
            continue;
        }
        for (int s : k.stores) {
            std::unordered_set<const ggml_tensor *> readers;
            for (int si : k.sinks) {
                if (P.base_of(sinks[si].src1) == s) {
                    readers.insert(sinks[si].matmuls.begin(), sinks[si].matmuls.end());
                }
            }
            if (readers.empty()) {
                continue;
            }
            std::vector<std::pair<int, bool>> ec;
            P.eff_cons(s, ec, false);
            bool covered = !(P.node(s)->flags & GGML_TENSOR_FLAG_OUTPUT) && ggml_node_get_use_count(P.g, s) == (int) P.cons[s].size();
            for (const auto & [c, via] : ec) {
                covered = covered && c >= 0 && readers.count(P.node(c)) && (!via || P.base_of(P.node(c)->src[1]) == s);
            }
            if (covered) {
                k.no_f32.push_back(s);
            }
        }
        std::sort(k.no_f32.begin(), k.no_f32.end());
    }
}

ggml_cuda_nvrtc_plan * ggml_cuda_nvrtc_fusion_plan(
        ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, std::vector<ggml_cuda_nvrtc_fp8_sink> * sinks) {
    if (!ggml_cuda_nvrtc_fusion_enabled() || cgraph->n_nodes == 0 || nf_small_batch(cgraph)) {
        return nullptr;
    }
    const auto t0 = std::chrono::steady_clock::now();
    nf_graph P;
    P.build(cgraph);
    if (P.n_alive() == 0) {
        return nullptr;
    }
    const auto t1 = std::chrono::steady_clock::now();
    double ms_lookup = 0.0;
    static const bool fp8_off = getenv("GGML_CUDA_NVRTC_FP8") != nullptr && atoi(getenv("GGML_CUDA_NVRTC_FP8")) == 0;
    if (sinks != nullptr && !fp8_off && !nf_verify()) {
        nf_assign_sinks(P, *sinks);
    }

    int n_gen_fail = 0, n_compile_fail = 0, n_mem_fail = 0;
    for (nf_kernel & k : P.ks) {
        if (!k.alive) {
            continue;
        }
        const std::vector<const ggml_tensor *> canon = nf_canonical(P, k);
        const std::string sig = nf_signature(P, k, canon, sinks, ctx.device);
        {
            std::lock_guard<std::mutex> lock(nf_cache_mutex);
            auto it = nf_cache.find(sig);
            if (it != nf_cache.end()) {
                const nf_cached_kernel & c = it->second;
                if (!c.ok) {
                    k.alive = false;
                    n_gen_fail++;
                    continue;
                }
                k.name   = c.name;
                k.fn     = c.fn;
                k.params = c.params;
                for (const auto & [slot, ci] : c.ptr_slots) {
                    k.params[slot] = (uint64_t) (uintptr_t) canon[ci]->data;
                }
                for (const auto & ss : c.sink_slots) {
                    const ggml_cuda_nvrtc_fp8_sink & sk = (*sinks)[k.sinks[ss[1]]];
                    if (ss[2] == 0) {
                        k.params[ss[0]] = (uint64_t) (uintptr_t) sk.values;
                    } else if (ss[2] == 1) {
                        k.params[ss[0]] = (uint64_t) (uintptr_t) sk.scale;
                    } else {
                        uint32_t u;
                        memcpy(&u, &sk.input_scale, sizeof(u));
                        k.params[ss[0]] = u;
                    }
                }
                k.ins.clear();
                for (int ci : c.ins) {
                    k.ins.push_back(canon[ci]);
                }
                k.ins_identity = c.ins_identity;
                k.store_slots  = c.store_slots;
                memcpy(k.grid, c.grid, sizeof(k.grid));
                memcpy(k.block, c.block, sizeof(k.block));
                continue;
            }
        }

        nf_gen gen(P, k);
        gen.sinks = sinks;
        const std::string src = gen.build();
        auto remember = [&](bool ok) {
            nf_cached_kernel c;
            c.ok = ok;
            if (ok) {
                auto index_of = [&](const ggml_tensor * t) {
                    const auto it = std::find(canon.begin(), canon.end(), t);
                    return it == canon.end() ? -1 : (int) (it - canon.begin());
                };
                c.name = k.name;
                c.fn = k.fn;
                c.params = k.params;
                for (const auto & [slot, t] : gen.ptr_slots) {
                    const int ci = index_of(t);
                    if (ci < 0) {
                        return;
                    }
                    c.ptr_slots.push_back({ slot, ci });
                }
                for (const auto & ss : gen.sink_slots) {
                    const int pos = (int) (std::find(k.sinks.begin(), k.sinks.end(), ss[1]) - k.sinks.begin());
                    c.sink_slots.push_back({ ss[0], pos, ss[2] });
                }
                for (const ggml_tensor * t : k.ins) {
                    const int ci = index_of(t);
                    if (ci < 0) {
                        return;
                    }
                    c.ins.push_back(ci);
                }
                c.ins_identity = k.ins_identity;
                c.store_slots  = k.store_slots;
                memcpy(c.grid, k.grid, sizeof(c.grid));
                memcpy(c.block, k.block, sizeof(c.block));
            }
            std::lock_guard<std::mutex> lock(nf_cache_mutex);
            nf_cache.emplace(sig, std::move(c));
        };
        if (src.empty()) {
            remember(false);
            if (nf_dump() > 1) {
                fprintf(stderr, "nvrtc-fusion: kernel at node %d not generated: %s\n", k.anchor, gen.err.c_str());
            }
            k.alive = false;
            n_gen_fail++;
            continue;
        }
        const auto tl = std::chrono::steady_clock::now();
        k.fn = nf_get_function(ctx.device, src, k.name);
        ms_lookup += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tl).count();
        remember(k.fn != nullptr);
        if (k.fn == nullptr) {
            k.alive = false;
            n_compile_fail++;
            continue;
        }
        if (nf_dump() > 2) {
            fprintf(stderr, "nvrtc-fusion: kernel at node %d (%s):\n%s\n", k.anchor, P.node(k.anchor)->name, src.c_str());
        }
    }

    const auto t2 = std::chrono::steady_clock::now();
    std::vector<int> anchor_of(P.n, -1);
    for (int iter = 0; iter < 16; ++iter) {
        P.settle();
        std::fill(anchor_of.begin(), anchor_of.end(), -1);
        for (int ik = 0; ik < (int) P.ks.size(); ++ik) {
            if (P.ks[ik].alive) {
                anchor_of[P.ks[ik].anchor] = ik;
            }
        }
        bool killed = false;
        for (nf_kernel & k : P.ks) {
            std::string why;
            if (k.alive && !nf_check_memory(P, k, anchor_of, why)) {
                k.alive = false;
                killed  = true;
                n_mem_fail++;
                if (nf_dump() > 1) {
                    fprintf(stderr, "nvrtc-fusion: kernel at node %d (%s) rejected: %s\n", k.anchor, P.node(k.anchor)->name, why.c_str());
                }
            }
        }
        if (!killed) {
            break;
        }
    }
    P.settle();
    const bool dump_now = nf_dump() > 1 && [&] {
        static std::unordered_set<int> seen;
        static std::mutex seen_mtx;
        std::lock_guard<std::mutex> lock(seen_mtx);
        return seen.insert(P.n).second;
    }();
    if (dump_now) {
        nf_dump_graph(P);
    }

    int n_sinks = 0, n_no_f32 = 0;
    if (sinks != nullptr) {
        for (const nf_kernel & k : P.ks) {
            if (!k.alive) {
                continue;
            }
            n_no_f32 += (int) k.no_f32.size();
            n_sinks  += (int) k.sinks.size();
            for (int si : k.sinks) {
                ggml_cuda_nvrtc_fp8_sink & sk = (*sinks)[si];
                sk.fulfilled    = true;
                sk.src1_written = !std::binary_search(k.no_f32.begin(), k.no_f32.end(), P.base_of(sk.src1));
            }
        }
    }

    ggml_cuda_nvrtc_plan * plan = new ggml_cuda_nvrtc_plan;
    plan->n_nodes = P.n;
    plan->claimed.assign(P.n, 0);
    plan->anchor_kernel.assign(P.n, -1);
    plan->next_claimed.assign(P.n + 1, P.n);
    int n_kernels = 0, n_claimed = 0;
    for (nf_kernel & k : P.ks) {
        if (!k.alive) {
            continue;
        }
        plan->anchor_kernel[k.anchor] = (int) plan->ks.size();
        nf_graph_lite gl;
        for (int s : k.stores) {
            gl.stores.push_back(P.node(s));
        }
        plan->stores.push_back(gl);
        plan->ks.push_back(std::move(k));
        n_kernels++;
    }
    for (int i = 0; i < P.n; ++i) {
        plan->claimed[i] = P.member[i] && !P.exec[i];
        n_claimed += plan->claimed[i];
    }
    for (int i = P.n - 1; i >= 0; --i) {
        plan->next_claimed[i] = plan->claimed[i] ? i : plan->next_claimed[i + 1];
    }

    if (nf_dump() > 0) {
        static std::unordered_set<int64_t> seen;
        static std::mutex seen_mtx;
        std::lock_guard<std::mutex> lock(seen_mtx);
        if (seen.insert(((int64_t) P.n << 32) | ((int64_t) n_kernels << 16) | (int64_t) (n_sinks * 64 + n_no_f32)).second) {
            const auto t3 = std::chrono::steady_clock::now();
            auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
            fprintf(stderr, "nvrtc-fusion: graph of %d nodes: %d kernels cover %d nodes, %d FP8 copies (%d F32 stores skipped) "
                    "(codegen fail %d, compile fail %d, memory fail %d), plan %.2f ms (group %.2f, codegen %.2f incl. lookup %.2f, checks %.2f)\n",
                    P.n, n_kernels, n_claimed, n_sinks, n_no_f32, n_gen_fail, n_compile_fail, n_mem_fail, ms(t0, t3), ms(t0, t1), ms(t1, t2),
                    ms_lookup, ms(t2, t3));
        }
    }
    if (n_kernels == 0) {
        delete plan;
        return nullptr;
    }
    return plan;
}

void ggml_cuda_nvrtc_fusion_free(ggml_cuda_nvrtc_plan * plan) {
    delete plan;
}

bool ggml_cuda_nvrtc_fusion_run(ggml_cuda_nvrtc_plan * plan, ggml_backend_cuda_context & ctx, int i) {
    if (i < 0 || i >= plan->n_nodes) {
        return false;
    }
    if (nf_verify()) {
        return false;
    }
    const int ik = plan->anchor_kernel[i];
    if (ik >= 0) {
        nf_kernel & k = plan->ks[ik];
        void * args[] = { k.params.data() };
        const CUresult res = cuLaunchKernel(k.fn, k.grid[0], k.grid[1], k.grid[2], k.block[0], k.block[1], k.block[2], 0,
                                            (CUstream) ctx.stream(), args, nullptr);
        if (res != CUDA_SUCCESS) {
            const char * msg = nullptr;
            cuGetErrorString(res, &msg);
            GGML_ABORT("nvrtc-fusion: launch failed: %s", msg ? msg : "?");
        }
        return true;
    }
    return plan->claimed[i] != 0;
}

void ggml_cuda_nvrtc_fusion_after(ggml_cuda_nvrtc_plan * plan, ggml_backend_cuda_context & ctx, int last_done) {
    if (!nf_verify()) {
        return;
    }
    for (; plan->verify_next <= last_done && plan->verify_next < plan->n_nodes; ++plan->verify_next) {
        const int ik = plan->anchor_kernel[plan->verify_next];
        if (ik >= 0) {
            nf_verify_kernel(ctx, plan->stores[ik], plan->ks[ik]);
        }
    }
}

int ggml_cuda_nvrtc_fusion_next_claimed(const ggml_cuda_nvrtc_plan * plan, int i) {
    if (nf_verify()) {
        return plan->n_nodes;
    }
    if (i + 1 >= plan->n_nodes) {
        return plan->n_nodes;
    }
    return plan->next_claimed[i + 1];
}

#endif // GGML_CUDA_NVRTC_FUSION
