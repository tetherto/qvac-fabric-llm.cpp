#include "l2-clip.cuh"

#include "ggml-impl.h"

#include <climits>

static constexpr int L2_CLIP_MAX_COLS       = 8192;
static constexpr int L2_CLIP_ROWS_PER_BLOCK = 4;

// one warp per row; every lane reads its columns of the row before the warp reduction and writes only those columns
// after it, so dst may be x itself (x is not __restrict__ for that reason)
template <bool has_w, bool has_f>
static __global__ void l2_clip_f32(const float * x, const float * __restrict__ w, const float * __restrict__ f, float * dst,
                                   const int ncols, const int nrows, const int w_rows, const int ne1, const int ne2,
                                   const int f_ne2, const int f_ne3, const float eps, const float t, const float t_bias) {
    const int row = blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    if (row >= nrows) {
        return;
    }
    const int lane = threadIdx.x % WARP_SIZE;

    const float * xr = x   + (int64_t) row * ncols;
    float *       dr = dst + (int64_t) row * ncols;
    const float * wr = has_w ? w + (int64_t) (row % w_rows) * ncols : nullptr;

    float sum = 0.0f;
    for (int c = lane; c < ncols; c += WARP_SIZE) {
        const float v = has_w ? xr[c] * wr[c] : xr[c];
        sum += v * v;
    }
    sum = warp_reduce_sum(sum);

    // same order of operations as the graph: the l2_norm scale, SCALE, then the per-token factor
    const float scale = rsqrtf(fmaxf(sum, eps * eps));
    float fv = 1.0f;
    if constexpr (has_f) {
        const int token = row / ne1;
        fv = f[(token / ne2 % f_ne3) * f_ne2 + token % ne2 % f_ne2];
    }

    for (int c = lane; c < ncols; c += WARP_SIZE) {
        const float v = has_w ? xr[c] * wr[c] : xr[c];
        const float y = t * (scale * v) + t_bias;
        dr[c] = has_f ? y * fv : y;
    }
}

static bool l2_clip_f32_contiguous(const ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t);
}

static bool l2_clip_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data;
    const char * b0 = (const char *) b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

// a weight broadcast over whole rows of x
static bool l2_clip_weight_ok(const ggml_tensor * w, const ggml_tensor * x) {
    return l2_clip_f32_contiguous(w) && w->ne[0] == x->ne[0] && w->ne[2] == 1 && w->ne[3] == 1 &&
           (w->ne[1] == x->ne[1] || w->ne[1] == 1);
}

// one factor per token, broadcast over the rows of that token
static bool l2_clip_factor_ok(const ggml_tensor * f, const ggml_tensor * x) {
    return l2_clip_f32_contiguous(f) && f->ne[0] == 1 && f->ne[1] == 1 &&
           (f->ne[2] == x->ne[2] || f->ne[2] == 1) && (f->ne[3] == x->ne[3] || f->ne[3] == 1);
}

static bool l2_clip_match_chain(const ggml_cgraph * cgraph, int i, bool has_w, bool has_f, ggml_cuda_l2_clip_match & m) {
    ggml_op ops[4];
    int     n = 0;
    if (has_w) {
        ops[n++] = GGML_OP_MUL;
    }
    ops[n++] = GGML_OP_L2_NORM;
    ops[n++] = GGML_OP_SCALE;
    if (has_f) {
        ops[n++] = GGML_OP_MUL;
    }
    if (i + n > cgraph->n_nodes) {
        return false;
    }
    for (int k = 0; k < n; ++k) {
        if (cgraph->nodes[i + k]->op != ops[k]) {
            return false;
        }
    }

    const ggml_tensor * mw  = has_w ? cgraph->nodes[i] : nullptr;
    const ggml_tensor * l2  = cgraph->nodes[i + (has_w ? 1 : 0)];
    const ggml_tensor * sc  = cgraph->nodes[i + (has_w ? 2 : 1)];
    ggml_tensor *       out = cgraph->nodes[i + n - 1];

    const ggml_tensor * x = has_w ? mw->src[0] : l2->src[0];
    const ggml_tensor * w = has_w ? mw->src[1] : nullptr;
    const ggml_tensor * f = has_f ? out->src[1] : nullptr;

    // topology of the chain
    if ((has_w && l2->src[0] != mw) || sc->src[0] != l2 || (has_f && out->src[0] != sc)) {
        return false;
    }
    // no in-place intermediates, and the intermediate nodes must not be read outside the chain
    for (const ggml_tensor * t : { mw, l2, has_f ? sc : nullptr }) {
        if (t && t->view_src) {
            return false;
        }
    }
    const int outputs[] = { i + n - 1 };
    if (!ggml_can_fuse_subgraph(cgraph, i, n, ops, outputs, 1)) {
        return false;
    }

    // rows of x, one norm per row
    if (!l2_clip_f32_contiguous(x) || !l2_clip_f32_contiguous(out) || !ggml_are_same_shape(x, out) ||
        x->ne[0] > L2_CLIP_MAX_COLS || ggml_nrows(x) > INT_MAX || l2->type != GGML_TYPE_F32 || sc->type != GGML_TYPE_F32) {
        return false;
    }
    if (has_w && (!ggml_are_same_shape(mw, x) || mw->type != GGML_TYPE_F32 || !l2_clip_weight_ok(w, x))) {
        return false;
    }
    if (has_f && !l2_clip_factor_ok(f, x)) {
        return false;
    }

    // out may be x itself (in place), but must not overlap x otherwise or the weight and the factor
    if ((out->data != x->data && l2_clip_overlap(out, x)) || (w && l2_clip_overlap(out, w)) || (f && l2_clip_overlap(out, f))) {
        return false;
    }

    m.x       = x;
    m.w       = w;
    m.f       = f;
    m.dst     = out;
    m.eps     = ggml_get_op_params_f32(l2, 0);
    m.t       = ggml_get_op_params_f32(sc, 0);
    m.t_bias  = ggml_get_op_params_f32(sc, 1);
    m.n_nodes = n;
    return true;
}

bool ggml_cuda_match_l2_clip(const ggml_cgraph * cgraph, int i, ggml_cuda_l2_clip_match & m) {
    const ggml_op op = cgraph->nodes[i]->op;
    if (op != GGML_OP_MUL && op != GGML_OP_L2_NORM) {
        return false;
    }
    const bool has_w = op == GGML_OP_MUL;
    // the longer chain first, so a following per-token MUL is absorbed
    return l2_clip_match_chain(cgraph, i, has_w, true, m) || l2_clip_match_chain(cgraph, i, has_w, false, m);
}

template <bool has_w, bool has_f>
static void l2_clip_f32_launch(const ggml_cuda_l2_clip_match & m, const dim3 grid, const dim3 block, cudaStream_t stream) {
    const ggml_tensor * x = m.x;
    l2_clip_f32<has_w, has_f><<<grid, block, 0, stream>>>(
        (const float *) x->data, has_w ? (const float *) m.w->data : nullptr, has_f ? (const float *) m.f->data : nullptr,
        (float *) m.dst->data, (int) x->ne[0], (int) ggml_nrows(x), has_w ? (int) m.w->ne[1] : 1, (int) x->ne[1],
        (int) x->ne[2], has_f ? (int) m.f->ne[2] : 1, has_f ? (int) m.f->ne[3] : 1, m.eps, m.t, m.t_bias);
}

void ggml_cuda_op_l2_clip(ggml_backend_cuda_context & ctx, const ggml_cuda_l2_clip_match & m) {
    const int nrows = (int) ggml_nrows(m.x);
    const dim3 block(WARP_SIZE * L2_CLIP_ROWS_PER_BLOCK, 1, 1);
    const dim3 grid((nrows + L2_CLIP_ROWS_PER_BLOCK - 1) / L2_CLIP_ROWS_PER_BLOCK, 1, 1);
    cudaStream_t stream = ctx.stream();

    if (m.w && m.f) {
        l2_clip_f32_launch<true, true>(m, grid, block, stream);
    } else if (m.w) {
        l2_clip_f32_launch<true, false>(m, grid, block, stream);
    } else if (m.f) {
        l2_clip_f32_launch<false, true>(m, grid, block, stream);
    } else {
        l2_clip_f32_launch<false, false>(m, grid, block, stream);
    }
}
