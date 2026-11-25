#include "l2-clip.cuh"

#include "ggml-impl.h"

#include <climits>

// one warp per row; every lane reads its columns of the row before the warp reduction and writes only those columns
// after it, so dst may be x itself (x is not __restrict__ for that reason)
template <bool has_scale>
static __global__ void l2_clip_f32(const float * x, const float * __restrict__ scale, float * dst,
                                   const int ncols, const int nrows, const int scale_rows,
                                   const float thr, const float b_scale, const float b_bias, const float lo, const float hi) {
    const int row = blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    if (row >= nrows) {
        return;
    }
    const int lane = threadIdx.x % WARP_SIZE;

    const float * xr = x   + (int64_t) row * ncols;
    float *       dr = dst + (int64_t) row * ncols;
    const float * sr = has_scale ? scale + (int64_t) (row % scale_rows) * ncols : nullptr;

    float sum = 0.0f;
    for (int c = lane; c < ncols; c += WARP_SIZE) {
        const float v = has_scale ? xr[c] * sr[c] : xr[c];
        sum += v * v;
    }
    sum = warp_reduce_sum(sum);

    // same order of operations as the graph: sqrt, scale and bias, thr / norm, clamp
    const float norm = sqrtf(sum) * b_scale + b_bias;
    const float f    = fminf(fmaxf(thr / norm, lo), hi);

    for (int c = lane; c < ncols; c += WARP_SIZE) {
        const float v = has_scale ? xr[c] * sr[c] : xr[c];
        dr[c] = v * f;
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

bool ggml_cuda_match_l2_clip(const ggml_cgraph * cgraph, int i, ggml_cuda_l2_clip_match & m) {
    const ggml_tensor * first = cgraph->nodes[i];
    const bool has_scale = first->op == GGML_OP_MUL;
    if (!has_scale && first->op != GGML_OP_SQR) {
        return false;
    }

    static const ggml_op ops_scale[] = { GGML_OP_MUL, GGML_OP_SQR, GGML_OP_SUM_ROWS, GGML_OP_SQRT, GGML_OP_SCALE,
                                         GGML_OP_FILL, GGML_OP_DIV, GGML_OP_CLAMP, GGML_OP_MUL };
    const ggml_op * ops = has_scale ? ops_scale : ops_scale + 1;
    const int n = has_scale ? 9 : 8;
    if (i + n > cgraph->n_nodes) {
        return false;
    }
    for (int k = 0; k < n; ++k) {
        if (cgraph->nodes[i + k]->op != ops[k]) {
            return false;
        }
    }

    const int b = has_scale ? 1 : 0; // index of the SQR
    const ggml_tensor * s    = has_scale ? cgraph->nodes[i] : nullptr;
    const ggml_tensor * sqr  = cgraph->nodes[i + b + 0];
    const ggml_tensor * sum  = cgraph->nodes[i + b + 1];
    const ggml_tensor * sq   = cgraph->nodes[i + b + 2];
    const ggml_tensor * sb   = cgraph->nodes[i + b + 3];
    const ggml_tensor * fill = cgraph->nodes[i + b + 4];
    const ggml_tensor * div  = cgraph->nodes[i + b + 5];
    const ggml_tensor * cl   = cgraph->nodes[i + b + 6];
    ggml_tensor *       out  = cgraph->nodes[i + b + 7];

    const ggml_tensor * x = has_scale ? s->src[0] : sqr->src[0];
    const ggml_tensor * v = has_scale ? s : x; // the clipped values

    // topology of the chain
    if (sqr->src[0] != v || sum->src[0] != sqr || sq->src[0] != sum || sb->src[0] != sq || fill->src[0] != sb ||
        div->src[0] != fill || div->src[1] != sb || cl->src[0] != div) {
        return false;
    }
    if (!((out->src[0] == v && out->src[1] == cl) || (out->src[0] == cl && out->src[1] == v))) {
        return false;
    }
    // no in-place intermediates: e.g. an in-place FILL overwrites the norm before the DIV reads it, which the kernel
    // does not reproduce
    for (const ggml_tensor * t : { s, sqr, sum, sq, sb, fill, div, cl }) {
        if (t && t->view_src) {
            return false;
        }
    }

    // the intermediate nodes must not be read outside the chain
    const int outputs[] = { i + n - 1 };
    if (!ggml_can_fuse_subgraph(cgraph, i, n, ops, outputs, 1)) {
        return false;
    }

    // rows of x, one norm per row, a scale broadcast over whole rows
    if (!l2_clip_f32_contiguous(x) || !l2_clip_f32_contiguous(out) || !ggml_are_same_shape(x, out) ||
        x->ne[0] > 8192 || ggml_nrows(x) > INT_MAX || ggml_nrows(sum) != ggml_nrows(x) || sum->ne[0] != 1) {
        return false;
    }
    for (const ggml_tensor * t : { sqr, sum, sq, sb, fill, div, cl }) {
        if (t->type != GGML_TYPE_F32) {
            return false;
        }
    }
    const ggml_tensor * scale = nullptr;
    if (has_scale) {
        scale = s->src[0] == x ? s->src[1] : s->src[0];
        if (!l2_clip_f32_contiguous(scale) || !ggml_are_same_shape(s, x) || scale->ne[0] != x->ne[0] ||
            scale->ne[2] != 1 || scale->ne[3] != 1 || (scale->ne[1] != x->ne[1] && scale->ne[1] != 1)) {
            return false;
        }
    }

    // out may be x itself (in place), but must not overlap x otherwise or the scale
    if ((out->data != x->data && l2_clip_overlap(out, x)) || (scale && l2_clip_overlap(out, scale))) {
        return false;
    }

    m.x       = x;
    m.scale   = scale;
    m.dst     = out;
    m.thr     = ggml_get_op_params_f32(fill, 0);
    m.b_scale = ggml_get_op_params_f32(sb, 0);
    m.b_bias  = ggml_get_op_params_f32(sb, 1);
    m.lo      = ggml_get_op_params_f32(cl, 0);
    m.hi      = ggml_get_op_params_f32(cl, 1);
    m.n_nodes = n;
    return true;
}

void ggml_cuda_op_l2_clip(ggml_backend_cuda_context & ctx, const ggml_cuda_l2_clip_match & m) {
    const int ncols = (int) m.x->ne[0];
    const int nrows = (int) ggml_nrows(m.x);
    const int rows_per_block = 4;
    const dim3 block(WARP_SIZE * rows_per_block, 1, 1);
    const dim3 grid((nrows + rows_per_block - 1) / rows_per_block, 1, 1);
    cudaStream_t stream = ctx.stream();

    const float * x   = (const float *) m.x->data;
    float *       dst = (float *) m.dst->data;
    if (m.scale) {
        l2_clip_f32<true><<<grid, block, 0, stream>>>(x, (const float *) m.scale->data, dst, ncols, nrows,
            (int) m.scale->ne[1], m.thr, m.b_scale, m.b_bias, m.lo, m.hi);
    } else {
        l2_clip_f32<false><<<grid, block, 0, stream>>>(x, nullptr, dst, ncols, nrows, 1,
            m.thr, m.b_scale, m.b_bias, m.lo, m.hi);
    }
}
