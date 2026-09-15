#include "gdn-gates.cuh"

#define GDN_GATES_MAX_BATCH 8
#define GDN_GATES_THREADS   256

// block (j, t): output row j for token t; j < n is an alpha row, otherwise a beta row. The row's 2k bytes of weights
// are spread over the block so every thread issues its loads up front (a warp per row was latency-bound)
template <typename T>
static __global__ void gdn_gates(const T * __restrict__ wa, const T * __restrict__ wb, const float * __restrict__ x,
                                 const float * __restrict__ dt, const float * __restrict__ a,
                                 float * __restrict__ g, float * __restrict__ b, const int k, const int n) {
    const int t   = blockIdx.y;
    const int j   = blockIdx.x;
    const int tid = threadIdx.x;

    const bool is_b = j >= n;
    const int  jj   = is_b ? j - n : j;

    const T     * w  = (is_b ? wb : wa) + (int64_t) jj*k;
    const float * xt = x + (int64_t) t*k;

    // same element math as mul_mat_vec_f: fp32 products of the converted weights, summed over the row
    float acc = 0.0f;
#pragma unroll 4
    for (int i = 2*tid; i < k; i += 2*GDN_GATES_THREADS) {
        const float2 x2 = *(const float2 *) (xt + i);
        if constexpr (std::is_same_v<T, half>) {
            const half2 w2 = *(const half2 *) (w + i);
            ggml_cuda_mad(acc, w2.x, x2.x);
            ggml_cuda_mad(acc, w2.y, x2.y);
        } else {
            const nv_bfloat162 w2 = *(const nv_bfloat162 *) (w + i);
            ggml_cuda_mad(acc, w2.x, x2.x);
            ggml_cuda_mad(acc, w2.y, x2.y);
        }
    }

    __shared__ float s_red[WARP_SIZE];
    acc = block_reduce<block_reduce_method::SUM, GDN_GATES_THREADS>(acc, s_red);

    if (tid == 0) {
        if (is_b) {
            b[(int64_t) t*n + jj] = 1.0f / (1.0f + expf(-acc));
        } else {
            const float s = acc + dt[jj];
            g[(int64_t) t*n + jj] = ((s > 20.0f) ? s : logf(1.0f + expf(s))) * a[jj];
        }
    }
}

bool ggml_cuda_gdn_gates_supported(const ggml_tensor * x, const ggml_tensor * w_alpha, const ggml_tensor * w_beta,
                                   const ggml_tensor * dt_bias, const ggml_tensor * a, const ggml_tensor * g, const ggml_tensor * b) {
    const int64_t k = x->ne[0];
    const int64_t n = w_alpha->ne[1];

    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[1] > GDN_GATES_MAX_BATCH || x->ne[2] != 1 || x->ne[3] != 1) {
        return false;
    }
    for (const ggml_tensor * w : { w_alpha, w_beta }) {
        if ((w->type != GGML_TYPE_F16 && w->type != GGML_TYPE_BF16) || w->type != w_alpha->type || !ggml_is_contiguous(w) ||
            w->ne[0] != k || w->ne[1] != n || w->ne[2] != 1 || w->ne[3] != 1) {
            return false;
        }
    }
    for (const ggml_tensor * v : { dt_bias, a }) {
        if (v->type != GGML_TYPE_F32 || !ggml_is_contiguous(v) || ggml_nelements(v) != n) {
            return false;
        }
    }
    for (const ggml_tensor * o : { g, b }) {
        if (o->type != GGML_TYPE_F32 || !ggml_is_contiguous(o) || ggml_nelements(o) != n*x->ne[1]) {
            return false;
        }
    }
    // paired loads
    return k % 2 == 0;
}

void ggml_cuda_op_gdn_gates(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * w_alpha, const ggml_tensor * w_beta,
                            const ggml_tensor * dt_bias, const ggml_tensor * a, ggml_tensor * g, ggml_tensor * b) {
    GGML_ASSERT(ggml_cuda_gdn_gates_supported(x, w_alpha, w_beta, dt_bias, a, g, b));

    const int k = x->ne[0];
    const int n = w_alpha->ne[1];

    const dim3 grid(2*n, x->ne[1], 1);
    const dim3 block(GDN_GATES_THREADS, 1, 1);

    if (w_alpha->type == GGML_TYPE_F16) {
        gdn_gates<half><<<grid, block, 0, ctx.stream()>>>(
            (const half *) w_alpha->data, (const half *) w_beta->data, (const float *) x->data,
            (const float *) dt_bias->data, (const float *) a->data, (float *) g->data, (float *) b->data, k, n);
    } else {
        gdn_gates<nv_bfloat16><<<grid, block, 0, ctx.stream()>>>(
            (const nv_bfloat16 *) w_alpha->data, (const nv_bfloat16 *) w_beta->data, (const float *) x->data,
            (const float *) dt_bias->data, (const float *) a->data, (float *) g->data, (float *) b->data, k, n);
    }
}
