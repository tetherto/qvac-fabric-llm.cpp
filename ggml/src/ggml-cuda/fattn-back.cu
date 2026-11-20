#include "fattn-back.cuh"
#include "fattn-back-mma.cuh"

#define FAB_D_MAX    256
#define FAB_PER_LANE (FAB_D_MAX/WARP_SIZE)
#define FAB_WARPS    8


// o and grad_o are [D, n_head, n_q, ne3], stats [2, n_q, n_head, ne3], contiguous like dst:
//   STATS: dst [2, n_q, n_head, ne3], (logsumexp, dot(grad_o, o)) per q row
//   DQ:    dst [D, n_q, n_head, ne3]
//   DKV:   dst [D, n_kv, n_head_kv, 2*ne3], dk in ne3 [0, ne3), dv in [ne3, 2*ne3)

static __device__ __forceinline__ float fab_to_float(const float x) { return x; }
static __device__ __forceinline__ float fab_to_float(const half  x) { return __half2float(x); }

template <typename T>
static __device__ __forceinline__ void fab_load_row(const T * row, const int64_t D, float * x) {
#pragma unroll
    for (int i = 0; i < FAB_PER_LANE; ++i) {
        const int d = threadIdx.x % WARP_SIZE + i*WARP_SIZE;
        x[i] = d < D ? fab_to_float(row[d]) : 0.0f;
    }
}

static __device__ __forceinline__ void fab_store_row(float * row, const int64_t D, const float * x) {
#pragma unroll
    for (int i = 0; i < FAB_PER_LANE; ++i) {
        const int d = threadIdx.x % WARP_SIZE + i*WARP_SIZE;
        if (d < D) {
            row[d] = x[i];
        }
    }
}

static __device__ __forceinline__ float fab_dot(const float * x, const float * y) {
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < FAB_PER_LANE; ++i) {
        s += x[i]*y[i];
    }
    return s;
}

// mask value of q row (i1, i2, i3) at kv position ic, the same for every lane of the warp
static __device__ __forceinline__ float fab_mask(
        const half * mask, const ggml_cuda_flash_attn_ext_back_kargs & args,
        const int64_t i1, const int64_t i2, const int64_t i3, const int64_t ic) {
    if (!mask) {
        return 0.0f;
    }
    const char * row = (const char *) mask + i1*args.nbm1 + (i2 % args.nem2)*args.nbm2 + (i3 % args.nem3)*args.nbm3;
    return __half2float(((const half *) row)[ic]);
}

template <typename T>
static __device__ __forceinline__ const T * fab_row(const T * base, const uint64_t offs) {
    return (const T *) ((const char *) base + offs);
}

// row of o and grad_o for q row (i1, i2, i3), heads are on dim 1 there
static __device__ __forceinline__ int64_t fab_o_row(
        const ggml_cuda_flash_attn_ext_back_kargs & args, const int64_t i1, const int64_t i2, const int64_t i3) {
    return (i2 + i1*args.n_head + i3*args.n_head*args.n_q)*args.D;
}

template <typename T_KV>
static __global__ void flash_attn_ext_back_stats(
        const float * q_ptr,
        const T_KV  * k_ptr,
        const half  * mask_ptr,
        const float * o_ptr,
        const float * grad_o_ptr,
              float * dst_ptr,
        const ggml_cuda_flash_attn_ext_back_kargs args) {
    const float * GGML_CUDA_RESTRICT q      = q_ptr;
    const T_KV  * GGML_CUDA_RESTRICT k      = k_ptr;
    const half  * GGML_CUDA_RESTRICT mask   = mask_ptr;
    const float * GGML_CUDA_RESTRICT o      = o_ptr;
    const float * GGML_CUDA_RESTRICT grad_o = grad_o_ptr;
          float * GGML_CUDA_RESTRICT dst    = dst_ptr;
    const int64_t ir = (int64_t) blockIdx.x*FAB_WARPS + threadIdx.x/WARP_SIZE;
    if (ir >= args.n_q*args.n_head*args.ne3) {
        return;
    }
    const int64_t i1  = ir % args.n_q;
    const int64_t i2  = (ir/args.n_q) % args.n_head;
    const int64_t i3  = ir/(args.n_q*args.n_head);
    const int64_t ik2 = i2/(args.n_head/args.n_head_kv);

    float x[FAB_PER_LANE];
    float y[FAB_PER_LANE];
    fab_load_row(fab_row(q, i1*args.nbq1 + i2*args.nbq2 + i3*args.nbq3), args.D, x);

    float S = 0.0f;
    float M = -INFINITY;
    for (int64_t ic = 0; ic < args.n_kv; ++ic) {
        const float mv = fab_mask(mask, args, i1, i2, i3, ic);
        if (mv == -INFINITY) {
            continue;
        }
        fab_load_row(fab_row(k, ic*args.nbk1 + ik2*args.nbk2 + i3*args.nbk3), args.D, y);
        const float s = warp_reduce_sum(fab_dot(x, y))*args.scale + mv;
        if (s > M) {
            S = S*expf(M - s) + 1.0f;
            M = s;
        } else {
            S += expf(s - M);
        }
    }

    const int64_t io = fab_o_row(args, i1, i2, i3);
    fab_load_row(o      + io, args.D, x);
    fab_load_row(grad_o + io, args.D, y);
    const float d = warp_reduce_sum(fab_dot(x, y));

    if (threadIdx.x % WARP_SIZE == 0) {
        dst[2*ir + 0] = S == 0.0f ? -INFINITY : M + logf(S);
        dst[2*ir + 1] = d;
    }
}

template <typename T_KV>
static __global__ void flash_attn_ext_back_dq(
        const float * q_ptr,
        const T_KV  * k_ptr,
        const T_KV  * v_ptr,
        const half  * mask_ptr,
        const float * grad_o_ptr,
        const float * stats_ptr,
              float * dst_ptr,
        const ggml_cuda_flash_attn_ext_back_kargs args) {
    const float * GGML_CUDA_RESTRICT q      = q_ptr;
    const T_KV  * GGML_CUDA_RESTRICT k      = k_ptr;
    const T_KV  * GGML_CUDA_RESTRICT v      = v_ptr;
    const half  * GGML_CUDA_RESTRICT mask   = mask_ptr;
    const float * GGML_CUDA_RESTRICT grad_o = grad_o_ptr;
    const float * GGML_CUDA_RESTRICT stats  = stats_ptr;
          float * GGML_CUDA_RESTRICT dst    = dst_ptr;
    const int64_t ir = (int64_t) blockIdx.x*FAB_WARPS + threadIdx.x/WARP_SIZE;
    if (ir >= args.n_q*args.n_head*args.ne3) {
        return;
    }
    const int64_t i1  = ir % args.n_q;
    const int64_t i2  = (ir/args.n_q) % args.n_head;
    const int64_t i3  = ir/(args.n_q*args.n_head);
    const int64_t ik2 = i2/(args.n_head/args.n_head_kv);

    float x[FAB_PER_LANE];
    float g[FAB_PER_LANE];
    float y[FAB_PER_LANE];
    float z[FAB_PER_LANE];
    float acc[FAB_PER_LANE] = {0.0f};
    fab_load_row(fab_row(q, i1*args.nbq1 + i2*args.nbq2 + i3*args.nbq3), args.D, x);
    fab_load_row(grad_o + fab_o_row(args, i1, i2, i3), args.D, g);

    const float L  = stats[2*ir + 0];
    const float Do = stats[2*ir + 1];

    for (int64_t ic = 0; ic < args.n_kv; ++ic) {
        const float mv = fab_mask(mask, args, i1, i2, i3, ic);
        if (mv == -INFINITY) {
            continue;
        }
        fab_load_row(fab_row(k, ic*args.nbk1 + ik2*args.nbk2 + i3*args.nbk3), args.D, y);
        fab_load_row(fab_row(v, ic*args.nbv1 + ik2*args.nbv2 + i3*args.nbv3), args.D, z);
        const float2 sd = warp_reduce_sum(make_float2(fab_dot(x, y), fab_dot(g, z)));
        const float p  = expf(sd.x*args.scale + mv - L);
        const float ds = p*(sd.y - Do);
#pragma unroll
        for (int i = 0; i < FAB_PER_LANE; ++i) {
            acc[i] += args.scale*ds*y[i];
        }
    }

    fab_store_row(dst + ir*args.D, args.D, acc);
}

template <typename T_KV>
static __global__ void flash_attn_ext_back_dkv(
        const float * q_ptr,
        const T_KV  * k_ptr,
        const T_KV  * v_ptr,
        const half  * mask_ptr,
        const float * grad_o_ptr,
        const float * stats_ptr,
              float * dst_ptr,
        const ggml_cuda_flash_attn_ext_back_kargs args) {
    const float * GGML_CUDA_RESTRICT q      = q_ptr;
    const T_KV  * GGML_CUDA_RESTRICT k      = k_ptr;
    const T_KV  * GGML_CUDA_RESTRICT v      = v_ptr;
    const half  * GGML_CUDA_RESTRICT mask   = mask_ptr;
    const float * GGML_CUDA_RESTRICT grad_o = grad_o_ptr;
    const float * GGML_CUDA_RESTRICT stats  = stats_ptr;
          float * GGML_CUDA_RESTRICT dst    = dst_ptr;
    const int64_t ir     = (int64_t) blockIdx.x*FAB_WARPS + threadIdx.x/WARP_SIZE;
    const int64_t n_rows = args.n_kv*args.n_head_kv*args.ne3;
    if (ir >= n_rows) {
        return;
    }
    const int64_t ik1 = ir % args.n_kv;
    const int64_t ik2 = (ir/args.n_kv) % args.n_head_kv;
    const int64_t ik3 = ir/(args.n_kv*args.n_head_kv);
    const int64_t rk2 = args.n_head/args.n_head_kv;

    float x[FAB_PER_LANE];
    float g[FAB_PER_LANE];
    float y[FAB_PER_LANE];
    float z[FAB_PER_LANE];
    float dk[FAB_PER_LANE] = {0.0f};
    float dv[FAB_PER_LANE] = {0.0f};
    fab_load_row(fab_row(k, ik1*args.nbk1 + ik2*args.nbk2 + ik3*args.nbk3), args.D, y);
    fab_load_row(fab_row(v, ik1*args.nbv1 + ik2*args.nbv2 + ik3*args.nbv3), args.D, z);

    // all q heads of this kv head (GQA reduce)
    for (int64_t iq2 = ik2*rk2; iq2 < (ik2 + 1)*rk2; ++iq2) {
        for (int64_t iq1 = 0; iq1 < args.n_q; ++iq1) {
            const float mv = fab_mask(mask, args, iq1, iq2, ik3, ik1);
            if (mv == -INFINITY) {
                continue;
            }
            fab_load_row(fab_row(q, iq1*args.nbq1 + iq2*args.nbq2 + ik3*args.nbq3), args.D, x);
            fab_load_row(grad_o + fab_o_row(args, iq1, iq2, ik3), args.D, g);
            const float * st = stats + 2*(iq1 + iq2*args.n_q + ik3*args.n_q*args.n_head);

            const float2 sd = warp_reduce_sum(make_float2(fab_dot(x, y), fab_dot(g, z)));
            const float p  = expf(sd.x*args.scale + mv - st[0]);
            const float ds = p*(sd.y - st[1]);
#pragma unroll
            for (int i = 0; i < FAB_PER_LANE; ++i) {
                dv[i] += p*g[i];
                dk[i] += args.scale*ds*x[i];
            }
        }
    }

    fab_store_row(dst +  ir          *args.D, args.D, dk);
    fab_store_row(dst + (ir + n_rows)*args.D, args.D, dv);
}

template <typename T_KV>
static void launch_flash_attn_ext_back(const ggml_tensor * dst, const ggml_cuda_flash_attn_ext_back_kargs & args, cudaStream_t stream) {
    const float * q      = (const float *) dst->src[0]->data;
    const T_KV  * k      = (const T_KV  *) dst->src[1]->data;
    const T_KV  * v      = (const T_KV  *) dst->src[2]->data;
    const half  * mask   = dst->src[3] ? (const half *) dst->src[3]->data : nullptr;
    const float * o      = (const float *) dst->src[4]->data;
    const float * grad_o = (const float *) dst->src[5]->data;
    const float * stats  = dst->src[6] ? (const float *) dst->src[6]->data : nullptr;
    float       * dst_d  = (float *) dst->data;

    const int64_t n_rows = args.target == GGML_FLASH_ATTN_EXT_BACK_DKV ?
        args.n_kv*args.n_head_kv*args.ne3 : args.n_q*args.n_head*args.ne3;
    const ggml_cuda_kernel_launch_params launch_params = {
        dim3((unsigned) ((n_rows + FAB_WARPS - 1)/FAB_WARPS)), dim3(FAB_WARPS*WARP_SIZE), 0, stream};

    switch (args.target) {
        case GGML_FLASH_ATTN_EXT_BACK_STATS:
            ggml_cuda_kernel_launch(flash_attn_ext_back_stats<T_KV>, launch_params, q, k, mask, o, grad_o, dst_d, args);
            break;
        case GGML_FLASH_ATTN_EXT_BACK_DQ:
            ggml_cuda_kernel_launch(flash_attn_ext_back_dq<T_KV>, launch_params, q, k, v, mask, grad_o, stats, dst_d, args);
            break;
        case GGML_FLASH_ATTN_EXT_BACK_DKV:
            ggml_cuda_kernel_launch(flash_attn_ext_back_dkv<T_KV>, launch_params, q, k, v, mask, grad_o, stats, dst_d, args);
            break;
        default:
            GGML_ABORT("flash_attn_ext_back: invalid target");
    }
}

void ggml_cuda_flash_attn_ext_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_flash_attn_ext_back_supported(ctx.device, dst));

    const ggml_tensor * q    = dst->src[0];
    const ggml_tensor * k    = dst->src[1];
    const ggml_tensor * v    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    ggml_cuda_flash_attn_ext_back_kargs args = {};
    args.target = ggml_get_op_params_i32(dst, 5);
    args.scale  = ggml_get_op_params_f32(dst, 0);

    args.D         = q->ne[0];
    args.n_q       = q->ne[1];
    args.n_kv      = k->ne[1];
    args.n_head    = q->ne[2];
    args.n_head_kv = k->ne[2];
    args.ne3       = q->ne[3];

    args.nbq1 = q->nb[1]; args.nbq2 = q->nb[2]; args.nbq3 = q->nb[3];
    args.nbk1 = k->nb[1]; args.nbk2 = k->nb[2]; args.nbk3 = k->nb[3];
    args.nbv1 = v->nb[1]; args.nbv2 = v->nb[2]; args.nbv3 = v->nb[3];

    args.nem2 = 1; args.nem3 = 1;
    if (mask) {
        args.nem2 = mask->ne[2]; args.nem3 = mask->ne[3];
        args.nbm1 = mask->nb[1]; args.nbm2 = mask->nb[2]; args.nbm3 = mask->nb[3];
    }

    if (ggml_cuda_flash_attn_ext_back_mma(ctx, dst, args)) {
        return;
    }
    if (k->type == GGML_TYPE_F16) {
        launch_flash_attn_ext_back<half>(dst, args, ctx.stream());
    } else {
        launch_flash_attn_ext_back<float>(dst, args, ctx.stream());
    }
}

bool ggml_cuda_flash_attn_ext_back_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);

    const ggml_tensor * q      = dst->src[0];
    const ggml_tensor * k      = dst->src[1];
    const ggml_tensor * v      = dst->src[2];
    const ggml_tensor * mask   = dst->src[3];
    const ggml_tensor * o      = dst->src[4];
    const ggml_tensor * grad_o = dst->src[5];
    const ggml_tensor * stats  = dst->src[6];

    // same types as the CPU kernel, K and V of one type
    if (q->type != GGML_TYPE_F32 || (k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_F16) || v->type != k->type) {
        return false;
    }
    if (mask && (mask->type != GGML_TYPE_F16 || !ggml_is_contiguous_rows(mask))) {
        return false;
    }
    if (q->ne[0] > FAB_D_MAX) {
        return false;
    }

    return ggml_is_contiguous_rows(q) && ggml_is_contiguous_rows(k) && ggml_is_contiguous_rows(v) &&
           ggml_is_contiguous(o) && ggml_is_contiguous(grad_o) && (!stats || ggml_is_contiguous(stats));
}
