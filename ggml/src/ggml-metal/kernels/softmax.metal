#include "common.h"

template<typename T>
kernel void kernel_soft_max(
        constant ggml_metal_kargs_soft_max & args,
        device const  char * src0,
        device const  char * src1,
        device const  char * src2,
        device        char * dst,
        threadgroup  float * buf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint3  tptg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i01 = tgpig.x;

    const int32_t i13 = i03%args.ne13;
    const int32_t i12 = i02%args.ne12;
    const int32_t i11 = i01;

    device const float * psrc0 =                (device const float *) (src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);
    device const     T * pmask = src1 != src0 ? (device const T *    ) (src1 + i11*args.nb11 + i12*args.nb12 + i13*args.nb13) : nullptr;
    device const float * psrc2 = src2 != src0 ? (device const float *) (src2)                                                 : nullptr;
    device       float * pdst  =                (device       float *) (dst  + i01*args.nb1  + i02*args.nb2  + i03*args.nb3);

    float slope = 1.0f;

    // ALiBi
    if (args.max_bias > 0.0f) {
        const int32_t h = i02;

        const float base = h < args.n_head_log2 ? args.m0 : args.m1;
        const int   exp  = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float lmax = psrc2 ? psrc2[i02] : -INFINITY;

    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {
        lmax = MAX(lmax, psrc0[i00]*args.scale + (pmask ? slope*pmask[i00] : 0.0f));
    }

    // find the max value in the block
    float max_val = simd_max(lmax);
    if (tptg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    // parallel sum
    float lsum = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {
        const float exp_psrc0 = exp((psrc0[i00]*args.scale + (pmask ? slope*pmask[i00] : 0.0f)) - max_val);
        lsum += exp_psrc0;
        pdst[i00] = exp_psrc0;
    }

    // This barrier fixes a failing test
    // ref: https://github.com/ggml-org/ggml/pull/621#discussion_r1425156335
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (tptg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    if (psrc2) {
        sum += exp(psrc2[i02] - max_val);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {
        pdst[i00] *= inv_sum;
    }
}

template<typename T>
kernel void kernel_soft_max_4(
        constant ggml_metal_kargs_soft_max & args,
        device const  char * src0,
        device const  char * src1,
        device const  char * src2,
        device        char * dst,
        threadgroup  float * buf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint3  tptg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i01 = tgpig.x;

    const int32_t i13 = i03%args.ne13;
    const int32_t i12 = i02%args.ne12;
    const int32_t i11 = i01;

    device const float4 * psrc4 =                (device const float4 *) (src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);
    device const      T * pmask = src1 != src0 ? (device const T *     ) (src1 + i11*args.nb11 + i12*args.nb12 + i13*args.nb13) : nullptr;
    device const float *  psrc2 = src2 != src0 ? (device const float * ) (src2)                                                 : nullptr;
    device       float4 * pdst4 =                (device       float4 *) (dst  + i01*args.nb1  + i02*args.nb2  + i03*args.nb3);

    float slope = 1.0f;

    if (args.max_bias > 0.0f) {
        const int32_t h = i02;

        const float base = h < args.n_head_log2 ? args.m0 : args.m1;
        const int   exp  = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float4 lmax4 = psrc2 ? psrc2[i02] : -INFINITY;

    for (int i00 = tpitg.x; i00 < args.ne00/4; i00 += tptg.x) {
        lmax4 = fmax(lmax4, psrc4[i00]*args.scale + (float4)((pmask ? slope*pmask[i00] : 0.0f)));
    }

    const float lmax = MAX(MAX(lmax4[0], lmax4[1]), MAX(lmax4[2], lmax4[3]));

    float max_val = simd_max(lmax);
    if (tptg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    // parallel sum
    float4 lsum4 = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00/4; i00 += tptg.x) {
        const float4 exp_psrc4 = exp((psrc4[i00]*args.scale + (float4)((pmask ? slope*pmask[i00] : 0.0f))) - max_val);
        lsum4 += exp_psrc4;
        pdst4[i00] = exp_psrc4;
    }

    const float lsum = lsum4[0] + lsum4[1] + lsum4[2] + lsum4[3];

    // This barrier fixes a failing test
    // ref: https://github.com/ggml-org/ggml/pull/621#discussion_r1425156335
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (tptg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    if (psrc2) {
        sum += exp(psrc2[i02] - max_val);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg.x; i00 < args.ne00/4; i00 += tptg.x) {
        pdst4[i00] *= inv_sum;
    }
}

typedef decltype(kernel_soft_max<float>)    kernel_soft_max_t;
typedef decltype(kernel_soft_max_4<float4>) kernel_soft_max_4_t;

template [[host_name("kernel_soft_max_f16")]]   kernel kernel_soft_max_t   kernel_soft_max<half>;
template [[host_name("kernel_soft_max_f32")]]   kernel kernel_soft_max_t   kernel_soft_max<float>;
template [[host_name("kernel_soft_max_f16_4")]] kernel kernel_soft_max_4_t kernel_soft_max_4<half4>;
template [[host_name("kernel_soft_max_f32_4")]] kernel kernel_soft_max_4_t kernel_soft_max_4<float4>;

[[host_name("kernel_soft_max_back")]]
kernel void kernel_soft_max_back(
        constant ggml_metal_kargs_soft_max_back & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const int i01 = tgpig.x;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;

    device const float * dy = (device const float *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device const float * y  = (device const float *) (src1 + i03*args.nb13 + i02*args.nb12 + i01*args.nb11);
    device       float * dx = (device       float *) (dst  + i03*args.nb3  + i02*args.nb2  + i01*args.nb1);

    float sum = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        sum += dy[i00] * y[i00];
    }

    sum = simd_sum(sum);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sum = shmem_f32[tiisg];
    sum = simd_sum(sum);

    const float scale = args.scale;

    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        dx[i00] = (dy[i00] - sum) * y[i00] * scale;
    }
}

[[host_name("kernel_soft_max_back_4")]]
kernel void kernel_soft_max_back_4(
        constant ggml_metal_kargs_soft_max_back & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const int i01 = tgpig.x;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;

    device const float4 * dy = (device const float4 *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device const float4 * y  = (device const float4 *) (src1 + i03*args.nb13 + i02*args.nb12 + i01*args.nb11);
    device       float4 * dx = (device       float4 *) (dst  + i03*args.nb3  + i02*args.nb2  + i01*args.nb1);

    float sum = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00_4; i00 += ntg.x) {
        sum += dot(dy[i00], y[i00]);
    }

    sum = simd_sum(sum);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sum = shmem_f32[tiisg];
    sum = simd_sum(sum);

    const float scale = args.scale;
    const float4 sum4 = float4(sum);

    for (int i00 = tpitg.x; i00 < args.ne00_4; i00 += ntg.x) {
        const float4 dy4 = dy[i00];
        const float4 y4  = y[i00];
        dx[i00] = (dy4 - sum4) * y4 * scale;
    }
}

constant int DM_MAX_STATES  = GGML_METAL_DEPTH_MIX_MAX_STATES;
constant int DM_MAX_QUERIES = GGML_METAL_DEPTH_MIX_MAX_QUERIES;
constant int DM_MAX_SG      = GGML_METAL_DEPTH_MIX_MAX_SIMDGROUPS;

// the token's float4 channel c of the stack states (0 past n_stack), of the last state and of the queries
template <int LAST_MODE>
static void depth_mix_load(
        constant ggml_metal_kargs_depth_mix & args,
        device const char * s_tok, device const char * l_tok, device const char * query, int c,
        thread float4 * s, thread float4 & x, thread float4 * w) {
    FOR_UNROLL (int n = 0; n < DM_MAX_STATES; ++n) {
        s[n] = n < args.n_stack ? *((device const float4 *) (s_tok + n*args.nbs1) + c) : float4(0.0f);
    }
    x = LAST_MODE == GGML_METAL_DEPTH_MIX_LAST_NONE ? float4(0.0f) : *((device const float4 *) l_tok + c);
    FOR_UNROLL (int q = 0; q < DM_MAX_QUERIES; ++q) {
        w[q] = q < args.n_queries ? *((device const float4 *) (query + q*args.nbq1) + c) : float4(0.0f);
    }
}

// the simdgroup sum of v in slot (q, n) of the threadgroup partials, one slot per simdgroup
static void depth_mix_store_partial(threadgroup float * shmem, int slot, float v, ushort sgitg, ushort tiisg) {
    const float r = simd_sum(v);
    if (tiisg == 0) {
        shmem[slot*DM_MAX_SG + sgitg] = r;
    }
}

// the sum over the simdgroups of slot, in every lane
static float depth_mix_total(threadgroup const float * shmem, int slot, int nsg, ushort tiisg) {
    return simd_sum(tiisg < nsg ? shmem[slot*DM_MAX_SG + tiisg] : 0.0f);
}

// as kernel_soft_max: exp(x - max), times 1/sum; the last state's weight also takes its norm scale
template <int LAST_MODE>
static void depth_mix_weights(
        constant ggml_metal_kargs_depth_mix & args, threadgroup const float * shmem, int q, int nsg, float norm_scale, ushort tiisg,
        thread float * p, thread float & p_last) {
    float x[DM_MAX_STATES];
    float m = -INFINITY;
    FOR_UNROLL (int n = 0; n < DM_MAX_STATES; ++n) {
        x[n] = n < args.n_stack ? args.scale*depth_mix_total(shmem, q*DM_MAX_STATES + n, nsg, tiisg) : -INFINITY;
        m = MAX(m, x[n]);
    }
    float xl = -INFINITY;
    if (LAST_MODE != GGML_METAL_DEPTH_MIX_LAST_NONE) {
        xl = args.scale*norm_scale*depth_mix_total(shmem, q*DM_MAX_STATES + args.n_stack, nsg, tiisg);
        m = MAX(m, xl);
    }
    float sum = 0.0f;
    FOR_UNROLL (int n = 0; n < DM_MAX_STATES; ++n) {
        x[n] = n < args.n_stack ? exp(x[n] - m) : 0.0f;
        sum += x[n];
    }
    const float el = LAST_MODE != GGML_METAL_DEPTH_MIX_LAST_NONE ? exp(xl - m) : 0.0f;
    sum += el;
    const float inv_sum = 1.0f/sum;
    FOR_UNROLL (int n = 0; n < DM_MAX_STATES; ++n) {
        p[n] = x[n]*inv_sum;
    }
    p_last = el*inv_sum*norm_scale;
}

// Inkling 2's depth mix, one threadgroup per token: per query, softmax over the states of scale * dot(state, query),
// then the weighted sum of the states. the states are a stack and a last state, absent, read, or RMS-normalized here
template <int LAST_MODE>
kernel void kernel_depth_mix(
        constant ggml_metal_kargs_depth_mix & args,
        device const char * stack,
        device const char * last,
        device const char * query,
        device       char * dst,
        threadgroup float * shmem [[threadgroup(0)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort tpitg [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort ntg   [[threads_per_threadgroup]]) {
    device const char * s_tok = stack + tgpig*args.nbs2;
    device const char * l_tok = last  + tgpig*args.nbl2;
    device       char * d_tok = dst   + tgpig*args.nbd2;

    float acc[DM_MAX_QUERIES][DM_MAX_STATES] = {};
    float acc_last[DM_MAX_QUERIES] = {};
    float sq = 0.0f;

    float4 s[DM_MAX_STATES];
    float4 x;
    float4 w[DM_MAX_QUERIES];
    for (int c = tpitg; c < args.n_embd4; c += ntg) {
        depth_mix_load<LAST_MODE>(args, s_tok, l_tok, query, c, s, x, w);
        FOR_UNROLL (int q = 0; q < DM_MAX_QUERIES; ++q) {
            FOR_UNROLL (int n = 0; n < DM_MAX_STATES; ++n) {
                acc[q][n] += dot(s[n], w[q]);
            }
            acc_last[q] += dot(x, w[q]);
        }
        sq += dot(x, x);
    }

    FOR_UNROLL (int q = 0; q < DM_MAX_QUERIES; ++q) {
        if (q < args.n_queries) {
            FOR_UNROLL (int n = 0; n < DM_MAX_STATES; ++n) {
                if (n < args.n_stack) {
                    depth_mix_store_partial(shmem, q*DM_MAX_STATES + n, acc[q][n], sgitg, tiisg);
                }
            }
            if (LAST_MODE != GGML_METAL_DEPTH_MIX_LAST_NONE) {
                depth_mix_store_partial(shmem, q*DM_MAX_STATES + args.n_stack, acc_last[q], sgitg, tiisg);
            }
        }
    }
    if (LAST_MODE == GGML_METAL_DEPTH_MIX_LAST_NORM) {
        depth_mix_store_partial(shmem, DM_MAX_QUERIES*DM_MAX_STATES, sq, sgitg, tiisg);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int nsg = (ntg + N_SIMDWIDTH - 1)/N_SIMDWIDTH;
    float norm_scale = 1.0f;
    if (LAST_MODE == GGML_METAL_DEPTH_MIX_LAST_NORM) {
        const float mean = depth_mix_total(shmem, DM_MAX_QUERIES*DM_MAX_STATES, nsg, tiisg)/(4*args.n_embd4);
        norm_scale = rsqrt(mean + args.eps);
    }

    float p[DM_MAX_QUERIES][DM_MAX_STATES];
    float p_last[DM_MAX_QUERIES];
    FOR_UNROLL (int q = 0; q < DM_MAX_QUERIES; ++q) {
        if (q < args.n_queries) {
            depth_mix_weights<LAST_MODE>(args, shmem, q, nsg, norm_scale, tiisg, p[q], p_last[q]);
        }
    }

    for (int c = tpitg; c < args.n_embd4; c += ntg) {
        depth_mix_load<LAST_MODE>(args, s_tok, l_tok, query, c, s, x, w);
        FOR_UNROLL (int q = 0; q < DM_MAX_QUERIES; ++q) {
            if (q < args.n_queries) {
                float4 o = p_last[q]*x;
                FOR_UNROLL (int n = 0; n < DM_MAX_STATES; ++n) {
                    o += p[q][n]*s[n];
                }
                *((device float4 *) (d_tok + q*args.nbd1) + c) = o;
            }
        }
    }
}

typedef decltype(kernel_depth_mix<GGML_METAL_DEPTH_MIX_LAST_NONE>) kernel_depth_mix_t;

template [[host_name("kernel_depth_mix_none")]] kernel kernel_depth_mix_t kernel_depth_mix<GGML_METAL_DEPTH_MIX_LAST_NONE>;
template [[host_name("kernel_depth_mix_read")]] kernel kernel_depth_mix_t kernel_depth_mix<GGML_METAL_DEPTH_MIX_LAST_READ>;
template [[host_name("kernel_depth_mix_norm")]] kernel kernel_depth_mix_t kernel_depth_mix<GGML_METAL_DEPTH_MIX_LAST_NORM>;
