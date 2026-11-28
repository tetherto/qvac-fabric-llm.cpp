#include "common.h"

// o and grad_o are [D, n_head, n_q, ne3], stats [2, n_q, n_head, ne3], contiguous like dst:
//   STATS: dst [2, n_q, n_head, ne3], (logsumexp, dot(grad_o, o)) per q row
//   DQ:    dst [D, n_q, n_head, ne3]
//   DKV:   dst [D, n_kv, n_head_kv, 2*ne3], dk in ne3 [0, ne3), dv in [ne3, 2*ne3)

template <typename T, uint FAB_PER_LANE>
inline void fab_load_row(device const T * row,
                         const uint D,
                         const uint lane,
                         thread float (&x)[FAB_PER_LANE]) {
    FOR_UNROLL (uint i = 0; i < FAB_PER_LANE; ++i) {
        const uint d = lane + i*N_SIMDWIDTH;
        x[i] = d < D ? float(row[d]) : 0.0f;
    }
}

template <uint FAB_PER_LANE>
inline void fab_store_row(device float * row,
                          const uint D,
                          const uint lane,
                          thread float (&x)[FAB_PER_LANE]) {
    FOR_UNROLL (uint i = 0; i < FAB_PER_LANE; ++i) {
        const uint d = lane + i*N_SIMDWIDTH;
        if (d < D) {
            row[d] = x[i];
        }
    }
}

template <uint FAB_PER_LANE>
inline float fab_dot(thread float (&x)[FAB_PER_LANE],
                     thread float (&y)[FAB_PER_LANE]) {
    float s = 0.0f;
    FOR_UNROLL (uint i = 0; i < FAB_PER_LANE; ++i) {
        s = fma(x[i], y[i], s);
    }
    return s;
}

// mask value of q row (i1, i2, i3) at kv position ic, the same for every lane of the warp
// mask is nullptr when absent, the host binds q in its slot and the kernels check for that
inline float fab_mask(constant ggml_metal_kargs_ext_back_kargs & args,
                      device const half * mask,
                      const uint64_t i1, const uint64_t i2,
                      const uint64_t i3, const uint64_t ic) {
    if (!mask) {
        return 0.0f;
    }
    device const char * row = (device const char *) mask + i1*args.nbm1 + (i2 % (uint64_t) args.nem2)*args.nbm2 + (i3 % (uint64_t) args.nem3)*args.nbm3;
    return float(((device const half *) row)[ic]);
}

template <typename T>
inline device const T * fab_row(device const T * base, const uint64_t offs) {
    return (device const T *) ((device const char *) base + offs);
}

// row of o and grad_o for q row (i1, i2, i3), heads are on dim 1 there
inline uint64_t fab_o_row(constant ggml_metal_kargs_ext_back_kargs & args,
                          const uint64_t i1, const uint64_t i2, const uint64_t i3) {
    const uint64_t n_q    = (uint64_t) args.n_q;
    const uint64_t n_head = (uint64_t) args.n_head;
    return (i2 + i1*n_head + i3*n_head*n_q)*(uint64_t) args.D;
}

template <typename T_KV, uint D_MAX, uint NSG>
kernel void flash_attn_ext_back_stats(
        constant ggml_metal_kargs_ext_back_kargs & args [[buffer(0)]],
        device const float * q      [[buffer(1)]],
        device const T_KV  * k      [[buffer(2)]],
        device const half  * mask   [[buffer(5)]],
        device const float * o      [[buffer(6)]],
        device const float * grad_o [[buffer(3)]],
              device float * dst    [[buffer(4)]],
                     uint    tgpig  [[threadgroup_position_in_grid]],
                     ushort  sgitg  [[simdgroup_index_in_threadgroup]],
                     ushort  tiisg  [[thread_index_in_simdgroup]]) {

    constexpr uint FAB_PER_LANE = D_MAX / N_SIMDWIDTH;
    device const half * pmask = (device const char *) mask != (device const char *) q ? mask : nullptr;

    const uint     D         = (uint)     args.D;
    const uint64_t n_q       = (uint64_t) args.n_q;
    const uint64_t n_kv      = (uint64_t) args.n_kv;
    const uint64_t n_head    = (uint64_t) args.n_head;
    const uint64_t n_head_kv = (uint64_t) args.n_head_kv;
    const uint64_t ne3       = (uint64_t) args.ne3;

    const uint64_t ir = (uint64_t) tgpig*NSG + sgitg;
    if (ir >= n_q*n_head*ne3) {
        return;
    }

    const uint64_t i1  = ir % n_q;
    const uint64_t i2  = (ir/n_q) % n_head;
    const uint64_t i3  = ir/(n_q*n_head);
    const uint64_t ik2 = i2/(n_head/n_head_kv);

    float x[FAB_PER_LANE];
    float y[FAB_PER_LANE];
    fab_load_row(fab_row(q, i1*args.nbq1 + i2*args.nbq2 + i3*args.nbq3), D, tiisg, x);

    float S = 0.0f;
    float M = -FLT_MAX/2;
    for (uint64_t ic = 0; ic < n_kv; ++ic) {
        const float mv = fab_mask(args, pmask, i1, i2, i3, ic);
        if (mv <= -MAXHALF) {
            continue;
        }
        fab_load_row(fab_row(k, ic*args.nbk1 + ik2*args.nbk2 + i3*args.nbk3), D, tiisg, y);
        const float s = simd_sum(fab_dot(x, y))*args.scale + mv;
        if (s > M) {
            S = S*exp(M - s) + 1.0f;
            M = s;
        } else {
            S += exp(s - M);
        }
    }

    const uint64_t io = fab_o_row(args, i1, i2, i3);
    fab_load_row(o      + io, D, tiisg, x);
    fab_load_row(grad_o + io, D, tiisg, y);
    const float d = simd_sum(fab_dot(x, y));

    if (tiisg == 0) {
        dst[2*ir + 0] = S == 0.0f ? -INFINITY : M + log(S);
        dst[2*ir + 1] = d;
    }
}

template <typename T_KV, uint D_MAX, uint NSG>
kernel void flash_attn_ext_back_dq(
        constant ggml_metal_kargs_ext_back_kargs & args [[buffer(0)]],
        device const float * q      [[buffer(1)]],
        device const T_KV  * k      [[buffer(2)]],
        device const T_KV  * v      [[buffer(7)]],
        device const half  * mask   [[buffer(5)]],
        device const float * grad_o [[buffer(3)]],
        device const float * stats  [[buffer(8)]],
              device float * dst    [[buffer(4)]],
                     uint    tgpig  [[threadgroup_position_in_grid]],
                     ushort  sgitg  [[simdgroup_index_in_threadgroup]],
                     ushort  tiisg  [[thread_index_in_simdgroup]]) {

    constexpr uint FAB_PER_LANE = D_MAX / N_SIMDWIDTH;
    device const half * pmask = (device const char *) mask != (device const char *) q ? mask : nullptr;

    const uint     D         = (uint)     args.D;
    const uint64_t n_q       = (uint64_t) args.n_q;
    const uint64_t n_kv      = (uint64_t) args.n_kv;
    const uint64_t n_head    = (uint64_t) args.n_head;
    const uint64_t n_head_kv = (uint64_t) args.n_head_kv;
    const uint64_t ne3       = (uint64_t) args.ne3;

    const uint64_t ir = (uint64_t) tgpig*NSG + sgitg;
    if (ir >= n_q*n_head*ne3) {
        return;
    }
    const uint64_t i1  = ir % n_q;
    const uint64_t i2  = (ir/n_q) % n_head;
    const uint64_t i3  = ir/(n_q*n_head);
    const uint64_t ik2 = i2/(n_head/n_head_kv);

    float x[FAB_PER_LANE];
    float g[FAB_PER_LANE];
    float y[FAB_PER_LANE];
    float z[FAB_PER_LANE];
    float acc[FAB_PER_LANE] = {0.0f};
    fab_load_row(fab_row(q, i1*args.nbq1 + i2*args.nbq2 + i3*args.nbq3), D, tiisg, x);
    fab_load_row(grad_o + fab_o_row(args, i1, i2, i3), D, tiisg, g);

    const float L  = stats[2*ir + 0];
    const float Do = stats[2*ir + 1];

    for (uint64_t ic = 0; ic < n_kv; ++ic) {
        const float mv = fab_mask(args, pmask, i1, i2, i3, ic);
        if (mv <= -MAXHALF) {
            continue;
        }
        fab_load_row(fab_row(k, ic*args.nbk1 + ik2*args.nbk2 + i3*args.nbk3), D, tiisg, y);
        fab_load_row(fab_row(v, ic*args.nbv1 + ik2*args.nbv2 + i3*args.nbv3), D, tiisg, z);
        const float2 sd = simd_sum(float2(fab_dot(x, y), fab_dot(g, z)));
        const float p  = exp(sd.x*args.scale + mv - L);
        const float ds = p*(sd.y - Do);
        FOR_UNROLL (uint i = 0; i < FAB_PER_LANE; ++i) {
            acc[i] += args.scale*ds*y[i];
        }
    }

    fab_store_row(dst + ir*D, D, tiisg, acc);
}

template <typename T_KV, uint D_MAX, uint NSG>
kernel void flash_attn_ext_back_dkv(
        constant ggml_metal_kargs_ext_back_kargs & args [[buffer(0)]],
        device const float * q      [[buffer(1)]],
        device const T_KV  * k      [[buffer(2)]],
        device const T_KV  * v      [[buffer(7)]],
        device const half  * mask   [[buffer(5)]],
        device const float * grad_o [[buffer(3)]],
        device const float * stats  [[buffer(8)]],
              device float * dst    [[buffer(4)]],
                     uint    tgpig  [[threadgroup_position_in_grid]],
                     ushort  sgitg  [[simdgroup_index_in_threadgroup]],
                     ushort  tiisg  [[thread_index_in_simdgroup]]) {

    constexpr uint FAB_PER_LANE = D_MAX / N_SIMDWIDTH;
    device const half * pmask = (device const char *) mask != (device const char *) q ? mask : nullptr;

    const uint     D         = (uint)     args.D;
    const uint64_t n_q       = (uint64_t) args.n_q;
    const uint64_t n_kv      = (uint64_t) args.n_kv;
    const uint64_t n_head    = (uint64_t) args.n_head;
    const uint64_t n_head_kv = (uint64_t) args.n_head_kv;
    const uint64_t ne3       = (uint64_t) args.ne3;

    const uint64_t ir     = (uint64_t) tgpig*NSG + sgitg;
    const uint64_t n_rows = n_kv*n_head_kv*ne3;
    if (ir >= n_rows) {
        return;
    }
    const uint64_t ik1 = ir % n_kv;
    const uint64_t ik2 = (ir/n_kv) % n_head_kv;
    const uint64_t ik3 = ir/(n_kv*n_head_kv);
    const uint64_t rk2 = n_head/n_head_kv;

    float x[FAB_PER_LANE];
    float g[FAB_PER_LANE];
    float y[FAB_PER_LANE];
    float z[FAB_PER_LANE];
    float dk[FAB_PER_LANE] = {0.0f};
    float dv[FAB_PER_LANE] = {0.0f};
    fab_load_row(fab_row(k, ik1*args.nbk1 + ik2*args.nbk2 + ik3*args.nbk3), D, tiisg, y);
    fab_load_row(fab_row(v, ik1*args.nbv1 + ik2*args.nbv2 + ik3*args.nbv3), D, tiisg, z);

    // all q heads of this kv head (GQA reduce)
    for (uint64_t iq2 = ik2*rk2; iq2 < (ik2 + 1)*rk2; ++iq2) {
        for (uint64_t iq1 = 0; iq1 < n_q; ++iq1) {
            const float mv = fab_mask(args, pmask, iq1, iq2, ik3, ik1);
            if (mv <= -MAXHALF) {
                continue;
            }
            fab_load_row(fab_row(q, iq1*args.nbq1 + iq2*args.nbq2 + ik3*args.nbq3), D, tiisg, x);
            fab_load_row(grad_o + fab_o_row(args, iq1, iq2, ik3), D, tiisg, g);
            device const float * st = stats + 2*(iq1 + iq2*n_q + ik3*n_q*n_head);

            const float2 sd = simd_sum(float2(fab_dot(x, y), fab_dot(g, z)));
            const float p  = exp(sd.x*args.scale + mv - st[0]);
            const float ds = p*(sd.y - st[1]);
            FOR_UNROLL (uint i = 0; i < FAB_PER_LANE; ++i) {
                dv[i] += p*g[i];
                dk[i] += args.scale*ds*x[i];
            }
        }
    }

    fab_store_row(dst +  ir          *D, D, tiisg, dk);
    fab_store_row(dst + (ir + n_rows)*D, D, tiisg, dv);
}

#define FAB_NSG 8

typedef decltype(flash_attn_ext_back_stats<float, 64, FAB_NSG>) flash_attn_ext_back_stats_f32_t;
typedef decltype(flash_attn_ext_back_stats<half,  64, FAB_NSG>) flash_attn_ext_back_stats_f16_t;
typedef decltype(flash_attn_ext_back_dq   <float, 64, FAB_NSG>) flash_attn_ext_back_dq_f32_t;
typedef decltype(flash_attn_ext_back_dq   <half,  64, FAB_NSG>) flash_attn_ext_back_dq_f16_t;
typedef decltype(flash_attn_ext_back_dkv  <float, 64, FAB_NSG>) flash_attn_ext_back_dkv_f32_t;
typedef decltype(flash_attn_ext_back_dkv  <half,  64, FAB_NSG>) flash_attn_ext_back_dkv_f16_t;

template [[host_name("kernel_flash_attn_ext_back_stats_f32_d64")]]  kernel flash_attn_ext_back_stats_f32_t flash_attn_ext_back_stats<float,  64, FAB_NSG>;
template [[host_name("kernel_flash_attn_ext_back_stats_f32_d128")]] kernel flash_attn_ext_back_stats_f32_t flash_attn_ext_back_stats<float, 128, FAB_NSG>;
template [[host_name("kernel_flash_attn_ext_back_stats_f32_d256")]] kernel flash_attn_ext_back_stats_f32_t flash_attn_ext_back_stats<float, 256, FAB_NSG>;
template [[host_name("kernel_flash_attn_ext_back_stats_f16_d64")]]  kernel flash_attn_ext_back_stats_f16_t flash_attn_ext_back_stats<half,   64, FAB_NSG>;
template [[host_name("kernel_flash_attn_ext_back_stats_f16_d128")]] kernel flash_attn_ext_back_stats_f16_t flash_attn_ext_back_stats<half,  128, FAB_NSG>;
template [[host_name("kernel_flash_attn_ext_back_stats_f16_d256")]] kernel flash_attn_ext_back_stats_f16_t flash_attn_ext_back_stats<half,  256, FAB_NSG>;

template [[host_name("kernel_flash_attn_ext_back_dq_f32_d64")]]     kernel flash_attn_ext_back_dq_f32_t    flash_attn_ext_back_dq   <float,  64, FAB_NSG>;
template [[host_name("kernel_flash_attn_ext_back_dq_f32_d128")]]    kernel flash_attn_ext_back_dq_f32_t    flash_attn_ext_back_dq   <float, 128, FAB_NSG>;
template [[host_name("kernel_flash_attn_ext_back_dq_f32_d256")]]    kernel flash_attn_ext_back_dq_f32_t    flash_attn_ext_back_dq   <float, 256, FAB_NSG>;
template [[host_name("kernel_flash_attn_ext_back_dq_f16_d64")]]     kernel flash_attn_ext_back_dq_f16_t    flash_attn_ext_back_dq   <half,   64, FAB_NSG>;
template [[host_name("kernel_flash_attn_ext_back_dq_f16_d128")]]    kernel flash_attn_ext_back_dq_f16_t    flash_attn_ext_back_dq   <half,  128, FAB_NSG>;
template [[host_name("kernel_flash_attn_ext_back_dq_f16_d256")]]    kernel flash_attn_ext_back_dq_f16_t    flash_attn_ext_back_dq   <half,  256, FAB_NSG>;

template [[host_name("kernel_flash_attn_ext_back_dkv_f32_d64")]]    kernel flash_attn_ext_back_dkv_f32_t   flash_attn_ext_back_dkv  <float,  64, FAB_NSG>;
template [[host_name("kernel_flash_attn_ext_back_dkv_f32_d128")]]   kernel flash_attn_ext_back_dkv_f32_t   flash_attn_ext_back_dkv  <float, 128, FAB_NSG>;
template [[host_name("kernel_flash_attn_ext_back_dkv_f32_d256")]]   kernel flash_attn_ext_back_dkv_f32_t   flash_attn_ext_back_dkv  <float, 256, FAB_NSG>;
template [[host_name("kernel_flash_attn_ext_back_dkv_f16_d64")]]    kernel flash_attn_ext_back_dkv_f16_t   flash_attn_ext_back_dkv  <half,   64, FAB_NSG>;
template [[host_name("kernel_flash_attn_ext_back_dkv_f16_d128")]]   kernel flash_attn_ext_back_dkv_f16_t   flash_attn_ext_back_dkv  <half,  128, FAB_NSG>;
template [[host_name("kernel_flash_attn_ext_back_dkv_f16_d256")]]   kernel flash_attn_ext_back_dkv_f16_t   flash_attn_ext_back_dkv  <half,  256, FAB_NSG>;
