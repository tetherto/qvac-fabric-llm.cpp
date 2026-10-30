#include "common.h"

kernel void kernel_argmax_f32(
        constant ggml_metal_kargs_argmax & args,
        device   const char * src0,
        device         char * dst,
        threadgroup    char * shmem [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        uint  tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint    ntg[[threads_per_threadgroup]]) {
    device const float * x_row = (device const float *) ((device const char *) src0 + tgpig * args.nb01);

    float   lmax = -INFINITY;
    int32_t larg = -1;

    for (int i00 = tpitg; i00 < args.ne00; i00 += ntg) {
        if (x_row[i00] > lmax) {
            lmax = x_row[i00];
            larg = i00;
        }
    }

    // find the argmax value in the block
    float max_val = simd_max(lmax);
    int32_t arg_val = simd_max(select(-1, larg, lmax == max_val));

    device int32_t * dst_i32 = (device int32_t *) dst;

    threadgroup   float * shared_maxval = (threadgroup   float *) shmem;
    threadgroup int32_t * shared_argmax = (threadgroup int32_t *) shmem + N_SIMDWIDTH;

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            shared_maxval[tiisg] = -INFINITY;
            shared_argmax[tiisg] = -1;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            shared_maxval[sgitg] = max_val;
            shared_argmax[sgitg] = arg_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = shared_maxval[tiisg];
        arg_val = shared_argmax[tiisg];

        float max_val_reduced   = simd_max(max_val);
        int32_t arg_val_reduced = simd_max(select(-1, arg_val, max_val == max_val_reduced));

        dst_i32[tgpig] = arg_val_reduced;

        return;
    }

    dst_i32[tgpig] = arg_val;
}

kernel void kernel_diag_f32(
        constant ggml_metal_kargs_diag & args,
        device   const char * src0,
        device         char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]]) {
    constexpr short NW = N_SIMDWIDTH;

    const int32_t i3 = tgpig.z;
    const int32_t i2 = tgpig.y;
    const int32_t i1 = tgpig.x;

    device const float * src0_ptr = (device const float *)(src0 +                i2*args.nb02 + i3*args.nb03);
    device       float * dst_ptr  = (device       float *)(dst  + i1*args.nb01 + i2*args.nb2  + i3*args.nb3);

    for (int i0 = tiitg; i0 < args.ne0; i0 += NW) {
        dst_ptr[i0] = i0 == i1 ? src0_ptr[i0] : 0.0f;
    }
}

kernel void kernel_roll_f32(
    constant ggml_metal_kargs_roll & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    device const float * src0_ptr = (device const float *) src0;
    device       float * dst_ptr  = (device       float *) dst;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        // apply shifts and wrap around
        int64_t i00 = i0 - args.s0;
        int64_t i01 = i1 - args.s1;
        int64_t i02 = i2 - args.s2;
        int64_t i03 = i3 - args.s3;

        if (i00 < 0) { i00 += args.ne00; } else if (i00 >= args.ne00) { i00 -= args.ne00; }
        if (i01 < 0) { i01 += args.ne01; } else if (i01 >= args.ne01) { i01 -= args.ne01; }
        if (i02 < 0) { i02 += args.ne02; } else if (i02 >= args.ne02) { i02 -= args.ne02; }
        if (i03 < 0) { i03 += args.ne03; } else if (i03 >= args.ne03) { i03 -= args.ne03; }

        int64_t src_idx = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00 + i00;
        int64_t dst_idx = i3 *args.ne2 *args.ne1 *args.ne0  + i2 *args.ne1 *args.ne0  + i1 *args.ne0  + i0;

        dst_ptr[dst_idx] = src0_ptr[src_idx];
    }
}

constant bool FC_pad_circular [[function_constant(FC_PAD + 0)]];

// circular means on a torus, so the coordinates wrap around
static inline int32_t wrap_around(int32_t coord, int32_t size) {
    return (coord + size) % size;
}

kernel void kernel_pad_f32(
    constant ggml_metal_kargs_pad & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {
    const int32_t i3 = tgpig.z;
    const int32_t i2 = tgpig.y;
    const int32_t k0 = tgpig.x/args.ne1;
    const int32_t i1 = tgpig.x - k0*args.ne1;

    const int32_t ne00 = args.ne00;
    const int32_t ne01 = args.ne01;
    const int32_t ne02 = args.ne02;
    const int32_t ne03 = args.ne03;

    int32_t i01 = i1 - args.lp1;
    int32_t i02 = i2 - args.lp2;
    int32_t i03 = i3 - args.lp3;

    if (FC_pad_circular) {
        i01 = wrap_around(i01, ne01);
        i02 = wrap_around(i02, ne02);
        i03 = wrap_around(i03, ne03);
    }

    device float * dst_ptr = (device float *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1);

    // the row lies in the padded region, so no source row backs it
    if (i01 < 0 || i01 >= ne01 ||
        i02 < 0 || i02 >= ne02 ||
        i03 < 0 || i03 >= ne03) {
        for (int32_t l0 = 0; l0 < 1024; l0 += ntg.x) {
            const int32_t i0 = k0*1024 + tpitg.x + l0;
            if (i0 >= args.ne0) {
                break;
            }

            dst_ptr[i0] = 0.0f;
        }

        return;
    }

    device const char * src0_row = src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01;

    for (int32_t l0 = 0; l0 < 1024; l0 += ntg.x) {
        const int32_t i0 = k0*1024 + tpitg.x + l0;
        if (i0 >= args.ne0) {
            break;
        }

        int32_t i00 = i0 - args.lp0;

        if (FC_pad_circular) {
            i00 = wrap_around(i00, ne00);
        }

        dst_ptr[i0] = i00 >= 0 && i00 < ne00 ? *((device const float *) (src0_row + i00*args.nb00)) : 0.0f;
    }
}

// TODO: this is slow - optimize
kernel void kernel_pad_reflect_1d_f32(
    constant   ggml_metal_kargs_pad_reflect_1d & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3  tgpg[[threadgroups_per_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    const int64_t i03 = i3;
    const int64_t i02 = i2;
    const int64_t i01 = i1;

    device const float * src0_ptr = (device const float *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device       float * dst_ptr  = (device       float *) (dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1);

    if (i1 < args.ne01 && i2 < args.ne02 && i3 < args.ne03) {
        for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
            if (i0 < args.p0) {
                dst_ptr[i0] = src0_ptr[args.p0 - i0];
            } else if (i0 < args.ne0 - args.p1) {
                dst_ptr[i0] = src0_ptr[i0 - args.p0];
            } else {
                dst_ptr[i0] = src0_ptr[(args.ne0 - args.p1 - args.p0) - (args.p1 + 1 - (args.ne0 - i0)) - 1];
            }
        }
    }
}

kernel void kernel_arange_f32(
    constant   ggml_metal_kargs_arange & args,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    device float * dst_ptr = (device float *) dst;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        dst_ptr[i0] = args.start + args.step * i0;
    }
}

kernel void kernel_timestep_embedding_f32(
    constant  ggml_metal_kargs_timestep_embedding & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    int i = tgpig.x;
    device float * embed_data = (device float *)(dst + i*args.nb1);

    int half_ = args.dim / 2;
    for (int j = tpitg.x; j < half_; j += ntg.x) {
        float timestep = ((device float *)src0)[i];
        float freq = (float)exp(-log((float)args.max_period) * j / half_);
        float arg = timestep * freq;
        embed_data[j        ] = cos(arg);
        embed_data[j + half_] = sin(arg);
    }

    if (args.dim % 2 != 0 && tpitg.x == 0) {
        embed_data[2 * half_] = 0.f;
    }
}

kernel void kernel_opt_step_adamw_f32(
        constant    ggml_metal_kargs_opt_step_adamw & args,
        device       float * x,
        device const float * g,
        device       float * g_m,
        device       float * g_v,
        device const float * pars,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= args.np) {
        return;
    }

    const float alpha  = pars[0];
    const float beta1  = pars[1];
    const float beta2  = pars[2];
    const float eps    = pars[3];
    const float wd     = pars[4];
    const float beta1h = pars[5];
    const float beta2h = pars[6];

    const float gi = g[gid];
    const float gmi = g_m[gid] * beta1 +      gi * (1.0f - beta1);
    const float gvi = g_v[gid] * beta2 + gi * gi * (1.0f - beta2);

    g_m[gid] = gmi;
    g_v[gid] = gvi;

    const float mh =      gmi * beta1h;
    const float vh = sqrt(gvi * beta2h) + eps;

    x[gid] = x[gid] * (1.0f - alpha * wd) - alpha * mh / vh;
}

kernel void kernel_opt_step_sgd_f32(
        constant    ggml_metal_kargs_opt_step_sgd & args,
        device       float * x,
        device const float * g,
        device const float * pars,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= args.np) {
        return;
    }

    x[gid] = x[gid] * (1.0f - pars[0] * pars[1]) - pars[0] * g[gid];
}

template<typename T>
kernel void kernel_memset(
        constant ggml_metal_kargs_memset & args,
        device T * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = args.val;
}

typedef decltype(kernel_memset<int64_t>) kernel_memset_t;

template [[host_name("kernel_memset_i64")]] kernel kernel_memset_t kernel_memset<int64_t>;

constant short FC_count_equal_nsg [[function_constant(FC_COUNT_EQUAL + 0)]];

template<typename T>
kernel void kernel_count_equal(
        constant ggml_metal_kargs_count_equal & args,
        device   const char * src0,
        device   const char * src1,
        device   atomic_int * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const short NSG = FC_count_equal_nsg;

    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;

    if (i3 >= args.ne03 || i2 >= args.ne02 || i1 >= args.ne01) {
        return;
    }

    int sum = 0;

    device const char * base0 = src0 + i1*args.nb01 + i2*args.nb02 + i3*args.nb03;
    device const char * base1 = src1 + i1*args.nb11 + i2*args.nb12 + i3*args.nb13;

    for (int64_t i0 = tpitg.x; i0 < args.ne00; i0 += ntg.x) {
        const T v0 = *(device const T *)(base0 + i0*args.nb00);
        const T v1 = *(device const T *)(base1 + i0*args.nb10);
        sum += (v0 == v1);
    }

    sum = simd_sum(sum);

    if (tiisg == 0) {
        shmem_i32[sgitg] = sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        float v = 0.0f;
        if (tpitg.x < NSG) {
            v = shmem_i32[tpitg.x];
        }

        float total = simd_sum(v);
        if (tpitg.x == 0) {
            atomic_fetch_add_explicit(dst, (int32_t) total, memory_order_relaxed);
        }
    }
}

typedef decltype(kernel_count_equal<int32_t>) kernel_count_equal_t;

template [[host_name("kernel_count_equal_i32")]] kernel kernel_count_equal_t kernel_count_equal<int32_t>;

template <typename T>
kernel void kernel_snake(
        constant ggml_metal_kargs_snake & args,
        device const T     * x,
        device const float * a,
        device const float * inv_b,
        device       T     * dst,
        uint         tgpig [[threadgroup_position_in_grid]],
        uint         tpitg [[thread_position_in_threadgroup]],
        uint         ntg   [[threads_per_threadgroup]]) {

    const uint64_t idx = (uint64_t) tgpig * ntg + tpitg;
    if (idx >= (uint64_t) args.ne) {
        return;
    }

    const uint64_t c  = idx / (uint64_t) args.T;  // x is [T, C], a / inv_b collapse to [1, C]
    const float xi = float(x[idx]);
    const float si = sin(a[c] * xi);
    dst[idx] = T(xi + si * si * inv_b[c]);
}

template [[host_name("kernel_snake_f32")]]  kernel void kernel_snake<float>(constant ggml_metal_kargs_snake &, device const float *, device const float *, device const float *, device float *, uint, uint, uint);
template [[host_name("kernel_snake_f16")]]  kernel void kernel_snake<half>(constant ggml_metal_kargs_snake &, device const half *, device const float *, device const float *, device half *, uint, uint, uint);
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_snake_bf16")]] kernel void kernel_snake<bfloat>(constant ggml_metal_kargs_snake &, device const bfloat *, device const float *, device const float *, device bfloat *, uint, uint, uint);
#endif

template<int N, typename src_t>
kernel void kernel_fwht(
        constant ggml_metal_kargs_fwht & args,
        device const src_t * src,
        device float * dst,
        device const float * signs,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort3  ntg[[threads_per_threadgroup]]) {

    constexpr int NW = N_SIMDWIDTH;
    constexpr int NE = N / NW;

    const float scale = 1.0f / sqrt((float) N);

    const int sg_per_tg = ntg.x / NW;
    const int64_t r = tgpig.x * sg_per_tg + sgitg;
    if (r >= args.nrows) {
        return;
    }
    signs += args.n_blk > 0 ? (r % args.n_blk) * N : 0;


    src += r * N;
    dst += r * N;

    const int lane = tiisg;

    float reg[NE];
    for (int i = 0; i < NE; i++) {
        const float s = args.n_blk > 0 ? signs[i*NW + lane] : 1.0f;
        reg[i] = float(src[i*NW + lane])*s*scale;
    }
    for (int i = 1; i < NW; i *= 2) {
        for (int j = 0; j < NE; j++) {
            const float val = reg[j];
            const float val2 = simd_shuffle_xor(val, i);
            reg[j] = val2 - val + 2*((lane & i) == 0)*val;
        }
    }

    for (int i = NW; i < N; i *= 2) {
        const int step = i / NW;
        for (int j = 0; j < NE; j += (2 * step)) {
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k ];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

    for (int i = 0; i < NE; i++) {
        dst[i*NW + lane] = reg[i];
    }
}

// in-register FWHT of an N-block spread over NT threads (NE = N/NT values each, element i*NT + tid)
template<int N, int NT>
static inline void fwht_tg_butterfly(thread float * reg, threadgroup float * shmem, int tid) {
    constexpr int NW = N_SIMDWIDTH;
    constexpr int NE = N / NT;

    for (int i = 1; i < NW; i *= 2) {
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = simd_shuffle_xor(val, i);
            reg[j] = (tid & i) == 0 ? val2 + val : val2 - val;
        }
    }

    for (int i = NW; i < NT; i *= 2) {
        for (int j = 0; j < NE; j++) {
            shmem[j*NT + tid] = reg[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = shmem[j*NT + (tid ^ i)];
            reg[j] = (tid & i) == 0 ? val2 + val : val2 - val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = NT; i < N; i *= 2) {
        const int step = i / NT;
        for (int j = 0; j < NE; j += (2 * step)) {
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k ];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
}

// Wide blocks: one row per threadgroup instead of per simdgroup, so each thread keeps
// N/NT values rather than N/32. Butterflies below the simdgroup width still shuffle;
// those up to NT go through threadgroup memory; the rest stay in registers.
// TODO: try avoiding branch https://github.com/ggml-org/llama.cpp/pull/29094#discussion_r4049563223
// TODO: try to unroll loops
template<int N, int NT, typename src_t>
kernel void kernel_fwht_tg(
        constant ggml_metal_kargs_fwht & args,
        device const src_t * src,
        device float * dst,
        device const float * signs,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort3  ntg[[threads_per_threadgroup]]) {

    constexpr int NW = N_SIMDWIDTH;
    constexpr int NE = N / NT;

    threadgroup float shmem[N];

    const float scale = 1.0f / sqrt((float) N);

    const int64_t r = tgpig.x;
    if (r >= args.nrows) {
        return;
    }
    signs += args.n_blk > 0 ? (r % args.n_blk) * N : 0;


    src += r * N;
    dst += r * N;

    const int tid = sgitg * NW + tiisg;

    float reg[NE];
    for (int i = 0; i < NE; i++) {
        const float s = args.n_blk > 0 ? signs[i*NT + tid] : 1.0f;
        reg[i] = float(src[i*NT + tid])*s*scale;
    }

    fwht_tg_butterfly<N, NT>(reg, shmem, tid);

    for (int i = 0; i < NE; i++) {
        dst[i*NT + tid] = reg[i];
    }
}

// RMS_NORM + MUL (weight) + signed FWHT: one threadgroup per FWHT block of a row; each recomputes the row sum, so a
// single decode row still spreads over n_blk threadgroups
template<int N, int NT>
kernel void kernel_rms_norm_mul_fwht(
        constant ggml_metal_kargs_rms_norm_fwht & args,
        device const float * src,
        device const float * weight,
        device const float * signs,
        device       float * dst,
        device       float * dst_norm,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]]) {

    constexpr int NW  = N_SIMDWIDTH;
    constexpr int NE  = N / NT;
    constexpr int NSG = NT / NW;

    threadgroup float shmem[N];

    const int64_t r = tgpig.x / args.n_blk;
    const int     b = tgpig.x % args.n_blk;

    const int tid = sgitg * NW + tiisg;

    device const float * x = src + r*args.ne00;

    // float4 loads keep the per-thread chain of dependent loads short (ne00 is a multiple of N; the fusion check aligns rows)
    device const float4 * x4 = (device const float4 *) x;

    float sumf = 0.0f;
    for (int i = tid; i < args.ne00/4; i += NT) {
        sumf += dot(x4[i], x4[i]);
    }
    sumf = simd_sum(sumf);

    if (tiisg == 0) {
        shmem[sgitg] = sumf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = simd_sum(tiisg < NSG ? shmem[tiisg] : 0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float rms   = 1.0f/sqrt(sumf/args.ne00 + args.eps);
    const float scale = 1.0f/sqrt((float) N);

    const int64_t offs = r*args.ne00 + b*N;

    float reg[NE];
    for (int i = 0; i < NE; i++) {
        const int j = b*N + i*NT + tid;
        const float y = (x[j]*rms)*weight[j];
        if (args.write_norm) {
            dst_norm[offs + i*NT + tid] = y;
        }
        reg[i] = y*signs[j]*scale;
    }

    fwht_tg_butterfly<N, NT>(reg, shmem, tid);

    for (int i = 0; i < NE; i++) {
        dst[offs + i*NT + tid] = reg[i];
    }
}

// element f of a contiguous CONT output: the f-th element of the CONT source view in its own shape
static inline float fwht_gather_load(constant ggml_metal_kargs_fwht_gather & args, device const char * src, int f) {
    const int i0 = f % args.ne0;
    const int i1 = (f / args.ne0) % args.ne1;
    const int i2 = (f / args.ne0 / args.ne1) % args.ne2;
    const int i3 =  f / args.ne0 / args.ne1 / args.ne2;

    return *(device const float *) (src + i0*args.nb0 + i1*args.nb1 + i2*args.nb2 + i3*args.nb3);
}

// CONT + signed FWHT: kernel_fwht_tg with its input loaded through the CONT source view instead of the CONT output
template<int N, int NT>
kernel void kernel_fwht_gather(
        constant ggml_metal_kargs_fwht_gather & args,
        device const char  * src,
        device       float * dst,
        device const float * signs,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]]) {

    constexpr int NW = N_SIMDWIDTH;
    constexpr int NE = N / NT;

    threadgroup float shmem[N];

    const float scale = 1.0f / sqrt((float) N);

    const int r   = tgpig.x;
    const int tid = sgitg * NW + tiisg;

    signs += (r % args.n_blk) * N;
    dst   += r * N;

    float reg[NE];
    for (int i = 0; i < NE; i++) {
        reg[i] = fwht_gather_load(args, src, r*N + i*NT + tid)*signs[i*NT + tid]*scale;
    }

    fwht_tg_butterfly<N, NT>(reg, shmem, tid);

    for (int i = 0; i < NE; i++) {
        dst[i*NT + tid] = reg[i];
    }
}

typedef decltype(kernel_fwht_gather<1024, GGML_METAL_FWHT_TG_NT>) kernel_fwht_gather_t;

template [[host_name("kernel_fwht_gather_f32_512")]]        kernel kernel_fwht_gather_t kernel_fwht_gather<512,  GGML_METAL_FWHT_TG_NT>;
template [[host_name("kernel_fwht_gather_f32_1024")]]       kernel kernel_fwht_gather_t kernel_fwht_gather<1024, GGML_METAL_FWHT_TG_NT>;
template [[host_name("kernel_fwht_gather_f32_2048")]]       kernel kernel_fwht_gather_t kernel_fwht_gather<2048, GGML_METAL_FWHT_TG_NT>;
template [[host_name("kernel_fwht_gather_f32_4096")]]       kernel kernel_fwht_gather_t kernel_fwht_gather<4096, GGML_METAL_FWHT_TG_NT>;
template [[host_name("kernel_fwht_gather_f32_8192")]]       kernel kernel_fwht_gather_t kernel_fwht_gather<8192, GGML_METAL_FWHT_TG_NT>;
template [[host_name("kernel_fwht_gather_f32_512_nt128")]]  kernel kernel_fwht_gather_t kernel_fwht_gather<512,  GGML_METAL_FWHT_TG_NT_FALLBACK>;
template [[host_name("kernel_fwht_gather_f32_1024_nt128")]] kernel kernel_fwht_gather_t kernel_fwht_gather<1024, GGML_METAL_FWHT_TG_NT_FALLBACK>;
template [[host_name("kernel_fwht_gather_f32_2048_nt128")]] kernel kernel_fwht_gather_t kernel_fwht_gather<2048, GGML_METAL_FWHT_TG_NT_FALLBACK>;
template [[host_name("kernel_fwht_gather_f32_4096_nt128")]] kernel kernel_fwht_gather_t kernel_fwht_gather<4096, GGML_METAL_FWHT_TG_NT_FALLBACK>;
template [[host_name("kernel_fwht_gather_f32_8192_nt128")]] kernel kernel_fwht_gather_t kernel_fwht_gather<8192, GGML_METAL_FWHT_TG_NT_FALLBACK>;

typedef decltype(kernel_rms_norm_mul_fwht<1024, GGML_METAL_FWHT_TG_NT>) kernel_rms_norm_mul_fwht_t;

template [[host_name("kernel_rms_norm_mul_fwht_f32_512")]]        kernel kernel_rms_norm_mul_fwht_t kernel_rms_norm_mul_fwht<512,  GGML_METAL_FWHT_TG_NT>;
template [[host_name("kernel_rms_norm_mul_fwht_f32_1024")]]       kernel kernel_rms_norm_mul_fwht_t kernel_rms_norm_mul_fwht<1024, GGML_METAL_FWHT_TG_NT>;
template [[host_name("kernel_rms_norm_mul_fwht_f32_2048")]]       kernel kernel_rms_norm_mul_fwht_t kernel_rms_norm_mul_fwht<2048, GGML_METAL_FWHT_TG_NT>;
template [[host_name("kernel_rms_norm_mul_fwht_f32_4096")]]       kernel kernel_rms_norm_mul_fwht_t kernel_rms_norm_mul_fwht<4096, GGML_METAL_FWHT_TG_NT>;
template [[host_name("kernel_rms_norm_mul_fwht_f32_8192")]]       kernel kernel_rms_norm_mul_fwht_t kernel_rms_norm_mul_fwht<8192, GGML_METAL_FWHT_TG_NT>;
template [[host_name("kernel_rms_norm_mul_fwht_f32_512_nt128")]]  kernel kernel_rms_norm_mul_fwht_t kernel_rms_norm_mul_fwht<512,  GGML_METAL_FWHT_TG_NT_FALLBACK>;
template [[host_name("kernel_rms_norm_mul_fwht_f32_1024_nt128")]] kernel kernel_rms_norm_mul_fwht_t kernel_rms_norm_mul_fwht<1024, GGML_METAL_FWHT_TG_NT_FALLBACK>;
template [[host_name("kernel_rms_norm_mul_fwht_f32_2048_nt128")]] kernel kernel_rms_norm_mul_fwht_t kernel_rms_norm_mul_fwht<2048, GGML_METAL_FWHT_TG_NT_FALLBACK>;
template [[host_name("kernel_rms_norm_mul_fwht_f32_4096_nt128")]] kernel kernel_rms_norm_mul_fwht_t kernel_rms_norm_mul_fwht<4096, GGML_METAL_FWHT_TG_NT_FALLBACK>;
template [[host_name("kernel_rms_norm_mul_fwht_f32_8192_nt128")]] kernel kernel_rms_norm_mul_fwht_t kernel_rms_norm_mul_fwht<8192, GGML_METAL_FWHT_TG_NT_FALLBACK>;

typedef decltype(kernel_fwht<64, float>) kernel_fwht_f32_t;
typedef decltype(kernel_fwht<64, half>)  kernel_fwht_f16_t;

template [[host_name("kernel_fwht_f32_64")]]         kernel kernel_fwht_f32_t kernel_fwht<64,   float>;
template [[host_name("kernel_fwht_f32_128")]]        kernel kernel_fwht_f32_t kernel_fwht<128,  float>;
template [[host_name("kernel_fwht_f32_256")]]        kernel kernel_fwht_f32_t kernel_fwht<256,  float>;
template [[host_name("kernel_fwht_f32_512")]]        kernel kernel_fwht_f32_t kernel_fwht_tg<512,  GGML_METAL_FWHT_TG_NT, float>;
template [[host_name("kernel_fwht_f32_1024")]]       kernel kernel_fwht_f32_t kernel_fwht_tg<1024, GGML_METAL_FWHT_TG_NT, float>;
template [[host_name("kernel_fwht_f32_2048")]]       kernel kernel_fwht_f32_t kernel_fwht_tg<2048, GGML_METAL_FWHT_TG_NT, float>;
template [[host_name("kernel_fwht_f32_4096")]]       kernel kernel_fwht_f32_t kernel_fwht_tg<4096, GGML_METAL_FWHT_TG_NT, float>;
template [[host_name("kernel_fwht_f32_8192")]]       kernel kernel_fwht_f32_t kernel_fwht_tg<8192, GGML_METAL_FWHT_TG_NT, float>;
template [[host_name("kernel_fwht_f32_512_nt128")]]  kernel kernel_fwht_f32_t kernel_fwht_tg<512,  GGML_METAL_FWHT_TG_NT_FALLBACK, float>;
template [[host_name("kernel_fwht_f32_1024_nt128")]] kernel kernel_fwht_f32_t kernel_fwht_tg<1024, GGML_METAL_FWHT_TG_NT_FALLBACK, float>;
template [[host_name("kernel_fwht_f32_2048_nt128")]] kernel kernel_fwht_f32_t kernel_fwht_tg<2048, GGML_METAL_FWHT_TG_NT_FALLBACK, float>;
template [[host_name("kernel_fwht_f32_4096_nt128")]] kernel kernel_fwht_f32_t kernel_fwht_tg<4096, GGML_METAL_FWHT_TG_NT_FALLBACK, float>;
template [[host_name("kernel_fwht_f32_8192_nt128")]] kernel kernel_fwht_f32_t kernel_fwht_tg<8192, GGML_METAL_FWHT_TG_NT_FALLBACK, float>;

template [[host_name("kernel_fwht_f16_64")]]         kernel kernel_fwht_f16_t kernel_fwht<64,   half>;
template [[host_name("kernel_fwht_f16_128")]]        kernel kernel_fwht_f16_t kernel_fwht<128,  half>;
template [[host_name("kernel_fwht_f16_256")]]        kernel kernel_fwht_f16_t kernel_fwht<256,  half>;
template [[host_name("kernel_fwht_f16_512")]]        kernel kernel_fwht_f16_t kernel_fwht_tg<512,  GGML_METAL_FWHT_TG_NT, half>;
template [[host_name("kernel_fwht_f16_1024")]]       kernel kernel_fwht_f16_t kernel_fwht_tg<1024, GGML_METAL_FWHT_TG_NT, half>;
template [[host_name("kernel_fwht_f16_2048")]]       kernel kernel_fwht_f16_t kernel_fwht_tg<2048, GGML_METAL_FWHT_TG_NT, half>;
template [[host_name("kernel_fwht_f16_4096")]]       kernel kernel_fwht_f16_t kernel_fwht_tg<4096, GGML_METAL_FWHT_TG_NT, half>;
template [[host_name("kernel_fwht_f16_8192")]]       kernel kernel_fwht_f16_t kernel_fwht_tg<8192, GGML_METAL_FWHT_TG_NT, half>;
template [[host_name("kernel_fwht_f16_512_nt128")]]  kernel kernel_fwht_f16_t kernel_fwht_tg<512,  GGML_METAL_FWHT_TG_NT_FALLBACK, half>;
template [[host_name("kernel_fwht_f16_1024_nt128")]] kernel kernel_fwht_f16_t kernel_fwht_tg<1024, GGML_METAL_FWHT_TG_NT_FALLBACK, half>;
template [[host_name("kernel_fwht_f16_2048_nt128")]] kernel kernel_fwht_f16_t kernel_fwht_tg<2048, GGML_METAL_FWHT_TG_NT_FALLBACK, half>;
template [[host_name("kernel_fwht_f16_4096_nt128")]] kernel kernel_fwht_f16_t kernel_fwht_tg<4096, GGML_METAL_FWHT_TG_NT_FALLBACK, half>;
template [[host_name("kernel_fwht_f16_8192_nt128")]] kernel kernel_fwht_f16_t kernel_fwht_tg<8192, GGML_METAL_FWHT_TG_NT_FALLBACK, half>;

constant int FC_dsv4_hc_n_hc [[function_constant(FC_DSV4_HC + 0)]];

kernel void kernel_dsv4_hc_comb_f32(
        constant ggml_metal_kargs_dsv4_hc_comb & args,
        device const char * mixes,
        device const char * scale,
        device const char * base,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr ushort hc = 4;
    constexpr ushort comb_offset = 2*hc;

    const int it = tgpig.x*ntg.y + sgitg;
    if (it >= args.n_tokens) {
        return;
    }

    float scale_lane = 0.0f;
    if (tiisg == 0) {
        scale_lane = *(device const float *) (scale + 2*args.nb_s0);
    }
    const float scale_comb = simd_shuffle(scale_lane, 0);

    float v = 0.0f;
    if (tiisg < hc*hc) {
        v = *(device const float *) (mixes + (comb_offset + tiisg)*args.nb_m0 + it*args.nb_m1)*scale_comb
          + *(device const float *) (base   + (comb_offset + tiisg)*args.nb_b0);
    }

    // Softmax across destinations (the four contiguous lanes for each source).
    float vmax = max(v, simd_shuffle_xor(v, 1));
    vmax = max(vmax, simd_shuffle_xor(vmax, 2));
    v = exp(v - vmax);

    float sum = v + simd_shuffle_xor(v, 1);
    sum += simd_shuffle_xor(sum, 2);
    v = v/sum + args.eps;

    // Normalize columns: equal destination indices are four lanes apart.
    sum = v + simd_shuffle_xor(v, 4);
    sum += simd_shuffle_xor(sum, 8);
    v /= sum + args.eps;

    for (int i = 1; i < args.n_iter; ++i) {
        sum = v + simd_shuffle_xor(v, 1);
        sum += simd_shuffle_xor(sum, 2);
        v /= sum + args.eps;

        sum = v + simd_shuffle_xor(v, 4);
        sum += simd_shuffle_xor(sum, 8);
        v /= sum + args.eps;
    }

    if (tiisg < hc*hc) {
        const ushort idst = tiisg & 3;
        const ushort isrc = tiisg >> 2;
        *(device float *) (dst + idst*args.nb_d0 + isrc*args.nb_d1 + it*args.nb_d2) = v;
    }
}

kernel void kernel_dsv4_hc_pre_f32(
        constant ggml_metal_kargs_dsv4_hc_pre & args,
        device const char * x,
        device const char * weights,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    if (i0 >= args.n_embd) {
        return;
    }

    device const char * xb = x + i0*args.nb_x0 + it*args.nb_x2;
    float result = 0.0f;
    FOR_UNROLL (int ih = 0; ih < FC_dsv4_hc_n_hc; ++ih) {
        const float xv = *(device const float *) (xb + ih*args.nb_x1);
        const float wv = *(device const float *) (weights + ih*args.nb_w0 + it*args.nb_w1);
        result = fma(xv, wv, result);
    }

    *(device float *) (dst + i0*args.nb_d0 + it*args.nb_d1) = args.scale*result;
}

kernel void kernel_dsv4_hc_pre_gated_f32(
        constant ggml_metal_kargs_dsv4_hc_pre & args,
        device const char * x,
        device const char * gate,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    if (i0 >= args.n_embd) {
        return;
    }

    device const char * xb = x    + i0*args.nb_x0 + it*args.nb_x2;
    device const char * gb = gate + i0*args.nb_w0 + it*args.nb_w2;
    float result = 0.0f;
    FOR_UNROLL (int ih = 0; ih < FC_dsv4_hc_n_hc; ++ih) {
        const float g  = 1.0f/(1.0f + exp(-*(device const float *) (gb + ih*args.nb_w1)));
        const float xv = *(device const float *) (xb + ih*args.nb_x1);
        result = fma(xv, g, result);
    }

    *(device float *) (dst + i0*args.nb_d0 + it*args.nb_d1) = args.scale*result;
}

kernel void kernel_dsv4_hc_post_nocomb_f32(
        constant ggml_metal_kargs_dsv4_hc_post & args,
        device const char * x,
        device const char * residual,
        device const char * post,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr ushort hc = 4;

    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    float post_lane = 0.0f;
    if (tiisg < hc) {
        post_lane = *(device const float *) (post + tiisg*args.nb_p0 + it*args.nb_p1);
    }

    float post_reg[hc];
    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        post_reg[idst] = simd_shuffle(post_lane, idst);
    }

    if (i0 >= args.n_embd) {
        return;
    }

    const float xv = *(device const float *) (x + i0*args.nb_x0 + it*args.nb_x1);
    device const char * rb = residual + i0*args.nb_r0 + it*args.nb_r2;
    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        const float rv = *(device const float *) (rb + idst*args.nb_r1);
        *(device float *) (dst + i0*args.nb_d0 + idst*args.nb_d1 + it*args.nb_d2) = xv*post_reg[idst] + rv;
    }
}

kernel void kernel_dsv4_hc_post_f32(
        constant ggml_metal_kargs_dsv4_hc_post & args,
        device const char * x,
        device const char * residual,
        device const char * post,
        device const char * comb,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr ushort hc = 4;

    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    float coeff_lane = 0.0f;
    if (tiisg < hc) {
        coeff_lane = *(device const float *) (post + tiisg*args.nb_p0 + it*args.nb_p1);
    } else if (tiisg < hc + hc*hc) {
        const ushort idx  = tiisg - hc;
        const ushort idst = idx & 3;
        const ushort isrc = idx >> 2;
        coeff_lane = *(device const float *) (comb + idst*args.nb_c0 + isrc*args.nb_c1 + it*args.nb_c2);
    }

    float post_reg[hc];
    float comb_reg[hc][hc];
    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        post_reg[idst] = simd_shuffle(coeff_lane, idst);
    }
    FOR_UNROLL (ushort isrc = 0; isrc < hc; ++isrc) {
        FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
            comb_reg[isrc][idst] = simd_shuffle(coeff_lane, hc + idst + hc*isrc);
        }
    }

    if (i0 >= args.n_embd) {
        return;
    }

    const float xv = *(device const float *) (x + i0*args.nb_x0 + it*args.nb_x1);
    float result[hc];
    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        result[idst] = xv*post_reg[idst];
    }

    device const char * rb = residual + i0*args.nb_r0 + it*args.nb_r2;
    FOR_UNROLL (ushort isrc = 0; isrc < hc; ++isrc) {
        const float rv = *(device const float *) (rb + isrc*args.nb_r1);
        FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
            result[idst] = fma(rv, comb_reg[isrc][idst], result[idst]);
        }
    }

    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        *(device float *) (dst + i0*args.nb_d0 + idst*args.nb_d1 + it*args.nb_d2) = result[idst];
    }
}


// gate and output slots of the LSTM cell and the transducer step control, in ggml's order
constant uint LSTM_GATE_INPUT  = 0;
constant uint LSTM_GATE_FORGET = 1;
constant uint LSTM_GATE_CELL   = 2;
constant uint LSTM_GATE_OUTPUT = 3;
constant uint LSTM_N_GATES     = 4;
constant uint LSTM_OUT_H       = 0;
constant uint LSTM_OUT_C       = 1;
constant uint LSTM_N_OUTS      = 2;
constant uint TDT_STEP_IN_T      = 0;
constant uint TDT_STEP_IN_S      = 1;
constant uint TDT_STEP_IN_N      = 2;
constant uint TDT_STEP_OUT_T     = 0;
constant uint TDT_STEP_OUT_S     = 1;
constant uint TDT_STEP_OUT_N     = 2;
constant uint TDT_STEP_OUT_UPD   = 3;
constant uint TDT_STEP_OUT_HOLD  = 4;
constant uint TDT_STEP_OUT_FRAME = 5;
constant uint TDT_STEP_OUT_TOKEN = 6;
constant uint TDT_STEP_OUT_DUR   = 7;

static inline void lstm_cell_pair_f32(
        device const float * g, float c_prev, uint H, uint j,
        thread float & h_new, thread float & c_new) {
    const float gi = 1 / (1 + exp(-g[LSTM_GATE_INPUT  * H + j]));
    const float gf = 1 / (1 + exp(-g[LSTM_GATE_FORGET * H + j]));
    const float gg = precise::tanh(g[LSTM_GATE_CELL   * H + j]);
    const float go = 1 / (1 + exp(-g[LSTM_GATE_OUTPUT * H + j]));

    // Metal compiles with fast math; the barrier stops the two products from
    // contracting into an FMA, which the decomposed mul/mul/add graph cannot do.
    {
        #pragma clang fp contract(off)
        c_new = gf * c_prev + gi * gg;
    }

    h_new = go * precise::tanh(c_new);
}

kernel void kernel_lstm_cell_f32(
        constant ggml_metal_kargs_lstm_cell & args,
        device const float * gates,  // [4H, N]
        device const float * c_prev, // [H, N]
        device       float * dst,    // [2H, N]
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {

    const uint H        = (uint) args.H;
    const uint prev_row = (uint) args.prev_row;
    const uint total    = H * (uint) args.N;

    const uint stride = ntg.x * tgpg.x;
    for (uint k = tgpig.x * ntg.x + tpitg.x; k < total; k += stride) {
        const uint n = k / H;
        const uint j = k - n * H;

        float h_new;
        float c_new;
        lstm_cell_pair_f32(gates + n * LSTM_N_GATES * H, c_prev[n * prev_row + j], H, j,
                           h_new, c_new);

        device float * out = dst + n * LSTM_N_OUTS * H;

        out[LSTM_OUT_H * H + j] = h_new;
        out[LSTM_OUT_C * H + j] = c_new;
    }
}

kernel void kernel_lstm_cell_masked_f32(
        constant ggml_metal_kargs_lstm_cell & args,
        device const float * gates,   // [4H, N]
        device const float * hc_prev, // [2H, N]
        device const int   * mask,    // [N] or [1]
        device       float * dst,     // [2H, N]
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {

    const uint H        = (uint) args.H;
    const uint prev_row = (uint) args.prev_row;
    const uint c_base   = (uint) args.c_base;
    const uint m_stride = (uint) args.mask_stride;
    const uint total    = H * (uint) args.N;

    const uint stride = ntg.x * tgpg.x;
    for (uint k = tgpig.x * ntg.x + tpitg.x; k < total; k += stride) {
        const uint n = k / H;
        const uint j = k - n * H;

        device const float * p   = hc_prev + n * prev_row;
        device       float * out = dst     + n * LSTM_N_OUTS * H;

        if (mask[n * m_stride] == 0) {
            out[LSTM_OUT_H * H + j] = p[j];
            out[LSTM_OUT_C * H + j] = p[c_base + j];
            continue;
        }

        float h_new;
        float c_new;
        lstm_cell_pair_f32(gates + n * LSTM_N_GATES * H, p[c_base + j], H, j, h_new, c_new);

        out[LSTM_OUT_H * H + j] = h_new;
        out[LSTM_OUT_C * H + j] = c_new;
    }
}

kernel void kernel_tdt_step_i32(
        constant ggml_metal_kargs_tdt_step & args,
        device const int * token,
        device const int * dur_idx,
        device const int * state,
        device const int * dur_table,
        device       int * dst) {
    const int t = state[TDT_STEP_IN_T];
    const int s = state[TDT_STEP_IN_S];
    const int n = state[TDT_STEP_IN_N];

    int t_next = t;
    int s_next = s;
    int update = 0;

    if (t < n) {
        const int raw = dur_idx[0];
        const int di  = raw < 0 ? 0 : (raw < args.n_dur ? raw : args.n_dur - 1);
        const int dur = args.rnnt ? 0 : dur_table[di];
        const int adv = args.rnnt ? 1 : (dur > 1 ? dur : 1);

        if (token[0] == args.blank_id) {
            t_next = t + adv;
            s_next = 0;
        } else {
            update = 1;
            s_next = s + 1;
            if ((args.rnnt == 0 && dur > 0) || s_next >= args.max_symbols) {
                t_next = t + adv;
                s_next = 0;
            }
        }
    }

    int frame = t_next > n - 1 ? n - 1 : t_next;
    if (frame < 0) {
        frame = 0;
    }

    dst[TDT_STEP_OUT_T]     = t_next;
    dst[TDT_STEP_OUT_S]     = s_next;
    dst[TDT_STEP_OUT_N]     = n;
    dst[TDT_STEP_OUT_UPD]   = update;
    dst[TDT_STEP_OUT_HOLD]  = 1 - update;
    dst[TDT_STEP_OUT_FRAME] = frame;
    dst[TDT_STEP_OUT_TOKEN] = token[0];
    dst[TDT_STEP_OUT_DUR]   = dur_idx[0];
}

kernel void kernel_scale_residual_f32(
    constant   ggml_metal_kargs_scale_residual & args,
    device  const float * x,
    device  const float * bias,
    device  const float * gamma,
    device  const float * residual,
    device        float * y,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int c = (int) tgpig.x;
    const int batch = (int) tgpig.y;
    if (c >= args.C || batch >= args.B) return;

    const int L = args.L;
    const int sxt = args.sxt, sxc = args.sxc;
    const int syt = args.syt, syc = args.syc;
    const int srt = args.srt, src = args.src;
    const float bv = bias[c];
    const float gv = gamma[c];
    const size_t batch_offset = (size_t) batch * L * args.C;

    device const float * x_c = x        + batch_offset + (size_t) c * sxc;
    device const float * r_c = residual + batch_offset + (size_t) c * src;
    device       float * y_c = y        + batch_offset + (size_t) c * syc;

    for (int t = (int) tpitg.x; t < L; t += (int) ntg.x) {
        y_c[(size_t) t * syt] = r_c[(size_t) t * srt] + (x_c[(size_t) t * sxt] + bv) * gv;
    }
}

kernel void kernel_bias_gelu_f32(
    constant   ggml_metal_kargs_bias_gelu & args,
    device  const float * x,
    device  const float * bias,
    device        float * y,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int c = (int) tgpig.x;
    const int batch = (int) tgpig.y;
    if (c >= args.C || batch >= args.B) return;

    const int L = args.L;
    const int sxt = args.sxt, sxc = args.sxc;
    const int syt = args.syt, syc = args.syc;
    const float bv = bias[c];
    const size_t batch_offset = (size_t) batch * L * args.C;

    device const float * x_c = x + batch_offset + (size_t) c * sxc;
    device       float * y_c = y + batch_offset + (size_t) c * syc;

    // Use the same erf_approx template as kernel_gelu_erf_f32 above
    // (Abramowitz & Stegun 7.1.26 / Hastings approximation) so the fused
    // bias_gelu output is bit-identical to the unfused add + gelu_erf path.
    for (int t = (int) tpitg.x; t < L; t += (int) ntg.x) {
        const float v = x_c[(size_t) t * sxt] + bv;
        y_c[(size_t) t * syt] = 0.5f * v * (1.0f + erf_approx<float>(v * SQRT_2_INV));
    }
}

kernel void kernel_pad_replicate_1d_f32(
    constant   ggml_metal_kargs_pad_replicate_1d & args,
    device  const float * x,
    device        float * y,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int c = (int) tgpig.x;
    if (c >= args.C) return;

    const int L_in  = args.L_in;
    const int L_out = args.L_out;
    const int pad   = args.pad_left;
    const int sxt = args.sxt, sxc = args.sxc;
    const int syt = args.syt, syc = args.syc;

    device const float * x_c = x + (size_t) c * sxc;
    device       float * y_c = y + (size_t) c * syc;

    for (int t = (int) tpitg.x; t < L_out; t += (int) ntg.x) {
        int src_t = t - pad;
        if (src_t < 0)      src_t = 0;
        if (src_t >= L_in)  src_t = L_in - 1;
        y_c[(size_t) t * syt] = x_c[(size_t) src_t * sxt];
    }
}
