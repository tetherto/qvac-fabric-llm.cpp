#include "common.h"
#include "dequantize.h"
#include "fa_common.metal"

// TODO: this is quite ugly. in the future these types will be hardcoded in the kernel, but for now keep them as
//       template to be able to explore different combinations

#define FA_TYPES \
    half,   half4,     simdgroup_half8x8,  \
    half,   half4x4,   simdgroup_half8x8,  \
    half,   half4x4,   simdgroup_half8x8,  \
    float,             simdgroup_float8x8, \
    float,  float2,    simdgroup_float8x8, \
    float,  float4,    simdgroup_float8x8
    //half,   half4,     simdgroup_half8x8

#define FA_TYPES_BF \
    bfloat, bfloat4,   simdgroup_bfloat8x8, \
    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
    float,             simdgroup_float8x8,  \
    float,  float2,    simdgroup_float8x8,  \
    half,   half4,     simdgroup_half8x8
    //float,  float4,    simdgroup_float8x8

#define FA_TYPES_F32 \
    half,   half4,     simdgroup_half8x8,  \
    float,  float4x4,  simdgroup_float8x8, \
    float,  float4x4,  simdgroup_float8x8, \
    float,             simdgroup_float8x8, \
    float,  float2,    simdgroup_float8x8, \
    float,  float4,    simdgroup_float8x8
    //half,   half4,     simdgroup_half8x8

typedef decltype(kernel_flash_attn_ext<FA_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 64, 64>) flash_attn_ext_t;

template [[host_name("kernel_flash_attn_ext_f16_dk32_dv32"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  32,  32>;
template [[host_name("kernel_flash_attn_ext_f16_dk40_dv40"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  40,  40>;
template [[host_name("kernel_flash_attn_ext_f16_dk48_dv48"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  48,  48>;
template [[host_name("kernel_flash_attn_ext_f16_dk64_dv64"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  64,  64>;
template [[host_name("kernel_flash_attn_ext_f16_dk72_dv72"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  72,  72>;
template [[host_name("kernel_flash_attn_ext_f16_dk80_dv80"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  80,  80>;
template [[host_name("kernel_flash_attn_ext_f16_dk96_dv96"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  96,  96>;
template [[host_name("kernel_flash_attn_ext_f16_dk96_dv64"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  96,  64>;
template [[host_name("kernel_flash_attn_ext_f16_dk112_dv112")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  112, 112>;
template [[host_name("kernel_flash_attn_ext_f16_dk128_dv128")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  128, 128>;
template [[host_name("kernel_flash_attn_ext_f16_dk192_dv192")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  192, 192>;
template [[host_name("kernel_flash_attn_ext_f16_dk192_dv128")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  192, 128>;
template [[host_name("kernel_flash_attn_ext_f16_dk256_dv256")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  256, 256>;
template [[host_name("kernel_flash_attn_ext_f16_dk320_dv256")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  320, 256>;
template [[host_name("kernel_flash_attn_ext_f16_dk512_dv512")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  512, 512>;
template [[host_name("kernel_flash_attn_ext_f16_dk576_dv512")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  576, 512>;

#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_bf16_dk32_dv32"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 32,  32>;
template [[host_name("kernel_flash_attn_ext_bf16_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 40,  40>;
template [[host_name("kernel_flash_attn_ext_bf16_dk48_dv48"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 48,  48>;
template [[host_name("kernel_flash_attn_ext_bf16_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 64,  64>;
template [[host_name("kernel_flash_attn_ext_bf16_dk72_dv72"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 72,  72>;
template [[host_name("kernel_flash_attn_ext_bf16_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 80,  80>;
template [[host_name("kernel_flash_attn_ext_bf16_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 96,  96>;
template [[host_name("kernel_flash_attn_ext_bf16_dk96_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 96,  64>;
template [[host_name("kernel_flash_attn_ext_bf16_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 112, 112>;
template [[host_name("kernel_flash_attn_ext_bf16_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 128, 128>;
template [[host_name("kernel_flash_attn_ext_bf16_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 192, 192>;
template [[host_name("kernel_flash_attn_ext_bf16_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 192, 128>;
template [[host_name("kernel_flash_attn_ext_bf16_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 256, 256>;
template [[host_name("kernel_flash_attn_ext_bf16_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 320, 256>;
template [[host_name("kernel_flash_attn_ext_bf16_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 512, 512>;
template [[host_name("kernel_flash_attn_ext_bf16_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 576, 512>;
#endif

#undef FA_TYPES
#undef FA_TYPES_BF
#undef FA_TYPES_F32

// prefill flash attention with F16 K/V for long KV, laid out like mul_mm: a threadgroup of 8 simdgroups takes 32
// query rows. Per KV tile of BK rows each simdgroup computes a 16x16 block of S = Q*K^T (Q staged as half in
// threadgroup memory, K read from device memory), the online softmax runs on S in threadgroup memory (4 rows per
// simdgroup), then each simdgroup adds P*V into its own 32-row x D/8-column slice of O, kept in registers.
// Needs ne11 % BK == 0; no mask block skipping, sinks, ALiBi or softcap
template<short D, short BK>
kernel void kernel_flash_attn_ext_tb(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device       char * dst,
        threadgroup  half * shmem [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short NSG = 8;
    constexpr short BQ  = 32;
    constexpr short LQ  = D + 8;   // row stride of the staged Q (halfs)
    constexpr short LS  = BK + 4;  // row stride of S (floats)
    constexpr short D8  = D/8;
    constexpr short OC  = D/NSG;   // O columns per simdgroup
    constexpr short OC8 = OC/8;
    constexpr short BK8 = BK/8;
    constexpr short NCB = BK/16;   // 16-column S blocks
    constexpr short NRB = BQ/16;   // 16-row S blocks
    static_assert(NCB*NRB == NSG, "S blocks must match the simdgroups");

    const int   iq1 = tgpig[0]*BQ;
    const short iq2 = tgpig[1];
    const short iq3 = tgpig[2];

    const short ikv2 = iq2/(args.ne02/args.ne_12_2);
    const short ikv3 = iq3/(args.ne03/args.ne_12_3);

    device const half * pk = (device const half *) (k + ikv2*args.nb12 + ikv3*args.nb13);
    device const half * pv = (device const half *) (v + ikv2*args.nb22 + ikv3*args.nb23);

    threadgroup half  * sq   = shmem;
    threadgroup float * ss   = (threadgroup float *) (shmem + BQ*LQ);
    threadgroup float * sfac = ss + BQ*LS;

    const short tid = sgitg*32 + tiisg;

    const short qid = tiisg/4;
    const short fm  = (qid & 4) + ((tiisg/2) % 4);
    const short fn  = (qid & 2)*2 + (tiisg % 2)*2;

    for (short e = tid; e < BQ*D/4; e += 32*NSG) {
        const short r  = e/(D/4);
        const short c4 = e%(D/4);
        half4 x = 0;
        if (iq1 + r < args.ne01) {
            x = (half4) ((device const float4 *) (q + (iq1 + r)*args.nb01 + iq2*args.nb02 + iq3*args.nb03))[c4];
        }
        *((threadgroup half4 *) (sq + r*LQ) + c4) = x;
    }

    // S block of this simdgroup
    const short sr = 16*(sgitg/NCB);
    const short sc = 16*(sgitg%NCB);

    // softmax rows of this simdgroup
    constexpr short RS = BQ/NSG;
    float M[RS];
    float S[RS];
    FOR_UNROLL (short j = 0; j < RS; ++j) {
        M[j] = -FLT_MAX/2;
        S[j] = 0.0f;
    }

    // O slice of this simdgroup: all BQ rows, columns OC*sgitg ...
    simdgroup_float8x8 lo[BQ/8][OC8];
    FOR_UNROLL (short rb = 0; rb < BQ/8; ++rb) {
        FOR_UNROLL (short cb = 0; cb < OC8; ++cb) {
            lo[rb][cb] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        }
    }

    device const half * pmask = (device const half *) (mask + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int ic = 0; ic < args.ne11; ic += BK) {
        {
            // S = Q*K^T, 16x16 per simdgroup
            simdgroup_float8x8 ms[2][2];
            FOR_UNROLL (short a = 0; a < 2; ++a) {
                FOR_UNROLL (short b = 0; b < 2; ++b) {
                    ms[a][b] = make_filled_simdgroup_matrix<float, 8>(0.0f);
                }
            }
            device const half * pk0 = pk + (int64_t) (ic + sc)*args.ns10;
            FOR_UNROLL (short i = 0; i < D8; ++i) {
                simdgroup_half8x8 mk0, mk1;
                simdgroup_half8x8 mq0, mq1;
                simdgroup_load(mq0, sq + (sr + 0)*LQ + 8*i, LQ);
                simdgroup_load(mq1, sq + (sr + 8)*LQ + 8*i, LQ);
                simdgroup_load(mk0, pk0 + 8*i,                 args.ns10, 0, true);
                simdgroup_load(mk1, pk0 + 8*args.ns10 + 8*i,   args.ns10, 0, true);
                simdgroup_multiply_accumulate(ms[0][0], mq0, mk0, ms[0][0]);
                simdgroup_multiply_accumulate(ms[0][1], mq0, mk1, ms[0][1]);
                simdgroup_multiply_accumulate(ms[1][0], mq1, mk0, ms[1][0]);
                simdgroup_multiply_accumulate(ms[1][1], mq1, mk1, ms[1][1]);
            }
            FOR_UNROLL (short a = 0; a < 2; ++a) {
                FOR_UNROLL (short b = 0; b < 2; ++b) {
                    simdgroup_store(ms[a][b], ss + (sr + 8*a)*LS + sc + 8*b, LS);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // online softmax in base 2, RS rows per simdgroup, BK/32 columns per thread
        FOR_UNROLL (short j = 0; j < RS; ++j) {
            const short r = RS*sgitg + j;
            const int row = iq1 + r;
            float s[BK/32];
            float mmax = -FLT_MAX/2;
            FOR_UNROLL (short t = 0; t < BK/32; ++t) {
                const short c = 32*t + tiisg;
                const float mk = row < args.ne31 ? (float) pmask[(int64_t) row*(args.nb31/2) + ic + c] : 0.0f;
                s[t] = (ss[r*LS + c]*args.scale + mk)*M_LOG2E_F;
                mmax = max(mmax, s[t]);
            }
            mmax = simd_max(mmax);
            const float mnew = max(M[j], mmax);
            const float fac  = exp2(M[j] - mnew);
            float rsum = 0.0f;
            FOR_UNROLL (short t = 0; t < BK/32; ++t) {
                const float p = exp2(s[t] - mnew);
                ss[r*LS + 32*t + tiisg] = p;
                rsum += p;
            }
            rsum = simd_sum(rsum);
            S[j] = S[j]*fac + rsum;
            M[j] = mnew;
            if (tiisg == 0) {
                sfac[r] = fac;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O = O*fac + P*V on this simdgroup's columns
        FOR_UNROLL (short rb = 0; rb < BQ/8; ++rb) {
            const float fac = sfac[8*rb + fm];
            FOR_UNROLL (short cb = 0; cb < OC8; ++cb) {
                thread auto & el = lo[rb][cb].thread_elements();
                el[0] *= fac;
                el[1] *= fac;
            }
        }

        device const half * pv0 = pv + (int64_t) ic*args.ns20 + OC*sgitg;
        FOR_UNROLL (short kb = 0; kb < BK8; ++kb) {
            simdgroup_float8x8 mp[BQ/8];
            FOR_UNROLL (short rb = 0; rb < BQ/8; ++rb) {
                simdgroup_load(mp[rb], ss + 8*rb*LS + 8*kb, LS);
            }
            FOR_UNROLL (short cb = 0; cb < OC8; ++cb) {
                simdgroup_half8x8 mv;
                simdgroup_load(mv, pv0 + (int64_t) 8*kb*args.ns20 + 8*cb, args.ns20);
                FOR_UNROLL (short rb = 0; rb < BQ/8; ++rb) {
                    simdgroup_multiply_accumulate(lo[rb][cb], mp[rb], mv, lo[rb][cb]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tiisg == 0) {
        FOR_UNROLL (short j = 0; j < RS; ++j) {
            sfac[RS*sgitg + j] = S[j] == 0.0f ? 0.0f : 1.0f/S[j];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL (short rb = 0; rb < BQ/8; ++rb) {
        const int row = iq1 + 8*rb + fm;
        if (row >= args.ne01) {
            continue;
        }
        const float inv = sfac[8*rb + fm];
        device float * d = (device float *) dst + ((int64_t) iq3*args.ne2*args.ne1 + iq2 + (int64_t) row*args.ne1)*D + OC*sgitg;
        FOR_UNROLL (short cb = 0; cb < OC8; ++cb) {
            thread auto & el = lo[rb][cb].thread_elements();
            *((device float2 *) (d + 8*cb + fn)) = float2(el[0], el[1])*inv;
        }
    }
}

typedef decltype(kernel_flash_attn_ext_tb<256, 64>) flash_attn_ext_tb_t;

template [[host_name("kernel_flash_attn_ext_tb_f16_dk256_bk64")]] kernel flash_attn_ext_tb_t kernel_flash_attn_ext_tb<256, 64>;
