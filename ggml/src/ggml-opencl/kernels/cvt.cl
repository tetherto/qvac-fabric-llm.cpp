//------------------------------------------------------------------------------
// This file is contains kernels for data conversion.
// These kernels are used when loading the model, so its performance is less
// important.
//------------------------------------------------------------------------------
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

#define QK4_0                   32
#define QR4_0                   2
#define QK4_1                   32
#define QR4_1                   2
#define QK5_0                   32
#define QR5_0                   2
#define QK5_1                   32
#define QR5_1                   2
#define QK8_0                   32
#define QR8_0                   1
#define QK1_0                   128
#define QR1_0                   1
#define QK_K                    256
#define K_SCALE_SIZE            (3 * QK_K / 64)
#define K_QUANTS_PER_ITERATION  2

typedef char int8_t;
typedef uchar uint8_t;
typedef short int16_t;
typedef ushort uint16_t;
typedef int int32_t;
typedef uint uint32_t;

// trans4_ns GEMM kernels consume bytes in least-significant-first numeric order.
//
// Everything here stays in uint registers. Narrowing intermediates to uchar
// locals is miscompiled by the Adreno E031.47 shader compiler: it keeps the
// first byte of each packed word and drops the rest, which silently corrupts
// every repacked expert weight.
static inline uint pack_uchar4(uint x0, uint x1, uint x2, uint x3) {
    return (x0 & 0xFFu) | ((x1 & 0xFFu) << 8) | ((x2 & 0xFFu) << 16) | ((x3 & 0xFFu) << 24);
}

static inline uchar unpack_uchar0(uint x) {
    return (uchar) (x & 0xFF);
}

static inline uchar unpack_uchar1(uint x) {
    return (uchar) ((x >> 8) & 0xFF);
}

static inline uchar unpack_uchar2(uint x) {
    return (uchar) ((x >> 16) & 0xFF);
}

static inline uchar unpack_uchar3(uint x) {
    return (uchar) ((x >> 24) & 0xFF);
}

// The low nibbles are combined with a multiply rather than `| (x << 4)`: the
// E031.47 compiler drops the mask on the shifted term, letting the odd byte's
// high nibble leak into the next byte of the packed word.
static inline uint pack_trans4_low(__global const uchar * q, uint offset) {
    uint w = 0;
    for (uint t = 0; t < 4; ++t) {
        const uint lo = q[offset + 2 * t + 0] & 0x0Fu;
        const uint hi = q[offset + 2 * t + 1] & 0x0Fu;
        w |= ((lo + hi * 16u) & 0xFFu) << (8u * t);
    }
    return w;
}

static inline uint pack_trans4_high(__global const uchar * q, uint offset) {
    const uint x0 = q[offset + 0];
    const uint x1 = q[offset + 1];
    const uint x2 = q[offset + 2];
    const uint x3 = q[offset + 3];
    const uint x4 = q[offset + 4];
    const uint x5 = q[offset + 5];
    const uint x6 = q[offset + 6];
    const uint x7 = q[offset + 7];
    return pack_uchar4(
        ((x0 & 0xF0u) >> 4) | (x1 & 0xF0u),
        ((x2 & 0xF0u) >> 4) | (x3 & 0xF0u),
        ((x4 & 0xF0u) >> 4) | (x5 & 0xF0u),
        ((x6 & 0xF0u) >> 4) | (x7 & 0xF0u));
}

static inline void restore_trans4(
    __global uchar * q,
    uint offset,
    uint low,
    uint high
) {
    uchar lo0 = unpack_uchar0(low);
    uchar lo1 = unpack_uchar1(low);
    uchar lo2 = unpack_uchar2(low);
    uchar lo3 = unpack_uchar3(low);
    uchar hi0 = unpack_uchar0(high);
    uchar hi1 = unpack_uchar1(high);
    uchar hi2 = unpack_uchar2(high);
    uchar hi3 = unpack_uchar3(high);
    q[offset + 0] = (lo0 & 0x0F) | ((hi0 & 0x0F) << 4);
    q[offset + 1] = ((lo0 & 0xF0) >> 4) | (hi0 & 0xF0);
    q[offset + 2] = (lo1 & 0x0F) | ((hi1 & 0x0F) << 4);
    q[offset + 3] = ((lo1 & 0xF0) >> 4) | (hi1 & 0xF0);
    q[offset + 4] = (lo2 & 0x0F) | ((hi2 & 0x0F) << 4);
    q[offset + 5] = ((lo2 & 0xF0) >> 4) | (hi2 & 0xF0);
    q[offset + 6] = (lo3 & 0x0F) | ((hi3 & 0x0F) << 4);
    q[offset + 7] = ((lo3 & 0xF0) >> 4) | (hi3 & 0xF0);
}

//------------------------------------------------------------------------------
// block_q1_0
//------------------------------------------------------------------------------
typedef struct {
    half d;             // delta
    uchar qs[QK1_0/8];  // 1-bit signs (16 bytes)
} block_q1_0;

//------------------------------------------------------------------------------
// block_q4_0
//------------------------------------------------------------------------------
struct block_q4_0
{
    half d;
    uint8_t qs[QK4_0 / 2];
};

//------------------------------------------------------------------------------
// block_q4_1
//------------------------------------------------------------------------------
struct block_q4_1 {
    half d; // delta
    half m; // min
    uchar qs[QK4_1 / 2]; // nibbles / quants
};

//------------------------------------------------------------------------------
// block_q5_0
//------------------------------------------------------------------------------
struct block_q5_0 {
    half d; // delta
    uchar qh[4]; // 5-th bit of quants
    uchar qs[QK5_0 / 2]; // nibbles / quants
};

//------------------------------------------------------------------------------
// block_q5_1
//------------------------------------------------------------------------------
struct block_q5_1 {
    half d; // delta
    half m; // min
    uchar qh[4]; // 5-th bit of quants
    uchar qs[QK5_1 / 2]; // nibbles / quants
};

//------------------------------------------------------------------------------
// block_q4_k
//------------------------------------------------------------------------------
struct block_q4_K {
    half d; // delta
    half dm; // min
    uchar s[K_SCALE_SIZE];
    uchar q[QK_K / 2]; // nibbles / quants
};

//------------------------------------------------------------------------------
// block_q5_k
//------------------------------------------------------------------------------
struct block_q5_K {
    half d; // delta
    half dm; // min
    uchar s[K_SCALE_SIZE];
    uchar qh[QK_K / 8];
    uchar qs[QK_K / 2]; // nibbles / quants
};

//------------------------------------------------------------------------------
// block_q6_K
//------------------------------------------------------------------------------
struct block_q6_K {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    half d;                  // super-block scale
};

//------------------------------------------------------------------------------
// block_iq4_nl
//------------------------------------------------------------------------------
#define QK4_NL 32

struct block_iq4_nl
{
    half d;
    uint8_t qs[QK4_NL / 2];
};

//------------------------------------------------------------------------------
// bf16 to f16
//------------------------------------------------------------------------------
kernel void kernel_convert_bf16_to_f16(
    global const ushort * src,
    global half * dst,
    ulong off_dst,
    ulong n
) {
    uint i = get_global_id(0);
    if (i >= n) {
        return;
    }

    dst[i + off_dst] = (half) as_float((uint) src[i] << 16);
}

//------------------------------------------------------------------------------
// f16 to bf16
//------------------------------------------------------------------------------
kernel void kernel_convert_f16_to_bf16(
    global const half * src,
    ulong off_src,
    global ushort * dst,
    ulong n
) {
    uint i = get_global_id(0);
    if (i >= n) {
        return;
    }

    float f = (float) src[i + off_src];
    uint bits = as_uint(f);
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        // nan to quiet nan
        dst[i] = (ushort)((bits >> 16) | 0x40u);
    } else {
        uint rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
        dst[i] = (ushort)(rounded >> 16);
    }
}

//------------------------------------------------------------------------------
// kernel_convert_block_q1_0
// Convert block_q1_0 (AOS) to 2 separate arrays (SOA): quant bytes + scales.
// q1_0 bits are stored in natural order (bit j of byte i -> weight 8*i + j)
//------------------------------------------------------------------------------
kernel void kernel_convert_block_q1_0(
    global block_q1_0 * src0,
    global uchar * dst_q,
    global half  * dst_d
) {
    global block_q1_0 * b = (global block_q1_0 *) src0 + get_global_id(0);
    global uchar      * q = (global uchar *) dst_q + (QK1_0/8)*get_global_id(0);
    global half       * d = (global half *) dst_d + get_global_id(0);

    *d = b->d;

    for (int i = 0; i < QK1_0/8; ++i) {
        q[i] = b->qs[i];
    }
}

kernel void kernel_restore_block_q1_0(
    global uchar * src_q,
    global half  * src_d,
    global block_q1_0 * dst
) {
    global block_q1_0 * b = (global block_q1_0 *) dst + get_global_id(0);
    global uchar      * q = (global uchar *) src_q + (QK1_0/8)*get_global_id(0);
    global half       * d = (global half *) src_d + get_global_id(0);

    b->d = *d;
    for (int i = 0; i < QK1_0/8; ++i) {
        b->qs[i] = q[i];
    }
}

//------------------------------------------------------------------------------
// kernel_convert_block_q4_0
// Convert the block_q4_0 format to 2 separate arrays (AOS -> SOA).
// This kernel does not deshuffle the bits.
//------------------------------------------------------------------------------
kernel void kernel_convert_block_q4_0(
    global struct block_q4_0 * src0,
    global uchar * dst_q,
    global half  * dst_d
) {
    global struct block_q4_0 * b = (global struct block_q4_0 *) src0 + get_global_id(0);
    global uchar * q = (global uchar *) dst_q + QK4_0/2*get_global_id(0);
    global half  * d = (global half *) dst_d + get_global_id(0);

    *d = b->d;

    for (int i = 0; i < QK4_0/2; ++i) {
        q[i] = b->qs[i];
    }
}

kernel void kernel_restore_block_q4_0(
    global uchar * src_q,
    global half  * src_d,
    global struct block_q4_0 * dst
) {
    global struct block_q4_0 * b = (global struct block_q4_0 *) dst + get_global_id(0);
    global uchar * q = (global uchar *) src_q + QK4_0/2*get_global_id(0);
    global half  * d = (global half *) src_d + get_global_id(0);

    b->d = *d;
    for (int i = 0; i < QK4_0/2; ++i) {
        b->qs[i] = q[i];
    }
}

//------------------------------------------------------------------------------
// kernel_convert_block_q4_0_noshuffle
// Flatten q4_0 weights and unshuffle the bits
//------------------------------------------------------------------------------

kernel void kernel_convert_block_q4_0_noshuffle(
    global struct block_q4_0 * src0,
    global uchar * dst_q,
    global half  * dst_d
) {
    global struct block_q4_0 * b = (global struct block_q4_0 *) src0 + get_global_id(0);
    global uchar * q = (global uchar *) dst_q + QK4_0/2*get_global_id(0);
    global half  * d = (global half *) dst_d + get_global_id(0);

    *d = b->d;
    for (int i = 0; i < QK4_0/4; ++i) {
        uchar x0 = b->qs[2*i + 0];
        uchar x1 = b->qs[2*i + 1];

        q[i + 0      ] = convert_uchar(x0 & 0x0F) | convert_uchar((x1 & 0x0F) << 4);
        q[i + QK4_0/4] = convert_uchar((x0 & 0xF0) >> 4) | convert_uchar(x1 & 0xF0);

#ifdef ADRENO_GPU
        // Workaround for adreno - must have the following printf statement for
        // the kernel to work properly. Otherwise it produces incorrect result.
        // convert_uchar above also seems necessary.
        // Compare against a large number so that it does not print anything.
        // get_sub_group_local_id() also works.
        if (get_global_id(0) == 65536*4096) {
            printf("%04x - %02x\n", *(global ushort*)d, ((x0 & 0xF0) >> 4) | (x1 & 0xF0));
        }
#endif
    }
}

kernel void kernel_restore_block_q4_0_noshuffle(
    global uchar * src_q,
    global half  * src_d,
    global struct block_q4_0 * dst,
    uchar mask_0F,
    uchar mask_F0
) {
    global struct block_q4_0 * b = (global struct block_q4_0 *) dst + get_global_id(0);
    global uchar * q = (global uchar *) src_q + QK4_0/2*get_global_id(0);
    global half  * d = (global half *) src_d + get_global_id(0);

    b->d = *d;
    for (int i = 0; i < QK4_0/4; ++i) {
        uchar x0 = q[i + 0      ] ;
        uchar x1 = q[i + QK4_0/4];

        b->qs[2*i + 0] = convert_uchar((x0 & mask_0F) | ((x1 & mask_0F) << 4));
        b->qs[2*i + 1] = convert_uchar(((x0 & mask_F0) >> 4) | (x1 & mask_F0));
    }
}

kernel void kernel_convert_block_q4_0_trans4_ns(
    global struct block_q4_0 * src0,
    __global uint * dst_q,
    __global half * dst_d,
    uint ne00,
    uint ne01
) {
    uint i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK4_0;
    uint src_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;

    global struct block_q4_0 * b = src0 + src_blk_offset;
    dst_d[dst_blk_offset] = b->d;

    uint q0 = pack_trans4_low(b->qs, 0);
    uint q1 = pack_trans4_low(b->qs, 8);
    uint q2 = pack_trans4_high(b->qs, 0);
    uint q3 = pack_trans4_high(b->qs, 8);

    uint offset = i02 * ne00_blk * ne01 * 4 + i00 * ne01 * 4 + i01;
    dst_q[offset] = q0;
    dst_q[offset + ne01] = q1;
    dst_q[offset + ne01 * 2] = q2;
    dst_q[offset + ne01 * 3] = q3;
}

kernel void kernel_restore_block_q4_0_trans4_ns(
    __global uint * src_q,
    __global half * src_d,
    __global struct block_q4_0 * dst0,
    uint ne00,
    uint ne01
) {
    uint i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK4_0;
    uint dst_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint src_d_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;

    __global struct block_q4_0 * b = dst0 + dst_blk_offset;
    b->d = src_d[src_d_offset];

    uint src_q_offset = i02 * ne00_blk * ne01 * 4 + i00 * ne01 * 4 + i01;
    uint q0 = src_q[src_q_offset];
    uint q1 = src_q[src_q_offset + ne01];
    uint q2 = src_q[src_q_offset + ne01 * 2];
    uint q3 = src_q[src_q_offset + ne01 * 3];
    restore_trans4(b->qs, 0, q0, q2);
    restore_trans4(b->qs, 8, q1, q3);
}

//------------------------------------------------------------------------------
// kernel_convert_block_q4_1
// Convert the block_q4_1 format to 2 separate arrays (AOS -> SOA).
// This kernel does not deshuffle the bits.
//------------------------------------------------------------------------------
kernel void kernel_convert_block_q4_1(
    global struct block_q4_1 * src0,
    global uchar * dst_q,
    global half  * dst_d,
    global half  * dst_m
) {
    global struct block_q4_1 * b = (global struct block_q4_1 *) src0 + get_global_id(0);
    global uchar * q = (global uchar *) dst_q + QK4_1/2*get_global_id(0);
    global half  * d = (global half *) dst_d + get_global_id(0);
    global half  * m = (global half *) dst_m + get_global_id(0);

    *d = b->d;
    *m = b->m;

    for (int i = 0; i < QK4_1/2; ++i) {
        q[i] = b->qs[i];
    }
}

kernel void kernel_restore_block_q4_1(
    global uchar * src_q,
    global half  * src_d,
    global half  * src_m,
    global struct block_q4_1 * dst
) {
    global struct block_q4_1 * b = (global struct block_q4_1 *) dst + get_global_id(0);
    global uchar * q = (global uchar *) src_q + QK4_1/2*get_global_id(0);
    global half  * d = (global half *) src_d + get_global_id(0);
    global half  * m = (global half *) src_m + get_global_id(0);

    b->d = *d;
    b->m = *m;
    for (int i = 0; i < QK4_1/2; ++i) {
        b->qs[i] = q[i];
    }
}

kernel void kernel_convert_block_q4_1_noshuffle(
    global struct block_q4_1 * src0,
    global uchar * dst_q,
    global half  * dst_d,
    global half  * dst_m
) {
    global struct block_q4_1 * b = (global struct block_q4_1 *) src0 + get_global_id(0);
    global uchar * q = (global uchar *) dst_q + QK4_1/2*get_global_id(0);
    global half  * d = (global half *) dst_d + get_global_id(0);
    global half  * m = (global half *) dst_m + get_global_id(0);

    *d = b->d;
    *m = b->m;
    for (int i = 0; i < QK4_1/4; ++i) {
        uchar x0 = b->qs[2*i + 0];
        uchar x1 = b->qs[2*i + 1];

        q[i + 0      ] = convert_uchar(x0 & 0x0F) | convert_uchar((x1 & 0x0F) << 4);
        q[i + QK4_1/4] = convert_uchar((x0 & 0xF0) >> 4) | convert_uchar(x1 & 0xF0);

#ifdef ADRENO_GPU
        if (get_global_id(0) == 65536*4096) {
            printf("%04x - %02x\n", *(global ushort*)d, ((x0 & 0xF0) >> 4) | (x1 & 0xF0));
        }
#endif
    }
}

kernel void kernel_restore_block_q4_1_noshuffle(
    global uchar * src_q,
    global half  * src_d,
    global half  * src_m,
    global struct block_q4_1 * dst,
    uchar mask_0F,
    uchar mask_F0
) {
    global struct block_q4_1 * b = (global struct block_q4_1 *) dst + get_global_id(0);
    global uchar * q = (global uchar *) src_q + QK4_1/2*get_global_id(0);
    global half  * d = (global half *) src_d + get_global_id(0);
    global half  * m = (global half *) src_m + get_global_id(0);

    b->d = *d;
    b->m = *m;
    for (int i = 0; i < QK4_1/4; ++i) {
        uchar x0 = q[i + 0      ] ;
        uchar x1 = q[i + QK4_1/4];

        b->qs[2*i + 0] = convert_uchar((x0 & mask_0F) | ((x1 & mask_0F) << 4));
        b->qs[2*i + 1] = convert_uchar(((x0 & mask_F0) >> 4) | (x1 & mask_F0));
    }
}

kernel void kernel_convert_block_q4_1_trans4_ns(
    __global struct block_q4_1 * src0,
    __global uint * dst_q,
    __global half * dst_d,
    __global half * dst_m,
    uint ne00,
    uint ne01
) {
    uint i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK4_1;
    uint src_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;

    global struct block_q4_1 * b = src0 + src_blk_offset;
    dst_d[dst_blk_offset] = b->d;
    dst_m[dst_blk_offset] = b->m;

    uint q0 = pack_trans4_low(b->qs, 0);
    uint q1 = pack_trans4_low(b->qs, 8);
    uint q2 = pack_trans4_high(b->qs, 0);
    uint q3 = pack_trans4_high(b->qs, 8);

    uint offset = i02 * ne00_blk * ne01 * 4 + i00 * ne01 * 4 + i01;
    dst_q[offset] = q0;
    dst_q[offset + ne01] = q1;
    dst_q[offset + ne01 * 2] = q2;
    dst_q[offset + ne01 * 3] = q3;
}

kernel void kernel_restore_block_q4_1_trans4_ns(
    __global uint * src_q,
    __global half * src_d,
    __global half * src_m,
    __global struct block_q4_1 * dst0,
    uint ne00,
    uint ne01
) {
    int i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK4_1;
    uint dst_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint src_dm_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;

    __global struct block_q4_1 * b = dst0 + dst_blk_offset;
    b->d = src_d[src_dm_offset];
    b->m = src_m[src_dm_offset];

    uint src_q_offset = i02 * ne00_blk * ne01 * 4 + i00 * ne01 * 4 + i01;
    uint q0 = src_q[src_q_offset];
    uint q1 = src_q[src_q_offset + ne01];
    uint q2 = src_q[src_q_offset + ne01 * 2];
    uint q3 = src_q[src_q_offset + ne01 * 3];
    restore_trans4(b->qs, 0, q0, q2);
    restore_trans4(b->qs, 8, q1, q3);
}

//------------------------------------------------------------------------------
// kernel_convert_block_q5_0
// Convert the block_q5_0 format to 3 separate arrays (AOS -> SOA).
// This kernel does not deshuffle the bits.
//------------------------------------------------------------------------------
kernel void kernel_convert_block_q5_0(
    global struct block_q5_0 * src0,
    global uchar * dst_qs,
    global uint  * dst_qh,
    global half  * dst_d,
    ulong n_blk
) {
    if (get_global_id(0) >= n_blk) {
        return;
    }

    global struct block_q5_0 * b = (global struct block_q5_0 *) src0 + get_global_id(0);
    global uchar * qs = (global uchar *) dst_qs + (QK5_0/2)*get_global_id(0);
    global uint  * qh = (global uint  *) dst_qh + get_global_id(0);
    global half  * d  = (global half  *) dst_d  + get_global_id(0);

    *d = b->d;
    *qh = *((global uint *)(b->qh));

    for (int i = 0; i < QK5_0/2; ++i) {
        qs[i] = b->qs[i];
    }
}

kernel void kernel_restore_block_q5_0(
    global uchar * src_qs,
    global uint  * src_qh,
    global half  * src_d,
    global struct block_q5_0 * dst
) {
    global struct block_q5_0 * b = (global struct block_q5_0 *) dst + get_global_id(0);
    global uchar * qs = (global uchar *) src_qs + (QK5_0/2)*get_global_id(0);
    global uint  * qh = (global uint  *) src_qh + get_global_id(0);
    global half  * d  = (global half  *) src_d  + get_global_id(0);

    b->d = *d;
    *((global uint *)(b->qh)) = *qh;
    for (int i = 0; i < QK5_0/2; ++i) {
        b->qs[i] = qs[i];
    }
}

kernel void kernel_convert_block_q5_0_noshuffle(
    global struct block_q5_0 * src0,
    global uchar * dst_q,
    global uint  * dst_qh,
    global half  * dst_d
) {
    global struct block_q5_0 * b = (global struct block_q5_0 *) src0 + get_global_id(0);
    global uchar * q  = (global uchar *) dst_q + QK5_0/2*get_global_id(0);
    global uint  * qh = (global uint  *) dst_qh + get_global_id(0);
    global half  * d  = (global half  *) dst_d  + get_global_id(0);

    *d = b->d;
    *qh = *((global uint *)(b->qh));

    for (int i = 0; i < QK5_0/4; ++i) {
        uchar x0 = b->qs[2*i + 0];
        uchar x1 = b->qs[2*i + 1];

        q[i + 0      ] = convert_uchar(x0 & 0x0F) | convert_uchar((x1 & 0x0F) << 4);
        q[i + QK5_0/4] = convert_uchar((x0 & 0xF0) >> 4) | convert_uchar(x1 & 0xF0);

#ifdef ADRENO_GPU
        if (get_global_id(0) == 65536*4096) {
            printf("%04x - %02x\n", *(global ushort*)d, ((x0 & 0xF0) >> 4) | (x1 & 0xF0));
        }
#endif
    }
}

kernel void kernel_restore_block_q5_0_noshuffle(
    global uchar * src_q,
    global uint  * src_qh,
    global half  * src_d,
    global struct block_q5_0 * dst,
    uchar mask_0F,
    uchar mask_F0
) {
    global struct block_q5_0 * b = (global struct block_q5_0 *) dst + get_global_id(0);
    global uchar * q  = (global uchar *) src_q + QK5_0/2*get_global_id(0);
    global uint  * qh = (global uint  *) src_qh + get_global_id(0);
    global half  * d  = (global half  *) src_d  + get_global_id(0);

    b->d = *d;
    *((global uint *)(b->qh)) = *qh;

    for (int i = 0; i < QK5_0/4; ++i) {
        uchar x0 = q[i + 0      ];
        uchar x1 = q[i + QK5_0/4];

        b->qs[2*i + 0] = convert_uchar((x0 & mask_0F) | ((x1 & mask_0F) << 4));
        b->qs[2*i + 1] = convert_uchar(((x0 & mask_F0) >> 4) | (x1 & mask_F0));
    }
}

kernel void kernel_convert_block_q5_0_trans4_ns(
    __global struct block_q5_0 * src0,
    __global uint * dst_qs,
    __global uint * dst_qh,
    __global half * dst_d,
    uint ne00,
    uint ne01
) {
    uint i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK5_0;
    uint src_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;

    global struct block_q5_0 * b = src0 + src_blk_offset;
    dst_d[dst_blk_offset] = b->d;

    dst_qh[dst_blk_offset] = pack_uchar4(b->qh[0], b->qh[1], b->qh[2], b->qh[3]);

    uint q0 = pack_trans4_low(b->qs, 0);
    uint q1 = pack_trans4_low(b->qs, 8);
    uint q2 = pack_trans4_high(b->qs, 0);
    uint q3 = pack_trans4_high(b->qs, 8);

    uint offset = i02 * ne00_blk * ne01 * 4 + i00 * ne01 * 4 + i01;
    dst_qs[offset] = q0;
    dst_qs[offset + ne01] = q1;
    dst_qs[offset + ne01 * 2] = q2;
    dst_qs[offset + ne01 * 3] = q3;
}

kernel void kernel_restore_block_q5_0_trans4_ns(
    __global uint * src_qs,
    __global uint * src_qh,
    __global half * src_d,
    __global struct block_q5_0 * dst0,
    uint ne00,
    uint ne01
) {
    int i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK5_0;
    uint dst_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint src_blk_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;

    __global struct block_q5_0 * b = dst0 + dst_blk_offset;
    b->d = src_d[src_blk_offset];

    uint qh = src_qh[src_blk_offset];
    b->qh[0] = unpack_uchar0(qh);
    b->qh[1] = unpack_uchar1(qh);
    b->qh[2] = unpack_uchar2(qh);
    b->qh[3] = unpack_uchar3(qh);

    uint src_q_offset = i02 * ne00_blk * ne01 * 4 + i00 * ne01 * 4 + i01;
    uint q0 = src_qs[src_q_offset];
    uint q1 = src_qs[src_q_offset + ne01];
    uint q2 = src_qs[src_q_offset + ne01 * 2];
    uint q3 = src_qs[src_q_offset + ne01 * 3];
    restore_trans4(b->qs, 0, q0, q2);
    restore_trans4(b->qs, 8, q1, q3);
}

//------------------------------------------------------------------------------
// kernel_convert_block_q5_1
// Convert the block_q5_1 format to 4 separate arrays (AOS -> SOA).
// This kernel does not deshuffle the bits.
//------------------------------------------------------------------------------
kernel void kernel_convert_block_q5_1(
    global struct block_q5_1 * src0,
    global uchar * dst_qs,
    global uint  * dst_qh,
    global half  * dst_d,
    global half  * dst_m,
    ulong n_blk
) {
    if (get_global_id(0) >= n_blk) {
        return;
    }

    global struct block_q5_1 * b = (global struct block_q5_1 *) src0 + get_global_id(0);
    global uchar * qs = (global uchar *) dst_qs + (QK5_1/2)*get_global_id(0);
    global uint  * qh = (global uint  *) dst_qh + get_global_id(0);
    global half  * d  = (global half  *) dst_d  + get_global_id(0);
    global half  * m  = (global half  *) dst_m  + get_global_id(0);

    *d = b->d;
    *m = b->m;
    *qh = *((global uint *)(b->qh));

    for (int i = 0; i < QK5_1/2; ++i) {
        qs[i] = b->qs[i];
    }
}

kernel void kernel_restore_block_q5_1(
    global uchar * src_qs,
    global uint  * src_qh,
    global half  * src_d,
    global half  * src_m,
    global struct block_q5_1 * dst
) {
    global struct block_q5_1 * b = (global struct block_q5_1 *) dst + get_global_id(0);
    global uchar * qs = (global uchar *) src_qs + (QK5_1/2)*get_global_id(0);
    global uint  * qh = (global uint  *) src_qh + get_global_id(0);
    global half  * d  = (global half  *) src_d  + get_global_id(0);
    global half  * m  = (global half  *) src_m  + get_global_id(0);

    b->d = *d;
    b->m = *m;
    *((global uint *)(b->qh)) = *qh;
    for (int i = 0; i < QK5_1/2; ++i) {
        b->qs[i] = qs[i];
    }
}

kernel void kernel_convert_block_q5_1_noshuffle(
    global struct block_q5_1 * src0,
    global uchar * dst_q,
    global uint  * dst_qh,
    global half  * dst_d,
    global half  * dst_m
) {
    global struct block_q5_1 * b = (global struct block_q5_1 *) src0 + get_global_id(0);
    global uchar * q  = (global uchar *) dst_q + QK5_1/2*get_global_id(0);
    global uint  * qh = (global uint  *) dst_qh + get_global_id(0);
    global half  * d  = (global half  *) dst_d  + get_global_id(0);
    global half  * m  = (global half  *) dst_m  + get_global_id(0);

    *d = b->d;
    *m = b->m;
    *qh = *((global uint *)(b->qh));

    for (int i = 0; i < QK5_1/4; ++i) {
        uchar x0 = b->qs[2*i + 0];
        uchar x1 = b->qs[2*i + 1];

        q[i + 0      ] = convert_uchar(x0 & 0x0F) | convert_uchar((x1 & 0x0F) << 4);
        q[i + QK5_1/4] = convert_uchar((x0 & 0xF0) >> 4) | convert_uchar(x1 & 0xF0);

#ifdef ADRENO_GPU
        if (get_global_id(0) == 65536*4096) {
            printf("%04x - %02x\n", *(global ushort*)d, ((x0 & 0xF0) >> 4) | (x1 & 0xF0));
        }
#endif
    }
}

kernel void kernel_restore_block_q5_1_noshuffle(
    global uchar * src_q,
    global uint  * src_qh,
    global half  * src_d,
    global half  * src_m,
    global struct block_q5_1 * dst,
    uchar mask_0F,
    uchar mask_F0
) {
    global struct block_q5_1 * b = (global struct block_q5_1 *) dst + get_global_id(0);
    global uchar * q  = (global uchar *) src_q + QK5_1/2*get_global_id(0);
    global uint  * qh = (global uint  *) src_qh + get_global_id(0);
    global half  * d  = (global half  *) src_d  + get_global_id(0);
    global half  * m  = (global half  *) src_m  + get_global_id(0);

    b->d = *d;
    b->m = *m;
    *((global uint *)(b->qh)) = *qh;

    for (int i = 0; i < QK5_1/4; ++i) {
        uchar x0 = q[i + 0      ];
        uchar x1 = q[i + QK5_1/4];

        b->qs[2*i + 0] = convert_uchar((x0 & mask_0F) | ((x1 & mask_0F) << 4));
        b->qs[2*i + 1] = convert_uchar(((x0 & mask_F0) >> 4) | (x1 & mask_F0));
    }
}

kernel void kernel_convert_block_q5_1_trans4_ns(
    __global struct block_q5_1 * src0,
    __global uint * dst_qs,
    __global uint * dst_qh,
    __global half * dst_d,
    __global half * dst_m,
    uint ne00,
    uint ne01
) {
    uint i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK5_1;
    uint src_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;

    global struct block_q5_1 * b = src0 + src_blk_offset;
    dst_d[dst_blk_offset] = b->d;
    dst_m[dst_blk_offset] = b->m;

    dst_qh[dst_blk_offset] = pack_uchar4(b->qh[0], b->qh[1], b->qh[2], b->qh[3]);

    uint q0 = pack_trans4_low(b->qs, 0);
    uint q1 = pack_trans4_low(b->qs, 8);
    uint q2 = pack_trans4_high(b->qs, 0);
    uint q3 = pack_trans4_high(b->qs, 8);

    uint offset = i02 * ne00_blk * ne01 * 4 + i00 * ne01 * 4 + i01;
    dst_qs[offset] = q0;
    dst_qs[offset + ne01] = q1;
    dst_qs[offset + ne01 * 2] = q2;
    dst_qs[offset + ne01 * 3] = q3;
}

kernel void kernel_restore_block_q5_1_trans4_ns(
    __global uint * src_qs,
    __global uint * src_qh,
    __global half * src_d,
    __global half * src_m,
    __global struct block_q5_1 * dst0,
    uint ne00,
    uint ne01
) {
    int i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK5_1;
    uint dst_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint src_blk_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;

    __global struct block_q5_1 * b = dst0 + dst_blk_offset;
    b->d = src_d[src_blk_offset];
    b->m = src_m[src_blk_offset];

    uint qh = src_qh[src_blk_offset];
    b->qh[0] = unpack_uchar0(qh);
    b->qh[1] = unpack_uchar1(qh);
    b->qh[2] = unpack_uchar2(qh);
    b->qh[3] = unpack_uchar3(qh);

    uint src_q_offset = i02 * ne00_blk * ne01 * 4 + i00 * ne01 * 4 + i01;
    uint q0 = src_qs[src_q_offset];
    uint q1 = src_qs[src_q_offset + ne01];
    uint q2 = src_qs[src_q_offset + ne01 * 2];
    uint q3 = src_qs[src_q_offset + ne01 * 3];
    restore_trans4(b->qs, 0, q0, q2);
    restore_trans4(b->qs, 8, q1, q3);
}

kernel void kernel_convert_block_q4_k_trans4_ns(
    __global struct block_q4_K * src0,
    __global uint  * dst_q,
    __global half  * dst_d,
    __global half  * dst_dm,
    __global uchar * dst_s,
    uint ne00,
    uint ne01,
    uchar mask_0F,
    uchar mask_F0
) {
    uint i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK_K;
    uint src_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i01 + i00 * ne01     + i02 * ne00_blk * ne01;

    __global struct block_q4_K * b = src0 + src_blk_offset;

    dst_d [dst_blk_offset] = b->d;
    dst_dm[dst_blk_offset] = b->dm;

    for (int i = 0; i < QK_K / 64; ++i) {
        uint src_offset = i * 32;
        uint dst_offset = i * 8;
        uint base = i02 * ne00_blk * ne01 * 32 + i00 * ne01 * 32 + i01;
        dst_q[base + (dst_offset + 0) * ne01] = pack_trans4_low(b->q, src_offset + 0);
        dst_q[base + (dst_offset + 1) * ne01] = pack_trans4_low(b->q, src_offset + 8);
        dst_q[base + (dst_offset + 2) * ne01] = pack_trans4_low(b->q, src_offset + 16);
        dst_q[base + (dst_offset + 3) * ne01] = pack_trans4_low(b->q, src_offset + 24);
        dst_q[base + (dst_offset + 4) * ne01] = pack_trans4_high(b->q, src_offset + 0);
        dst_q[base + (dst_offset + 5) * ne01] = pack_trans4_high(b->q, src_offset + 8);
        dst_q[base + (dst_offset + 6) * ne01] = pack_trans4_high(b->q, src_offset + 16);
        dst_q[base + (dst_offset + 7) * ne01] = pack_trans4_high(b->q, src_offset + 24);
    }

    __global uchar * s_dst = dst_s + (i02 * ne01 + i01) * ne00_blk * K_SCALE_SIZE + i00 * K_SCALE_SIZE;
    #pragma unroll
    for (int i = 0; i < K_SCALE_SIZE; ++i) {
        s_dst[i] = b->s[i];
    }
}

kernel void kernel_restore_block_q4_k_trans4_ns(
    __global uint  * src_q,
    __global half  * src_d,
    __global half  * src_dm,
    __global uchar * src_s,
    __global struct block_q4_K * dst0,
    uint ne00,
    uint ne01,
    uchar mask_0F,
    uchar mask_F0
) {
    uint i00 = get_global_id(1);  // block index along K
    uint i01 = get_global_id(0);  // row index
    uint i02 = get_global_id(2);  // batch index

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK_K;

    uint src_blk_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;

    __global struct block_q4_K * b = dst0 + dst_blk_offset;

    b->d  = src_d[src_blk_offset];
    b->dm = src_dm[src_blk_offset];

    __global uchar * s_src = src_s + (i02 * ne01 + i01) * ne00_blk * K_SCALE_SIZE + i00 * K_SCALE_SIZE;
    for (int i = 0; i < K_SCALE_SIZE; ++i) {
        b->s[i] = s_src[i];
    }

    uint base = i02 * ne00_blk * ne01 * 32 + i00 * ne01 * 32 + i01;

    for (int i = 0; i < QK_K / 64; ++i) {
        uint dst_offset = i * 32;
        uint src_offset = i * 8;
        restore_trans4(b->q, dst_offset + 0,
            src_q[base + (src_offset + 0) * ne01], src_q[base + (src_offset + 4) * ne01]);
        restore_trans4(b->q, dst_offset + 8,
            src_q[base + (src_offset + 1) * ne01], src_q[base + (src_offset + 5) * ne01]);
        restore_trans4(b->q, dst_offset + 16,
            src_q[base + (src_offset + 2) * ne01], src_q[base + (src_offset + 6) * ne01]);
        restore_trans4(b->q, dst_offset + 24,
            src_q[base + (src_offset + 3) * ne01], src_q[base + (src_offset + 7) * ne01]);
    }
}

kernel void kernel_convert_block_q5_k_trans4_ns(
    __global struct block_q5_K * src0,
    __global uint  * dst_qs,
    __global uint  * dst_qh,
    __global half  * dst_d,
    __global half  * dst_dm,
    __global uchar * dst_s,
    uint ne00,
    uint ne01,
    uchar mask_0F,
    uchar mask_F0
) {
    uint i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK_K;
    uint src_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i01 + i00 * ne01     + i02 * ne00_blk * ne01;

    __global struct block_q5_K * b = src0 + src_blk_offset;

    dst_d [dst_blk_offset] = b->d;
    dst_dm[dst_blk_offset] = b->dm;

    for (int k = 0; k < 8; k++) {
        uchar b0 = 0, b1 = 0, b2 = 0, b3 = 0;
        for (int bit = 0; bit < 8; bit++) {
            b0 |= (uchar)(((b->qh[bit]      >> k) & 1) << bit);
            b1 |= (uchar)(((b->qh[8  + bit] >> k) & 1) << bit);
            b2 |= (uchar)(((b->qh[16 + bit] >> k) & 1) << bit);
            b3 |= (uchar)(((b->qh[24 + bit] >> k) & 1) << bit);
        }
        uint packed = (uint)b0 | ((uint)b1 << 8) | ((uint)b2 << 16) | ((uint)b3 << 24);
        dst_qh[i01 + (i00 * 8 + k) * ne01 + i02 * ne00_blk * 8 * ne01] = packed;
    }

    for (int i = 0; i < QK_K / 64; ++i) {
        uint src_offset = i * 32;
        uint dst_offset = i * 8;
        uint base = i02 * ne00_blk * ne01 * 32 + i00 * ne01 * 32 + i01;
        dst_qs[base + (dst_offset + 0) * ne01] = pack_trans4_low(b->qs, src_offset + 0);
        dst_qs[base + (dst_offset + 1) * ne01] = pack_trans4_low(b->qs, src_offset + 8);
        dst_qs[base + (dst_offset + 2) * ne01] = pack_trans4_low(b->qs, src_offset + 16);
        dst_qs[base + (dst_offset + 3) * ne01] = pack_trans4_low(b->qs, src_offset + 24);
        dst_qs[base + (dst_offset + 4) * ne01] = pack_trans4_high(b->qs, src_offset + 0);
        dst_qs[base + (dst_offset + 5) * ne01] = pack_trans4_high(b->qs, src_offset + 8);
        dst_qs[base + (dst_offset + 6) * ne01] = pack_trans4_high(b->qs, src_offset + 16);
        dst_qs[base + (dst_offset + 7) * ne01] = pack_trans4_high(b->qs, src_offset + 24);
    }

    __global uchar * s_dst = dst_s + (i02 * ne01 + i01) * ne00_blk * K_SCALE_SIZE + i00 * K_SCALE_SIZE;
    #pragma unroll
    for (int i = 0; i < K_SCALE_SIZE; ++i) {
        s_dst[i] = b->s[i];
    }
}

kernel void kernel_restore_block_q5_k_trans4_ns(
    __global uint  * src_qs,
    __global uint  * src_qh,
    __global half  * src_d,
    __global half  * src_dm,
    __global uchar * src_s,
    __global struct block_q5_K * dst0,
    uint ne00,
    uint ne01,
    uchar mask_0F,
    uchar mask_F0
) {
    uint i00 = get_global_id(1);  // block index along K
    uint i01 = get_global_id(0);  // row index
    uint i02 = get_global_id(2);  // batch index

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK_K;

    uint src_blk_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;

    __global struct block_q5_K * b = dst0 + dst_blk_offset;

    b->d  = src_d[src_blk_offset];
    b->dm = src_dm[src_blk_offset];

    for (int j = 0; j < 32; j++) b->qh[j] = 0;
    for (int k = 0; k < 8; k++) {
        uint packed = src_qh[i01 + (i00 * 8 + k) * ne01 + i02 * ne00_blk * 8 * ne01];
        uchar b0 = (uchar)(packed & 0xFF);
        uchar b1 = (uchar)((packed >> 8) & 0xFF);
        uchar b2 = (uchar)((packed >> 16) & 0xFF);
        uchar b3 = (uchar)((packed >> 24) & 0xFF);
        for (int bit = 0; bit < 8; bit++) {
            b->qh[bit]      |= (uchar)(((b0 >> bit) & 1) << k);
            b->qh[8  + bit] |= (uchar)(((b1 >> bit) & 1) << k);
            b->qh[16 + bit] |= (uchar)(((b2 >> bit) & 1) << k);
            b->qh[24 + bit] |= (uchar)(((b3 >> bit) & 1) << k);
        }
    }

    __global uchar * s_src = src_s + (i02 * ne01 + i01) * ne00_blk * K_SCALE_SIZE + i00 * K_SCALE_SIZE;
    for (int i = 0; i < K_SCALE_SIZE; ++i) {
        b->s[i] = s_src[i];
    }

    uint base = i02 * ne00_blk * ne01 * 32 + i00 * ne01 * 32 + i01;

    for (int i = 0; i < QK_K / 64; ++i) {
        uint dst_offset = i * 32;
        uint src_offset = i * 8;
        restore_trans4(b->qs, dst_offset + 0,
            src_qs[base + (src_offset + 0) * ne01], src_qs[base + (src_offset + 4) * ne01]);
        restore_trans4(b->qs, dst_offset + 8,
            src_qs[base + (src_offset + 1) * ne01], src_qs[base + (src_offset + 5) * ne01]);
        restore_trans4(b->qs, dst_offset + 16,
            src_qs[base + (src_offset + 2) * ne01], src_qs[base + (src_offset + 6) * ne01]);
        restore_trans4(b->qs, dst_offset + 24,
            src_qs[base + (src_offset + 3) * ne01], src_qs[base + (src_offset + 7) * ne01]);
    }
}

kernel void kernel_convert_block_q6_k_trans4_ns(
    __global struct block_q6_K * src0,
    __global uint  * dst_ql,
    __global uint  * dst_qh,
    __global half  * dst_d,
    __global char  * dst_s,
    uint ne00,
    uint ne01,
    uchar mask_0F,
    uchar mask_F0
) {
    uint i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK_K;

    uint src_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i01 + i00 * ne01     + i02 * ne00_blk * ne01;

    __global struct block_q6_K * b = src0 + src_blk_offset;

    dst_d[dst_blk_offset] = b->d;

    uint ql_base = i02 * ne00_blk * ne01 * 32 + i00 * ne01 * 32 + i01;
    for (int i = 0; i < 2; ++i) {
        uint src_offset = i * 64;
        uint dst_offset = i * 16;
        for (int j = 0; j < 4; ++j) {
            uint chunk_offset = j * 8;
            dst_ql[ql_base + (dst_offset + j) * ne01] =
                pack_trans4_low(b->ql, src_offset + chunk_offset);
            dst_ql[ql_base + (dst_offset + 4 + j) * ne01] =
                pack_trans4_low(b->ql, src_offset + 32 + chunk_offset);
            dst_ql[ql_base + (dst_offset + 8 + j) * ne01] =
                pack_trans4_high(b->ql, src_offset + chunk_offset);
            dst_ql[ql_base + (dst_offset + 12 + j) * ne01] =
                pack_trans4_high(b->ql, src_offset + 32 + chunk_offset);
        }
    }

    uint qhv[16] = {0};

    for (int n = 0; n < 2; ++n) {
        for (int l = 0; l < 32; ++l) {
            uchar h = b->qh[n*32 + l];
            int u = l / 16;
            int bit_pos = (l % 16) * 2;
            qhv[(n*4 + 0)*2 + u] |= ((uint)((h >> 0) & 0x03)) << bit_pos;
            qhv[(n*4 + 1)*2 + u] |= ((uint)((h >> 2) & 0x03)) << bit_pos;
            qhv[(n*4 + 2)*2 + u] |= ((uint)((h >> 4) & 0x03)) << bit_pos;
            qhv[(n*4 + 3)*2 + u] |= ((uint)((h >> 6) & 0x03)) << bit_pos;
        }
    }

    uint qh_base = i02 * ne00_blk * ne01 * 16 + i00 * ne01 * 16 + i01;

    for (int p = 0; p < 16; ++p) {
        dst_qh[qh_base + p * ne01] = qhv[p];
    }

    __global char * s_dst = dst_s + (i02 * ne01 + i01) * ne00_blk * 16 + i00 * 16;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        s_dst[i] = b->scales[i];
    }
}

kernel void kernel_restore_block_q6_k_trans4_ns(
    __global uint  * src_ql,
    __global uint  * src_qh,
    __global half  * src_d,
    __global char  * src_s,
    __global struct block_q6_K * dst0,
    uint ne00,
    uint ne01,
    uchar mask_0F,
    uchar mask_F0
) {
    uint i00 = get_global_id(1);  // block index along K
    uint i01 = get_global_id(0);  // row index
    uint i02 = get_global_id(2);  // batch index

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK_K;

    uint src_blk_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;

    __global struct block_q6_K * b = dst0 + dst_blk_offset;

    b->d = src_d[src_blk_offset];

    uint ql_base = i02 * ne00_blk * ne01 * 32 + i00 * ne01 * 32 + i01;
    for (int i = 0; i < 2; ++i) {
        uint dst_offset = i * 64;
        uint src_offset = i * 16;
        for (int j = 0; j < 4; ++j) {
            restore_trans4(b->ql, dst_offset + j * 8,
                src_ql[ql_base + (src_offset + j) * ne01],
                src_ql[ql_base + (src_offset + 8 + j) * ne01]);
            restore_trans4(b->ql, dst_offset + 32 + j * 8,
                src_ql[ql_base + (src_offset + 4 + j) * ne01],
                src_ql[ql_base + (src_offset + 12 + j) * ne01]);
        }
    }

    uint qh_base = i02 * ne00_blk * ne01 * 16 + i00 * ne01 * 16 + i01;
    uint qhv[16];
    for (int p = 0; p < 16; ++p) {
        qhv[p] = src_qh[qh_base + p * ne01];
    }

    for (int n = 0; n < 2; ++n) {
        for (int l = 0; l < 32; ++l) {
            int u = l / 16;
            int bit_pos = (l % 16) * 2;
            uchar v0 = (uchar)((qhv[(n*4 + 0)*2 + u] >> bit_pos) & 0x03);
            uchar v1 = (uchar)((qhv[(n*4 + 1)*2 + u] >> bit_pos) & 0x03);
            uchar v2 = (uchar)((qhv[(n*4 + 2)*2 + u] >> bit_pos) & 0x03);
            uchar v3 = (uchar)((qhv[(n*4 + 3)*2 + u] >> bit_pos) & 0x03);
            b->qh[n*32 + l] = v0 | (v1 << 2) | (v2 << 4) | (v3 << 6);
        }
    }

    __global char * s_src = src_s + (i02 * ne01 + i01) * ne00_blk * 16 + i00 * 16;
    for (int i = 0; i < 16; ++i) {
        b->scales[i] = s_src[i];
    }
}

//------------------------------------------------------------------------------
// block_mxfp4
//------------------------------------------------------------------------------
#define QK_MXFP4 32
struct block_mxfp4 {
    uchar e; // E8M0
    uchar qs[QK_MXFP4 / 2];
};

//------------------------------------------------------------------------------
// kernel_convert_block_mxfp4
// Convert the block_mxfp4 format to 2 separate arrays (AOS -> SOA).
// This kernel does not deshuffle the bits.
//------------------------------------------------------------------------------
kernel void kernel_convert_block_mxfp4(
    global struct block_mxfp4 * src0,
    global uchar * dst_q,
    global uchar * dst_e
) {
    global struct block_mxfp4 * b = (global struct block_mxfp4 *) src0 + get_global_id(0);
    global uchar * q = (global uchar *) dst_q + QK_MXFP4 / 2 * get_global_id(0);
    global uchar * e = (global uchar *) dst_e + get_global_id(0);

    *e = b->e;

    for (int i = 0; i < QK_MXFP4 / 2; ++i) {
        q[i] = b->qs[i];
    }
}

kernel void kernel_convert_block_mxfp4_trans(
    global struct block_mxfp4 * src0,
    __global uint4 * dst_q,
    __global uchar * dst_e,
    uint ne00,
    uint ne01
) {
    int i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    uint ne00_blk = ne00 / QK_MXFP4;
    uint src_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;

    global struct block_mxfp4 * b = src0 + src_blk_offset;

    dst_q[dst_blk_offset] = ((global uint4 *)(&(b->qs[0])))[0];
    dst_e[dst_blk_offset] = b->e;
}

kernel void kernel_restore_block_mxfp4(
    global uchar * src_q,
    global half  * src_e,
    global struct block_mxfp4 * dst
) {
    global struct block_mxfp4 * b = (global struct block_mxfp4 *) dst + get_global_id(0);
    global uchar * q = (global uchar *) src_q + QK_MXFP4 / 2 * get_global_id(0);
    global uchar * e = (global uchar *) src_e + get_global_id(0);

    b->e = *e;
    for (int i = 0; i < QK_MXFP4 / 2; ++i) {
        b->qs[i] = q[i];
    }
}

kernel void kernel_restore_block_mxfp4_trans(
    __global uint4 * src_q,
    __global uchar * src_e,
    global struct block_mxfp4 * dst,
    uint ne00,
    uint ne01
) {
    int i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    uint ne00_blk = ne00 / QK_MXFP4;
    uint src_blk_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;

    global struct block_mxfp4 * b = dst + dst_blk_offset;

    ((global uint4 *)(&(b->qs[0])))[0] = src_q[src_blk_offset];
    b->e = src_e[src_blk_offset];
}

kernel void kernel_convert_block_mxfp4_trans4_ns(
    global struct block_mxfp4 * src0,
    __global uint * dst_q,
    __global uchar * dst_e,
    uint ne00,
    uint ne01
) {
    uint i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK_MXFP4;
    uint src_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint dst_blk_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;

    global struct block_mxfp4 * b = src0 + src_blk_offset;
    dst_e[dst_blk_offset] = b->e;

    uint q0 = pack_trans4_low(b->qs, 0);
    uint q1 = pack_trans4_low(b->qs, 8);
    uint q2 = pack_trans4_high(b->qs, 0);
    uint q3 = pack_trans4_high(b->qs, 8);

    uint offset = i02 * ne00_blk * ne01 * 4 + i00 * ne01 * 4 + i01;
    dst_q[offset] = q0;
    dst_q[offset + ne01] = q1;
    dst_q[offset + ne01 * 2] = q2;
    dst_q[offset + ne01 * 3] = q3;
}

kernel void kernel_restore_block_mxfp4_trans4_ns(
    __global uint * src_q,
    __global uchar * src_e,
    __global struct block_mxfp4 * dst0,
    uint ne00,
    uint ne01
) {
    uint i00 = get_global_id(1);
    uint i01 = get_global_id(0);
    uint i02 = get_global_id(2);

    if (i01 >= ne01) {
        return;
    }

    uint ne00_blk = ne00 / QK_MXFP4;
    uint dst_blk_offset = i00 + i01 * ne00_blk + i02 * ne00_blk * ne01;
    uint src_d_offset = i01 + i00 * ne01 + i02 * ne00_blk * ne01;

    __global struct block_mxfp4 * b = dst0 + dst_blk_offset;
    b->e = src_e[src_d_offset];

    uint src_q_offset = i02 * ne00_blk * ne01 * 4 + i00 * ne01 * 4 + i01;
    uint q0 = src_q[src_q_offset];
    uint q1 = src_q[src_q_offset + ne01];
    uint q2 = src_q[src_q_offset + ne01 * 2];
    uint q3 = src_q[src_q_offset + ne01 * 3];
    restore_trans4(b->qs, 0, q0, q2);
    restore_trans4(b->qs, 8, q1, q3);
}


//------------------------------------------------------------------------------
// block_q8_0
//------------------------------------------------------------------------------
typedef struct {
    half d;       // delta
    char qs[QK8_0]; // quants
} block_q8_0;

kernel void kernel_convert_block_q8_0(
    global block_q8_0 * src0,
    global uchar * dst_q,
    global half  * dst_d
) {
    global block_q8_0 * b = (global block_q8_0 *) src0 + get_global_id(0);
    global uchar      * q = (global uchar *) dst_q + QK8_0*get_global_id(0);
    global half       * d = (global half *) dst_d + get_global_id(0);

    *d = b->d;

    for (int i = 0; i < QK8_0; ++i) {
        q[i] = b->qs[i];
    }
}

kernel void kernel_restore_block_q8_0(
    global uchar * src_q,
    global half  * src_d,
    global block_q8_0 * dst
) {
    global block_q8_0 * b = (global block_q8_0 *) dst + get_global_id(0);
    global uchar      * q = (global uchar *) src_q + QK8_0*get_global_id(0);
    global half       * d = (global half *) src_d + get_global_id(0);

    b->d = *d;
    for (int i = 0; i < QK8_0; ++i) {
        b->qs[i] = q[i];
    }
}

// View-aware AoS q8_0 -> f32 dequant (f32/f32 FA path).
kernel void kernel_dequant_q8_0_f32_view_aos(
    global char * src,
    ulong         src_offset,
    ulong         src_nb1,
    ulong         src_nb2,
    ulong         src_nb3,
    int           nblk0,
    int           ne1,
    int           ne2,
    int           ne3,
    global float * dst
) {
    int blk_i0 = get_global_id(0);
    int i1     = get_global_id(1);
    int batch  = get_global_id(2);

    if (blk_i0 >= nblk0) return;
    if (i1     >= ne1)   return;

    int i2 = batch % ne2;
    int i3 = batch / ne2;
    if (i3 >= ne3) return;

    global char * block = src + src_offset + (ulong)i3*src_nb3 + (ulong)i2*src_nb2 + (ulong)i1*src_nb1 + (ulong)blk_i0 * (2 + QK8_0);
    float d = vload_half(0, (global half *)block);
    global char * qs = block + 2;

    ulong dst_row_base = ((ulong)i3 * ne2 * ne1 + (ulong)i2 * ne1 + (ulong)i1) * nblk0;
    global float * out = dst + (dst_row_base + blk_i0) * QK8_0;

    for (int i = 0; i < QK8_0; ++i) {
        out[i] = d * (float)qs[i];
    }
}

// View-aware AoS q8_0 -> f16 dequant. Rows tight, batch strides may be gapped.
kernel void kernel_dequant_q8_0_f16_view_aos(
    global char * src,
    ulong         src_offset,
    ulong         src_nb1,
    ulong         src_nb2,
    ulong         src_nb3,
    int           nblk0,
    int           ne1,
    int           ne2,
    int           ne3,
    global half * dst
) {
    int blk_i0 = get_global_id(0);
    int i1     = get_global_id(1);
    int batch  = get_global_id(2);

    if (blk_i0 >= nblk0) return;
    if (i1     >= ne1)   return;

    int i2 = batch % ne2;
    int i3 = batch / ne2;
    if (i3 >= ne3) return;

    global char * block = src + src_offset + (ulong)i3*src_nb3 + (ulong)i2*src_nb2 + (ulong)i1*src_nb1 + (ulong)blk_i0 * (2 + QK8_0);
    float d = vload_half(0, (global half *)block);
    global char * qs = block + 2;

    ulong dst_row_base = ((ulong)i3 * ne2 * ne1 + (ulong)i2 * ne1 + (ulong)i1) * nblk0;
    global half * out = dst + (dst_row_base + blk_i0) * QK8_0;

    for (int i = 0; i < QK8_0; ++i) {
        out[i] = (half)(d * (float)qs[i]);
    }
}

// View-aware AoS q4_0 -> f32 dequant (mirrors the q8_0 view variant).
kernel void kernel_dequant_q4_0_f32_view_aos(
    global char * src,
    ulong         src_offset,
    ulong         src_nb1,
    ulong         src_nb2,
    ulong         src_nb3,
    int           nblk0,
    int           ne1,
    int           ne2,
    int           ne3,
    global float * dst
) {
    int blk_i0 = get_global_id(0);
    int i1     = get_global_id(1);
    int batch  = get_global_id(2);

    if (blk_i0 >= nblk0) return;
    if (i1     >= ne1)   return;

    int i2 = batch % ne2;
    int i3 = batch / ne2;
    if (i3 >= ne3) return;

    global char * block = src + src_offset + (ulong)i3*src_nb3 + (ulong)i2*src_nb2 + (ulong)i1*src_nb1 + (ulong)blk_i0 * (2 + QK4_0/2);
    float d = vload_half(0, (global half *)block);
    global uchar * qs = (global uchar *)(block + 2);

    ulong dst_row_base = ((ulong)i3 * ne2 * ne1 + (ulong)i2 * ne1 + (ulong)i1) * nblk0;
    global float * out = dst + (dst_row_base + blk_i0) * QK4_0;

    for (int i = 0; i < QK4_0/2; ++i) {
        uchar byte = qs[i];
        int q0 = (int)(byte & 0x0F) - 8;
        int q1 = (int)(byte >> 4)   - 8;
        out[i]            = d * (float)q0;
        out[i + QK4_0/2]  = d * (float)q1;
    }
}

// View-aware AoS q4_0 -> f16 dequant (mirrors the q8_0 view variant).
kernel void kernel_dequant_q4_0_f16_view_aos(
    global char * src,
    ulong         src_offset,
    ulong         src_nb1,
    ulong         src_nb2,
    ulong         src_nb3,
    int           nblk0,
    int           ne1,
    int           ne2,
    int           ne3,
    global half * dst
) {
    int blk_i0 = get_global_id(0);
    int i1     = get_global_id(1);
    int batch  = get_global_id(2);

    if (blk_i0 >= nblk0) return;
    if (i1     >= ne1)   return;

    int i2 = batch % ne2;
    int i3 = batch / ne2;
    if (i3 >= ne3) return;

    global char * block = src + src_offset + (ulong)i3*src_nb3 + (ulong)i2*src_nb2 + (ulong)i1*src_nb1 + (ulong)blk_i0 * (2 + QK4_0/2);
    float d = vload_half(0, (global half *)block);
    global uchar * qs = (global uchar *)(block + 2);

    ulong dst_row_base = ((ulong)i3 * ne2 * ne1 + (ulong)i2 * ne1 + (ulong)i1) * nblk0;
    global half * out = dst + (dst_row_base + blk_i0) * QK4_0;

    for (int i = 0; i < QK4_0/2; ++i) {
        uchar byte = qs[i];
        int q0 = (int)(byte & 0x0F) - 8;
        int q1 = (int)(byte >> 4)   - 8;
        out[i]          = (half)(d * (float)q0);
        out[i + QK4_0/2] = (half)(d * (float)q1);
    }
}

kernel void kernel_restore_block_q8_0_trans(
    global uchar * src_q,
    global half  * src_d,
    global block_q8_0 * dst,
    uint ne00,
    uint ne01
){
    // One work item per row, and callers round the dispatch up to the work-group size, so the
    // tail items have no row to restore. Without this they write past the end of dst.
    if (get_global_id(0) >= ne01) {
        return;
    }

    uint num_blk_per_row = ne00 / QK8_0;

    global block_q8_0 * b = (global block_q8_0 *) dst + get_global_id(0) * num_blk_per_row;
    global uchar      * q = (global uchar *) src_q + get_global_id(0) * 4; // 4 8-bit packed
    global half       * d = (global half *) src_d + get_global_id(0);

    for (uint blk = 0; blk < num_blk_per_row; blk++) {
        b->d = *d;

        for (uint i = 0; i < QK8_0; i+=4) {
            b->qs[i]   = q[0];
            b->qs[i+1] = q[1];
            b->qs[i+2] = q[2];
            b->qs[i+3] = q[3];

            q += 4 * ne01; // M stride
        }

        d += ne01;

        b++;
    }
}

//------------------------------------------------------------------------------
// kernel_convert_block_q4_K
// Convert the block_q4_K format to 4 separate arrays (AOS -> SOA).
// This kernel does not deshuffle the bits.
// Each thread processes a super block.
// Mask args are just to keep the signature consistent with the no-shuffle
// version and they are not used in this kernel.
//------------------------------------------------------------------------------
kernel void kernel_convert_block_q4_K(
    global struct block_q4_K * src0,
    global uchar * dst_q,
    global uchar * dst_s,
    global half  * dst_d,
    global half  * dst_dm,
    uchar mask_0F,
    uchar mask_F0
) {
    global struct block_q4_K * b = (global struct block_q4_K *) src0 + get_global_id(0);
    global uchar * q  = (global uchar *) dst_q  + QK_K/2*get_global_id(0);
    global uchar * s  = (global uchar *) dst_s  + K_SCALE_SIZE*get_global_id(0);
    global half  * d  = (global half  *) dst_d  + get_global_id(0);
    global half  * dm = (global half  *) dst_dm + get_global_id(0);

    *d  = b->d;
    *dm = b->dm;

    for (int i = 0; i < QK_K/2; ++i) {
        q[i] = b->q[i];
    }
    for (int i = 0; i < K_SCALE_SIZE; ++i) {
        s[i] = b->s[i];
    }
}

// Restore block_q4_K from flattened arrays.
// Each thread processes a super block.
// Mask args are just to keep the signature consistent with the no-shuffle ones.
kernel void kernel_restore_block_q4_K(
    global uchar * src_q,
    global uchar * src_s,
    global half  * src_d,
    global half  * src_dm,
    global struct block_q4_K * dst,
    uchar mask_0F,
    uchar mask_F0
) {
    global struct block_q4_K * b = (global struct block_q4_K *) dst + get_global_id(0);
    global uchar * q  = (global uchar *) src_q  + QK_K/2*get_global_id(0);
    global uchar * s  = (global uchar *) src_s + K_SCALE_SIZE*get_global_id(0);
    global half  * d  = (global half  *) src_d  + get_global_id(0);
    global half  * dm = (global half  *) src_dm  + get_global_id(0);

    b->d  = *d;
    b->dm = *dm;

    for (int i = 0; i < QK_K/2; ++i) {
        b->q[i] = q[i];
    }
    for (int i = 0; i < K_SCALE_SIZE; ++i) {
        b->s[i] = s[i];
    }
}

kernel void kernel_convert_block_q4_K_noshuffle(
    global struct block_q4_K * src0,
    global uchar * dst_q,
    global uchar * dst_s,
    global half  * dst_d,
    global half  * dst_dm,
    uchar mask_0F,
    uchar mask_F0
) {
    global struct block_q4_K * b = (global struct block_q4_K *) src0 + get_global_id(0);
    global uchar * q  = (global uchar *) dst_q  + QK_K/2 * get_global_id(0);
    global uchar * s  = (global uchar *) dst_s  + K_SCALE_SIZE * get_global_id(0);
    global half  * d  = (global half  *) dst_d  + get_global_id(0);
    global half  * dm = (global half  *) dst_dm + get_global_id(0);

    *d  = b->d;
    *dm = b->dm;

    for (int i = 0; i < QK_K / 64; ++i) {
        for (int j = 0; j < 16; ++j) {
            uchar x0 = b->q[i*32 + 2*j];
            uchar x1 = b->q[i*32 + 2*j + 1];
            q[i*32 + j]      = convert_uchar(x0 & mask_0F) | convert_uchar((x1 & mask_0F) << 4);
            q[i*32 + j + 16] = convert_uchar((x0 & mask_F0) >> 4)   | convert_uchar(x1 & mask_F0);
        }
    }

    for (int i = 0; i < K_SCALE_SIZE; ++i) {
        s[i] = b->s[i];
    }
}

kernel void kernel_restore_block_q4_K_noshuffle(
    global uchar * src_q,
    global uchar * src_s,
    global half  * src_d,
    global half  * src_dm,
    global struct block_q4_K * dst,
    uchar mask_0F,
    uchar mask_F0
) {
    global struct block_q4_K * b = (global struct block_q4_K *) dst + get_global_id(0);
    global uchar * q  = (global uchar *) src_q  + QK_K/2 * get_global_id(0);
    global uchar * s  = (global uchar *) src_s  + K_SCALE_SIZE * get_global_id(0);
    global half  * d  = (global half  *) src_d  + get_global_id(0);
    global half  * dm = (global half  *) src_dm + get_global_id(0);

    b->d  = *d;
    b->dm = *dm;

    for (int i = 0; i < QK_K / 64; ++i) {
        for (int j = 0; j < 16; ++j) {
            uchar lo = q[i*32 + j];
            uchar hi = q[i*32 + j + 16];
            b->q[i*32 + 2*j]     = convert_uchar((lo & mask_0F) | ((hi & mask_0F) << 4));
            b->q[i*32 + 2*j + 1] = convert_uchar(((lo & mask_F0) >> 4) | (hi & mask_F0));
        }
    }

    for (int i = 0; i < K_SCALE_SIZE; ++i) {
        b->s[i] = s[i];
    }
}

//------------------------------------------------------------------------------
// kernel_convert_block_q5_K
// Convert the block_q5_K format to 5 separate arrays (AOS -> SOA).
// Each thread processes a super block.
//------------------------------------------------------------------------------
kernel void kernel_convert_block_q5_K(
    global struct block_q5_K * src0,
    global uchar * dst_q,
    global uchar * dst_qh,
    global uchar * dst_s,
    global half  * dst_d,
    global half  * dst_dm,
    uchar mask_0F,
    uchar mask_F0
) {
    global struct block_q5_K * b  = (global struct block_q5_K *) src0 + get_global_id(0);
    global uchar * q  = (global uchar *) dst_q  + QK_K/2*get_global_id(0);
    global uchar * qh = (global uchar *) dst_qh + QK_K/8*get_global_id(0);
    global uchar * s  = (global uchar *) dst_s  + K_SCALE_SIZE*get_global_id(0);
    global half  * d  = (global half  *) dst_d  + get_global_id(0);
    global half  * dm = (global half  *) dst_dm + get_global_id(0);

    *d  = b->d;
    *dm = b->dm;

    for (int i = 0; i < QK_K/2; ++i) {
        q[i] = b->qs[i];
    }
    for (int i = 0; i < QK_K/8; ++i) {
        qh[i] = b->qh[i];
    }
    for (int i = 0; i < K_SCALE_SIZE; ++i) {
        s[i] = b->s[i];
    }
}

// Restore block_q5_K from flattened arrays.
// Each thread processes a super block.
kernel void kernel_restore_block_q5_K(
    global uchar * src_q,
    global uchar * src_qh,
    global uchar * src_s,
    global half  * src_d,
    global half  * src_dm,
    global struct block_q5_K * dst,
    uchar mask_0F,
    uchar mask_F0
) {
    global struct block_q5_K * b  = (global struct block_q5_K *) dst + get_global_id(0);
    global uchar * q  = (global uchar *) src_q  + QK_K/2*get_global_id(0);
    global uchar * qh = (global uchar *) src_qh + QK_K/8*get_global_id(0);
    global uchar * s  = (global uchar *) src_s  + K_SCALE_SIZE*get_global_id(0);
    global half  * d  = (global half  *) src_d  + get_global_id(0);
    global half  * dm = (global half  *) src_dm + get_global_id(0);

    b->d    = *d;
    b->dm = *dm;

    for (int i = 0; i < QK_K/2; ++i) {
        b->qs[i] = q[i];
    }
    for (int i = 0; i < QK_K/8; ++i) {
        b->qh[i] = qh[i];
    }
    for (int i = 0; i < K_SCALE_SIZE; ++i) {
        b->s[i] = s[i];
    }
}

kernel void kernel_convert_block_q5_K_noshuffle(
    global struct block_q5_K * src0,
    global uchar * dst_q,
    global uchar * dst_qh,
    global uchar * dst_s,
    global half  * dst_d,
    global half  * dst_dm,
    uchar mask_0F,
    uchar mask_F0
) {
    global struct block_q5_K * b  = (global struct block_q5_K *) src0 + get_global_id(0);
    global uchar * q  = (global uchar *) dst_q  + QK_K/2       * get_global_id(0);
    global uchar * qh = (global uchar *) dst_qh + QK_K/8       * get_global_id(0);
    global uchar * s  = (global uchar *) dst_s  + K_SCALE_SIZE * get_global_id(0);
    global half  * d  = (global half  *) dst_d  + get_global_id(0);
    global half  * dm = (global half  *) dst_dm + get_global_id(0);

    *d  = b->d;
    *dm = b->dm;

    for (int i = 0; i < QK_K / 64; ++i) {
        for (int j = 0; j < 16; ++j) {
            uchar x0 = b->qs[i*32 + 2*j];
            uchar x1 = b->qs[i*32 + 2*j + 1];
            q[i*32 + j]      = convert_uchar(x0 & mask_0F) | convert_uchar((x1 & mask_0F) << 4);
            q[i*32 + j + 16] = convert_uchar((x0 & mask_F0) >> 4) | convert_uchar(x1 & mask_F0);
        }
    }

    for (int l = 0; l < QK_K/8; ++l) {
        uchar x0 = 0;
        for (int i = 0; i < 8; ++i) {
            x0 |= ((b->qh[(l%4)*8+i] >> (l/4)) & 0x01) << i;
        }
        qh[l] = x0;
    }

    for (int i = 0; i < K_SCALE_SIZE; ++i) {
        s[i] = b->s[i];
    }
}

kernel void kernel_restore_block_q5_K_noshuffle(
    global uchar * src_q,
    global uchar * src_qh,
    global uchar * src_s,
    global half  * src_d,
    global half  * src_dm,
    global struct block_q5_K * dst,
    uchar mask_0F,
    uchar mask_F0
) {
    global struct block_q5_K * b  = (global struct block_q5_K *) dst + get_global_id(0);
    global uchar * q  = (global uchar *) src_q  + QK_K/2       * get_global_id(0);
    global uchar * qh = (global uchar *) src_qh + QK_K/8       * get_global_id(0);
    global uchar * s  = (global uchar *) src_s  + K_SCALE_SIZE * get_global_id(0);
    global half  * d  = (global half  *) src_d  + get_global_id(0);
    global half  * dm = (global half  *) src_dm + get_global_id(0);

    b->d  = *d;
    b->dm = *dm;

    for (int i = 0; i < QK_K / 64; ++i) {
        for (int j = 0; j < 16; ++j) {
            uchar lo = q[i*32 + j];
            uchar hi = q[i*32 + j + 16];
            b->qs[i*32 + 2*j]     = convert_uchar((lo & mask_0F) | ((hi & mask_0F) << 4));
            b->qs[i*32 + 2*j + 1] = convert_uchar(((lo & mask_F0) >> 4) | (hi & mask_F0));
        }
    }

    for (int g = 0; g < 4; ++g) {
        for (int i = 0; i < 8; ++i) {
            uchar x0 = 0;
            for (int k = 0; k < 8; ++k) {
                x0 |= ((qh[4*k+g] >> i) & 0x01) << k;
            }
            b->qh[g*8+i] = x0;
        }
    }

    for (int i = 0; i < K_SCALE_SIZE; ++i) {
        b->s[i] = s[i];
    }
}

//------------------------------------------------------------------------------
// kernel_convert_block_q6_K
// Convert the block_q6_K format to 3 separate arrays (AOS -> SOA).
// This kernel does not deshuffle the bits.
// Each thread processes a super block.
//------------------------------------------------------------------------------
kernel void kernel_convert_block_q6_K(
    global struct block_q6_K * src0,
    global uchar * dst_ql,
    global uchar * dst_qh,
    global char  * dst_s,
    global half  * dst_d,
    uchar          mask_lsb_8,
    ulong          n_blk
) {
    if (get_global_id(0) >= n_blk) {
        return;
    }
    global struct block_q6_K * b = (global struct block_q6_K *) src0 + get_global_id(0);
    global uchar * ql = (global uchar *) dst_ql + QK_K/2*get_global_id(0);
    global uchar * qh = (global uchar *) dst_qh + QK_K/4*get_global_id(0);
    global char  * s  = (global char  *) dst_s  + QK_K/16*get_global_id(0);
    global half  * d  = (global half  *) dst_d  + get_global_id(0);

    *d = b->d;

    for (int i = 0; i < QK_K/2; ++i) {
        ql[i] = b->ql[i];
    }
    for (int i = 0; i < QK_K/4; ++i) {
        qh[i] = b->qh[i];
    }
    for (int i = 0; i < QK_K/16; ++i) {
        s[i] = b->scales[i];
    }
}

// Restore block_q6_K from flattened arrays.
// Each thread processes a super block.
kernel void kernel_restore_block_q6_K(
    global uchar * dst_ql,
    global uchar * dst_qh,
    global char  * dst_s,
    global half  * dst_d,
    global struct block_q6_K * dst,
    uchar mask_lsb_8,
    ulong n_blk
) {
    if (get_global_id(0) >= n_blk) {
        return;
    }
    global struct block_q6_K * b = (global struct block_q6_K *) dst + get_global_id(0);
    global uchar * ql = (global uchar *) dst_ql + QK_K/2*get_global_id(0);
    global uchar * qh = (global uchar *) dst_qh + QK_K/4*get_global_id(0);
    global char  * s  = (global char  *) dst_s  + QK_K/16*get_global_id(0);
    global half  * d  = (global half  *) dst_d  + get_global_id(0);

    b->d = *d;

    for (int i = 0; i < QK_K/2; ++i) {
        b->ql[i] = ql[i];
    }
    for (int i = 0; i < QK_K/4; ++i) {
        b->qh[i] = qh[i];
    }
    for (int i = 0; i < QK_K/16; ++i) {
        b->scales[i] = s[i];
    }
}

kernel void kernel_convert_block_q6_K_noshuffle(
    global struct block_q6_K * src0,
    global uchar * dst_ql,
    global uchar * dst_qh,
    global char  * dst_s,
    global half  * dst_d,
    uchar          mask_lsb_8,
    ulong          n_blk
) {
    if (get_global_id(0) >= n_blk) {
        return;
    }
    global struct block_q6_K * b = (global struct block_q6_K *) src0 + get_global_id(0);
    global uchar * ql = (global uchar *) dst_ql + QK_K/2*get_global_id(0);
    global uchar * qh = (global uchar *) dst_qh + QK_K/4*get_global_id(0);
    global char  * s  = (global char  *) dst_s  + QK_K/16*get_global_id(0);
    global half  * d  = (global half  *) dst_d  + get_global_id(0);

    *d = b->d;

    for (int i = 0; i < QK_K/2/4; ++i) {
        uchar x0 = b->ql[i*2 + 0] & mask_lsb_8;
        uchar x1 = b->ql[i*2 + 1] & mask_lsb_8;
        ql[i +  0] = (x0 & 0x0F)        | ((x1 & 0x0F) << 4);
        ql[i + 32] = ((x0 & 0xF0) >> 4) | (x1 & 0xF0);

        uchar x2 = b->ql[i*2 + 0 + 64] & mask_lsb_8;
        uchar x3 = b->ql[i*2 + 1 + 64] & mask_lsb_8;
        ql[i + 64] = (x2 & 0x0F)        | ((x3 & 0x0F) << 4);
        ql[i + 96] = ((x2 & 0xF0) >> 4) | (x3 & 0xF0);
    }

    for (int i = 0; i < QK_K/4/8; ++i) {
        uchar x0 = b->qh[i*4 + 0] & mask_lsb_8;
        uchar x1 = b->qh[i*4 + 1] & mask_lsb_8;
        uchar x2 = b->qh[i*4 + 2] & mask_lsb_8;
        uchar x3 = b->qh[i*4 + 3] & mask_lsb_8;
        qh[i +  0] = (x0 & 0x03)        | ((x1 & 0x03) << 2) | ((x2 & 0x03) << 4) | ((x3 & 0x03) << 6);
        qh[i +  8] = ((x0 & 0x0C) >> 2) | (x1 & 0x0C)        | ((x2 & 0x0C) << 2) | ((x3 & 0x0C) << 4);
        qh[i + 16] = ((x0 & 0x30) >> 4) | ((x1 & 0x30) >> 2) | (x2 & 0x30)        | ((x3 & 0x30) << 2);
        qh[i + 24] = ((x0 & 0xC0) >> 6) | ((x1 & 0xC0) >> 4) | ((x2 & 0xC0) >> 2) | (x3 & 0xC0);

        uchar x4 = b->qh[i*4 + 0 + 32] & mask_lsb_8;
        uchar x5 = b->qh[i*4 + 1 + 32] & mask_lsb_8;
        uchar x6 = b->qh[i*4 + 2 + 32] & mask_lsb_8;
        uchar x7 = b->qh[i*4 + 3 + 32] & mask_lsb_8;
        qh[i + 32] = (x4 & 0x03)        | ((x5 & 0x03) << 2) | ((x6 & 0x03) << 4) | ((x7 & 0x03) << 6);
        qh[i + 40] = ((x4 & 0x0C) >> 2) | (x5 & 0x0C)        | ((x6 & 0x0C) << 2) | ((x7 & 0x0C) << 4);
        qh[i + 48] = ((x4 & 0x30) >> 4) | ((x5 & 0x30) >> 2) | (x6 & 0x30)        | ((x7 & 0x30) << 2);
        qh[i + 56] = ((x4 & 0xC0) >> 6) | ((x5 & 0xC0) >> 4) | ((x6 & 0xC0) >> 2) | (x7 & 0xC0);
    }

    for (int i = 0; i < QK_K/16; ++i) {
        s[i] = b->scales[i];
    }
}

kernel void kernel_restore_block_q6_K_noshuffle(
    global uchar * src_ql,
    global uchar * src_qh,
    global char  * src_s,
    global half  * src_d,
    global struct block_q6_K * dst,
    uchar          mask_lsb_8,
    ulong          n_blk
) {
    if (get_global_id(0) >= n_blk) {
        return;
    }
    global struct block_q6_K * b = (global struct block_q6_K *) dst + get_global_id(0);
    global uchar * ql = (global uchar *) src_ql + QK_K/2*get_global_id(0);
    global uchar * qh = (global uchar *) src_qh + QK_K/4*get_global_id(0);
    global char  * s  = (global char  *) src_s  + QK_K/16*get_global_id(0);
    global half  * d  = (global half  *) src_d  + get_global_id(0);

    b->d = *d;

    for (int i = 0; i < QK_K/2/4; ++i) {
        uchar x0   = ql[i +  0] & mask_lsb_8;
        uchar x1   = ql[i + 32] & mask_lsb_8;
        b->ql[i*2 + 0] = (x0 & 0x0F)        | ((x1 & 0x0F) << 4);
        b->ql[i*2 + 1] = ((x0 & 0xF0) >> 4) | (x1 & 0xF0);

        uchar x2   = ql[i + 64] & mask_lsb_8;
        uchar x3   = ql[i + 96] & mask_lsb_8;
        b->ql[i*2 + 0 + 64] = (x2 & 0x0F)        | ((x3 & 0x0F) << 4);
        b->ql[i*2 + 1 + 64] = ((x2 & 0xF0) >> 4) | (x3 & 0xF0);
    }

    for (int i = 0; i < QK_K/4/8; ++i) {
        uchar x0 = qh[i +  0] & mask_lsb_8;
        uchar x1 = qh[i +  8] & mask_lsb_8;
        uchar x2 = qh[i + 16] & mask_lsb_8;
        uchar x3 = qh[i + 24] & mask_lsb_8;
        b->qh[i*4 + 0] = (x0 & 0x03)        | ((x1 & 0x03) << 2) | ((x2 & 0x03) << 4) | ((x3 & 0x03) << 6);
        b->qh[i*4 + 1] = ((x0 & 0x0C) >> 2) | (x1 & 0x0C)        | ((x2 & 0x0C) << 2) | ((x3 & 0x0C) << 4);
        b->qh[i*4 + 2] = ((x0 & 0x30) >> 4) | ((x1 & 0x30) >> 2) | (x2 & 0x30)        | ((x3 & 0x30) << 2);
        b->qh[i*4 + 3] = ((x0 & 0xC0) >> 6) | ((x1 & 0xC0) >> 4) | ((x2 & 0xC0) >> 2) | (x3 & 0xC0);

        uchar x4 = qh[i +  0 + 32] & mask_lsb_8;
        uchar x5 = qh[i +  8 + 32] & mask_lsb_8;
        uchar x6 = qh[i + 16 + 32] & mask_lsb_8;
        uchar x7 = qh[i + 24 + 32] & mask_lsb_8;
        b->qh[i*4 + 0 + 32] = (x4 & 0x03)        | ((x5 & 0x03) << 2) | ((x6 & 0x03) << 4) | ((x7 & 0x03) << 6);
        b->qh[i*4 + 1 + 32] = ((x4 & 0x0C) >> 2) | (x5 & 0x0C)        | ((x6 & 0x0C) << 2) | ((x7 & 0x0C) << 4);
        b->qh[i*4 + 2 + 32] = ((x4 & 0x30) >> 4) | ((x5 & 0x30) >> 2) | (x6 & 0x30)        | ((x7 & 0x30) << 2);
        b->qh[i*4 + 3 + 32] = ((x4 & 0xC0) >> 6) | ((x5 & 0xC0) >> 4) | ((x6 & 0xC0) >> 2) | (x7 & 0xC0);
    }

    for (int i = 0; i < QK_K/16; ++i) {
        b->scales[i] = s[i];
    }
}

//------------------------------------------------------------------------------
// kernel_convert_block_iq4_nl
// Convert the block_iq4_nl format to 2 separate arrays (AOS -> SOA).
//------------------------------------------------------------------------------
kernel void kernel_convert_block_iq4_nl(
    global struct block_iq4_nl * src0,
    global uchar * dst_q,
    global half  * dst_d,
    uchar          mask_0F,
    uchar          mask_F0,
    ulong          n_blk
) {
    if (get_global_id(0) >= n_blk) {
        return;
    }
    global struct block_iq4_nl * b = (global struct block_iq4_nl *) src0 + get_global_id(0);
    global uchar * q = (global uchar *) dst_q + QK4_NL/2*get_global_id(0);
    global half  * d = (global half *) dst_d + get_global_id(0);

    *d = b->d;

    for (int i = 0; i < QK4_NL/2; ++i) {
        q[i] = b->qs[i];
    }
}

kernel void kernel_restore_block_iq4_nl(
    global uchar * src_q,
    global half  * src_d,
    global struct block_iq4_nl * dst,
    ulong          n_blk
) {
    if (get_global_id(0) >= n_blk) {
        return;
    }
    global struct block_iq4_nl * b = (global struct block_iq4_nl *) dst + get_global_id(0);
    global uchar * q = (global uchar *) src_q + QK4_NL/2*get_global_id(0);
    global half  * d = (global half *) src_d + get_global_id(0);

    b->d = *d;

    for (int i = 0; i < QK4_NL/2; ++i) {
        b->qs[i] = q[i];
    }
}

kernel void kernel_convert_block_iq4_nl_noshuffle(
    global struct block_iq4_nl * src0,
    global uchar * dst_q,
    global half  * dst_d,
    uchar          mask_0F,
    uchar          mask_F0,
    ulong          n_blk
) {
    if (get_global_id(0) >= n_blk) {
        return;
    }
    global struct block_iq4_nl * b = (global struct block_iq4_nl *) src0 + get_global_id(0);
    global uchar * q = (global uchar *) dst_q + QK4_NL/2*get_global_id(0);
    global half  * d = (global half *) dst_d + get_global_id(0);

    *d = b->d;
    for (int i = 0; i < QK4_NL/4; ++i) {
        uchar x0 = b->qs[2*i + 0];
        uchar x1 = b->qs[2*i + 1];

        q[i + 0       ] = convert_uchar(x0 & mask_0F) | convert_uchar((x1 & mask_0F) << 4);
        q[i + QK4_NL/4] = convert_uchar((x0 & mask_F0) >> 4) | convert_uchar(x1 & mask_F0);
    }
}

kernel void kernel_restore_block_iq4_nl_noshuffle(
    global uchar * src_q,
    global half  * src_d,
    global struct block_iq4_nl * dst,
    uchar mask_0F,
    uchar mask_F0,
    ulong n_blk
) {
    if (get_global_id(0) >= n_blk) {
        return;
    }
    global struct block_iq4_nl * b = (global struct block_iq4_nl *) dst + get_global_id(0);
    global uchar * q = (global uchar *) src_q + QK4_NL/2*get_global_id(0);
    global half  * d = (global half *) src_d + get_global_id(0);

    b->d = *d;
    for (int i = 0; i < QK4_NL/4; ++i) {
        uchar x0 = q[i + 0       ];
        uchar x1 = q[i + QK4_NL/4];

        b->qs[2*i + 0] = convert_uchar((x0 & mask_0F) | ((x1 & mask_0F) << 4));
        b->qs[2*i + 1] = convert_uchar(((x0 & mask_F0) >> 4) | (x1 & mask_F0));
    }
}

// ---------------------------------------------------------------------------
// kernel_moe_expand_scale_q8_0
//
// Expand the q8_0 per-32-block scale d (one half/block, [expert][row][block]) into
// the UNIFORM scale[16] format the generic dp4a MoE GEMM (kernel_gemm_moe_q8_1_dp4a,
// MOE_QT=80) consumes: 16 f16 per 256-superblock (per-16-element segment), where the
// two segments of each 32-block share the block's d. q8_0 is symmetric -> no min
// buffer (the GEMM runs with has_min=0). The int8 weight codes are reused verbatim
// from the existing flat q8_0 weight buffer (extra0_q8_0->q), so only the scale is
// rebuilt here. One work-item per (row, superblock, expert).
// ---------------------------------------------------------------------------
kernel void kernel_moe_expand_scale_q8_0(
    global const half * src_d,      // [expert][row][block], one scale per 32-block
    global       half * dst_scale,  // [expert][row][block][2] (FLAT per-32-block)
    int ne00,
    int ne01
) {
    int row = get_global_id(0);
    int blk = get_global_id(1);   // 32-block index along K
    int e   = get_global_id(2);
    if (row >= ne01) { return; }

    long nb = ne00 / 32;          // 32-blocks per row (K only needs % 32 == 0)
    half d  = src_d[((long)e*ne01 + row)*nb + blk];
    long b  = (((long)e*ne01 + row)*nb + blk) * 2;
    dst_scale[b + 0] = d;
    dst_scale[b + 1] = d;
}

// ---------------------------------------------------------------------------
// kernel_moe_expand_scale_q5_0
//
// q5_0 = symmetric, value = d*(code-16), code = nibble | (hi<<4) in 0..31. The
// generic dp4a MoE GEMM keeps the unsigned code and centers via the min term:
//   scale*dp4a(code,a) - min*sum(a),  scale = d,  min = d*16.
// Reads the existing q5_0 d ([expert][block][row], one half/32-block, from the
// trans4 convert) and writes the FLAT per-32-block uniform scale[2]/min[1] in
// [expert][row][block] order (a transpose). One work-item per (row, block, expert).
// ---------------------------------------------------------------------------
kernel void kernel_moe_expand_scale_q5_0(
    global const half * src_d,      // [expert][block][row]
    global       half * dst_scale,  // [expert][row][block][2]
    global       half * dst_min,    // [expert][row][block]
    int ne00,
    int ne01
) {
    int row = get_global_id(0);
    int blk = get_global_id(1);
    int e   = get_global_id(2);
    if (row >= ne01) { return; }

    long nb = ne00 / 32;
    half d  = src_d[(long)e*nb*ne01 + (long)blk*ne01 + row];   // [expert][block][row]
    long sb = (((long)e*ne01 + row)*nb + blk) * 2;
    long mb = ((long)e*ne01 + row)*nb + blk;
    dst_scale[sb + 0] = d;
    dst_scale[sb + 1] = d;
    dst_min[mb] = (half)((float)d * 16.0f);
}

// ---------------------------------------------------------------------------
// kernel_moe_expand_scale_q5_K
//
// q5_K value = d*sv*code + (-dm*mn), with the 6-bit packed per-sub-block scale sv
// and min mn (8 sub-blocks of 32 per 256-superblock, decoded by get_scale_min_k4
// from the 12-byte s[]). The generic dp4a MoE GEMM (kernel_gemm_moe_q8_1_dp4a,
// MOE_QT=5) keeps the unsigned 5-bit code and applies scale/min via the uniform
// per-32-block buffers:
//   acc += sc0*a_d*raw1 + sc1*a_d*raw2 - mn_u*a_s,
//   sc0 = sc1 = d*sv (both per-16 segments of a 32-block share the sub-block scale),
//   mn_u = dm*mn (positive; the GEMM subtracts it -> the -dm*mn min term).
// q5_K's q_img (low nibbles) + qh (hi-bit plane) are already in the layout the GEMM
// reads (same trans4_ns convert that feeds gemm_moe_q5_k_f32_ns), so only the scale
// is rebuilt here.
//
// One work-item per (row, superblock, expert); each emits 8 sub-blocks.
// ---------------------------------------------------------------------------
kernel void kernel_moe_expand_scale_q5_K(
    global const uchar * src_s,     // [expert][row][superblock][12]
    global const half  * src_d,     // [expert][superblock][row]
    global const half  * src_dm,    // [expert][superblock][row]
    global       half  * dst_scale, // [expert][row][32block][2]
    global       half  * dst_min,   // [expert][row][32block]
    int ne00,
    int ne01
) {
    int row = get_global_id(0);
    int sb  = get_global_id(1);   // superblock index along K
    int e   = get_global_id(2);
    if (row >= ne01) { return; }

    long nsb    = ne00 / 256;     // superblocks per row
    long nblk32 = ne00 / 32;      // 32-blocks per row

    float d  = (float)src_d [((long)e*nsb + sb)*ne01 + row];
    float dm = (float)src_dm[((long)e*nsb + sb)*ne01 + row];

    __global const uchar * sc = src_s + ((long)e*ne01 + row)*nsb*12 + (long)sb*12;

    for (int j = 0; j < 8; ++j) {
        uchar sv, mn;
        // get_scale_min_k4 (6-bit packed scale/min for sub-block j of 8)
        if (j < 4) {
            sv = sc[j]   & 63;
            mn = sc[j+4] & 63;
        } else {
            sv = (sc[j+4] & 0x0F) | ((sc[j-4] & 0xC0) >> 2);
            mn = ((sc[j+4] >> 4) & 0x0F) | ((sc[j]   & 0xC0) >> 2);
        }
        long sub   = (long)sb*8 + j;
        long sbase = (((long)e*ne01 + row)*nblk32 + sub) * 2;
        half s_val = (half)(d  * (float)sv);
        dst_scale[sbase + 0] = s_val;
        dst_scale[sbase + 1] = s_val;
        dst_min[((long)e*ne01 + row)*nblk32 + sub] = (half)(dm * (float)mn);
    }
}
