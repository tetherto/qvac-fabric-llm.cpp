#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#endif

#define QK_K  256
#define NSUBGROUPS 4
#define SUBGROUP_SIZE 64

// scales are transposed: consecutive codes of a row are `stride` apart
inline void get_scale_min_k4(
    int j,
    global const uchar * q,
    uint stride,
    uchar * d,
    uchar * m,
    uchar mask_d6,
    uchar mask_d4,
    uchar mask_hi2
) {
    if (j < 4) {
        *d = q[j*stride]     & mask_d6;
        *m = q[(j+4)*stride] & mask_d6;
    } else {
        *d = (q[(j+4)*stride] & mask_d4) | ((q[(j-4)*stride] & mask_hi2) >> 2);
        *m = ((q[(j+4)*stride] >> 4) & mask_d4) | ((q[j*stride] & mask_hi2) >> 2);
    }
}

// fetch(v, e, lane) yields element e of `lane`'s activations; ya serves lanes 0 and 2, yb lanes 1 and 3.
#define ACT_BROADCAST(v, e, lane) sub_group_broadcast(v.e, lane)

#define dequantizeBlockAccum_ns_sgbroadcast_1_hi(total_sums, bits4, scale, minv, y) \
    dequantizeBlockAccum_ns_fetch_hi(total_sums, bits4, scale, minv, ACT_BROADCAST, y, y)

#define dequantizeBlockAccum_ns_fetch_hi(total_sums, bits4, scale, minv, fetch, ya, yb) \
    float shared_y; \
    shared_y = fetch(ya, s0, 0); \
    total_sums.s0 += ((bits4.s0 & 0x000F) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((bits4.s1 & 0x000F) * scale.s1 - minv.s1) * shared_y; \
    shared_y = fetch(ya, s1, 0); \
    total_sums.s0 += (((bits4.s0 & 0x00F0) >> 4) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0x00F0) >> 4) * scale.s1 - minv.s1) * shared_y; \
    shared_y = fetch(ya, s2, 0); \
    total_sums.s0 += (((bits4.s0 & 0x0F00) >> 8) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0x0F00) >> 8) * scale.s1 - minv.s1) * shared_y; \
    shared_y = fetch(ya, s3, 0); \
    total_sums.s0 += (((bits4.s0 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y; \
    shared_y = fetch(ya, s4, 0); \
    total_sums.s0 += ((bits4.s2 & 0x000F) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((bits4.s3 & 0x000F) * scale.s1 - minv.s1) * shared_y; \
    shared_y = fetch(ya, s5, 0); \
    total_sums.s0 += (((bits4.s2 & 0x00F0) >> 4) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0x00F0) >> 4) * scale.s1 - minv.s1) * shared_y; \
    shared_y = fetch(ya, s6, 0); \
    total_sums.s0 += (((bits4.s2 & 0x0F00) >> 8) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0x0F00) >> 8) * scale.s1 - minv.s1) * shared_y; \
    shared_y = fetch(ya, s7, 0); \
    total_sums.s0 += (((bits4.s2 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y; \
    shared_y = fetch(yb, s0, 1); \
    total_sums.s0 += ((bits4.s4 & 0x000F) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((bits4.s5 & 0x000F) * scale.s1 - minv.s1) * shared_y; \
    shared_y = fetch(yb, s1, 1); \
    total_sums.s0 += (((bits4.s4 & 0x00F0) >> 4) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0x00F0) >> 4) * scale.s1 - minv.s1) * shared_y; \
    shared_y = fetch(yb, s2, 1); \
    total_sums.s0 += (((bits4.s4 & 0x0F00) >> 8) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0x0F00) >> 8) * scale.s1 - minv.s1) * shared_y; \
    shared_y = fetch(yb, s3, 1); \
    total_sums.s0 += (((bits4.s4 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y; \
    shared_y = fetch(yb, s4, 1); \
    total_sums.s0 += ((bits4.s6 & 0x000F) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((bits4.s7 & 0x000F) * scale.s1 - minv.s1) * shared_y; \
    shared_y = fetch(yb, s5, 1); \
    total_sums.s0 += (((bits4.s6 & 0x00F0) >> 4) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0x00F0) >> 4) * scale.s1 - minv.s1) * shared_y; \
    shared_y = fetch(yb, s6, 1); \
    total_sums.s0 += (((bits4.s6 & 0x0F00) >> 8) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0x0F00) >> 8) * scale.s1 - minv.s1) * shared_y; \
    shared_y = fetch(yb, s7, 1); \
    total_sums.s0 += (((bits4.s6 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y; \


#define dequantizeBlockAccum_ns_sgbroadcast_1_lo(total_sums, bits4, scale, minv, y) \
    dequantizeBlockAccum_ns_fetch_lo(total_sums, bits4, scale, minv, ACT_BROADCAST, y, y)

#define dequantizeBlockAccum_ns_fetch_lo(total_sums, bits4, scale, minv, fetch, ya, yb) \
    shared_y = fetch(ya, s0, 2); \
    total_sums.s0 += ((bits4.s0 & 0x000F) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((bits4.s1 & 0x000F) * scale.s1 - minv.s1) * shared_y; \
    shared_y = fetch(ya, s1, 2); \
    total_sums.s0 += (((bits4.s0 & 0x00F0) >> 4) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0x00F0) >> 4) * scale.s1 - minv.s1) * shared_y; \
    shared_y = fetch(ya, s2, 2); \
    total_sums.s0 += (((bits4.s0 & 0x0F00) >> 8) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0x0F00) >> 8) * scale.s1 - minv.s1) * shared_y; \
    shared_y = fetch(ya, s3, 2); \
    total_sums.s0 += (((bits4.s0 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y; \
    shared_y = fetch(ya, s4, 2); \
    total_sums.s0 += ((bits4.s2 & 0x000F) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((bits4.s3 & 0x000F) * scale.s1 - minv.s1) * shared_y; \
    shared_y = fetch(ya, s5, 2); \
    total_sums.s0 += (((bits4.s2 & 0x00F0) >> 4) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0x00F0) >> 4) * scale.s1 - minv.s1) * shared_y; \
    shared_y = fetch(ya, s6, 2); \
    total_sums.s0 += (((bits4.s2 & 0x0F00) >> 8) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0x0F00) >> 8) * scale.s1 - minv.s1) * shared_y; \
    shared_y = fetch(ya, s7, 2); \
    total_sums.s0 += (((bits4.s2 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y; \
    shared_y = fetch(yb, s0, 3); \
    total_sums.s0 += ((bits4.s4 & 0x000F) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((bits4.s5 & 0x000F) * scale.s1 - minv.s1) * shared_y; \
    shared_y = fetch(yb, s1, 3); \
    total_sums.s0 += (((bits4.s4 & 0x00F0) >> 4) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0x00F0) >> 4) * scale.s1 - minv.s1) * shared_y; \
    shared_y = fetch(yb, s2, 3); \
    total_sums.s0 += (((bits4.s4 & 0x0F00) >> 8) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0x0F00) >> 8) * scale.s1 - minv.s1) * shared_y; \
    shared_y = fetch(yb, s3, 3); \
    total_sums.s0 += (((bits4.s4 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y; \
    shared_y = fetch(yb, s4, 3); \
    total_sums.s0 += ((bits4.s6 & 0x000F) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((bits4.s7 & 0x000F) * scale.s1 - minv.s1) * shared_y; \
    shared_y = fetch(yb, s5, 3); \
    total_sums.s0 += (((bits4.s6 & 0x00F0) >> 4) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0x00F0) >> 4) * scale.s1 - minv.s1) * shared_y; \
    shared_y = fetch(yb, s6, 3); \
    total_sums.s0 += (((bits4.s6 & 0x0F00) >> 8) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0x0F00) >> 8) * scale.s1 - minv.s1) * shared_y; \
    shared_y = fetch(yb, s7, 3); \
    total_sums.s0 += (((bits4.s6 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y; \


#define dequantizeBlockAccum_ns_sgbroadcast_8_hi(total_sums, bits4, scale, minv, y) \
    float8 shared_y; \
    shared_y = sub_group_broadcast(y, 0); \
    total_sums.s0 += ((bits4.s0 & 0x000F)         * scale.s0 - minv.s0) * shared_y.s0; \
    total_sums.s0 += (((bits4.s0 & 0x00F0) >> 4)  * scale.s0 - minv.s0) * shared_y.s1; \
    total_sums.s0 += (((bits4.s0 & 0x0F00) >> 8)  * scale.s0 - minv.s0) * shared_y.s2; \
    total_sums.s0 += (((bits4.s0 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y.s3; \
    total_sums.s0 += ((bits4.s2 & 0x000F)         * scale.s0 - minv.s0) * shared_y.s4; \
    total_sums.s0 += (((bits4.s2 & 0x00F0) >> 4)  * scale.s0 - minv.s0) * shared_y.s5; \
    total_sums.s0 += (((bits4.s2 & 0x0F00) >> 8)  * scale.s0 - minv.s0) * shared_y.s6; \
    total_sums.s0 += (((bits4.s2 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y.s7; \
    total_sums.s1 += ((bits4.s1 & 0x000F)         * scale.s1 - minv.s1) * shared_y.s0; \
    total_sums.s1 += (((bits4.s1 & 0x00F0) >> 4)  * scale.s1 - minv.s1) * shared_y.s1; \
    total_sums.s1 += (((bits4.s1 & 0x0F00) >> 8)  * scale.s1 - minv.s1) * shared_y.s2; \
    total_sums.s1 += (((bits4.s1 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y.s3; \
    total_sums.s1 += ((bits4.s3 & 0x000F)         * scale.s1 - minv.s1) * shared_y.s4; \
    total_sums.s1 += (((bits4.s3 & 0x00F0) >> 4)  * scale.s1 - minv.s1) * shared_y.s5; \
    total_sums.s1 += (((bits4.s3 & 0x0F00) >> 8)  * scale.s1 - minv.s1) * shared_y.s6; \
    total_sums.s1 += (((bits4.s3 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y.s7; \
    shared_y = sub_group_broadcast(y, 1); \
    total_sums.s0 += ((bits4.s4 & 0x000F)         * scale.s0 - minv.s0) * shared_y.s0; \
    total_sums.s0 += (((bits4.s4 & 0x00F0) >> 4)  * scale.s0 - minv.s0) * shared_y.s1; \
    total_sums.s0 += (((bits4.s4 & 0x0F00) >> 8)  * scale.s0 - minv.s0) * shared_y.s2; \
    total_sums.s0 += (((bits4.s4 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y.s3; \
    total_sums.s0 += ((bits4.s6 & 0x000F)         * scale.s0 - minv.s0) * shared_y.s4; \
    total_sums.s0 += (((bits4.s6 & 0x00F0) >> 4)  * scale.s0 - minv.s0) * shared_y.s5; \
    total_sums.s0 += (((bits4.s6 & 0x0F00) >> 8)  * scale.s0 - minv.s0) * shared_y.s6; \
    total_sums.s0 += (((bits4.s6 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y.s7; \
    total_sums.s1 += ((bits4.s5 & 0x000F)         * scale.s1 - minv.s1) * shared_y.s0; \
    total_sums.s1 += (((bits4.s5 & 0x00F0) >> 4)  * scale.s1 - minv.s1) * shared_y.s1; \
    total_sums.s1 += (((bits4.s5 & 0x0F00) >> 8)  * scale.s1 - minv.s1) * shared_y.s2; \
    total_sums.s1 += (((bits4.s5 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y.s3; \
    total_sums.s1 += ((bits4.s7 & 0x000F)         * scale.s1 - minv.s1) * shared_y.s4; \
    total_sums.s1 += (((bits4.s7 & 0x00F0) >> 4)  * scale.s1 - minv.s1) * shared_y.s5; \
    total_sums.s1 += (((bits4.s7 & 0x0F00) >> 8)  * scale.s1 - minv.s1) * shared_y.s6; \
    total_sums.s1 += (((bits4.s7 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y.s7; \


#define dequantizeBlockAccum_ns_sgbroadcast_8_lo(total_sums, bits4, scale, minv, y) \
    shared_y = sub_group_broadcast(y, 2); \
    total_sums.s0 += ((bits4.s0 & 0x000F)         * scale.s0 - minv.s0) * shared_y.s0; \
    total_sums.s0 += (((bits4.s0 & 0x00F0) >> 4)  * scale.s0 - minv.s0) * shared_y.s1; \
    total_sums.s0 += (((bits4.s0 & 0x0F00) >> 8)  * scale.s0 - minv.s0) * shared_y.s2; \
    total_sums.s0 += (((bits4.s0 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y.s3; \
    total_sums.s0 += ((bits4.s2 & 0x000F)         * scale.s0 - minv.s0) * shared_y.s4; \
    total_sums.s0 += (((bits4.s2 & 0x00F0) >> 4)  * scale.s0 - minv.s0) * shared_y.s5; \
    total_sums.s0 += (((bits4.s2 & 0x0F00) >> 8)  * scale.s0 - minv.s0) * shared_y.s6; \
    total_sums.s0 += (((bits4.s2 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y.s7; \
    total_sums.s1 += ((bits4.s1 & 0x000F)         * scale.s1 - minv.s1) * shared_y.s0; \
    total_sums.s1 += (((bits4.s1 & 0x00F0) >> 4)  * scale.s1 - minv.s1) * shared_y.s1; \
    total_sums.s1 += (((bits4.s1 & 0x0F00) >> 8)  * scale.s1 - minv.s1) * shared_y.s2; \
    total_sums.s1 += (((bits4.s1 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y.s3; \
    total_sums.s1 += ((bits4.s3 & 0x000F)         * scale.s1 - minv.s1) * shared_y.s4; \
    total_sums.s1 += (((bits4.s3 & 0x00F0) >> 4)  * scale.s1 - minv.s1) * shared_y.s5; \
    total_sums.s1 += (((bits4.s3 & 0x0F00) >> 8)  * scale.s1 - minv.s1) * shared_y.s6; \
    total_sums.s1 += (((bits4.s3 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y.s7; \
    shared_y = sub_group_broadcast(y, 3); \
    total_sums.s0 += ((bits4.s4 & 0x000F)         * scale.s0 - minv.s0) * shared_y.s0; \
    total_sums.s0 += (((bits4.s4 & 0x00F0) >> 4)  * scale.s0 - minv.s0) * shared_y.s1; \
    total_sums.s0 += (((bits4.s4 & 0x0F00) >> 8)  * scale.s0 - minv.s0) * shared_y.s2; \
    total_sums.s0 += (((bits4.s4 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y.s3; \
    total_sums.s0 += ((bits4.s6 & 0x000F)         * scale.s0 - minv.s0) * shared_y.s4; \
    total_sums.s0 += (((bits4.s6 & 0x00F0) >> 4)  * scale.s0 - minv.s0) * shared_y.s5; \
    total_sums.s0 += (((bits4.s6 & 0x0F00) >> 8)  * scale.s0 - minv.s0) * shared_y.s6; \
    total_sums.s0 += (((bits4.s6 & 0xF000) >> 12) * scale.s0 - minv.s0) * shared_y.s7; \
    total_sums.s1 += ((bits4.s5 & 0x000F)         * scale.s1 - minv.s1) * shared_y.s0; \
    total_sums.s1 += (((bits4.s5 & 0x00F0) >> 4)  * scale.s1 - minv.s1) * shared_y.s1; \
    total_sums.s1 += (((bits4.s5 & 0x0F00) >> 8)  * scale.s1 - minv.s1) * shared_y.s2; \
    total_sums.s1 += (((bits4.s5 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y.s3; \
    total_sums.s1 += ((bits4.s7 & 0x000F)         * scale.s1 - minv.s1) * shared_y.s4; \
    total_sums.s1 += (((bits4.s7 & 0x00F0) >> 4)  * scale.s1 - minv.s1) * shared_y.s5; \
    total_sums.s1 += (((bits4.s7 & 0x0F00) >> 8)  * scale.s1 - minv.s1) * shared_y.s6; \
    total_sums.s1 += (((bits4.s7 & 0xF000) >> 12) * scale.s1 - minv.s1) * shared_y.s7; \

#ifndef MC_N_COLS
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemv_noshuffle_q4_k_f32(
        read_only  image1d_buffer_t src0_q,
        global half2  * src0_d,
        global half2  * src0_m,
        global uchar  * src0_s,
        read_only  image1d_buffer_t src1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        uchar mask_d6,
        uchar mask_d4,
        uchar mask_hi2)
{
    uint groupId = get_local_id(1);
    uint gid     = get_global_id(0);
    ushort slid  = get_sub_group_local_id();
    // K-split factor = #subgroups in the WG. Read from the launch (NOT a compile
    // constant) so small-M projections (Kcur/Vcur/Qcur) can dispatch a wider
    // K-split (more waves/SP -> latency hiding) while large-M keeps 4. The
    // physical weight layout stride below is INDEPENDENT of this (see BLOCK_STRIDE_A).
    uint nsg = get_local_size(1);

    uint K = ne00;
    uint M = ne01;

    uint LINE_STRIDE_A  = M / 2;
    // Physical per-K-block stride in the packed image: 8 uints/block-row-pair *
    // (M/2) row-pairs = 4*M uints. This is a layout constant, not tied to nsg.
    uint BLOCK_STRIDE_A = 4 * M;
    uint scales_per_row = (K / QK_K) * 12;

    // The x-grid is padded to CEIL_DIV(ne01/2,64)*64, so when ne01 % 128 != 0 the
    // tail lanes hold gid >= ne01/2. The output stores below are guarded, but the
    // input fetches are not: src0_d and src0_m are raw global half2 pointers,
    // src0_s is a raw global uchar pointer, and read_imageui on an
    // image1d_buffer_t is UNDEFINED out of range -- an image clamps only for
    // SAMPLER reads, which these are not. Those lanes therefore read past the end
    // of all three allocations. For a [2816, 2112] weight (2112 % 128 == 64) the
    // top tail lane is gid = 1087 while only gid < 1056 is backed, and it runs
    // 32 half2 past src0_d/src0_m, 31 uints past the quant image, and 63 bytes
    // past src0_s.
    //
    // Clamp the row used for every fetch. The lanes stay ACTIVE, which the
    // sub_group_broadcast in the dequant macros requires, and their results are
    // still discarded by the existing output guard. No-op and byte-identical
    // whenever ne01 % 128 == 0.
    uint gid_s = min(gid, LINE_STRIDE_A - 1);

    private uint4     regA;
    private half2     regS;
    private half2     regM;
    private float8    regB;

    private float2 totalSum = (float2)(0.0f);

    for (uint k = groupId; k < (K / 32); k += nsg) {
        uint sb = k / 8;
        uint j  = k % 8;

        half2 d   = src0_d[gid_s + sb * LINE_STRIDE_A];
        half2 dm  = src0_m[gid_s + sb * LINE_STRIDE_A];

        global const uchar * sc0 = src0_s + sb * 12 * M + 2 * gid_s;
        global const uchar * sc1 = sc0 + 1;

        uchar sv0, mn0, sv1, mn1;
        get_scale_min_k4(j, sc0, M, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(j, sc1, M, &sv1, &mn1, mask_d6, mask_d4, mask_hi2);

        regS = convert_half2(convert_float2(d)  * convert_float2((uchar2)(sv0, sv1)));
        regM = convert_half2(convert_float2(dm) * convert_float2((uchar2)(mn0, mn1)));

        if (slid < 4) {
            regB.s0123 = read_imagef(src1, (slid * 2 + k * 8));
            regB.s4567 = read_imagef(src1, (1 + slid * 2 + k * 8));
        }

        // load half weights for two blocks in consecutive rows
        regA.s0 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 0)).x;
        regA.s1 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 1)).x;
        regA.s2 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 2)).x;
        regA.s3 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 3)).x;
#ifdef VECTOR_SUB_GROUP_BROADCAST
        dequantizeBlockAccum_ns_sgbroadcast_8_hi(totalSum, as_ushort8(regA), regS, regM, regB);
#else
        dequantizeBlockAccum_ns_sgbroadcast_1_hi(totalSum, as_ushort8(regA), regS, regM, regB);
#endif // VECTOR_SUB_GROUP_BROADCAST

        regA.s0 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 4)).x;
        regA.s1 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 5)).x;
        regA.s2 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 6)).x;
        regA.s3 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 7)).x;
#ifdef VECTOR_SUB_GROUP_BROADCAST
        dequantizeBlockAccum_ns_sgbroadcast_8_lo(totalSum, as_ushort8(regA), regS, regM, regB);
#else
        dequantizeBlockAccum_ns_sgbroadcast_1_lo(totalSum, as_ushort8(regA), regS, regM, regB);
#endif // VECTOR_SUB_GROUP_BROADCAST
    }

    // Cross-subgroup reduction in local memory. Generalized to nsg subgroups
    // (was a hard-coded 4-wave unroll). Sized for up to 16 subgroups (the widest
    // K-split we dispatch for small M). At nsg==4 the accumulation order is
    // identical to the original unroll -> byte-identical for the large-M path.
    local float2 reduceLM[SUBGROUP_SIZE * 15];
    if (groupId > 0) {
        reduceLM[SUBGROUP_SIZE * (groupId - 1) + slid] = totalSum;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (groupId == 0) {
        for (uint i = 0; i < nsg - 1; ++i) {
            totalSum += reduceLM[SUBGROUP_SIZE * i + slid];
        }
    }

    // 2 outputs per fiber in wave 0
    if (groupId == 0) {
        dst = (global float*)((global char*)dst + offsetd);
        // Guard the two output rows. The x-grid is padded to CEIL_DIV(ne01/2,64)*64,
        // so when ne01 is not a multiple of 128 the tail row-pairs run past row ne01
        // and would overrun dst into the adjacent tensor. No-op / byte-identical when
        // ne01 % 128 == 0 (M/2 already a multiple of 64 -> no padding).
        if (gid * 2 + 0 < M) dst[gid * 2 + 0] = totalSum.s0;
        if (gid * 2 + 1 < M) dst[gid * 2 + 1] = totalSum.s1;
    }

}

// --- Fused gate+up GEMV + GLU epilogue (FFN) ------------------------------------
// Folds the FFN's two decode GEMVs (ffn_gate, ffn_up) and the following GLU into a
// SINGLE dispatch: {MUL_MAT(Wg,x), MUL_MAT(Wu,x), GLU}. Both matmuls share the same
// activation x (ffn_norm), so the activation image read is issued ONCE per K-block
// and reused for the gate and up dot products (the per-op path re-reads it twice and
// also materializes the two full ffn-wide intermediates to global, which the GLU
// then re-reads). The gate/up partial sums are accumulated in the SAME per-fiber
// order and reduced in the SAME cross-subgroup order as the standalone GEMV, and the
// GLU formula is the exact scalar expression from kernels/glu.cl, so the output is
// BYTE-IDENTICAL to the per-op matmul+matmul+glu path -> safe to default on.
//   glu_op: REGLU=0, GEGLU=1, SWIGLU=2, GEGLU_ERF=4, GEGLU_QUICK=5 (ggml_glu_op).
// Weights: src0g_* = gate (= GLU src[0]); src0u_* = up (= GLU src[1]).
#define GLU_GEGLU_COEF_A      0.044715f
#define GLU_SQRT_2_OVER_PI    0.79788456080286535587989211986876f
#define GLU_SQRT_2_INV        0.70710678118654752440084436210484f
#define GLU_QUICK_COEF       -1.702f

inline float glu_apply(int glu_op, float g, float u) {
    float act;
    if (glu_op == 1) {        // GEGLU (tanh-approx gelu)
        act = 0.5f*g*(1.0f + tanh(GLU_SQRT_2_OVER_PI*g*(1.0f + GLU_GEGLU_COEF_A*g*g)));
    } else if (glu_op == 2) { // SWIGLU (silu)
        act = g / (1.0f + exp(-g));
    } else if (glu_op == 0) { // REGLU
        return g*u*(g > 0.0f);
    } else if (glu_op == 4) { // GEGLU_ERF
        act = 0.5f*g*(1.0f + erf(g*GLU_SQRT_2_INV));
    } else {                  // GEGLU_QUICK (glu_op == 5)
        act = g*(1.0f/(1.0f + exp(GLU_QUICK_COEF*g)));
    }
    return act*u;
}

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemv_noshuffle_q4_k_f32_glu(
        read_only  image1d_buffer_t src0g_q,
        global half2  * src0g_d,
        global half2  * src0g_m,
        global uchar  * src0g_s,
        read_only  image1d_buffer_t src0u_q,
        global half2  * src0u_d,
        global half2  * src0u_m,
        global uchar  * src0u_s,
        read_only  image1d_buffer_t src1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int glu_op,
        uchar mask_d6,
        uchar mask_d4,
        uchar mask_hi2)
{
    uint groupId = get_local_id(1);
    uint gid     = get_global_id(0);
    ushort slid  = get_sub_group_local_id();
    uint nsg     = get_local_size(1);

    uint K = ne00;
    uint M = ne01;

    uint LINE_STRIDE_A  = M / 2;
    uint BLOCK_STRIDE_A = 4 * M;

    private uint4  regA;
    private half2  regS, regM;
    private float8 regB;

    private float2 gateSum = (float2)(0.0f);
    private float2 upSum   = (float2)(0.0f);

    // Two SEQUENTIAL K-loops (gate fully, then up). Keeping only one weight's
    // working set live at a time holds the kernel's register footprint at ~the
    // base single-weight GEMV's, so its max WG stays 1024 (16 subgroups) and the
    // per-subgroup K-split matches the standalone wide GEMV exactly -> the gate
    // and up partial sums are BYTE-IDENTICAL to the per-op path. The macro body
    // is the base kernel's inner loop verbatim, parameterized by weight source.
#define Q4K_GLU_LOOP(SUM, Q, DD, MM, SS)                                                       \
    for (uint k = groupId; k < (K / 32); k += nsg) {                                           \
        uint sb = k / 8;                                                                       \
        uint j  = k % 8;                                                                       \
        half2 d   = DD[gid + sb * LINE_STRIDE_A];                                              \
        half2 dm  = MM[gid + sb * LINE_STRIDE_A];                                              \
        global const uchar * sc0 = SS + sb * 12 * M + 2 * gid;                                 \
        global const uchar * sc1 = sc0 + 1;                                                    \
        uchar sv0, mn0, sv1, mn1;                                                              \
        get_scale_min_k4(j, sc0, M, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);                      \
        get_scale_min_k4(j, sc1, M, &sv1, &mn1, mask_d6, mask_d4, mask_hi2);                      \
        regS = convert_half2(convert_float2(d)  * convert_float2((uchar2)(sv0, sv1)));         \
        regM = convert_half2(convert_float2(dm) * convert_float2((uchar2)(mn0, mn1)));         \
        if (slid < 4) {                                                                        \
            regB.s0123 = read_imagef(src1, (slid * 2 + k * 8));                                \
            regB.s4567 = read_imagef(src1, (1 + slid * 2 + k * 8));                            \
        }                                                                                      \
        regA.s0 = read_imageui(Q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 0)).x;           \
        regA.s1 = read_imageui(Q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 1)).x;           \
        regA.s2 = read_imageui(Q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 2)).x;           \
        regA.s3 = read_imageui(Q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 3)).x;           \
        DEQ_HI(SUM, as_ushort8(regA), regS, regM, regB);                                       \
        regA.s0 = read_imageui(Q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 4)).x;           \
        regA.s1 = read_imageui(Q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 5)).x;           \
        regA.s2 = read_imageui(Q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 6)).x;           \
        regA.s3 = read_imageui(Q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 7)).x;           \
        DEQ_LO(SUM, as_ushort8(regA), regS, regM, regB);                                       \
    }

#ifdef VECTOR_SUB_GROUP_BROADCAST
#define DEQ_HI dequantizeBlockAccum_ns_sgbroadcast_8_hi
#define DEQ_LO dequantizeBlockAccum_ns_sgbroadcast_8_lo
#else
#define DEQ_HI dequantizeBlockAccum_ns_sgbroadcast_1_hi
#define DEQ_LO dequantizeBlockAccum_ns_sgbroadcast_1_lo
#endif

    Q4K_GLU_LOOP(gateSum, src0g_q, src0g_d, src0g_m, src0g_s)
    Q4K_GLU_LOOP(upSum,   src0u_q, src0u_d, src0u_m, src0u_s)

#undef DEQ_HI
#undef DEQ_LO
#undef Q4K_GLU_LOOP

    // Cross-subgroup reduction in local memory. Packs gate (xy) + up (zw) into a
    // float4 so both reduce in one pass; summation order matches the base GEMV's
    // per-channel loop -> byte-identical partial sums.
    local float4 reduceLM[SUBGROUP_SIZE * 15];
    if (groupId > 0) {
        reduceLM[SUBGROUP_SIZE * (groupId - 1) + slid] = (float4)(gateSum, upSum);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (groupId == 0) {
        for (uint i = 0; i < nsg - 1; ++i) {
            float4 p = reduceLM[SUBGROUP_SIZE * i + slid];
            gateSum += p.xy;
            upSum   += p.zw;
        }
        dst = (global float*)((global char*)dst + offsetd);
        dst[gid * 2 + 0] = glu_apply(glu_op, gateSum.s0, upSum.s0);
        dst[gid * 2 + 1] = glu_apply(glu_op, gateSum.s1, upSum.s1);
    }
}

// --- Split-K-across-workgroups decode GEMV (small-M projections) ----------------
// A single-token GEMV makes only ceil(M/2/64) workgroups; a WG runs on one Adreno
// compute unit, so for small M (Kcur/Vcur, M=512 -> 4 WGs) most of the 16 CUs sit
// idle and the matmul is bandwidth-starved even with a wide intra-WG K-split. This
// variant adds a SECOND grid dimension of `ksplit` workgroups that each reduce a
// disjoint slice of K and write a per-slice partial; kernel_gemv_splitk_reduce_f32
// then sums the partials into dst. Identical math/layout to the base kernel
// (physical block stride 4*M, get_scale_min_k4) -> coherent. Gated host-side to
// M<=1024 (M>=2048
// already fills the CUs and the extra reduce dispatch only hurts).
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemv_noshuffle_q4_k_f32_splitk(
        read_only  image1d_buffer_t src0_q,
        global half2  * src0_d,
        global half2  * src0_m,
        global uchar  * src0_s,
        read_only  image1d_buffer_t src1,
        global float * partial,          // [ksplit * M], slice-major
        int ne00,
        int ne01,
        uchar mask_d6,
        uchar mask_d4,
        uchar mask_hi2)
{
    uint groupId = get_local_id(1);
    uint gid     = get_global_id(0);
    ushort slid  = get_sub_group_local_id();
    uint nsg     = get_local_size(1);
    uint ksplit  = get_num_groups(1);
    uint kslice  = get_group_id(1);

    uint K = ne00;
    uint M = ne01;
    uint LINE_STRIDE_A  = M / 2;
    uint BLOCK_STRIDE_A = 4 * M;      // physical, independent of the K-split

    private uint4  regA;
    private half2  regS, regM;
    private float8 regB;
    private float2 totalSum = (float2)(0.0f);

    // each (kslice, subgroup) pair owns a disjoint set of K-blocks
    for (uint k = kslice * nsg + groupId; k < (K / 32); k += ksplit * nsg) {
        uint sb = k / 8;
        uint j  = k % 8;
        half2 d   = src0_d[gid + sb * LINE_STRIDE_A];
        half2 dm  = src0_m[gid + sb * LINE_STRIDE_A];
        global const uchar * sc0 = src0_s + sb * 12 * M + 2 * gid;
        global const uchar * sc1 = sc0 + 1;
        uchar sv0, mn0, sv1, mn1;
        get_scale_min_k4(j, sc0, M, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(j, sc1, M, &sv1, &mn1, mask_d6, mask_d4, mask_hi2);
        regS = convert_half2(convert_float2(d)  * convert_float2((uchar2)(sv0, sv1)));
        regM = convert_half2(convert_float2(dm) * convert_float2((uchar2)(mn0, mn1)));
        if (slid < 4) {
            regB.s0123 = read_imagef(src1, (slid * 2 + k * 8));
            regB.s4567 = read_imagef(src1, (1 + slid * 2 + k * 8));
        }
        regA.s0 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 0)).x;
        regA.s1 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 1)).x;
        regA.s2 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 2)).x;
        regA.s3 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 3)).x;
#ifdef VECTOR_SUB_GROUP_BROADCAST
        dequantizeBlockAccum_ns_sgbroadcast_8_hi(totalSum, as_ushort8(regA), regS, regM, regB);
#else
        dequantizeBlockAccum_ns_sgbroadcast_1_hi(totalSum, as_ushort8(regA), regS, regM, regB);
#endif
        regA.s0 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 4)).x;
        regA.s1 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 5)).x;
        regA.s2 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 6)).x;
        regA.s3 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 7)).x;
#ifdef VECTOR_SUB_GROUP_BROADCAST
        dequantizeBlockAccum_ns_sgbroadcast_8_lo(totalSum, as_ushort8(regA), regS, regM, regB);
#else
        dequantizeBlockAccum_ns_sgbroadcast_1_lo(totalSum, as_ushort8(regA), regS, regM, regB);
#endif
    }

    local float2 reduceLM[SUBGROUP_SIZE * 15];
    if (groupId > 0) {
        reduceLM[SUBGROUP_SIZE * (groupId - 1) + slid] = totalSum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (groupId == 0) {
        for (uint i = 0; i < nsg - 1; ++i) {
            totalSum += reduceLM[SUBGROUP_SIZE * i + slid];
        }
        vstore2(totalSum, 0, &(partial[kslice * M + gid * 2]));
    }
}

// Sum the per-slice partials [ksplit * M] into dst[M]; applies the dst byte offset.
kernel void kernel_gemv_splitk_reduce_f32(
        global float * partial,
        global float * dst,
        ulong offsetd,
        int   ne01,         // M
        int   ksplit)
{
    uint r = get_global_id(0);
    if (r >= (uint)ne01) return;
    float acc = 0.0f;
    for (uint s = 0; s < (uint)ksplit; ++s) {
        acc += partial[s * (uint)ne01 + r];
    }
    dst = (global float*)((global char*)dst + offsetd);
    dst[r] = acc;
}
#endif // MC_N_COLS

#ifdef MC_N_COLS
// Multi-column kernel_gemv_noshuffle_q4_k_f32, built once per MC_N_COLS (2..8): weights load once per K-block and
// each column runs the 1-column dequant-accumulate (bit-identical to it at the same work-group height).
#define MC_MAX_NSG 16
// From MC_STAGE_MIN_COLS columns on, the activations come from a local-memory slice instead of per-column broadcasts.
#define MC_STAGE_MIN_COLS 4
#define MC_BLOCK_PIXELS   8 // float4 activation pixels per 32-element block

#if MC_N_COLS < MC_STAGE_MIN_COLS
#ifdef VECTOR_SUB_GROUP_BROADCAST
#define Q4K_MC_DEQ_HI dequantizeBlockAccum_ns_sgbroadcast_8_hi
#define Q4K_MC_DEQ_LO dequantizeBlockAccum_ns_sgbroadcast_8_lo
#else
#define Q4K_MC_DEQ_HI dequantizeBlockAccum_ns_sgbroadcast_1_hi
#define Q4K_MC_DEQ_LO dequantizeBlockAccum_ns_sgbroadcast_1_lo
#endif

#define Q4K_MC_COL(ts, c) { \
    if (slid < 4) { regB.s0123 = read_imagef(src1, (c) * COL_STRIDE +     slid * 2 + k * 8); \
                    regB.s4567 = read_imagef(src1, (c) * COL_STRIDE + 1 + slid * 2 + k * 8); } \
    Q4K_MC_DEQ_HI(ts, as_ushort8(regA_hi), regS, regM, regB); \
    Q4K_MC_DEQ_LO(ts, as_ushort8(regA_lo), regS, regM, regB); }
#else
// Every lane reads the staged block itself, so the lane argument is not needed.
#define ACT_LOCAL(v, e, lane) v.e
// The fence keeps the compiler from hoisting every column's local loads, which spilled registers at n8 on A740.
#define Q4K_MC_COL(ts, c) { \
    mem_fence(CLK_LOCAL_MEM_FENCE); \
    local const float4 * sp = mc_stage + (groupId * MC_N_COLS + (c)) * MC_BLOCK_PIXELS; \
    const float8 y0 = (float8)(sp[0], sp[1]); \
    const float8 y1 = (float8)(sp[2], sp[3]); \
    const float8 y2 = (float8)(sp[4], sp[5]); \
    const float8 y3 = (float8)(sp[6], sp[7]); \
    dequantizeBlockAccum_ns_fetch_hi(ts, as_ushort8(regA_hi), regS, regM, ACT_LOCAL, y0, y1); \
    dequantizeBlockAccum_ns_fetch_lo(ts, as_ushort8(regA_lo), regS, regM, ACT_LOCAL, y2, y3); }
#endif

// Sums the nsg subgroup partials of one column in subgroup order and stores its two rows.
inline void reduce_store_col_q4k(float2 ts, uint col, local float2 * lm, uint sg, ushort slid, uint nsg,
                                 uint gid, uint M, global float * dst) {
    if (sg > 0) {
        lm[SUBGROUP_SIZE * (sg - 1) + slid] = ts;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg == 0) {
        for (uint i = 0; i < nsg - 1; ++i) {
            ts += lm[SUBGROUP_SIZE * i + slid];
        }
        if (gid * 2 + 0 < M) dst[col * M + gid * 2 + 0] = ts.s0;
        if (gid * 2 + 1 < M) dst[col * M + gid * 2 + 1] = ts.s1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemv_noshuffle_q4_k_f32_mc(
        read_only  image1d_buffer_t src0_q,
        global half2  * src0_d,
        global half2  * src0_m,
        global uchar  * src0_s,
        read_only  image1d_buffer_t src1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        uchar mask_d6,
        uchar mask_d4,
        uchar mask_hi2)
{
    uint groupId = get_local_id(1);
    uint gid     = get_global_id(0);
    ushort slid  = get_sub_group_local_id();
    uint nsg     = get_local_size(1);

    uint K = ne00;
    uint M = ne01;

    uint LINE_STRIDE_A  = M / 2;
    uint BLOCK_STRIDE_A = 4 * M;
    uint COL_STRIDE     = K / 4;
    // Tail lanes (ne01 % 128 != 0) fetch a clamped row; the store guard drops their results.
    uint gid_s = min(gid, LINE_STRIDE_A - 1);

    private uint4  regA_hi, regA_lo;
    private half2  regS, regM;
    private float8 regB;

    float2 ts0 = 0.0f, ts1 = 0.0f, ts2 = 0.0f, ts3 = 0.0f;
    float2 ts4 = 0.0f, ts5 = 0.0f, ts6 = 0.0f, ts7 = 0.0f;

#if MC_N_COLS >= MC_STAGE_MIN_COLS
    local float4 mc_stage[MC_MAX_NSG * MC_N_COLS * MC_BLOCK_PIXELS];
#endif

    for (uint k = groupId; k < (K / 32); k += nsg) {
        uint sb = k / 8;
        uint j  = k % 8;

        half2 d   = src0_d[gid_s + sb * LINE_STRIDE_A];
        half2 dm  = src0_m[gid_s + sb * LINE_STRIDE_A];

        global const uchar * sc0 = src0_s + sb * 12 * M + 2 * gid_s;
        global const uchar * sc1 = sc0 + 1;

        uchar sv0, mn0, sv1, mn1;
        get_scale_min_k4(j, sc0, M, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(j, sc1, M, &sv1, &mn1, mask_d6, mask_d4, mask_hi2);

        regS = convert_half2(convert_float2(d)  * convert_float2((uchar2)(sv0, sv1)));
        regM = convert_half2(convert_float2(dm) * convert_float2((uchar2)(mn0, mn1)));

        regA_hi.s0 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 0)).x;
        regA_hi.s1 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 1)).x;
        regA_hi.s2 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 2)).x;
        regA_hi.s3 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 3)).x;
        regA_lo.s0 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 4)).x;
        regA_lo.s1 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 5)).x;
        regA_lo.s2 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 6)).x;
        regA_lo.s3 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 7)).x;

#if MC_N_COLS >= MC_STAGE_MIN_COLS
        // Stage this block's activations for every column in the subgroup's own slice. The first barrier keeps the
        // previous block's reads ahead of the overwrite, the second makes the writes visible to every lane.
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
        if (slid < MC_N_COLS * MC_BLOCK_PIXELS) {
            mc_stage[groupId * MC_N_COLS * MC_BLOCK_PIXELS + slid] = read_imagef(src1,
                (slid / MC_BLOCK_PIXELS) * COL_STRIDE + k * MC_BLOCK_PIXELS + slid % MC_BLOCK_PIXELS);
        }
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
#endif
        Q4K_MC_COL(ts0, 0);
        Q4K_MC_COL(ts1, 1);
        if (MC_N_COLS > 2) Q4K_MC_COL(ts2, 2);
        if (MC_N_COLS > 3) Q4K_MC_COL(ts3, 3);
        if (MC_N_COLS > 4) Q4K_MC_COL(ts4, 4);
        if (MC_N_COLS > 5) Q4K_MC_COL(ts5, 5);
        if (MC_N_COLS > 6) Q4K_MC_COL(ts6, 6);
        if (MC_N_COLS > 7) Q4K_MC_COL(ts7, 7);
    }

    local float2 reduceLM[SUBGROUP_SIZE * (MC_MAX_NSG - 1)];
    dst = (global float*)((global char*)dst + offsetd);
    reduce_store_col_q4k(ts0, 0, reduceLM, groupId, slid, nsg, gid, M, dst);
    reduce_store_col_q4k(ts1, 1, reduceLM, groupId, slid, nsg, gid, M, dst);
    if (MC_N_COLS > 2) reduce_store_col_q4k(ts2, 2, reduceLM, groupId, slid, nsg, gid, M, dst);
    if (MC_N_COLS > 3) reduce_store_col_q4k(ts3, 3, reduceLM, groupId, slid, nsg, gid, M, dst);
    if (MC_N_COLS > 4) reduce_store_col_q4k(ts4, 4, reduceLM, groupId, slid, nsg, gid, M, dst);
    if (MC_N_COLS > 5) reduce_store_col_q4k(ts5, 5, reduceLM, groupId, slid, nsg, gid, M, dst);
    if (MC_N_COLS > 6) reduce_store_col_q4k(ts6, 6, reduceLM, groupId, slid, nsg, gid, M, dst);
    if (MC_N_COLS > 7) reduce_store_col_q4k(ts7, 7, reduceLM, groupId, slid, nsg, gid, M, dst);
}
#endif // MC_N_COLS
