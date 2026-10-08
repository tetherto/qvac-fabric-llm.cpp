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

inline void get_scale_min_k4(
    int j,
    global const uchar * q,
    uchar * d,
    uchar * m,
    uchar mask_d6,
    uchar mask_d4,
    uchar mask_hi2
) {
    if (j < 4) {
        *d = q[j]   & mask_d6;
        *m = q[j+4] & mask_d6;
    } else {
        *d = (q[j+4] & mask_d4) | ((q[j-4] & mask_hi2) >> 2);
        *m = ((q[j+4] >> 4) & mask_d4) | ((q[j]   & mask_hi2) >> 2);
    }
}

#define dequantizeBlockAccum_ns_sgbroadcast_1_hi(total_sums, bits4, bits1, scale, minv, y) \
    float shared_y; \
    shared_y = sub_group_broadcast(y.s0, 0); \
    total_sums.s0 += (((bits4.s0 & 0x000F) | ((bits1.s0 & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0x000F) | ((bits1.s1 & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 0); \
    total_sums.s0 += ((((bits4.s0 & 0x00F0) >> 4) | (((bits1.s0 >> 1) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s1 & 0x00F0) >> 4) | (((bits1.s1 >> 1) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 0); \
    total_sums.s0 += ((((bits4.s0 & 0x0F00) >> 8) | (((bits1.s0 >> 2) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s1 & 0x0F00) >> 8) | (((bits1.s1 >> 2) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 0); \
    total_sums.s0 += ((((bits4.s0 & 0xF000) >> 12) | (((bits1.s0 >> 3) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s1 & 0xF000) >> 12) | (((bits1.s1 >> 3) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 0); \
    total_sums.s0 += (((bits4.s2 & 0x000F) | (((bits1.s0 >> 4) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0x000F) | (((bits1.s1 >> 4) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 0); \
    total_sums.s0 += ((((bits4.s2 & 0x00F0) >> 4) | (((bits1.s0 >> 5) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s3 & 0x00F0) >> 4) | (((bits1.s1 >> 5) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 0); \
    total_sums.s0 += ((((bits4.s2 & 0x0F00) >> 8) | (((bits1.s0 >> 6) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s3 & 0x0F00) >> 8) | (((bits1.s1 >> 6) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 0); \
    total_sums.s0 += ((((bits4.s2 & 0xF000) >> 12) | (((bits1.s0 >> 7) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s3 & 0xF000) >> 12) | (((bits1.s1 >> 7) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s0, 1); \
    total_sums.s0 += (((bits4.s4 & 0x000F) | ((bits1.s2 & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0x000F) | ((bits1.s3 & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 1); \
    total_sums.s0 += ((((bits4.s4 & 0x00F0) >> 4) | (((bits1.s2 >> 1) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s5 & 0x00F0) >> 4) | (((bits1.s3 >> 1) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 1); \
    total_sums.s0 += ((((bits4.s4 & 0x0F00) >> 8) | (((bits1.s2 >> 2) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s5 & 0x0F00) >> 8) | (((bits1.s3 >> 2) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 1); \
    total_sums.s0 += ((((bits4.s4 & 0xF000) >> 12) | (((bits1.s2 >> 3) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s5 & 0xF000) >> 12) | (((bits1.s3 >> 3) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 1); \
    total_sums.s0 += (((bits4.s6 & 0x000F) | (((bits1.s2 >> 4) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0x000F) | (((bits1.s3 >> 4) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 1); \
    total_sums.s0 += ((((bits4.s6 & 0x00F0) >> 4) | (((bits1.s2 >> 5) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s7 & 0x00F0) >> 4) | (((bits1.s3 >> 5) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 1); \
    total_sums.s0 += ((((bits4.s6 & 0x0F00) >> 8) | (((bits1.s2 >> 6) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s7 & 0x0F00) >> 8) | (((bits1.s3 >> 6) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 1); \
    total_sums.s0 += ((((bits4.s6 & 0xF000) >> 12) | (((bits1.s2 >> 7) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s7 & 0xF000) >> 12) | (((bits1.s3 >> 7) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \


#define dequantizeBlockAccum_ns_sgbroadcast_1_lo(total_sums, bits4, bits1, scale, minv, y) \
    shared_y = sub_group_broadcast(y.s0, 2); \
    total_sums.s0 += (((bits4.s0 & 0x000F) | ((bits1.s4 & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0x000F) | ((bits1.s5 & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 2); \
    total_sums.s0 += ((((bits4.s0 & 0x00F0) >> 4) | (((bits1.s4 >> 1) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s1 & 0x00F0) >> 4) | (((bits1.s5 >> 1) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 2); \
    total_sums.s0 += ((((bits4.s0 & 0x0F00) >> 8) | (((bits1.s4 >> 2) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s1 & 0x0F00) >> 8) | (((bits1.s5 >> 2) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 2); \
    total_sums.s0 += ((((bits4.s0 & 0xF000) >> 12) | (((bits1.s4 >> 3) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s1 & 0xF000) >> 12) | (((bits1.s5 >> 3) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 2); \
    total_sums.s0 += (((bits4.s2 & 0x000F) | (((bits1.s4 >> 4) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0x000F) | (((bits1.s5 >> 4) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 2); \
    total_sums.s0 += ((((bits4.s2 & 0x00F0) >> 4) | (((bits1.s4 >> 5) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s3 & 0x00F0) >> 4) | (((bits1.s5 >> 5) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 2); \
    total_sums.s0 += ((((bits4.s2 & 0x0F00) >> 8) | (((bits1.s4 >> 6) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s3 & 0x0F00) >> 8) | (((bits1.s5 >> 6) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 2); \
    total_sums.s0 += ((((bits4.s2 & 0xF000) >> 12) | (((bits1.s4 >> 7) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s3 & 0xF000) >> 12) | (((bits1.s5 >> 7) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s0, 3); \
    total_sums.s0 += (((bits4.s4 & 0x000F) | ((bits1.s6 & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0x000F) | ((bits1.s7 & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 3); \
    total_sums.s0 += ((((bits4.s4 & 0x00F0) >> 4) | (((bits1.s6 >> 1) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s5 & 0x00F0) >> 4) | (((bits1.s7 >> 1) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 3); \
    total_sums.s0 += ((((bits4.s4 & 0x0F00) >> 8) | (((bits1.s6 >> 2) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s5 & 0x0F00) >> 8) | (((bits1.s7 >> 2) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 3); \
    total_sums.s0 += ((((bits4.s4 & 0xF000) >> 12) | (((bits1.s6 >> 3) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s5 & 0xF000) >> 12) | (((bits1.s7 >> 3) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 3); \
    total_sums.s0 += (((bits4.s6 & 0x000F) | (((bits1.s6 >> 4) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0x000F) | (((bits1.s7 >> 4) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 3); \
    total_sums.s0 += ((((bits4.s6 & 0x00F0) >> 4) | (((bits1.s6 >> 5) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s7 & 0x00F0) >> 4) | (((bits1.s7 >> 5) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 3); \
    total_sums.s0 += ((((bits4.s6 & 0x0F00) >> 8) | (((bits1.s6 >> 6) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s7 & 0x0F00) >> 8) | (((bits1.s7 >> 6) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 3); \
    total_sums.s0 += ((((bits4.s6 & 0xF000) >> 12) | (((bits1.s6 >> 7) & 0x01) << 4)) * scale.s0 - minv.s0) * shared_y; \
    total_sums.s1 += ((((bits4.s7 & 0xF000) >> 12) | (((bits1.s7 >> 7) & 0x01) << 4)) * scale.s1 - minv.s1) * shared_y; \


#define dequantizeBlockAccum_ns_sgbroadcast_8_hi(total_sums, bits4, bits1, scale, minv, y) \
    float8 shared_y; \
    shared_y = sub_group_broadcast(y, 0); \
    total_sums.s0 += (((bits4.s0 & 0x000F)         | ((bits1.s0 & 0x01) << 4))         * scale.s0 - minv.s0) * shared_y.s0; \
    total_sums.s0 += ((((bits4.s0 & 0x00F0) >> 4)  | (((bits1.s0 >> 1) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s1; \
    total_sums.s0 += ((((bits4.s0 & 0x0F00) >> 8)  | (((bits1.s0 >> 2) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s2; \
    total_sums.s0 += ((((bits4.s0 & 0xF000) >> 12) | (((bits1.s0 >> 3) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s3; \
    total_sums.s0 += (((bits4.s2 & 0x000F)         | (((bits1.s0 >> 4) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s4; \
    total_sums.s0 += ((((bits4.s2 & 0x00F0) >> 4)  | (((bits1.s0 >> 5) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s5; \
    total_sums.s0 += ((((bits4.s2 & 0x0F00) >> 8)  | (((bits1.s0 >> 6) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s6; \
    total_sums.s0 += ((((bits4.s2 & 0xF000) >> 12) | (((bits1.s0 >> 7) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s7; \
    total_sums.s1 += (((bits4.s1 & 0x000F)         | ((bits1.s1 & 0x01) << 4))         * scale.s1 - minv.s1) * shared_y.s0; \
    total_sums.s1 += ((((bits4.s1 & 0x00F0) >> 4)  | (((bits1.s1 >> 1) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s1; \
    total_sums.s1 += ((((bits4.s1 & 0x0F00) >> 8)  | (((bits1.s1 >> 2) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s2; \
    total_sums.s1 += ((((bits4.s1 & 0xF000) >> 12) | (((bits1.s1 >> 3) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s3; \
    total_sums.s1 += (((bits4.s3 & 0x000F)         | (((bits1.s1 >> 4) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s4; \
    total_sums.s1 += ((((bits4.s3 & 0x00F0) >> 4)  | (((bits1.s1 >> 5) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s5; \
    total_sums.s1 += ((((bits4.s3 & 0x0F00) >> 8)  | (((bits1.s1 >> 6) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s6; \
    total_sums.s1 += ((((bits4.s3 & 0xF000) >> 12) | (((bits1.s1 >> 7) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s7; \
    shared_y = sub_group_broadcast(y, 1); \
    total_sums.s0 += (((bits4.s4 & 0x000F)         | ((bits1.s2 & 0x01) << 4))         * scale.s0 - minv.s0) * shared_y.s0; \
    total_sums.s0 += ((((bits4.s4 & 0x00F0) >> 4)  | (((bits1.s2 >> 1) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s1; \
    total_sums.s0 += ((((bits4.s4 & 0x0F00) >> 8)  | (((bits1.s2 >> 2) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s2; \
    total_sums.s0 += ((((bits4.s4 & 0xF000) >> 12) | (((bits1.s2 >> 3) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s3; \
    total_sums.s0 += (((bits4.s6 & 0x000F)         | (((bits1.s2 >> 4) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s4; \
    total_sums.s0 += ((((bits4.s6 & 0x00F0) >> 4)  | (((bits1.s2 >> 5) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s5; \
    total_sums.s0 += ((((bits4.s6 & 0x0F00) >> 8)  | (((bits1.s2 >> 6) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s6; \
    total_sums.s0 += ((((bits4.s6 & 0xF000) >> 12) | (((bits1.s2 >> 7) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s7; \
    total_sums.s1 += (((bits4.s5 & 0x000F)         | ((bits1.s3 & 0x01) << 4))         * scale.s1 - minv.s1) * shared_y.s0; \
    total_sums.s1 += ((((bits4.s5 & 0x00F0) >> 4)  | (((bits1.s3 >> 1) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s1; \
    total_sums.s1 += ((((bits4.s5 & 0x0F00) >> 8)  | (((bits1.s3 >> 2) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s2; \
    total_sums.s1 += ((((bits4.s5 & 0xF000) >> 12) | (((bits1.s3 >> 3) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s3; \
    total_sums.s1 += (((bits4.s7 & 0x000F)         | (((bits1.s3 >> 4) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s4; \
    total_sums.s1 += ((((bits4.s7 & 0x00F0) >> 4)  | (((bits1.s3 >> 5) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s5; \
    total_sums.s1 += ((((bits4.s7 & 0x0F00) >> 8)  | (((bits1.s3 >> 6) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s6; \
    total_sums.s1 += ((((bits4.s7 & 0xF000) >> 12) | (((bits1.s3 >> 7) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s7; \


#define dequantizeBlockAccum_ns_sgbroadcast_8_lo(total_sums, bits4, bits1, scale, minv, y) \
    shared_y = sub_group_broadcast(y, 2); \
    total_sums.s0 += (((bits4.s0 & 0x000F)         | ((bits1.s4 & 0x01) << 4))         * scale.s0 - minv.s0) * shared_y.s0; \
    total_sums.s0 += ((((bits4.s0 & 0x00F0) >> 4)  | (((bits1.s4 >> 1) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s1; \
    total_sums.s0 += ((((bits4.s0 & 0x0F00) >> 8)  | (((bits1.s4 >> 2) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s2; \
    total_sums.s0 += ((((bits4.s0 & 0xF000) >> 12) | (((bits1.s4 >> 3) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s3; \
    total_sums.s0 += (((bits4.s2 & 0x000F)         | (((bits1.s4 >> 4) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s4; \
    total_sums.s0 += ((((bits4.s2 & 0x00F0) >> 4)  | (((bits1.s4 >> 5) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s5; \
    total_sums.s0 += ((((bits4.s2 & 0x0F00) >> 8)  | (((bits1.s4 >> 6) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s6; \
    total_sums.s0 += ((((bits4.s2 & 0xF000) >> 12) | (((bits1.s4 >> 7) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s7; \
    total_sums.s1 += (((bits4.s1 & 0x000F)         | ((bits1.s5 & 0x01) << 4))         * scale.s1 - minv.s1) * shared_y.s0; \
    total_sums.s1 += ((((bits4.s1 & 0x00F0) >> 4)  | (((bits1.s5 >> 1) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s1; \
    total_sums.s1 += ((((bits4.s1 & 0x0F00) >> 8)  | (((bits1.s5 >> 2) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s2; \
    total_sums.s1 += ((((bits4.s1 & 0xF000) >> 12) | (((bits1.s5 >> 3) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s3; \
    total_sums.s1 += (((bits4.s3 & 0x000F)         | (((bits1.s5 >> 4) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s4; \
    total_sums.s1 += ((((bits4.s3 & 0x00F0) >> 4)  | (((bits1.s5 >> 5) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s5; \
    total_sums.s1 += ((((bits4.s3 & 0x0F00) >> 8)  | (((bits1.s5 >> 6) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s6; \
    total_sums.s1 += ((((bits4.s3 & 0xF000) >> 12) | (((bits1.s5 >> 7) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s7; \
    shared_y = sub_group_broadcast(y, 3); \
    total_sums.s0 += (((bits4.s4 & 0x000F)         | ((bits1.s6 & 0x01) << 4))         * scale.s0 - minv.s0) * shared_y.s0; \
    total_sums.s0 += ((((bits4.s4 & 0x00F0) >> 4)  | (((bits1.s6 >> 1) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s1; \
    total_sums.s0 += ((((bits4.s4 & 0x0F00) >> 8)  | (((bits1.s6 >> 2) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s2; \
    total_sums.s0 += ((((bits4.s4 & 0xF000) >> 12) | (((bits1.s6 >> 3) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s3; \
    total_sums.s0 += (((bits4.s6 & 0x000F)         | (((bits1.s6 >> 4) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s4; \
    total_sums.s0 += ((((bits4.s6 & 0x00F0) >> 4)  | (((bits1.s6 >> 5) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s5; \
    total_sums.s0 += ((((bits4.s6 & 0x0F00) >> 8)  | (((bits1.s6 >> 6) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s6; \
    total_sums.s0 += ((((bits4.s6 & 0xF000) >> 12) | (((bits1.s6 >> 7) & 0x01) << 4))  * scale.s0 - minv.s0) * shared_y.s7; \
    total_sums.s1 += (((bits4.s5 & 0x000F)         | ((bits1.s7 & 0x01) << 4))         * scale.s1 - minv.s1) * shared_y.s0; \
    total_sums.s1 += ((((bits4.s5 & 0x00F0) >> 4)  | (((bits1.s7 >> 1) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s1; \
    total_sums.s1 += ((((bits4.s5 & 0x0F00) >> 8)  | (((bits1.s7 >> 2) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s2; \
    total_sums.s1 += ((((bits4.s5 & 0xF000) >> 12) | (((bits1.s7 >> 3) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s3; \
    total_sums.s1 += (((bits4.s7 & 0x000F)         | (((bits1.s7 >> 4) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s4; \
    total_sums.s1 += ((((bits4.s7 & 0x00F0) >> 4)  | (((bits1.s7 >> 5) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s5; \
    total_sums.s1 += ((((bits4.s7 & 0x0F00) >> 8)  | (((bits1.s7 >> 6) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s6; \
    total_sums.s1 += ((((bits4.s7 & 0xF000) >> 12) | (((bits1.s7 >> 7) & 0x01) << 4))  * scale.s1 - minv.s1) * shared_y.s7; \

#ifndef MC_N_COLS
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemv_noshuffle_q5_k_f32(
        read_only  image1d_buffer_t src0_q,
        read_only  image1d_buffer_t src0_qh,
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

    uint K = ne00;
    uint M = ne01;

    uint LINE_STRIDE_A     = M / 2;
    uint BLOCK_STRIDE_A    = NSUBGROUPS * M;

    uint LINE_STRIDE_A_QH  = M / 2;
    uint BLOCK_STRIDE_A_QH = NSUBGROUPS * M / 2;
    uint scales_per_row    = (K / QK_K) * 12;

    private uint4     regA;
    private ushort4   regH;
    private half2     regS;
    private half2     regM;
    private float8    regB;

    private float2 totalSum = (float2)(0.0f);

    for (uint k = groupId; k < (K / 32); k += NSUBGROUPS) {
        uint sb = k / 8;
        uint j  = k % 8;

        half2 d   = src0_d[gid + sb * LINE_STRIDE_A];
        half2 dm  = src0_m[gid + sb * LINE_STRIDE_A];

        global const uchar * sc0 = src0_s + 2 * gid * scales_per_row + sb * 12;
        global const uchar * sc1 = src0_s + (2 * gid + 1) * scales_per_row + sb * 12;

        uchar sv0, mn0, sv1, mn1;
        get_scale_min_k4(j, sc0, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(j, sc1, &sv1, &mn1, mask_d6, mask_d4, mask_hi2);

        regS = convert_half2(convert_float2(d)  * convert_float2((uchar2)(sv0, sv1)));
        regM = convert_half2(convert_float2(dm) * convert_float2((uchar2)(mn0, mn1)));

        if (slid < 4) {
            regB.s0123 = read_imagef(src1, (slid * 2 + k * 8));
            regB.s4567 = read_imagef(src1, (1 + slid * 2 + k * 8));
        }

        regH.s0 = as_ushort(read_imageh(src0_qh, (gid + k * BLOCK_STRIDE_A_QH + LINE_STRIDE_A_QH * 0)).x);
        regH.s1 = as_ushort(read_imageh(src0_qh, (gid + k * BLOCK_STRIDE_A_QH + LINE_STRIDE_A_QH * 1)).x);
        regH.s2 = as_ushort(read_imageh(src0_qh, (gid + k * BLOCK_STRIDE_A_QH + LINE_STRIDE_A_QH * 2)).x);
        regH.s3 = as_ushort(read_imageh(src0_qh, (gid + k * BLOCK_STRIDE_A_QH + LINE_STRIDE_A_QH * 3)).x);

        regA.s0 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 0)).x;
        regA.s1 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 1)).x;
        regA.s2 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 2)).x;
        regA.s3 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 3)).x;
#ifdef VECTOR_SUB_GROUP_BROADCAST
        dequantizeBlockAccum_ns_sgbroadcast_8_hi(totalSum, as_ushort8(regA), as_uchar8(regH), regS, regM, regB);
#else
        dequantizeBlockAccum_ns_sgbroadcast_1_hi(totalSum, as_ushort8(regA), as_uchar8(regH), regS, regM, regB);
#endif // VECTOR_SUB_GROUP_BROADCAST

        regA.s0 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 4)).x;
        regA.s1 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 5)).x;
        regA.s2 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 6)).x;
        regA.s3 = read_imageui(src0_q, (gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 7)).x;
#ifdef VECTOR_SUB_GROUP_BROADCAST
        dequantizeBlockAccum_ns_sgbroadcast_8_lo(totalSum, as_ushort8(regA), as_uchar8(regH), regS, regM, regB);
#else
        dequantizeBlockAccum_ns_sgbroadcast_1_lo(totalSum, as_ushort8(regA), as_uchar8(regH), regS, regM, regB);
#endif // VECTOR_SUB_GROUP_BROADCAST
    }

    // reduction in local memory, assumes #wave=4
    local float2 reduceLM[SUBGROUP_SIZE * 3];
    if (groupId == 1) {
        reduceLM[SUBGROUP_SIZE * 0 + slid] = totalSum;
    }
    if (groupId == 2) {
        reduceLM[SUBGROUP_SIZE * 1 + slid] = totalSum;
    }
    if (groupId == 3) {
        reduceLM[SUBGROUP_SIZE * 2 + slid] = totalSum;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (groupId == 0) {
        totalSum += reduceLM[SUBGROUP_SIZE * 0 + slid];
    }
    if (groupId == 0) {
        totalSum += reduceLM[SUBGROUP_SIZE * 1 + slid];
    }
    if (groupId == 0) {
        totalSum += reduceLM[SUBGROUP_SIZE * 2 + slid];
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
#endif // MC_N_COLS

#ifdef MC_N_COLS
// Multi-column kernel_gemv_noshuffle_q5_k_f32, built once per MC_N_COLS (2..8): weights load once per K-block and
// each column runs the 1-column dequant-accumulate with the same K-split, so it is bit-identical to it.
#ifdef VECTOR_SUB_GROUP_BROADCAST
#define Q5K_MC_DEQ_HI dequantizeBlockAccum_ns_sgbroadcast_8_hi
#define Q5K_MC_DEQ_LO dequantizeBlockAccum_ns_sgbroadcast_8_lo
#else
#define Q5K_MC_DEQ_HI dequantizeBlockAccum_ns_sgbroadcast_1_hi
#define Q5K_MC_DEQ_LO dequantizeBlockAccum_ns_sgbroadcast_1_lo
#endif

#define Q5K_MC_COL(ts, c) { \
    if (slid < 4) { regB.s0123 = read_imagef(src1, (c) * COL_STRIDE +     slid * 2 + k * 8); \
                    regB.s4567 = read_imagef(src1, (c) * COL_STRIDE + 1 + slid * 2 + k * 8); } \
    Q5K_MC_DEQ_HI(ts, as_ushort8(regA_hi), as_uchar8(regH), regS, regM, regB); \
    Q5K_MC_DEQ_LO(ts, as_ushort8(regA_lo), as_uchar8(regH), regS, regM, regB); }

// Sums the subgroup partials of one column in subgroup order and stores its two rows.
inline void reduce_store_col_q5k(float2 ts, uint col, local float2 * lm, uint sg, ushort slid,
                                 uint gid, uint M, global float * dst) {
    if (sg > 0) {
        lm[SUBGROUP_SIZE * (sg - 1) + slid] = ts;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg == 0) {
        for (uint i = 0; i < NSUBGROUPS - 1; ++i) {
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
kernel void kernel_gemv_noshuffle_q5_k_f32_mc(
        read_only  image1d_buffer_t src0_q,
        read_only  image1d_buffer_t src0_qh,
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

    uint K = ne00;
    uint M = ne01;

    uint LINE_STRIDE_A     = M / 2;
    uint BLOCK_STRIDE_A    = NSUBGROUPS * M;
    uint LINE_STRIDE_A_QH  = M / 2;
    uint BLOCK_STRIDE_A_QH = NSUBGROUPS * M / 2;
    uint scales_per_row    = (K / QK_K) * 12;
    uint COL_STRIDE        = K / 4;
    // Tail lanes (ne01 % 128 != 0) fetch a clamped row; the store guard drops their results.
    uint gid_s = min(gid, LINE_STRIDE_A - 1);

    private uint4   regA_hi, regA_lo;
    private ushort4 regH;
    private half2   regS, regM;
    private float8  regB;

    float2 ts0 = 0.0f, ts1 = 0.0f, ts2 = 0.0f, ts3 = 0.0f;
    float2 ts4 = 0.0f, ts5 = 0.0f, ts6 = 0.0f, ts7 = 0.0f;

    for (uint k = groupId; k < (K / 32); k += NSUBGROUPS) {
        uint sb = k / 8;
        uint j  = k % 8;

        half2 d   = src0_d[gid_s + sb * LINE_STRIDE_A];
        half2 dm  = src0_m[gid_s + sb * LINE_STRIDE_A];

        global const uchar * sc0 = src0_s + 2 * gid_s * scales_per_row + sb * 12;
        global const uchar * sc1 = src0_s + (2 * gid_s + 1) * scales_per_row + sb * 12;

        uchar sv0, mn0, sv1, mn1;
        get_scale_min_k4(j, sc0, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4(j, sc1, &sv1, &mn1, mask_d6, mask_d4, mask_hi2);

        regS = convert_half2(convert_float2(d)  * convert_float2((uchar2)(sv0, sv1)));
        regM = convert_half2(convert_float2(dm) * convert_float2((uchar2)(mn0, mn1)));

        regH.s0 = as_ushort(read_imageh(src0_qh, (gid_s + k * BLOCK_STRIDE_A_QH + LINE_STRIDE_A_QH * 0)).x);
        regH.s1 = as_ushort(read_imageh(src0_qh, (gid_s + k * BLOCK_STRIDE_A_QH + LINE_STRIDE_A_QH * 1)).x);
        regH.s2 = as_ushort(read_imageh(src0_qh, (gid_s + k * BLOCK_STRIDE_A_QH + LINE_STRIDE_A_QH * 2)).x);
        regH.s3 = as_ushort(read_imageh(src0_qh, (gid_s + k * BLOCK_STRIDE_A_QH + LINE_STRIDE_A_QH * 3)).x);

        regA_hi.s0 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 0)).x;
        regA_hi.s1 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 1)).x;
        regA_hi.s2 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 2)).x;
        regA_hi.s3 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 3)).x;
        regA_lo.s0 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 4)).x;
        regA_lo.s1 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 5)).x;
        regA_lo.s2 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 6)).x;
        regA_lo.s3 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 7)).x;

        Q5K_MC_COL(ts0, 0);
        Q5K_MC_COL(ts1, 1);
        if (MC_N_COLS > 2) Q5K_MC_COL(ts2, 2);
        if (MC_N_COLS > 3) Q5K_MC_COL(ts3, 3);
        if (MC_N_COLS > 4) Q5K_MC_COL(ts4, 4);
        if (MC_N_COLS > 5) Q5K_MC_COL(ts5, 5);
        if (MC_N_COLS > 6) Q5K_MC_COL(ts6, 6);
        if (MC_N_COLS > 7) Q5K_MC_COL(ts7, 7);
    }

    local float2 reduceLM[SUBGROUP_SIZE * (NSUBGROUPS - 1)];
    dst = (global float*)((global char*)dst + offsetd);
    reduce_store_col_q5k(ts0, 0, reduceLM, groupId, slid, gid, M, dst);
    reduce_store_col_q5k(ts1, 1, reduceLM, groupId, slid, gid, M, dst);
    if (MC_N_COLS > 2) reduce_store_col_q5k(ts2, 2, reduceLM, groupId, slid, gid, M, dst);
    if (MC_N_COLS > 3) reduce_store_col_q5k(ts3, 3, reduceLM, groupId, slid, gid, M, dst);
    if (MC_N_COLS > 4) reduce_store_col_q5k(ts4, 4, reduceLM, groupId, slid, gid, M, dst);
    if (MC_N_COLS > 5) reduce_store_col_q5k(ts5, 5, reduceLM, groupId, slid, gid, M, dst);
    if (MC_N_COLS > 6) reduce_store_col_q5k(ts6, 6, reduceLM, groupId, slid, gid, M, dst);
    if (MC_N_COLS > 7) reduce_store_col_q5k(ts7, 7, reduceLM, groupId, slid, gid, M, dst);
}
#endif // MC_N_COLS
