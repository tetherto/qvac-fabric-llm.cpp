#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

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

#define NSUBGROUPS 4
#define SUBGROUP_SIZE 64

#define dequantize_block_acc_bcast_8_hi(total_sum, bits4, bits2, scale_d, scale_s, y) \
    float8 shared_y; \
    shared_y = sub_group_broadcast(y, 0); \
    total_sum.s0 += ((float)(((bits4.s0 & 0x000F)      ) | ((bits2.s0 & 0x03) << 4)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y.s0; \
    total_sum.s0 += ((float)(((bits4.s0 & 0x00F0) >>  4) | ((bits2.s0 & 0x0C) << 2)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y.s1; \
    total_sum.s0 += ((float)(((bits4.s0 & 0x0F00) >>  8) | ((bits2.s0 & 0x30)     )) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y.s2; \
    total_sum.s0 += ((float)(((bits4.s0 & 0xF000) >> 12) | ((bits2.s0 & 0xC0) >> 2)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y.s3; \
    total_sum.s0 += ((float)(((bits4.s2 & 0x000F)      ) | ((bits2.s2 & 0x03) << 4)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y.s4; \
    total_sum.s0 += ((float)(((bits4.s2 & 0x00F0) >>  4) | ((bits2.s2 & 0x0C) << 2)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y.s5; \
    total_sum.s0 += ((float)(((bits4.s2 & 0x0F00) >>  8) | ((bits2.s2 & 0x30)     )) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y.s6; \
    total_sum.s0 += ((float)(((bits4.s2 & 0xF000) >> 12) | ((bits2.s2 & 0xC0) >> 2)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y.s7; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x000F)      ) | ((bits2.s1 & 0x03) << 4)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y.s0; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x00F0) >>  4) | ((bits2.s1 & 0x0C) << 2)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y.s1; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x0F00) >>  8) | ((bits2.s1 & 0x30)     )) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y.s2; \
    total_sum.s1 += ((float)(((bits4.s1 & 0xF000) >> 12) | ((bits2.s1 & 0xC0) >> 2)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y.s3; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x000F)      ) | ((bits2.s3 & 0x03) << 4)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y.s4; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x00F0) >>  4) | ((bits2.s3 & 0x0C) << 2)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y.s5; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x0F00) >>  8) | ((bits2.s3 & 0x30)     )) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y.s6; \
    total_sum.s1 += ((float)(((bits4.s3 & 0xF000) >> 12) | ((bits2.s3 & 0xC0) >> 2)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y.s7; \
    shared_y = sub_group_broadcast(y, 1); \
    total_sum.s0 += ((float)(((bits4.s4 & 0x000F)      ) | ((bits2.s4 & 0x03) << 4)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y.s0; \
    total_sum.s0 += ((float)(((bits4.s4 & 0x00F0) >>  4) | ((bits2.s4 & 0x0C) << 2)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y.s1; \
    total_sum.s0 += ((float)(((bits4.s4 & 0x0F00) >>  8) | ((bits2.s4 & 0x30)     )) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y.s2; \
    total_sum.s0 += ((float)(((bits4.s4 & 0xF000) >> 12) | ((bits2.s4 & 0xC0) >> 2)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y.s3; \
    total_sum.s0 += ((float)(((bits4.s6 & 0x000F)      ) | ((bits2.s6 & 0x03) << 4)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y.s4; \
    total_sum.s0 += ((float)(((bits4.s6 & 0x00F0) >>  4) | ((bits2.s6 & 0x0C) << 2)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y.s5; \
    total_sum.s0 += ((float)(((bits4.s6 & 0x0F00) >>  8) | ((bits2.s6 & 0x30)     )) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y.s6; \
    total_sum.s0 += ((float)(((bits4.s6 & 0xF000) >> 12) | ((bits2.s6 & 0xC0) >> 2)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y.s7; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x000F)      ) | ((bits2.s5 & 0x03) << 4)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y.s0; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x00F0) >>  4) | ((bits2.s5 & 0x0C) << 2)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y.s1; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x0F00) >>  8) | ((bits2.s5 & 0x30)     )) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y.s2; \
    total_sum.s1 += ((float)(((bits4.s5 & 0xF000) >> 12) | ((bits2.s5 & 0xC0) >> 2)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y.s3; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x000F)      ) | ((bits2.s7 & 0x03) << 4)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y.s4; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x00F0) >>  4) | ((bits2.s7 & 0x0C) << 2)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y.s5; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x0F00) >>  8) | ((bits2.s7 & 0x30)     )) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y.s6; \
    total_sum.s1 += ((float)(((bits4.s7 & 0xF000) >> 12) | ((bits2.s7 & 0xC0) >> 2)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y.s7; \

#define dequantize_block_acc_bcast_8_lo(total_sum, bits4, bits2, scale_d, scale_s, y) \
    shared_y = sub_group_broadcast(y, 2); \
    total_sum.s0 += ((float)(((bits4.s0 & 0x000F)      ) | ((bits2.s0 & 0x03) << 4)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y.s0; \
    total_sum.s0 += ((float)(((bits4.s0 & 0x00F0) >>  4) | ((bits2.s0 & 0x0C) << 2)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y.s1; \
    total_sum.s0 += ((float)(((bits4.s0 & 0x0F00) >>  8) | ((bits2.s0 & 0x30)     )) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y.s2; \
    total_sum.s0 += ((float)(((bits4.s0 & 0xF000) >> 12) | ((bits2.s0 & 0xC0) >> 2)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y.s3; \
    total_sum.s0 += ((float)(((bits4.s2 & 0x000F)      ) | ((bits2.s2 & 0x03) << 4)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y.s4; \
    total_sum.s0 += ((float)(((bits4.s2 & 0x00F0) >>  4) | ((bits2.s2 & 0x0C) << 2)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y.s5; \
    total_sum.s0 += ((float)(((bits4.s2 & 0x0F00) >>  8) | ((bits2.s2 & 0x30)     )) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y.s6; \
    total_sum.s0 += ((float)(((bits4.s2 & 0xF000) >> 12) | ((bits2.s2 & 0xC0) >> 2)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y.s7; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x000F)      ) | ((bits2.s1 & 0x03) << 4)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y.s0; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x00F0) >>  4) | ((bits2.s1 & 0x0C) << 2)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y.s1; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x0F00) >>  8) | ((bits2.s1 & 0x30)     )) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y.s2; \
    total_sum.s1 += ((float)(((bits4.s1 & 0xF000) >> 12) | ((bits2.s1 & 0xC0) >> 2)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y.s3; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x000F)      ) | ((bits2.s3 & 0x03) << 4)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y.s4; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x00F0) >>  4) | ((bits2.s3 & 0x0C) << 2)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y.s5; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x0F00) >>  8) | ((bits2.s3 & 0x30)     )) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y.s6; \
    total_sum.s1 += ((float)(((bits4.s3 & 0xF000) >> 12) | ((bits2.s3 & 0xC0) >> 2)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y.s7; \
    shared_y = sub_group_broadcast(y, 3); \
    total_sum.s0 += ((float)(((bits4.s4 & 0x000F)      ) | ((bits2.s4 & 0x03) << 4)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y.s0; \
    total_sum.s0 += ((float)(((bits4.s4 & 0x00F0) >>  4) | ((bits2.s4 & 0x0C) << 2)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y.s1; \
    total_sum.s0 += ((float)(((bits4.s4 & 0x0F00) >>  8) | ((bits2.s4 & 0x30)     )) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y.s2; \
    total_sum.s0 += ((float)(((bits4.s4 & 0xF000) >> 12) | ((bits2.s4 & 0xC0) >> 2)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y.s3; \
    total_sum.s0 += ((float)(((bits4.s6 & 0x000F)      ) | ((bits2.s6 & 0x03) << 4)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y.s4; \
    total_sum.s0 += ((float)(((bits4.s6 & 0x00F0) >>  4) | ((bits2.s6 & 0x0C) << 2)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y.s5; \
    total_sum.s0 += ((float)(((bits4.s6 & 0x0F00) >>  8) | ((bits2.s6 & 0x30)     )) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y.s6; \
    total_sum.s0 += ((float)(((bits4.s6 & 0xF000) >> 12) | ((bits2.s6 & 0xC0) >> 2)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y.s7; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x000F)      ) | ((bits2.s5 & 0x03) << 4)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y.s0; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x00F0) >>  4) | ((bits2.s5 & 0x0C) << 2)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y.s1; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x0F00) >>  8) | ((bits2.s5 & 0x30)     )) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y.s2; \
    total_sum.s1 += ((float)(((bits4.s5 & 0xF000) >> 12) | ((bits2.s5 & 0xC0) >> 2)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y.s3; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x000F)      ) | ((bits2.s7 & 0x03) << 4)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y.s4; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x00F0) >>  4) | ((bits2.s7 & 0x0C) << 2)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y.s5; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x0F00) >>  8) | ((bits2.s7 & 0x30)     )) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y.s6; \
    total_sum.s1 += ((float)(((bits4.s7 & 0xF000) >> 12) | ((bits2.s7 & 0xC0) >> 2)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y.s7; \

// fetch(v, e, lane) yields element e of `lane`'s activations; ya serves lanes 0 and 2, yb lanes 1 and 3.
#define ACT_BROADCAST(v, e, lane) sub_group_broadcast(v.e, lane)

#define dequantize_block_acc_bcast_1_hi(total_sum, bits4, bits2, scale_d, scale_s, y) \
    dequantize_block_acc_fetch_hi(total_sum, bits4, bits2, scale_d, scale_s, ACT_BROADCAST, y, y)

#define dequantize_block_acc_fetch_hi(total_sum, bits4, bits2, scale_d, scale_s, fetch, ya, yb) \
    float shared_y; \
    shared_y = fetch(ya, s0, 0); \
    total_sum.s0 += ((float)(((bits4.s0 & 0x000F)      ) | ((bits2.s0 & 0x03) << 4)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x000F)      ) | ((bits2.s1 & 0x03) << 4)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y; \
    shared_y = fetch(ya, s1, 0); \
    total_sum.s0 += ((float)(((bits4.s0 & 0x00F0) >>  4) | ((bits2.s0 & 0x0C) << 2)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x00F0) >>  4) | ((bits2.s1 & 0x0C) << 2)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y; \
    shared_y = fetch(ya, s2, 0); \
    total_sum.s0 += ((float)(((bits4.s0 & 0x0F00) >>  8) | ((bits2.s0 & 0x30)     )) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x0F00) >>  8) | ((bits2.s1 & 0x30)     )) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y; \
    shared_y = fetch(ya, s3, 0); \
    total_sum.s0 += ((float)(((bits4.s0 & 0xF000) >> 12) | ((bits2.s0 & 0xC0) >> 2)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s1 & 0xF000) >> 12) | ((bits2.s1 & 0xC0) >> 2)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y; \
    shared_y = fetch(ya, s4, 0); \
    total_sum.s0 += ((float)(((bits4.s2 & 0x000F)      ) | ((bits2.s2 & 0x03) << 4)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x000F)      ) | ((bits2.s3 & 0x03) << 4)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y; \
    shared_y = fetch(ya, s5, 0); \
    total_sum.s0 += ((float)(((bits4.s2 & 0x00F0) >>  4) | ((bits2.s2 & 0x0C) << 2)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x00F0) >>  4) | ((bits2.s3 & 0x0C) << 2)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y; \
    shared_y = fetch(ya, s6, 0); \
    total_sum.s0 += ((float)(((bits4.s2 & 0x0F00) >>  8) | ((bits2.s2 & 0x30)     )) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x0F00) >>  8) | ((bits2.s3 & 0x30)     )) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y; \
    shared_y = fetch(ya, s7, 0); \
    total_sum.s0 += ((float)(((bits4.s2 & 0xF000) >> 12) | ((bits2.s2 & 0xC0) >> 2)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s3 & 0xF000) >> 12) | ((bits2.s3 & 0xC0) >> 2)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y; \
    shared_y = fetch(yb, s0, 1); \
    total_sum.s0 += ((float)(((bits4.s4 & 0x000F)      ) | ((bits2.s4 & 0x03) << 4)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x000F)      ) | ((bits2.s5 & 0x03) << 4)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y; \
    shared_y = fetch(yb, s1, 1); \
    total_sum.s0 += ((float)(((bits4.s4 & 0x00F0) >>  4) | ((bits2.s4 & 0x0C) << 2)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x00F0) >>  4) | ((bits2.s5 & 0x0C) << 2)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y; \
    shared_y = fetch(yb, s2, 1); \
    total_sum.s0 += ((float)(((bits4.s4 & 0x0F00) >>  8) | ((bits2.s4 & 0x30)     )) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x0F00) >>  8) | ((bits2.s5 & 0x30)     )) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y; \
    shared_y = fetch(yb, s3, 1); \
    total_sum.s0 += ((float)(((bits4.s4 & 0xF000) >> 12) | ((bits2.s4 & 0xC0) >> 2)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s5 & 0xF000) >> 12) | ((bits2.s5 & 0xC0) >> 2)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y; \
    shared_y = fetch(yb, s4, 1); \
    total_sum.s0 += ((float)(((bits4.s6 & 0x000F)      ) | ((bits2.s6 & 0x03) << 4)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x000F)      ) | ((bits2.s7 & 0x03) << 4)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y; \
    shared_y = fetch(yb, s5, 1); \
    total_sum.s0 += ((float)(((bits4.s6 & 0x00F0) >>  4) | ((bits2.s6 & 0x0C) << 2)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x00F0) >>  4) | ((bits2.s7 & 0x0C) << 2)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y; \
    shared_y = fetch(yb, s6, 1); \
    total_sum.s0 += ((float)(((bits4.s6 & 0x0F00) >>  8) | ((bits2.s6 & 0x30)     )) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x0F00) >>  8) | ((bits2.s7 & 0x30)     )) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y; \
    shared_y = fetch(yb, s7, 1); \
    total_sum.s0 += ((float)(((bits4.s6 & 0xF000) >> 12) | ((bits2.s6 & 0xC0) >> 2)) - 32.f) * scale_s.s0 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s7 & 0xF000) >> 12) | ((bits2.s7 & 0xC0) >> 2)) - 32.f) * scale_s.s2 * scale_d.s1 * shared_y; \

#define dequantize_block_acc_bcast_1_lo(total_sum, bits4, bits2, scale_d, scale_s, y) \
    dequantize_block_acc_fetch_lo(total_sum, bits4, bits2, scale_d, scale_s, ACT_BROADCAST, y, y)

#define dequantize_block_acc_fetch_lo(total_sum, bits4, bits2, scale_d, scale_s, fetch, ya, yb) \
    shared_y = fetch(ya, s0, 2); \
    total_sum.s0 += ((float)(((bits4.s0 & 0x000F)      ) | ((bits2.s0 & 0x03) << 4)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x000F)      ) | ((bits2.s1 & 0x03) << 4)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y; \
    shared_y = fetch(ya, s1, 2); \
    total_sum.s0 += ((float)(((bits4.s0 & 0x00F0) >>  4) | ((bits2.s0 & 0x0C) << 2)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x00F0) >>  4) | ((bits2.s1 & 0x0C) << 2)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y; \
    shared_y = fetch(ya, s2, 2); \
    total_sum.s0 += ((float)(((bits4.s0 & 0x0F00) >>  8) | ((bits2.s0 & 0x30)     )) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s1 & 0x0F00) >>  8) | ((bits2.s1 & 0x30)     )) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y; \
    shared_y = fetch(ya, s3, 2); \
    total_sum.s0 += ((float)(((bits4.s0 & 0xF000) >> 12) | ((bits2.s0 & 0xC0) >> 2)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s1 & 0xF000) >> 12) | ((bits2.s1 & 0xC0) >> 2)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y; \
    shared_y = fetch(ya, s4, 2); \
    total_sum.s0 += ((float)(((bits4.s2 & 0x000F)      ) | ((bits2.s2 & 0x03) << 4)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x000F)      ) | ((bits2.s3 & 0x03) << 4)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y; \
    shared_y = fetch(ya, s5, 2); \
    total_sum.s0 += ((float)(((bits4.s2 & 0x00F0) >>  4) | ((bits2.s2 & 0x0C) << 2)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x00F0) >>  4) | ((bits2.s3 & 0x0C) << 2)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y; \
    shared_y = fetch(ya, s6, 2); \
    total_sum.s0 += ((float)(((bits4.s2 & 0x0F00) >>  8) | ((bits2.s2 & 0x30)     )) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s3 & 0x0F00) >>  8) | ((bits2.s3 & 0x30)     )) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y; \
    shared_y = fetch(ya, s7, 2); \
    total_sum.s0 += ((float)(((bits4.s2 & 0xF000) >> 12) | ((bits2.s2 & 0xC0) >> 2)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s3 & 0xF000) >> 12) | ((bits2.s3 & 0xC0) >> 2)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y; \
    shared_y = fetch(yb, s0, 3); \
    total_sum.s0 += ((float)(((bits4.s4 & 0x000F)      ) | ((bits2.s4 & 0x03) << 4)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x000F)      ) | ((bits2.s5 & 0x03) << 4)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y; \
    shared_y = fetch(yb, s1, 3); \
    total_sum.s0 += ((float)(((bits4.s4 & 0x00F0) >>  4) | ((bits2.s4 & 0x0C) << 2)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x00F0) >>  4) | ((bits2.s5 & 0x0C) << 2)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y; \
    shared_y = fetch(yb, s2, 3); \
    total_sum.s0 += ((float)(((bits4.s4 & 0x0F00) >>  8) | ((bits2.s4 & 0x30)     )) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s5 & 0x0F00) >>  8) | ((bits2.s5 & 0x30)     )) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y; \
    shared_y = fetch(yb, s3, 3); \
    total_sum.s0 += ((float)(((bits4.s4 & 0xF000) >> 12) | ((bits2.s4 & 0xC0) >> 2)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s5 & 0xF000) >> 12) | ((bits2.s5 & 0xC0) >> 2)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y; \
    shared_y = fetch(yb, s4, 3); \
    total_sum.s0 += ((float)(((bits4.s6 & 0x000F)      ) | ((bits2.s6 & 0x03) << 4)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x000F)      ) | ((bits2.s7 & 0x03) << 4)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y; \
    shared_y = fetch(yb, s5, 3); \
    total_sum.s0 += ((float)(((bits4.s6 & 0x00F0) >>  4) | ((bits2.s6 & 0x0C) << 2)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x00F0) >>  4) | ((bits2.s7 & 0x0C) << 2)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y; \
    shared_y = fetch(yb, s6, 3); \
    total_sum.s0 += ((float)(((bits4.s6 & 0x0F00) >>  8) | ((bits2.s6 & 0x30)     )) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s7 & 0x0F00) >>  8) | ((bits2.s7 & 0x30)     )) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y; \
    shared_y = fetch(yb, s7, 3); \
    total_sum.s0 += ((float)(((bits4.s6 & 0xF000) >> 12) | ((bits2.s6 & 0xC0) >> 2)) - 32.f) * scale_s.s1 * scale_d.s0 * shared_y; \
    total_sum.s1 += ((float)(((bits4.s7 & 0xF000) >> 12) | ((bits2.s7 & 0xC0) >> 2)) - 32.f) * scale_s.s3 * scale_d.s1 * shared_y; \

#ifndef MC_N_COLS
#if defined(ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemv_noshuffle_q6_K_f32(
    read_only image1d_buffer_t src0_ql,
    read_only image1d_buffer_t src0_qh,
    global half2 * src0_s,
    global half2 * src0_d,
    read_only image1d_buffer_t src1,
    global float * dst,
    ulong offsetd,
    int ne00,
    int ne01
) {
    int grp = get_local_id(1);
    int gid = get_global_id(0);
    ushort slid = get_sub_group_local_id();

    int nb = ne00 / 32;

    uint4    reg_a_l;
    ushort4  reg_a_h;
    half2    reg_d;
    char4    reg_s;
    float8   reg_b;

    float2  total_sum = 0.0f;

    int line_stride_a = ne01 / 2;
    int block_stride_a = NSUBGROUPS * ne01;

    for (int k = grp; k < nb; k += NSUBGROUPS) {
        reg_d = src0_d[gid + k/8 * line_stride_a];
        reg_s = as_char4(src0_s[gid + k * line_stride_a]);

        if (slid < 4) {
            reg_b.s0123 = read_imagef(src1, 0 + slid*2 + k*8);
            reg_b.s4567 = read_imagef(src1, 1 + slid*2 + k*8);
        }

        reg_a_l.s0 = read_imageui(src0_ql, gid + k*block_stride_a + line_stride_a*0).x;
        reg_a_l.s1 = read_imageui(src0_ql, gid + k*block_stride_a + line_stride_a*1).x;
        reg_a_l.s2 = read_imageui(src0_ql, gid + k*block_stride_a + line_stride_a*2).x;
        reg_a_l.s3 = read_imageui(src0_ql, gid + k*block_stride_a + line_stride_a*3).x;

        reg_a_h.s0 = as_ushort(read_imageh(src0_qh, gid + k*block_stride_a + line_stride_a*0).x);
        reg_a_h.s1 = as_ushort(read_imageh(src0_qh, gid + k*block_stride_a + line_stride_a*1).x);
        reg_a_h.s2 = as_ushort(read_imageh(src0_qh, gid + k*block_stride_a + line_stride_a*2).x);
        reg_a_h.s3 = as_ushort(read_imageh(src0_qh, gid + k*block_stride_a + line_stride_a*3).x);

#ifdef VECTOR_SUB_GROUP_BROADCAT
        dequantize_block_acc_bcast_8_hi(total_sum, as_ushort8(reg_a_l), as_uchar8(reg_a_h), reg_d, reg_s, reg_b);
#else
        dequantize_block_acc_bcast_1_hi(total_sum, as_ushort8(reg_a_l), as_uchar8(reg_a_h), reg_d, reg_s, reg_b);
#endif // VECTOR_SUB_GROUP_BROADCAT

        reg_a_l.s0 = read_imageui(src0_ql, gid + k*block_stride_a + line_stride_a*4).x;
        reg_a_l.s1 = read_imageui(src0_ql, gid + k*block_stride_a + line_stride_a*5).x;
        reg_a_l.s2 = read_imageui(src0_ql, gid + k*block_stride_a + line_stride_a*6).x;
        reg_a_l.s3 = read_imageui(src0_ql, gid + k*block_stride_a + line_stride_a*7).x;

        reg_a_h.s0 = as_ushort(read_imageh(src0_qh, gid + k*block_stride_a + line_stride_a*4).x);
        reg_a_h.s1 = as_ushort(read_imageh(src0_qh, gid + k*block_stride_a + line_stride_a*5).x);
        reg_a_h.s2 = as_ushort(read_imageh(src0_qh, gid + k*block_stride_a + line_stride_a*6).x);
        reg_a_h.s3 = as_ushort(read_imageh(src0_qh, gid + k*block_stride_a + line_stride_a*7).x);

#ifdef VECTOR_SUB_GROUP_BROADCAT
        dequantize_block_acc_bcast_8_lo(total_sum, as_ushort8(reg_a_l), as_uchar8(reg_a_h), reg_d, reg_s, reg_b);
#else
        dequantize_block_acc_bcast_1_lo(total_sum, as_ushort8(reg_a_l), as_uchar8(reg_a_h), reg_d, reg_s, reg_b);
#endif // VECTOR_SUB_GROUP_BROADCAT
    }

    local float2 reduce_lm[SUBGROUP_SIZE * 3];
    if (grp == 1) {
        reduce_lm[SUBGROUP_SIZE*0 + slid] = total_sum;
    }
    if (grp == 2) {
        reduce_lm[SUBGROUP_SIZE*1 + slid] = total_sum;
    }
    if (grp == 3) {
        reduce_lm[SUBGROUP_SIZE*2 + slid] = total_sum;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (grp == 0) {
        total_sum += reduce_lm[SUBGROUP_SIZE*0 + slid];
    }
    if (grp == 0) {
        total_sum += reduce_lm[SUBGROUP_SIZE*1 + slid];
    }
    if (grp == 0) {
        total_sum += reduce_lm[SUBGROUP_SIZE*2 + slid];
    }

    if (grp == 0) {
        dst = (global float*)((global char*)dst + offsetd);
        // Guard the two output rows. The x-grid is padded to CEIL_DIV(ne01/2,64)*64,
        // so when ne01 is not a multiple of 128 the tail row-pairs run past row ne01
        // and would overrun dst into the adjacent tensor (garbage downstream).
        // No-op / byte-identical when ne01 % 128 == 0 (no padding).
        if (gid * 2 + 0 < ne01) dst[gid * 2 + 0] = total_sum.s0;
        if (gid * 2 + 1 < ne01) dst[gid * 2 + 1] = total_sum.s1;
    }
}
#endif // MC_N_COLS

#ifdef MC_N_COLS
// Multi-column kernel_gemv_noshuffle_q6_K_f32, built once per MC_N_COLS (2..8): weights load once per K-block and
// each column runs the 1-column dequant-accumulate with the same K-split, so it is bit-identical to it.
// From MC_STAGE_MIN_COLS columns on, the activations come from a local-memory slice instead of per-column broadcasts.
#define MC_STAGE_MIN_COLS 4
#define MC_BLOCK_PIXELS   8 // float4 activation pixels per 32-element block

#if MC_N_COLS < MC_STAGE_MIN_COLS
#ifdef VECTOR_SUB_GROUP_BROADCAT
#define Q6K_MC_DEQ_HI dequantize_block_acc_bcast_8_hi
#define Q6K_MC_DEQ_LO dequantize_block_acc_bcast_8_lo
#else
#define Q6K_MC_DEQ_HI dequantize_block_acc_bcast_1_hi
#define Q6K_MC_DEQ_LO dequantize_block_acc_bcast_1_lo
#endif

#define Q6K_MC_COL(ts, c) { \
    if (slid < 4) { reg_b.s0123 = read_imagef(src1, (c) * col_stride + 0 + slid * 2 + k * 8); \
                    reg_b.s4567 = read_imagef(src1, (c) * col_stride + 1 + slid * 2 + k * 8); } \
    Q6K_MC_DEQ_HI(ts, as_ushort8(ql_hi), as_uchar8(qh_hi), reg_d, reg_s, reg_b); \
    Q6K_MC_DEQ_LO(ts, as_ushort8(ql_lo), as_uchar8(qh_lo), reg_d, reg_s, reg_b); }
#else
// Every lane reads the staged block itself, so the lane argument is not needed.
#define ACT_LOCAL(v, e, lane) v.e
// The fence keeps the compiler from hoisting every column's local loads, which spilled registers at n8 on A740.
#define Q6K_MC_COL(ts, c) { \
    mem_fence(CLK_LOCAL_MEM_FENCE); \
    local const float4 * sp = mc_stage + (grp * MC_N_COLS + (c)) * MC_BLOCK_PIXELS; \
    const float8 y0 = (float8)(sp[0], sp[1]); \
    const float8 y1 = (float8)(sp[2], sp[3]); \
    const float8 y2 = (float8)(sp[4], sp[5]); \
    const float8 y3 = (float8)(sp[6], sp[7]); \
    dequantize_block_acc_fetch_hi(ts, as_ushort8(ql_hi), as_uchar8(qh_hi), reg_d, reg_s, ACT_LOCAL, y0, y1); \
    dequantize_block_acc_fetch_lo(ts, as_ushort8(ql_lo), as_uchar8(qh_lo), reg_d, reg_s, ACT_LOCAL, y2, y3); }
#endif

// Sums the subgroup partials of one column in subgroup order and stores its two rows.
inline void reduce_store_col_q6k(float2 ts, int col, local float2 * lm, int sg, ushort slid,
                                 int gid, int ne01, global float * dst) {
    if (sg > 0) {
        lm[SUBGROUP_SIZE * (sg - 1) + slid] = ts;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg == 0) {
        for (int i = 0; i < NSUBGROUPS - 1; ++i) {
            ts += lm[SUBGROUP_SIZE * i + slid];
        }
        if (gid * 2 + 0 < ne01) dst[col * ne01 + gid * 2 + 0] = ts.s0;
        if (gid * 2 + 1 < ne01) dst[col * ne01 + gid * 2 + 1] = ts.s1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

#if defined(ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemv_noshuffle_q6_K_f32_mc(
    read_only image1d_buffer_t src0_ql,
    read_only image1d_buffer_t src0_qh,
    global half2 * src0_s,
    global half2 * src0_d,
    read_only image1d_buffer_t src1,
    global float * dst,
    ulong offsetd,
    int ne00,
    int ne01
) {
    int grp  = get_local_id(1);
    int gid  = get_global_id(0);
    ushort slid = get_sub_group_local_id();

    int nb = ne00 / 32;
    int line_stride_a  = ne01 / 2;
    int block_stride_a = NSUBGROUPS * ne01;
    int col_stride     = ne00 / 4;
    // Tail lanes (ne01 % 128 != 0) fetch a clamped row; the store guard drops their results.
    int gid_s = min(gid, line_stride_a - 1);

    uint4   ql_hi, ql_lo;
    ushort4 qh_hi, qh_lo;
    half2   reg_d;
    char4   reg_s;
    float8  reg_b;

    float2 ts0 = 0.0f, ts1 = 0.0f, ts2 = 0.0f, ts3 = 0.0f;
    float2 ts4 = 0.0f, ts5 = 0.0f, ts6 = 0.0f, ts7 = 0.0f;

#if MC_N_COLS >= MC_STAGE_MIN_COLS
    local float4 mc_stage[NSUBGROUPS * MC_N_COLS * MC_BLOCK_PIXELS];
#endif

    for (int k = grp; k < nb; k += NSUBGROUPS) {
        reg_d = src0_d[gid_s + k/8 * line_stride_a];
        reg_s = as_char4(src0_s[gid_s + k * line_stride_a]);

        ql_hi.s0 = read_imageui(src0_ql, gid_s + k*block_stride_a + line_stride_a*0).x;
        ql_hi.s1 = read_imageui(src0_ql, gid_s + k*block_stride_a + line_stride_a*1).x;
        ql_hi.s2 = read_imageui(src0_ql, gid_s + k*block_stride_a + line_stride_a*2).x;
        ql_hi.s3 = read_imageui(src0_ql, gid_s + k*block_stride_a + line_stride_a*3).x;
        qh_hi.s0 = as_ushort(read_imageh(src0_qh, gid_s + k*block_stride_a + line_stride_a*0).x);
        qh_hi.s1 = as_ushort(read_imageh(src0_qh, gid_s + k*block_stride_a + line_stride_a*1).x);
        qh_hi.s2 = as_ushort(read_imageh(src0_qh, gid_s + k*block_stride_a + line_stride_a*2).x);
        qh_hi.s3 = as_ushort(read_imageh(src0_qh, gid_s + k*block_stride_a + line_stride_a*3).x);

        ql_lo.s0 = read_imageui(src0_ql, gid_s + k*block_stride_a + line_stride_a*4).x;
        ql_lo.s1 = read_imageui(src0_ql, gid_s + k*block_stride_a + line_stride_a*5).x;
        ql_lo.s2 = read_imageui(src0_ql, gid_s + k*block_stride_a + line_stride_a*6).x;
        ql_lo.s3 = read_imageui(src0_ql, gid_s + k*block_stride_a + line_stride_a*7).x;
        qh_lo.s0 = as_ushort(read_imageh(src0_qh, gid_s + k*block_stride_a + line_stride_a*4).x);
        qh_lo.s1 = as_ushort(read_imageh(src0_qh, gid_s + k*block_stride_a + line_stride_a*5).x);
        qh_lo.s2 = as_ushort(read_imageh(src0_qh, gid_s + k*block_stride_a + line_stride_a*6).x);
        qh_lo.s3 = as_ushort(read_imageh(src0_qh, gid_s + k*block_stride_a + line_stride_a*7).x);

#if MC_N_COLS >= MC_STAGE_MIN_COLS
        // Stage this block's activations for every column in the subgroup's own slice. The first barrier keeps the
        // previous block's reads ahead of the overwrite, the second makes the writes visible to every lane.
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
        if (slid < MC_N_COLS * MC_BLOCK_PIXELS) {
            mc_stage[grp * MC_N_COLS * MC_BLOCK_PIXELS + slid] = read_imagef(src1,
                (slid / MC_BLOCK_PIXELS) * col_stride + k * MC_BLOCK_PIXELS + slid % MC_BLOCK_PIXELS);
        }
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
#endif
        Q6K_MC_COL(ts0, 0);
        Q6K_MC_COL(ts1, 1);
        if (MC_N_COLS > 2) Q6K_MC_COL(ts2, 2);
        if (MC_N_COLS > 3) Q6K_MC_COL(ts3, 3);
        if (MC_N_COLS > 4) Q6K_MC_COL(ts4, 4);
        if (MC_N_COLS > 5) Q6K_MC_COL(ts5, 5);
        if (MC_N_COLS > 6) Q6K_MC_COL(ts6, 6);
        if (MC_N_COLS > 7) Q6K_MC_COL(ts7, 7);
    }

    local float2 reduce_lm[SUBGROUP_SIZE * (NSUBGROUPS - 1)];
    dst = (global float*)((global char*)dst + offsetd);
    reduce_store_col_q6k(ts0, 0, reduce_lm, grp, slid, gid, ne01, dst);
    reduce_store_col_q6k(ts1, 1, reduce_lm, grp, slid, gid, ne01, dst);
    if (MC_N_COLS > 2) reduce_store_col_q6k(ts2, 2, reduce_lm, grp, slid, gid, ne01, dst);
    if (MC_N_COLS > 3) reduce_store_col_q6k(ts3, 3, reduce_lm, grp, slid, gid, ne01, dst);
    if (MC_N_COLS > 4) reduce_store_col_q6k(ts4, 4, reduce_lm, grp, slid, gid, ne01, dst);
    if (MC_N_COLS > 5) reduce_store_col_q6k(ts5, 5, reduce_lm, grp, slid, gid, ne01, dst);
    if (MC_N_COLS > 6) reduce_store_col_q6k(ts6, 6, reduce_lm, grp, slid, gid, ne01, dst);
    if (MC_N_COLS > 7) reduce_store_col_q6k(ts7, 7, reduce_lm, grp, slid, gid, ne01, dst);
}
#endif // MC_N_COLS
