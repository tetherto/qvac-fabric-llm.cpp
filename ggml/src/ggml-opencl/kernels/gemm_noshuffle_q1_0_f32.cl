#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

inline bool tile_finite(float8 c0, float8 c1, float8 c2, float8 c3) {
    return all(isfinite(c0)) && all(isfinite(c1)) && all(isfinite(c2)) && all(isfinite(c3));
}

inline float4 dequantize_rows(
        global const uint * wptr,
        global const half * sptr,
        int m,
        int kk
) {
    uint4 pack4 = vload4(0, wptr + (kk / 32) * m);
    float4 scale = convert_float4(vload4(0, sptr + (kk / 128) * m));
    float4 sign = convert_float4((pack4 >> (uint) (kk % 32)) & (uint) 1) * 2.0f - 1.0f;
    return sign * scale;
}

inline void accumulate_tile_fp32(
        read_only image1d_buffer_t src1,
        global const uint * wptr,
        global const half * sptr,
        int m,
        int n_4,
        int k,
        int gy,
        float8 * c0,
        float8 * c1,
        float8 * c2,
        float8 * c3
) {
    *c0 = 0; *c1 = 0; *c2 = 0; *c3 = 0;
    for (int kk = 0; kk < k; ++kk) {
        half8 B;
        B.s0123 = read_imageh(src1, gy * 2 + kk * n_4);
        B.s4567 = read_imageh(src1, gy * 2 + kk * n_4 + 1);
        float8 activations = convert_float8(B);
        float4 weights = dequantize_rows(wptr, sptr, m, kk);
        *c0 += activations * weights.s0;
        *c1 += activations * weights.s1;
        *c2 += activations * weights.s2;
        *c3 += activations * weights.s3;
    }
}

// each work-item computes a 4 (rows of A / m) x 8 (cols of B / n) output tile.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_gemm_noshuffle_q1_0_f32(
        global const uint * src0_q,
        global const half  * src0_d,
        read_only image1d_buffer_t src1,
        global float * dst,
        int k,
        int m,
        int n,
        int n_no_padding,
        ulong offsetd
) {
    int n_4 = n >> 2;

    int gy   = get_global_id(0);
    int gx   = get_global_id(1);
    int gx_2 = gx << 2;
    dst  = (global float *)((global char*)dst  + offsetd);

    float8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    half8 B;

    global const uint* wptr = src0_q + gx_2;
    global const half* sptr = src0_d + gx_2;

    // 32 weights per uint32, 128 weights (one block / one scale) per 4 uint32.
    for (int i = 0; i < k; i += 32) {
        uint4 pack4 = vload4(0, wptr + (i / 32)  * m); // 4 rows, 32 K-values each
        half4 scale = vload4(0, sptr + (i / 128) * m); // 4 rows, one scale per 128

        half8 b0 = 0, b1 = 0, b2 = 0, b3 = 0;

        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            B.s0123 = read_imageh(src1, gy * 2 + (i + j) * n_4);
            B.s4567 = read_imageh(src1, gy * 2 + (i + j) * n_4 + 1);

            // sign bit -> +-1 (half arithmetic avoids unsigned underflow)
            half4 wj = (half4)(
                2.0h * (half)((pack4.s0 >> j) & 1u) - 1.0h,
                2.0h * (half)((pack4.s1 >> j) & 1u) - 1.0h,
                2.0h * (half)((pack4.s2 >> j) & 1u) - 1.0h,
                2.0h * (half)((pack4.s3 >> j) & 1u) - 1.0h) * scale;

            b0 += B * wj.s0;
            b1 += B * wj.s1;
            b2 += B * wj.s2;
            b3 += B * wj.s3;
        }

        c0 += convert_float8(b0);
        c1 += convert_float8(b1);
        c2 += convert_float8(b2);
        c3 += convert_float8(b3);
    }

    if (!tile_finite(c0, c1, c2, c3)) {
        accumulate_tile_fp32(src1, wptr, sptr, m, n_4, k, gy, &c0, &c1, &c2, &c3);
    }

    int idx = (gy << 3) * m + (gx << 2);

    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s0, c1.s0, c2.s0, c3.s0), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s1, c1.s1, c2.s1, c3.s1), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s2, c1.s2, c2.s2, c3.s2), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s3, c1.s3, c2.s3, c3.s3), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s4, c1.s4, c2.s4, c3.s4), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s5, c1.s5, c2.s5, c3.s5), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s6, c1.s6, c2.s6, c3.s6), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s7, c1.s7, c2.s7, c3.s7), 0, dst + idx);
    }
}
