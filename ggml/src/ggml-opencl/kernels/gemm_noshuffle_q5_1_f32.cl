#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

#define QK5_1 32

inline bool tile_finite(float8 c0, float8 c1, float8 c2, float8 c3) {
    return all(isfinite(c0)) && all(isfinite(c1)) && all(isfinite(c2)) && all(isfinite(c3));
}

inline float4 dequantize_rows(
        global const ushort * weight_ptr,
        global const uchar * qh_ptr,
        global const half * scale_ptr,
        global const half * min_ptr,
        int m,
        int kk
) {
    ushort4 bits4 = vload4(0, weight_ptr + (kk/4)*m);
    uchar4  qh    = vload4(0, qh_ptr + (kk/8)*m);
    float4 scale = convert_float4(vload4(0, scale_ptr + (kk/QK5_1)*m));
    float4 minv  = convert_float4(vload4(0, min_ptr + (kk/QK5_1)*m));
    ushort4 nibble = (bits4 >> (ushort) (4 * (kk % 4))) & (ushort) 0xF;
    ushort4 high = convert_ushort4((qh >> (uchar) (kk % 8)) & (uchar) 1);
    return convert_float4(nibble | (high << (ushort) 4)) * scale + minv;
}

inline void accumulate_tile_fp32(
        read_only image1d_buffer_t src1,
        global const ushort * weight_ptr,
        global const uchar * qh_ptr,
        global const half * scale_ptr,
        global const half * min_ptr,
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
        float4 weights = dequantize_rows(weight_ptr, qh_ptr, scale_ptr, min_ptr, m, kk);
        *c0 += activations * weights.s0;
        *c1 += activations * weights.s1;
        *c2 += activations * weights.s2;
        *c3 += activations * weights.s3;
    }
}

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif

kernel void kernel_gemm_noshuffle_q5_1_f32(
        global const ushort * src0_qs,      // quantized A
        global const uchar  * src0_qh,      // 5th bits
        global const half   * src0_d,       // A scales
        global const half   * src0_m,       // A mins
        __read_only image1d_buffer_t src1,  // B (1d image)
        global float * dst,                 // C
        int m,                              // M
        int n,                              // N with padding
        int k,                              // K
        int n_no_padding                    // N without padding
) {

    int n_4 = n >> 2;

    int gy = get_global_id(0);
    int gx = get_global_id(1);
    int gx_2 = gx << 2;

    float8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    half8 B;
    half4 dequantized_weights;

    global const ushort * weight_ptr = src0_qs + gx_2;
    global const uchar  * qh_ptr    = src0_qh + gx_2;
    global const half   * scale_ptr = src0_d  + gx_2;
    global const half   * min_ptr   = src0_m  + gx_2;

    for (int ib = 0; ib < k; ib += QK5_1) {
        half4 scale = vload4(0, scale_ptr + (ib / QK5_1)*m);
        half4 minv  = vload4(0, min_ptr   + (ib / QK5_1)*m);
        half8 b0 = 0, b1 = 0, b2 = 0, b3 = 0;

        for (int i = ib; i < ib + QK5_1; i += 4) {
            B.s0123 = read_imageh(src1, gy*2 + i*n_4);
            B.s4567 = read_imageh(src1, gy*2 + i*n_4 + 1);

            ushort4 bits4 = vload4(0, weight_ptr + (i >> 2)*m);
            uchar4  bits1 = vload4(0, qh_ptr + (i >> 3)*m);
            uchar4  qh = bits1 >> (uchar4)(i & 4);

            // j=0
            dequantized_weights.s0 = convert_half((bits4.s0 & 0x000F) | ((qh.s0 & 0x01) << 4)) * scale.s0 + minv.s0;
            dequantized_weights.s1 = convert_half((bits4.s1 & 0x000F) | ((qh.s1 & 0x01) << 4)) * scale.s1 + minv.s1;
            dequantized_weights.s2 = convert_half((bits4.s2 & 0x000F) | ((qh.s2 & 0x01) << 4)) * scale.s2 + minv.s2;
            dequantized_weights.s3 = convert_half((bits4.s3 & 0x000F) | ((qh.s3 & 0x01) << 4)) * scale.s3 + minv.s3;
            b0 += B * dequantized_weights.s0;
            b1 += B * dequantized_weights.s1;
            b2 += B * dequantized_weights.s2;
            b3 += B * dequantized_weights.s3;

            // j=1
            B.s0123 = read_imageh(src1, gy*2 + (i+1)*n_4);
            B.s4567 = read_imageh(src1, gy*2 + (i+1)*n_4 + 1);
            dequantized_weights.s0 = convert_half(((bits4.s0 & 0x00F0) >> 4) | ((qh.s0 & 0x02) << 3)) * scale.s0 + minv.s0;
            dequantized_weights.s1 = convert_half(((bits4.s1 & 0x00F0) >> 4) | ((qh.s1 & 0x02) << 3)) * scale.s1 + minv.s1;
            dequantized_weights.s2 = convert_half(((bits4.s2 & 0x00F0) >> 4) | ((qh.s2 & 0x02) << 3)) * scale.s2 + minv.s2;
            dequantized_weights.s3 = convert_half(((bits4.s3 & 0x00F0) >> 4) | ((qh.s3 & 0x02) << 3)) * scale.s3 + minv.s3;
            b0 += B * dequantized_weights.s0;
            b1 += B * dequantized_weights.s1;
            b2 += B * dequantized_weights.s2;
            b3 += B * dequantized_weights.s3;

            // j=2
            B.s0123 = read_imageh(src1, gy*2 + (i+2)*n_4);
            B.s4567 = read_imageh(src1, gy*2 + (i+2)*n_4 + 1);
            dequantized_weights.s0 = convert_half(((bits4.s0 & 0x0F00) >> 8) | ((qh.s0 & 0x04) << 2)) * scale.s0 + minv.s0;
            dequantized_weights.s1 = convert_half(((bits4.s1 & 0x0F00) >> 8) | ((qh.s1 & 0x04) << 2)) * scale.s1 + minv.s1;
            dequantized_weights.s2 = convert_half(((bits4.s2 & 0x0F00) >> 8) | ((qh.s2 & 0x04) << 2)) * scale.s2 + minv.s2;
            dequantized_weights.s3 = convert_half(((bits4.s3 & 0x0F00) >> 8) | ((qh.s3 & 0x04) << 2)) * scale.s3 + minv.s3;
            b0 += B * dequantized_weights.s0;
            b1 += B * dequantized_weights.s1;
            b2 += B * dequantized_weights.s2;
            b3 += B * dequantized_weights.s3;

            // j=3
            B.s0123 = read_imageh(src1, gy*2 + (i+3)*n_4);
            B.s4567 = read_imageh(src1, gy*2 + (i+3)*n_4 + 1);
            dequantized_weights.s0 = convert_half(((bits4.s0 & 0xF000) >> 12) | ((qh.s0 & 0x08) << 1)) * scale.s0 + minv.s0;
            dequantized_weights.s1 = convert_half(((bits4.s1 & 0xF000) >> 12) | ((qh.s1 & 0x08) << 1)) * scale.s1 + minv.s1;
            dequantized_weights.s2 = convert_half(((bits4.s2 & 0xF000) >> 12) | ((qh.s2 & 0x08) << 1)) * scale.s2 + minv.s2;
            dequantized_weights.s3 = convert_half(((bits4.s3 & 0xF000) >> 12) | ((qh.s3 & 0x08) << 1)) * scale.s3 + minv.s3;
            b0 += B * dequantized_weights.s0;
            b1 += B * dequantized_weights.s1;
            b2 += B * dequantized_weights.s2;
            b3 += B * dequantized_weights.s3;
        }

        c0 += convert_float8(b0);
        c1 += convert_float8(b1);
        c2 += convert_float8(b2);
        c3 += convert_float8(b3);
    }

    if (!tile_finite(c0, c1, c2, c3)) {
        accumulate_tile_fp32(src1, weight_ptr, qh_ptr, scale_ptr, min_ptr, m, n_4, k, gy, &c0, &c1, &c2, &c3);
    }

    int idx = (gy<<3)*m + (gx<<2);

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
