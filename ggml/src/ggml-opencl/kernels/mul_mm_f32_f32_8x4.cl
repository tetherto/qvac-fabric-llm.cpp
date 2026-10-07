#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

// f32 x f32 GEMM in the shape the Adreno image path wants: A pre-transposed to [k][m]
// with m contiguous so each lane vload4s four m values, B staged as an RGBA-f32 image.
// Everything stays float, so this is the same arithmetic the generic kernel does.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif

// `m` is dst's row stride, `m_pad` is src0_t's: A may be padded up to a multiple of 4
// rows while dst keeps its true stride. Equal when the caller's A is already aligned.
kernel void kernel_mul_mm_f32_f32_8x4(
        global const float * src0_t,
        __read_only image1d_buffer_t src1,
        global float * dst,
        int k,
        int m,
        int m_pad,
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
    float8 B;

    __global const float * aptr = src0_t + gx_2;

    for (int i = 0; i < k; i += 4) {
        float4 w0 = vload4(0, aptr + (i + 0) * m_pad);
        float4 w1 = vload4(0, aptr + (i + 1) * m_pad);
        float4 w2 = vload4(0, aptr + (i + 2) * m_pad);
        float4 w3 = vload4(0, aptr + (i + 3) * m_pad);

        // ------------------- j = 0 (k = i+0) -------------------
        B.s0123 = read_imagef(src1, gy * 2 + (i + 0) * n_4);
        B.s4567 = read_imagef(src1, gy * 2 + (i + 0) * n_4 + 1);

        c0 += B * w0.s0;
        c1 += B * w0.s1;
        c2 += B * w0.s2;
        c3 += B * w0.s3;

        // ------------------- j = 1 (k = i+1) -------------------
        B.s0123 = read_imagef(src1, gy * 2 + (i + 1) * n_4);
        B.s4567 = read_imagef(src1, gy * 2 + (i + 1) * n_4 + 1);

        c0 += B * w1.s0;
        c1 += B * w1.s1;
        c2 += B * w1.s2;
        c3 += B * w1.s3;

        // ------------------- j = 2 (k = i+2) -------------------
        B.s0123 = read_imagef(src1, gy * 2 + (i + 2) * n_4);
        B.s4567 = read_imagef(src1, gy * 2 + (i + 2) * n_4 + 1);

        c0 += B * w2.s0;
        c1 += B * w2.s1;
        c2 += B * w2.s2;
        c3 += B * w2.s3;

        // ------------------- j = 3 (k = i+3) -------------------
        B.s0123 = read_imagef(src1, gy * 2 + (i + 3) * n_4);
        B.s4567 = read_imagef(src1, gy * 2 + (i + 3) * n_4 + 1);

        c0 += B * w3.s0;
        c1 += B * w3.s1;
        c2 += B * w3.s2;
        c3 += B * w3.s3;
    }

    // gws[1] is ceil(m/4), so the last row group is short when m % 4 != 0. For m % 4 == 0
    // this is exactly the old flat-index guard `idx+3 < m*n_no_padding`.
    const int nrow = min(m - gx_2, 4);
    int col = gy << 3;

#define STORE_COL(v0, v1, v2, v3)                                 \
    if (col < n_no_padding) {                                     \
        global float * p = dst + col * m + gx_2;                  \
        if (nrow == 4) {                                          \
            vstore4((float4)(v0, v1, v2, v3), 0, p);              \
        } else {                                                  \
            p[0] = v0;                                            \
            if (nrow > 1) { p[1] = v1; }                          \
            if (nrow > 2) { p[2] = v2; }                          \
        }                                                         \
    }                                                             \
    col++;

    STORE_COL(c0.s0, c1.s0, c2.s0, c3.s0)
    STORE_COL(c0.s1, c1.s1, c2.s1, c3.s1)
    STORE_COL(c0.s2, c1.s2, c2.s2, c3.s2)
    STORE_COL(c0.s3, c1.s3, c2.s3, c3.s3)
    STORE_COL(c0.s4, c1.s4, c2.s4, c3.s4)
    STORE_COL(c0.s5, c1.s5, c2.s5, c3.s5)
    STORE_COL(c0.s6, c1.s6, c2.s6, c3.s6)
    STORE_COL(c0.s7, c1.s7, c2.s7, c3.s7)

#undef STORE_COL
}
